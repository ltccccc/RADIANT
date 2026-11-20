import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..data.clinical import ClinicalDataset, check_clinical_manifest, check_exposure
from ..data.manifest import read_manifest, sha256
from ..models.encoder import load_backbone, Encoder
from ..models.stage3 import RadiantStage3
from .clinical_losses import focal_loss, cox_loss
from .clinical_metrics import binary_metrics, concordance_index
from .common import (
    setup,
    read_config,
    fingerprint,
    rng_state,
    restore_rng,
    save,
    log,
    move,
)


def build_model(config, use_immunotherapy):
    encoder = Encoder(
        load_backbone(config["model_dir"], config["stage2_encoder_weights"]),
        blocks=(12,),
        checkpointing=False,
    )
    return RadiantStage3(
        encoder,
        task=config["task"],
        use_immunotherapy=use_immunotherapy,
        dose_widths=tuple(config.get("dose_widths", [64, 128, 256])),
        fusion_channels=config.get("fusion_channels", 512),
        hidden_dim=config.get("hidden_dim", 512),
        dropout=config.get("dropout", 0.3),
    )


def prediction(model, batch):
    return model(batch["ct"], batch["dose"], batch.get("immunotherapy"))


def classification_epoch(
    model, batches, optimizer, scaler, device, amp, clip_grad, gamma=2.0
):
    model.train()
    total, patients, steps = 0.0, 0, 0
    for batch in batches:
        batch = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = prediction(model, batch)
            loss = focal_loss(logits, batch["label"], gamma)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite classification loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_grad, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        total += float(loss.detach()) * len(logits)
        patients += len(logits)
    return {"loss": total / patients, "optimizer_steps": steps}


def survival_epoch(
    model, batches, optimizer, scaler, device, amp, clip_grad, ties="efron"
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    risks, times, events, states, identities = [], [], [], [], []
    with torch.no_grad():
        for batch in batches:
            states.append(rng_state())
            identities.append(list(batch["case_id"]))
            batch = move(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                values = prediction(model, batch)
            risks.append(values.float().cpu())
            times.append(batch["time"].cpu())
            events.append(batch["event"].cpu())
    end_rng = rng_state()
    risk_leaf = torch.cat(risks).double().requires_grad_(True)
    loss = cox_loss(risk_leaf, torch.cat(times), torch.cat(events), ties)
    gradients = torch.autograd.grad(loss, risk_leaf)[0]
    offset, replay_error = 0, 0.0
    for index, batch in enumerate(batches):
        if list(batch["case_id"]) != identities[index]:
            raise RuntimeError("Survival replay requires a fixed patient order")
        batch = move(batch, device)
        restore_rng(states[index])
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            values = prediction(model, batch)
        expected = risks[index].to(device)
        error = (values.float() - expected).abs().max()
        replay_error = max(replay_error, float(error.detach()))
        if not torch.allclose(values.float(), expected, rtol=1e-4, atol=1e-5):
            raise RuntimeError(
                "Cox replay predictions changed before the optimizer step"
            )
        grad = gradients[offset : offset + len(values)].to(
            device=device, dtype=torch.float32
        )
        scaler.scale((values.float() * grad).sum()).backward()
        offset += len(values)
    restore_rng(end_rng)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip_grad, error_if_nonfinite=True
    )
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach()),
        "optimizer_steps": 1,
        "risk_set_patients": len(risk_leaf),
        "replay_max_error": replay_error,
    }


@torch.no_grad()
def evaluate_model(
    model, batches, device, task, amp=False, ties="efron", gamma=2.0, threshold=0.5
):
    model.eval()
    scores, labels, times, events, identifiers = [], [], [], [], []
    for batch in batches:
        identifiers.extend(zip(batch["case_id"], batch["patient_id"]))
        batch = move(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            values = prediction(model, batch)
        scores.append(values.float().cpu())
        if task == "classification":
            labels.append(batch["label"].cpu())
        else:
            times.append(batch["time"].cpu())
            events.append(batch["event"].cpu())
    scores = torch.cat(scores)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite clinical predictions")
    if task == "classification":
        labels = torch.cat(labels)
        probability = scores.sigmoid().numpy()
        metrics = binary_metrics(labels.numpy(), probability, threshold)
        metrics["loss"] = float(focal_loss(scores, labels, gamma))
        records = [
            {
                "case_id": ids[0],
                "patient_id": ids[1],
                "label": float(label),
                "probability": float(value),
            }
            for ids, label, value in zip(identifiers, labels, probability)
        ]
    else:
        times, events = torch.cat(times), torch.cat(events)
        metrics = concordance_index(times.numpy(), events.numpy(), scores.numpy())
        metrics["loss"] = float(cox_loss(scores, times, events, ties))
        records = [
            {
                "case_id": ids[0],
                "patient_id": ids[1],
                "time": float(time),
                "event": int(event),
                "log_relative_risk": float(value),
            }
            for ids, time, event, value in zip(identifiers, times, events, scores)
        ]
    return metrics, records


def datasets(config, rows, use_immunotherapy, split):
    chosen = [row for row in rows if row["split"] == split]
    if not chosen:
        raise ValueError("Empty clinical split: " + split)
    return ClinicalDataset(
        chosen,
        config["task"],
        use_immunotherapy,
        sha256(config["dose_generator_checkpoint"])
        if config.get("dose_generator_checkpoint")
        else None,
    )


def train(config, use_immunotherapy=None, resume=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "Stage 3 uses one process per endpoint to preserve full-cohort Cox risk sets"
        )
    cfg = read_config(config)
    flag = (
        cfg.get("use_immunotherapy", False)
        if use_immunotherapy is None
        else use_immunotherapy
    )
    if not isinstance(flag, bool):
        raise ValueError("use_immunotherapy must be True or False")
    cfg["use_immunotherapy"] = flag
    if cfg["task"] not in ["classification", "survival"]:
        raise ValueError("Unsupported downstream task")
    device, _, _ = setup(cfg["seed"])
    rows = read_manifest(cfg["manifest"])
    check_clinical_manifest(rows, cfg["task"], flag)
    check_exposure(rows, cfg["foundation_manifests"])
    training, validation = (
        datasets(cfg, rows, flag, "train"),
        datasets(cfg, rows, flag, "val"),
    )
    selected_rows = training.rows + validation.rows
    cfg["clinical_inputs"] = {
        row["case_id"]: {
            key: sha256(row[key])
            for key in [
                "ct_path",
                "predicted_dose_path",
                "dose_provenance_path",
                "roi_path",
            ]
            if row.get(key)
        }
        for row in selected_rows
    }
    identity = fingerprint(cfg, cfg["manifest"])
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "latest.pt").exists():
        raise FileExistsError("Use --resume for an existing clinical run")
    model = build_model(cfg, flag).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 20)
    )
    amp = cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    start, step, best, best_loss = 1, 0, -float("inf"), float("inf")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["fingerprint"] != identity or state["use_immunotherapy"] != flag:
            raise ValueError("Clinical resume configuration/inputs mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, step, best, best_loss = (
            state["epoch"] + 1,
            state["step"],
            state["best_metric"],
            state["best_loss"],
        )
        restore_rng(state["rng_by_rank"][0])
    (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    for epoch in range(start, cfg.get("epochs", 20) + 1):
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        train_loader = DataLoader(
            training,
            batch_size=cfg.get("batch_size", 1),
            shuffle=cfg["task"] == "classification",
            num_workers=cfg.get("workers", 2),
            generator=generator,
        )
        val_loader = DataLoader(
            validation, batch_size=1, shuffle=False, num_workers=cfg.get("workers", 2)
        )
        lr = optimizer.param_groups[0]["lr"]
        if cfg["task"] == "classification":
            result = classification_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                amp,
                cfg.get("clip_grad", 1.0),
                cfg.get("focal_gamma", 2.0),
            )
        else:
            result = survival_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                amp,
                cfg.get("clip_grad", 1.0),
                cfg.get("cox_ties", "efron"),
            )
        step += result["optimizer_steps"]
        metrics, _ = evaluate_model(
            model,
            val_loader,
            device,
            cfg["task"],
            amp,
            cfg.get("cox_ties", "efron"),
            cfg.get("focal_gamma", 2.0),
        )
        key = "auc" if cfg["task"] == "classification" else "c_index"
        score = metrics[key]
        if score is None:
            raise ValueError(
                "Validation requires both classes or comparable survival pairs"
            )
        improved = score > best or (score == best and metrics["loss"] < best_loss)
        if improved:
            best, best_loss = score, metrics["loss"]
        scheduler.step()
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_metric": best,
            "best_loss": best_loss,
            "task": cfg["task"],
            "use_immunotherapy": flag,
            "fingerprint": identity,
            "config": cfg,
        }
        save(output / "latest.pt", state, 0, 1)
        if improved:
            save(output / "best.pt", state, 0, 1)
        log(
            output,
            {
                "epoch": epoch,
                "step": step,
                "learning_rate": lr,
                "training": result,
                "validation": metrics,
                "selection_metric": key,
                "best_metric": best,
                "use_immunotherapy": flag,
            },
            0,
        )
    return str(output / "best.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p.add_argument(
        "--use-immunotherapy", action=argparse.BooleanOptionalAction, default=None
    )
    args = p.parse_args()
    train(args.config, args.use_immunotherapy, args.resume)


if __name__ == "__main__":
    main()
