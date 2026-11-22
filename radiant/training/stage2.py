import argparse
import math
from pathlib import Path
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from ..data.manifest import read_manifest
from ..data.dataset import Stage2Dataset
from ..models.encoder import load_backbone, Encoder
from ..models.stage2 import RadiantStage2
from .losses import joint_loss
from .common import (
    setup,
    sync_parameters,
    scaled_distributed_step,
    allreduce,
    loader,
    factor,
    fingerprint,
    save,
    read_config,
    log,
    move,
    restore_rng,
)


@torch.no_grad()
def validate(model, batches, device, amp=True):
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for batch in batches:
        batch = move(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            out = model(batch["ct"], spacing_dhw=batch["spacing_dhw"])
        for i in range(len(batch["ct"])):
            one = {
                k: v[i : i + 1] for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            pred = {k: v[i : i + 1] for k, v in out.items()}
            loss, parts = joint_loss(pred, one)
            totals += torch.tensor(
                [
                    float(loss),
                    float(parts["gtv"]),
                    float(parts["ctv"]),
                    float(parts["dose"]),
                ],
                device=device,
            )
    count = allreduce(
        torch.tensor(len(batches.sampler), device=device, dtype=torch.float64)
    )
    totals = allreduce(totals)
    if not count:
        raise ValueError("Empty validation set")
    return dict(zip(["loss", "gtv", "ctv", "dose"], (totals / count).tolist()))


def run(config, resume=None):
    cfg = read_config(config)
    device, rank, world = setup(cfg["seed"])
    rows = read_manifest(cfg["manifest"])
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "val"]
    if not training or not validation:
        raise ValueError("Explicit train and val patients required")
    train, sampler = loader(
        Stage2Dataset(training, True),
        cfg["batch_size"],
        cfg["workers"],
        rank,
        world,
        True,
    )
    val, _ = loader(Stage2Dataset(validation), 1, cfg["workers"], rank, world, False)
    model = RadiantStage2(
        Encoder(load_backbone(cfg["model_dir"], cfg["stage1_weights"])),
        tuple(cfg["decoder_widths"]),
    ).to(device)
    sync_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.encoder.parameters()),
                "lr": cfg["encoder_lr"],
                "base_lr": cfg["encoder_lr"],
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if not name.startswith("encoder.")
                ],
                "lr": cfg["decoder_lr"],
                "base_lr": cfg["decoder_lr"],
            },
        ],
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg["amp"] and device.type == "cuda", init_scale=1024.0
    )
    identity = fingerprint(cfg, cfg["manifest"])
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    start, step, best, stale = 1, 0, float("inf"), 0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck["fingerprint"] != identity or ck["world"] != world:
            raise ValueError("Resume fingerprint/world size mismatch")
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start, step, best, stale = ck["epoch"] + 1, ck["step"], ck["best"], ck["stale"]
        restore_rng(ck["rng_by_rank"][rank])
    elif (out / "latest.pt").exists():
        raise FileExistsError("Use --resume for an existing run")
    total = cfg["epochs"] * math.ceil(len(train) / cfg["accumulation"])
    for epoch in range(start, cfg["epochs"] + 1):
        sampler.set_epoch(epoch)
        model.set_epoch(epoch, cfg["freeze_epochs"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=device)
        for j, batch in enumerate(train):
            batch = move(batch, device)
            group_start = j // cfg["accumulation"] * cfg["accumulation"]
            denominator = min(cfg["accumulation"], len(train) - group_start)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=cfg["amp"] and device.type == "cuda",
            ):
                pred = model(
                    batch["ct"],
                    batch["gtv"],
                    batch["ctv"],
                    batch["distance"],
                    teacher_forcing=True,
                )
                loss, _ = joint_loss(pred, batch, synchronize=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite joint loss")
            scaler.scale(loss / denominator).backward()
            totals[:2] += torch.tensor([float(loss.detach()), 1], device=device)
            if (j + 1) % cfg["accumulation"] == 0 or j + 1 == len(train):
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * factor(
                        step, total, cfg["warmup_fraction"]
                    )
                updated = scaled_distributed_step(
                    model, optimizer, scaler, world, cfg["clip_grad"]
                )
                optimizer.zero_grad(set_to_none=True)
                step += int(updated)
                totals[2] += int(updated)
        metrics = validate(model, val, device, cfg["amp"])
        totals = allreduce(totals)
        improved = metrics["loss"] < best
        if improved:
            best, stale = metrics["loss"], 0
        elif epoch > cfg["freeze_epochs"]:
            stale += 1
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best": best,
            "stale": stale,
            "fingerprint": identity,
            "config": cfg,
        }
        save(out / "latest.pt", payload, rank, world)
        if improved:
            save(out / "best.pt", payload, rank, world)
            if rank == 0:
                save_file(
                    {
                        "model." + k: v.detach().cpu().contiguous()
                        for k, v in model.encoder.backbone.state_dict().items()
                    },
                    str(out / "encoder_stage2_best.safetensors"),
                )
        log(
            out,
            {
                "epoch": epoch,
                "step": step,
                "train_loss": float(totals[0] / totals[1]),
                "optimizer_steps": int(totals[2] / world),
                "validation": metrics,
                "encoder_frozen": model.encoder_frozen,
                "best": best,
            },
            rank,
        )
        if stale >= cfg["patience"] and epoch > cfg["freeze_epochs"]:
            break
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    args = p.parse_args()
    run(args.config, args.resume)


if __name__ == "__main__":
    main()
