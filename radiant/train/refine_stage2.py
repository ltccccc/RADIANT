import argparse
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from radiant.data import Stage2Dataset, load_manifest, split_manifest
from radiant.losses import DiceFocalLoss, DoseRegressionLoss
from radiant.metrics.dose import dose_metrics
from radiant.metrics.segmentation import hd95
from radiant.models import RadiantStage2
from radiant.train.common import finite_or_raise, is_main_process, load_weights, set_seed, unwrap_model
from radiant.train.ddp import cleanup_ddp, configure_runtime, cosine_lr, ddp_model, load_train_checkpoint, log_args, make_loader, progress, prune_checkpoints, reduce_sum_count, save_train_checkpoint, set_optimizer_lr, setup_ddp


def parse_tuple(value, cast=int):
    return tuple(cast(x) for x in value.split(","))


def masked_mean(losses, flags):
    indices = torch.where(flags)[0]
    if len(indices) == 0:
        return None
    return torch.stack([losses[int(i)] for i in indices]).mean()


def dice_score(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    denom = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return ((2 * inter + eps) / (denom + eps)).detach()


def compute_loss(outputs, batch, seg_loss, dose_loss, args):
    ct = batch["ct"]
    gtv = batch["gtv"]
    ctv = batch["ctv"]
    dose = batch["dose"]
    has_gtv = batch["has_gtv"]
    has_ctv = batch["has_ctv"]
    has_dose = batch["has_dose"]
    gtv_terms = [seg_loss(outputs["gtv_logits"][i : i + 1], gtv[i : i + 1]) for i in range(ct.size(0))]
    ctv_terms = [seg_loss(outputs["ctv_logits"][i : i + 1], ctv[i : i + 1]) for i in range(ct.size(0))]
    dose_terms = [dose_loss(outputs["dose"][i : i + 1], dose[i : i + 1]) for i in range(ct.size(0))]
    pieces = []
    logs = {}
    for name, weight, value in (
        ("gtv", args.lambda_gtv, masked_mean(gtv_terms, has_gtv)),
        ("ctv", args.lambda_ctv, masked_mean(ctv_terms, has_ctv)),
        ("dose", args.lambda_dose, masked_mean(dose_terms, has_dose)),
    ):
        if value is not None:
            pieces.append(weight * value)
            logs[f"{name}_loss"] = float(value.detach())
        else:
            logs[f"{name}_loss"] = 0.0
    loss = torch.stack(pieces).sum() if pieces else outputs["dose"].sum() * 0
    return loss, logs


@torch.no_grad()
def validate(model, loader, device, seg_loss, dose_loss, args):
    model.eval()
    totals = {"loss": [0.0, 0], "gtv_dice": [0.0, 0], "ctv_dice": [0.0, 0], "dose_l1": [0.0, 0]}
    for name in ("gtv_hd95", "ctv_hd95", "delta_hi", "delta_d98", "delta_d2", "delta_dmax", "delta_dmean"):
        totals[name] = [0.0, 0]
    spacing = parse_tuple(args.spacing, float)
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
            outputs = model(batch["ct"])
            loss, _ = compute_loss(outputs, batch, seg_loss, dose_loss, args)
        n = batch["ct"].size(0)
        totals["loss"][0] += float(loss.detach()) * n
        totals["loss"][1] += n
        if batch["has_gtv"].any():
            scores = dice_score(outputs["gtv_logits"][batch["has_gtv"]], batch["gtv"][batch["has_gtv"]])
            totals["gtv_dice"][0] += float(scores.sum())
            totals["gtv_dice"][1] += int(scores.numel())
            probs = torch.sigmoid(outputs["gtv_logits"][batch["has_gtv"]]).detach().float().cpu().numpy()
            targets = batch["gtv"][batch["has_gtv"]].detach().float().cpu().numpy()
            for pred_i, target_i in zip(probs[:, 0], targets[:, 0]):
                value = hd95(pred_i, target_i, spacing=spacing)
                if np.isfinite(value):
                    totals["gtv_hd95"][0] += float(value)
                    totals["gtv_hd95"][1] += 1
        if batch["has_ctv"].any():
            scores = dice_score(outputs["ctv_logits"][batch["has_ctv"]], batch["ctv"][batch["has_ctv"]])
            totals["ctv_dice"][0] += float(scores.sum())
            totals["ctv_dice"][1] += int(scores.numel())
            probs = torch.sigmoid(outputs["ctv_logits"][batch["has_ctv"]]).detach().float().cpu().numpy()
            targets = batch["ctv"][batch["has_ctv"]].detach().float().cpu().numpy()
            for pred_i, target_i in zip(probs[:, 0], targets[:, 0]):
                value = hd95(pred_i, target_i, spacing=spacing)
                if np.isfinite(value):
                    totals["ctv_hd95"][0] += float(value)
                    totals["ctv_hd95"][1] += 1
        if batch["has_dose"].any():
            l1 = torch.abs(outputs["dose"][batch["has_dose"]] - batch["dose"][batch["has_dose"]]).mean(dim=(1, 2, 3, 4))
            totals["dose_l1"][0] += float(l1.sum())
            totals["dose_l1"][1] += int(l1.numel())
        if "has_ptv" in batch and (batch["has_dose"] & batch["has_ptv"]).any():
            indices = torch.where(batch["has_dose"] & batch["has_ptv"])[0]
            pred_np = outputs["dose"].detach().float().cpu().numpy()
            dose_np = batch["dose"].detach().float().cpu().numpy()
            ptv_np = batch["ptv"].detach().float().cpu().numpy()
            for i in indices.detach().cpu().tolist():
                values = dose_metrics(pred_np[i, 0], dose_np[i, 0], ptv_np[i, 0])
                for name, value in values.items():
                    if np.isfinite(value):
                        totals[name][0] += float(value)
                        totals[name][1] += 1
    metrics = {k: reduce_sum_count(v[0], v[1], device) for k, v in totals.items()}
    model.train()
    return metrics


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_manifest", default="")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--stage1_ckpt", default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps_per_epoch", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--lr_decoders", type=float, default=1e-4)
    p.add_argument("--min_lr_scale", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--freeze_encoder_epochs", type=int, default=20)
    p.add_argument("--lambda_gtv", type=float, default=1.0)
    p.add_argument("--lambda_ctv", type=float, default=1.0)
    p.add_argument("--lambda_dose", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--image_size", default="96,336,336")
    p.add_argument("--ct_clip", default="-1000,1000")
    p.add_argument("--spacing", default="2,2,5")
    p.add_argument("--patch_size", default="4,4,4")
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depths", default="2,2,24,2")
    p.add_argument("--num_heads", default="12,24,48,96")
    p.add_argument("--window_size", default="4,7,7")
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--save_every_steps", type=int, default=1000)
    p.add_argument("--save_every_epochs", type=int, default=1)
    p.add_argument("--val_every_epochs", type=int, default=1)
    p.add_argument("--ckpt_keep", type=int, default=5)
    return p


def main():
    configure_runtime()
    args = make_parser().parse_args()
    distributed, rank, local_rank, world_size, device = setup_ddp()
    set_seed(args.seed + rank)
    out = Path(args.output_dir)
    log_args(args, out)

    image_size = parse_tuple(args.image_size)
    ct_clip = parse_tuple(args.ct_clip, float)
    patch_size = parse_tuple(args.patch_size)
    depths = parse_tuple(args.depths)
    num_heads = parse_tuple(args.num_heads)
    window_size = parse_tuple(args.window_size)
    frame = load_manifest(args.manifest, ["patient_id", "ct_path", "prescription_dose"])
    if args.val_manifest:
        train_frame = frame
        val_frame = load_manifest(args.val_manifest, ["patient_id", "ct_path", "prescription_dose"])
    else:
        train_frame, val_frame = split_manifest(frame, args.val_fraction, args.seed)
    train_dataset = Stage2Dataset(train_frame, image_size=image_size, ct_clip=ct_clip, augment=True)
    val_dataset = Stage2Dataset(val_frame, image_size=image_size, ct_clip=ct_clip, augment=False)
    train_loader, train_sampler = make_loader(train_dataset, args.batch_size, args.num_workers, train=True, drop_last=distributed)
    val_loader, _ = make_loader(val_dataset, args.batch_size, args.num_workers, train=False, drop_last=False)

    model = RadiantStage2(encoder_kwargs={"patch_size": patch_size, "embed_dim": args.embed_dim, "depths": depths, "num_heads": num_heads, "window_size": window_size, "use_checkpoint": args.use_checkpoint}).to(device)
    if args.stage1_ckpt:
        load_weights(model.encoder, Path(args.stage1_ckpt), strict=False)
    model = ddp_model(model, local_rank, find_unused_parameters=True)
    raw_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": raw_model.encoder.parameters(), "lr": args.lr_encoder},
            {"params": list(raw_model.gtv_decoder.parameters()) + list(raw_model.ctv_decoder.parameters()) + list(raw_model.dose_decoder.parameters()), "lr": args.lr_decoders},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        foreach=False,
    )
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    seg_loss = DiceFocalLoss().to(device)
    dose_loss = DoseRegressionLoss().to(device)

    start_epoch = 1
    global_step = 0
    if args.resume:
        ckpt = out / "latest.pt" if args.resume == "latest" else Path(args.resume)
        if ckpt.exists():
            loaded_epoch, global_step, _ = load_train_checkpoint(ckpt, model, optimizer, scaler, strict=True)
            start_epoch = loaded_epoch + 1
            if is_main_process():
                print(f"resumed {ckpt} epoch={loaded_epoch} step={global_step}")

    steps_per_epoch = args.steps_per_epoch or len(train_loader)
    total_steps = max(args.epochs * steps_per_epoch // max(args.grad_accum, 1), 1)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            raw_model = unwrap_model(model)
            for param in raw_model.encoder.parameters():
                param.requires_grad = epoch > args.freeze_encoder_epochs
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_count = 0
            iterator = progress(train_loader, is_main_process(), desc=f"stage2 epoch {epoch}/{args.epochs}", total=min(len(train_loader), steps_per_epoch))
            for batch_idx, batch in enumerate(iterator, start=1):
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                    outputs = model(batch["ct"])
                    loss, logs = compute_loss(outputs, batch, seg_loss, dose_loss, args)
                    loss = finite_or_raise(loss) / args.grad_accum
                scaler.scale(loss).backward()
                epoch_loss += float(loss.detach()) * args.grad_accum
                epoch_count += 1
                if batch_idx % args.grad_accum == 0:
                    enc_lr = cosine_lr(args.lr_encoder, global_step, total_steps, args.warmup_steps, args.lr_encoder * args.min_lr_scale)
                    dec_lr = cosine_lr(args.lr_decoders, global_step, total_steps, args.warmup_steps, args.lr_decoders * args.min_lr_scale)
                    set_optimizer_lr(optimizer, [enc_lr, dec_lr])
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if is_main_process() and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(loss=f"{epoch_loss / max(epoch_count, 1):.4f}", gtv=f"{logs['gtv_loss']:.3f}", ctv=f"{logs['ctv_loss']:.3f}", dose=f"{logs['dose_loss']:.3f}", step=global_step)
                    if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                        save_train_checkpoint(out / f"epoch{epoch:03d}_step{global_step:07d}.pt", model, optimizer, scaler, epoch, global_step, args, {"loss": epoch_loss / max(epoch_count, 1)})
                        prune_checkpoints(out, "epoch*_step*.pt", args.ckpt_keep)
                if batch_idx >= steps_per_epoch:
                    break
            metrics = {"train_loss": epoch_loss / max(epoch_count, 1)}
            if args.val_every_epochs > 0 and epoch % args.val_every_epochs == 0 and len(val_dataset) > 0:
                metrics.update({f"val_{k}": v for k, v in validate(model, val_loader, device, seg_loss, dose_loss, args).items()})
                if is_main_process():
                    print(metrics)
            if is_main_process() and (epoch % args.save_every_epochs == 0 or epoch == args.epochs):
                save_train_checkpoint(out / f"stage2_epoch{epoch:03d}.pt", model, optimizer, scaler, epoch, global_step, args, metrics)
        if is_main_process():
            path = out / "radiant_encoder_stage2.pt"
            tmp = path.with_suffix(".pt.tmp")
            torch.save(unwrap_model(model).encoder.state_dict(), tmp)
            tmp.replace(path)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
