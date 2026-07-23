import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast

from radiant.data import DownstreamDataset, load_manifest, split_manifest
from radiant.losses import BinaryFocalLoss, cox_ph_loss
from radiant.metrics.classification import binary_metrics
from radiant.metrics.survival import concordance_index, time_dependent_auc
from radiant.models import RadiantDownstream
from radiant.train.common import finite_or_raise, is_main_process, load_weights, set_seed
from radiant.train.ddp import cleanup_ddp, configure_runtime, cosine_lr, ddp_model, load_train_checkpoint, log_args, make_loader, progress, prune_checkpoints, reduce_sum_count, save_train_checkpoint, set_optimizer_lr, setup_ddp


def parse_tuple(value, cast=int):
    return tuple(cast(x) for x in value.split(","))


def gather_object_list(values):
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, values)
        out = []
        for part in gathered:
            out.extend(part)
        return out
    return values


def classification_loss(logits, labels, loss_fn):
    return loss_fn(logits, labels)


@torch.no_grad()
def validate(model, loader, device, args, cls_loss=None):
    model.eval()
    loss_total = 0.0
    loss_count = 0
    probs = []
    labels = []
    risks = []
    times = []
    events = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
            pred = model(batch["ct"], batch["dose"])
            if args.task == "classification":
                loss = classification_loss(pred, batch["label"], cls_loss)
            else:
                loss = cox_ph_loss(pred, batch["time"], batch["event"])
        n = batch["ct"].size(0)
        loss_total += float(loss.detach()) * n
        loss_count += n
        if args.task == "classification":
            probs.extend(torch.sigmoid(pred).detach().cpu().tolist())
            labels.extend(batch["label"].detach().cpu().tolist())
        else:
            risks.extend(pred.detach().cpu().tolist())
            times.extend(batch["time"].detach().cpu().tolist())
            events.extend(batch["event"].detach().cpu().tolist())
    metrics = {"loss": reduce_sum_count(loss_total, loss_count, device)}
    if args.task == "classification":
        probs = gather_object_list(probs)
        labels = gather_object_list(labels)
        if is_main_process():
            metrics.update(binary_metrics(np.asarray(probs), np.asarray(labels)))
    else:
        risks = gather_object_list(risks)
        times = gather_object_list(times)
        events = gather_object_list(events)
        if is_main_process():
            metrics["c_index"] = concordance_index(np.asarray(risks), np.asarray(times), np.asarray(events))
            metrics.update(time_dependent_auc(np.asarray(risks), np.asarray(times), np.asarray(events)))
    model.train()
    return metrics


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_manifest", default="")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--encoder_ckpt", default="")
    p.add_argument("--task", choices=["classification", "survival"], default="classification")
    p.add_argument("--roi_mode", choices=["auto", "roi", "tumor", "lung", "bowel", "rectum", "bladder", "full"], default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps_per_epoch", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--image_size", default="96,336,336")
    p.add_argument("--ct_clip", default="-1000,1000")
    p.add_argument("--margin_mm", type=float, default=30.0)
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--restrict_to_provided_roi", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
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
    required = ["patient_id", "ct_path", "generated_dose_path"]
    required += ["label"] if args.task == "classification" else ["time", "event"]
    frame = load_manifest(args.manifest, required)
    if args.val_manifest:
        train_frame = frame
        val_frame = load_manifest(args.val_manifest, required)
    else:
        train_frame, val_frame = split_manifest(frame, args.val_fraction, args.seed)
    train_dataset = DownstreamDataset(train_frame, image_size=image_size, ct_clip=ct_clip, augment=True, roi_mode=args.roi_mode, margin_mm=args.margin_mm, mask_threshold=args.mask_threshold, restrict_to_provided_roi=args.restrict_to_provided_roi)
    val_dataset = DownstreamDataset(val_frame, image_size=image_size, ct_clip=ct_clip, augment=False, roi_mode=args.roi_mode, margin_mm=args.margin_mm, mask_threshold=args.mask_threshold, restrict_to_provided_roi=args.restrict_to_provided_roi)
    train_loader, train_sampler = make_loader(train_dataset, args.batch_size, args.num_workers, train=True, drop_last=distributed)
    val_loader, _ = make_loader(val_dataset, args.batch_size, args.num_workers, train=False, drop_last=False)

    model = RadiantDownstream(task=args.task, encoder_kwargs={"patch_size": patch_size, "embed_dim": args.embed_dim, "depths": depths, "num_heads": num_heads, "window_size": window_size, "use_checkpoint": args.use_checkpoint}).to(device)
    if args.encoder_ckpt:
        load_weights(model.encoder, Path(args.encoder_ckpt), strict=False)
    model = ddp_model(model, local_rank)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), foreach=False)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    cls_loss = BinaryFocalLoss(args.focal_alpha, args.focal_gamma).to(device) if args.task == "classification" else None

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
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_count = 0
            iterator = progress(train_loader, is_main_process(), desc=f"downstream epoch {epoch}/{args.epochs}", total=min(len(train_loader), steps_per_epoch))
            for batch_idx, batch in enumerate(iterator, start=1):
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                with autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                    pred = model(batch["ct"], batch["dose"])
                    if args.task == "classification":
                        loss = classification_loss(pred, batch["label"], cls_loss)
                    else:
                        loss = cox_ph_loss(pred, batch["time"], batch["event"])
                    loss = finite_or_raise(loss) / args.grad_accum
                scaler.scale(loss).backward()
                epoch_loss += float(loss.detach()) * args.grad_accum
                epoch_count += batch["ct"].size(0)
                if batch_idx % args.grad_accum == 0:
                    lr = cosine_lr(args.lr, global_step, total_steps, args.warmup_steps, args.min_lr)
                    set_optimizer_lr(optimizer, lr)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if is_main_process() and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(loss=f"{epoch_loss / max(batch_idx, 1):.4f}", lr=f"{lr:.2e}", step=global_step)
                    if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                        save_train_checkpoint(out / f"epoch{epoch:03d}_step{global_step:07d}.pt", model, optimizer, scaler, epoch, global_step, args, {"loss": epoch_loss / max(batch_idx, 1)})
                        prune_checkpoints(out, "epoch*_step*.pt", args.ckpt_keep)
                if batch_idx >= steps_per_epoch:
                    break
            metrics = {"train_loss": epoch_loss / max(len(train_loader), 1)}
            if args.val_every_epochs > 0 and epoch % args.val_every_epochs == 0 and len(val_dataset) > 0:
                metrics.update({f"val_{k}": v for k, v in validate(model, val_loader, device, args, cls_loss).items()})
                if is_main_process():
                    print(metrics)
            if is_main_process() and (epoch % args.save_every_epochs == 0 or epoch == args.epochs):
                save_train_checkpoint(out / f"downstream_{args.task}_epoch{epoch:03d}.pt", model, optimizer, scaler, epoch, global_step, args, metrics)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
