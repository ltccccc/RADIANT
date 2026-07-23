import argparse
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from radiant.data import PretrainDataset, load_manifest
from radiant.losses import MAEReconstructionLoss
from radiant.models import RadiantMAE
from radiant.train.common import finite_or_raise, is_main_process, set_seed, unwrap_model
from radiant.train.ddp import cleanup_ddp, configure_runtime, cosine_lr, ddp_model, load_train_checkpoint, log_args, make_loader, progress, prune_checkpoints, save_train_checkpoint, set_optimizer_lr, setup_ddp


def masked_voxel_grid(mask, target_shape, patch_size):
    b, _, d, h, w = target_shape
    gd = (d + patch_size[0] - 1) // patch_size[0]
    gh = (h + patch_size[1] - 1) // patch_size[1]
    gw = (w + patch_size[2] - 1) // patch_size[2]
    mask = mask.reshape(b, 1, gd, gh, gw).float()
    mask = mask.repeat_interleave(patch_size[0], 2).repeat_interleave(patch_size[1], 3).repeat_interleave(patch_size[2], 4)
    return mask[:, :, :d, :h, :w]


def parse_tuple(value):
    return tuple(int(x) for x in value.split(","))


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps_per_epoch", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--alpha_ssim", type=float, default=0.5)
    p.add_argument("--image_size", default="96,336,336")
    p.add_argument("--ct_clip", default="-1000,1000")
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
    patch_size = parse_tuple(args.patch_size)
    depths = parse_tuple(args.depths)
    num_heads = parse_tuple(args.num_heads)
    window_size = parse_tuple(args.window_size)
    ct_clip = tuple(float(x) for x in args.ct_clip.split(","))
    frame = load_manifest(args.manifest, ["patient_id", "ct_path"])
    dataset = PretrainDataset(frame, image_size=image_size, ct_clip=ct_clip)
    loader, sampler = make_loader(dataset, args.batch_size, args.num_workers, train=True, drop_last=distributed)

    model = RadiantMAE(
        mask_ratio=args.mask_ratio,
        patch_size=patch_size,
        embed_dim=args.embed_dim,
        encoder_kwargs={"depths": depths, "num_heads": num_heads, "window_size": window_size, "use_checkpoint": args.use_checkpoint},
    ).to(device)
    model = ddp_model(model, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), foreach=False)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loss_fn = MAEReconstructionLoss(args.alpha_ssim)

    start_epoch = 1
    global_step = 0
    if args.resume:
        ckpt = out / "latest.pt" if args.resume == "latest" else Path(args.resume)
        if ckpt.exists():
            loaded_epoch, global_step, _ = load_train_checkpoint(ckpt, model, optimizer, scaler, strict=True)
            start_epoch = loaded_epoch + 1
            if is_main_process():
                print(f"resumed {ckpt} epoch={loaded_epoch} step={global_step}")

    steps_per_epoch = args.steps_per_epoch or len(loader)
    total_steps = max(args.epochs * steps_per_epoch // max(args.grad_accum, 1), 1)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_count = 0
            iterator = progress(loader, is_main_process(), desc=f"stage1 epoch {epoch}/{args.epochs}", total=min(len(loader), steps_per_epoch))
            for batch_idx, batch in enumerate(iterator, start=1):
                ct = batch["ct"].to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                    recon, patch_mask = model(ct)
                    voxel_mask = masked_voxel_grid(patch_mask, ct.shape, patch_size)
                    loss = finite_or_raise(loss_fn(recon, ct, voxel_mask)) / args.grad_accum
                scaler.scale(loss).backward()
                epoch_loss += float(loss.detach()) * args.grad_accum
                epoch_count += 1
                if batch_idx % args.grad_accum == 0:
                    lr = cosine_lr(args.lr, global_step, total_steps, args.warmup_steps, args.min_lr)
                    set_optimizer_lr(optimizer, lr)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if is_main_process() and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(loss=f"{epoch_loss / max(epoch_count, 1):.4f}", lr=f"{lr:.2e}", step=global_step)
                    if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                        save_train_checkpoint(out / f"epoch{epoch:03d}_step{global_step:07d}.pt", model, optimizer, scaler, epoch, global_step, args, {"loss": epoch_loss / max(epoch_count, 1)})
                        prune_checkpoints(out, "epoch*_step*.pt", args.ckpt_keep)
                if batch_idx >= steps_per_epoch:
                    break
            if is_main_process() and (epoch % args.save_every_epochs == 0 or epoch == args.epochs):
                save_train_checkpoint(out / f"mae_epoch{epoch:03d}.pt", model, optimizer, scaler, epoch, global_step, args, {"loss": epoch_loss / max(epoch_count, 1)})
        if is_main_process():
            path = out / "radiant_encoder_stage1.pt"
            tmp = path.with_suffix(".pt.tmp")
            torch.save(unwrap_model(model).encoder.state_dict(), tmp)
            tmp.replace(path)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
