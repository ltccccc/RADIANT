import argparse
import math
from pathlib import Path
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from ..data.manifest import read_manifest
from ..data.views import MultiCropDataset, global_masks, augment_views
from ..models.encoder import load_backbone, configure_drop_path
from ..models.ssl import SSLModel
from .stage1_optim import parameter_groups, update_optimizer
from .common import (
    setup,
    sync_parameters,
    sync_gradients,
    allreduce,
    loader,
    fingerprint,
    save,
    read_config,
    log,
    restore_rng,
)


def run(config, resume=None):
    cfg = read_config(config)
    device, rank, world = setup(cfg["seed"])
    if cfg["batch_size"] < 2:
        raise ValueError("Stage 1 needs at least two patients per GPU for KoLeo")
    rows = [r for r in read_manifest(cfg["manifest"]) if r["split"] == "train"]
    train, sampler = loader(
        MultiCropDataset(rows, local_views=cfg.get("local_views", 8), seed=cfg["seed"]),
        cfg["batch_size"],
        cfg["workers"],
        rank,
        world,
        True,
    )
    student = (
        SSLModel(load_backbone(cfg["model_dir"]), cfg["prototypes"]).to(device).float()
    )
    teacher = (
        SSLModel(load_backbone(cfg["model_dir"]), cfg["prototypes"]).to(device).float()
    )
    configure_drop_path(student.backbone, cfg.get("drop_path", 0.2))
    configure_drop_path(teacher.backbone, 0.0)
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    sync_parameters(student)
    sync_parameters(teacher)
    optimizer = torch.optim.AdamW(
        parameter_groups(
            student,
            cfg["encoder_lr"],
            cfg["head_lr"],
            cfg.get("layer_decay", 0.9),
            cfg["weight_decay"],
        ),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(cfg, cfg["manifest"])
    start, step = 1, 0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck["fingerprint"] != identity or ck["world"] != world:
            raise ValueError("Resume fingerprint/world size mismatch")
        student.load_state_dict(ck["student"], strict=True)
        teacher.load_state_dict(ck["teacher"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        start, step = ck["epoch"] + 1, ck["step"]
        restore_rng(ck["rng_by_rank"][rank])
    elif (out / "latest.pt").exists():
        raise FileExistsError("Use --resume for an existing run")
    total = cfg["epochs"] * math.ceil(len(train) / cfg["accumulation"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for epoch in range(start, cfg["epochs"] + 1):
        sampler.set_epoch(epoch)
        train.dataset.set_epoch(epoch - 1)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        center = [
            torch.zeros_like(student.center_cls),
            0,
            torch.zeros_like(student.center_patch),
            0,
        ]
        totals = torch.zeros(2, device=device)
        for j, batch in enumerate(train):
            g, l = augment_views(batch, device)
            masks = global_masks(
                len(g),
                device=device,
                seed=cfg["seed"] + (epoch - 1) * 1000003 + j * world + rank,
            )
            temperature = 0.04 + 0.03 * min(1, step / max(1, 0.3 * total))
            loss, _, stats = student.loss(
                g, l, masks, teacher, temperature, cfg["patch_chunk"]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite self-supervised loss")
            denominator = min(
                cfg["accumulation"],
                len(train) - j // cfg["accumulation"] * cfg["accumulation"],
            )
            (loss / denominator).backward()
            center[0] += stats[0]
            center[1] += stats[1]
            center[2] += stats[2]
            center[3] += stats[3]
            totals += torch.tensor([float(loss.detach()), 1], device=device)
            if (j + 1) % cfg["accumulation"] == 0 or j + 1 == len(train):
                lr_factor, current_decay = update_optimizer(
                    optimizer,
                    step,
                    total,
                    cfg.get("warmup_fraction", 0.2),
                    cfg["weight_decay"],
                    cfg.get("weight_decay_end", 0.4),
                )
                sync_gradients(student, world)
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), cfg["clip_grad"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                momentum = (
                    0.992
                    + 0.008 * (1 - math.cos(math.pi * step / max(1, total - 1))) / 2
                )
                with torch.no_grad():
                    for t, s in zip(teacher.parameters(), student.parameters()):
                        t.mul_(momentum).add_(s, alpha=1 - momentum)
                    for index, name in [(0, "center_cls"), (2, "center_patch")]:
                        n = allreduce(
                            torch.tensor(float(center[index + 1]), device=device)
                        )
                        value = allreduce(center[index]) / n.clamp_min(1)
                        getattr(student, name).mul_(0.9).add_(value, alpha=0.1)
                center = [
                    torch.zeros_like(student.center_cls),
                    0,
                    torch.zeros_like(student.center_patch),
                    0,
                ]
                step += 1
        save(
            out / "latest.pt",
            {
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
                "fingerprint": identity,
                "config": cfg,
            },
            rank,
            world,
        )
        if rank == 0:
            save_file(
                {
                    "model." + k: v.detach().cpu().contiguous()
                    for k, v in teacher.backbone.state_dict().items()
                },
                str(out / f"encoder_epoch_{epoch:03d}.safetensors"),
            )
        totals = allreduce(totals)
        log(
            out,
            {
                "epoch": epoch,
                "step": step,
                "loss": float(totals[0] / totals[1]),
                "teacher_temperature": temperature,
                "teacher_momentum": momentum,
                "effective_batch": world * cfg["batch_size"] * cfg["accumulation"],
                "weight_decay": current_decay,
                "learning_rate_factor": lr_factor,
                "layer_decay": cfg.get("layer_decay", 0.9),
                "student_drop_path": cfg.get("drop_path", 0.2),
            },
            rank,
        )
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
