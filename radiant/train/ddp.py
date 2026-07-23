import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .common import is_main_process, unwrap_model


def configure_runtime():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, init_method="env://", timeout=timedelta(minutes=60))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return distributed, rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def ddp_model(model, local_rank, find_unused_parameters=False):
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            return DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)
        return DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)
    return model


def make_loader(dataset, batch_size, num_workers, train, drop_last=False):
    sampler = DistributedSampler(dataset, shuffle=train, drop_last=drop_last) if dist.is_available() and dist.is_initialized() else None
    kwargs = {
        "batch_size": batch_size,
        "shuffle": train and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs), sampler


def reduce_mean(value, device):
    tensor = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.detach().cpu())


def reduce_sum_count(total, count, device):
    pair = torch.tensor([float(total), float(count)], device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    total, count = pair.detach().cpu().tolist()
    return total / max(count, 1.0)


def cosine_lr(base_lr, step, total_steps, warmup_steps=0, min_lr=0.0):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    denom = max(total_steps - warmup_steps, 1)
    t = min(max(step - warmup_steps, 0), denom) / denom
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def set_optimizer_lr(optimizer, lr_values):
    if not isinstance(lr_values, (list, tuple)):
        lr_values = [lr_values] * len(optimizer.param_groups)
    for group, lr in zip(optimizer.param_groups, lr_values):
        group["lr"] = lr


def save_train_checkpoint(path, model, optimizer, scaler, epoch, global_step, args, metrics=None, scheduler=None):
    if not is_main_process():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "metrics": metrics or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    latest = path.parent / "latest.pt"
    latest_tmp = latest.with_suffix(".pt.tmp")
    torch.save(state, latest_tmp)
    latest_tmp.replace(latest)


def load_train_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None, strict=True):
    state = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("epoch", 0)), int(state.get("global_step", 0)), state


def prune_checkpoints(directory, pattern, keep):
    if keep <= 0 or not is_main_process():
        return
    paths = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime)
    for path in paths[:-keep]:
        path.unlink(missing_ok=True)


def log_args(args, output_dir):
    if not is_main_process():
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(vars(args), indent=2, sort_keys=True)
    (output_dir / "args.json").write_text(payload + "\n")
    print(payload)


def progress(iterable, enabled, **kwargs):
    return tqdm(iterable, **kwargs) if enabled else iterable


class Timer:
    def __init__(self):
        self.t = time.time()

    def reset(self):
        now = time.time()
        elapsed = now - self.t
        self.t = now
        return elapsed
