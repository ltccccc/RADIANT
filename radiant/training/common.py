import hashlib
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from ..data.manifest import sha256


def setup(seed):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = (
        torch.device("cuda", local)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    return device, rank, world


def allreduce(value):
    if dist.is_initialized():
        dist.all_reduce(value)
    return value


def sync_parameters(model):
    if dist.is_initialized():
        for p in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(p.data, 0)


def sync_gradients(model, world):
    if world > 1:
        for p in model.parameters():
            if p.requires_grad:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad.div_(world)


def loader(dataset, batch, workers, rank, world, training):
    if training:
        sampler = DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True
        )
    else:
        sampler = list(range(rank, len(dataset), world))
    result = DataLoader(
        dataset,
        batch_size=batch,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
    )
    if training and not len(result):
        raise ValueError("No complete training minibatch")
    return result, sampler


def factor(step, total, warmup=0.2):
    n = max(1, int(total * warmup))
    if step < n:
        return (step + 1) / n
    return 0.01 + 0.99 * 0.5 * (
        1 + math.cos(math.pi * min(1, (step - n) / max(1, total - n - 1)))
    )


def fingerprint(config, manifest):
    source = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))
    }
    assets = {}
    if config.get("model_dir"):
        folder = Path(config["model_dir"])
        assets.update(
            {
                str(p): sha256(p)
                for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix in [".py", ".json", ".safetensors"]
            }
        )
    for key in ["stage1_weights", "stage2_encoder_weights", "dose_generator_checkpoint"]:
        if config.get(key):
            assets[config[key]] = sha256(config[key])
    for path in config.get("foundation_manifests", []):
        assets[path] = sha256(path)
    return hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "manifest": sha256(manifest),
                "sources": sources,
                "assets": assets,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def save(path, payload, rank, world):
    states = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(rng_state(), states, dst=0)
    else:
        states = [rng_state()]
    error = [None]
    if rank == 0:
        try:
            payload = {**payload, "rng_by_rank": states, "world": world}
            tmp = Path(str(path) + ".tmp")
            torch.save(payload, tmp)
            tmp.replace(path)
        except Exception as exc:
            error[0] = repr(exc)
    if world > 1:
        dist.broadcast_object_list(error, 0)
    if error[0]:
        raise RuntimeError(error[0])


def read_config(path):
    return json.loads(Path(path).read_text())


def log(root, record, rank):
    if rank == 0:
        with (root / "history.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        (root / "progress.json").write_text(
            json.dumps(record, indent=2, allow_nan=False)
        )


def move(batch, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
