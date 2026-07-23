import random
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from(name):
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def setup_distributed(device_name="auto"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, device_from(device_name)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, local_rank, device


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path, model, optimizer=None, epoch=0, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": unwrap_model(model).state_dict(), "epoch": epoch}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    latest = path.parent / "latest.pt"
    latest_tmp = latest.with_suffix(".pt.tmp")
    torch.save(state, latest_tmp)
    latest_tmp.replace(latest)


def load_weights(model, path, key="model", strict=False):
    state = torch.load(path, map_location="cpu")
    weights = state.get(key, state)
    unwrap_model(model).load_state_dict(weights, strict=strict)
    return state


def load_training_state(path, model, optimizer=None, strict=False):
    state = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)), state


def finite_or_raise(loss, name="loss"):
    if not torch.isfinite(loss).all():
        raise FloatingPointError(f"Non-finite {name}: {float(loss.detach().cpu())}")
    return loss
