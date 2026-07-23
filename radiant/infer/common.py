from pathlib import Path

import torch


def resolve_checkpoint(checkpoint="", checkpoint_dir="", checkpoint_name="latest.pt"):
    if checkpoint:
        path = Path(checkpoint)
    elif checkpoint_dir:
        path = Path(checkpoint_dir) / checkpoint_name
    else:
        raise ValueError("Provide --checkpoint or --checkpoint_dir")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_model_weights(model, path, key="model", strict=False):
    state = torch.load(path, map_location="cpu")
    weights = state.get(key, state)
    model.load_state_dict(weights, strict=strict)
    return state
