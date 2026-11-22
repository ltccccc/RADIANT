import math


def parameter_groups(model, encoder_lr, head_lr, layer_decay=0.9, weight_decay=0.04):
    if not 0 < layer_decay <= 1:
        raise ValueError("layer_decay must be in (0,1]")
    groups = {}
    depth_count = len(model.backbone.blocks)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.blocks."):
            depth = int(name.split(".")[2]) + 1
            scale = layer_decay ** (depth_count + 1 - depth)
            base = encoder_lr
        elif name.startswith("backbone."):
            scale = (
                1.0
                if name.startswith("backbone.norm.")
                else layer_decay ** (depth_count + 1)
            )
            base = encoder_lr
        else:
            scale, base = 1.0, head_lr
        magnitude = "last_layer." in name and (
            name.endswith("weight_g")
            or name.endswith("parametrizations.weight.original0")
        )
        decay = not (parameter.ndim == 1 or name.endswith(".bias") or magnitude)
        key = (base * scale, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "base_lr": base * scale,
                "lr": base * scale,
                "decay": decay,
                "weight_decay": weight_decay if decay else 0.0,
            }
        groups[key]["params"].append(parameter)
    return list(groups.values())


def schedule(step, total, warmup_fraction=0.2, weight_decay=0.04, weight_decay_end=0.4):
    if total < 1 or step < 0 or step >= total or not 0 <= warmup_fraction <= 1:
        raise ValueError("Invalid Stage 1 schedule position")
    warmup = max(1, round(warmup_fraction * total))
    if step < warmup:
        learning_rate = (step + 1) / warmup
    else:
        learning_rate = 0.01 + 0.99 * 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))
        )
    position = step / max(1, total - 1)
    decay = weight_decay + (weight_decay_end - weight_decay) * 0.5 * (
        1 - math.cos(math.pi * position)
    )
    return learning_rate, decay


def update_optimizer(
    optimizer, step, total, warmup_fraction=0.2, weight_decay=0.04, weight_decay_end=0.4
):
    learning_rate, decay = schedule(
        step, total, warmup_fraction, weight_decay, weight_decay_end
    )
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * learning_rate
        group["weight_decay"] = decay if group["decay"] else 0.0
    return learning_rate, decay
