import torch
import torch.distributed as dist
from torch.nn import functional as F


def segmentation_loss(logits, target):
    dims = tuple(range(1, logits.ndim))
    p = logits.float().sigmoid()
    y = target.float()
    dice = 1 - (2 * (p * y).sum(dims) + 1) / (p.sum(dims) + y.sum(dims) + 1)
    bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none").mean(
        dims
    )
    return (0.6 * dice + 0.4 * bce).mean()


def dose_loss(prediction, target, ctv, body, synchronize=False):
    error = (prediction.float() - target.float()).abs()
    pieces = []
    for region in [body.bool() & ctv.bool(), body.bool() & ~ctv.bool()]:
        for i in range(len(error)):
            if region[i].any():
                pieces.append(error[i][region[i]].mean())
    count = torch.tensor(float(len(pieces)), device=prediction.device)
    distributed = synchronize and dist.is_initialized()
    if distributed:
        dist.all_reduce(count)
    if count == 0:
        raise ValueError("No valid dose supervision region")
    total = torch.stack(pieces).sum() if pieces else prediction.float().sum() * 0
    return total * (dist.get_world_size() if distributed else 1) / count


def joint_loss(output, batch, synchronize=False):
    gtv = segmentation_loss(output["gtv_logits"], batch["gtv"])
    ctv = segmentation_loss(output["ctv_logits"], batch["ctv"])
    dose = dose_loss(
        output["dose"], batch["dose"], batch["ctv"], batch["body"], synchronize
    )
    return gtv + ctv + dose, {
        "gtv": gtv.detach(),
        "ctv": ctv.detach(),
        "dose": dose.detach(),
    }
