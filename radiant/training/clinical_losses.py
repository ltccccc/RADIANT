import torch
from torch.nn import functional as F


def focal_loss(logits, labels, gamma=2.0):
    if gamma < 0:
        raise ValueError("Focal gamma must be nonnegative")
    logits, labels = logits.float().reshape(-1), labels.float().reshape(-1)
    if (
        logits.shape != labels.shape
        or not torch.isfinite(labels).all()
        or not ((labels == 0) | (labels == 1)).all()
    ):
        raise ValueError("Binary classification requires 0/1 labels")
    cross_entropy = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probability = torch.exp(-cross_entropy)
    return ((1 - probability).pow(gamma) * cross_entropy).mean()


def cox_loss(log_risk, time, event, ties="efron"):
    risk = log_risk.reshape(-1).double()
    time = time.to(risk.device).reshape(-1).double()
    event = event.to(risk.device).reshape(-1)
    if risk.shape != time.shape or risk.shape != event.shape or not len(risk):
        raise ValueError("Risk, time and event must have the same nonempty shape")
    if (
        not torch.isfinite(risk).all()
        or not torch.isfinite(time).all()
        or (time < 0).any()
        or not ((event == 0) | (event == 1)).all()
    ):
        raise ValueError(
            "Finite risks, nonnegative times and binary events are required"
        )
    if ties not in ["efron", "breslow"]:
        raise ValueError("Unsupported Cox ties method")
    observed = event.bool()
    count = observed.sum()
    if count == 0:
        raise ValueError("Cox training/evaluation requires at least one observed event")
    likelihood = risk.sum() * 0
    for value in torch.unique(time[observed]):
        failures = observed & (time == value)
        m = int(failures.sum())
        at_risk = time >= value
        log_total = torch.logsumexp(risk[at_risk], 0)
        if ties == "efron":
            ratio = torch.exp(torch.logsumexp(risk[failures], 0) - log_total).clamp(
                max=1
            )
            fractions = torch.arange(m, dtype=risk.dtype, device=risk.device) / m
            denominator = (log_total + torch.log1p(-fractions * ratio)).sum()
        else:
            denominator = m * log_total
        likelihood = likelihood + risk[failures].sum() - denominator
    return -likelihood / count
