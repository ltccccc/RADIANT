import torch


def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    loss = -(risk - log_cumsum) * event
    return loss.sum() / event.sum().clamp_min(1.0)

