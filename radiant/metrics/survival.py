import numpy as np
from .classification import roc_auc


def concordance_index(risk, time, event):
    risk = np.asarray(risk)
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    total = 0
    correct = 0.0
    for i in range(len(risk)):
        for j in range(len(risk)):
            if time[i] < time[j] and event[i]:
                total += 1
                correct += risk[i] > risk[j]
                correct += 0.5 * (risk[i] == risk[j])
    return float(correct / total) if total else float("nan")


def time_dependent_auc(risk, time, event, months=(6, 12, 18, 24)):
    out = {}
    for m in months:
        case = (time <= m) & (event > 0)
        control = time > m
        keep = case | control
        y = case[keep].astype(int)
        out[f"auc_{m}m"] = roc_auc(np.asarray(risk)[keep], y) if keep.any() else float("nan")
    return out

