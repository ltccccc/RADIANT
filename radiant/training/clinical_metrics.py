import numpy as np


def binary_metrics(labels, scores, threshold=0.5):
    y, scores = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float)
    if (
        y.shape != scores.shape
        or y.ndim != 1
        or not len(y)
        or not np.isfinite(scores).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Invalid binary metric inputs")
    positive, negative = scores[y == 1], scores[y == 0]
    auc = None
    if len(positive) and len(negative):
        auc = float(
            np.mean(
                [
                    (np.sum(value > negative) + 0.5 * np.sum(value == negative))
                    / len(negative)
                    for value in positive
                ]
            )
        )
    predicted = scores >= threshold
    tp = int((predicted & (y == 1)).sum())
    tn = int((~predicted & (y == 0)).sum())
    fp = int((predicted & (y == 0)).sum())
    fn = int((~predicted & (y == 1)).sum())
    return {
        "auc": auc,
        "accuracy": float((predicted == y).mean()),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "threshold": threshold,
    }


def concordance_index(time, event, risk, tolerance=1e-8):
    time, event, risk = np.asarray(time), np.asarray(event), np.asarray(risk)
    if (
        time.shape != event.shape
        or time.shape != risk.shape
        or time.ndim != 1
        or not np.isfinite(time).all()
        or not np.isfinite(risk).all()
        or (time < 0).any()
        or not np.isin(event, [0, 1]).all()
    ):
        raise ValueError("Invalid survival metric inputs")
    score, pairs = 0.0, 0
    for i in range(len(time)):
        if event[i] != 1:
            continue
        comparable = (time > time[i]) | ((time == time[i]) & (event == 0))
        differences = risk[i] - risk[comparable]
        score += np.sum(differences > tolerance) + 0.5 * np.sum(
            np.abs(differences) <= tolerance
        )
        pairs += len(differences)
    return {
        "c_index": float(score / pairs) if pairs else None,
        "comparable_pairs": pairs,
    }
