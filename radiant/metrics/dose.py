import numpy as np


def dose_metrics(pred, target, ptv_mask):
    pred = np.asarray(pred)[ptv_mask > 0]
    target = np.asarray(target)[ptv_mask > 0]
    if len(pred) == 0 or len(target) == 0:
        return {name: float("nan") for name in ("delta_hi", "delta_d98", "delta_d2", "delta_dmax", "delta_dmean")}
    return {
        "delta_hi": abs(homogeneity_index(pred) - homogeneity_index(target)),
        "delta_d98": abs(percentile(pred, 2) - percentile(target, 2)),
        "delta_d2": abs(percentile(pred, 98) - percentile(target, 98)),
        "delta_dmax": float(abs(pred.max() - target.max())),
        "delta_dmean": float(abs(pred.mean() - target.mean())),
    }


def percentile(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def homogeneity_index(x):
    d2 = percentile(x, 98)
    d98 = percentile(x, 2)
    d50 = percentile(x, 50)
    return (d2 - d98) / max(d50, 1e-8)
