import numpy as np


def dice_score(pred, target, threshold=0.5):
    pred = np.asarray(pred) > threshold
    target = np.asarray(target) > 0
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, target).sum() / denom)


def hd95(pred, target, spacing=(1, 1, 1), threshold=0.5):
    pred = np.asarray(pred) > threshold
    target = np.asarray(target) > 0
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
        ps = pred ^ binary_erosion(pred)
        ts = target ^ binary_erosion(target)
        dt_pred = distance_transform_edt(~ps, sampling=spacing)
        dt_target = distance_transform_edt(~ts, sampling=spacing)
        distances = np.concatenate([dt_target[ps], dt_pred[ts]])
    except Exception:
        ps = surface_points(pred)
        ts = surface_points(target)
        distances = np.concatenate([nearest_distances(ps, ts, spacing), nearest_distances(ts, ps, spacing)])
    return float(np.percentile(distances, 95))


def surface_points(mask):
    pts = np.argwhere(mask)
    keep = []
    shape = np.array(mask.shape)
    for p in pts:
        for off in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            q = p + np.array(off)
            if ((q < 0) | (q >= shape)).any() or not mask[tuple(q)]:
                keep.append(p)
                break
    return np.asarray(keep, dtype=np.float32)


def nearest_distances(a, b, spacing):
    if len(a) == 0 or len(b) == 0:
        return np.asarray([float("inf")])
    a = a * np.asarray(spacing)
    b = b * np.asarray(spacing)
    out = []
    for p in a:
        out.append(np.sqrt(((b - p) ** 2).sum(1)).min())
    return np.asarray(out)
