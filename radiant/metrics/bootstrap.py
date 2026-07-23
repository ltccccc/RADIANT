import numpy as np


def bootstrap_ci(values, fn, n=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(values), len(values))
        stats.append(fn(values[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(values)), float(lo), float(hi)


def paired_bootstrap_ci(arrays, fn, n=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(x) for x in arrays]
    length = len(arrays[0])
    stats = []
    for _ in range(n):
        idx = rng.integers(0, length, length)
        stats.append(fn(*[x[idx] for x in arrays]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(*arrays)), float(lo), float(hi)
