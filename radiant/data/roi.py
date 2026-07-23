import numpy as np
from .preprocessing import pad_or_center_crop


def margin_voxels(margin_mm, spacing):
    return tuple(max(int(round(margin_mm / s)), 1) for s in spacing)


def mask_to_binary(mask, threshold=0.5):
    mask = np.asarray(mask)
    if mask.size == 0:
        return mask.astype(np.float32)
    return (mask >= threshold).astype(np.float32)


def dilate_mask(mask, margin_mm, spacing, threshold=0.5):
    mask = mask_to_binary(mask, threshold)
    radius = margin_voxels(margin_mm, spacing)
    grid = np.ogrid[
        -radius[0] : radius[0] + 1,
        -radius[1] : radius[1] + 1,
        -radius[2] : radius[2] + 1,
    ]
    footprint = sum((g / r) ** 2 for g, r in zip(grid, radius)) <= 1
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask > 0, structure=footprint).astype(np.float32)
    except Exception:
        coords = np.argwhere(mask > 0)
        out = np.zeros_like(mask, dtype=bool)
        offsets = np.argwhere(footprint) - np.array(radius)
        shape = np.array(mask.shape)
        for c in coords:
            pts = c + offsets
            pts = pts[((pts >= 0) & (pts < shape)).all(1)]
            out[pts[:, 0], pts[:, 1], pts[:, 2]] = True
        return out.astype(np.float32)


def crop_around_mask(x, mask, size):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return pad_or_center_crop(x, size)
    center = coords.mean(axis=0).round().astype(int)
    pads = [(s, s) for s in size]
    padded = np.pad(x, pads, mode="constant")
    center = center + np.array(size)
    slices = tuple(slice(c - s // 2, c - s // 2 + s) for c, s in zip(center, size))
    return padded[slices]
