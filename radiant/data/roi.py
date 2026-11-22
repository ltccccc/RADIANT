import numpy as np
import SimpleITK as sitk
from .geometry import same_grid


def roi_grid(ct, mask, margin_mm=30):
    if not same_grid(ct, mask):
        raise ValueError("ROI mask must be aligned to CT")
    points = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)
    if not len(points) or margin_mm < 0:
        raise ValueError("Empty mask or negative margin")
    spacing = np.asarray(ct.GetSpacing())
    lo = np.maximum(0, points.min(0)[::-1] - np.ceil(margin_mm / spacing).astype(int))
    hi = np.minimum(
        np.asarray(ct.GetSize()) - 1,
        points.max(0)[::-1] + np.ceil(margin_mm / spacing).astype(int),
    )
    return sitk.RegionOfInterest(ct, (hi - lo + 1).tolist(), lo.tolist())
