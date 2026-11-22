import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, distance_transform_edt


def check_image(image):
    if image.GetDimension() != 3:
        raise ValueError("A three-dimensional image is required")
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError("Image contains non-finite values")
    if not np.all(np.asarray(image.GetSpacing()) > 0):
        raise ValueError("Image spacing must be positive")
    direction = np.array(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Image direction must be orthonormal")


def same_grid(a, b):
    return a.GetSize() == b.GetSize() and all(
        np.allclose(x, y, atol=1e-5, rtol=0)
        for x, y in [
            (a.GetSpacing(), b.GetSpacing()),
            (a.GetOrigin(), b.GetOrigin()),
            (a.GetDirection(), b.GetDirection()),
        ]
    )


def grid(ct, spacing_xyz=None):
    check_image(ct)
    if spacing_xyz is None:
        return ct
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing_xyz must contain three positive millimetre values")
    extent = (np.asarray(ct.GetSize()) - 1) * ct.GetSpacing()
    size = np.ceil(extent / spacing).astype(int) + 1
    result = sitk.Image(size.tolist(), sitk.sitkFloat32)
    result.SetOrigin(ct.GetOrigin())
    result.SetDirection(ct.GetDirection())
    result.SetSpacing(spacing.tolist())
    return result


def resample(image, reference, label=False, default=0):
    check_image(image)
    if label and not np.isin(sitk.GetArrayViewFromImage(image), [0, 1]).all():
        raise ValueError(
            "Expected a binary mask; select/resolve ROI labels before preprocessing"
        )
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if label else sitk.sitkLinear,
        float(default),
        sitk.sitkUInt8 if label else sitk.sitkFloat32,
    )


def crop_pad(image, shape_dhw, center_xyz=None, fill=0):
    shape = np.asarray(shape_dhw[::-1], dtype=int)
    if shape.shape != (3,) or (shape <= 0).any():
        raise ValueError("shape_dhw must contain three positive integers")
    center = (
        (np.asarray(image.GetSize()) - 1) / 2
        if center_xyz is None
        else np.asarray(
            image.TransformPhysicalPointToContinuousIndex(tuple(center_xyz))
        )
    )
    start = np.floor(center - (shape - 1) / 2).astype(int)
    ref = sitk.Image(shape.tolist(), image.GetPixelID())
    ref.CopyInformation(image) if tuple(shape) == image.GetSize() else None
    ref.SetSpacing(image.GetSpacing())
    ref.SetDirection(image.GetDirection())
    ref.SetOrigin(
        image.TransformContinuousIndexToPhysicalPoint(start.astype(float).tolist())
    )
    return resample(image, ref, image.GetPixelID() == sitk.sitkUInt8, fill)


def distance_map(mask, spacing_dhw):
    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing_dhw, dtype=float)
    if (
        mask.ndim != 3
        or spacing.shape != (3,)
        or not np.isfinite(spacing).all()
        or (spacing <= 0).any()
    ):
        raise ValueError("Invalid mask or physical spacing")
    if not mask.any():
        return np.ones(mask.shape, dtype=np.float32)
    if mask.all():
        return -np.ones(mask.shape, dtype=np.float32)
    return np.clip(
        (
            distance_transform_edt(~mask, sampling=spacing)
            - distance_transform_edt(mask, sampling=spacing)
        )
        / 100.0,
        -1,
        1,
    ).astype(np.float32)


def loss_region(ct_hu, ctv):
    a = np.asarray(ct_hu) > -500
    return np.stack([binary_fill_holes(s) for s in a]) | np.asarray(ctv, dtype=bool)


def from_array(array, reference):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image
