import math
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import SimpleITK as sitk


def resized_crop(array, size, scale, rng):
    d, h, w = array.shape
    for _ in range(10):
        area = rng.uniform(*scale) * h * w
        ratio = math.exp(rng.uniform(math.log(0.75), math.log(4 / 3)))
        ch = int(round(math.sqrt(area / ratio)))
        cw = int(round(math.sqrt(area * ratio)))
        if 0 < ch <= h and 0 < cw <= w:
            break
    else:
        ch, cw = h, w
    y = rng.randrange(h - ch + 1)
    x = rng.randrange(w - cw + 1)
    z = rng.randrange(max(1, d - 12 + 1))
    view = torch.from_numpy(array[z : z + 12, y : y + ch, x : x + cw].copy())[None]
    if view.shape[1] < 12:
        view = F.pad(view, (0, 0, 0, 0, 0, 12 - view.shape[1]), value=-1000)
    view = F.interpolate(
        view[None], size=(12, size, size), mode="trilinear", align_corners=False
    )[0]
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            view = view.flip(axis)
    return view


def augment_intensity(x, gamma=None, sigma=None):
    if gamma is not None:
        x = x.clamp_min(0).pow(gamma)
    if sigma is not None:
        axis = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (axis / sigma).square())
        kernel = kernel / kernel.sum()
        for dim in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[dim + 2] = 5
            padding = [0, 0, 0]
            padding[dim] = 2
            x = F.conv3d(x[None], kernel.reshape(shape), padding=tuple(padding))[0]
    return x


def augment_views(batch, device):
    global_views = batch["global"].to(device, non_blocking=True).float()
    local_views = batch["local"].to(device, non_blocking=True).float()
    parameters = batch["params"]
    views = []
    for k in range(2 + local_views.shape[1]):
        current = global_views[:, k] if k < 2 else local_views[:, k - 2]
        augmented = []
        for j, view in enumerate(current):
            gamma, sigma = parameters[j, k].tolist()
            x = (view.clamp(-1000, 1000) + 1000) / 2000
            augmented.append(
                augment_intensity(
                    x, gamma if gamma > 0 else None, sigma if sigma > 0 else None
                )
            )
        views.append(torch.stack(augmented))
    return torch.cat(views[:2]), torch.cat(views[2:])


class MultiCropDataset(Dataset):
    def __init__(self, rows, local_views=8, seed=42):
        self.rows = rows
        self.local_views = local_views
        self.seed = seed
        self.epoch = 0
        if any(row.get("ct_units") != "HU" for row in rows):
            raise ValueError("Stage 1 expects preprocessed CT in HU")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = sitk.ReadImage(self.rows[index]["ct_path"])
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise ValueError("Invalid CT")
        rng = random.Random(self.seed + 1000003 * self.epoch + 97 * index)
        crops = [resized_crop(array, 224, (0.32, 1), rng) for _ in range(2)]
        crops.extend(
            resized_crop(array, 96, (0.05, 0.32), rng) for _ in range(self.local_views)
        )
        parameters = []
        for k in range(len(crops)):
            gamma = rng.uniform(0.7, 1.5) if rng.random() < 0.8 else -1
            blur_probability = 1.0 if k == 0 else 0.1 if k == 1 else 0.5
            sigma = rng.uniform(0.1, 2) if rng.random() < blur_probability else -1
            parameters.append([gamma, sigma])
        return {
            "global": torch.stack(crops[:2]),
            "local": torch.stack(crops[2:]),
            "params": torch.tensor(parameters),
        }


def global_masks(batch, shape=(3, 28, 28), device="cpu", seed=None):
    rng = random.Random(seed)
    result = np.zeros((batch, *shape), dtype=bool)
    count = math.prod(shape)
    for i in rng.sample(range(batch), batch // 2):
        target = int(count * rng.uniform(0.1, 0.5))
        mask = result[i]
        for _ in range(100):
            remaining = target - int(mask.sum())
            if remaining <= 0:
                break
            volume = rng.uniform(min(4, remaining), remaining)
            rhw = math.exp(rng.uniform(math.log(0.3), math.log(1 / 0.3)))
            rhd = math.exp(rng.uniform(math.log(0.3), math.log(1 / 0.3)))
            hh = max(1, round((volume * rhw * rhd) ** (1 / 3)))
            ww = max(1, round(hh / rhw))
            dd = max(1, round(hh / rhd))
            dd, hh, ww = [min(v, s) for v, s in zip((dd, hh, ww), shape)]
            z, y, x = [rng.randrange(s - v + 1) for v, s in zip((dd, hh, ww), shape)]
            region = mask[z : z + dd, y : y + hh, x : x + ww]
            if int((~region).sum()) <= remaining:
                region[:] = True
        remaining = target - int(mask.sum())
        if remaining > 0:
            flat = mask.reshape(-1)
            available = np.flatnonzero(~flat).tolist()
            flat[rng.sample(available, remaining)] = True
    return torch.from_numpy(result.reshape(batch, count)).to(device)
