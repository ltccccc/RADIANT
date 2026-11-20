import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from .geometry import same_grid, distance_map, loss_region


class Stage2Dataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment
        for r in rows:
            if r.get("ct_units") != "HU" or r.get("dose_units") != "RxNormalized":
                raise ValueError(
                    "Preprocessed HU CT and prescription-normalized dose required"
                )
            if not all(
                r.get(k + "_path") for k in ["gtv", "ctv", "dose", "dose_valid"]
            ):
                raise ValueError(
                    "Joint training release requires complete CT/GTV/CTV/dose annotations"
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ims = {
            k: sitk.ReadImage(row[k + "_path"])
            for k in ["ct", "gtv", "ctv", "dose", "dose_valid"]
        }
        if not all(same_grid(ims["ct"], v) for v in ims.values()):
            raise ValueError("Prepared modalities do not share a physical grid")
        arrays = {
            k: sitk.GetArrayFromImage(v).astype(np.float32) for k, v in ims.items()
        }
        spacing = ims["ct"].GetSpacing()[::-1]
        if self.augment:
            for axis in range(3):
                if torch.rand(()) < 0.5:
                    arrays = {k: np.flip(v, axis).copy() for k, v in arrays.items()}
        arrays["distance"] = distance_map(arrays["ctv"] > 0, spacing)
        arrays["body"] = loss_region(arrays["ct"], arrays["ctv"]) & (
            arrays["dose_valid"] > 0
        )
        if not (arrays["dose_valid"][arrays["ctv"] > 0] > 0).all():
            raise ValueError("CTV outside dose coverage")
        arrays["ct"] = (np.clip(arrays["ct"], -1000, 1000) + 1000) / 2000
        result = {
            k: torch.from_numpy(np.ascontiguousarray(v))[None].float()
            for k, v in arrays.items()
        }
        result["spacing_dhw"] = torch.tensor(spacing)
        result["case_id"] = row["case_id"]
        return result
