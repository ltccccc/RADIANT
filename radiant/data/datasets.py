from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from .io import read_volume
from .preprocessing import normalize_ct, normalize_dose, pad_or_center_crop, ensure_binary, random_flip
from .roi import crop_around_mask, dilate_mask, mask_to_binary


def present(value):
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if not isinstance(value, str):
        value = str(value)
    return bool(value) and value.lower() != "nan"


class BaseRadiantDataset(Dataset):
    def __init__(self, manifest, image_size=(96, 336, 336), ct_clip=(-1000, 1000), augment=False):
        self.frame = manifest.reset_index(drop=True)
        self.image_size = tuple(image_size)
        self.ct_clip = tuple(ct_clip)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def _tensor(self, x):
        return torch.from_numpy(np.ascontiguousarray(x[None].astype(np.float32)))

    def _load_ct(self, row):
        ct, _ = read_volume(row.ct_path)
        return normalize_ct(ct, self.ct_clip)


class PretrainDataset(BaseRadiantDataset):
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        ct = pad_or_center_crop(self._load_ct(row), self.image_size)
        return {"ct": self._tensor(ct), "patient_id": str(row.patient_id)}


class Stage2Dataset(BaseRadiantDataset):
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        ct = self._load_ct(row)
        items = [ct]
        gtv = self._optional_mask(row, "gtv_path", ct.shape)
        ctv = self._optional_mask(row, "ctv_path", ct.shape)
        ptv = self._optional_mask(row, "ptv_path", ct.shape)
        dose = self._optional_dose(row, ct.shape)
        if self.augment:
            items = random_flip([ct, gtv, ctv, ptv, dose])
            ct, gtv, ctv, ptv, dose = items
        ct = pad_or_center_crop(ct, self.image_size)
        gtv = pad_or_center_crop(gtv, self.image_size)
        ctv = pad_or_center_crop(ctv, self.image_size)
        ptv = pad_or_center_crop(ptv, self.image_size)
        dose = pad_or_center_crop(dose, self.image_size)
        return {
            "ct": self._tensor(ct),
            "gtv": self._tensor(gtv),
            "ctv": self._tensor(ctv),
            "ptv": self._tensor(ptv),
            "dose": self._tensor(dose),
            "has_gtv": torch.tensor(present(getattr(row, "gtv_path", "")), dtype=torch.bool),
            "has_ctv": torch.tensor(present(getattr(row, "ctv_path", "")), dtype=torch.bool),
            "has_ptv": torch.tensor(present(getattr(row, "ptv_path", "")), dtype=torch.bool),
            "has_dose": torch.tensor(present(getattr(row, "dose_path", "")), dtype=torch.bool),
            "patient_id": str(row.patient_id),
        }

    def _optional_mask(self, row, col, shape):
        path = getattr(row, col, "")
        if present(path):
            x, _ = read_volume(path)
            return ensure_binary(x)
        return np.zeros(shape, dtype=np.float32)

    def _optional_dose(self, row, shape):
        path = getattr(row, "dose_path", "")
        if present(path):
            x, _ = read_volume(path)
            return normalize_dose(x, getattr(row, "prescription_dose", 0))
        return np.zeros(shape, dtype=np.float32)


class DownstreamDataset(BaseRadiantDataset):
    def __init__(self, manifest, image_size=(96, 336, 336), ct_clip=(-1000, 1000), augment=False, roi_mode="auto", spacing=(2, 2, 5), margin_mm=30, mask_threshold=0.5, restrict_to_provided_roi=True):
        super().__init__(manifest, image_size, ct_clip, augment)
        self.roi_mode = roi_mode
        self.spacing = spacing
        self.margin_mm = margin_mm
        self.mask_threshold = mask_threshold
        self.restrict_to_provided_roi = restrict_to_provided_roi

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        ct = self._load_ct(row)
        dose, _ = read_volume(row.generated_dose_path)
        dose = dose.astype(np.float32)
        mask, source = self._roi_mask(row, ct.shape)
        ct = crop_around_mask(ct, mask, self.image_size)
        dose = crop_around_mask(dose, mask, self.image_size)
        if source == "provided" and self.restrict_to_provided_roi:
            roi = crop_around_mask(mask, mask, self.image_size)
            ct = ct * roi
            dose = dose * roi
        if self.augment:
            ct, dose = random_flip([ct, dose])
        out = {"ct": self._tensor(ct), "dose": self._tensor(dose), "patient_id": str(row.patient_id)}
        if "label" in self.frame.columns:
            out["label"] = torch.tensor(int(row.label), dtype=torch.long)
        if "time" in self.frame.columns and "event" in self.frame.columns:
            out["time"] = torch.tensor(float(row.time), dtype=torch.float32)
            out["event"] = torch.tensor(float(row.event), dtype=torch.float32)
        return out

    def _roi_mask(self, row, shape):
        if self.roi_mode == "full":
            return np.ones(shape, dtype=np.float32), "full"
        if self.roi_mode != "tumor":
            mask = self._provided_roi(row)
            if mask is not None:
                return mask, "provided"
        path = getattr(row, "pred_target_path", "")
        if present(path):
            mask, _ = read_volume(path)
            return dilate_mask(mask, self.margin_mm, self.spacing, self.mask_threshold), "tumor"
        return np.ones(shape, dtype=np.float32), "full"

    def _provided_roi(self, row):
        columns = {
            "auto": ("roi_mask_path", "lung_mask_path", "bowel_mask_path", "rectum_mask_path", "bladder_mask_path"),
            "roi": ("roi_mask_path",),
            "lung": ("lung_mask_path",),
            "bowel": ("bowel_mask_path",),
            "rectum": ("rectum_mask_path",),
            "bladder": ("bladder_mask_path",),
        }.get(self.roi_mode, ())
        for col in columns:
            path = getattr(row, col, "")
            if present(path):
                mask, _ = read_volume(path)
                return mask_to_binary(mask, self.mask_threshold)
        return None
