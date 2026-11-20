import json
import math
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from .geometry import same_grid, resample, crop_pad, check_image
from .manifest import read_manifest, sha256
from .roi import roi_grid


def check_clinical_manifest(rows, task, use_immunotherapy=False):
    if task not in ["classification", "survival"]:
        raise ValueError("Unsupported clinical task")
    seen = set()
    for row in rows:
        if str(row["patient_id"]) in seen:
            raise ValueError("Select one study per patient for this endpoint")
        seen.add(str(row["patient_id"]))
        if row["split"] == "excluded":
            continue
        if row.get("ct_units") != "HU" or not row.get("predicted_dose_path"):
            raise ValueError("HU CT and an explicit predicted_dose_path are required")
        fields = ["label"] if task == "classification" else ["time", "event"]
        if use_immunotherapy:
            fields.append("immunotherapy")
        for field in fields:
            value = float(row[field])
            if (
                not math.isfinite(value)
                or (
                    field in ["label", "event", "immunotherapy"] and value not in [0, 1]
                )
                or (field == "time" and value < 0)
            ):
                raise ValueError("Invalid clinical field: " + field)


def check_exposure(rows, manifests):
    if not manifests:
        raise ValueError(
            "Provide the Stage 1/2 foundation manifests for downstream exclusion checks"
        )
    downstream = {str(row["patient_id"]) for row in rows if row["split"] != "excluded"}
    for path in manifests:
        foundation = {
            str(row["patient_id"])
            for row in read_manifest(path)
            if row["split"] != "excluded"
        }
        if downstream & foundation:
            raise ValueError("Downstream patients overlap the foundation-model cohort")


class ClinicalDataset(Dataset):
    def __init__(self, rows, task, use_immunotherapy=False, generator_sha256=None):
        check_clinical_manifest(rows, task, use_immunotherapy)
        self.rows, self.task = rows, task
        self.use_immunotherapy = use_immunotherapy
        generator_ids = set()
        for row in rows:
            if not row.get("dose_provenance_path"):
                declared = row.get("dose_generator_sha256")
                if declared:
                    generator_ids.add(declared)
                continue
            with open(row["dose_provenance_path"]) as stream:
                provenance = json.load(stream)
            if (
                provenance.get("source") != "RADIANT"
                or provenance.get("conditioning") != "predicted_ctv"
                or provenance.get("dose_units") != "RxNormalized"
            ):
                raise ValueError(
                    "Dose must be generated from the predicted-CTV cascade"
                )
            for field, name in [
                ("ct_path", "input_ct_sha256"),
                ("predicted_dose_path", "dose_sha256"),
            ]:
                if sha256(row[field]) != provenance.get(name):
                    raise ValueError("Generated dose/CT provenance mismatch")
            generator_ids.add(provenance.get("generator_checkpoint_sha256"))
        if (
            len(generator_ids) > 1
            or None in generator_ids
            or (
                generator_sha256
                and generator_ids
                and generator_ids != {generator_sha256}
            )
        ):
            raise ValueError(
                "All cases must use the same specified fixed dose generator"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ct = sitk.ReadImage(row["ct_path"], sitk.sitkFloat32)
        dose = sitk.ReadImage(row["predicted_dose_path"], sitk.sitkFloat32)
        check_image(ct)
        check_image(dose)
        if not same_grid(ct, dose):
            raise ValueError(
                "Clinical CT and generated dose must share a physical grid"
            )
        if row.get("roi_path"):
            mask = sitk.ReadImage(row["roi_path"])
            margin = row.get("roi_margin_mm")
            ct = roi_grid(ct, mask, 30.0 if margin in [None, ""] else float(margin))
            dose = resample(dose, ct)
        shape = tuple(
            int(math.ceil(n / p) * p) for n, p in zip(ct.GetSize()[::-1], (4, 8, 8))
        )
        ct, dose = crop_pad(ct, shape, fill=-1000), crop_pad(dose, shape, fill=0)
        a, d = sitk.GetArrayFromImage(ct), sitk.GetArrayFromImage(dose)
        if not np.isfinite(a).all() or not np.isfinite(d).all() or (d < 0).any():
            raise ValueError("Nonfinite CT or invalid generated dose")
        result = {
            "ct": torch.from_numpy((np.clip(a, -1000, 1000) + 1000) / 2000)[None],
            "dose": torch.from_numpy(d)[None],
            "case_id": row["case_id"],
            "patient_id": row["patient_id"],
        }
        if self.task == "classification":
            result["label"] = torch.tensor(float(row["label"]))
        else:
            result["time"] = torch.tensor(float(row["time"]), dtype=torch.float64)
            result["event"] = torch.tensor(float(row["event"]))
        if self.use_immunotherapy:
            result["immunotherapy"] = torch.tensor(float(row["immunotherapy"]))
        return result
