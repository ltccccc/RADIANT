import csv
import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    p = Path(path).resolve()
    with p.open() as f:
        rows = json.load(f) if p.suffix == ".json" else list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    patients = {}
    for row in rows:
        if not all(row.get(k) for k in ["case_id", "patient_id", "split", "ct_path"]):
            raise ValueError("Missing case_id, patient_id, split or ct_path")
        if row["split"] not in ["train", "val", "test", "excluded"]:
            raise ValueError("Unknown split")
        if row["case_id"] in seen:
            raise ValueError("Duplicate case_id")
        seen.add(row["case_id"])
        if (
            row["patient_id"] in patients
            and patients[row["patient_id"]] != row["split"]
        ):
            raise ValueError("Patient appears in multiple splits")
        patients[row["patient_id"]] = row["split"]
        oars = row.get("oar_paths") or {}
        if isinstance(oars, str):
            oars = json.loads(oars)
        if not isinstance(oars, dict) or any(
            not re.fullmatch(r"[A-Za-z0-9_]+", name) or not value
            for name, value in oars.items()
        ):
            raise ValueError("oar_paths must map safe ROI names to binary mask paths")
        row["oar_paths"] = {
            name: str(
                Path(value)
                if Path(value).is_absolute()
                else (p.parent / value).resolve()
            )
            for name, value in oars.items()
        }
        for key, value in list(row.items()):
            if key.endswith("_path") and value:
                q = Path(value)
                row[key] = str(q if q.is_absolute() else (p.parent / q).resolve())
    return rows
