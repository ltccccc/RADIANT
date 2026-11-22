import argparse
import json
import re
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from ..data.geometry import check_image, grid, resample, crop_pad, from_array
from ..data.manifest import read_manifest, sha256

'''
把每个病例的原始数据整理成模型可以使用的格式：以 CT 的空间坐标为基准，
对齐 GTV、CTV、Dose map 和可选的 OAR masks，CT 和剂量采用线性插值，轮廓采用最近邻插值；
将 CT 强度截断到 −1000～1000 HU，将剂量除以该患者的处方剂量，并保留大于 1 的热点；
对所有数据同步进行中心裁剪或补边，默认得到 96×336×336（深×高×宽） 的尺寸，同时正确更新空间原点。
'''


def prepare_case(row, output, spacing_xyz=None, shape_dhw=(96, 336, 336)):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["case_id"]) or row["case_id"] in [
        ".",
        "..",
    ]:
        raise ValueError("Unsafe case_id")
    if row.get("ct_units") != "HU":
        raise ValueError(
            "ct_units must explicitly be HU; exported NRRD is never rescaled twice"
        )
    ct = sitk.ReadImage(row["ct_path"], sitk.sitkFloat32)
    check_image(ct)
    reference = grid(ct, spacing_xyz)
    images = {"ct": resample(ct, reference, default=-1000)}
    for name in ["gtv", "ctv"]:
        if row.get(name + "_path"):
            image = sitk.ReadImage(row[name + "_path"])
            images[name] = resample(image, reference, label=True)
            if not sitk.GetArrayViewFromImage(images[name]).any():
                raise ValueError(name + " is empty after alignment")
    for name, path in row.get("oar_paths", {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("Unsafe OAR name")
        images["oar_" + name] = resample(sitk.ReadImage(path), reference, label=True)
    if row.get("dose_path"):
        rx = float(row["prescription_gy"]) #处方剂量
        if not np.isfinite(rx) or rx <= 0:
            raise ValueError("Verified positive prescription_gy required")
        units = row.get("dose_units")
        if units not in ["Gy", "cGy", "RxNormalized"]:
            raise ValueError("Explicit dose units required")
        dose = sitk.ReadImage(row["dose_path"], sitk.sitkFloat32)
        check_image(dose)
        raw = sitk.GetArrayFromImage(dose)
        if (raw < 0).any():
            raise ValueError("Negative reference dose")
        valid = from_array(np.ones(raw.shape, dtype=np.uint8), dose)
        coverage = sitk.GetArrayFromImage(resample(valid, reference, True)) > 0
        if (
            "ctv" in images
            and not coverage[sitk.GetArrayViewFromImage(images["ctv"]) > 0].all()
        ):
            raise ValueError("Dose does not cover CTV")
        images["dose_valid"] = resample(valid, reference, True)
        images["dose"] = (
            resample(dose, reference)
            * ({"Gy": 1 / rx, "cGy": 0.01 / rx, "RxNormalized": 1}[units])
        )
    original_volumes = {
        k: np.count_nonzero(sitk.GetArrayViewFromImage(images[k]))
        for k in ["gtv", "ctv"]
        if k in images
    }
    if shape_dhw is not None:
        images = {
            k: crop_pad(v, shape_dhw, fill=-1000 if k == "ct" else 0)
            for k, v in images.items()
        }
    for k, count in original_volumes.items():
        if np.count_nonzero(sitk.GetArrayViewFromImage(images[k])) != count:
            raise ValueError(
                k
                + " is truncated by the fixed field of view; review the crop before training"
            )
    for k in ["gtv", "ctv"]:
        if k in images and not sitk.GetArrayViewFromImage(images[k]).any():
            raise ValueError(k + " lost by cropping")
    hu = sitk.GetArrayFromImage(images["ct"])
    images["ct"] = from_array(np.clip(hu, -1000, 1000).astype(np.float32), images["ct"])
    destination = Path(output) / row["case_id"]
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(row)
    result["oar_paths"] = {}
    for k, v in images.items():
        p = destination / (k + ".nrrd")
        sitk.WriteImage(v, str(p), True)
        if k.startswith("oar_"):
            result["oar_paths"][k[4:]] = str(p.resolve())
        else:
            result[k + "_path"] = str(p.resolve())
    result["dose_units"] = "RxNormalized" if "dose" in images else ""
    result["ct_units"] = "HU"
    metadata = {
        "source": row,
        "source_sha256": {
            k: sha256(v) for k, v in row.items() if k.endswith("_path") and v
        },
        "spacing_xyz": images["ct"].GetSpacing(),
        "origin_xyz": images["ct"].GetOrigin(),
        "direction": images["ct"].GetDirection(),
        "shape_dhw": sitk.GetArrayViewFromImage(images["ct"]).shape,
    }
    metadata["source_sha256"].update(
        {
            "oar_paths." + name: sha256(path)
            for name, path in row.get("oar_paths", {}).items()
        }
    )
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--spacing-xyz", nargs=3, type=float)
    p.add_argument("--shape-dhw", nargs=3, type=int, default=[96, 336, 336])
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("Prepared manifest already exists")
    rows = read_manifest(args.manifest)
    result = [
        prepare_case(r, root, args.spacing_xyz, args.shape_dhw)
        for r in rows
        if r["split"] != "excluded"
    ]
    (root / "manifest.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
