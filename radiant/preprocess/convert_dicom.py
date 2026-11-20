import argparse
import json
import re
from pathlib import Path
import numpy as np
import pydicom
import SimpleITK as sitk
from ..data.geometry import from_array, check_image


def check_ct_series(slices):
    first = slices[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    normal = np.cross(orientation[:3], orientation[3:])
    if not np.allclose(
        np.stack([orientation[:3], orientation[3:], normal])
        @ np.stack([orientation[:3], orientation[3:], normal]).T,
        np.eye(3),
        atol=1e-5,
    ):
        raise ValueError("CT directions must be orthonormal")
    positions = np.asarray([s.ImagePositionPatient for s in slices], dtype=float)
    for item in slices:
        if (
            item.Modality != "CT"
            or not np.allclose(item.ImageOrientationPatient, orientation, atol=1e-5)
            or not np.allclose(item.PixelSpacing, first.PixelSpacing, atol=1e-5)
            or (item.Rows, item.Columns) != (first.Rows, first.Columns)
        ):
            raise ValueError("CT series contains inconsistent planes")
    if len(positions) > 1:
        steps = np.diff(positions, axis=0)
        projected = steps @ normal
        if (
            np.any(np.abs(projected) < 1e-5)
            or not np.allclose(steps, projected[:, None] * normal, atol=1e-3)
            or not np.allclose(projected, projected[0], atol=1e-3)
        ):
            raise ValueError(
                "CT planes must form a regular orthogonal grid; irregular spacing or gantry tilt requires explicit conversion"
            )


def dose_image(ds):
    values = np.asarray(ds.pixel_array, dtype=np.float32)
    scaling = float(ds.DoseGridScaling)
    if (
        values.ndim != 3
        or values.shape[0] < 2
        or not np.isfinite(scaling)
        or scaling <= 0
    ):
        raise ValueError("A positive-scale multi-plane RTDOSE is required")
    orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
    origin = np.asarray(ds.ImagePositionPatient, dtype=float)
    offsets = np.asarray(ds.GridFrameOffsetVector, dtype=float)
    if len(offsets) != len(values) or not np.isfinite(offsets).all():
        raise ValueError("RTDOSE frame offsets do not match pixel data")
    if not np.isclose(offsets[0], 0, atol=1e-5):
        if not np.allclose(
            orientation, [1, 0, 0, 0, 1, 0], atol=1e-5
        ) or not np.isclose(offsets[0], origin[2], atol=1e-5):
            raise ValueError("Invalid RTDOSE frame-offset convention")
    steps = np.diff(offsets)
    if np.isclose(steps[0], 0) or not np.allclose(steps, steps[0], atol=1e-4):
        raise ValueError("Nonuniform dose grid requires explicit conversion")
    direction = np.column_stack(
        [
            orientation[:3],
            orientation[3:],
            np.cross(orientation[:3], orientation[3:]) * np.sign(steps[0]),
        ]
    )
    result = sitk.GetImageFromArray(values * scaling)
    result.SetOrigin(origin.tolist())
    result.SetSpacing(
        [float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), float(abs(steps[0]))]
    )
    result.SetDirection(direction.ravel().tolist())
    check_image(result)
    return result


def convert(ct_dir, series_uid, rtstruct, rtdose, roi_names, output, rtplan=None):
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in roi_names):
        raise ValueError("Unsafe ROI output name")
    if roi_names and not rtstruct:
        raise ValueError("ROI names require RTSTRUCT")
    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(ct_dir), series_uid))
    if not files:
        raise ValueError("CT series not found")
    slices = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    frames = {str(s.FrameOfReferenceUID) for s in slices}
    if len(frames) != 1:
        raise ValueError("Multiple CT frames")
    check_ct_series(slices)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    ct = reader.Execute()
    check_image(ct)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    sitk.WriteImage(sitk.Cast(ct, sitk.sitkFloat32), str(out / "ct.nrrd"), True)
    if rtstruct:
        from rt_utils import RTStructBuilder

        rt = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir), rt_struct_path=str(rtstruct)
        )
        source = {str(s.SOPInstanceUID): i for i, s in enumerate(rt.series_data)}
        if set(source) != {str(s.SOPInstanceUID) for s in slices}:
            raise ValueError(
                "RTSTRUCT conversion folder must contain exactly the selected CT series"
            )
        referenced = {
            str(x.FrameOfReferenceUID) for x in rt.ds.ReferencedFrameOfReferenceSequence
        }
        if referenced != frames:
            raise ValueError("CT/RTSTRUCT frame mismatch")
        for name, clinical_name in roi_names.items():
            mask = rt.get_roi_mask_by_name(clinical_name)
            a = np.stack(
                [mask[:, :, source[str(s.SOPInstanceUID)]] for s in slices]
            ).astype(np.uint8)
            if a.shape != sitk.GetArrayViewFromImage(ct).shape:
                raise ValueError("RTSTRUCT mask/CT shape mismatch")
            sitk.WriteImage(from_array(a, ct), str(out / (name + ".nrrd")), True)
    if rtdose:
        if not rtplan or not rtstruct:
            raise ValueError(
                "RTPLAN and RTSTRUCT are required to verify a dose-plan association"
            )
        plan = pydicom.dcmread(rtplan, stop_before_pixels=True)
        struct = pydicom.dcmread(rtstruct, stop_before_pixels=True)
        if str(struct.SOPInstanceUID) not in {
            str(v.ReferencedSOPInstanceUID) for v in plan.ReferencedStructureSetSequence
        }:
            raise ValueError("RTPLAN does not reference this RTSTRUCT")
        ds = pydicom.dcmread(rtdose)
        if str(plan.SOPInstanceUID) not in {
            str(v.ReferencedSOPInstanceUID) for v in ds.ReferencedRTPlanSequence
        }:
            raise ValueError("RTDOSE does not reference this RTPLAN")
        if (
            str(ds.FrameOfReferenceUID) not in frames
            or ds.DoseUnits != "GY"
            or ds.DoseSummationType != "PLAN"
        ):
            raise ValueError("A PLAN dose in Gy matching the CT frame is required")
        if ds.DoseType != "PHYSICAL":
            raise ValueError("Only physical dose is supported")
        sitk.WriteImage(dose_image(ds), str(out / "dose.nrrd"), True)
    (out / "conversion.json").write_text(
        json.dumps(
            {
                "ct_series_uid": series_uid,
                "frame_uid": list(frames)[0],
                "roi_names": roi_names,
                "ct_units": "HU",
                "dose_units": "Gy",
                "prescription_source": "User-verified plan metadata required separately",
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ct-dir", required=True)
    p.add_argument("--series-uid", required=True)
    p.add_argument("--rtstruct")
    p.add_argument("--rtdose")
    p.add_argument("--rtplan")
    p.add_argument("--gtv-name")
    p.add_argument("--ctv-name")
    p.add_argument("--oar", action="append", default=[])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    names = {k: v for k, v in {"gtv": args.gtv_name, "ctv": args.ctv_name}.items() if v}
    for item in args.oar:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            p.error("--oar requires NAME=CLINICAL_ROI_NAME")
        names["oar_" + key] = value
    convert(
        args.ct_dir,
        args.series_uid,
        args.rtstruct,
        args.rtdose,
        names,
        args.output,
        args.rtplan,
    )


if __name__ == "__main__":
    main()
