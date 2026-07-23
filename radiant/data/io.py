from pathlib import Path
import numpy as np


def read_volume(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32), {}
    if path.suffix == ".npz":
        z = np.load(path)
        key = "data" if "data" in z.files else z.files[0]
        return z[key].astype(np.float32), {}
    if path.suffix == ".nrrd":
        import nrrd
        data, header = nrrd.read(str(path))
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
            header = dict(header)
            header["array_order"] = "DHW_from_NRRD_xyz"
        return data.astype(np.float32), header
    raise ValueError(f"Unsupported volume format: {path}")


def write_volume(path, array, header=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, array.astype(np.float32))
        return
    if path.suffix == ".nrrd":
        import nrrd
        payload = array.astype(np.float32)
        header = dict(header or {})
        if payload.ndim == 3:
            payload = np.transpose(payload, (2, 1, 0))
            header.pop("array_order", None)
        nrrd.write(str(path), payload, header)
        return
    raise ValueError(f"Unsupported output format: {path}")


def require_columns(frame, columns):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")
