import pandas as pd
from .io import require_columns


def load_manifest(path, required=None):
    frame = pd.read_csv(path)
    if required:
        require_columns(frame, required)
    return frame


def existing_path_columns(frame, columns):
    missing = {}
    for col in columns:
        if col not in frame.columns:
            continue
        paths = frame[col].dropna().astype(str)
        bad = [p for p in paths if p and p.lower() != "nan" and not pd.io.common.file_exists(p)]
        if bad:
            missing[col] = bad[:10]
    if missing:
        raise FileNotFoundError(f"Missing files in manifest: {missing}")


def split_manifest(frame, val_fraction=0.2, seed=42, group_col="patient_id"):
    ids = frame[group_col].drop_duplicates().sample(frac=1.0, random_state=seed)
    n_val = max(int(len(ids) * val_fraction), 1)
    val_ids = set(ids.iloc[:n_val])
    train = frame[~frame[group_col].isin(val_ids)].reset_index(drop=True)
    val = frame[frame[group_col].isin(val_ids)].reset_index(drop=True)
    return train, val
