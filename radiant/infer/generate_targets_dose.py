import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from radiant.data import PretrainDataset, load_manifest
from radiant.data.io import write_volume
from radiant.models import RadiantStage2
from radiant.train.common import device_from
from radiant.infer.common import load_model_weights, resolve_checkpoint


def parse_tuple(value, cast=int):
    return tuple(cast(x) for x in value.split(","))


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--manifest_out", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--checkpoint_name", default="latest.pt")
    p.add_argument("--checkpoint_key", default="model")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", default="96,336,336")
    p.add_argument("--ct_clip", default="-1000,1000")
    p.add_argument("--patch_size", default="4,4,4")
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depths", default="2,2,24,2")
    p.add_argument("--num_heads", default="12,24,48,96")
    p.add_argument("--window_size", default="4,7,7")
    p.add_argument("--output_format", choices=["npy", "nrrd"], default="npy")
    p.add_argument("--mask_threshold", type=float, default=0.5)
    return p


def main():
    args = make_parser().parse_args()
    device = device_from(args.device)
    image_size = parse_tuple(args.image_size)
    ct_clip = parse_tuple(args.ct_clip, float)
    patch_size = parse_tuple(args.patch_size)
    depths = parse_tuple(args.depths)
    num_heads = parse_tuple(args.num_heads)
    window_size = parse_tuple(args.window_size)
    checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoint_dir, args.checkpoint_name)

    frame = load_manifest(args.manifest, ["patient_id", "ct_path"])
    row_lookup = {str(row["patient_id"]): row.dropna().to_dict() for _, row in frame.iterrows()}
    loader = DataLoader(PretrainDataset(frame, image_size=image_size, ct_clip=ct_clip), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = RadiantStage2(encoder_kwargs={"patch_size": patch_size, "embed_dim": args.embed_dim, "depths": depths, "num_heads": num_heads, "window_size": window_size}).to(device)
    load_model_weights(model, checkpoint, key=args.checkpoint_key, strict=args.strict)
    model.eval()

    out = Path(args.output_dir)
    suffix = f".{args.output_format}"
    rows = []
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(device, non_blocking=True)
            y = model(ct)
            gtv = torch.sigmoid(y["gtv_logits"]).cpu()
            ctv = torch.sigmoid(y["ctv_logits"]).cpu()
            dose = y["dose"].cpu()
            for i, pid in enumerate(batch["patient_id"]):
                gtv_prob_path = out / f"{pid}_pred_gtv_prob{suffix}"
                ctv_prob_path = out / f"{pid}_pred_ctv_prob{suffix}"
                gtv_path = out / f"{pid}_pred_gtv_mask{suffix}"
                ctv_path = out / f"{pid}_pred_ctv_mask{suffix}"
                dose_path = out / f"{pid}_pred_dose{suffix}"
                gtv_prob = gtv[i, 0].numpy()
                ctv_prob = ctv[i, 0].numpy()
                write_volume(gtv_prob_path, gtv_prob)
                write_volume(ctv_prob_path, ctv_prob)
                write_volume(gtv_path, (gtv_prob >= args.mask_threshold).astype("float32"))
                write_volume(ctv_path, (ctv_prob >= args.mask_threshold).astype("float32"))
                write_volume(dose_path, dose[i, 0].numpy())
                row = dict(row_lookup.get(str(pid), {"patient_id": pid}))
                row.update({
                    "patient_id": pid,
                    "pred_gtv_path": str(gtv_path),
                    "pred_ctv_path": str(ctv_path),
                    "pred_gtv_prob_path": str(gtv_prob_path),
                    "pred_ctv_prob_path": str(ctv_prob_path),
                    "pred_target_path": str(ctv_path),
                    "generated_dose_path": str(dose_path),
                })
                rows.append(row)
    if args.manifest_out:
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.manifest_out, index=False)


if __name__ == "__main__":
    main()
