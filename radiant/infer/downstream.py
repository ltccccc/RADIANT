import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from radiant.data import DownstreamDataset, load_manifest
from radiant.models import RadiantDownstream
from radiant.train.common import device_from
from radiant.infer.common import load_model_weights, resolve_checkpoint


def parse_tuple(value, cast=int):
    return tuple(cast(x) for x in value.split(","))


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--checkpoint_name", default="latest.pt")
    p.add_argument("--checkpoint_key", default="model")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--task", choices=["classification", "survival"], default="classification")
    p.add_argument("--roi_mode", choices=["auto", "roi", "tumor", "lung", "bowel", "rectum", "bladder", "full"], default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", default="96,336,336")
    p.add_argument("--ct_clip", default="-1000,1000")
    p.add_argument("--margin_mm", type=float, default=30.0)
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--restrict_to_provided_roi", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--patch_size", default="4,4,4")
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depths", default="2,2,24,2")
    p.add_argument("--num_heads", default="12,24,48,96")
    p.add_argument("--window_size", default="4,7,7")
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

    frame = load_manifest(args.manifest, ["patient_id", "ct_path", "generated_dose_path"])
    dataset = DownstreamDataset(frame, image_size=image_size, ct_clip=ct_clip, augment=False, roi_mode=args.roi_mode, margin_mm=args.margin_mm, mask_threshold=args.mask_threshold, restrict_to_provided_roi=args.restrict_to_provided_roi)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = RadiantDownstream(task=args.task, encoder_kwargs={"patch_size": patch_size, "embed_dim": args.embed_dim, "depths": depths, "num_heads": num_heads, "window_size": window_size}).to(device)
    load_model_weights(model, checkpoint, key=args.checkpoint_key, strict=args.strict)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(device, non_blocking=True)
            dose = batch["dose"].to(device, non_blocking=True)
            pred = model(ct, dose).detach().cpu()
            for i, pid in enumerate(batch["patient_id"]):
                row = {"patient_id": pid}
                if args.task == "classification":
                    logit = float(pred[i])
                    row["logit"] = logit
                    row["probability"] = float(torch.sigmoid(pred[i]))
                else:
                    row["risk"] = float(pred[i])
                rows.append(row)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
