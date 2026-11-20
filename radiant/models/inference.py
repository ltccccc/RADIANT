import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from ..data.geometry import check_image, from_array
from .encoder import load_backbone, Encoder
from .stage2 import RadiantStage2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ct", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RadiantStage2(
        Encoder(load_backbone(args.model_dir)), tuple(state["config"]["decoder_widths"])
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    ct = sitk.ReadImage(args.ct, sitk.sitkFloat32)
    check_image(ct)
    a = sitk.GetArrayFromImage(ct)
    x = torch.as_tensor((np.clip(a, -1000, 1000) + 1000) / 2000, device=device)[
        None, None
    ].float()
    spacing = torch.tensor([ct.GetSpacing()[::-1]], device=device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=state["config"].get("amp", True) and device.type == "cuda",
        ),
    ):
        out = model(x, spacing_dhw=spacing)
    if not all(torch.isfinite(value).all() for value in out.values()):
        raise FloatingPointError("Nonfinite cascade output")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for name in ["gtv", "ctv"]:
        mask = (out[name + "_logits"].sigmoid()[0, 0].cpu().numpy() >= 0.5).astype(
            np.uint8
        )
        sitk.WriteImage(
            from_array(mask, ct), str(root / (name + "_predicted.nrrd")), True
        )
    dose = out["dose"][0, 0].float().cpu().numpy()
    sitk.WriteImage(
        from_array(dose, ct), str(root / "dose_predicted_RxNormalized.nrrd"), True
    )


if __name__ == "__main__":
    main()
