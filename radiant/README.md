# radiant

Core Python package for RADIANT.

## Structure

- `data/`: volume IO, manifests, datasets, ROI cropping.
- `models/`: 3D Swin encoder, MAE, Stage 2 target/dose model, downstream model.
- `losses/`: MAE reconstruction, segmentation, dose, survival losses.
- `metrics/`: segmentation, dose, classification, survival, bootstrap metrics.
- `train/`: DDP training entrypoints for Stage 1, Stage 2, and downstream tasks.
- `infer/`: Stage 2 target/dose generation and downstream prediction.
- `config.py`: default configuration dataclasses.

## Training

Run from the project root:

```bash
torchrun --nproc_per_node=8 -m radiant.train.pretrain_mae --manifest /path/to/pretrain.csv --output_dir outputs/stage1
torchrun --nproc_per_node=8 -m radiant.train.refine_stage2 --manifest /path/to/stage2.csv --stage1_ckpt outputs/stage1/radiant_encoder_stage1.pt --output_dir outputs/stage2
torchrun --nproc_per_node=8 -m radiant.train.downstream --manifest /path/to/downstream.csv --encoder_ckpt outputs/stage2/radiant_encoder_stage2.pt --output_dir outputs/downstream
```

All training scripts support DDP, AMP, gradient accumulation, checkpoint resume with `--resume latest`, and rank-0-only checkpoint writing.

## Inference

```bash
python -m radiant.infer.generate_targets_dose --manifest /path/to/cases.csv --checkpoint_dir outputs/stage2 --checkpoint_name latest.pt --output_dir outputs/generated --manifest_out outputs/generated/manifest.csv
python -m radiant.infer.downstream --manifest outputs/generated/manifest.csv --checkpoint_dir outputs/downstream --checkpoint_name latest.pt --output_csv outputs/predictions.csv
```

## Defaults

- CT input clip: `-1000,1000`
- Stage 1 image size: `96,336,336`
- Stage 2 image size: `96,336,336`
- Downstream image size: `96,336,336`
- Encoder: `embed_dim=384`, `depths=2,2,24,2`, `window_size=4,7,7`
