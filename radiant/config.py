from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    manifest: Path
    image_size: tuple[int, int, int] = (96, 336, 336)
    spacing: tuple[float, float, float] = (2.0, 2.0, 5.0)
    ct_clip: tuple[float, float] = (-1000.0, 1000.0)
    margin_mm: float = 30.0


@dataclass
class SwinConfig:
    patch_size: tuple[int, int, int] = (4, 4, 4)
    embed_dim: int = 384
    depths: tuple[int, int, int, int] = (2, 2, 24, 2)
    num_heads: tuple[int, int, int, int] = (12, 24, 48, 96)
    window_size: tuple[int, int, int] = (4, 7, 7)
    drop_path_rate: float = 0.2
    use_checkpoint: bool = False


@dataclass
class Stage1Config:
    epochs: int = 50
    mask_ratio: float = 0.75
    alpha_ssim: float = 0.5
    learning_rate: float = 1.0e-4
    weight_decay: float = 5.0e-2


@dataclass
class Stage2Config:
    epochs: int = 100
    freeze_encoder_epochs: int = 20
    lr_encoder: float = 1.0e-5
    lr_decoders: float = 1.0e-4
    weight_decay: float = 1.0e-2
    lambda_gtv: float = 1.0
    lambda_ctv: float = 1.0
    lambda_dose: float = 1.0
    grad_clip: float = 1.0


@dataclass
class DownstreamConfig:
    epochs: int = 20
    learning_rate: float = 1.0e-4
    dropout: float = 0.3
    hidden_dim: int = 512
    aspp_rates: tuple[int, int, int, int] = (1, 6, 12, 18)
    freeze_encoder: bool = True


@dataclass
class TrainConfig:
    output_dir: Path = Path("outputs")
    batch_size: int = 1
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    distributed: bool = False
    device: str = "cuda"


@dataclass
class RadiantConfig:
    data: DataConfig
    swin: SwinConfig = field(default_factory=SwinConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
