import types
import importlib
import importlib.util
import sys
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file


def attention_sdpa(self, x, attn_bias=None):
    if attn_bias is not None:
        raise ValueError("Packed attention is unsupported")
    b, n, c = x.shape
    q, k, v = (
        self.qkv(x)
        .reshape(b, n, 3, self.num_heads, c // self.num_heads)
        .permute(2, 0, 3, 1, 4)
        .unbind(0)
    )
    out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=self.attn_drop.p if self.training else 0
    )
    return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, c)))


def enable_sdpa(backbone):
    for block in backbone.blocks:
        block.attn.forward = types.MethodType(attention_sdpa, block.attn)


class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * (mask / keep)


def configure_drop_path(backbone, maximum):
    if not 0 <= maximum < 1:
        raise ValueError("DropPath must be in [0,1)")
    for i, block in enumerate(backbone.blocks):
        probability = maximum * i / max(1, len(backbone.blocks) - 1)
        block.drop_path1 = DropPath(probability) if probability else nn.Identity()
        block.drop_path2 = DropPath(probability) if probability else nn.Identity()
        block.sample_drop_ratio = probability


def load_backbone(model_dir, encoder_weights=None):
    folder = Path(model_dir).resolve()
    name = "_radiant_tapct_" + hashlib.sha256(str(folder).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    cls = importlib.import_module(name + ".modeling_tapct").TAPCTModel
    model, info = cls.from_pretrained(
        str(folder), local_files_only=True, output_loading_info=True
    )
    if any(
        info.get(k)
        for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
    ):
        raise ValueError(str(info))
    if encoder_weights:
        model.load_state_dict(load_file(encoder_weights), strict=True)
    backbone = model.model
    if (
        backbone.num_register_tokens != 4
        or len(backbone.blocks) != 12
        or backbone.embed_dim != 768
    ):
        raise ValueError("TAP-CT Base 3D with four registers is required")
    enable_sdpa(backbone)
    return backbone


class Encoder(nn.Module):
    def __init__(
        self, backbone, patch_size=(4, 8, 8), blocks=(3, 6, 9, 12), checkpointing=True
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = tuple(patch_size)
        self.blocks = tuple(blocks)
        self.embed_dim = backbone.embed_dim
        self.checkpointing = checkpointing

    def forward(self, ct):
        if any(n % p for n, p in zip(ct.shape[2:], self.patch_size)):
            raise ValueError(
                "Input spatial dimensions must be divisible by TAP-CT patch dimensions"
            )
        shape = tuple(n // p for n, p in zip(ct.shape[2:], self.patch_size))
        x = self.backbone._prepare_tokens_with_masks(ct, None)
        outputs = []
        for i, block in enumerate(self.backbone.blocks, 1):
            x = (
                checkpoint(block, x, use_reentrant=False)
                if self.checkpointing and self.training and torch.is_grad_enabled()
                else block(x)
            )
            if i in self.blocks:
                y = self.backbone.norm(x)[:, 1 + self.backbone.num_register_tokens :]
                outputs.append(
                    y.transpose(1, 2).reshape(ct.shape[0], self.embed_dim, *shape)
                )
        return outputs
