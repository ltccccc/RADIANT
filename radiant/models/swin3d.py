import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_chans=1, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, c, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (d, h, w)


class Mlp(nn.Module):
    def __init__(self, dim, ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, heads, window_size, drop=0.0):
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)
        size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, heads))
        self.register_buffer("relative_position_index", self._relative_index(window_size), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _relative_index(window_size):
        dz, dy, dx = [torch.arange(s) for s in window_size]
        coords = torch.stack(torch.meshgrid(dz, dy, dx, indexing="ij"))
        flat = coords.flatten(1)
        rel = flat[:, :, None] - flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 2] += window_size[2] - 1
        rel[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        rel[:, :, 1] *= 2 * window_size[2] - 1
        return rel.sum(-1)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(n, n, self.heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, n, n)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class WindowBlock3D(nn.Module):
    def __init__(self, dim, heads, window_size=(4, 7, 7), shift=False, drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift = tuple(s // 2 for s in window_size) if shift else (0, 0, 0)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, heads, window_size, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, drop=drop)

    def forward(self, x, resolution):
        b, l, c = x.shape
        d, h, w = resolution
        y = self.norm1(x).view(b, d, h, w, c)
        if any(self.shift):
            y = torch.roll(y, shifts=tuple(-s for s in self.shift), dims=(1, 2, 3))
        windows, pads = self._partition(y)
        windows = windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], c)
        mask = self._attention_mask((d, h, w), pads, x.device) if any(self.shift) else None
        windows = self.attn(windows, mask)
        y = self._reverse(windows, b, c, resolution, pads)
        if any(self.shift):
            y = torch.roll(y, shifts=self.shift, dims=(1, 2, 3))
        y = y.reshape(b, l, c)
        x = x + y
        return x + self.mlp(self.norm2(x))

    def _partition(self, x):
        b, d, h, w, c = x.shape
        wd, wh, ww = self.window_size
        pd, ph, pw = (wd - d % wd) % wd, (wh - h % wh) % wh, (ww - w % ww) % ww
        x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
        d2, h2, w2 = d + pd, h + ph, w + pw
        x = x.view(b, d2 // wd, wd, h2 // wh, wh, w2 // ww, ww, c)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x, (pd, ph, pw)

    def _reverse(self, windows, b, c, resolution, pads):
        d, h, w = resolution
        pd, ph, pw = pads
        wd, wh, ww = self.window_size
        d2, h2, w2 = d + pd, h + ph, w + pw
        x = windows.view(b, d2 // wd, h2 // wh, w2 // ww, wd, wh, ww, c)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d2, h2, w2, c)
        return x[:, :d, :h, :w]

    def _attention_mask(self, resolution, pads, device):
        d, h, w = resolution
        pd, ph, pw = pads
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift
        d2, h2, w2 = d + pd, h + ph, w + pw
        img_mask = torch.zeros((1, d2, h2, w2, 1), device=device)
        cnt = 0
        slices = (
            (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)),
            (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)),
            (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)),
        )
        for ds in slices[0]:
            for hs in slices[1]:
                for ws in slices[2]:
                    img_mask[:, ds, hs, ws, :] = cnt
                    cnt += 1
        mask_windows, _ = self._partition(img_mask)
        mask_windows = mask_windows.view(-1, wd * wh * ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 8)
        self.reduction = nn.Linear(dim * 8, dim * 2, bias=False)

    def forward(self, x, resolution):
        b, l, c = x.shape
        d, h, w = resolution
        x = x.view(b, d, h, w, c)
        pd, ph, pw = d % 2, h % 2, w % 2
        x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
        d, h, w = x.shape[1:4]
        parts = [x[:, i::2, j::2, k::2] for i in range(2) for j in range(2) for k in range(2)]
        x = torch.cat(parts, dim=-1).view(b, -1, c * 8)
        return self.reduction(self.norm(x)), (d // 2, h // 2, w // 2)


class SwinStage3D(nn.Module):
    def __init__(self, dim, depth, heads, window_size, downsample=True, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([WindowBlock3D(dim, heads, window_size, shift=i % 2 == 1) for i in range(depth)])
        self.downsample = PatchMerging3D(dim) if downsample else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, resolution):
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(lambda y: block(y, resolution), x, use_reentrant=False)
            else:
                x = block(x, resolution)
        b, _, c = x.shape
        d, h, w = resolution
        feat = x.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        if self.downsample is None:
            return x, resolution, feat
        x, resolution = self.downsample(x, resolution)
        return x, resolution, feat


class SwinTransformer3D(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), in_chans=1, embed_dim=384, depths=(2, 2, 24, 2), num_heads=(12, 24, 48, 96), window_size=(4, 7, 7), use_checkpoint=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim)
        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.stages.append(SwinStage3D(embed_dim * 2**i, depth, num_heads[i], window_size, downsample=i < len(depths) - 1, use_checkpoint=use_checkpoint))
        self.feature_dims = [embed_dim * 2**i for i in range(len(depths))]

    def forward(self, x):
        x, resolution = self.patch_embed(x)
        features = []
        for stage in self.stages:
            x, resolution, feat = stage(x, resolution)
            features.append(feat)
        return features


def create_radiant_encoder(**kwargs):
    return SwinTransformer3D(**kwargs)
