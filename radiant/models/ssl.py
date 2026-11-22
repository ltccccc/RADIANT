import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ProjectionHead(nn.Module):
    def __init__(self, dim, prototypes, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck, prototypes, bias=False)
        )
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.last_layer(F.normalize(self.mlp(x), dim=-1, eps=1e-12))


def tokens(backbone, x, masks=None, recompute=False):
    x = backbone._prepare_tokens_with_masks(x, masks)
    for block in backbone.blocks:
        x = (
            checkpoint(block, x, use_reentrant=False)
            if recompute and torch.is_grad_enabled()
            else block(x)
        )
    x = backbone.norm(x)
    return x[:, 0], x[:, 1 + backbone.num_register_tokens :]


def koleo(x):
    if len(x) < 2:
        raise ValueError(
            "KoLeo requires at least two patients per global view on each rank"
        )
    x = F.normalize(x.float(), dim=-1)
    similarity = (x @ x.T).detach()
    similarity.fill_diagonal_(-float("inf"))
    nearest = x[similarity.argmax(1)]
    return -torch.log(
        torch.linalg.vector_norm(x - nearest, dim=1).clamp_min(1e-8)
    ).mean()


class SSLModel(nn.Module):
    def __init__(self, backbone, prototypes=65536):
        super().__init__()
        self.backbone = backbone
        self.dino_head = ProjectionHead(backbone.embed_dim, prototypes)
        self.ibot_head = ProjectionHead(backbone.embed_dim, prototypes)
        self.register_buffer("center_cls", torch.zeros(1, prototypes))
        self.register_buffer("center_patch", torch.zeros(1, prototypes))

    def loss(self, global_views, local_views, masks, teacher, temperature, chunk=128):
        if len(global_views) % 2 or len(local_views) % (len(global_views) // 2):
            raise ValueError("Invalid multicrop batch")
        with torch.no_grad():
            tc, tp = tokens(teacher.backbone, global_views)
            teacher_logits = teacher.dino_head(tc).float()
            teacher_prob = F.softmax(
                (teacher_logits - self.center_cls) / temperature, -1
            ).chunk(2)
        sc, sp = tokens(self.backbone, global_views, masks, True)
        lc, _ = tokens(self.backbone, local_views, recompute=True)
        sg = self.dino_head(sc).float().chunk(2)
        sl = (
            self.dino_head(lc)
            .float()
            .chunk(len(local_views) // (len(global_views) // 2))
        )

        def ce(logits, target):
            return -(F.log_softmax(logits / 0.1, -1) * target).sum(-1)

        pieces = [ce(sg[0], teacher_prob[1]).mean(), ce(sg[1], teacher_prob[0]).mean()]
        for view in sl:
            pieces.extend(
                [ce(view, teacher_prob[0]).mean(), ce(view, teacher_prob[1]).mean()]
            )
        dino = torch.stack(pieces).mean()
        weights = masks.sum(-1).clamp_min(1).reciprocal().float()[:, None].expand_as(
            masks
        )[masks] / len(masks)
        sp, tp = sp[masks], tp[masks]
        ibot = sc.sum() * 0
        patch_sum = torch.zeros_like(self.center_patch)
        for offset in range(0, len(sp), chunk):
            s, t, weight = (
                sp[offset : offset + chunk],
                tp[offset : offset + chunk],
                weights[offset : offset + chunk],
            )
            with torch.no_grad():
                patch_sum += teacher.ibot_head(t).float().sum(0, keepdim=True)

            def term(s, t, weight):
                with torch.no_grad():
                    target = F.softmax(
                        (teacher.ibot_head(t).float() - self.center_patch)
                        / temperature,
                        -1,
                    )
                return (ce(self.ibot_head(s).float(), target) * weight).sum()

            ibot = ibot + checkpoint(term, s, t, weight, use_reentrant=False)
        regularizer = sum(koleo(x) for x in sc.chunk(2))
        return (
            dino + ibot + 0.1 * regularizer,
            {
                "dino": dino.detach(),
                "ibot": ibot.detach(),
                "koleo": regularizer.detach(),
            },
            [
                teacher_logits.sum(0, keepdim=True),
                len(teacher_logits),
                patch_sum,
                len(sp),
            ],
        )
