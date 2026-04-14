import torch
from torch import nn
from baseline_adapter import ARPredictor, ConditionalBlock, Transformer


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class InverseDynamicsModel(nn.Module):
    """Infer a macro-action from current and target latents."""

    def __init__(self, embed_dim: int, macro_action_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, macro_action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        assert z_t.ndim == 2 and z_target.ndim == 2, "Expected (B, D) tensors"
        assert z_t.shape == z_target.shape, "z_t and z_target must match shape"
        assert z_t.shape[-1] == self.embed_dim, "Unexpected embed dim"
        x = torch.cat([z_t, z_target], dim=-1)
        return self.net(x)


class ConditionedSingleStepPredictor(nn.Module):
    """Single-step predictor used at Level-2 (tactical) and Level-3 (strategic)."""

    def __init__(
        self,
        embed_dim: int,
        macro_action_dim: int,
        depth: int = 3,
        heads: int = 8,
        mlp_dim: int = 1024,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_proj = nn.Linear(macro_action_dim, embed_dim)
        self.anchor_proj = nn.Linear(embed_dim, embed_dim)
        self.transformer = Transformer(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=ConditionalBlock,
        )

    def forward(
        self,
        z_current: torch.Tensor,
        macro_action: torch.Tensor,
        z_anchor: torch.Tensor = None,
    ) -> torch.Tensor:
        assert z_current.ndim == 2 and macro_action.ndim == 2, "Expected (B, D) and (B, A)"
        _b, d = z_current.shape
        assert d == self.embed_dim, "Unexpected embed dim in z_current"

        tokens = z_current.unsqueeze(1)

        cond = self.action_proj(macro_action)
        if z_anchor is not None:
            assert z_anchor.shape == z_current.shape, "z_anchor must be (B,D)"
            cond = cond + self.anchor_proj(z_anchor)

        cond = cond.unsqueeze(1).expand_as(tokens)
        out = self.transformer(tokens, cond)
        return out[:, 0]


class ARPredictorAnchored(ARPredictor):
    """Anchored autoregressive predictor used at Level-1 (reactive rollout)."""

    def __init__(self, *, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.anchor_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor, z_anchor: torch.Tensor = None):
        assert x.ndim == 3 and c.ndim == 3, "x and c must be (B,T,D)"
        if z_anchor is not None:
            assert z_anchor.ndim == 2, "z_anchor must be (B,D)"
            assert z_anchor.shape[0] == x.shape[0], "Batch mismatch on anchor"
            a = self.anchor_proj(z_anchor).unsqueeze(1).expand(-1, c.shape[1], -1)
            c = c + a
        return super().forward(x, c)
