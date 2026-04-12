# `ippo` Tasks (Model Core)

## Scope Ownership

- `module.py`
- `work/tasks/ippo/*`

## File-Level Goal

Add 3 reusable building blocks in `module.py`:

1. `InverseDynamicsModel`
2. `ConditionedSingleStepPredictor`
3. `ARPredictorAnchored`

Keep current LeWM path unchanged.

## Task 1: Implement `InverseDynamicsModel`

Edit target:

- add class after `MLP` in `module.py`

Implementation snippet:

```python
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
        return self.net(x)  # (B, macro_action_dim)
```

Done criteria:

- Output is `(B, macro_action_dim)` for random `(B, D)` input.
- No side effects on existing classes.

## Task 2: Implement `ConditionedSingleStepPredictor`

Edit target:

- add class after `InverseDynamicsModel` in `module.py`

Implementation snippet:

```python
class ConditionedSingleStepPredictor(nn.Module):
    """
    Predict one future latent from [z_current] or [z_current, z_anchor]
    conditioned by a macro-action through AdaLN blocks.
    """

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
        B, D = z_current.shape
        assert D == self.embed_dim, "Unexpected embed dim in z_current"

        tokens = z_current.unsqueeze(1)  # (B,1,D)
        if z_anchor is not None:
            assert z_anchor.shape == z_current.shape, "z_anchor must be (B,D)"
            tokens = torch.cat([tokens, z_anchor.unsqueeze(1)], dim=1)  # (B,2,D)

        cond = self.action_proj(macro_action).unsqueeze(1).expand_as(tokens)
        out = self.transformer(tokens, cond)
        return out[:, 0]  # (B,D)
```

Done criteria:

- Works for both L3 call (`z_anchor=None`) and L2 call (`z_anchor` set).

## Task 3: Implement `ARPredictorAnchored`

Edit target:

- add class below existing `ARPredictor` in `module.py`

Implementation snippet:

```python
class ARPredictorAnchored(ARPredictor):
    """ARPredictor with optional global anchor conditioning."""

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
```

Done criteria:

- If `z_anchor=None`, output matches base behavior.
- If `z_anchor` is used, no shape regressions.

## Task 4: Smoke Test Script

Create `work/tasks/ippo/smoke_module_test.md` with command and output summary.

Suggested command snippet:

```bash
python - <<'PY'
import torch
from module import InverseDynamicsModel, ConditionedSingleStepPredictor, ARPredictorAnchored

B,D,A,T = 4,192,64,3
id3 = InverseDynamicsModel(D, A)
pred = ConditionedSingleStepPredictor(D, A)
ar = ARPredictorAnchored(
    embed_dim=D, num_frames=T, depth=2, heads=4, mlp_dim=256,
    input_dim=D, hidden_dim=D, output_dim=D, dim_head=32
)
z = torch.randn(B,D); z2 = torch.randn(B,D)
a = id3(z, z2)
print("id3:", a.shape)
print("pred3:", pred(z, a).shape)
print("pred2:", pred(z, a, z_anchor=z2).shape)
x = torch.randn(B,T,D); c = torch.randn(B,T,D)
print("ar:", ar(x, c, z_anchor=z2).shape)
PY
```

Expected output shapes:

- `id3: torch.Size([4, 64])`
- `pred3: torch.Size([4, 192])`
- `pred2: torch.Size([4, 192])`
- `ar: torch.Size([4, 3, 192])`

## Handoff Artifact

- PR description section: `Module APIs` with the 3 class signatures.
