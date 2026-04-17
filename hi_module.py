from __future__ import annotations

import torch
from torch import nn


class LatentActionEncoder(nn.Module):
    """Encode variable-length primitive action chunks into latent macro-actions."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        model_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        self.max_seq_len = int(max_seq_len)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len + 1, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, latent_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        action_chunks: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            action_chunks: (B, T_chunk, A)
            action_mask: (B, T_chunk) boolean mask, True for valid tokens
        Returns:
            macro_actions: (B, latent_dim)
        """
        if action_chunks.ndim != 3:
            raise ValueError("action_chunks must be a 3D tensor with shape (B, T, A)")

        x = self.input_proj(action_chunks.float())
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.size(1) > self.max_seq_len + 1:
            raise ValueError(
                f"Action chunk length {x.size(1) - 1} exceeds max_seq_len={self.max_seq_len}"
            )
        x = x + self.pos_embedding[:, : x.size(1)]

        key_padding_mask = None
        if action_mask is not None:
            if action_mask.ndim != 2:
                raise ValueError("action_mask must be shape (B, T)")
            if action_mask.size(0) != b or action_mask.size(1) != action_chunks.size(1):
                raise ValueError("action_mask shape must match action_chunks (B, T)")
            valid = action_mask.to(dtype=torch.bool)
            cls_valid = torch.ones((b, 1), dtype=torch.bool, device=valid.device)
            full_valid = torch.cat([cls_valid, valid], dim=1)
            # Transformer expects True where tokens should be ignored.
            key_padding_mask = ~full_valid

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        macro = self.output_proj(h[:, 0])
        return macro
