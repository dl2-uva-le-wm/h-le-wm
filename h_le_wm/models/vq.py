from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    """Nearest-neighbor vector quantizer with straight-through gradients."""

    def __init__(self, *, num_codes: int, code_dim: int):
        super().__init__()
        if num_codes <= 1:
            raise ValueError("num_codes must be > 1")
        if code_dim <= 0:
            raise ValueError("code_dim must be > 0")
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.codebook = nn.Embedding(self.num_codes, self.code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def _nearest_code(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = latents.reshape(-1, self.code_dim)
        codebook = self.codebook.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        quantized = self.codebook(indices).reshape_as(latents)
        return quantized, indices.reshape(latents.shape[:-1])

    def forward(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        if latents.size(-1) != self.code_dim:
            raise ValueError(
                f"Expected latent dim {self.code_dim}, got {latents.size(-1)}"
            )

        quantized, indices = self._nearest_code(latents)
        quantized_st = latents + (quantized - latents).detach()
        codebook_loss = F.mse_loss(quantized, latents.detach())
        commitment_loss = F.mse_loss(latents, quantized.detach())

        one_hot = F.one_hot(indices.reshape(-1), num_classes=self.num_codes).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * avg_probs.clamp_min(1e-10).log()).sum())
        active_codes = (avg_probs > 0).float().sum()

        return {
            "quantized": quantized,
            "quantized_st": quantized_st,
            "indices": indices,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "active_codes": active_codes,
        }

    def quantize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.size(-1) != self.code_dim:
            raise ValueError(
                f"Expected latent dim {self.code_dim}, got {latents.size(-1)}"
            )
        quantized, _ = self._nearest_code(latents)
        return quantized

    def latents_to_codes(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.size(-1) != self.code_dim:
            raise ValueError(
                f"Expected latent dim {self.code_dim}, got {latents.size(-1)}"
            )
        _, indices = self._nearest_code(latents)
        return indices

    def codes_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices.to(dtype=torch.long))

    def code_probs_to_latents(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.size(-1) != self.num_codes:
            raise ValueError(
                f"Expected last dim {self.num_codes}, got {probs.size(-1)}"
            )
        return probs.to(dtype=self.codebook.weight.dtype) @ self.codebook.weight


class VQActionEncoder(nn.Module):
    """Action-chunk encoder with a VQ bottleneck and masked reconstruction loss."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        num_codes: int,
        model_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        decoder_hidden_dim: int = 768,
    ):
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
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
        self.quantizer = VectorQuantizer(num_codes=num_codes, code_dim=latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, self.max_seq_len * input_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def _encode_chunks(
        self,
        action_chunks: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_chunks.ndim != 3:
            raise ValueError("action_chunks must be a 3D tensor with shape (B, T, A)")

        x = self.input_proj(action_chunks.float())
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
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
            if action_mask.shape != action_chunks.shape[:2]:
                raise ValueError("action_mask shape must match action_chunks (B, T)")
            valid = action_mask.to(dtype=torch.bool)
            cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=valid.device)
            key_padding_mask = ~torch.cat([cls_valid, valid], dim=1)

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.output_proj(h[:, 0])

    def _reconstruction_loss(
        self,
        macro_actions: torch.Tensor,
        action_chunks: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = action_chunks.float()
        pred = self.decoder(macro_actions).reshape(
            action_chunks.size(0), self.max_seq_len, self.input_dim
        )
        pred = pred[:, : action_chunks.size(1)]

        if action_mask is None:
            return F.mse_loss(pred, target)

        mask = action_mask.to(dtype=target.dtype).unsqueeze(-1)
        err = (pred - target).pow(2) * mask
        denom = mask.sum().clamp(min=1.0) * target.size(-1)
        return err.sum() / denom

    def encode_with_info(
        self,
        action_chunks: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self._encode_chunks(action_chunks, action_mask=action_mask)
        quantized = self.quantizer(encoded)
        recon_loss = self._reconstruction_loss(
            quantized["quantized_st"],
            action_chunks,
            action_mask=action_mask,
        )
        return {
            "macro_actions": quantized["quantized_st"],
            "code_indices": quantized["indices"],
            "commitment_loss": quantized["commitment_loss"],
            "codebook_loss": quantized["codebook_loss"],
            "recon_loss": recon_loss,
            "perplexity": quantized["perplexity"],
            "active_codes": quantized["active_codes"],
        }

    def forward(
        self,
        action_chunks: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_with_info(action_chunks, action_mask=action_mask)["macro_actions"]

    def quantize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize_latents(latents)

    @property
    def num_codes(self) -> int:
        return int(self.quantizer.num_codes)

    def get_codebook(self) -> torch.Tensor:
        return self.quantizer.codebook.weight

    def latents_to_codes(self, latents: torch.Tensor) -> torch.Tensor:
        return self.quantizer.latents_to_codes(latents)

    def codes_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.codes_to_latents(indices)

    def code_probs_to_latents(self, probs: torch.Tensor) -> torch.Tensor:
        return self.quantizer.code_probs_to_latents(probs)
