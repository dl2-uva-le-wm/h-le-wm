from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid, save_image

from h_le_wm.baseline.adapter import BASELINE_ROOT


class DecoderBlock(nn.Module):
    """Transformer decoder block for latent-to-patch decoding."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), memory, memory, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class LatentToPixelDecoder(nn.Module):
    """Decode a single latent vector into a full image via learned patch queries."""

    def __init__(
        self,
        *,
        latent_dim: int,
        img_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        out_channels: int = 3,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.latent_dim = int(latent_dim)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.out_channels = int(out_channels)
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = self.patch_size * self.patch_size * self.out_channels

        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_patches, self.hidden_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=self.hidden_dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.patch_head = nn.Linear(self.hidden_dim, self.patch_dim)

    def unpatchify(self, patch_pixels: torch.Tensor) -> torch.Tensor:
        if patch_pixels.ndim != 3:
            raise ValueError("patch_pixels must be shape (B, P, patch_dim)")

        b = patch_pixels.size(0)
        patch_pixels = patch_pixels.view(
            b,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        patch_pixels = patch_pixels.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patch_pixels.view(b, self.out_channels, self.img_size, self.img_size)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 2:
            raise ValueError("latents must be shape (B, D)")
        memory = self.latent_proj(latents).unsqueeze(1)
        x = self.query_tokens + self.pos_embedding
        x = x.expand(latents.size(0), -1, -1)
        for block in self.blocks:
            x = block(x, memory)
        x = self.norm(x)
        patch_pixels = self.patch_head(x)
        return self.unpatchify(patch_pixels)


def load_hi_checkpoint(path: str | Path):
    """Load a saved HiJEPA object checkpoint."""

    baseline_root = str(BASELINE_ROOT)
    if baseline_root not in sys.path:
        sys.path.insert(0, baseline_root)

    ckpt_path = Path(path).expanduser()
    try:
        model_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        model_obj = torch.load(ckpt_path, map_location="cpu")

    model = model_obj.model if hasattr(model_obj, "model") else model_obj
    required = (
        "encoder",
        "low_predictor",
        "action_encoder",
        "high_predictor",
        "latent_action_encoder",
        "macro_to_condition",
        "projector",
        "low_pred_proj",
        "high_pred_proj",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            f"Loaded checkpoint does not look like a HiJEPA model. Missing attrs: {missing}"
        )
    return model


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def imagenet_mean_std() -> tuple[torch.Tensor, torch.Tensor]:
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1, 1, 1)
    return mean, std


def denormalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    mean, std = imagenet_mean_std()
    mean = mean.to(device=images.device, dtype=images.dtype)
    std = std.to(device=images.device, dtype=images.dtype)
    return (images * std + mean).clamp(0.0, 1.0)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return -10.0 * torch.log10(mse.clamp_min(eps))


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_size: int = 11,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape for SSIM")
    if pred.ndim != 4:
        raise ValueError("pred and target must be BCHW tensors for SSIM")

    pad = kernel_size // 2
    mu_x = F.avg_pool2d(pred, kernel_size=kernel_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size=kernel_size, stride=1, padding=pad) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, kernel_size=kernel_size, stride=1, padding=pad) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=kernel_size, stride=1, padding=pad) - (mu_x * mu_y)

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(1e-8)
    return ssim_map.mean()


def make_comparison_panel(
    *,
    context_pixels: torch.Tensor | None = None,
    target_pixels: torch.Tensor,
    decoded_true: torch.Tensor,
    decoded_pred: torch.Tensor | None = None,
    max_items: int = 4,
) -> torch.Tensor:
    max_items = max(1, min(int(max_items), int(target_pixels.size(0))))
    columns = []
    if context_pixels is not None:
        columns.append(denormalize_imagenet(context_pixels[:max_items]))
    columns.extend(
        [
            denormalize_imagenet(target_pixels[:max_items]),
            denormalize_imagenet(decoded_true[:max_items]),
        ]
    )
    if decoded_pred is not None:
        columns.append(denormalize_imagenet(decoded_pred[:max_items]))

    rows = []
    for i in range(max_items):
        for col in columns:
            rows.append(col[i])
    return make_grid(rows, nrow=len(columns), padding=4)


def save_panel(path: str | Path, panel: torch.Tensor) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(panel, output_path)


def save_probe_bundle(
    path: str | Path,
    *,
    decoder: nn.Module,
    cfg: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "decoder_state_dict": decoder.state_dict(),
        "cfg": cfg,
        "metadata": metadata or {},
    }
    torch.save(payload, Path(path))


def load_decoder_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "decoder_state_dict" in payload:
        return payload["decoder_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = {}
        for key, value in payload["state_dict"].items():
            if key.startswith("decoder."):
                state_dict[key[len("decoder.") :]] = value
        if state_dict:
            return state_dict
    if isinstance(payload, dict) and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise ValueError(f"Unsupported decoder checkpoint format: {path}")


def infer_latent_dim(model: nn.Module) -> int:
    if hasattr(model, "high_pred_proj") and hasattr(model.high_pred_proj, "net"):
        last = model.high_pred_proj.net[-1]
        if hasattr(last, "out_features"):
            return int(last.out_features)
    if hasattr(model, "high_pred_proj") and hasattr(model.high_pred_proj, "out_features"):
        return int(model.high_pred_proj.out_features)
    if hasattr(model, "projector") and hasattr(model.projector, "net"):
        last = model.projector.net[-1]
        if hasattr(last, "out_features"):
            return int(last.out_features)
    if hasattr(model, "projector") and hasattr(model.projector, "out_features"):
        return int(model.projector.out_features)
    if hasattr(model, "low_predictor") and hasattr(model.low_predictor, "pos_embedding"):
        return int(model.low_predictor.pos_embedding.shape[-1])
    raise ValueError("Unable to infer latent dimension from HiJEPA checkpoint.")


def format_metric(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def epoch_filename(prefix: str, epoch: int, suffix: str) -> str:
    return f"{prefix}_epoch_{epoch}_{suffix}"


def step_filename(prefix: str, step: int, suffix: str) -> str:
    return f"{prefix}_step_{step}_{suffix}"


def ensure_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return tensor


def sqrt_int(value: int) -> int:
    root = int(math.isqrt(value))
    if root * root != value:
        raise ValueError(f"Expected a perfect square, got {value}")
    return root
