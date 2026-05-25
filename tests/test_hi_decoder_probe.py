from __future__ import annotations

import pytest
import torch

pytest.importorskip("stable_pretraining")

from h_le_wm.probe.model import LatentToPixelDecoder, compute_psnr, compute_ssim


def test_latent_to_pixel_decoder_output_shape():
    decoder = LatentToPixelDecoder(
        latent_dim=192,
        img_size=224,
        patch_size=16,
        hidden_dim=128,
        depth=2,
        heads=4,
    )
    latents = torch.randn(3, 192)
    out = decoder(latents)
    assert out.shape == (3, 3, 224, 224)


def test_psnr_and_ssim_sane_for_identical_images():
    x = torch.rand(2, 3, 224, 224)
    psnr = compute_psnr(x, x)
    ssim = compute_ssim(x, x)
    assert torch.isfinite(psnr)
    assert psnr.item() > 60.0
    assert torch.isfinite(ssim)
    assert abs(ssim.item() - 1.0) < 1e-4
