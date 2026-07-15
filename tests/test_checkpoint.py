from __future__ import annotations

import torch

from configs.config import paths_config
from models.vae import VAE


def test_checkpoint_exists_and_loads() -> None:
    assert paths_config.model_path.exists()

    model = VAE()
    checkpoint = torch.load(paths_config.model_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)

    model.eval()
    with torch.no_grad():
        sample = torch.randn(1, 1, 28, 28)
        reconstruction, mu, logvar = model(sample)

    assert reconstruction.shape == (1, 1, 28, 28)
    assert mu.shape == (1, model.latent_dim)
    assert logvar.shape == (1, model.latent_dim)
