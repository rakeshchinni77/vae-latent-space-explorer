from __future__ import annotations

import torch

from models.vae import VAE


def test_vae_forward_and_methods() -> None:
    model = VAE()
    inputs = torch.randn(2, 1, 28, 28)

    reconstruction, mu, logvar = model(inputs)
    assert reconstruction.shape == (2, 1, 28, 28)
    assert mu.shape == (2, model.latent_dim)
    assert logvar.shape == (2, model.latent_dim)

    encoded_mu, encoded_logvar = model.encode(inputs)
    assert encoded_mu.shape == (2, model.latent_dim)
    assert encoded_logvar.shape == (2, model.latent_dim)

    z = model.reparameterize(encoded_mu, encoded_logvar)
    assert z.shape == (2, model.latent_dim)

    decoded = model.decode(z)
    assert decoded.shape == (2, 1, 28, 28)

    sampled = model.sample(4)
    assert sampled.shape == (4, 1, 28, 28)

    generated = model.generate(inputs)
    assert generated.shape == (2, 1, 28, 28)

    assert model.count_parameters() > 0
    summary = model.model_summary()
    assert summary["latent_dim"] == model.latent_dim
    assert summary["model_name"] == "VAE"
