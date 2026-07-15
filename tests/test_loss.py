from __future__ import annotations

import pytest
import torch

from models.loss import ELBOLoss


def test_elbo_loss_components() -> None:
    loss_module = ELBOLoss()
    reconstructed = torch.rand(2, 1, 28, 28)
    target = torch.rand(2, 1, 28, 28)
    mu = torch.zeros(2, 16)
    logvar = torch.zeros(2, 16)

    total_loss, reconstruction_loss, kl_divergence = loss_module(
        reconstructed, target, mu, logvar, beta=1.0
    )

    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(reconstruction_loss, torch.Tensor)
    assert isinstance(kl_divergence, torch.Tensor)
    assert torch.isfinite(total_loss).all()
    assert torch.isfinite(reconstruction_loss).all()
    assert torch.isfinite(kl_divergence).all()
    assert torch.isclose(total_loss, reconstruction_loss + kl_divergence).item()


def test_elbo_loss_with_beta_variations() -> None:
    loss_module = ELBOLoss()
    reconstructed = torch.rand(2, 1, 28, 28)
    target = torch.rand(2, 1, 28, 28)
    mu = torch.zeros(2, 16)
    logvar = torch.zeros(2, 16)

    for beta in (0.0, 0.5, 1.0):
        total_loss, reconstruction_loss, kl_divergence = loss_module(
            reconstructed, target, mu, logvar, beta=beta
        )
        assert torch.isclose(total_loss, reconstruction_loss + beta * kl_divergence).item()


def test_elbo_loss_negative_beta_raises() -> None:
    loss_module = ELBOLoss()
    reconstructed = torch.rand(2, 1, 28, 28)
    target = torch.rand(2, 1, 28, 28)
    mu = torch.zeros(2, 16)
    logvar = torch.zeros(2, 16)

    with pytest.raises(ValueError):
        loss_module(reconstructed, target, mu, logvar, beta=-0.1)
