"""ELBO loss functions for the VAE model.

This module implements the variational autoencoder objective used in the
original Kingma and Welling formulation. The reconstruction term uses binary
cross entropy with reduction="sum", while the KL divergence term is computed
analytically for a diagonal Gaussian posterior.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ELBOLoss(nn.Module):
    """Evidence Lower Bound (ELBO) loss for a VAE.

    The loss is the sum of a reconstruction term and a KL regularization term.
    The reconstruction loss is computed with binary cross entropy over the
    reconstructed image tensor and the target image tensor, while the KL term
    measures the divergence between the approximate posterior and the standard
    normal prior.
    """

    def __init__(self) -> None:
        """Initialize the loss module."""
        super().__init__()

    def _validate_inputs(
        self,
        reconstructed: Tensor,
        target: Tensor,
        mu: Tensor,
        logvar: Tensor,
        beta: float,
    ) -> None:
        """Validate tensor shapes and parameters before computing the loss."""
        if reconstructed.shape != target.shape:
            raise ValueError(
                "reconstructed and target must have matching shapes; "
                f"got {tuple(reconstructed.shape)} and {tuple(target.shape)}"
            )
        if reconstructed.ndim != 4:
            raise ValueError(
                f"reconstructed must be rank 4, but got shape {tuple(reconstructed.shape)}"
            )
        if target.ndim != 4:
            raise ValueError(
                f"target must be rank 4, but got shape {tuple(target.shape)}"
            )
        if mu.ndim != 2:
            raise ValueError(f"mu must be rank 2, but got shape {tuple(mu.shape)}")
        if logvar.ndim != 2:
            raise ValueError(
                f"logvar must be rank 2, but got shape {tuple(logvar.shape)}"
            )
        if mu.shape != logvar.shape:
            raise ValueError(
                "mu and logvar must have matching shapes; "
                f"got {tuple(mu.shape)} and {tuple(logvar.shape)}"
            )
        if reconstructed.shape[0] != mu.shape[0]:
            raise ValueError(
                "Batch size mismatch between reconstructed/target and latent parameters; "
                f"got {reconstructed.shape[0]} and {mu.shape[0]}"
            )
        if not torch.isfinite(reconstructed).all():
            raise RuntimeError("reconstructed contains NaN or Inf values")
        if not torch.isfinite(target).all():
            raise RuntimeError("target contains NaN or Inf values")
        if not torch.isfinite(mu).all():
            raise RuntimeError("mu contains NaN or Inf values")
        if not torch.isfinite(logvar).all():
            raise RuntimeError("logvar contains NaN or Inf values")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, but got {beta}")

    @staticmethod
    def metrics_dict(
        total_loss: Tensor,
        reconstruction_loss: Tensor,
        kl_divergence: Tensor,
        beta: float,
    ) -> dict[str, float]:
        """Convert ELBO loss outputs into a plain dictionary for CSV logging.

        Args:
            total_loss: Total ELBO loss for the current batch.
            reconstruction_loss: Binary cross entropy reconstruction loss.
            kl_divergence: KL divergence term for the batch.
            beta: Weight applied to the KL divergence term.

        Returns:
            A plain Python dictionary with scalar metric values suitable for
            CSV writing.
        """
        return {
            "loss": float(total_loss.detach()),
            "reconstruction_loss": float(reconstruction_loss.detach()),
            "kl_divergence": float(kl_divergence.detach()),
            "beta": float(beta),
        }

    def forward(
        self,
        reconstructed: Tensor,
        target: Tensor,
        mu: Tensor,
        logvar: Tensor,
        beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the ELBO loss components for a VAE batch.

        Args:
            reconstructed: Reconstructed image tensor of shape ``(B, 1, 28, 28)``.
            target: Target image tensor of shape ``(B, 1, 28, 28)``.
            mu: Posterior mean tensor of shape ``(B, latent_dim)``.
            logvar: Posterior log-variance tensor of shape ``(B, latent_dim)``.
            beta: Weight applied to the KL divergence term.

        Returns:
            A tuple ``(total_loss, reconstruction_loss, kl_divergence)``.
        """
        self._validate_inputs(reconstructed, target, mu, logvar, beta)

        reconstruction_loss = F.binary_cross_entropy(
            reconstructed,
            target,
            reduction="sum",
        )
        kl_divergence = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        total_loss = reconstruction_loss + beta * kl_divergence
        return total_loss, reconstruction_loss, kl_divergence


def loss_function(
    reconstructed: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convenience function that evaluates the ELBO loss using the same logic.

    Args:
        reconstructed: Reconstructed image tensor of shape ``(B, 1, 28, 28)``.
        target: Target image tensor of shape ``(B, 1, 28, 28)``.
        mu: Posterior mean tensor of shape ``(B, latent_dim)``.
        logvar: Posterior log-variance tensor of shape ``(B, latent_dim)``.
        beta: Weight applied to the KL divergence term.

    Returns:
        A tuple ``(total_loss, reconstruction_loss, kl_divergence)``.
    """
    return ELBOLoss()(reconstructed, target, mu, logvar, beta)
