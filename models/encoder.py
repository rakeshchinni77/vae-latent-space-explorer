"""Convolutional encoder for the VAE latent space explorer.

This module defines a lightweight CNN encoder that maps FashionMNIST images
into Gaussian posterior parameters for the latent space.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from configs.config import model_config


class Encoder(nn.Module):
    """CNN encoder producing latent mean and log-variance parameters.

    The encoder expects input tensors shaped ``(B, 1, 28, 28)`` and returns
    two tensors of shape ``(B, latent_dim)`` corresponding to the posterior
    mean and log-variance.

    Feature maps produced by the convolution stack have shape ``(B, 64, 7, 7)``.
    After flattening, the representation has dimension ``3136``. The projector
    reduces this hidden state to a ``256``-dimensional embedding before
    producing ``mu`` and ``logvar`` outputs.
    """

    def __init__(self) -> None:
        """Initialize the convolutional encoder layers."""
        super().__init__()
        self.latent_dim: int = model_config.latent_dim
        self.feature_channels: int = 64
        self.feature_height: int = 7
        self.feature_width: int = 7
        self.feature_shape = (
            self.feature_channels,
            self.feature_height,
            self.feature_width,
        )
        self.flatten_dim: int = (
            self.feature_channels * self.feature_height * self.feature_width
        )

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.feature_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.projector = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256, self.latent_dim)
        self.fc_logvar = nn.Linear(256, self.latent_dim)

    def _validate_inputs(self, x: Tensor) -> None:
        """Validate the input tensor shape before applying the encoder."""
        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor with 4 dimensions, but got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"Expected single-channel input, but got {x.shape[1]} channels"
            )
        if x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError(
                f"Expected input shape (B, 1, 28, 28), but got {tuple(x.shape)}"
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode an input batch into latent mean and log-variance.

        Args:
            x: Input tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            A tuple ``(mu, logvar)`` containing latent parameters with shape
            ``(B, latent_dim)``.

        Raises:
            ValueError: If the input tensor does not match the expected shape.
        """
        self._validate_inputs(x)

        features = self.features(x)
        flattened = self.flatten(features)
        hidden = self.projector(flattened)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
