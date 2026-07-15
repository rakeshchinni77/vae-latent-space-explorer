"""Decoder module for reconstructing FashionMNIST-style images from latent vectors.

The decoder is the mathematical mirror of the encoder and reconstructs an
input image tensor of shape ``(B, 1, 28, 28)`` from a latent representation of
shape ``(B, latent_dim)``.
"""

from __future__ import annotations

from torch import Tensor, nn

from configs.config import model_config


class Decoder(nn.Module):
    """Convolutional decoder that reconstructs images from latent vectors.

    The decoder expects a latent tensor of shape ``(B, latent_dim)`` and returns
    a reconstructed image tensor of shape ``(B, 1, 28, 28)`` with pixel values
    constrained to the range ``[0, 1]`` by ``nn.Sigmoid()``.
    """

    def __init__(self) -> None:
        """Initialize the decoder layers using the configured latent dimension."""
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

        self.latent_projector = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.flatten_dim),
            nn.ReLU(inplace=True),
        )

        self.reshape = nn.Unflatten(1, self.feature_shape)

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def _validate_inputs(self, z: Tensor) -> None:
        """Validate the latent tensor before decoding."""
        if z.ndim != 2:
            raise ValueError(
                f"Expected latent tensor with 2 dimensions, but got shape {tuple(z.shape)}"
            )
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, but got {z.shape[1]}"
            )
        if z.shape[0] == 0:
            raise ValueError("Input batch size cannot be empty")

    def forward(self, z: Tensor) -> Tensor:
        """Decode a latent batch into reconstructed images.

        Args:
            z: Latent tensor of shape ``(B, latent_dim)``.

        Returns:
            Reconstructed image tensor of shape ``(B, 1, 28, 28)``.

        Raises:
            ValueError: If the latent tensor does not match the expected shape.
        """
        self._validate_inputs(z)

        hidden = self.latent_projector(z)
        features = self.reshape(hidden)
        reconstructed = self.upsampler(features)
        return reconstructed
