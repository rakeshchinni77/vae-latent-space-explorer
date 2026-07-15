"""Variational Autoencoder (VAE) module for the latent space explorer.

The VAE combines the encoder and decoder into a single model that maps input
images to latent Gaussian parameters and reconstructs images from sampled
latent vectors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from models.decoder import Decoder
from models.encoder import Encoder


class VAE(nn.Module):
    """Variational Autoencoder for FashionMNIST-style image reconstruction.

    The model accepts input tensors of shape ``(B, 1, 28, 28)`` and returns a
    reconstruction together with the posterior mean and log-variance for the
    latent space.
    """

    def __init__(self) -> None:
        """Initialize the encoder, decoder, and reusable model components."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @property
    def latent_dim(self) -> int:
        """Return the latent dimension configured for the encoder and decoder."""
        return self.encoder.latent_dim

    def _validate_inputs(self, x: Tensor) -> None:
        """Validate the input tensor before encoding."""
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

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters in the model."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def model_summary(self) -> dict[str, object]:
        """Return a compact summary of the model for logging and debugging."""
        parameters = list(self.parameters())
        device = str(parameters[0].device) if parameters else "cpu"
        return {
            "model_name": self.__class__.__name__,
            "latent_dim": self.latent_dim,
            "trainable_parameters": self.count_parameters(),
            "device": device,
        }

    def __repr__(self) -> str:
        """Return a readable representation of the model configuration."""
        summary = self.model_summary()
        return (
            f"{self.__class__.__name__}("
            f"latent_dim={summary['latent_dim']}, "
            f"trainable_parameters={summary['trainable_parameters']}, "
            f"device={summary['device']})"
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample a latent vector using the standard VAE reparameterization trick.

        Args:
            mu: Mean of the approximate posterior of shape ``(B, latent_dim)``.
            logvar: Log-variance of the approximate posterior of shape
                ``(B, latent_dim)``.

        Returns:
            A latent tensor ``z`` of shape ``(B, latent_dim)``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode an input image batch into latent mean and log-variance.

        Args:
            x: Input tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            A tuple ``(mu, logvar)`` with shape ``(B, latent_dim)``.
        """
        self._validate_inputs(x)
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode a latent tensor into a reconstructed image batch.

        Args:
            z: Latent tensor of shape ``(B, latent_dim)``.

        Returns:
            Reconstructed image tensor of shape ``(B, 1, 28, 28)``.
        """
        return self.decoder(z)

    def generate(self, x: Tensor) -> Tensor:
        """Encode and reconstruct an input image batch.

        Args:
            x: Input tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            Reconstructed image tensor of shape ``(B, 1, 28, 28)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def sample(self, num_samples: int) -> Tensor:
        """Sample random latent vectors and decode them into images.

        Args:
            num_samples: Number of images to generate.

        Returns:
            A tensor of generated images with shape ``(num_samples, 1, 28, 28)``.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero")

        with torch.no_grad():
            z = torch.randn(num_samples, self.encoder.latent_dim)
            return self.decode(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, reparameterize, and decode an input image batch.

        Args:
            x: Input tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            A tuple ``(reconstructed_x, mu, logvar)``.
        """
        self._validate_inputs(x)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar
