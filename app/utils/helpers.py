from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def build_latent_vector(slider_values: list[float]) -> Tensor:
    """Build a latent tensor from slider values with the correct shape and dtype."""
    return torch.tensor([slider_values], dtype=torch.float32)


def format_latent_vector(z: Tensor) -> str:
    """Format the latent vector for display with three decimal precision."""
    values = z.squeeze().cpu().numpy().tolist()
    formatted = [f"{float(value):.3f}" for value in values]
    return "[" + ", ".join(formatted) + "]"


def compute_norm(z: Tensor) -> float:
    """Compute the Euclidean norm for the latent vector."""
    return float(torch.norm(z).item())


def get_default_latent_vector(latent_dim: int) -> list[float]:
    """Return the default latent vector values for slider initialization."""
    return [0.0] * latent_dim


def get_random_latent_vector(latent_dim: int) -> list[float]:
    """Return a random latent vector sampled from N(0,1)."""
    return torch.randn(latent_dim).cpu().tolist()


def checkpoint_file_label(model_path: Path) -> str:
    """Return a normalized relative file label for display."""
    return model_path.name


def normalize_device_label(device: str) -> str:
    """Return a user-friendly device label from a PyTorch device string."""
    return "CUDA (NVIDIA GPU)" if device.startswith("cuda") else "CPU"
