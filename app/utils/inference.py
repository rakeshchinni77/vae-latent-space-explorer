from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import numpy as np

from models.vae import VAE


@st.cache_resource
def load_vae_checkpoint(model_path: Path, device: str) -> VAE:
    """Load the trained VAE model from a checkpoint and cache the result."""
    model = VAE()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def decode_latent_image(model: VAE, latent_tensor: torch.Tensor) -> np.ndarray:
    """Decode a latent tensor into a grayscale image array in the 0-1 range."""
    with torch.no_grad():
        output = model.decode(latent_tensor)
    output = output.clamp(0.0, 1.0)
    image = output.squeeze().cpu().numpy()
    return image


def run_reconstruction(
    model: VAE, image_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a full VAE reconstruction pipeline on a single image tensor."""
    with torch.no_grad():
        reconstruction, mu, logvar = model(image_tensor)
        z = model.reparameterize(mu, logvar)
    return reconstruction, mu, logvar, z
