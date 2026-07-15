from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dataset import get_test_dataset
from configs.config import paths_config, training_config
from models.vae import VAE


def get_device() -> torch.device:
    """Return the best available device for inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_results_directory() -> Path:
    """Ensure the results directory exists and return its path."""
    results_path = paths_config.results_dir
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def load_model() -> VAE:
    """Load the trained VAE model from the configured checkpoint."""
    device = get_device()
    checkpoint_path = paths_config.model_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = VAE()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_test_dataset():
    """Load the FashionMNIST test dataset."""
    return get_test_dataset()


def get_device_name() -> str:
    """Return a friendsly device name for display."""
    return "cuda" if torch.cuda.is_available() else "cpu"
