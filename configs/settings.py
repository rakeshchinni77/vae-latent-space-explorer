"""Application configuration settings loaded from environment variables.

This module centralizes environment-driven configuration for the repository.
It uses python-dotenv so local development works smoothly with a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


def _get_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int = 0) -> int:
    """Parse an integer environment variable."""
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _get_float(name: str, default: float = 0.0) -> float:
    """Parse a float environment variable."""
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _get_str(name: str, default: str = "") -> str:
    """Read a string environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip()


def _resolve_path(base_dir: Path, value: str | None, default: str) -> Path:
    """Resolve a filesystem path from an environment variable."""
    raw_value = value if value is not None and str(value).strip() else default
    return base_dir / Path(raw_value)


def _resolve_device(value: str | None) -> str:
    """Resolve the active compute device from the environment."""
    raw_value = str(value).strip().lower() if value is not None else ""
    if raw_value == "auto":
        try:
            import torch
        except Exception:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw_value or "cpu"


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

STREAMLIT_PORT: int = _get_int("STREAMLIT_PORT", 8501)
STREAMLIT_HOST: str = _get_str("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_SERVER_ADDRESS: str = _get_str(
    "STREAMLIT_SERVER_ADDRESS", f"{STREAMLIT_HOST}:{STREAMLIT_PORT}"
)
STREAMLIT_HEADLESS: bool = _get_bool("STREAMLIT_HEADLESS", True)
STREAMLIT_SERVER_FILE_WATCHER_TYPE: str = _get_str(
    "STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll"
)

MODEL_PATH: Path = _resolve_path(PROJECT_ROOT, os.getenv("MODEL_PATH"), "models/vae.pt")
LATENT_DIM: int = _get_int("LATENT_DIM", 16)
DEVICE: str = _resolve_device(os.getenv("DEVICE"))

DATA_DIR: Path = _resolve_path(PROJECT_ROOT, os.getenv("DATA_DIR"), "data")
RESULTS_DIR: Path = _resolve_path(PROJECT_ROOT, os.getenv("RESULTS_DIR"), "results")

BATCH_SIZE: int = _get_int("BATCH_SIZE", 128)
LEARNING_RATE: float = _get_float("LEARNING_RATE", 0.001)
EPOCHS: int = _get_int("EPOCHS", _get_int("NUM_EPOCHS", 30))
KL_ANNEAL_EPOCHS: int = _get_int(
    "KL_ANNEAL_EPOCHS", _get_int("KL_ANNEALING_EPOCHS", 15)
)
SEED: int = _get_int("SEED", 42)

__all__: list[str] = [
    "PROJECT_ROOT",
    "STREAMLIT_PORT",
    "STREAMLIT_HOST",
    "STREAMLIT_SERVER_ADDRESS",
    "STREAMLIT_HEADLESS",
    "STREAMLIT_SERVER_FILE_WATCHER_TYPE",
    "MODEL_PATH",
    "LATENT_DIM",
    "DEVICE",
    "DATA_DIR",
    "RESULTS_DIR",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "EPOCHS",
    "KL_ANNEAL_EPOCHS",
    "SEED",
]
