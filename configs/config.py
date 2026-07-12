"""Structured configuration objects for the project.

This module exposes typed configuration dataclasses that build on the values in
settings.py. The singleton objects are intended to be reused by future training,
data, and UI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from configs.settings import (
    BATCH_SIZE,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    KL_ANNEAL_EPOCHS,
    LATENT_DIM,
    LEARNING_RATE,
    MODEL_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    STREAMLIT_HEADLESS,
    STREAMLIT_HOST,
    STREAMLIT_PORT,
    STREAMLIT_SERVER_ADDRESS,
    STREAMLIT_SERVER_FILE_WATCHER_TYPE,
)


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths used across the project."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    model_path: Path = MODEL_PATH


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for data loading and preprocessing."""

    data_dir: Path = DATA_DIR


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the model architecture."""

    latent_dim: int = LATENT_DIM
    model_path: Path = MODEL_PATH


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for optimization and training behavior."""

    device: str = DEVICE
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    kl_anneal_epochs: int = KL_ANNEAL_EPOCHS
    seed: int = SEED


@dataclass(frozen=True)
class StreamlitConfig:
    """Configuration for the Streamlit application."""

    port: int = STREAMLIT_PORT
    host: str = STREAMLIT_HOST
    server_address: str = STREAMLIT_SERVER_ADDRESS
    headless: bool = STREAMLIT_HEADLESS
    server_file_watcher_type: str = STREAMLIT_SERVER_FILE_WATCHER_TYPE


paths_config: PathsConfig = PathsConfig()
dataset_config: DatasetConfig = DatasetConfig()
model_config: ModelConfig = ModelConfig()
training_config: TrainingConfig = TrainingConfig()
streamlit_config: StreamlitConfig = StreamlitConfig()

__all__ = [
    "PathsConfig",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "StreamlitConfig",
    "paths_config",
    "dataset_config",
    "model_config",
    "training_config",
    "streamlit_config",
]
