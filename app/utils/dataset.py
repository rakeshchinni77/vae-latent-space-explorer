"""Dataset utilities for the VAE latent space explorer project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from configs.config import dataset_config, training_config


logger = logging.getLogger(__name__)


def _ensure_data_directory(data_dir: Path) -> Path:
    """Create the dataset directory if it does not already exist."""
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _dataset_exists(dataset_root: Path) -> bool:
    """Check whether the FashionMNIST dataset files already exist."""
    dataset_dir = dataset_root / "FashionMNIST"
    return dataset_dir.exists() and any(dataset_dir.iterdir())


def get_transforms() -> transforms.Compose:
    """Return the image transformation pipeline for FashionMNIST data."""
    return transforms.Compose([transforms.ToTensor()])


def get_train_dataset() -> Dataset:
    """Return the FashionMNIST training dataset."""
    dataset_root = _ensure_data_directory(dataset_config.data_dir / "raw")
    if not _dataset_exists(dataset_root):
        raise FileNotFoundError(
            "FashionMNIST dataset not found. Run 'python data/download_dataset.py' first."
        )
    return datasets.FashionMNIST(
        root=str(dataset_root),
        train=True,
        download=False,
        transform=get_transforms(),
    )


def get_test_dataset() -> Dataset:
    """Return the FashionMNIST test dataset."""
    dataset_root = _ensure_data_directory(dataset_config.data_dir / "raw")
    if not _dataset_exists(dataset_root):
        raise FileNotFoundError(
            "FashionMNIST dataset not found. Run 'python data/download_dataset.py' first."
        )
    return datasets.FashionMNIST(
        root=str(dataset_root),
        train=False,
        download=False,
        transform=get_transforms(),
    )


def get_train_loader() -> DataLoader:
    """Return a DataLoader for the training split."""
    dataset = get_train_dataset()
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )


def get_test_loader() -> DataLoader:
    """Return a DataLoader for the test split."""
    dataset = get_test_dataset()
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Return both train and test loaders for the current dataset."""
    return get_train_loader(), get_test_loader()
