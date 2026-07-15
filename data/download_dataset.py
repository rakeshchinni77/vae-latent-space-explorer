"""Download the FashionMNIST dataset into the configured raw data directory."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dataset import get_transforms
from configs.config import dataset_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _resolve_dataset_root() -> Path:
    """Return the dataset root directory for raw FashionMNIST files."""
    dataset_root = dataset_config.data_dir / "raw"
    dataset_root.mkdir(parents=True, exist_ok=True)
    return dataset_root


def download_dataset() -> None:
    """Download FashionMNIST train and test splits if they are not already present."""
    dataset_root = _resolve_dataset_root()
    dataset_dir = dataset_root / "FashionMNIST"

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        logger.info("Dataset already exists at %s; skipping download.", dataset_dir)
        train_dataset = datasets.FashionMNIST(
            root=str(dataset_root),
            train=True,
            download=False,
            transform=get_transforms(),
        )
        test_dataset = datasets.FashionMNIST(
            root=str(dataset_root),
            train=False,
            download=False,
            transform=get_transforms(),
        )
    else:
        try:
            train_dataset = datasets.FashionMNIST(
                root=str(dataset_root),
                train=True,
                download=True,
                transform=get_transforms(),
            )
            test_dataset = datasets.FashionMNIST(
                root=str(dataset_root),
                train=False,
                download=True,
                transform=get_transforms(),
            )
        except Exception as exc:  # pragma: no cover - explicit runtime guard
            logger.error("Failed to download the FashionMNIST dataset: %s", exc)
            raise

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")

    sample_image, sample_label = train_dataset[0]
    print(f"Image shape: {tuple(sample_image.shape)}")
    print(f"Label: {sample_label}")
    logger.info("Dataset stored under %s", dataset_root)


def main() -> int:
    """Run the dataset download workflow and return an exit code."""
    try:
        download_dataset()
    except Exception as exc:
        logger.error("Dataset download failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
