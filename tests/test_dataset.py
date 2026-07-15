from __future__ import annotations

import torch

from app.utils.dataset import get_dataloaders


def test_dataloaders_load_and_return_batches() -> None:
    train_loader, test_loader = get_dataloaders()
    assert train_loader is not None
    assert test_loader is not None

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    train_images, train_labels = train_batch
    test_images, test_labels = test_batch

    assert train_images.dtype == torch.float32
    assert test_images.dtype == torch.float32
    assert train_images.ndim == 4
    assert test_images.ndim == 4
    assert train_images.shape[1:] == (1, 28, 28)
    assert test_images.shape[1:] == (1, 28, 28)
    assert train_labels.ndim == 1
    assert test_labels.ndim == 1
    assert torch.all(train_images >= 0.0) and torch.all(train_images <= 1.0)
    assert torch.all(test_images >= 0.0) and torch.all(test_images <= 1.0)
