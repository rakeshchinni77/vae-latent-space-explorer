"""Training entry point for the VAE model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dataset import get_train_loader
from configs.config import training_config
from models.loss import ELBOLoss
from models.trainer import Trainer
from models.vae import VAE


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the VAE model")
    parser.add_argument("--epochs", type=int, default=int(training_config.epochs))
    return parser.parse_args()


def main() -> None:
    """Run the training pipeline."""
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    set_seed(int(training_config.seed))

    train_loader = get_train_loader()
    if train_loader is None:
        raise RuntimeError("Training dataloader could not be created")

    model = VAE()
    criterion = ELBOLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config.learning_rate))

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        device=training_config.device,
        training_config_obj=training_config,
    )
    trainer.epochs = args.epochs
    trainer.train()


if __name__ == "__main__":
    main()
