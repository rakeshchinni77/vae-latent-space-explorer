"""Reusable training pipeline for the VAE model."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import paths_config, training_config
from models.loss import ELBOLoss
from models.vae import VAE


logger = logging.getLogger(__name__)


class Trainer:
    """Coordinate VAE training, logging, and checkpointing."""

    def __init__(
        self,
        model: VAE,
        criterion: ELBOLoss,
        optimizer: Adam,
        train_loader: DataLoader,
        device: torch.device | str,
        training_config_obj: Any,
    ) -> None:
        """Initialize the trainer with injected dependencies."""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = torch.device(device)
        self.training_config_obj = training_config_obj
        self.epochs = int(self.training_config_obj.epochs)
        self.kl_anneal_epochs = int(self.training_config_obj.kl_anneal_epochs)
        self.model_path = Path(paths_config.model_path)
        self.results_dir = Path(paths_config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.log_path = self.results_dir / "training_log.csv"
        self.best_loss: float | None = None

    def _initialize_log(self) -> None:
        """Reset the training log for a fresh run."""
        fieldnames = ["epoch", "reconstruction_loss", "kl_divergence"]
        with self.log_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    def _write_epoch_log(self, epoch: int, average_loss: float, average_bce: float, average_kl: float) -> None:
        """Append a single epoch summary row to the CSV log."""
        fieldnames = ["epoch", "reconstruction_loss", "kl_divergence"]
        file_exists = self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "epoch": epoch,
                    "reconstruction_loss": average_bce,
                    "kl_divergence": average_kl,
                }
            )

    def _save_checkpoint(self, epoch: int, current_loss: float) -> None:
        """Save the model checkpoint when the loss improves."""
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save(self.model.state_dict(), str(self.model_path))
            logger.info("Saved checkpoint for epoch %s to %s", epoch, self.model_path)

    def _beta_for_epoch(self, epoch: int) -> float:
        """Compute the KL annealing coefficient for the current epoch."""
        if self.kl_anneal_epochs <= 0:
            return 1.0
        return min(1.0, (epoch + 1) / self.kl_anneal_epochs)

    def train(self) -> None:
        """Run the full training loop."""
        self.model.to(self.device)
        self.model.train()

        self._initialize_log()

        print("Model: VAE")
        print(f"Latent Dimension: {self.model.latent_dim}")
        print(f"Trainable Parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print(self.model.model_summary())

        for epoch in range(self.epochs):
            epoch_loss_total = 0.0
            epoch_bce_total = 0.0
            epoch_kl_total = 0.0

            progress_bar = tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=True,
            )
            self.optimizer.zero_grad(set_to_none=True)

            for batch_idx, (images, _) in enumerate(self.train_loader):
                images = images.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                reconstructed, mu, logvar = self.model(images)
                beta = self._beta_for_epoch(epoch)

                total_loss, reconstruction_loss, kl_divergence = self.criterion(
                    reconstructed,
                    images,
                    mu,
                    logvar,
                    beta=beta,
                )

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss_total += float(total_loss.detach().item())
                epoch_bce_total += float(reconstruction_loss.detach().item())
                epoch_kl_total += float(kl_divergence.detach().item())

                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{self.epochs} | beta={beta:.3f} | loss={float(total_loss.detach().item()):.4f} | bce={float(reconstruction_loss.detach().item()):.4f} | kl={float(kl_divergence.detach().item()):.4f}"
                )
                progress_bar.update(1)

            progress_bar.close()

            num_batches = max(1, len(self.train_loader))
            avg_loss = epoch_loss_total / num_batches
            avg_bce = epoch_bce_total / num_batches
            avg_kl = epoch_kl_total / num_batches
            self._write_epoch_log(epoch + 1, avg_loss, avg_bce, avg_kl)
            self._save_checkpoint(epoch + 1, avg_loss)

            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"beta={beta:.3f} | "
                f"avg_loss={avg_loss:.4f} | "
                f"avg_bce={avg_bce:.4f} | "
                f"avg_kl={avg_kl:.4f}"
            )

        if not self.model_path.exists():
            raise RuntimeError(f"Checkpoint was not created at {self.model_path}")

        print(f"Training complete. Log saved to {self.log_path}")
        print(f"Final checkpoint saved to {self.model_path}")
