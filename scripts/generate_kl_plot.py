from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_utils import ensure_results_directory, get_device, load_model, load_test_dataset


def compute_kl(mu: torch.Tensor, logvar: torch.Tensor) -> np.ndarray:
    """Compute per-dimension KL divergence values for a single sample."""
    kl_tensor = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_tensor.squeeze(0).cpu().numpy()


def save_kl_plot(kl_values: np.ndarray, output_path: Path) -> None:
    """Save a bar chart of KL divergence values to disk."""
    figure, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    dimensions = np.arange(len(kl_values))
    ax.bar(dimensions, kl_values, color="#4c78a8", edgecolor="#1f3b5d")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence by Latent Dimension")
    ax.set_xticks(dimensions)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> int:
    """Generate a KL divergence bar plot for one test sample."""
    try:
        device = get_device()
        model = load_model()
        dataset = load_test_dataset()
        results_path = ensure_results_directory()

        image, _ = dataset[0]
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            mu, logvar = model.encode(input_tensor)

        kl_values = compute_kl(mu, logvar)
        output_path = results_path / "kl_plot.png"
        save_kl_plot(kl_values, output_path)

        print("KL plot saved successfully.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
