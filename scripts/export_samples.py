from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from eval_utils import ensure_results_directory, get_device, load_model


def save_generated_samples(model: torch.nn.Module, output_path: Path, device: torch.device) -> None:
    """Generate a 4x4 grid of decoded samples from random latent vectors."""
    latent_dim = model.latent_dim
    with torch.no_grad():
        random_latents = torch.randn(16, latent_dim, device=device)
        generated = model.decode(random_latents).clamp(0.0, 1.0).cpu()

    figure, axes = plt.subplots(4, 4, figsize=(8, 8), constrained_layout=True)
    for index, ax in enumerate(axes.flatten()):
        image = generated[index].squeeze(0).numpy()
        ax.imshow(image, cmap="gray")
        ax.axis("off")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> int:
    """Generate a 4x4 grid of decoded random samples."""
    try:
        device = get_device()
        model = load_model()
        results_path = ensure_results_directory()
        output_path = results_path / "generated_samples.png"
        save_generated_samples(model, output_path, device)

        print("Generated samples exported successfully.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
