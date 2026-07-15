from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from eval_utils import ensure_results_directory, get_device, load_model, load_test_dataset


def save_reconstruction_figure(original: torch.Tensor, reconstruction: torch.Tensor, error: torch.Tensor, output_path: Path) -> None:
    """Save a reconstruction comparison figure to disk."""
    original_image = original.squeeze().cpu().numpy()
    reconstruction_image = reconstruction.squeeze().cpu().numpy()
    error_image = error.squeeze().cpu().numpy()

    figure, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(reconstruction_image, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(error_image, cmap="inferno")
    axes[2].set_title("Error Heatmap")
    axes[2].axis("off")

    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> int:
    """Generate a reconstruction figure for one test sample."""
    try:
        device = get_device()
        model = load_model()
        dataset = load_test_dataset()
        results_path = ensure_results_directory()

        image, _ = dataset[0]
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            reconstruction, _, _ = model(input_tensor)

        error = torch.abs(input_tensor - reconstruction)
        output_path = results_path / "reconstruction.png"
        save_reconstruction_figure(input_tensor, reconstruction, error, output_path)

        print("Reconstruction image saved successfully.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
