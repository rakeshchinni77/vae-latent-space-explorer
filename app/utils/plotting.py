from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def make_error_heatmap(error_tensor: torch.Tensor) -> plt.Figure:
    """Create a figure for the absolute error heatmap."""
    heatmap = error_tensor.squeeze().cpu().numpy()
    figure, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.imshow(heatmap, cmap="inferno", interpolation="nearest")
    ax.axis("off")
    figure.tight_layout()
    return figure
