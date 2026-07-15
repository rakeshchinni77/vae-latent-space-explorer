from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from app.utils.dataset import get_test_dataset
from app.utils.inference import load_vae_checkpoint
from configs.config import model_config, paths_config, training_config

PAGE_TITLE = "KL Divergence Analysis"
PAGE_DESCRIPTION = (
    "Inspect the contribution of each latent dimension to the KL divergence in a trained VAE. "
    "KL divergence regularizes the approximate posterior toward the standard normal prior, "
    "and low per-dimension KL values may indicate posterior collapse."
)
COLLAPSE_THRESHOLD = 0.01


def configure_page() -> None:
    """Configure Streamlit page metadata for the KL analysis page."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📊", layout="wide")


def load_sample_index() -> int:
    """Return the current KL analysis sample index from session state."""
    if "kl_analysis_index" not in st.session_state:
        st.session_state.kl_analysis_index = 0
    return int(st.session_state.kl_analysis_index)


def render_sidebar_controls(max_index: int) -> int:
    """Render the sample selection controls and return the selected index."""
    st.sidebar.header("Sample Selection")
    st.sidebar.write("Choose a test image from the FashionMNIST dataset.")

    if st.sidebar.button("🎲 Random Test Image"):
        st.session_state.kl_analysis_index = int(torch.randint(0, max_index + 1, (1,)).item())

    selected_index = st.sidebar.number_input(
        label="Manual Image Index",
        min_value=0,
        max_value=max_index,
        value=load_sample_index(),
        step=1,
    )
    st.session_state.kl_analysis_index = int(selected_index)
    return int(selected_index)


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Prepare a single FashionMNIST image tensor for model encoding."""
    return image.unsqueeze(0).to(training_config.device)


def compute_kl_per_dimension(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence per latent dimension for a single sample."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def build_kl_figure(kl_values: np.ndarray) -> plt.Figure:
    """Build a bar chart figure for per-dimension KL divergence."""
    figure, ax = plt.subplots(figsize=(10, 4), dpi=120)
    dimensions = np.arange(len(kl_values))
    ax.bar(dimensions, kl_values, color="#4c78a8", edgecolor="#1f3b5d")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence by Latent Dimension")
    ax.set_xticks(dimensions)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def build_summary_table(mu: torch.Tensor, logvar: torch.Tensor, kl_values: np.ndarray) -> pd.DataFrame:
    """Build a sorted summary table for latent statistics."""
    data = {
        "Dimension": np.arange(len(kl_values)),
        "Mu": mu.squeeze(0).cpu().numpy(),
        "LogVar": logvar.squeeze(0).cpu().numpy(),
        "KL": kl_values,
    }
    dataframe = pd.DataFrame(data)
    return dataframe.sort_values("KL", ascending=False).reset_index(drop=True)


def render_summary_metrics(kl_values: np.ndarray, mu: torch.Tensor, logvar: torch.Tensor, collapsed_count: int) -> None:
    """Render summary metrics for the KL analysis results."""
    total_kl = float(np.sum(kl_values))
    mean_kl = float(np.mean(kl_values))
    max_kl = float(np.max(kl_values))
    min_kl = float(np.min(kl_values))
    avg_abs_mu = float(torch.abs(mu).mean().item())
    avg_logvar = float(logvar.mean().item())

    cols = st.columns(6)
    cols[0].metric("Total KL", f"{total_kl:.4f}")
    cols[1].metric("Mean KL", f"{mean_kl:.4f}")
    cols[2].metric("Max KL", f"{max_kl:.4f}")
    cols[3].metric("Min KL", f"{min_kl:.4f}")
    cols[4].metric("Collapsed Dimensions", f"{collapsed_count} / {model_config.latent_dim}")
    cols[5].metric("Avg |μ|", f"{avg_abs_mu:.4f}")

    with st.expander("KL Summary Details"):
        st.markdown(f"- **Average logσ²:** {avg_logvar:.4f}")
        if collapsed_count >= 3:
            st.warning(
                "Multiple dimensions exhibit very low KL divergence, which may indicate posterior collapse. "
                "Investigate model capacity and training dynamics."
            )
        else:
            st.success("Latent utilization is healthy for the selected sample.")


def render_latent_table(summary_table: pd.DataFrame) -> None:
    """Render the detailed latent statistics table."""
    st.subheader("KL Divergence Table")
    st.dataframe(summary_table, use_container_width=True)


def render_explanation() -> None:
    """Render an explanation of posterior collapse and KL divergence."""
    st.expander("What is Posterior Collapse?").write(
        "KL divergence measures how much the approximate posterior differs from the prior. "
        "In a VAE, each latent dimension should carry useful information without straying too far from N(0,1). "
        "Posterior collapse occurs when one or more dimensions contribute almost no KL divergence, "
        "meaning the encoder ignores those dimensions and the decoder receives little information from them. "
        "Monitoring per-dimension KL values helps identify underutilized latent dimensions and improve model training."
    )


def main() -> None:
    """Render the KL Divergence Analysis page."""
    configure_page()
    st.title(PAGE_TITLE)
    st.write(PAGE_DESCRIPTION)

    if not paths_config.model_path.exists():
        st.error("Model checkpoint not found.")
        return

    try:
        dataset = get_test_dataset()
    except Exception as exc:
        st.error("Unable to load the FashionMNIST test dataset.")
        st.exception(exc)
        return

    model = load_vae_checkpoint(paths_config.model_path, training_config.device)
    selected_index = render_sidebar_controls(len(dataset) - 1)

    try:
        image, _ = dataset[selected_index]
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            mu, logvar = model.encode(input_tensor)

        kl_tensor = compute_kl_per_dimension(mu, logvar).squeeze(0)
        kl_values = kl_tensor.cpu().numpy()
        collapsed_count = int(np.sum(kl_values < COLLAPSE_THRESHOLD))

        chart = build_kl_figure(kl_values)
        st.pyplot(chart)

        render_summary_metrics(kl_values, mu, logvar, collapsed_count)
        render_latent_table(build_summary_table(mu, logvar, kl_values))
        render_explanation()
    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
