from __future__ import annotations

import streamlit as st
import torch
import torch.nn.functional as F

from app.utils.dataset import get_test_dataset
from app.utils.inference import load_vae_checkpoint, run_reconstruction
from app.utils.plotting import make_error_heatmap
from configs.config import model_config, paths_config, training_config


PAGE_TITLE = "Reconstruction Explorer"
PAGE_DESCRIPTION = (
    "Visualize the VAE reconstruction pipeline for FashionMNIST: original image, encoded latent "
    "parameters, decoded reconstruction, and absolute error heatmap."
)


def configure_page() -> None:
    """Configure Streamlit page metadata for the reconstruction explorer."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🧪", layout="wide")


def load_sample_index() -> int:
    """Return the current sample index from session state."""
    if "reconstruction_index" not in st.session_state:
        st.session_state.reconstruction_index = 0
    return int(st.session_state.reconstruction_index)


def render_controls(max_index: int) -> int:
    """Render sample selection controls and return the selected index."""
    st.sidebar.header("Sample Selection")
    st.sidebar.write("Choose a test image index or sample a random one.")

    if st.sidebar.button("🎲 Random Test Image"):
        st.session_state.reconstruction_index = int(torch.randint(0, max_index + 1, (1,)).item())

    selected_index = st.sidebar.number_input(
        label="Manual Image Index",
        min_value=0,
        max_value=max_index,
        value=load_sample_index(),
        step=1,
    )
    st.session_state.reconstruction_index = int(selected_index)
    return int(selected_index)


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Prepare a single image tensor for model inference."""
    return image.unsqueeze(0).to(training_config.device)


def render_image_columns(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    heatmap_figure: plt.Figure,
) -> None:
    """Render the original image, reconstructed image, and heatmap in three columns."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(original, clamp=True, use_column_width=True)
    with col2:
        st.subheader("Reconstructed")
        st.image(reconstruction, clamp=True, use_column_width=True)
    with col3:
        st.subheader("Error Heatmap")
        st.pyplot(heatmap_figure)


def compute_metrics(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    latent_vector: torch.Tensor,
) -> dict[str, float]:
    """Compute reconstruction metrics for display."""
    abs_error = torch.abs(original - reconstruction)
    bce = F.binary_cross_entropy(reconstruction, original, reduction="mean")
    return {
        "MAE": float(abs_error.mean().item()),
        "Max Error": float(abs_error.max().item()),
        "Min Error": float(abs_error.min().item()),
        "Reconstruction BCE": float(bce.item()),
        "Latent Norm": float(torch.norm(latent_vector).item()),
    }


def render_metrics(metrics: dict[str, float]) -> None:
    """Render the reconstruction metrics below the image columns."""
    cols = st.columns(5)
    for label, value in metrics.items():
        cols[list(metrics.keys()).index(label)].metric(label, f"{value:.4f}")


def render_latent_info(mu: torch.Tensor, logvar: torch.Tensor, index: int) -> None:
    """Render detailed latent statistics inside an expander."""
    with st.expander("Latent Information"):
        st.markdown(f"- **μ Mean:** {mu.mean().item():.4f}")
        st.markdown(f"- **μ Std:** {mu.std().item():.4f}")
        st.markdown(f"- **Log Variance Mean:** {logvar.mean().item():.4f}")
        st.markdown(f"- **Log Variance Std:** {logvar.std().item():.4f}")
        st.markdown(f"- **Latent Dimension:** {model_config.latent_dim}")
        st.markdown(f"- **Current Sample Index:** {index}")


def render_explanation() -> None:
    """Render the explanatory information for the reconstruction explorer."""
    st.info(
        "The left image is the original FashionMNIST sample. The center image is reconstructed by the VAE. "
        "The heatmap visualizes absolute reconstruction error. Bright regions indicate pixels the model struggles to reconstruct."
    )


def main() -> None:
    """Render the reconstruction explorer page."""
    configure_page()
    st.title(PAGE_TITLE)
    st.write(PAGE_DESCRIPTION)

    if not paths_config.model_path.exists():
        st.error("Model checkpoint not found at the expected path. Please add models/vae.pt.")
        return

    try:
        dataset = get_test_dataset()
    except Exception as exc:
        st.error("Unable to load the FashionMNIST test dataset. Ensure the data is available.")
        st.exception(exc)
        return

    model = load_vae_checkpoint(paths_config.model_path, training_config.device)
    selected_index = render_controls(len(dataset) - 1)

    try:
        image, _ = dataset[selected_index]
        original = preprocess_image(image)
        reconstruction, mu, logvar, z = run_reconstruction(model, original)
        error = torch.abs(original - reconstruction)
        heatmap_figure = make_error_heatmap(error.squeeze(0))

        render_image_columns(
            original.squeeze(0).squeeze(0).cpu().numpy(),
            reconstruction.squeeze(0).squeeze(0).cpu().numpy(),
            heatmap_figure,
        )
        metrics = compute_metrics(original, reconstruction, mu, logvar, z)
        render_metrics(metrics)
        render_latent_info(mu, logvar, selected_index)
        render_explanation()
    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
