from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from app.utils.helpers import (
    build_latent_vector,
    checkpoint_file_label,
    compute_norm,
    format_latent_vector,
    get_default_latent_vector,
    get_random_latent_vector,
    normalize_device_label,
)
from app.utils.inference import decode_latent_image, load_vae_checkpoint
from configs.config import model_config, paths_config, training_config


PAGE_TITLE = "Latent Space Explorer"
PAGE_DESCRIPTION = (
    "Explore the trained VAE latent space by adjusting each dimension with sliders. "
    "The decoder generates FashionMNIST images directly from the latent vector." 
)


def configure_page() -> None:
    """Configure Streamlit page metadata for the latent explorer."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🧭", layout="wide")


def initialize_slider_state(latent_dim: int) -> list[float]:
    """Return the current slider values stored in session state."""
    if "latent_vector" not in st.session_state:
        st.session_state.latent_vector = get_default_latent_vector(latent_dim)
    return st.session_state.latent_vector


def render_sidebar_controls(latent_dim: int) -> list[float]:
    """Render latent dimension controls in the sidebar and return the latent vector."""
    st.sidebar.header("Latent Space Controls")
    st.sidebar.write(f"Latent Dimension: {latent_dim}")

    if st.sidebar.button("Reset Latent Vector"):
        st.session_state.latent_vector = get_default_latent_vector(latent_dim)

    if st.sidebar.button("Random Latent Vector"):
        st.session_state.latent_vector = get_random_latent_vector(latent_dim)

    latent_vector = initialize_slider_state(latent_dim)
    for index in range(latent_dim):
        latent_vector[index] = st.sidebar.slider(
            label=f"Latent Dimension {index}",
            min_value=-3.0,
            max_value=3.0,
            value=float(latent_vector[index]),
            step=0.1,
        )

    st.sidebar.divider()
    return latent_vector


def render_latent_statistics(z: torch.Tensor) -> None:
    """Render latent statistics such as norm and distribution information."""
    st.subheader("Latent Statistics")
    st.markdown("- **Expected Prior:** N(0,1)")
    st.markdown(f"- **Current Norm:** {compute_norm(z):.3f}")
    st.markdown(f"- **Device:** {normalize_device_label(str(z.device))}")


def render_writer(z: torch.Tensor, generated_image: torch.Tensor) -> None:
    """Render the generated image and current latent vector information."""
    image = generated_image
    st.header("Generated Image")
    st.image(image, caption="Generated FashionMNIST sample", clamp=True, use_column_width=True)
    st.markdown(
        f"**Minimum pixel:** {float(image.min()):.3f}  \
         **Maximum pixel:** {float(image.max()):.3f}  \
         **Mean pixel intensity:** {float(image.mean()):.3f}"
    )

    st.subheader("Current Latent Vector")
    st.code(format_latent_vector(z))


def render_explanation() -> None:
    """Render the explanatory information for the latent space explorer."""
    st.info(
        "A Variational Autoencoder learns a continuous latent space. Moving a slider changes one latent "
        "dimension while keeping the others fixed. The decoder transforms the latent vector into a generated "
        "FashionMNIST image. Values outside ±3 move into low-density regions where image quality may degrade."
    )


def main() -> None:
    """Render the latent space explorer page."""
    configure_page()
    st.title(PAGE_TITLE)
    st.write(PAGE_DESCRIPTION)

    if not paths_config.model_path.exists():
        st.error("Model checkpoint not found at the expected path. Please add models/vae.pt.")
        return

    model = load_vae_checkpoint(paths_config.model_path, training_config.device)
    latent_vector_values = render_sidebar_controls(model_config.latent_dim)

    try:
        z = build_latent_vector(latent_vector_values).to(training_config.device)
        generated_image = decode_latent_image(model, z)
    except Exception as exc:
        st.exception(exc)
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        render_writer(z, generated_image)
    with col2:
        render_latent_statistics(z)

    render_explanation()


if __name__ == "__main__":
    main()
