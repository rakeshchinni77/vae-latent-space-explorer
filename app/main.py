from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Ensure local package imports work when Streamlit runs from inside app/.
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from configs.config import model_config, paths_config, training_config

PAGE_TITLE = "VAE Latent Space Explorer"
PAGE_ICON = "🧠"
HOMEPAGE_LABEL = "🏠 Home"
APPLICATION_VERSION = "1.0.0"
FRAMEWORK = "PyTorch"
DATASET = "FashionMNIST"
MODEL_NAME = "Variational Autoencoder"
PROJECT_DESCRIPTION = (
    "A lightweight Streamlit application for exploring a trained Variational Autoencoder. "
    "The project demonstrates a PyTorch-based VAE trained on FashionMNIST with ELBO loss, "
    "KL annealing, and a simple interface for future latent space exploration."
)


def get_device_label() -> str:
    """Return a user-friendly device label for display."""
    try:
        import torch
    except ImportError:
        return training_config.device.upper()

    if torch.cuda.is_available():
        return "CUDA (NVIDIA GPU)"
    return "CPU"


def get_checkpoint_display() -> tuple[str, str]:
    """Return a display label and relative file name for the checkpoint."""
    checkpoint_exists = paths_config.model_path.exists()
    relative_path = paths_config.model_path.relative_to(paths_config.project_root).as_posix()

    if checkpoint_exists:
        return "✓ Checkpoint Loaded", relative_path
    return "🔴 Not Found", relative_path


def configure_page() -> None:
    """Configure the Streamlit page metadata and layout."""
    st.set_page_config(
        page_title=HOMEPAGE_LABEL,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> None:
    """Render the application sidebar with model and project metadata."""
    checkpoint_exists = paths_config.model_path.exists()
    checkpoint_label, _ = get_checkpoint_display()

    st.sidebar.title(HOMEPAGE_LABEL)
    st.sidebar.markdown(f"**Project:** {PAGE_TITLE}")
    st.sidebar.markdown(f"**Version:** {APPLICATION_VERSION}")
    st.sidebar.markdown(f"**Model:** {MODEL_NAME}")
    st.sidebar.markdown(f"**Device:** {get_device_label()}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Checkpoint Status**")

    if checkpoint_exists:
        st.sidebar.success(checkpoint_label)
    else:
        st.sidebar.error(checkpoint_label)
        st.sidebar.write(
            "The checkpoint was not found. A trained model file is required for later phases."
        )


def render_project_info() -> None:
    """Render the model and dataset summary cards on the homepage."""
    status_label, checkpoint_file = get_checkpoint_display()

    st.header("Project Information")

    model_card, training_card, checkpoint_card = st.columns(3)
    with model_card:
        st.subheader("Project Overview")
        st.markdown(f"- **Model Name:** {MODEL_NAME}")
        st.markdown(f"- **Dataset:** {DATASET}")
        st.markdown(f"- **Latent Dimension:** {model_config.latent_dim}")
        st.markdown(f"- **Framework:** {FRAMEWORK}")

    with training_card:
        st.subheader("Training Configuration")
        st.markdown(f"- **Device:** {get_device_label()}")
        st.markdown(f"- **Epochs:** {training_config.epochs}")
        st.markdown(f"- **Batch Size:** {training_config.batch_size}")
        st.markdown(f"- **Learning Rate:** {training_config.learning_rate:.4f}")

    with checkpoint_card:
        st.subheader("Checkpoint")
        st.markdown(f"- **Status:** {status_label}")
        st.markdown(f"- **File:** `{checkpoint_file}`")


def render_homepage() -> None:
    """Render the homepage content for the Streamlit application."""
    st.title(PAGE_TITLE)
    st.write(PROJECT_DESCRIPTION)
    st.info(
        "This application is built around a Variational Autoencoder trained with ELBO loss and "
        "KL annealing on the FashionMNIST dataset. Use the available pages to explore the latent "
        "space, visualize reconstructions, and analyze KL divergence in future phases."
    )
    render_project_info()

    if not paths_config.model_path.exists():
        st.error(
            "Model checkpoint not found. Please verify that the file exists at the configured checkpoint path."
        )


def main() -> None:
    """Entry point for the Streamlit homepage."""
    configure_page()
    render_sidebar()
    render_homepage()


if __name__ == "__main__":
    main()
