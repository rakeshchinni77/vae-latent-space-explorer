# Architecture

## System Overview

The VAE Latent Space Explorer is designed as a modular generative modeling pipeline with a clear separation between data ingestion, model computation, and user-facing analysis.

## Data Flow

```text
FashionMNIST
↓
DataLoader
↓
Encoder
↓
μ + logσ²
↓
Reparameterization
↓
Latent Space
↓
Decoder
↓
Reconstruction
↓
ELBO Loss
↓
Optimizer
```

## Components

### DataLoader

`app/utils/dataset.py` loads FashionMNIST from the local `data/` directory and exposes train/test dataloaders. It ensures the dataset is available before training and evaluation.

### Encoder

`models/encoder.py` is a convolutional encoder that transforms input images into latent distribution parameters:

- Input shape: `(B, 1, 28, 28)`
- Output: `(B, latent_dim)` for `mu` and `logvar`

### Reparameterization

The VAE reparameterization trick samples latent vectors from the approximate posterior while preserving gradients. The sampled `z` is used by the decoder to reconstruct images.

### Latent Space

The model uses a 16-dimensional latent space. The prior is a standard normal distribution, and KL divergence regularization encourages the posterior to remain close to that prior.

### Decoder

`models/decoder.py` reconstructs images from latent vectors:

- Input shape: `(B, latent_dim)`
- Output shape: `(B, 1, 28, 28)`
- Final activation: Sigmoid for pixel values in `[0, 1]`

### ELBO Loss

`models/loss.py` implements the ELBO objective with:

- Reconstruction loss: binary cross-entropy
- KL divergence: analytical Gaussian formula

The combined loss encourages both accurate reconstruction and latent regularization.

## Streamlit Pages

- `app/main.py`: homepage and sidebar configuration
- `app/pages/1_Latent_Explorer.py`: interactive latent vector generation and sample decoding
- `app/pages/2_Reconstruction.py`: original vs reconstructed image comparison with error heatmap
- `app/pages/3_KL_Analysis.py`: latent dimension KL divergence analysis and collapse detection

## Evaluation Scripts

Standalone scripts provide offline analysis without the dashboard:

- `scripts/generate_reconstruction.py`
- `scripts/generate_kl_plot.py`
- `scripts/export_samples.py`

## Configuration

The project uses `.env` and `configs/settings.py` to centralize runtime values. `configs/config.py` exposes a typed configuration layer for data paths, model settings, training parameters, and Streamlit options.
