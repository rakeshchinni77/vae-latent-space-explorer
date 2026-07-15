# VAE Latent Space Explorer

A production-ready Variational Autoencoder (VAE) project for FashionMNIST with an interactive Streamlit dashboard for latent space exploration, image reconstruction, and KL divergence analysis.

## Project Features

- CNN-based Variational Autoencoder
- FashionMNIST Dataset
- Latent Space Exploration
- Image Reconstruction
- KL Divergence Analysis
- ELBO Loss
- KL Annealing
- Interactive Streamlit Dashboard
- Evaluation Scripts
- Pytest Test Suite
- Docker Support
- Professional Project Structure

## Repository Structure

```text
app/
configs/
data/
models/
results/
scripts/
tests/
docs/
screenshots/
```

## Installation

1. Clone the repository:

```powershell
git clone https://github.com/rakeshchinni77/vae-latent-space-explorer
cd vae-latent-space-explorer
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Configure environment variables:

```powershell
copy .env.example .env
```

5. Download the FashionMNIST dataset:

```powershell
python data/download_dataset.py
```

6. Train the model:

```powershell
python scripts/train.py
```

7. Launch the Streamlit application:

```powershell
streamlit run app/main.py
```

## Running the Project

Start the training and dashboard with:

```powershell
python scripts/train.py
streamlit run app/main.py
```

## Evaluation Scripts

The project includes standalone evaluation scripts for batch image generation and analysis:

```powershell
python scripts/generate_reconstruction.py
python scripts/generate_kl_plot.py
python scripts/export_samples.py
```

Generated outputs are saved to the `results/` directory:

- `results/reconstruction.png`
- `results/kl_plot.png`
- `results/generated_samples.png`

## Testing

Validate the repository with pytest:

```powershell
pytest
pytest --cov=. --cov-report=term-missing
```

Current status: 14 tests passing.

## Results

This project includes the following artifacts:

- Model checkpoint: `models/vae.pt`
- Training log: `results/training_log.csv`
- Generated samples: `results/generated_samples.png`
- Reconstruction output: `results/reconstruction.png`
- KL plot: `results/kl_plot.png`

## Technologies

- Python
- PyTorch
- Torchvision
- Streamlit
- Matplotlib
- NumPy
- Pandas
- Pytest
- Docker

## Future Improvements

Potential future enhancements include:

- GPU Training
- Conditional VAE
- Higher Resolution Images
- Latent Interpolation Animation
- Model Comparison

## License

This project is licensed under the MIT License.
