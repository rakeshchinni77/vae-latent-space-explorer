from __future__ import annotations

import pytest
import torch

from models.encoder import Encoder


def test_encoder_outputs_shapes() -> None:
    encoder = Encoder()
    inputs = torch.randn(4, 1, 28, 28)
    mu, logvar = encoder(inputs)

    assert mu.shape == (4, encoder.latent_dim)
    assert logvar.shape == (4, encoder.latent_dim)


def test_encoder_invalid_input_raises() -> None:
    encoder = Encoder()
    invalid_input = torch.randn(4, 3, 28, 28)
    with pytest.raises(ValueError):
        encoder(invalid_input)
