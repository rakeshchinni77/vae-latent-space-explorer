from __future__ import annotations

import pytest
import torch

from models.decoder import Decoder


def test_decoder_outputs_shape() -> None:
    decoder = Decoder()
    latent = torch.randn(4, decoder.latent_dim)
    output = decoder(latent)

    assert output.shape == (4, 1, 28, 28)


def test_decoder_invalid_latent_dim_raises() -> None:
    decoder = Decoder()
    invalid_latent = torch.randn(4, decoder.latent_dim + 1)

    with pytest.raises(ValueError):
        decoder(invalid_latent)
