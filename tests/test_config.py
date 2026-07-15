from __future__ import annotations

from configs.config import (
    dataset_config,
    model_config,
    paths_config,
    streamlit_config,
    training_config,
)


def test_config_objects_load_correctly() -> None:
    assert model_config.latent_dim > 0
    assert training_config.batch_size > 0
    assert training_config.epochs > 0
    assert training_config.learning_rate > 0
    assert paths_config.model_path.exists()
    assert dataset_config.data_dir.exists()
    assert streamlit_config.port > 0
