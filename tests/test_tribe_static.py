import json

import numpy as np
import pytest
import yaml

from src.models.encoding_factory import build_encoder, load_encoder, save_encoder
from src.models.nonlinear_encoding import StaticTransformerConfig, StaticTransformerEncoder

torch = pytest.importorskip("torch")


def test_frozen_config_builds_selected_transformer():
    with open("config.yaml") as handle:
        config = yaml.safe_load(handle)

    encoder = build_encoder(
        config=config,
        input_dim=768,
        output_dim=100,
        feature_slices={"clip": (0, 768)},
    )

    assert config["release"]["status"] == "frozen"
    assert config["subjects"] == {"train": [1, 2, 3, 4, 5, 6], "test": [7]}
    assert encoder.architecture == "tribe_static_transformer"
    assert encoder.config.transformer_d_model == 384
    assert encoder.config.transformer_layers == 8
    assert encoder.config.transformer_heads == 8
    assert encoder.config.dropout == 0.2
    assert encoder.config.learning_rate == pytest.approx(0.0002084109344364613)
    assert encoder.config.weight_decay == pytest.approx(3.0201957739732042e-05)


def test_transformer_fit_save_load(tmp_path):
    rng = np.random.RandomState(3)
    X = rng.randn(36, 10).astype(np.float32)
    Z = (X[:, :4] @ rng.randn(4, 3)).astype(np.float32)
    cfg = StaticTransformerConfig(
        architecture="tribe_static_transformer",
        transformer_d_model=16,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_multiplier=2.0,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=12,
        max_epochs=2,
        patience=2,
        val_fraction=0.2,
        device="cpu",
        seed=3,
    )
    encoder = StaticTransformerEncoder(
        input_dim=10,
        output_dim=3,
        config=cfg,
        feature_slices={"clip": (0, 10)},
    )

    encoder.fit(X, Z, sample_groups=np.arange(X.shape[0], dtype=np.int64))
    before = encoder.predict(X[:4])
    save_encoder(encoder, str(tmp_path))
    loaded = load_encoder(str(tmp_path))

    with open(tmp_path / "encoder" / "metadata.json") as handle:
        metadata = json.load(handle)
    assert metadata["architecture"] == "tribe_static_transformer"
    np.testing.assert_allclose(loaded.predict(X[:4]), before, atol=1e-6)


def test_transformer_validates_attention_width():
    with pytest.raises(ValueError, match="divisible"):
        StaticTransformerConfig(
            architecture="tribe_static_transformer",
            transformer_d_model=15,
            transformer_heads=4,
        )


def test_grouped_validation_keeps_stimuli_disjoint():
    cfg = StaticTransformerConfig(
        architecture="tribe_static_transformer",
        val_fraction=0.25,
        validation_mode="stimulus_grouped",
        seed=7,
    )
    encoder = StaticTransformerEncoder(
        input_dim=2,
        output_dim=2,
        config=cfg,
        feature_slices={"clip": (0, 2)},
    )
    groups = np.repeat(np.arange(8), 3)

    train_idx, val_idx = encoder._validation_indices(len(groups), groups)

    assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
    assert encoder.validation_summary["n_val_groups"] == 2
