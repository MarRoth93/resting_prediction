"""
Encoder construction and loading utilities.
"""

from __future__ import annotations

import os
from typing import Any

from src.models.nonlinear_encoding import StaticTransformerConfig, StaticTransformerEncoder


def build_encoder(
    config: dict[str, Any],
    input_dim: int,
    output_dim: int,
    feature_slices: dict[str, tuple[int, int]] | None = None,
) -> StaticTransformerEncoder:
    """Build the frozen static TRIBE transformer."""
    encoding_cfg = config.get("encoding", {}) or {}
    transformer_cfg = StaticTransformerConfig.from_config(
        encoding_cfg,
        seed=int(config.get("random_seed", 42)),
    )
    if transformer_cfg.architecture != "tribe_static_transformer":
        raise ValueError("Only encoding.architecture=tribe_static_transformer is supported.")
    return StaticTransformerEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        config=transformer_cfg,
        feature_slices=feature_slices,
    )


def load_encoder(model_dir: str) -> StaticTransformerEncoder:
    """Load the frozen transformer artifact."""
    encoder_dir = os.path.join(model_dir, "encoder")
    if os.path.exists(os.path.join(encoder_dir, "metadata.json")):
        encoder = StaticTransformerEncoder.load(encoder_dir)
        if encoder.architecture != "tribe_static_transformer":
            raise ValueError(f"Unsupported encoder artifact: {encoder.architecture!r}.")
        return encoder
    raise FileNotFoundError(f"Missing frozen encoder artifact: {encoder_dir}/metadata.json")


def save_encoder(
    encoder: StaticTransformerEncoder,
    model_dir: str,
) -> str:
    """Save the transformer artifact."""
    if not isinstance(encoder, StaticTransformerEncoder):
        raise TypeError("Only StaticTransformerEncoder artifacts are supported.")
    out_dir = os.path.join(model_dir, "encoder")
    encoder.save(out_dir)
    return out_dir
