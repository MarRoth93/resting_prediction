"""Project configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load and minimally validate the single supported pipeline config."""
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    if config.get("release", {}).get("status") != "frozen":
        raise ValueError(f"Expected a frozen release config in {path}.")
    if config.get("features", {}).get("type") != "clip":
        raise ValueError("The frozen pipeline supports only CLIP features.")
    if config.get("encoding", {}).get("architecture") != "tribe_static_transformer":
        raise ValueError("The frozen pipeline supports only the static TRIBE transformer.")
    alignment = config.get("alignment", {})
    if alignment.get("connectivity_mode") != "external_seed_bank":
        raise ValueError("The frozen pipeline supports only external seed-bank connectivity.")
    if alignment.get("experiment_mode") != "hybrid_cha":
        raise ValueError("The frozen pipeline supports only hybrid CHA alignment.")
    if config.get("analysis_mask", {}).get("mode") != "nsdgeneral":
        raise ValueError("The frozen pipeline supports only the nsdgeneral mask.")
    return config
