"""Train the FOR-compatible, parcel-level Schaefer-400 multi-expert model."""

from __future__ import annotations

import logging
from pathlib import Path

from src.pipelines.schaefer400_support import (
    fit_schaefer400_fusion,
    prepare_schaefer400_training,
    save_schaefer400_model,
)
from src.schaefer400_config import load_schaefer400_config, resolve_schaefer400_roots


logger = logging.getLogger(__name__)


def train_schaefer400(
    *,
    config_path: str = "config_schaefer400.yaml",
    data_root: str | None = None,
    raw_data_root: str | None = None,
    output_dir: str | None = None,
) -> dict:
    config = resolve_schaefer400_roots(
        load_schaefer400_config(config_path),
        data_root=data_root,
        raw_data_root=raw_data_root,
    )
    destination = Path(output_dir or (Path(config["output_root"]) / "model"))
    prepared = prepare_schaefer400_training(config)
    encoder = fit_schaefer400_fusion(prepared)
    manifest = save_schaefer400_model(prepared, encoder, output_dir=destination)
    logger.info("Saved Schaefer-400 model to %s", destination)
    return manifest


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_schaefer400.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--raw-data-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    train_schaefer400(
        config_path=args.config,
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
        output_dir=args.output_dir,
    )
