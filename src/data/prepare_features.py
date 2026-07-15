"""
Extract frozen CLIP ViT-L/14 features from NSD images.
"""

from __future__ import annotations

import logging
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.data.shared_paths import default_stimuli_hdf5

logger = logging.getLogger(__name__)


def extract_clip_features(
    stimuli_path: str,
    output_path: str,
    model_name: str = "ViT-L/14",
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract CLIP features for all NSD stimuli.

    Args:
        stimuli_path: path to nsd_stimuli.hdf5
        output_path: where to save features .npy
        model_name: CLIP model variant
        batch_size: images per batch
        device: 'cuda' or 'cpu'

    Returns:
        (N_stimuli, feature_dim) float32 array
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    model = model.to(device).eval()

    f = h5py.File(stimuli_path, "r")
    images = f["imgBrick"]
    n_images = images.shape[0]

    # Determine feature dim with a dummy forward pass
    from PIL import Image
    dummy = Image.fromarray(images[0])
    dummy_tensor = preprocess(dummy).unsqueeze(0).to(device)
    with torch.no_grad():
        dummy_feat = model.encode_image(dummy_tensor)
    feat_dim = dummy_feat.shape[1]
    logger.info(f"CLIP {model_name}: feature_dim={feat_dim}, n_images={n_images}")

    features = np.zeros((n_images, feat_dim), dtype=np.float32)

    for start in tqdm(range(0, n_images, batch_size), desc="CLIP features"):
        end = min(start + batch_size, n_images)
        batch_imgs = []
        for i in range(start, end):
            img = Image.fromarray(images[i])
            batch_imgs.append(preprocess(img))
        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor)
        features[start:end] = batch_features.cpu().numpy()

    f.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features)
    logger.info(f"Saved CLIP features: {features.shape} to {output_path}")
    return features


def extract_features(
    stimuli_path: str = default_stimuli_hdf5(),
    output_dir: str = "data/processed/features",
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract the one feature representation used by the frozen model."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clip_features.npy")
    if os.path.exists(output_path):
        logger.info("CLIP features already exist at %s, skipping", output_path)
        return np.load(output_path, mmap_mode="r")
    return extract_clip_features(
        stimuli_path,
        output_path,
        batch_size=batch_size,
        device=device,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Extract frozen CLIP stimulus features")
    parser.add_argument("--stimuli", default=default_stimuli_hdf5())
    parser.add_argument("--output-dir", default="data/processed/features")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    extract_features(args.stimuli, args.output_dir, args.device, args.batch_size)
