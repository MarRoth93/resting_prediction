"""
Extract stimulus features from NSD images using pretrained vision models.

Supports: CLIP ViT-L/14, DINOv2 ViT-L/14
"""

import logging
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

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


def extract_dinov2_features(
    stimuli_path: str,
    output_path: str,
    model_name: str = "dinov2_vitl14",
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract DINOv2 features for all NSD stimuli.

    Returns:
        (N_stimuli, feature_dim) float32 array
    """
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device).eval()

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    f = h5py.File(stimuli_path, "r")
    images = f["imgBrick"]
    n_images = images.shape[0]

    # Determine feature dim
    from PIL import Image
    dummy = Image.fromarray(images[0])
    dummy_tensor = preprocess(dummy).unsqueeze(0).to(device)
    with torch.no_grad():
        dummy_feat = model(dummy_tensor)
    feat_dim = dummy_feat.shape[1]
    logger.info(f"DINOv2 {model_name}: feature_dim={feat_dim}, n_images={n_images}")

    features = np.zeros((n_images, feat_dim), dtype=np.float32)

    for start in tqdm(range(0, n_images, batch_size), desc="DINOv2 features"):
        end = min(start + batch_size, n_images)
        batch_imgs = []
        for i in range(start, end):
            img = Image.fromarray(images[i])
            batch_imgs.append(preprocess(img))
        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            batch_features = model(batch_tensor)
        features[start:end] = batch_features.cpu().numpy()

    f.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features)
    logger.info(f"Saved DINOv2 features: {features.shape} to {output_path}")
    return features


def extract_all_features(
    stimuli_path: str = "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
    output_dir: str = "processed_data/features",
    models: list[str] | None = None,
    device: str = "cuda",
    batch_size: int = 64,
):
    """Extract features using all specified models."""
    if models is None:
        models = ["clip", "dinov2"]

    os.makedirs(output_dir, exist_ok=True)

    for model_type in models:
        output_path = os.path.join(output_dir, f"{model_type}_features.npy")
        if os.path.exists(output_path):
            logger.info(f"{model_type} features already exist at {output_path}, skipping")
            continue

        if model_type == "clip":
            extract_clip_features(stimuli_path, output_path, batch_size=batch_size, device=device)
        elif model_type == "dinov2":
            extract_dinov2_features(stimuli_path, output_path, batch_size=batch_size, device=device)
        else:
            logger.warning(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Extract stimulus features")
    parser.add_argument("--stimuli", default="nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    parser.add_argument("--output-dir", default="processed_data/features")
    parser.add_argument("--models", nargs="+", default=["clip", "dinov2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    extract_all_features(args.stimuli, args.output_dir, args.models, args.device, args.batch_size)
