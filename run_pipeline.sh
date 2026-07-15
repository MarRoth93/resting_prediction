#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/psycontrol/miniforge3/envs/resting-prediction/bin/python}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/media/psycontrol/HDD/Datasets/brain-diffuser/data}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/processed}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/artifacts/model}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${ROOT}/third_party}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/resting_prediction-matplotlib}"
SUBJECT="${SUBJECT:-7}"
SHOTS="${SHOTS:-100}"
PREDICTION_DIR="${PREDICTION_DIR:-${ROOT}/artifacts/predictions/subj$(printf '%02d' "${SUBJECT}")}"

usage() {
  cat <<'EOF'
Usage: ./run_pipeline.sh COMMAND

Commands:
  check        Validate the frozen config, model, data, and decoder assets.
  train        Reproduce the frozen seed-42 model in artifacts/retrained_model.
  predict      Run zero-shot and 100-shot prediction for subject 7.
  reconstruct  Reconstruct images from the saved predictions.

Override defaults with SUBJECT, SHOTS, PYTHON, RAW_DATA_ROOT, DATA_ROOT,
MODEL_DIR, PREDICTION_DIR, or THIRD_PARTY_ROOT.
EOF
}

case "${1:-}" in
  check)
    DATA_ROOT="${DATA_ROOT}" MODEL_DIR="${MODEL_DIR}" THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT}" \
      PYTHONPATH=. "${PYTHON}" - <<'PY'
import os
import json
from pathlib import Path

import numpy as np

from src.alignment.shared_space import SharedSpaceBuilder
from src.config import load_config
from src.models.encoding_factory import load_encoder

config = load_config("config.yaml")
assert config["features"]["streams"] == ["clip"]
assert config["subjects"] == {"train": [1, 2, 3, 4, 5, 6], "test": [7]}
data_root = Path(os.environ["DATA_ROOT"])
model_dir = Path(os.environ["MODEL_DIR"])
third_party = Path(os.environ["THIRD_PARTY_ROOT"])
required = [data_root / "features" / "clip_features.npy"]
for subject in range(1, 8):
    subject_dir = data_root / f"subj{subject:02d}"
    required.extend(
        subject_dir / name
        for name in ("train_fmri.npy", "test_fmri.npy", "mask.npy", "rest_run1.npy")
    )
recon = data_root / "reconstruction_features" / "subj07"
required.extend((
    data_root / "subj07" / "test_fmri_trials.npy",
    data_root / "subj07" / "test_trial_labels.npy",
))
required.extend(recon / name for name in (
    "vdvae_features.npz", "ref_latents.npz",
    "cliptext_train.npy", "cliptext_test.npy",
    "clipvision_train.npy", "clipvision_test.npy",
    "cliptext_train_stim_idx.npy", "cliptext_test_stim_idx.npy",
    "clipvision_train_stim_idx.npy", "clipvision_test_stim_idx.npy",
))
required.extend((
    model_dir / "builder.npz",
    model_dir / "shared_stim_idx.npy",
    model_dir / "external_seed_info.json",
    model_dir / "encoder" / "model.pt",
    third_party / "vdvae" / "model" / "imagenet64-iter-1600000-model.th",
    third_party / "versatile_diffusion" / "pretrained" / "vd-four-flow-v1-0-fp16-deprecated.pth",
))
for path in required:
    if not path.exists():
        raise FileNotFoundError(path)
builder = SharedSpaceBuilder.load(str(model_dir))
encoder = load_encoder(str(model_dir))
assert builder.connectivity_mode == "external_seed_bank"
assert builder.experiment_mode == "hybrid_cha"
assert builder.template_fingerprint.shape == (499, 100)
assert encoder.architecture == "tribe_static_transformer"
assert encoder.input_dim == 768 and encoder.output_dim == 100
with np.load(recon / "vdvae_features.npz") as vdvae:
    assert set(vdvae.files) == {"train_latents", "test_latents", "train_stim_idx", "test_stim_idx"}
    assert vdvae["train_latents"].shape[0] == vdvae["train_stim_idx"].shape[0]
    assert vdvae["test_latents"].shape[0] == vdvae["test_stim_idx"].shape[0]
for feature_name in ("cliptext", "clipvision"):
    for split in ("train", "test"):
        features = np.load(recon / f"{feature_name}_{split}.npy", mmap_mode="r")
        stimuli = np.load(recon / f"{feature_name}_{split}_stim_idx.npy", mmap_mode="r")
        assert features.shape[0] == stimuli.shape[0]
with open(model_dir / "external_seed_info.json") as handle:
    assert json.load(handle)["n_seeds"] == 499
print("Frozen pipeline is ready.")
PY
    ;;
  train)
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.train_shared_space \
      --config config.yaml \
      --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" \
      --output-dir artifacts/retrained_model
    ;;
  predict)
    mkdir -p "${PREDICTION_DIR}"
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.predict_subject \
      --mode zero_shot --test-sub "${SUBJECT}" \
      --config config.yaml \
      --model-dir "${MODEL_DIR}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" \
      --output-dir "${PREDICTION_DIR}"
    PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.predict_subject \
      --mode few_shot --test-sub "${SUBJECT}" --n-shots "${SHOTS}" --seed 42 \
      --config config.yaml \
      --model-dir "${MODEL_DIR}" --data-root "${DATA_ROOT}" \
      --raw-data-root "${RAW_DATA_ROOT}" \
      --output-dir "${PREDICTION_DIR}"
    ;;
  reconstruct)
    mkdir -p "${MPLCONFIGDIR}"
    MPLCONFIGDIR="${MPLCONFIGDIR}" PYTHONPATH=. "${PYTHON}" -u -m src.pipelines.benchmark_reconstructions_vdvae_vd \
      --test-sub "${SUBJECT}" --config config.yaml --data-root "${DATA_ROOT}" \
      --predictions-dir "${PREDICTION_DIR}" --fewshot-dir "${PREDICTION_DIR}" \
      --output-dir "artifacts/reconstructions/subj$(printf '%02d' "${SUBJECT}")" \
      --recon-model-root "${THIRD_PARTY_ROOT}" \
      --fewshot-n-shots "${SHOTS}" --fewshot-seed 42 --device cuda
    ;;
  *)
    usage
    exit 2
    ;;
esac
