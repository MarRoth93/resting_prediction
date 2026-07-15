# Resting-State Image-to-fMRI Prediction

This repository contains one frozen pipeline: static images are encoded with
CLIP, a TRIBE-style transformer predicts a shared fMRI response, and resting
state fMRI aligns that response to a specific subject's voxels. The predicted
responses can optionally be decoded into images with VDVAE and Versatile
Diffusion.

## Pipeline

```text
image -> CLIP ViT-L/14 -> frozen transformer -> 100-D shared response
resting fMRI -> 499-region connectivity -> subject alignment -> voxel response
voxel response -> VDVAE/CLIP regressors -> Versatile Diffusion -> image
```

REST is used for subject alignment. It is not concatenated to each transformer
input. Training uses task fMRI from subjects 1-6. Subject 7 is evaluation-only
and has already been used once, so it must not be used for further model
selection.

## Repository Layout

```text
config.yaml            Frozen model and preprocessing settings
run_pipeline.sh        The four supported user commands
src/alignment/         REST connectivity and shared-space alignment
src/data/              NSD preparation and CLIP/reconstruction features
src/models/            Frozen static transformer and artifact loader
src/pipelines/         Training, prediction, evaluation, and reconstruction
tests/                 Tests for the retained pipeline
data/processed/        Current Friston-24 processed dataset
artifacts/model/       Frozen seed-42 model
artifacts/results.json Final LOSO and seed-robustness result
third_party/           VDVAE and Versatile Diffusion code/checkpoints
```

Raw NSD data remains outside the repository at
`/media/psycontrol/HDD/Datasets/brain-diffuser/data`.

## Frozen Model

The authoritative settings are in `config.yaml`:

| Setting | Value |
|---|---:|
| Input | CLIP ViT-L/14, 768 dimensions |
| Shared response | 100 dimensions |
| Transformer | 8 blocks, width 384, 8 heads |
| Feed-forward width | 1536 |
| Parameters | 14,531,812 |
| Dropout | 0.20 |
| Learning rate | 2.084109344364613e-4 |
| Weight decay | 3.0201957739732042e-5 |
| Batch size | 512 |
| Early stopping | 200 epochs maximum, patience 20 |
| Validation | Stimulus-grouped Pearson correlation |
| Release seed | 42 |
| Evaluation | 200 fixed images, split seed 42 |

Across seeds 42-46, mean LOSO median voxel correlation was `0.17818`, versus
`0.17432` for ridge. All five seed-level means beat ridge. The exact retained
summary is in `artifacts/results.json`.

The optional image decoder is also fixed in `config.yaml`: VDVAE, CLIP-text,
and CLIP-vision ridge alphas are `50000`, `100000`, and `60000`; Versatile
Diffusion uses strength `0.5`, image/text mixing `0.2`, guidance `20`, and 50
DDIM steps. These ridge regressors are part of image reconstruction, not an
alternative to the frozen transformer brain-encoding model.

## Setup

The tested environment is `resting-prediction` under
`/home/psycontrol/miniforge3/envs/resting-prediction`. To recreate a compatible
environment:

```bash
conda create -n resting-prediction python=3.10 -y
conda activate resting-prediction
pip install -r requirements.txt
```

From the repository root, validate every required artifact:

```bash
./run_pipeline.sh check
```

## Use The Frozen Model

Run zero-shot prediction followed by 100-shot subject alignment for subject 7:

```bash
./run_pipeline.sh predict
```

Outputs are written to `artifacts/predictions/subj07/`. Zero-shot uses only the
subject's REST data for alignment. Few-shot additionally uses the requested
number of task responses but does not alter the frozen transformer weights.

Use another prepared subject or shot count with environment overrides:

```bash
SUBJECT=8 SHOTS=50 ./run_pipeline.sh predict
```

The supported CLI predicts stimuli already represented in
`data/processed/features/clip_features.npy`. Arbitrary images require a
compatible CLIP ViT-L/14 embedding and a prepared subject alignment; there is no
single-image CLI.

## Reproduce Training

Training a new seed-42 copy does not overwrite the frozen release:

```bash
./run_pipeline.sh train
```

The result is saved under `artifacts/retrained_model/`. The training path reads
subjects 1-6 from `data/processed`, builds the external NSD ROI seed-bank
alignment, and fits the transformer defined in `config.yaml`.

## Reconstruct Images

First create predictions, then run the retained VDVAE + Versatile Diffusion
decoder:

```bash
./run_pipeline.sh predict
./run_pipeline.sh reconstruct
```

Reconstruction uses the feature bundle in
`data/processed/reconstruction_features/subj07/` and writes images and metrics
to `artifacts/reconstructions/subj07/`.

## Regenerate Data

The retained data can be rebuilt from raw NSD files with these modules:

```bash
for subject in 1 2 3 4 5 6 7; do
  python -m src.data.prepare_task_data --sub "$subject" \
    --data-root /media/psycontrol/HDD/Datasets/brain-diffuser/data \
    --output-root data/processed
  python -m src.data.prepare_rest_data --sub "$subject" --config config.yaml \
    --data-root /media/psycontrol/HDD/Datasets/brain-diffuser/data \
    --output-root data/processed
done

python -m src.data.prepare_features \
  --output-dir data/processed/features --device cuda
```

Reconstruction features are large and already retained. To regenerate them:

```bash
python -m src.data.prepare_reconstruction_features --subject 7 \
  --data-root data/processed --recon-model-root third_party \
  --output-dir data/processed/reconstruction_features/subj07 \
  --device cuda
```

## Development

Run the retained test suite with:

```bash
PYTHONPATH=. python -m pytest -q
```

Generated predictions, reconstructions, model copies, caches, and logs are
ignored by Git. `artifacts/model` and `artifacts/results.json` are the frozen
release artifacts and should not be modified during experiments.
