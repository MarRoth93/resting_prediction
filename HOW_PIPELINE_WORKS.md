# How The Pipeline Works

This file explains the logic of the project: what goes in, what happens at each stage, and what comes out.

If you just want to run it, use [RUN_PIPELINE.md](/home/rothermm/resting_prediction/RUN_PIPELINE.md).

## One-Sentence Summary

The project learns how to predict task-evoked fMRI responses for a held-out subject using:
- resting-state fMRI to build a cross-subject shared space
- task fMRI to anchor that shared space across subjects
- image features from the NSD stimuli to train a stimulus-to-brain encoder

## Big Picture

There are three different data streams:

1. task fMRI
- used as the target the model tries to predict
- also used to align subjects through shared stimuli

2. resting-state fMRI
- used to estimate subject-specific connectivity structure
- helps define a common space across subjects

3. image features
- CLIP, DINOv2, or CLIP+DINOv2
- used as the input representation of each NSD stimulus image

The model then learns:
- image features -> shared latent brain space
- shared latent brain space -> subject-specific voxel predictions

## Default Experimental Setup

The default split in [config.yaml](/home/rothermm/resting_prediction/config.yaml:4) is:
- train on subjects `1-6`
- test on subject `7`

The main feature backbones in [config.yaml](/home/rothermm/resting_prediction/config.yaml:12) are:
- `clip`
- `dinov2`
- `clip_dinov2`

The default alignment settings in [config.yaml](/home/rothermm/resting_prediction/config.yaml:36) are:
- `connectivity_mode: parcellation`
- `experiment_mode: hybrid_cha`
- `atlas_type: combined_rois`

## Stage 1: Prepare Task Data

Script:
- [src/data/prepare_task_data.py](/home/rothermm/resting_prediction/src/data/prepare_task_data.py)

Job wrapper:
- [slurm_scripts/01_prepare_task_data_array.sh](/home/rothermm/resting_prediction/slurm_scripts/01_prepare_task_data_array.sh)

What happens:
- load subject-level task beta volumes
- apply the `nsdgeneral` mask
- form train and test matrices per subject
- save the stimulus indices for each row
- keep trial-level test data for evaluation and noise-ceiling estimates

Main outputs per subject:
- `train_fmri.npy`
- `test_fmri.npy`
- `train_stim_idx.npy`
- `test_stim_idx.npy`
- `test_fmri_trials.npy`
- `test_trial_labels.npy`
- `mask.npy`

Interpretation:
- each row is a stimulus presentation or stimulus-average response
- each column is a voxel within that subject's mask

## Stage 2: Prepare REST Data

Script:
- [src/data/prepare_rest_data.py](/home/rothermm/resting_prediction/src/data/prepare_rest_data.py)

Job wrapper:
- [slurm_scripts/02_prepare_rest_data_array.sh](/home/rothermm/resting_prediction/slurm_scripts/02_prepare_rest_data_array.sh)

What happens:
- load REST runs for each subject
- discard initial TRs
- detrend and high-pass filter
- optionally censor motion-heavy time points
- z-score the cleaned time series
- save each usable run as a 2D array

Main outputs per subject:
- `rest_run1.npy`, `rest_run2.npy`, ...
- `rest_run_manifest.json`

Interpretation:
- each REST array is `timepoints x voxels`
- these runs are later used to estimate connectivity structure for alignment

## Stage 3: Extract Stimulus Features

Script:
- [src/data/prepare_features.py](/home/rothermm/resting_prediction/src/data/prepare_features.py)

Job wrapper:
- [slurm_scripts/03_extract_features_job.sh](/home/rothermm/resting_prediction/slurm_scripts/03_extract_features_job.sh)

What happens:
- load all NSD stimulus images
- pass them through one or more pretrained vision models
- save one feature vector per image

Supported backbones:
- `clip`
- `dinov2`

Combined backbone:
- `clip_dinov2`
  - this is not stored separately on disk
  - it is built by concatenating CLIP and DINOv2 features at load time in [src/data/nsd_loader.py](/home/rothermm/resting_prediction/src/data/nsd_loader.py:111)

Main outputs:
- `processed_data/features/clip_features.npy`
- `processed_data/features/dinov2_features.npy`

Why this matters:
- these image features are the model inputs
- the project is not predicting voxels directly from pixels
- it predicts from pretrained visual representations of the stimuli

## Stage 4: Build The Shared Space

Script:
- [src/pipelines/train_shared_space.py](/home/rothermm/resting_prediction/src/pipelines/train_shared_space.py)

Core alignment class:
- [src/alignment/shared_space.py](/home/rothermm/resting_prediction/src/alignment/shared_space.py)

What happens:
- load all training subjects' task and REST data
- load the ROI/parcellation atlas
- harmonize atlas labels across subjects
- find the intersection of shared test stimuli across training subjects
- combine REST connectivity structure with shared task responses
- fit subject-specific projections and rotations into a common latent space

Important idea:
- the shared space is built from brain data, not from image features
- image features are only used after the shared space exists

Why REST is useful:
- subjects do not share identical voxel geometry
- REST provides subject-specific connectivity structure
- that structure helps create a more comparable space across subjects

Why shared task stimuli are useful:
- they give a common anchor across subjects
- the project uses them to stabilize alignment in `hybrid_cha` mode

## Stage 5: Train The Encoder

This happens inside the same training pipeline as Stage 4.

Key step in code:
- [src/pipelines/train_shared_space.py](/home/rothermm/resting_prediction/src/pipelines/train_shared_space.py:377)

What happens:
- for each training subject, get the image features for that subject's training stimuli
- project that subject's task fMRI into the shared latent space
- pool all subjects together
- fit a ridge-regression encoder from image features `X` to shared latent responses `Z`

Conceptually:
- input: image features for a stimulus
- target: where that stimulus lands in the common brain space

Outputs:
- `builder.npz`
- `encoder.npz`
- `shared_stim_idx.npy`
- `metadata.json`

## Stage 6: Predict A Held-Out Subject

Script:
- [src/pipelines/predict_subject.py](/home/rothermm/resting_prediction/src/pipelines/predict_subject.py)

Job wrapper:
- [slurm_scripts/05_predict_and_ablate_job.sh](/home/rothermm/resting_prediction/slurm_scripts/05_predict_and_ablate_job.sh)

Two modes:
- zero-shot
- few-shot

### Zero-shot

What it means:
- predict the held-out subject without using held-out-subject training examples from the evaluation set

What happens:
- load the trained shared-space builder and encoder
- load the held-out subject's image features
- map image features through the encoder
- adapt the prediction back into the held-out subject's voxel space
- compare predicted voxel responses with true task fMRI

### Few-shot

What it means:
- allow a small number of shared stimuli from the held-out subject
- use those examples to adapt or refine prediction

Why it exists:
- zero-shot tests pure cross-subject transfer
- few-shot tests how quickly performance improves with small subject-specific supervision

## Stage 7: Run Ablations

Script:
- [src/pipelines/run_ablations.py](/home/rothermm/resting_prediction/src/pipelines/run_ablations.py)

What happens:
- repeat few-shot prediction for different numbers of examples
- optionally compare multiple feature backbones
- summarize uncertainty with bootstrap and permutation statistics

Questions this stage answers:
- how much does performance improve as shots increase?
- does `clip`, `dinov2`, or `clip_dinov2` work better?
- are gains statistically reliable?

## Stage 8: Visualize Predictions

Script:
- [src/pipelines/visualize_prediction_maps.py](/home/rothermm/resting_prediction/src/pipelines/visualize_prediction_maps.py)

What happens:
- select representative examples
- render side-by-side maps or figures for qualitative inspection

Purpose:
- fast human sanity check
- useful for presentations and debugging

## Stage 9: Reconstruction Benchmark

Main scripts:
- [src/data/prepare_reconstruction_features.py](/home/rothermm/resting_prediction/src/data/prepare_reconstruction_features.py)
- [src/pipelines/benchmark_reconstructions_vdvae_vd.py](/home/rothermm/resting_prediction/src/pipelines/benchmark_reconstructions_vdvae_vd.py)

Job wrappers:
- [slurm_scripts/03b_extract_recon_features_job.sh](/home/rothermm/resting_prediction/slurm_scripts/03b_extract_recon_features_job.sh)
- [slurm_scripts/08_benchmark_recon_vdvae_vd_job.sh](/home/rothermm/resting_prediction/slurm_scripts/08_benchmark_recon_vdvae_vd_job.sh)

What happens:
- build local VDVAE, CLIP-text, and CLIP-vision feature bundles for the test subject
- map predicted fMRI into reconstruction-model feature spaces
- reconstruct images using the brain-diffuser-style stack

Purpose:
- evaluate predictions in a more perceptual space
- compare GT, zero-shot, and few-shot conditions through reconstructed images

This is downstream of the main prediction pipeline. It is not required to train or evaluate voxelwise prediction metrics.

## Where DINOv2 Fits

DINOv2 is one of the image-feature backbones, not the alignment method itself.

It enters the pipeline here:
1. feature extraction writes `dinov2_features.npy`
2. training loads those features for the relevant stimuli
3. the encoder learns `dinov2 features -> shared brain space`
4. prediction uses DINOv2 features for the held-out subject's test images

So:
- REST/task fMRI define the brain alignment
- DINOv2 defines one possible visual representation of the stimuli

## What The Main Output Means

The central output is not a single image or single score. It is a set of trained artifacts and evaluation files.

Main outputs:
- `outputs/shared_space/`: trained shared-space model
- `outputs/predictions/`: held-out subject predictions and metrics
- `outputs/ablations/`: few-shot sweep results and statistics

Core scientific question:
- can resting-state structure plus cross-subject alignment support prediction of task responses in a new subject?

## Mental Model For A New Reader

If you want the shortest useful mental model, use this:

1. preprocess task fMRI into subject matrices
2. preprocess REST fMRI into clean runs
3. convert every image into CLIP or DINOv2 features
4. build a shared brain space across training subjects
5. learn a mapping from image features into that shared space
6. transfer that mapping to a held-out subject
7. score how well predicted voxels match real task fMRI

## Related Files

- practical run guide: [RUN_PIPELINE.md](/home/rothermm/resting_prediction/RUN_PIPELINE.md)
- configuration: [config.yaml](/home/rothermm/resting_prediction/config.yaml)
- SLURM job overview: [slurm_scripts/README.md](/home/rothermm/resting_prediction/slurm_scripts/README.md)
