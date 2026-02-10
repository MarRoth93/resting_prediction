"""
Prepare averaged task fMRI data for one subject.

Refactored from prepare_nsddata.py with:
- Dynamic session discovery (no hardcoded 37)
- Canonical stimulus ordering by NSD image ID
- Trial-level data saved for test/shared set (noise ceiling)
- float32 throughout
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import scipy.io as spio

logger = logging.getLogger(__name__)


def loadmat(filename: str) -> dict:
    """Load .mat file handling nested structs properly."""
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def discover_sessions(betas_dir: str) -> list[int]:
    """Discover available beta sessions dynamically via glob."""
    pattern = os.path.join(betas_dir, "betas_session*.nii.gz")
    files = sorted(glob.glob(pattern))
    sessions = []
    for f in files:
        basename = os.path.basename(f)
        # Extract session number from betas_sessionNN.nii.gz
        num_str = basename.replace("betas_session", "").replace(".nii.gz", "")
        sessions.append(int(num_str))
    if not sessions:
        raise FileNotFoundError(f"No beta session files found in {betas_dir}")
    logger.info(f"Discovered {len(sessions)} sessions in {betas_dir}")
    return sorted(sessions)


def prepare_task_data(
    sub: int,
    data_root: str = ".",
    output_root: str = "processed_data",
) -> dict:
    """
    Prepare averaged task fMRI for one subject.

    Returns dict with:
        train_fmri: (N_train, V_sub) float32 — averaged betas per stimulus
        test_fmri: (N_test, V_sub) float32 — averaged betas per stimulus
        train_stim_idx: (N_train,) int — NSD image indices, sorted
        test_stim_idx: (N_test,) int — NSD image indices, sorted
        test_fmri_trials: (N_test_trials, V_sub) float32 — trial-level
        test_trial_labels: (N_test_trials,) int — stimulus ID per trial
        mask: (X, Y, Z) bool — nsdgeneral mask
        num_voxels: int
    """
    # Paths
    stim_order_f = os.path.join(data_root, "nsddata/experiments/nsd/nsd_expdesign.mat")
    roi_dir = os.path.join(data_root, f"nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/")
    betas_dir = os.path.join(data_root, f"nsddata_betas/ppdata/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/")
    out_dir = os.path.join(output_root, f"subj{sub:02d}")
    os.makedirs(out_dir, exist_ok=True)

    # Load experiment design
    stim_order = loadmat(stim_order_f)
    masterordering = np.array(stim_order["masterordering"])
    subjectim = np.array(stim_order["subjectim"])

    # Load mask
    mask = nib.load(os.path.join(roi_dir, "nsdgeneral.nii.gz")).get_fdata() > 0
    num_voxels = int(mask.sum())
    logger.info(f"Subject {sub}: {num_voxels} voxels in nsdgeneral mask")

    # Discover sessions dynamically
    sessions = discover_sessions(betas_dir)
    num_sessions = len(sessions)
    trials_per_session = 750
    num_trials = num_sessions * trials_per_session

    # Separate train/test trial indices and map to NSD image IDs
    sig_train = {}  # nsdId -> [trial_indices]
    sig_test = {}
    for idx in range(num_trials):
        if idx >= len(masterordering):
            break
        nsd_id = int(subjectim[sub - 1, masterordering[idx] - 1] - 1)
        if masterordering[idx] > 1000:
            sig_train.setdefault(nsd_id, []).append(idx)
        else:
            sig_test.setdefault(nsd_id, []).append(idx)

    # Sort by NSD image ID for canonical ordering
    train_stim_idx = np.array(sorted(sig_train.keys()), dtype=np.int64)
    test_stim_idx = np.array(sorted(sig_test.keys()), dtype=np.int64)

    logger.info(f"Subject {sub}: {len(train_stim_idx)} train stimuli, {len(test_stim_idx)} test stimuli")

    # Load all betas
    fmri = np.zeros((num_trials, num_voxels), dtype=np.float32)
    for i, sess in enumerate(sessions):
        beta_file = os.path.join(betas_dir, f"betas_session{sess:02d}.nii.gz")
        beta_data = nib.load(beta_file).get_fdata().astype(np.float32)
        fmri[i * trials_per_session:(i + 1) * trials_per_session] = beta_data[mask].T
        del beta_data
        logger.info(f"  Loaded session {sess}/{num_sessions}")

    # Average train betas per stimulus (canonical order)
    train_fmri = np.zeros((len(train_stim_idx), num_voxels), dtype=np.float32)
    for i, nsd_id in enumerate(train_stim_idx):
        trial_indices = sorted(sig_train[nsd_id])
        train_fmri[i] = fmri[trial_indices].mean(axis=0)

    # Average test betas per stimulus (canonical order) + keep trial-level
    test_fmri = np.zeros((len(test_stim_idx), num_voxels), dtype=np.float32)
    test_trials_list = []
    test_labels_list = []
    for i, nsd_id in enumerate(test_stim_idx):
        trial_indices = sorted(sig_test[nsd_id])
        test_fmri[i] = fmri[trial_indices].mean(axis=0)
        # Keep trial-level for noise ceiling
        for tidx in trial_indices:
            test_trials_list.append(fmri[tidx])
            test_labels_list.append(i)  # index into test_stim_idx

    test_fmri_trials = np.array(test_trials_list, dtype=np.float32)
    test_trial_labels = np.array(test_labels_list, dtype=np.int64)

    # Save
    np.save(os.path.join(out_dir, "train_fmri.npy"), train_fmri)
    np.save(os.path.join(out_dir, "test_fmri.npy"), test_fmri)
    np.save(os.path.join(out_dir, "train_stim_idx.npy"), train_stim_idx)
    np.save(os.path.join(out_dir, "test_stim_idx.npy"), test_stim_idx)
    np.save(os.path.join(out_dir, "test_fmri_trials.npy"), test_fmri_trials)
    np.save(os.path.join(out_dir, "test_trial_labels.npy"), test_trial_labels)
    np.save(os.path.join(out_dir, "mask.npy"), mask)

    logger.info(f"Subject {sub}: saved to {out_dir}")
    logger.info(f"  train_fmri: {train_fmri.shape}, test_fmri: {test_fmri.shape}")
    logger.info(f"  test_fmri_trials: {test_fmri_trials.shape}")

    del fmri  # free memory

    return {
        "train_fmri": train_fmri,
        "test_fmri": test_fmri,
        "train_stim_idx": train_stim_idx,
        "test_stim_idx": test_stim_idx,
        "test_fmri_trials": test_fmri_trials,
        "test_trial_labels": test_trial_labels,
        "mask": mask,
        "num_voxels": num_voxels,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Prepare task data for one subject")
    parser.add_argument("-sub", "--sub", type=int, required=True, choices=[1, 2, 5, 7])
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-root", default="processed_data")
    args = parser.parse_args()

    prepare_task_data(args.sub, args.data_root, args.output_root)
