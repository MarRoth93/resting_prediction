"""
Pytest fixtures: synthetic data generators for unit and integration tests.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def synthetic_subject(rng):
    """Create synthetic subject data with known properties."""
    V = 500  # voxels
    T = 200  # timepoints per REST run
    N_train = 100  # training stimuli
    N_test = 50  # test stimuli
    k = 20  # components
    R = 30  # parcels

    # Generate a "true" basis
    P_true = np.linalg.qr(rng.randn(V, k))[0][:, :k].astype(np.float32)

    # REST runs (2 runs)
    rest_runs = [rng.randn(T, V).astype(np.float32) for _ in range(2)]

    # Task betas
    train_fmri = rng.randn(N_train, V).astype(np.float32)
    test_fmri = rng.randn(N_test, V).astype(np.float32)

    # Atlas
    atlas_masked = np.zeros(V, dtype=np.int32)
    parcel_size = V // R
    for p in range(R):
        start = p * parcel_size
        end = start + parcel_size if p < R - 1 else V
        atlas_masked[start:end] = p + 1

    return {
        "V": V,
        "k": k,
        "R": R,
        "P_true": P_true,
        "rest_runs": rest_runs,
        "train_fmri": train_fmri,
        "test_fmri": test_fmri,
        "atlas_masked": atlas_masked,
        "train_stim_idx": np.arange(N_train),
        "test_stim_idx": np.arange(N_train, N_train + N_test),
    }


@pytest.fixture
def three_subjects(rng):
    """Create 3 synthetic subjects with different voxel counts but shared structure."""
    k = 15
    R = 25
    N_shared = 50
    F = 64  # feature dim

    # Shared latent structure
    W_true = rng.randn(F, k).astype(np.float32)  # true encoding weights

    subjects = {}
    for sub_id, V in zip([1, 2, 3], [500, 450, 400]):
        # Per-subject basis (orthogonal)
        P = np.linalg.qr(rng.randn(V, k))[0][:, :k].astype(np.float32)
        R_rot = np.linalg.qr(rng.randn(k, k))[0].astype(np.float32)

        # Atlas
        atlas = np.zeros(V, dtype=np.int32)
        ps = V // R
        for p in range(R):
            start = p * ps
            end = start + ps if p < R - 1 else V
            atlas[start:end] = p + 1

        # REST data
        rest_runs = [rng.randn(100, V).astype(np.float32) for _ in range(2)]

        # Shared stimuli features and responses
        X_shared = rng.randn(N_shared, F).astype(np.float32)
        Z_shared = X_shared @ W_true @ R_rot  # (N_shared, k) in subject's component space
        Y_shared = Z_shared @ P.T  # (N_shared, V) voxel responses

        subjects[sub_id] = {
            "V": V,
            "P": P,
            "R_rot": R_rot,
            "atlas_masked": atlas,
            "rest_runs": rest_runs,
            "Y_shared": Y_shared,
            "X_shared": X_shared,
        }

    return subjects, k, R, N_shared, F
