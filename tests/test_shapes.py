"""
Unit tests: matrix shape contracts through the pipeline.
"""

import numpy as np
import pytest

from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.utils import compute_svd_basis, procrustes_align
from src.evaluation.metrics import (
    noise_ceiling_split_half,
    pattern_correlation,
    voxelwise_correlation,
)
from src.models.encoding import SharedSpaceEncoder


class TestSVDBasis:
    def test_parcellation_shape(self):
        """P has shape (V, k_actual) with k_actual <= min(R, n_components)."""
        V, R, k = 1000, 25, 50
        C = np.random.randn(R, V).astype(np.float32)
        P = compute_svd_basis(C, n_components=k, min_k=5)
        assert P.shape[0] == V
        assert P.shape[1] <= min(k, R - 1)  # rank-limited
        assert P.shape[1] == 24  # R=25 → max rank 24

    def test_voxel_correlation_shape(self):
        """P from V×V has k_actual = n_components (no rank limit for large V)."""
        V, k = 200, 20
        C = np.random.randn(V, V).astype(np.float32)
        C = (C + C.T) / 2  # symmetric
        P = compute_svd_basis(C, n_components=k, min_k=5)
        assert P.shape == (V, k)

    def test_min_k_raises(self):
        """Fail-fast when k_actual < min_k."""
        V, R = 1000, 5
        C = np.random.randn(R, V).astype(np.float32)
        with pytest.raises(ValueError, match="min_k"):
            compute_svd_basis(C, n_components=50, min_k=10)

    def test_basis_orthogonal_columns(self):
        """Columns of P should be approximately orthonormal."""
        V, R = 1000, 30
        C = np.random.randn(R, V).astype(np.float32)
        P = compute_svd_basis(C, n_components=20, min_k=5)
        # P.T @ P should be close to identity
        PtP = P.T @ P
        np.testing.assert_allclose(PtP, np.eye(P.shape[1]), atol=1e-4)


class TestProcrustes:
    def test_rotation_orthogonal(self):
        """R @ R.T ≈ I."""
        k = 20
        source = np.random.randn(100, k).astype(np.float32)
        target = np.random.randn(100, k).astype(np.float32)
        R = procrustes_align(source, target)
        assert R.shape == (k, k)
        np.testing.assert_allclose(R @ R.T, np.eye(k), atol=1e-4)

    def test_nan_raises(self):
        """NaN input should raise."""
        source = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
        target = np.random.randn(2, 2).astype(np.float32)
        with pytest.raises(ValueError, match="NaN"):
            procrustes_align(source, target)

    def test_constant_column_raises(self):
        """Near-constant column should raise."""
        source = np.ones((100, 2), dtype=np.float32)
        target = np.random.randn(100, 2).astype(np.float32)
        with pytest.raises(ValueError, match="constant"):
            procrustes_align(source, target)


class TestConnectivity:
    def test_parcellation_output_shape(self, synthetic_subject):
        """Parcellation connectivity is (R, V)."""
        d = synthetic_subject
        C = compute_rest_connectivity(
            d["rest_runs"],
            mode="parcellation",
            atlas_masked=d["atlas_masked"],
            n_parcels=d["R"],
        )
        assert C.shape == (d["R"], d["V"])

    def test_voxel_correlation_output_shape(self, synthetic_subject):
        """Voxel connectivity is (V, V)."""
        d = synthetic_subject
        C = compute_rest_connectivity(d["rest_runs"], mode="voxel_correlation")
        assert C.shape == (d["V"], d["V"])


class TestEncoder:
    def test_fit_predict_shapes(self):
        """Encoder input/output shapes."""
        N, F, k = 100, 64, 20
        X = np.random.randn(N, F).astype(np.float32)
        Z = np.random.randn(N, k).astype(np.float32)

        enc = SharedSpaceEncoder(alpha=1.0)
        enc.fit(X, Z)

        assert enc.W.shape == (F, k)
        assert enc.b.shape == (k,)

        Z_pred = enc.predict(X)
        assert Z_pred.shape == (N, k)

    def test_predict_voxels_shape(self):
        """predict_voxels returns (N, V)."""
        N, F, k, V = 50, 64, 20, 500
        X = np.random.randn(N, F).astype(np.float32)
        Z = np.random.randn(N, k).astype(np.float32)
        P = np.random.randn(V, k).astype(np.float32)
        R = np.eye(k, dtype=np.float32)

        enc = SharedSpaceEncoder(alpha=1.0)
        enc.fit(X, Z)

        Y_pred = enc.predict_voxels(X, P, R)
        assert Y_pred.shape == (N, V)


class TestMetrics:
    def test_voxelwise_correlation_shape(self):
        """Returns (V,)."""
        N, V = 50, 100
        Y_true = np.random.randn(N, V).astype(np.float32)
        Y_pred = np.random.randn(N, V).astype(np.float32)
        corrs = voxelwise_correlation(Y_true, Y_pred)
        assert corrs.shape == (V,)
        assert np.all(np.isfinite(corrs))

    def test_voxelwise_perfect_correlation(self):
        """Identical inputs → r = 1."""
        N, V = 50, 10
        Y = np.random.randn(N, V).astype(np.float32)
        corrs = voxelwise_correlation(Y, Y)
        np.testing.assert_allclose(corrs, 1.0, atol=1e-5)

    def test_zero_variance_voxel(self):
        """Zero-variance voxels should return 0, not NaN."""
        N, V = 50, 10
        Y_true = np.random.randn(N, V).astype(np.float32)
        Y_pred = np.random.randn(N, V).astype(np.float32)
        Y_true[:, 0] = 5.0  # constant
        corrs = voxelwise_correlation(Y_true, Y_pred)
        assert corrs[0] == 0.0
        assert np.all(np.isfinite(corrs))

    def test_pattern_correlation_shape(self):
        """Returns (N,)."""
        N, V = 50, 100
        Y_true = np.random.randn(N, V).astype(np.float32)
        Y_pred = np.random.randn(N, V).astype(np.float32)
        corrs = pattern_correlation(Y_true, Y_pred)
        assert corrs.shape == (N,)

    def test_noise_ceiling_shape(self):
        """Returns (V,) with values in [0, 1]."""
        V = 100
        N_trials = 150
        N_stim = 50
        trial_fmri = np.random.randn(N_trials, V).astype(np.float32)
        trial_labels = np.repeat(np.arange(N_stim), 3)  # 3 reps each
        nc = noise_ceiling_split_half(trial_fmri, trial_labels)
        assert nc.shape == (V,)
        assert np.all(nc >= 0)
        assert np.all(nc <= 1)
