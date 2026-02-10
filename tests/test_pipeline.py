"""
Integration test: full pipeline with synthetic data.
"""

import numpy as np
import pytest

from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.shared_space import SharedSpaceBuilder
from src.alignment.utils import compute_svd_basis, procrustes_align
from src.evaluation.metrics import voxelwise_correlation
from src.models.encoding import SharedSpaceEncoder


class TestFullPipelineSynthetic:
    """End-to-end test with synthetic subjects sharing a common structure."""

    def test_shared_space_alignment(self, three_subjects):
        """
        Verify that shared space alignment recovers cross-subject structure.
        After alignment, subjects' component-space responses should be similar.
        """
        subjects, k, R, N_shared, F = three_subjects

        # Step 1: Compute connectivity and bases
        bases = {}
        connectivity = {}
        for sub_id, data in subjects.items():
            C = compute_rest_connectivity(
                data["rest_runs"],
                mode="parcellation",
                atlas_masked=data["atlas_masked"],
                n_parcels=R,
            )
            connectivity[sub_id] = C
            P = compute_svd_basis(C, n_components=k, min_k=5)
            bases[sub_id] = P

        # Step 2: Project shared responses
        Z_all = {}
        for sub_id, data in subjects.items():
            Z = data["Y_shared"] @ bases[sub_id]
            Z_all[sub_id] = Z

        # Step 3: Procrustes alignment
        template = Z_all[1].copy()
        rotations = {}
        for sub_id in subjects:
            R_rot = procrustes_align(Z_all[sub_id], template)
            rotations[sub_id] = R_rot

        # After alignment, rotated Zs should be more similar to template
        for sub_id in [2, 3]:
            Z_aligned = Z_all[sub_id] @ rotations[sub_id]
            corr = np.corrcoef(Z_aligned.ravel(), template.ravel())[0, 1]
            # Should have positive correlation after alignment
            assert corr > 0, f"Subject {sub_id}: aligned correlation {corr:.3f} should be > 0"

    def test_encoder_prediction(self, three_subjects):
        """
        Verify encoder can learn and predict in shared space.
        """
        subjects, k, R, N_shared, F = three_subjects

        # Build training data
        X_all, Z_all = [], []
        for sub_id, data in subjects.items():
            X_all.append(data["X_shared"])
            # Simple component projection (no alignment for this test)
            C = compute_rest_connectivity(
                data["rest_runs"],
                mode="parcellation",
                atlas_masked=data["atlas_masked"],
                n_parcels=R,
            )
            P = compute_svd_basis(C, n_components=k, min_k=5)
            Z = data["Y_shared"] @ P
            Z_all.append(Z)

        X = np.concatenate(X_all, axis=0)
        Z = np.concatenate(Z_all, axis=0)

        # Fit encoder
        enc = SharedSpaceEncoder(alpha=1.0)
        enc.fit(X, Z)

        # Predict
        Z_pred = enc.predict(X)
        assert Z_pred.shape == Z.shape

        # Should have some correlation with actual Z (model learned something)
        corr = np.corrcoef(Z_pred.ravel(), Z.ravel())[0, 1]
        assert corr > 0.1, f"Encoder correlation {corr:.3f} too low"

    def test_shared_space_builder_roundtrip(self, three_subjects, tmp_path):
        """Test save/load of SharedSpaceBuilder."""
        subjects, k, R, N_shared, F = three_subjects

        rest_runs = {sid: data["rest_runs"] for sid, data in subjects.items()}
        task_shared = {sid: data["Y_shared"] for sid, data in subjects.items()}
        atlas = {sid: data["atlas_masked"] for sid, data in subjects.items()}

        builder = SharedSpaceBuilder(
            n_components=k,
            min_k=5,
            connectivity_mode="parcellation",
            experiment_mode="hybrid_cha",
        )
        builder.fit(rest_runs, task_shared, atlas, R)

        # Save and reload
        save_dir = str(tmp_path / "model")
        builder.save(save_dir)

        loaded = SharedSpaceBuilder.load(save_dir)
        assert loaded.k_global == builder.k_global
        assert set(loaded.subject_bases.keys()) == set(builder.subject_bases.keys())

        for sub_id in builder.subject_bases:
            np.testing.assert_allclose(
                loaded.subject_bases[sub_id],
                builder.subject_bases[sub_id],
                atol=1e-5,
            )
