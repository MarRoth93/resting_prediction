"""Focused contracts for interchangeable alignment experts."""

import numpy as np
import pytest

from src.alignment.experts import HybridCHAExpert, SubjectTransform


def _orthonormal_basis(rng: np.random.RandomState, n_rows: int, n_cols: int) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.randn(n_rows, n_cols))
    return basis.astype(np.float32)


class TestSubjectTransform:
    def test_project_and_reconstruct_contract(self, rng):
        basis = _orthonormal_basis(rng, n_rows=13, n_cols=4)
        transform = SubjectTransform(
            basis=basis,
            expert_name="test_expert",
            subject_id=7,
        )
        responses = rng.randn(9, 13).astype(np.float32)

        latents = transform.project(responses)
        reconstructed = transform.reconstruct(latents)

        assert latents.shape == (9, 4)
        assert reconstructed.shape == (9, 13)
        np.testing.assert_allclose(latents, responses @ basis, atol=1e-6)
        np.testing.assert_allclose(reconstructed, responses @ basis @ basis.T, atol=1e-6)

    def test_rejects_invalid_shapes_and_nonorthogonal_basis(self, rng):
        with pytest.raises(ValueError, match="orthonormal"):
            SubjectTransform(
                basis=np.ones((8, 3), dtype=np.float32),
                expert_name="bad",
            )

        transform = SubjectTransform(
            basis=_orthonormal_basis(rng, 8, 3),
            expert_name="good",
        )
        with pytest.raises(ValueError, match="voxel count"):
            transform.project(np.zeros((2, 7), dtype=np.float32))
        with pytest.raises(ValueError, match="component count"):
            transform.reconstruct(np.zeros((2, 2), dtype=np.float32))

    def test_save_load_roundtrip_and_compatibility_checks(self, tmp_path, rng):
        transform = SubjectTransform(
            basis=_orthonormal_basis(rng, 11, 3),
            expert_name="test_expert",
            subject_id=4,
            metadata={"alignment": "zero_shot", "n_runs": np.int64(2)},
        )
        path = tmp_path / "transform.npz"
        transform.save(path)

        loaded = SubjectTransform.load(
            path,
            expected_expert_name="test_expert",
            expected_n_components=3,
            expected_n_voxels=11,
        )
        np.testing.assert_array_equal(loaded.basis, transform.basis)
        assert loaded.subject_id == 4
        assert loaded.metadata == {"alignment": "zero_shot", "n_runs": 2}
        assert loaded.fingerprint == transform.fingerprint

        with pytest.raises(ValueError, match="component count mismatch"):
            SubjectTransform.load(path, expected_n_components=2)
        with pytest.raises(ValueError, match="expert name mismatch"):
            SubjectTransform.load(path, expected_expert_name="another_expert")


def test_hybrid_wrapper_composes_basis_and_rotation(rng):
    expert = HybridCHAExpert(n_components=4, min_k=2)
    P = _orthonormal_basis(rng, 12, 4)
    R = _orthonormal_basis(rng, 4, 4)
    expert.builder.k_global = 4
    expert.builder.subject_bases[1] = P
    expert.builder.subject_rotations[1] = R

    transform = expert.training_transform(1)

    assert transform.expert_name == "hybrid_cha"
    np.testing.assert_allclose(transform.basis, P @ R, atol=1e-6)
    np.testing.assert_allclose(transform.basis.T @ transform.basis, np.eye(4), atol=1e-5)


def test_hybrid_save_load_checks_inference_templates(tmp_path, rng):
    rest_runs = {}
    seed_runs = {}
    task = {}
    for subject, voxels in [(1, 7), (2, 9)]:
        rest_runs[subject] = []
        seed_runs[subject] = []
        mixing = rng.randn(5, voxels).astype(np.float32)
        for _ in range(2):
            seeds = rng.randn(35, 5).astype(np.float32)
            seed_runs[subject].append(seeds)
            rest_runs[subject].append(
                (seeds @ mixing + 0.1 * rng.randn(35, voxels)).astype(np.float32)
            )
        task[subject] = rng.randn(12, voxels).astype(np.float32)
    expert = HybridCHAExpert(
        n_components=3,
        min_k=2,
        ensemble_method="concat",
        max_iters=3,
    ).fit(rest_runs, task, seed_runs)
    expert.save(tmp_path)
    HybridCHAExpert.load(tmp_path, expected_n_components=3)

    builder_path = tmp_path / "builder.npz"
    with np.load(builder_path, allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    values["template_Z"] = values["template_Z"].copy()
    values["template_Z"][0, 0] += 0.5
    np.savez(builder_path, **values)
    with pytest.raises(ValueError, match="task-template checksum"):
        HybridCHAExpert.load(tmp_path, expected_n_components=3)


def test_hybrid_zero_shot_masks_zero_filled_missing_seed_rows(rng):
    rest_runs = {}
    seed_runs = {}
    task = {}
    for subject, voxels in [(1, 7), (2, 8)]:
        seeds = rng.randn(40, 5).astype(np.float32)
        mixing = rng.randn(5, voxels).astype(np.float32)
        seed_runs[subject] = [seeds]
        rest_runs[subject] = [
            (seeds @ mixing + 0.05 * rng.randn(40, voxels)).astype(np.float32)
        ]
        task[subject] = rng.randn(12, voxels).astype(np.float32)
    expert = HybridCHAExpert(
        n_components=2,
        min_k=2,
        ensemble_method="concat",
        max_iters=3,
        seed_manifest_fingerprint="seed-manifest",
    ).fit(rest_runs, task, seed_runs)

    new_seeds = rng.randn(40, 5).astype(np.float32)
    new_seeds[:, -1] = 0.0
    new_rest = (
        new_seeds @ rng.randn(5, 9) + 0.05 * rng.randn(40, 9)
    ).astype(np.float32)
    transform = expert.align_new_subject_zeroshot(
        [new_rest],
        [new_seeds],
        subject_id=9,
        seed_manifest_fingerprint="seed-manifest",
    )

    assert transform.metadata["available_seed_rows"] == 4
    assert transform.metadata["total_seed_rows"] == 5
    np.testing.assert_allclose(transform.basis.T @ transform.basis, np.eye(2), atol=1e-5)
