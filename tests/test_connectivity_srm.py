"""Synthetic recovery and persistence tests for connectivity-SRM."""

import numpy as np
import pytest

from src.alignment.connectivity_srm import ConnectivitySRMExpert


def _synthetic_connectivity(
    seed: int = 42,
    n_seeds: int = 36,
    k: int = 5,
    noise: float = 0.01,
) -> dict[int, np.ndarray]:
    rng = np.random.RandomState(seed)
    shared = rng.randn(n_seeds, k)
    connectivity = {}
    for subject_id, n_voxels in [(1, 19), (2, 23), (3, 27)]:
        weights, _ = np.linalg.qr(rng.randn(n_voxels, k))
        C = shared @ weights.T + noise * rng.randn(n_seeds, n_voxels)
        connectivity[subject_id] = C.astype(np.float32)
    return connectivity


def _task_responses(
    connectivity: dict[int, np.ndarray],
    n_rows: int = 20,
) -> dict[int, np.ndarray]:
    rng = np.random.RandomState(123)
    return {
        subject_id: rng.randn(n_rows, C.shape[1]).astype(np.float32)
        for subject_id, C in connectivity.items()
    }


def test_synthetic_recovery_convergence_and_orthogonality():
    connectivity = _synthetic_connectivity()
    expert = ConnectivitySRMExpert(
        n_components=5,
        min_k=2,
        max_iters=20,
        tol=1e-7,
    ).fit_connectivity(connectivity)

    assert expert.k_global == 5
    assert expert.converged
    assert 1 <= len(expert.delta_history) <= 20
    assert len(expert.objective_history) == len(expert.delta_history) + 1
    assert expert.objective_history[-1] <= expert.objective_history[0] + 1e-12
    assert np.all(np.diff(expert.objective_history) <= 1e-10)

    for subject_id, C in connectivity.items():
        W = expert.subject_weights[subject_id]
        np.testing.assert_allclose(W.T @ W, np.eye(5), atol=1e-5)
        relative_error = np.linalg.norm(C - expert.shared_response @ W.T) / np.linalg.norm(C)
        assert relative_error < 0.04


def test_fit_is_deterministic():
    connectivity = _synthetic_connectivity()
    kwargs = dict(n_components=5, min_k=2, max_iters=20, tol=1e-7)
    first = ConnectivitySRMExpert(**kwargs).fit_connectivity(connectivity)
    second = ConnectivitySRMExpert(**kwargs).fit_connectivity(connectivity)

    np.testing.assert_array_equal(first.shared_response, second.shared_response)
    for subject_id in connectivity:
        np.testing.assert_array_equal(
            first.subject_weights[subject_id],
            second.subject_weights[subject_id],
        )


def test_task_template_does_not_change_rest_fit():
    connectivity = _synthetic_connectivity()
    task = _task_responses(connectivity)
    expert = ConnectivitySRMExpert(
        n_components=5,
        min_k=2,
        max_iters=20,
        tol=1e-7,
    ).fit_connectivity(connectivity)
    shared_before = expert.shared_response.copy()
    weights_before = {key: value.copy() for key, value in expert.subject_weights.items()}

    expert._build_task_template(task)

    np.testing.assert_array_equal(expert.shared_response, shared_before)
    for subject_id in connectivity:
        np.testing.assert_array_equal(expert.subject_weights[subject_id], weights_before[subject_id])
    assert expert.task_template.shape == (20, 5)


def test_new_subject_basis_is_orthonormal():
    connectivity = _synthetic_connectivity()
    expert = ConnectivitySRMExpert(
        n_components=5,
        min_k=2,
    ).fit_connectivity(connectivity)
    new_connectivity = _synthetic_connectivity(seed=99)[1]

    basis = expert._basis_for_connectivity(new_connectivity)

    assert basis.shape == (new_connectivity.shape[1], 5)
    np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-5)


def test_save_load_roundtrip_and_manifest_compatibility(tmp_path):
    connectivity = _synthetic_connectivity()
    expert = ConnectivitySRMExpert(
        n_components=5,
        min_k=2,
        max_iters=20,
        tol=1e-7,
        seed_manifest_fingerprint="seed-order-v1",
    ).fit_connectivity(connectivity)
    expert._build_task_template(_task_responses(connectivity))
    expert.save(tmp_path)

    loaded = ConnectivitySRMExpert.load(
        tmp_path,
        expected_n_components=5,
        expected_seed_manifest_fingerprint="seed-order-v1",
    )

    assert loaded.converged == expert.converged
    assert loaded.objective_history == expert.objective_history
    assert loaded.delta_history == expert.delta_history
    np.testing.assert_array_equal(loaded.shared_response, expert.shared_response)
    np.testing.assert_array_equal(loaded.task_template, expert.task_template)
    for subject_id in connectivity:
        np.testing.assert_array_equal(
            loaded.training_transform(subject_id).basis,
            expert.training_transform(subject_id).basis,
        )

    with pytest.raises(ValueError, match="component count mismatch"):
        ConnectivitySRMExpert.load(tmp_path, expected_n_components=4)
    with pytest.raises(ValueError, match="Seed manifest fingerprint mismatch"):
        ConnectivitySRMExpert.load(
            tmp_path,
            expected_seed_manifest_fingerprint="different-seed-order",
        )


def test_rejects_inconsistent_seed_rows():
    connectivity = _synthetic_connectivity()
    connectivity[2] = connectivity[2][:-1]
    with pytest.raises(ValueError, match="same ordered seed rows"):
        ConnectivitySRMExpert(n_components=5, min_k=2).fit_connectivity(connectivity)
