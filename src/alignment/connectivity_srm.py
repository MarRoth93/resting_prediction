"""Native connectivity Shared Response Model (cSRM) alignment expert.

This implementation factorizes each subject's fixed seed-to-voxel REST
connectivity matrix ``C_s`` as ``S @ W_s.T``.  The common response ``S`` and
orthonormal subject maps ``W_s`` are learned exclusively from REST.  Shared
task responses are retained only as an optional template for later few-shot
calibration of a new subject.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.alignment.experts import (
    EXPERT_SCHEMA_VERSION,
    AlignmentExpert,
    SubjectTransform,
    _array_sha256,
    _validate_expected,
    _validate_seed_manifest_fingerprint,
)
from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.utils import compute_global_k, compute_svd_basis, procrustes_align


logger = logging.getLogger(__name__)


def _orthogonal_factor(cross_covariance: np.ndarray) -> np.ndarray:
    """Return the rectangular orthogonal polar factor of ``cross_covariance``."""
    U, _, Vt = np.linalg.svd(cross_covariance, full_matrices=False)
    return U @ Vt


def _normalized_objective(
    connectivity: dict[int, np.ndarray],
    shared_response: np.ndarray,
    weights: dict[int, np.ndarray],
) -> float:
    """Compute the cSRM reconstruction objective without large temporaries."""
    residual = 0.0
    scale = 0.0
    shared_gram = shared_response.T @ shared_response
    for subject_id, C in connectivity.items():
        W = weights[subject_id]
        c_norm = float(np.sum(C * C))
        model_norm = float(np.trace(shared_gram @ (W.T @ W)))
        cross = float(np.trace(W.T @ C.T @ shared_response))
        residual += max(0.0, c_norm + model_norm - 2.0 * cross)
        scale += c_norm
    return residual / max(scale, np.finfo(np.float64).eps)


class ConnectivitySRMExpert(AlignmentExpert):
    """REST-only shared-response factorization in connectivity space."""

    expert_name = "connectivity_srm"

    def __init__(
        self,
        n_components: int = 50,
        min_k: int = 10,
        ensemble_method: str = "average",
        max_iters: int = 20,
        tol: float = 1e-5,
        seed_manifest_fingerprint: str | None = None,
    ) -> None:
        if int(n_components) < 1:
            raise ValueError("n_components must be positive")
        if int(min_k) < 1:
            raise ValueError("min_k must be positive")
        if int(max_iters) < 1:
            raise ValueError("max_iters must be positive")
        if float(tol) <= 0:
            raise ValueError("tol must be positive")
        self.n_components = int(n_components)
        self.min_k = int(min_k)
        self.ensemble_method = str(ensemble_method)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.seed_manifest_fingerprint = seed_manifest_fingerprint

        self.k_global: int | None = None
        self.n_seed_rows: int | None = None
        self.shared_response: np.ndarray | None = None
        self.task_template: np.ndarray | None = None
        self.subject_weights: dict[int, np.ndarray] = {}
        self.objective_history: list[float] = []
        self.delta_history: list[float] = []
        self.converged = False

    def fit(
        self,
        rest_runs: dict[int, list[np.ndarray]],
        task_responses_shared: dict[int, np.ndarray] | None,
        external_seed_runs: dict[int, list[np.ndarray]],
    ) -> "ConnectivitySRMExpert":
        subject_ids = sorted(int(v) for v in rest_runs)
        if not subject_ids:
            raise ValueError("rest_runs must contain at least one training subject")
        if sorted(int(v) for v in external_seed_runs) != subject_ids:
            raise ValueError("external_seed_runs must contain exactly the training subjects")

        connectivity: dict[int, np.ndarray] = {}
        for subject_id in subject_ids:
            connectivity[subject_id] = compute_rest_connectivity(
                rest_runs[subject_id],
                seed_runs=external_seed_runs[subject_id],
                ensemble=self.ensemble_method,
            )

        # This step is deliberately completed before task data is inspected.
        self.fit_connectivity(
            connectivity,
            task_responses_shared=task_responses_shared,
        )
        return self

    def fit_connectivity(
        self,
        connectivity: dict[int, np.ndarray],
        task_responses_shared: dict[int, np.ndarray] | None = None,
    ) -> "ConnectivitySRMExpert":
        """Fit from precomputed ``(shared seeds, subject voxels)`` matrices."""
        if not connectivity:
            raise ValueError("connectivity must contain at least one training subject")
        subject_ids = sorted(int(v) for v in connectivity)
        matrices: dict[int, np.ndarray] = {}
        seed_counts: set[int] = set()
        for subject_id in subject_ids:
            C = np.asarray(connectivity[subject_id], dtype=np.float64)
            if C.ndim != 2 or min(C.shape) < 2:
                raise ValueError(
                    f"Subject {subject_id}: connectivity must be a non-trivial 2D array, "
                    f"got {C.shape}"
                )
            if not np.all(np.isfinite(C)):
                raise ValueError(f"Subject {subject_id}: connectivity contains NaN/Inf")
            matrices[subject_id] = C
            seed_counts.add(int(C.shape[0]))
        if len(seed_counts) != 1:
            raise ValueError(
                "All connectivity matrices must use the same ordered seed rows; "
                f"got counts {sorted(seed_counts)}"
            )

        self.k_global = compute_global_k(
            matrices,
            n_components=self.n_components,
            min_k=self.min_k,
        )
        self.n_seed_rows = seed_counts.pop()
        k = self.k_global

        # Deterministic right-SVD initialization.  Fingerprints are oriented to
        # the first subject before averaging so arbitrary SVD rotations do not
        # cancel in the initial shared response.
        weights: dict[int, np.ndarray] = {}
        fingerprints: dict[int, np.ndarray] = {}
        for subject_id in subject_ids:
            W = compute_svd_basis(
                matrices[subject_id],
                n_components=k,
                min_k=self.min_k,
            )[:, :k].astype(np.float64)
            weights[subject_id] = W
            fingerprints[subject_id] = matrices[subject_id] @ W

        reference = fingerprints[subject_ids[0]]
        for subject_id in subject_ids[1:]:
            rotation = _orthogonal_factor(fingerprints[subject_id].T @ reference)
            weights[subject_id] = weights[subject_id] @ rotation

        shared = np.mean(
            [matrices[subject_id] @ weights[subject_id] for subject_id in subject_ids],
            axis=0,
        )
        objectives = [_normalized_objective(matrices, shared, weights)]
        deltas: list[float] = []
        converged = False

        for iteration in range(self.max_iters):
            previous = shared
            for subject_id in subject_ids:
                cross_covariance = matrices[subject_id].T @ previous
                weights[subject_id] = _orthogonal_factor(cross_covariance)
            shared = np.mean(
                [matrices[subject_id] @ weights[subject_id] for subject_id in subject_ids],
                axis=0,
            )
            delta = float(
                np.linalg.norm(shared - previous)
                / max(np.linalg.norm(previous), np.finfo(np.float64).eps)
            )
            objective = _normalized_objective(matrices, shared, weights)
            deltas.append(delta)
            objectives.append(objective)
            logger.info(
                "connectivity-SRM iteration %d: delta=%.6g objective=%.6g",
                iteration + 1,
                delta,
                objective,
            )
            if delta < self.tol:
                converged = True
                break

        self.shared_response = np.asarray(shared, dtype=np.float32)
        self.subject_weights = {
            subject_id: np.asarray(weights[subject_id], dtype=np.float32)
            for subject_id in subject_ids
        }
        self.objective_history = [float(v) for v in objectives]
        self.delta_history = [float(v) for v in deltas]
        self.converged = converged
        self.task_template = None
        # Task data is deliberately inspected only after the REST factorization
        # has been finalized.  This optional argument lets callers reuse a
        # connectivity matrix already computed for another expert.
        self._build_task_template(task_responses_shared)
        return self

    def _build_task_template(
        self,
        task_responses_shared: dict[int, np.ndarray] | None,
    ) -> None:
        if task_responses_shared is None:
            self.task_template = None
            return
        subject_ids = sorted(self.subject_weights)
        if sorted(int(v) for v in task_responses_shared) != subject_ids:
            raise ValueError(
                "task_responses_shared must contain exactly the fitted subjects"
            )
        projected: list[np.ndarray] = []
        n_rows: int | None = None
        for subject_id in subject_ids:
            Y = np.asarray(task_responses_shared[subject_id], dtype=np.float32)
            transform = self.training_transform(subject_id)
            if n_rows is None:
                n_rows = int(Y.shape[0])
            elif int(Y.shape[0]) != n_rows:
                raise ValueError("Shared task responses must have the same row count")
            projected.append(transform.project(Y))
        self.task_template = np.asarray(np.mean(projected, axis=0), dtype=np.float32)

    def _require_fitted(self) -> None:
        if self.k_global is None or self.n_seed_rows is None or self.shared_response is None:
            raise ValueError("connectivity_srm expert is not fitted")

    def training_transform(self, subject_id: int) -> SubjectTransform:
        subject_id = int(subject_id)
        if subject_id not in self.subject_weights:
            raise KeyError(f"No fitted connectivity_srm transform for subject {subject_id}")
        return SubjectTransform(
            basis=self.subject_weights[subject_id],
            expert_name=self.expert_name,
            subject_id=subject_id,
            metadata={"alignment": "training"},
        )

    def _new_subject_connectivity(
        self,
        rest_runs: list[np.ndarray],
        external_seed_runs: list[np.ndarray],
        seed_manifest_fingerprint: str | None,
    ) -> np.ndarray:
        self._require_fitted()
        _validate_seed_manifest_fingerprint(
            self.seed_manifest_fingerprint,
            seed_manifest_fingerprint,
        )
        C = compute_rest_connectivity(
            rest_runs,
            seed_runs=external_seed_runs,
            ensemble=self.ensemble_method,
        )
        if int(C.shape[0]) != self.n_seed_rows:
            raise ValueError(
                f"Seed count mismatch: artifact={self.n_seed_rows}, supplied={C.shape[0]}"
            )
        return C

    def _basis_for_connectivity(self, connectivity: np.ndarray) -> np.ndarray:
        self._require_fitted()
        C = np.asarray(connectivity, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != self.n_seed_rows:
            raise ValueError(
                f"Connectivity shape mismatch: expected ({self.n_seed_rows}, V), got {C.shape}"
            )
        if C.shape[1] < int(self.k_global):
            raise ValueError(
                f"New subject has {C.shape[1]} voxels but k={self.k_global}"
            )
        if not np.all(np.isfinite(C)):
            raise ValueError("New-subject connectivity contains NaN/Inf")
        cross_covariance = C.T @ np.asarray(self.shared_response, dtype=np.float64)
        return np.asarray(_orthogonal_factor(cross_covariance), dtype=np.float32)

    def align_new_subject_zeroshot(
        self,
        rest_runs: list[np.ndarray],
        external_seed_runs: list[np.ndarray],
        *,
        subject_id: int | None = None,
        seed_manifest_fingerprint: str | None = None,
    ) -> SubjectTransform:
        C = self._new_subject_connectivity(
            rest_runs,
            external_seed_runs,
            seed_manifest_fingerprint,
        )
        return SubjectTransform(
            basis=self._basis_for_connectivity(C),
            expert_name=self.expert_name,
            subject_id=subject_id,
            metadata={"alignment": "zero_shot"},
        )

    def align_new_subject_fewshot(
        self,
        rest_runs: list[np.ndarray],
        task_fmri_shared: np.ndarray,
        external_seed_runs: list[np.ndarray],
        *,
        shot_indices: np.ndarray | None = None,
        subject_id: int | None = None,
        seed_manifest_fingerprint: str | None = None,
    ) -> SubjectTransform:
        if self.task_template is None:
            raise ValueError(
                "Few-shot alignment is unavailable because no task template was fitted"
            )
        C = self._new_subject_connectivity(
            rest_runs,
            external_seed_runs,
            seed_manifest_fingerprint,
        )
        basis = self._basis_for_connectivity(C)
        zero_transform = SubjectTransform(
            basis=basis,
            expert_name=self.expert_name,
            subject_id=subject_id,
        )
        task_fmri_shared = np.asarray(task_fmri_shared, dtype=np.float32)
        Z_new = zero_transform.project(task_fmri_shared)
        if shot_indices is None:
            template_rows = self.task_template[: Z_new.shape[0]]
            logger.warning(
                "Few-shot cSRM alignment: shot_indices not supplied; using the first %d "
                "task-template rows.",
                Z_new.shape[0],
            )
        else:
            shot_indices = np.asarray(shot_indices, dtype=np.int64)
            if shot_indices.ndim != 1 or shot_indices.shape[0] != Z_new.shape[0]:
                raise ValueError(
                    "shot_indices must be a 1D array with one entry per task response"
                )
            if np.any(shot_indices < 0) or np.any(shot_indices >= self.task_template.shape[0]):
                raise IndexError("shot_indices contain rows outside the fitted task template")
            template_rows = self.task_template[shot_indices]
        if template_rows.shape[0] != Z_new.shape[0]:
            raise ValueError(
                f"Task template has only {template_rows.shape[0]} matching rows for "
                f"{Z_new.shape[0]} shots"
            )
        rotation = procrustes_align(Z_new, template_rows)
        return SubjectTransform(
            basis=basis @ rotation,
            expert_name=self.expert_name,
            subject_id=subject_id,
            metadata={"alignment": "few_shot", "n_shots": int(Z_new.shape[0])},
        )

    def save(self, output_dir: str | Path) -> None:
        self._require_fitted()
        output_dir = Path(output_dir)
        transform_dir = output_dir / "transforms"
        transform_dir.mkdir(parents=True, exist_ok=True)

        transform_files: dict[str, str] = {}
        transform_hashes: dict[str, str] = {}
        for subject_id in sorted(self.subject_weights):
            relative = f"transforms/subject_{subject_id}.npz"
            transform = self.training_transform(subject_id)
            transform.save(output_dir / relative)
            transform_files[str(subject_id)] = relative
            transform_hashes[str(subject_id)] = transform.fingerprint

        task_template = (
            self.task_template if self.task_template is not None else np.empty((0, 0), dtype=np.float32)
        )
        np.savez_compressed(
            output_dir / "expert_state.npz",
            shared_response=self.shared_response,
            task_template=task_template,
            objective_history=np.asarray(self.objective_history, dtype=np.float64),
            delta_history=np.asarray(self.delta_history, dtype=np.float64),
        )
        manifest = {
            "schema_version": EXPERT_SCHEMA_VERSION,
            "expert_name": self.expert_name,
            "n_components_requested": self.n_components,
            "min_k": self.min_k,
            "k_global": self.k_global,
            "ensemble_method": self.ensemble_method,
            "max_iters": self.max_iters,
            "tol": self.tol,
            "n_seed_rows": self.n_seed_rows,
            "seed_manifest_fingerprint": self.seed_manifest_fingerprint,
            "training_subjects": sorted(self.subject_weights),
            "transform_files": transform_files,
            "transform_sha256": transform_hashes,
            "shared_response_sha256": _array_sha256(self.shared_response),
            "task_template_sha256": (
                _array_sha256(self.task_template) if self.task_template is not None else None
            ),
            "converged": self.converged,
            "n_iterations": len(self.delta_history),
        }
        (output_dir / "expert_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        output_dir: str | Path,
        *,
        expected_n_components: int | None = None,
        expected_seed_manifest_fingerprint: str | None = None,
    ) -> "ConnectivitySRMExpert":
        output_dir = Path(output_dir)
        manifest = json.loads((output_dir / "expert_manifest.json").read_text(encoding="utf-8"))
        _validate_expected(
            actual=int(manifest["schema_version"]),
            expected=EXPERT_SCHEMA_VERSION,
            label="expert schema version",
        )
        _validate_expected(
            actual=str(manifest["expert_name"]),
            expected=cls.expert_name,
            label="expert name",
        )
        _validate_expected(
            actual=int(manifest["k_global"]),
            expected=expected_n_components,
            label="component count",
        )
        if expected_seed_manifest_fingerprint is not None:
            _validate_seed_manifest_fingerprint(
                manifest.get("seed_manifest_fingerprint"),
                expected_seed_manifest_fingerprint,
            )

        expert = cls(
            n_components=int(manifest["n_components_requested"]),
            min_k=int(manifest["min_k"]),
            ensemble_method=str(manifest["ensemble_method"]),
            max_iters=int(manifest["max_iters"]),
            tol=float(manifest["tol"]),
            seed_manifest_fingerprint=manifest.get("seed_manifest_fingerprint"),
        )
        expert.k_global = int(manifest["k_global"])
        expert.n_seed_rows = int(manifest["n_seed_rows"])
        with np.load(output_dir / "expert_state.npz", allow_pickle=False) as state:
            expert.shared_response = np.asarray(state["shared_response"], dtype=np.float32)
            task_template = np.asarray(state["task_template"], dtype=np.float32)
            expert.task_template = task_template if task_template.size else None
            expert.objective_history = [float(v) for v in state["objective_history"]]
            expert.delta_history = [float(v) for v in state["delta_history"]]
        expert.converged = bool(manifest["converged"])

        if expert.shared_response.shape != (expert.n_seed_rows, expert.k_global):
            raise ValueError(
                "Shared-response shape does not match the saved seed/component dimensions"
            )
        if _array_sha256(expert.shared_response) != manifest["shared_response_sha256"]:
            raise ValueError("Shared-response checksum mismatch")
        expected_task_hash = manifest.get("task_template_sha256")
        if (expert.task_template is None) != (expected_task_hash is None):
            raise ValueError("Task-template presence does not match expert manifest")
        if expert.task_template is not None:
            if expert.task_template.shape[1] != expert.k_global:
                raise ValueError("Task-template component count mismatch")
            if _array_sha256(expert.task_template) != expected_task_hash:
                raise ValueError("Task-template checksum mismatch")

        expected_subjects = [int(v) for v in manifest["training_subjects"]]
        for subject_id in expected_subjects:
            transform = SubjectTransform.load(
                output_dir / manifest["transform_files"][str(subject_id)],
                expected_expert_name=cls.expert_name,
                expected_n_components=expert.k_global,
            )
            if transform.subject_id != subject_id:
                raise ValueError(f"Subject id mismatch in transform for subject {subject_id}")
            if transform.fingerprint != manifest["transform_sha256"][str(subject_id)]:
                raise ValueError(f"Transform checksum mismatch for subject {subject_id}")
            expert.subject_weights[subject_id] = transform.basis
        if sorted(expert.subject_weights) != expected_subjects:
            raise ValueError("Training-subject list does not match saved transforms")
        if len(expert.delta_history) != int(manifest["n_iterations"]):
            raise ValueError("Convergence-history length does not match expert manifest")
        return expert
