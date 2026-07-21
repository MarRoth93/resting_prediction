"""Common contracts for interchangeable REST-alignment experts.

The frozen pipeline still uses :class:`SharedSpaceBuilder` directly.  This
module adds a small adapter layer for the experimental multi-expert pipeline
without changing that established path.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.alignment.cha_alignment import align_via_connectivity_fingerprint
from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.shared_space import SharedSpaceBuilder
from src.alignment.utils import compute_svd_basis


TRANSFORM_SCHEMA_VERSION = 1
EXPERT_SCHEMA_VERSION = 1


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _validate_expected(
    *,
    actual: Any,
    expected: Any | None,
    label: str,
) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} mismatch: artifact={actual!r}, expected={expected!r}")


def _validate_seed_manifest_fingerprint(
    stored: str | None,
    supplied: str | None,
) -> None:
    """Reject an explicitly incompatible seed definition/order."""
    if stored is None:
        if supplied is not None:
            raise ValueError(
                "Seed manifest fingerprint mismatch: artifact has no fingerprint, "
                f"supplied={supplied!r}"
            )
        return
    if supplied is None:
        raise ValueError(
            "This alignment expert requires a seed_manifest_fingerprint; "
            "the caller did not supply one."
        )
    if supplied != stored:
        raise ValueError(
            "Seed manifest fingerprint mismatch: "
            f"artifact={stored!r}, supplied={supplied!r}"
        )


@dataclass
class SubjectTransform:
    """A subject's effective voxel-to-shared-space orthonormal basis.

    ``project`` maps voxel responses ``(samples, voxels)`` into an expert's
    shared latent space.  ``reconstruct`` applies the matching transpose map.
    Storing the already-composed basis makes this contract identical for CHA
    (``P @ R``) and connectivity-SRM (``W``).
    """

    basis: np.ndarray
    expert_name: str
    subject_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=np.float32)
        if basis.ndim != 2 or min(basis.shape) < 1:
            raise ValueError(f"basis must be a non-empty 2D array, got {basis.shape}")
        if not np.all(np.isfinite(basis)):
            raise ValueError("basis contains NaN/Inf")
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), atol=5e-4, rtol=5e-4):
            max_error = float(np.max(np.abs(gram - np.eye(basis.shape[1]))))
            raise ValueError(
                "basis columns must be orthonormal for transpose reconstruction; "
                f"maximum Gram-matrix error is {max_error:.3e}"
            )
        if not str(self.expert_name):
            raise ValueError("expert_name must be non-empty")
        self.basis = np.ascontiguousarray(basis)
        self.expert_name = str(self.expert_name)
        self.subject_id = None if self.subject_id is None else int(self.subject_id)
        self.metadata = dict(self.metadata)

    @property
    def n_voxels(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.basis.shape[1])

    @property
    def fingerprint(self) -> str:
        return _array_sha256(self.basis)

    def project(self, responses: np.ndarray) -> np.ndarray:
        responses = np.asarray(responses, dtype=np.float32)
        if responses.ndim != 2:
            raise ValueError(f"responses must be 2D, got {responses.shape}")
        if responses.shape[1] != self.n_voxels:
            raise ValueError(
                f"voxel count mismatch: responses={responses.shape[1]}, "
                f"transform={self.n_voxels}"
            )
        if not np.all(np.isfinite(responses)):
            raise ValueError("responses contain NaN/Inf")
        return np.asarray(responses @ self.basis, dtype=np.float32)

    def reconstruct(self, latents: np.ndarray) -> np.ndarray:
        latents = np.asarray(latents, dtype=np.float32)
        if latents.ndim != 2:
            raise ValueError(f"latents must be 2D, got {latents.shape}")
        if latents.shape[1] != self.n_components:
            raise ValueError(
                f"component count mismatch: latents={latents.shape[1]}, "
                f"transform={self.n_components}"
            )
        if not np.all(np.isfinite(latents)):
            raise ValueError("latents contain NaN/Inf")
        return np.asarray(latents @ self.basis.T, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(
            self.metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        np.savez_compressed(
            path,
            schema_version=np.int64(TRANSFORM_SCHEMA_VERSION),
            basis=self.basis,
            expert_name=np.str_(self.expert_name),
            subject_id=np.int64(-1 if self.subject_id is None else self.subject_id),
            metadata_json=np.str_(metadata_json),
            basis_sha256=np.str_(self.fingerprint),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_expert_name: str | None = None,
        expected_n_components: int | None = None,
        expected_n_voxels: int | None = None,
    ) -> "SubjectTransform":
        with np.load(Path(path), allow_pickle=False) as data:
            schema_version = int(data["schema_version"])
            _validate_expected(
                actual=schema_version,
                expected=TRANSFORM_SCHEMA_VERSION,
                label="SubjectTransform schema version",
            )
            basis = np.asarray(data["basis"], dtype=np.float32)
            expert_name = str(data["expert_name"])
            subject_raw = int(data["subject_id"])
            metadata = json.loads(str(data["metadata_json"]))
            stored_hash = str(data["basis_sha256"])

        _validate_expected(
            actual=expert_name,
            expected=expected_expert_name,
            label="expert name",
        )
        _validate_expected(
            actual=int(basis.shape[1]),
            expected=expected_n_components,
            label="component count",
        )
        _validate_expected(
            actual=int(basis.shape[0]),
            expected=expected_n_voxels,
            label="voxel count",
        )
        if _array_sha256(basis) != stored_hash:
            raise ValueError(f"SubjectTransform checksum mismatch: {path}")
        return cls(
            basis=basis,
            expert_name=expert_name,
            subject_id=None if subject_raw < 0 else subject_raw,
            metadata=metadata,
        )


class AlignmentExpert(ABC):
    """Interface consumed by the experimental multi-expert pipeline."""

    expert_name: str
    k_global: int | None

    @abstractmethod
    def fit(
        self,
        rest_runs: dict[int, list[np.ndarray]],
        task_responses_shared: dict[int, np.ndarray] | None,
        external_seed_runs: dict[int, list[np.ndarray]],
    ) -> "AlignmentExpert":
        """Fit the expert on training subjects."""

    @abstractmethod
    def training_transform(self, subject_id: int) -> SubjectTransform:
        """Return the fitted transform for a training subject."""

    @abstractmethod
    def align_new_subject_zeroshot(
        self,
        rest_runs: list[np.ndarray],
        external_seed_runs: list[np.ndarray],
        *,
        subject_id: int | None = None,
        seed_manifest_fingerprint: str | None = None,
    ) -> SubjectTransform:
        """Infer a transform using REST only."""

    @abstractmethod
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
        """Infer a REST transform and calibrate it with shared task responses."""

    @abstractmethod
    def save(self, output_dir: str | Path) -> None:
        """Persist all fitted state."""


class HybridCHAExpert(AlignmentExpert):
    """Adapter exposing the frozen hybrid-CHA builder through the expert API."""

    expert_name = "hybrid_cha"

    def __init__(
        self,
        n_components: int = 50,
        min_k: int = 10,
        ensemble_method: str = "average",
        max_iters: int = 10,
        tol: float = 1e-5,
        seed_manifest_fingerprint: str | None = None,
    ) -> None:
        self.n_components = int(n_components)
        self.min_k = int(min_k)
        self.ensemble_method = str(ensemble_method)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.seed_manifest_fingerprint = seed_manifest_fingerprint
        self.n_seed_rows: int | None = None
        self.builder = SharedSpaceBuilder(
            n_components=self.n_components,
            min_k=self.min_k,
            ensemble_method=self.ensemble_method,
            max_iters=self.max_iters,
            tol=self.tol,
        )

    @property
    def k_global(self) -> int | None:
        return self.builder.k_global

    def fit(
        self,
        rest_runs: dict[int, list[np.ndarray]],
        task_responses_shared: dict[int, np.ndarray] | None,
        external_seed_runs: dict[int, list[np.ndarray]],
    ) -> "HybridCHAExpert":
        if task_responses_shared is None:
            raise ValueError("hybrid_cha requires shared task responses during fit")
        self.builder.fit(rest_runs, task_responses_shared, external_seed_runs)
        seed_rows = {int(C.shape[0]) for C in self.builder.subject_connectivity.values()}
        if len(seed_rows) != 1:
            raise ValueError(f"Training subjects have inconsistent seed counts: {seed_rows}")
        self.n_seed_rows = seed_rows.pop()
        return self

    def training_transform(self, subject_id: int) -> SubjectTransform:
        subject_id = int(subject_id)
        if subject_id not in self.builder.subject_bases:
            raise KeyError(f"No fitted hybrid_cha transform for subject {subject_id}")
        basis = self.builder.subject_bases[subject_id] @ self.builder.subject_rotations[subject_id]
        return SubjectTransform(
            basis=basis,
            expert_name=self.expert_name,
            subject_id=subject_id,
            metadata={"alignment": "training"},
        )

    def _validate_new_seed_runs(
        self,
        external_seed_runs: list[np.ndarray],
        seed_manifest_fingerprint: str | None,
    ) -> None:
        _validate_seed_manifest_fingerprint(
            self.seed_manifest_fingerprint,
            seed_manifest_fingerprint,
        )
        if self.n_seed_rows is None:
            raise ValueError("hybrid_cha expert is not fitted")
        row_counts = {int(run.shape[1]) for run in external_seed_runs if run.ndim == 2}
        if row_counts != {self.n_seed_rows}:
            raise ValueError(
                f"Seed count mismatch: artifact={self.n_seed_rows}, supplied={sorted(row_counts)}"
            )

    def align_new_subject_zeroshot(
        self,
        rest_runs: list[np.ndarray],
        external_seed_runs: list[np.ndarray],
        *,
        subject_id: int | None = None,
        seed_manifest_fingerprint: str | None = None,
    ) -> SubjectTransform:
        self._validate_new_seed_runs(external_seed_runs, seed_manifest_fingerprint)
        C = compute_rest_connectivity(
            rest_runs,
            seed_runs=external_seed_runs,
            ensemble=self.ensemble_method,
        )
        if self.builder.template_fingerprint is None:
            raise ValueError("Missing hybrid-CHA fingerprint template.")
        P = compute_svd_basis(
            C,
            n_components=int(self.k_global),
            min_k=self.min_k,
        )[:, : int(self.k_global)]
        available_rows = np.linalg.norm(C, axis=1) > 1e-10
        n_available = int(available_rows.sum())
        if n_available <= int(self.k_global):
            raise ValueError(
                "Too few nonzero seed rows to align the new subject: "
                f"available={n_available}, k={self.k_global}."
            )
        R = align_via_connectivity_fingerprint(
            P_new=P,
            C_new=C[available_rows],
            template_fingerprint=self.builder.template_fingerprint[available_rows],
        )
        return SubjectTransform(
            basis=P @ R,
            expert_name=self.expert_name,
            subject_id=subject_id,
            metadata={
                "alignment": "zero_shot",
                "available_seed_rows": n_available,
                "total_seed_rows": int(C.shape[0]),
            },
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
        self._validate_new_seed_runs(external_seed_runs, seed_manifest_fingerprint)
        P, R = self.builder.align_new_subject_fewshot(
            rest_runs,
            task_fmri_shared,
            external_seed_runs,
            shot_indices=shot_indices,
        )
        return SubjectTransform(
            basis=P @ R,
            expert_name=self.expert_name,
            subject_id=subject_id,
            metadata={"alignment": "few_shot", "n_shots": int(task_fmri_shared.shape[0])},
        )

    def save(self, output_dir: str | Path) -> None:
        if self.k_global is None or self.n_seed_rows is None:
            raise ValueError("Cannot save an unfitted hybrid_cha expert")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.builder.save(str(output_dir))
        transforms = {
            str(subject_id): self.training_transform(subject_id).fingerprint
            for subject_id in sorted(self.builder.subject_bases)
        }
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
            "training_subjects": sorted(self.builder.subject_bases),
            "transform_sha256": transforms,
            "task_template_sha256": _array_sha256(self.builder.template_Z),
            "connectivity_template_sha256": _array_sha256(
                self.builder.template_fingerprint
            ),
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
    ) -> "HybridCHAExpert":
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
        expert.builder = SharedSpaceBuilder.load(str(output_dir))
        expert.n_seed_rows = int(manifest["n_seed_rows"])
        if _array_sha256(expert.builder.template_Z) != manifest["task_template_sha256"]:
            raise ValueError("Hybrid-CHA task-template checksum mismatch")
        if (
            _array_sha256(expert.builder.template_fingerprint)
            != manifest["connectivity_template_sha256"]
        ):
            raise ValueError("Hybrid-CHA connectivity-template checksum mismatch")
        expected_subjects = [int(v) for v in manifest["training_subjects"]]
        if sorted(expert.builder.subject_bases) != expected_subjects:
            raise ValueError("Training-subject list does not match saved hybrid_cha files")
        for subject_id in expected_subjects:
            transform = expert.training_transform(subject_id)
            expected_hash = manifest["transform_sha256"][str(subject_id)]
            if transform.fingerprint != expected_hash:
                raise ValueError(f"Transform checksum mismatch for subject {subject_id}")
        return expert


def load_alignment_expert(
    expert_name: str,
    output_dir: str | Path,
    **compatibility: Any,
) -> AlignmentExpert:
    """Load a named expert while keeping pipeline dispatch in one place."""
    if expert_name == HybridCHAExpert.expert_name:
        return HybridCHAExpert.load(output_dir, **compatibility)
    if expert_name == "connectivity_srm":
        from src.alignment.connectivity_srm import ConnectivitySRMExpert

        return ConnectivitySRMExpert.load(output_dir, **compatibility)
    raise ValueError(f"Unknown alignment expert: {expert_name!r}")
