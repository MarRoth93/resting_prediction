"""
SharedSpaceBuilder: Build a shared representational space from multiple subjects.

Handles variable voxel counts by operating in k-dimensional component space.
Supports hybrid_cha (default) and strict_rest_cha experiment modes.
"""

import logging
from typing import Optional

import numpy as np

from src.alignment.cha_alignment import align_via_connectivity_fingerprint
from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.utils import (
    column_sign_fix,
    compute_global_k,
    compute_svd_basis,
    procrustes_align,
)

logger = logging.getLogger(__name__)


class SharedSpaceBuilder:
    """
    Build and manage a shared representational space across subjects.

    Solves the variable voxel count problem by projecting each subject's
    brain activity into a common k-dimensional space via REST-derived bases.
    """

    def __init__(
        self,
        n_components: int = 50,
        min_k: int = 10,
        connectivity_mode: str = "parcellation",
        experiment_mode: str = "hybrid_cha",
        ensemble_method: str = "average",
        max_iters: int = 10,
        tol: float = 1e-5,
    ):
        self.n_components = n_components
        self.min_k = min_k
        self.connectivity_mode = connectivity_mode
        self.experiment_mode = experiment_mode
        self.ensemble_method = ensemble_method
        self.max_iters = max_iters
        self.tol = tol

        # Learned state
        self.k_global: int | None = None
        self.subject_bases: dict[int, np.ndarray] = {}  # sub -> P (V_s × k)
        self.subject_rotations: dict[int, np.ndarray] = {}  # sub -> R (k × k)
        self.subject_connectivity: dict[int, np.ndarray] = {}  # sub -> C
        self.subject_fingerprints: dict[int, np.ndarray] = {}  # sub -> F (R × k)
        self.template_Z: np.ndarray | None = None  # (N_shared × k)
        self.template_fingerprint: np.ndarray | None = None  # (R × k) for zero-shot

    def fit(
        self,
        rest_runs: dict[int, list[np.ndarray]],
        task_responses_shared: dict[int, np.ndarray],
        atlas_masked: dict[int, np.ndarray] | None = None,
        n_parcels: int | None = None,
    ) -> "SharedSpaceBuilder":
        """
        Fit shared space using training subjects.

        hybrid_cha mode:
        1. REST → connectivity → SVD → P_s (REST only)
        2. Project shared task responses → Z_s = Y_shared @ P_s
        3. Iterative Procrustes → template + rotations R_s
        4. Build calibrated fingerprint template for zero-shot

        strict_rest_cha mode:
        1. REST → connectivity → SVD → P_s
        2. Compute fingerprints F_s = C_s @ P_s
        3. Iterative Procrustes on fingerprints → template + rotations

        Args:
            rest_runs: sub_id -> list of (T, V_s) REST arrays
            task_responses_shared: sub_id -> (N_shared, V_s) responses to shared1000
                Must be in CANONICAL NSD image ID order.
            atlas_masked: sub_id -> (V_s,) integer parcel labels (for parcellation mode)
            n_parcels: number of parcels (after harmonization)
        """
        sub_ids = sorted(rest_runs.keys())
        logger.info(f"Fitting shared space with subjects {sub_ids}, mode={self.experiment_mode}")

        # Step 1: Compute connectivity and bases for all subjects
        connectivity = {}
        for sub_id in sub_ids:
            logger.info(f"Subject {sub_id}: computing REST connectivity ({self.connectivity_mode})")
            C = compute_rest_connectivity(
                rest_runs[sub_id],
                mode=self.connectivity_mode,
                atlas_masked=atlas_masked.get(sub_id) if atlas_masked else None,
                n_parcels=n_parcels,
                ensemble=self.ensemble_method,
            )
            connectivity[sub_id] = C

        # Compute global k
        self.k_global = compute_global_k(connectivity, self.n_components, self.min_k)

        # Compute bases with global k
        for sub_id in sub_ids:
            P = compute_svd_basis(
                connectivity[sub_id],
                n_components=self.k_global,
                min_k=self.min_k,
            )
            # Ensure exactly k_global columns
            P = P[:, :self.k_global]
            self.subject_bases[sub_id] = P
            self.subject_connectivity[sub_id] = connectivity[sub_id]
            logger.info(f"Subject {sub_id}: basis P shape = {P.shape}")

        # Step 2 & 3: Alignment depends on experiment mode
        if self.experiment_mode == "hybrid_cha":
            self._fit_hybrid(sub_ids, task_responses_shared)
        elif self.experiment_mode == "strict_rest_cha":
            self._fit_strict_rest(sub_ids)
        else:
            raise ValueError(f"Unknown experiment_mode: {self.experiment_mode}")

        # Step 4: Build fingerprint template for zero-shot inference
        if self.connectivity_mode == "parcellation":
            self._build_fingerprint_template(sub_ids)

        return self

    def _fit_hybrid(
        self,
        sub_ids: list[int],
        task_responses_shared: dict[int, np.ndarray],
    ):
        """Hybrid CHA: use task responses for Procrustes template orientation."""
        # Project shared responses to component space
        Z_all = {}
        for sub_id in sub_ids:
            Y = np.array(task_responses_shared[sub_id], dtype=np.float32)
            P = self.subject_bases[sub_id]
            Z = Y @ P  # (N_shared, k)
            Z_all[sub_id] = Z

        # Iterative Procrustes alignment
        template = np.mean(list(Z_all.values()), axis=0)  # (N_shared, k)

        for iteration in range(self.max_iters):
            rotations = {}
            aligned = []
            for sub_id in sub_ids:
                R = procrustes_align(Z_all[sub_id], template)
                rotations[sub_id] = R
                aligned.append(Z_all[sub_id] @ R)

            new_template = np.mean(aligned, axis=0)
            delta = np.linalg.norm(new_template - template) / np.linalg.norm(template)
            template = new_template
            logger.info(f"  Procrustes iteration {iteration + 1}: delta = {delta:.6f}")

            if delta < self.tol:
                logger.info(f"  Converged at iteration {iteration + 1}")
                break

        self.template_Z = template
        self.subject_rotations = rotations

    def _fit_strict_rest(self, sub_ids: list[int]):
        """Strict REST CHA: align using connectivity fingerprints only."""
        # Compute fingerprints
        F_all = {}
        for sub_id in sub_ids:
            C = self.subject_connectivity[sub_id]
            P = self.subject_bases[sub_id]
            F = C @ P  # (R, k) or (V, k)
            F_all[sub_id] = F

        # Iterative Procrustes on fingerprints
        template = np.mean(list(F_all.values()), axis=0)

        for iteration in range(self.max_iters):
            rotations = {}
            aligned = []
            for sub_id in sub_ids:
                R = procrustes_align(F_all[sub_id], template)
                rotations[sub_id] = R
                aligned.append(F_all[sub_id] @ R)

            new_template = np.mean(aligned, axis=0)
            delta = np.linalg.norm(new_template - template) / np.linalg.norm(template)
            template = new_template
            logger.info(f"  Procrustes iteration {iteration + 1}: delta = {delta:.6f}")

            if delta < self.tol:
                break

        self.template_Z = template  # In fingerprint space for strict mode
        self.subject_rotations = rotations
        for sub_id in sub_ids:
            self.subject_fingerprints[sub_id] = F_all[sub_id]

    def _build_fingerprint_template(self, sub_ids: list[int]):
        """
        Build calibrated fingerprint template for zero-shot alignment.

        In hybrid mode: project task-aligned rotations into fingerprint space.
        F_template = mean(C_s @ P_s @ R_s) for training subjects.
        """
        F_aligned = []
        for sub_id in sub_ids:
            C = self.subject_connectivity[sub_id]
            P = self.subject_bases[sub_id]
            R = self.subject_rotations[sub_id]
            F = C @ P @ R  # (R, k)
            F_aligned.append(F)
            self.subject_fingerprints[sub_id] = C @ P  # unrotated

        self.template_fingerprint = np.mean(F_aligned, axis=0).astype(np.float32)
        logger.info(f"Fingerprint template shape: {self.template_fingerprint.shape}")

    def align_new_subject_zeroshot(
        self,
        rest_runs: list[np.ndarray],
        atlas_masked: np.ndarray | None = None,
        n_parcels: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align a new subject using only REST data (zero-shot).

        Returns:
            P_new: (V_new, k) basis
            R_new: (k, k) rotation to shared space
        """
        if self.connectivity_mode != "parcellation":
            raise ValueError(
                "Zero-shot alignment requires parcellation mode "
                "(voxel_correlation produces subject-specific V dimensions)"
            )

        # Compute connectivity and basis
        C_new = compute_rest_connectivity(
            rest_runs,
            mode="parcellation",
            atlas_masked=atlas_masked,
            n_parcels=n_parcels,
            ensemble=self.ensemble_method,
        )

        P_new = compute_svd_basis(
            C_new, n_components=self.k_global, min_k=self.min_k
        )
        P_new = P_new[:, :self.k_global]

        # Align via connectivity fingerprint
        R_new = align_via_connectivity_fingerprint(
            P_new=P_new,
            C_new=C_new,
            template_fingerprint=self.template_fingerprint,
        )

        return P_new, R_new

    def align_new_subject_fewshot(
        self,
        rest_runs: list[np.ndarray],
        task_fmri_shared: np.ndarray,
        shot_indices: np.ndarray | None = None,
        atlas_masked: np.ndarray | None = None,
        n_parcels: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align a new subject using REST + shared task responses (few-shot).

        Args:
            rest_runs: list of (T, V_new) REST arrays
            task_fmri_shared: (N_shot, V_new) responses to shot stimuli
            shot_indices: (N_shot,) indices into the shared stimulus set,
                used to select corresponding rows from template_Z.
                If None, assumes first N_shot stimuli (for backward compat).
            atlas_masked: optional, for parcellation connectivity
            n_parcels: optional

        Returns:
            P_new: (V_new, k) basis
            R_new: (k, k) rotation to shared space

        Raises:
            ValueError: if experiment_mode is strict_rest_cha (template_Z is
                in fingerprint space, incompatible with task-projected Z_new)
        """
        if self.experiment_mode == "strict_rest_cha":
            raise ValueError(
                "Few-shot alignment is not supported with strict_rest_cha mode. "
                "In strict mode, template_Z is in fingerprint space (R, k) which "
                "is incompatible with task-projected Z_new (N_shot, k). "
                "Use hybrid_cha mode for few-shot prediction, or use zero-shot."
            )

        # Compute basis from REST
        C_new = compute_rest_connectivity(
            rest_runs,
            mode=self.connectivity_mode,
            atlas_masked=atlas_masked,
            n_parcels=n_parcels,
            ensemble=self.ensemble_method,
        )

        P_new = compute_svd_basis(
            C_new, n_components=self.k_global, min_k=self.min_k
        )
        P_new = P_new[:, :self.k_global]

        # Project task responses to component space
        Z_new = task_fmri_shared @ P_new  # (N_shot, k)

        # Select corresponding template rows for Procrustes alignment
        if shot_indices is not None:
            template_rows = self.template_Z[shot_indices]
        else:
            template_rows = self.template_Z[:Z_new.shape[0]]
            logger.warning(
                "Few-shot alignment: shot_indices not provided, using first "
                f"{Z_new.shape[0]} template rows. This is only correct if "
                "the shot stimuli correspond to the first rows of template_Z."
            )

        R_new = procrustes_align(Z_new, template_rows)

        return P_new, R_new

    def save(self, output_dir: str):
        """Save builder state to disk."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        np.savez(
            os.path.join(output_dir, "builder.npz"),
            k_global=self.k_global,
            template_Z=self.template_Z,
            template_fingerprint=self.template_fingerprint if self.template_fingerprint is not None else np.array([]),
            connectivity_mode=self.connectivity_mode,
            experiment_mode=self.experiment_mode,
            n_components=self.n_components,
            min_k=self.min_k,
            ensemble_method=self.ensemble_method,
            max_iters=self.max_iters,
            tol=self.tol,
        )

        for sub_id in self.subject_bases:
            np.savez(
                os.path.join(output_dir, f"subject_{sub_id}.npz"),
                basis=self.subject_bases[sub_id],
                rotation=self.subject_rotations[sub_id],
                connectivity=self.subject_connectivity.get(sub_id, np.array([])),
            )

        logger.info(f"Saved SharedSpaceBuilder to {output_dir}")

    @classmethod
    def load(cls, output_dir: str) -> "SharedSpaceBuilder":
        """Load builder state from disk."""
        import os
        import glob as globmod

        data = np.load(os.path.join(output_dir, "builder.npz"), allow_pickle=True)
        builder = cls(
            n_components=int(data["n_components"]),
            connectivity_mode=str(data["connectivity_mode"]),
            experiment_mode=str(data["experiment_mode"]),
            min_k=int(data["min_k"]) if "min_k" in data else 10,
            ensemble_method=str(data["ensemble_method"]) if "ensemble_method" in data else "average",
            max_iters=int(data["max_iters"]) if "max_iters" in data else 10,
            tol=float(data["tol"]) if "tol" in data else 1e-5,
        )
        builder.k_global = int(data["k_global"])
        builder.template_Z = data["template_Z"]
        fp = data["template_fingerprint"]
        builder.template_fingerprint = fp if fp.size > 0 else None

        for f in sorted(globmod.glob(os.path.join(output_dir, "subject_*.npz"))):
            sub_id = int(os.path.basename(f).replace("subject_", "").replace(".npz", ""))
            sub_data = np.load(f, allow_pickle=True)
            builder.subject_bases[sub_id] = sub_data["basis"]
            builder.subject_rotations[sub_id] = sub_data["rotation"]
            C = sub_data["connectivity"]
            if C.size > 0:
                builder.subject_connectivity[sub_id] = C

        logger.info(f"Loaded SharedSpaceBuilder from {output_dir}")
        return builder
