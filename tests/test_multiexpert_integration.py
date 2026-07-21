from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.alignment.connectivity_srm import ConnectivitySRMExpert
from src.alignment.experts import HybridCHAExpert
from src.data.multiexpert_batching import stimulus_disjoint_split
from src.models.multiexpert_encoding import MultiExpertFusionConfig, MultiExpertFusionNetwork
from src.models.multiexpert_training import (
    FittedMultiExpertEncoder,
    FusionOptimizationConfig,
    SubjectFusionData,
)
from src.pipelines.multiexpert_support import fit_alignment_experts, training_transforms


def test_rest_experts_to_voxel_fusion_small_integration():
    rng = np.random.RandomState(21)
    subjects = {}
    seed_runs = {}
    shared_task = {}
    for subject, voxels in [(1, 8), (2, 10)]:
        subject_rest = []
        subject_seeds = []
        mixing = rng.randn(6, voxels).astype(np.float32)
        for _ in range(2):
            seeds = rng.randn(45, 6).astype(np.float32)
            rest = seeds @ mixing + 0.1 * rng.randn(45, voxels).astype(np.float32)
            subject_seeds.append(seeds)
            subject_rest.append(rest.astype(np.float32))
        subjects[subject] = SimpleNamespace(rest_runs=subject_rest)
        seed_runs[subject] = subject_seeds
        shared_task[subject] = rng.randn(14, voxels).astype(np.float32)

    experts = {
        "hybrid_cha": HybridCHAExpert(
            n_components=3,
            min_k=2,
            ensemble_method="concat",
            max_iters=3,
            seed_manifest_fingerprint="synthetic-seeds",
        ),
        "connectivity_srm": ConnectivitySRMExpert(
            n_components=3,
            min_k=2,
            ensemble_method="concat",
            max_iters=5,
            seed_manifest_fingerprint="synthetic-seeds",
        ),
    }
    fit_alignment_experts(
        experts,
        subjects=subjects,
        external_seed_runs=seed_runs,
        task_responses_shared=shared_task,
    )

    features = rng.randn(40, 6).astype(np.float32)
    stimulus_ids = {1: np.arange(20), 2: np.arange(10, 30)}
    split = stimulus_disjoint_split(stimulus_ids, val_fraction=0.25, seed=4)
    views = {}
    for subject, voxels in [(1, 8), (2, 10)]:
        transforms = training_transforms(experts, subject)
        responses = rng.randn(20, voxels).astype(np.float32)
        views[subject] = SubjectFusionData(
            subject_id=subject,
            stimulus_ids=stimulus_ids[subject],
            responses=responses,
            latent_targets={
                name: transform.project(responses)
                for name, transform in transforms.items()
            },
            transforms=transforms,
            voxel_groups=np.arange(voxels) % 2,
            train_indices=split.by_subject[subject].train_indices,
            val_indices=split.by_subject[subject].val_indices,
        )

    network = MultiExpertFusionNetwork(
        input_dim=6,
        expert_dims={"hybrid_cha": 3, "connectivity_srm": 3},
        num_regions=2,
        feature_slices={"clip": (0, 6)},
        config=MultiExpertFusionConfig(
            backbone_dim=12,
            backbone_layers=1,
            backbone_heads=3,
            backbone_dropout=0.0,
            fusion_dim=8,
            fusion_layers=1,
            fusion_heads=2,
            fusion_dropout=0.0,
            method_dropout=0.25,
            seed=4,
        ),
    )
    encoder = FittedMultiExpertEncoder(
        network,
        FusionOptimizationConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=8,
            max_epochs=1,
            patience=1,
            device="cpu",
            seed=4,
        ),
    ).fit(features, views)
    view = views[2]
    rows = view.val_indices
    result = encoder.predict_subject(
        features[view.stimulus_ids[rows]],
        transforms=view.transforms,
        voxel_groups=view.voxel_groups,
    )

    assert result["fused"].shape == (rows.size, 10)
    assert result["regional_weights"].shape == (rows.size, 2, 2)
    np.testing.assert_allclose(result["regional_weights"].sum(axis=-1), 1.0, atol=1e-6)
