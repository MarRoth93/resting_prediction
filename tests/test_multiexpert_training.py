import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.alignment.experts import SubjectTransform
from src.data.multiexpert_batching import stimulus_disjoint_split
from src.models.multiexpert_encoding import MultiExpertFusionConfig, MultiExpertFusionNetwork
from src.models.multiexpert_training import (
    FittedMultiExpertEncoder,
    FusionOptimizationConfig,
    SubjectFusionData,
)


def _basis(rng, voxels, components):
    q, _ = np.linalg.qr(rng.randn(voxels, components))
    return q.astype(np.float32)


def _tiny_problem():
    rng = np.random.RandomState(11)
    features = rng.randn(40, 6).astype(np.float32)
    stimulus_ids = {1: np.arange(0, 20), 2: np.arange(10, 30)}
    split = stimulus_disjoint_split(stimulus_ids, val_fraction=0.25, seed=3)
    views = {}
    for subject, voxels in [(1, 7), (2, 9)]:
        transforms = {
            "hybrid_cha": SubjectTransform(
                _basis(rng, voxels, 2), "hybrid_cha", subject
            ),
            "connectivity_srm": SubjectTransform(
                _basis(rng, voxels, 3), "connectivity_srm", subject
            ),
        }
        responses = rng.randn(20, voxels).astype(np.float32)
        targets = {
            name: transform.project(responses)
            for name, transform in transforms.items()
        }
        subject_split = split.by_subject[subject]
        views[subject] = SubjectFusionData(
            subject_id=subject,
            stimulus_ids=stimulus_ids[subject],
            responses=responses,
            latent_targets=targets,
            transforms=transforms,
            voxel_groups=np.arange(voxels) % 3,
            train_indices=subject_split.train_indices,
            val_indices=subject_split.val_indices,
        )
    network = MultiExpertFusionNetwork(
        input_dim=6,
        expert_dims={"hybrid_cha": 2, "connectivity_srm": 3},
        num_regions=3,
        feature_slices={"clip": (0, 6)},
        config=MultiExpertFusionConfig(
            backbone_dim=12,
            backbone_layers=1,
            backbone_heads=3,
            backbone_ff_multiplier=2.0,
            backbone_dropout=0.0,
            fusion_dim=8,
            fusion_layers=1,
            fusion_heads=2,
            fusion_ff_multiplier=2.0,
            fusion_dropout=0.0,
            method_dropout=0.25,
            seed=3,
        ),
    )
    fitted = FittedMultiExpertEncoder(
        network,
        FusionOptimizationConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=8,
            max_epochs=2,
            patience=2,
            latent_loss_weight=0.25,
            device="cpu",
            seed=3,
        ),
    )
    return features, views, fitted


def test_small_variable_voxel_training_and_roundtrip(tmp_path):
    features, views, fitted = _tiny_problem()
    fitted.fit(features, views)

    view = views[2]
    rows = view.val_indices
    before = fitted.predict_subject(
        features[view.stimulus_ids[rows]],
        transforms=view.transforms,
        voxel_groups=view.voxel_groups,
        batch_size=4,
    )
    assert before["fused"].shape == (rows.size, 9)
    assert before["regional_weights"].shape == (rows.size, 3, 2)
    np.testing.assert_allclose(before["regional_weights"].sum(axis=-1), 1.0, atol=1e-6)

    fitted.save(tmp_path)
    loaded = FittedMultiExpertEncoder.load(
        tmp_path,
        expected_expert_order=["hybrid_cha", "connectivity_srm"],
        expected_expert_dims={"hybrid_cha": 2, "connectivity_srm": 3},
    )
    after = loaded.predict_subject(
        features[view.stimulus_ids[rows]],
        transforms=view.transforms,
        voxel_groups=view.voxel_groups,
        batch_size=4,
    )
    np.testing.assert_allclose(after["fused"], before["fused"], atol=1e-6)
    assert loaded.validation_summary["metric"] == "mean_subject_median_voxel_correlation"


def test_equal_average_and_single_expert_masks():
    features, views, fitted = _tiny_problem()
    fitted.fit(features, views)
    view = views[1]
    rows = view.val_indices

    result = fitted.predict_subject(
        features[view.stimulus_ids[rows]],
        transforms=view.transforms,
        voxel_groups=view.voxel_groups,
        active_experts=["hybrid_cha"],
        equal_weights=True,
    )

    np.testing.assert_array_equal(result["regional_weights"][..., 0], 1.0)
    np.testing.assert_array_equal(result["regional_weights"][..., 1], 0.0)
    np.testing.assert_allclose(result["fused"], result["per_expert"]["hybrid_cha"])
