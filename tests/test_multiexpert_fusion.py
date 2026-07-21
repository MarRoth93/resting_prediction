import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.nsd_loader import FeatureBundle
from src.models.multiexpert_encoding import (
    MultiExpertFusionConfig,
    MultiExpertFusionNetwork,
    decode_and_fuse,
    sample_method_mask,
)
from src.pipelines.multiexpert_support import build_fusion_network


def _tiny_config(*, method_dropout=0.25):
    return MultiExpertFusionConfig(
        backbone_dim=16,
        backbone_layers=1,
        backbone_heads=4,
        backbone_ff_multiplier=2.0,
        backbone_dropout=0.0,
        fusion_dim=8,
        fusion_layers=1,
        fusion_heads=2,
        fusion_ff_multiplier=2.0,
        fusion_dropout=0.0,
        method_dropout=method_dropout,
        seed=7,
    )


def _tiny_model():
    torch.manual_seed(7)
    return MultiExpertFusionNetwork(
        input_dim=6,
        expert_dims={"hybrid_cha": 2, "connectivity_srm": 3},
        num_regions=3,
        feature_slices={"clip": (0, 6)},
        config=_tiny_config(),
    )


def test_default_stage1_architecture_contract():
    config = MultiExpertFusionConfig()

    assert config.backbone_dim == 384
    assert config.backbone_layers == 8
    assert config.backbone_heads == 8
    assert config.fusion_dim == 128
    assert config.fusion_layers == 2
    assert config.fusion_heads == 4
    assert config.fusion_dropout == pytest.approx(0.10)
    assert config.method_dropout == pytest.approx(0.25)


def test_method_dropout_always_retains_an_expert():
    generator = torch.Generator().manual_seed(4)

    mask = sample_method_mask(
        batch_size=256,
        num_methods=2,
        dropout_probability=0.75,
        generator=generator,
    )

    assert mask.dtype == torch.bool
    assert torch.all(mask.sum(dim=1) >= 1)


def test_method_dropout_is_bernoulli_conditioned_on_a_nonempty_row():
    generator = torch.Generator().manual_seed(19)
    mask = sample_method_mask(
        batch_size=30000,
        num_methods=2,
        dropout_probability=0.25,
        generator=generator,
    )

    # P(both retained | at least one retained) = .75^2 / (1 - .25^2) = .6.
    both_retained = float((mask.sum(dim=1) == 2).float().mean())
    assert both_retained == pytest.approx(0.60, abs=0.015)


def test_method_dropout_rejects_certain_drop_probability():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        sample_method_mask(2, 2, 1.0)


def test_dropped_weights_are_exact_zero_and_survivors_sum_to_one():
    model = _tiny_model().eval()
    features = torch.randn(3, 6)
    active = torch.tensor(
        [[True, False], [False, True], [True, True]], dtype=torch.bool
    )

    output = model(features, method_mask=active)

    assert torch.equal(output.active_mask, active)
    assert torch.equal(output.weights[0, :, 1], torch.zeros(3))
    assert torch.equal(output.weights[1, :, 0], torch.zeros(3))
    torch.testing.assert_close(
        output.weights.sum(dim=-1),
        torch.ones((3, 3)),
        rtol=0.0,
        atol=1e-7,
    )


def test_eval_is_deterministic_and_uses_every_method():
    model = _tiny_model().eval()
    bundle = FeatureBundle.from_streams(
        {"clip": np.random.RandomState(2).randn(4, 6).astype(np.float32)}
    )

    first = model(bundle)
    second = model(bundle)

    assert torch.all(first.active_mask)
    assert torch.all(second.active_mask)
    for name in model.expert_order:
        torch.testing.assert_close(first.latents[name], second.latents[name])
    torch.testing.assert_close(first.weights, second.weights)


@pytest.mark.parametrize("num_voxels", [5, 11])
def test_decode_and_fuse_supports_varying_voxel_and_latent_dimensions(num_voxels):
    torch.manual_seed(num_voxels)
    latents = {
        "hybrid_cha": torch.randn(4, 2, requires_grad=True),
        "connectivity_srm": torch.randn(4, 3, requires_grad=True),
    }
    bases = {
        "hybrid_cha": torch.randn(num_voxels, 2),
        "connectivity_srm": torch.randn(num_voxels, 3),
    }
    weights = torch.empty(4, 3, 2)
    weights[..., 0] = 0.25
    weights[..., 1] = 0.75
    groups = torch.arange(num_voxels) % 3

    decoded = decode_and_fuse(latents, bases, weights, groups)
    expected_a = latents["hybrid_cha"] @ bases["hybrid_cha"].T
    expected_b = latents["connectivity_srm"] @ bases["connectivity_srm"].T

    assert decoded.fused.shape == (4, num_voxels)
    assert decoded.voxel_weights.shape == (4, num_voxels, 2)
    torch.testing.assert_close(decoded.fused, 0.25 * expected_a + 0.75 * expected_b)
    decoded.fused.square().mean().backward()
    assert latents["hybrid_cha"].grad is not None
    assert latents["connectivity_srm"].grad is not None


def test_fusion_model_save_load_round_trip(tmp_path):
    model = _tiny_model().eval()
    features = torch.randn(3, 6)
    before = model(features)

    model.save(str(tmp_path))
    loaded = MultiExpertFusionNetwork.load(str(tmp_path))
    after = loaded(features)

    with open(tmp_path / "metadata.json") as handle:
        metadata = json.load(handle)
    assert metadata["expert_order"] == ["hybrid_cha", "connectivity_srm"]
    assert metadata["expert_dims"] == {"hybrid_cha": 2, "connectivity_srm": 3}
    for name in model.expert_order:
        torch.testing.assert_close(before.latents[name], after.latents[name])
    torch.testing.assert_close(before.weights, after.weights)


def test_pipeline_network_initialization_is_seeded_before_construction():
    config = {
        "random_seed": 17,
        "fusion": {
            "backbone": {
                "d_model": 12,
                "n_layers": 1,
                "n_heads": 3,
                "ff_multiplier": 2.0,
                "dropout": 0.0,
            },
            "method_projection_dim": 8,
            "transformer_layers": 1,
            "transformer_heads": 2,
            "transformer_ff_multiplier": 2.0,
            "transformer_dropout": 0.0,
            "method_dropout": 0.25,
        },
    }
    kwargs = {
        "config": config,
        "input_dim": 6,
        "expert_dims": {"hybrid_cha": 2, "connectivity_srm": 2},
        "num_regions": 3,
        "feature_slices": {"clip": (0, 6)},
    }
    first = build_fusion_network(**kwargs)
    _ = torch.randn(500)
    second = build_fusion_network(**kwargs)

    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])
