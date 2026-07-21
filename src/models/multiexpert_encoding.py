"""Neural fusion of multiple resting-state alignment experts.

The network predicts one latent response per alignment method and regional
mixture weights.  Subject-specific transforms stay outside the network so one
model can be trained across subjects with different voxel counts.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultiExpertFusionConfig:
    """Architecture settings for the Stage-1 multi-expert encoder."""

    backbone_dim: int = 384
    backbone_layers: int = 8
    backbone_heads: int = 8
    backbone_ff_multiplier: float = 4.0
    backbone_dropout: float = 0.20
    fusion_dim: int = 128
    fusion_layers: int = 2
    fusion_heads: int = 4
    fusion_ff_multiplier: float = 4.0
    fusion_dropout: float = 0.10
    method_dropout: float = 0.25
    seed: int = 42

    def __post_init__(self) -> None:
        if self.backbone_dim < 1 or self.backbone_layers < 1:
            raise ValueError("Backbone dimensions and layer count must be positive.")
        if self.fusion_dim < 1 or self.fusion_layers < 1:
            raise ValueError("Fusion dimensions and layer count must be positive.")
        if self.backbone_heads < 1 or self.backbone_dim % self.backbone_heads:
            raise ValueError("backbone_dim must be divisible by backbone_heads.")
        if self.fusion_heads < 1 or self.fusion_dim % self.fusion_heads:
            raise ValueError("fusion_dim must be divisible by fusion_heads.")
        if self.backbone_ff_multiplier <= 0 or self.fusion_ff_multiplier <= 0:
            raise ValueError("Transformer feed-forward multipliers must be positive.")
        if not 0.0 <= self.backbone_dropout < 1.0:
            raise ValueError("backbone_dropout must be in [0, 1).")
        if not 0.0 <= self.fusion_dropout < 1.0:
            raise ValueError("fusion_dropout must be in [0, 1).")
        if not 0.0 <= self.method_dropout < 1.0:
            raise ValueError("method_dropout must be in [0, 1).")

    @classmethod
    def from_config(
        cls,
        fusion_config: Mapping[str, Any] | None,
        *,
        seed: int = 42,
    ) -> "MultiExpertFusionConfig":
        """Build architecture settings from ``config_multiexpert.yaml``."""

        fusion = dict(fusion_config or {})
        backbone = dict(fusion.get("backbone", {}) or {})
        return cls(
            backbone_dim=int(backbone.get("d_model", 384)),
            backbone_layers=int(backbone.get("n_layers", 8)),
            backbone_heads=int(backbone.get("n_heads", 8)),
            backbone_ff_multiplier=float(backbone.get("ff_multiplier", 4.0)),
            backbone_dropout=float(backbone.get("dropout", 0.20)),
            fusion_dim=int(fusion.get("method_projection_dim", 128)),
            fusion_layers=int(fusion.get("transformer_layers", 2)),
            fusion_heads=int(fusion.get("transformer_heads", 4)),
            fusion_ff_multiplier=float(
                fusion.get("transformer_ff_multiplier", 4.0)
            ),
            fusion_dropout=float(fusion.get("transformer_dropout", 0.10)),
            method_dropout=float(fusion.get("method_dropout", 0.25)),
            seed=int(seed),
        )


def _stream_specs(
    input_dim: int,
    feature_slices: Mapping[str, Sequence[int]] | None,
) -> tuple[tuple[str, int, int], ...]:
    if not feature_slices:
        return (("features", 0, int(input_dim)),)

    specs = tuple(
        (str(name), int(bounds[0]), int(bounds[1]))
        for name, bounds in feature_slices.items()
    )
    expected_start = 0
    for name, start, end in specs:
        if start != expected_start or end <= start:
            raise ValueError(
                f"Feature slices must be contiguous; {name!r} has ({start}, {end}) "
                f"after expected start {expected_start}."
            )
        expected_start = end
    if expected_start != int(input_dim):
        raise ValueError(
            f"Feature slices cover {expected_start} columns, expected {input_dim}."
        )
    return specs


class StaticFeatureBackbone(nn.Module):
    """Shared static-image backbone matching the frozen TRIBE encoder pattern."""

    def __init__(
        self,
        input_dim: int,
        feature_slices: Mapping[str, Sequence[int]] | None,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_multiplier: float,
        dropout: float,
    ) -> None:
        super().__init__()
        specs = _stream_specs(input_dim, feature_slices)
        self.input_dim = int(input_dim)
        self.stream_names = tuple(name for name, _, _ in specs)
        self.stream_slices = tuple((start, end) for _, start, end in specs)
        self.stream_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(end - start, d_model, bias=False),
                    nn.LayerNorm(d_model),
                )
                for _, start, end in specs
            ]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.stream_embeddings = nn.Parameter(
            torch.zeros(1, len(specs) + 1, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=max(1, int(round(d_model * ff_multiplier))),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.stream_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected features shaped (batch, {self.input_dim}), got {tuple(x.shape)}."
            )
        stream_tokens = [
            projection(x[:, start:end])
            for projection, (start, end) in zip(
                self.stream_projections, self.stream_slices
            )
        ]
        tokens = torch.stack(stream_tokens, dim=1)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.stream_embeddings
        return self.transformer(tokens)[:, 0]


def sample_method_mask(
    batch_size: int,
    num_methods: int,
    dropout_probability: float,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample independent method dropout while retaining at least one method."""

    if batch_size < 1 or num_methods < 1:
        raise ValueError("batch_size and num_methods must be positive.")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must be in [0, 1).")

    keep = torch.rand(
        (batch_size, num_methods),
        device=device,
        generator=generator,
    ) >= float(dropout_probability)
    empty_rows = torch.nonzero(~keep.any(dim=1), as_tuple=False).flatten()
    while empty_rows.numel():
        keep[empty_rows] = torch.rand(
            (empty_rows.numel(), num_methods),
            device=keep.device,
            generator=generator,
        ) >= float(dropout_probability)
        empty_rows = empty_rows[~keep[empty_rows].any(dim=1)]
    return keep


@dataclass
class MultiExpertOutput:
    """Predicted standardized latents and regional fusion weights."""

    latents: dict[str, torch.Tensor]
    weights: torch.Tensor
    active_mask: torch.Tensor
    cls: torch.Tensor


@dataclass
class DecodedFusion:
    """Subject-voxel predictions before and after regional fusion."""

    fused: torch.Tensor
    per_expert: dict[str, torch.Tensor]
    voxel_weights: torch.Tensor


class MultiExpertFusionNetwork(nn.Module):
    """Shared image encoder with method-token fusion and regional gating."""

    architecture = "tribe_multiexpert_fusion"

    def __init__(
        self,
        input_dim: int,
        expert_dims: Mapping[str, int],
        num_regions: int,
        *,
        feature_slices: Mapping[str, Sequence[int]] | None = None,
        config: MultiExpertFusionConfig | None = None,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if not expert_dims:
            raise ValueError("At least one alignment expert is required.")
        if num_regions < 1:
            raise ValueError("num_regions must be positive.")

        self.input_dim = int(input_dim)
        self.expert_order = tuple(str(name) for name in expert_dims)
        if len(set(self.expert_order)) != len(self.expert_order):
            raise ValueError("Expert names must be unique.")
        self.expert_dims = {
            str(name): int(dim) for name, dim in expert_dims.items()
        }
        if any(dim < 1 for dim in self.expert_dims.values()):
            raise ValueError("Every expert latent dimension must be positive.")
        self.num_regions = int(num_regions)
        self.feature_slices = {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in (feature_slices or {}).items()
        }
        self.config = config or MultiExpertFusionConfig()

        self.backbone = StaticFeatureBackbone(
            input_dim=self.input_dim,
            feature_slices=self.feature_slices,
            d_model=self.config.backbone_dim,
            n_layers=self.config.backbone_layers,
            n_heads=self.config.backbone_heads,
            ff_multiplier=self.config.backbone_ff_multiplier,
            dropout=self.config.backbone_dropout,
        )
        self.method_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.config.backbone_dim, self.config.fusion_dim),
                    nn.LayerNorm(self.config.fusion_dim),
                    nn.GELU(),
                )
                for _ in self.expert_order
            ]
        )
        self.method_embeddings = nn.Parameter(
            torch.zeros(1, len(self.expert_order), self.config.fusion_dim)
        )
        self.fusion_cls = nn.Parameter(
            torch.zeros(1, 1, self.config.fusion_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.fusion_dim,
            nhead=self.config.fusion_heads,
            dim_feedforward=max(
                1,
                int(round(self.config.fusion_dim * self.config.fusion_ff_multiplier)),
            ),
            dropout=self.config.fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.config.fusion_layers,
            norm=nn.LayerNorm(self.config.fusion_dim),
            enable_nested_tensor=False,
        )
        self.latent_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.config.fusion_dim, self.expert_dims[name])
                for name in self.expert_order
            }
        )
        self.weight_head = nn.Linear(
            self.config.fusion_dim,
            self.num_regions * len(self.expert_order),
        )
        nn.init.normal_(self.method_embeddings, std=0.02)
        nn.init.normal_(self.fusion_cls, std=0.02)

    @property
    def num_experts(self) -> int:
        return len(self.expert_order)

    def _feature_tensor(self, features: Any) -> torch.Tensor:
        if hasattr(features, "array"):
            features = features.array
        if torch.is_tensor(features):
            return features.to(dtype=torch.float32)
        device = next(self.parameters()).device
        return torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)

    def _active_methods(
        self,
        batch_size: int,
        device: torch.device,
        method_mask: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor:
        if method_mask is None:
            if self.training:
                return sample_method_mask(
                    batch_size,
                    self.num_experts,
                    self.config.method_dropout,
                    device=device,
                )
            return torch.ones(
                (batch_size, self.num_experts), dtype=torch.bool, device=device
            )

        active = torch.as_tensor(method_mask, dtype=torch.bool, device=device)
        expected = (batch_size, self.num_experts)
        if tuple(active.shape) != expected:
            raise ValueError(
                f"method_mask has shape {tuple(active.shape)}, expected {expected}."
            )
        if not torch.all(active.any(dim=1)):
            raise ValueError("Every sample must retain at least one alignment expert.")
        return active

    def forward(
        self,
        features: torch.Tensor | np.ndarray | Any,
        method_mask: torch.Tensor | np.ndarray | None = None,
    ) -> MultiExpertOutput:
        x = self._feature_tensor(features)
        shared = self.backbone(x)
        active = self._active_methods(x.shape[0], x.device, method_mask)

        method_tokens = torch.stack(
            [projection(shared) for projection in self.method_projections],
            dim=1,
        )
        method_tokens = method_tokens + self.method_embeddings
        cls = self.fusion_cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, method_tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device),
                ~active,
            ],
            dim=1,
        )
        contextualized = self.fusion_transformer(
            tokens,
            src_key_padding_mask=padding_mask,
        )
        cls_context = contextualized[:, 0]
        method_context = contextualized[:, 1:]
        latents = {
            name: self.latent_heads[name](method_context[:, index])
            for index, name in enumerate(self.expert_order)
        }

        logits = self.weight_head(cls_context).reshape(
            x.shape[0], self.num_regions, self.num_experts
        )
        logits = logits.masked_fill(~active[:, None, :], -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        # Keep the masking contract exact even if softmax implementation changes.
        weights = weights * active[:, None, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return MultiExpertOutput(
            latents=latents,
            weights=weights,
            active_mask=active,
            cls=cls_context,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "artifact_version": 1,
            "architecture": self.architecture,
            "input_dim": self.input_dim,
            "expert_order": list(self.expert_order),
            "expert_dims": self.expert_dims,
            "num_regions": self.num_regions,
            "feature_slices": self.feature_slices,
            "config": asdict(self.config),
        }

    def save(self, path_or_dir: str) -> None:
        os.makedirs(path_or_dir, exist_ok=True)
        state = {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
        }
        torch.save(state, os.path.join(path_or_dir, "model.pt"))
        with open(os.path.join(path_or_dir, "metadata.json"), "w") as handle:
            json.dump(self.manifest(), handle, indent=2)

    @classmethod
    def load(
        cls,
        path_or_dir: str,
        *,
        map_location: torch.device | str = "cpu",
    ) -> "MultiExpertFusionNetwork":
        with open(os.path.join(path_or_dir, "metadata.json")) as handle:
            metadata = json.load(handle)
        if metadata.get("architecture") != cls.architecture:
            raise ValueError(
                f"Unsupported fusion artifact {metadata.get('architecture')!r}."
            )
        expert_order = [str(name) for name in metadata["expert_order"]]
        dims = metadata["expert_dims"]
        expert_dims = {name: int(dims[name]) for name in expert_order}
        model = cls(
            input_dim=int(metadata["input_dim"]),
            expert_dims=expert_dims,
            num_regions=int(metadata["num_regions"]),
            feature_slices=metadata.get("feature_slices") or None,
            config=MultiExpertFusionConfig(**metadata["config"]),
        )
        state = torch.load(
            os.path.join(path_or_dir, "model.pt"),
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.to(map_location)
        model.eval()
        return model


def decode_and_fuse(
    latents: Mapping[str, torch.Tensor],
    subject_bases: Mapping[str, torch.Tensor | np.ndarray | Any],
    regional_weights: torch.Tensor,
    voxel_groups: torch.Tensor | np.ndarray,
    *,
    expert_order: Sequence[str] | None = None,
) -> DecodedFusion:
    """Decode expert latents to one subject and combine them by voxel region.

    Each basis is shaped ``(subject_voxels, expert_latent_dim)``.  Calling this
    function once per subject permits different voxel counts without padding.
    """

    order = tuple(expert_order or latents.keys())
    if not order:
        raise ValueError("At least one expert latent is required.")
    if set(order) != set(latents) or set(order) != set(subject_bases):
        raise ValueError("latents, subject_bases, and expert_order must name the same experts.")
    if regional_weights.ndim != 3:
        raise ValueError("regional_weights must have shape (batch, regions, experts).")
    if regional_weights.shape[2] != len(order):
        raise ValueError(
            f"regional_weights has {regional_weights.shape[2]} experts, expected {len(order)}."
        )

    batch_size = int(regional_weights.shape[0])
    device = regional_weights.device
    dtype = regional_weights.dtype
    decoded: dict[str, torch.Tensor] = {}
    num_voxels: int | None = None
    for name in order:
        latent = latents[name]
        if latent.ndim != 2 or latent.shape[0] != batch_size:
            raise ValueError(
                f"Latent {name!r} must have shape ({batch_size}, k), got {tuple(latent.shape)}."
            )
        basis_value = getattr(subject_bases[name], "basis", subject_bases[name])
        basis = torch.as_tensor(basis_value, dtype=dtype, device=device)
        if basis.ndim != 2 or basis.shape[1] != latent.shape[1]:
            raise ValueError(
                f"Basis {name!r} has shape {tuple(basis.shape)} but latent dimension "
                f"is {latent.shape[1]}."
            )
        if num_voxels is None:
            num_voxels = int(basis.shape[0])
        elif int(basis.shape[0]) != num_voxels:
            raise ValueError("All expert bases for a subject must have the same voxel count.")
        decoded[name] = latent.to(device=device, dtype=dtype) @ basis.T

    assert num_voxels is not None
    groups = torch.as_tensor(voxel_groups, dtype=torch.long, device=device)
    if groups.ndim != 1 or groups.shape[0] != num_voxels:
        raise ValueError(
            f"voxel_groups must have shape ({num_voxels},), got {tuple(groups.shape)}."
        )
    if groups.numel() and (
        int(groups.min().item()) < 0
        or int(groups.max().item()) >= int(regional_weights.shape[1])
    ):
        raise ValueError("voxel_groups contains an out-of-range regional index.")

    per_expert = torch.stack([decoded[name] for name in order], dim=-1)
    voxel_weights = regional_weights[:, groups, :]
    fused = (per_expert * voxel_weights).sum(dim=-1)
    return DecodedFusion(
        fused=fused,
        per_expert=decoded,
        voxel_weights=voxel_weights,
    )
