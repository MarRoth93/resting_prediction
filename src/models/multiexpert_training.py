"""Training and inference wrapper for the Stage-1 multi-expert fusion network."""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from src.alignment.experts import SubjectTransform
from src.data.multiexpert_batching import SubjectHomogeneousBatchSampler
from src.models.multiexpert_encoding import (
    MultiExpertFusionNetwork,
    decode_and_fuse,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionOptimizationConfig:
    learning_rate: float = 2.084109344364613e-4
    weight_decay: float = 3.0201957739732042e-5
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 20
    latent_loss_weight: float = 0.25
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("Batch size, epochs, and patience must be positive.")
        if self.latent_loss_weight < 0:
            raise ValueError("latent_loss_weight must be non-negative.")

    @classmethod
    def from_pipeline_config(cls, config: dict, *, seed: int | None = None):
        fusion = config["fusion"]
        return cls(
            learning_rate=float(fusion["learning_rate"]),
            weight_decay=float(fusion["weight_decay"]),
            batch_size=int(fusion["batch_size"]),
            max_epochs=int(fusion["max_epochs"]),
            patience=int(fusion["patience"]),
            latent_loss_weight=float(fusion["latent_loss_weight"]),
            device=str(fusion["device"]),
            seed=int(config["random_seed"] if seed is None else seed),
        )


@dataclass
class SubjectFusionData:
    """One subject's variable-voxel training view."""

    subject_id: int
    stimulus_ids: np.ndarray
    responses: np.ndarray
    latent_targets: dict[str, np.ndarray]
    transforms: dict[str, SubjectTransform]
    voxel_groups: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray

    def __post_init__(self) -> None:
        self.subject_id = int(self.subject_id)
        self.stimulus_ids = np.asarray(self.stimulus_ids, dtype=np.int64)
        self.voxel_groups = np.asarray(self.voxel_groups, dtype=np.int64)
        self.train_indices = np.asarray(self.train_indices, dtype=np.int64)
        self.val_indices = np.asarray(self.val_indices, dtype=np.int64)
        if self.stimulus_ids.ndim != 1 or np.asarray(self.responses).ndim != 2:
            raise ValueError("stimulus_ids and responses must be 1D and 2D, respectively.")
        n_rows, n_voxels = np.asarray(self.responses).shape
        if self.stimulus_ids.shape[0] != n_rows:
            raise ValueError("Stimulus and response row counts differ.")
        if self.voxel_groups.shape != (n_voxels,):
            raise ValueError(
                f"Subject {self.subject_id}: voxel_groups has {self.voxel_groups.shape}, "
                f"expected ({n_voxels},)."
            )
        if set(self.latent_targets) != set(self.transforms):
            raise ValueError("latent_targets and transforms must name the same experts.")
        for name, target in self.latent_targets.items():
            target = np.asarray(target, dtype=np.float32)
            transform = self.transforms[name]
            if target.shape != (n_rows, transform.n_components):
                raise ValueError(
                    f"Subject {self.subject_id} target {name!r} has shape {target.shape}, "
                    f"expected {(n_rows, transform.n_components)}."
                )
            if transform.n_voxels != n_voxels:
                raise ValueError(
                    f"Subject {self.subject_id} transform {name!r} has "
                    f"{transform.n_voxels} voxels, expected {n_voxels}."
                )
            self.latent_targets[name] = target
        all_indices = np.concatenate([self.train_indices, self.val_indices])
        if all_indices.size != n_rows or set(all_indices.tolist()) != set(range(n_rows)):
            raise ValueError("Training/validation indices must partition every response row.")
        if np.intersect1d(self.train_indices, self.val_indices).size:
            raise ValueError("Training and validation rows overlap.")


class _RunningMoments:
    def __init__(self, width: int):
        self.count = 0
        self.mean = np.zeros(int(width), dtype=np.float64)
        self.m2 = np.zeros(int(width), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.shape[0]:
            raise ValueError("Moment-update dimensions do not match.")
        if values.shape[0] == 0:
            return
        count = int(values.shape[0])
        batch_mean = values.mean(axis=0)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)
        if self.count == 0:
            self.count = count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + count
        self.mean += delta * (count / total)
        self.m2 += batch_m2 + delta * delta * (self.count * count / total)
        self.count = total

    def finalize(self, *, constant_value: float) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 1:
            raise ValueError("Cannot finalize empty running moments.")
        std = np.sqrt(self.m2 / self.count)
        std[std < 1e-8] = float(constant_value)
        return self.mean.astype(np.float32), std.astype(np.float32)


class _VoxelCorrelationAccumulator:
    def __init__(self, n_voxels: int):
        self.n = 0
        self.true_sum = np.zeros(n_voxels, dtype=np.float64)
        self.pred_sum = np.zeros(n_voxels, dtype=np.float64)
        self.true_sq = np.zeros(n_voxels, dtype=np.float64)
        self.pred_sq = np.zeros(n_voxels, dtype=np.float64)
        self.cross = np.zeros(n_voxels, dtype=np.float64)

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        truth = np.asarray(truth, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if truth.shape != prediction.shape or truth.ndim != 2:
            raise ValueError("Correlation batches must be matching 2D arrays.")
        self.n += int(truth.shape[0])
        self.true_sum += truth.sum(axis=0)
        self.pred_sum += prediction.sum(axis=0)
        self.true_sq += np.sum(truth * truth, axis=0)
        self.pred_sq += np.sum(prediction * prediction, axis=0)
        self.cross += np.sum(truth * prediction, axis=0)

    def correlation(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.true_sum, dtype=np.float32)
        numerator = self.cross - self.true_sum * self.pred_sum / self.n
        true_var = self.true_sq - self.true_sum * self.true_sum / self.n
        pred_var = self.pred_sq - self.pred_sum * self.pred_sum / self.n
        denominator = np.sqrt(np.maximum(true_var, 0.0) * np.maximum(pred_var, 0.0))
        result = np.zeros_like(numerator)
        valid = denominator > 1e-12
        result[valid] = numerator[valid] / denominator[valid]
        return np.clip(result, -1.0, 1.0).astype(np.float32)


class FittedMultiExpertEncoder:
    """Own the neural model, fitted standardizers, training, and inference."""

    artifact_version = 1

    def __init__(
        self,
        network: MultiExpertFusionNetwork,
        optimization: FusionOptimizationConfig | None = None,
    ) -> None:
        self.network = network
        self.optimization = optimization or FusionOptimizationConfig()
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.z_mean: dict[str, np.ndarray] = {}
        self.z_std: dict[str, np.ndarray] = {}
        self.history: list[dict] = []
        self.validation_summary: dict = {}

    @property
    def expert_order(self) -> tuple[str, ...]:
        return self.network.expert_order

    def _device(self) -> torch.device:
        if self.optimization.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(self.optimization.device)

    def _validate_views(
        self,
        feature_matrix: np.ndarray,
        subjects: Mapping[int, SubjectFusionData],
    ) -> None:
        if np.asarray(feature_matrix).ndim != 2:
            raise ValueError("feature_matrix must be 2D.")
        if int(feature_matrix.shape[1]) != self.network.input_dim:
            raise ValueError(
                f"Feature width is {feature_matrix.shape[1]}, expected {self.network.input_dim}."
            )
        if not subjects:
            raise ValueError("At least one subject is required for fusion training.")
        for subject_id, view in subjects.items():
            if int(subject_id) != view.subject_id:
                raise ValueError("Subject mapping key does not match SubjectFusionData.subject_id.")
            if tuple(view.latent_targets) != self.expert_order:
                raise ValueError(
                    f"Subject {subject_id} expert order {tuple(view.latent_targets)} does not "
                    f"match network order {self.expert_order}."
                )
            if view.voxel_groups.size and (
                int(view.voxel_groups.min()) < 0
                or int(view.voxel_groups.max()) >= self.network.num_regions
            ):
                raise ValueError(f"Subject {subject_id} has out-of-range region indices.")
            if view.stimulus_ids.size and (
                int(view.stimulus_ids.min()) < 0
                or int(view.stimulus_ids.max()) >= int(feature_matrix.shape[0])
            ):
                raise ValueError(f"Subject {subject_id} has out-of-range stimulus IDs.")

    def _fit_standardizers(
        self,
        feature_matrix: np.ndarray,
        subjects: Mapping[int, SubjectFusionData],
    ) -> dict[int, np.ndarray]:
        x_moments = _RunningMoments(self.network.input_dim)
        z_moments = {
            name: _RunningMoments(self.network.expert_dims[name])
            for name in self.expert_order
        }
        voxel_scales: dict[int, np.ndarray] = {}
        for subject_id in sorted(subjects):
            view = subjects[subject_id]
            rows = view.train_indices
            for start in range(0, rows.size, 512):
                chunk = rows[start : start + 512]
                x_moments.update(feature_matrix[view.stimulus_ids[chunk]])
                for name in self.expert_order:
                    z_moments[name].update(view.latent_targets[name][chunk])
            y_moments = _RunningMoments(np.asarray(view.responses).shape[1])
            for start in range(0, rows.size, 256):
                y_moments.update(np.asarray(view.responses[rows[start : start + 256]]))
            _, voxel_scales[subject_id] = y_moments.finalize(constant_value=1.0)

        self.x_mean, self.x_std = x_moments.finalize(constant_value=1.0)
        for name in self.expert_order:
            self.z_mean[name], self.z_std[name] = z_moments[name].finalize(
                constant_value=1.0
            )
        return voxel_scales

    def fit(
        self,
        feature_matrix: np.ndarray,
        subjects: Mapping[int, SubjectFusionData],
    ) -> "FittedMultiExpertEncoder":
        self._validate_views(feature_matrix, subjects)
        random.seed(self.optimization.seed)
        np.random.seed(self.optimization.seed)
        torch.manual_seed(self.optimization.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.optimization.seed)

        voxel_scales = self._fit_standardizers(feature_matrix, subjects)
        assert self.x_mean is not None and self.x_std is not None
        device = self._device()
        model = self.network.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.optimization.learning_rate,
            weight_decay=self.optimization.weight_decay,
        )
        train_sampler = SubjectHomogeneousBatchSampler(
            {subject: view.train_indices for subject, view in subjects.items()},
            batch_size=self.optimization.batch_size,
            shuffle=True,
            seed=self.optimization.seed,
        )
        basis_cache = {
            subject: {
                name: torch.as_tensor(
                    view.transforms[name].basis,
                    dtype=torch.float32,
                    device=device,
                )
                for name in self.expert_order
            }
            for subject, view in subjects.items()
        }
        groups_cache = {
            subject: torch.as_tensor(view.voxel_groups, dtype=torch.long, device=device)
            for subject, view in subjects.items()
        }
        voxel_scale_cache = {
            subject: torch.as_tensor(scale, dtype=torch.float32, device=device)
            for subject, scale in voxel_scales.items()
        }
        x_mean_t = torch.as_tensor(self.x_mean, dtype=torch.float32, device=device)
        x_std_t = torch.as_tensor(self.x_std, dtype=torch.float32, device=device)
        z_mean_t = {
            name: torch.as_tensor(self.z_mean[name], dtype=torch.float32, device=device)
            for name in self.expert_order
        }
        z_std_t = {
            name: torch.as_tensor(self.z_std[name], dtype=torch.float32, device=device)
            for name in self.expert_order
        }

        best_score = -np.inf
        best_state = None
        bad_epochs = 0
        self.history = []
        for epoch in range(1, self.optimization.max_epochs + 1):
            train_sampler.set_epoch(epoch - 1)
            model.train()
            total_loss = 0.0
            total_voxel_loss = 0.0
            total_latent_loss = 0.0
            total_rows = 0
            for keys in train_sampler:
                subject = keys[0][0]
                if any(key[0] != subject for key in keys):
                    raise RuntimeError("SubjectHomogeneousBatchSampler emitted a mixed batch.")
                rows = np.asarray([key[1] for key in keys], dtype=np.int64)
                view = subjects[subject]
                x = torch.as_tensor(
                    np.asarray(feature_matrix[view.stimulus_ids[rows]], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                x = (x - x_mean_t) / x_std_t
                truth = torch.as_tensor(
                    np.asarray(view.responses[rows], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                targets = {
                    name: torch.as_tensor(
                        view.latent_targets[name][rows], dtype=torch.float32, device=device
                    )
                    for name in self.expert_order
                }
                standardized_targets = {
                    name: (targets[name] - z_mean_t[name]) / z_std_t[name]
                    for name in self.expert_order
                }

                optimizer.zero_grad(set_to_none=True)
                output = model(x)
                decoded = decode_and_fuse(
                    {
                        name: output.latents[name] * z_std_t[name] + z_mean_t[name]
                        for name in self.expert_order
                    },
                    basis_cache[subject],
                    output.weights,
                    groups_cache[subject],
                    expert_order=self.expert_order,
                )
                voxel_loss = torch.mean(
                    ((decoded.fused - truth) / voxel_scale_cache[subject]) ** 2
                )
                latent_sum = torch.zeros((), dtype=torch.float32, device=device)
                active_count = torch.zeros((), dtype=torch.float32, device=device)
                for method_index, name in enumerate(self.expert_order):
                    per_sample_mse = torch.mean(
                        (output.latents[name] - standardized_targets[name]) ** 2,
                        dim=1,
                    )
                    active = output.active_mask[:, method_index].to(torch.float32)
                    latent_sum = latent_sum + torch.sum(per_sample_mse * active)
                    active_count = active_count + torch.sum(active)
                latent_loss = latent_sum / torch.clamp(active_count, min=1.0)
                loss = voxel_loss + self.optimization.latent_loss_weight * latent_loss
                loss.backward()
                optimizer.step()

                batch_rows = int(rows.size)
                total_rows += batch_rows
                total_loss += float(loss.detach().cpu()) * batch_rows
                total_voxel_loss += float(voxel_loss.detach().cpu()) * batch_rows
                total_latent_loss += float(latent_loss.detach().cpu()) * batch_rows

            validation = self._validation_score(
                feature_matrix,
                subjects,
                batch_size=self.optimization.batch_size,
            )
            score = float(validation["mean_subject_median_r"])
            history_row = {
                "epoch": epoch,
                "train_loss": total_loss / max(total_rows, 1),
                "train_voxel_loss": total_voxel_loss / max(total_rows, 1),
                "train_latent_loss": total_latent_loss / max(total_rows, 1),
                **validation,
            }
            self.history.append(history_row)
            logger.info(
                "Multi-expert epoch %d: loss=%.5f mean_subject_median_r=%.5f",
                epoch,
                history_row["train_loss"],
                score,
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.optimization.patience:
                    logger.info("Multi-expert early stopping at epoch %d.", epoch)
                    break

        if best_state is None:
            raise RuntimeError("Fusion training did not produce a valid checkpoint.")
        model.load_state_dict(best_state)
        self.validation_summary = {
            "metric": "mean_subject_median_voxel_correlation",
            "best_score": float(best_score),
            "best_epoch": int(max(self.history, key=lambda row: row["mean_subject_median_r"])["epoch"]),
            "n_subjects": len(subjects),
            "n_train_rows": int(sum(view.train_indices.size for view in subjects.values())),
            "n_val_rows": int(sum(view.val_indices.size for view in subjects.values())),
        }
        return self

    def _validation_score(
        self,
        feature_matrix: np.ndarray,
        subjects: Mapping[int, SubjectFusionData],
        *,
        batch_size: int,
    ) -> dict:
        medians: dict[str, float] = {}
        for subject in sorted(subjects):
            view = subjects[subject]
            accumulator = _VoxelCorrelationAccumulator(np.asarray(view.responses).shape[1])
            for start in range(0, view.val_indices.size, batch_size):
                rows = view.val_indices[start : start + batch_size]
                result = self.predict_subject(
                    feature_matrix[view.stimulus_ids[rows]],
                    transforms=view.transforms,
                    voxel_groups=view.voxel_groups,
                    batch_size=batch_size,
                )
                accumulator.update(
                    np.asarray(view.responses[rows], dtype=np.float32),
                    result["fused"],
                )
            medians[str(subject)] = float(np.median(accumulator.correlation()))
        return {
            "mean_subject_median_r": float(np.mean(list(medians.values()))),
            "subject_median_r": medians,
        }

    def _require_fitted(self) -> None:
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Multi-expert encoder is not fitted.")
        if tuple(self.z_mean) != self.expert_order or tuple(self.z_std) != self.expert_order:
            raise RuntimeError("Multi-expert target standardizers are incomplete or reordered.")

    def predict_subject(
        self,
        features: np.ndarray,
        *,
        transforms: Mapping[str, SubjectTransform],
        voxel_groups: np.ndarray,
        batch_size: int = 256,
        active_experts: Sequence[str] | None = None,
        equal_weights: bool = False,
    ) -> dict:
        self._require_fitted()
        if tuple(transforms) != self.expert_order:
            raise ValueError("Prediction transforms do not match the trained expert order.")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.network.input_dim:
            raise ValueError("Prediction features have an incompatible shape.")
        selected = tuple(self.expert_order if active_experts is None else active_experts)
        if not selected or not set(selected).issubset(self.expert_order):
            raise ValueError("active_experts must be a non-empty subset of trained experts.")
        device = self._device()
        model = self.network.to(device)
        model.eval()
        bases = {
            name: torch.as_tensor(transforms[name].basis, dtype=torch.float32, device=device)
            for name in self.expert_order
        }
        groups = torch.as_tensor(voxel_groups, dtype=torch.long, device=device)
        x_mean = torch.as_tensor(self.x_mean, dtype=torch.float32, device=device)
        x_std = torch.as_tensor(self.x_std, dtype=torch.float32, device=device)
        z_mean = {
            name: torch.as_tensor(self.z_mean[name], dtype=torch.float32, device=device)
            for name in self.expert_order
        }
        z_std = {
            name: torch.as_tensor(self.z_std[name], dtype=torch.float32, device=device)
            for name in self.expert_order
        }
        active_template = torch.tensor(
            [name in selected for name in self.expert_order], dtype=torch.bool, device=device
        )
        fused_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        expert_chunks = {name: [] for name in self.expert_order}
        latent_chunks = {name: [] for name in self.expert_order}
        with torch.no_grad():
            for start in range(0, features.shape[0], int(batch_size)):
                x = torch.as_tensor(
                    np.array(
                        features[start : start + int(batch_size)],
                        dtype=np.float32,
                        copy=True,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                x = (x - x_mean) / x_std
                mask = active_template[None, :].expand(x.shape[0], -1)
                output = model(x, method_mask=mask)
                weights = output.weights
                if equal_weights:
                    weights = mask[:, None, :].to(torch.float32)
                    weights = weights.expand(-1, self.network.num_regions, -1)
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                original_latents = {
                    name: output.latents[name] * z_std[name] + z_mean[name]
                    for name in self.expert_order
                }
                decoded = decode_and_fuse(
                    original_latents,
                    bases,
                    weights,
                    groups,
                    expert_order=self.expert_order,
                )
                fused_chunks.append(decoded.fused.cpu().numpy().astype(np.float32))
                weight_chunks.append(weights.cpu().numpy().astype(np.float32))
                for name in self.expert_order:
                    expert_chunks[name].append(
                        decoded.per_expert[name].cpu().numpy().astype(np.float32)
                    )
                    latent_chunks[name].append(
                        original_latents[name].cpu().numpy().astype(np.float32)
                    )
        return {
            "fused": np.concatenate(fused_chunks, axis=0),
            "per_expert": {
                name: np.concatenate(expert_chunks[name], axis=0)
                for name in self.expert_order
            },
            "regional_weights": np.concatenate(weight_chunks, axis=0),
            "latents": {
                name: np.concatenate(latent_chunks[name], axis=0)
                for name in self.expert_order
            },
            "active_experts": list(selected),
            "weight_mode": "equal" if equal_weights else "learned",
        }

    def save(self, path_or_dir: str | Path) -> None:
        self._require_fitted()
        path = Path(path_or_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.network.save(str(path))
        np.savez(
            path / "feature_standardizer.npz",
            x_mean=self.x_mean,
            x_std=self.x_std,
        )
        target_values = {}
        for index, name in enumerate(self.expert_order):
            target_values[f"mean_{index}"] = self.z_mean[name]
            target_values[f"std_{index}"] = self.z_std[name]
        np.savez(path / "target_standardizers.npz", **target_values)
        state = {
            "artifact_version": self.artifact_version,
            "expert_order": list(self.expert_order),
            "optimization": asdict(self.optimization),
            "validation_summary": self.validation_summary,
            "history": self.history,
        }
        (path / "training_state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n"
        )
        if self.history:
            with (path / "train_history.csv").open("w", newline="") as handle:
                flat_rows = [
                    {key: value for key, value in row.items() if key != "subject_median_r"}
                    for row in self.history
                ]
                writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
                writer.writeheader()
                writer.writerows(flat_rows)

    @classmethod
    def load(
        cls,
        path_or_dir: str | Path,
        *,
        expected_expert_order: Sequence[str] | None = None,
        expected_expert_dims: Mapping[str, int] | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "FittedMultiExpertEncoder":
        path = Path(path_or_dir)
        network = MultiExpertFusionNetwork.load(str(path), map_location=map_location)
        state = json.loads((path / "training_state.json").read_text())
        if int(state.get("artifact_version", -1)) != cls.artifact_version:
            raise ValueError("Unsupported fitted multi-expert encoder artifact version.")
        order = tuple(str(name) for name in state["expert_order"])
        if order != network.expert_order:
            raise ValueError("Encoder training state and network expert order differ.")
        if expected_expert_order is not None and order != tuple(expected_expert_order):
            raise ValueError("Encoder expert order does not match the model manifest.")
        if expected_expert_dims is not None:
            actual_dims = {name: int(network.expert_dims[name]) for name in order}
            expected_dims = {name: int(expected_expert_dims[name]) for name in order}
            if actual_dims != expected_dims:
                raise ValueError("Encoder expert dimensions do not match the model manifest.")
        fitted = cls(
            network=network,
            optimization=FusionOptimizationConfig(**state["optimization"]),
        )
        feature_std = np.load(path / "feature_standardizer.npz", allow_pickle=False)
        fitted.x_mean = np.asarray(feature_std["x_mean"], dtype=np.float32)
        fitted.x_std = np.asarray(feature_std["x_std"], dtype=np.float32)
        target_std = np.load(path / "target_standardizers.npz", allow_pickle=False)
        for index, name in enumerate(order):
            fitted.z_mean[name] = np.asarray(target_std[f"mean_{index}"], dtype=np.float32)
            fitted.z_std[name] = np.asarray(target_std[f"std_{index}"], dtype=np.float32)
        fitted.validation_summary = dict(state.get("validation_summary", {}))
        fitted.history = list(state.get("history", []))
        fitted._require_fitted()
        return fitted
