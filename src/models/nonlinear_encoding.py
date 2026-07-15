"""
PyTorch nonlinear encoder from static image features to shared-space responses.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "The static TRIBE transformer requires PyTorch."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _pearson_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_float32(y_true)
    y_pred = _as_float32(y_pred)
    true_c = y_true - y_true.mean(axis=0, keepdims=True)
    pred_c = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((true_c**2).sum(axis=0) * (pred_c**2).sum(axis=0))
    valid = denom > 1e-8
    if not np.any(valid):
        return 0.0
    corr = (true_c[:, valid] * pred_c[:, valid]).sum(axis=0) / denom[valid]
    return float(np.mean(corr))


@dataclass
class StaticTransformerConfig:
    architecture: str = "tribe_static_transformer"
    transformer_d_model: int = 384
    transformer_layers: int = 8
    transformer_heads: int = 8
    transformer_ff_multiplier: float = 4.0
    dropout: float = 0.10
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 512
    max_epochs: int = 200
    patience: int = 20
    val_fraction: float = 0.10
    val_metric: str = "pearson"
    validation_mode: str = "stimulus_grouped"
    standardize_targets: bool = True
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self) -> None:
        aliases = {
            "tribe": "tribe_static_transformer",
            "tribe_static": "tribe_static_transformer",
            "tribe_static_transformer": "tribe_static_transformer",
        }
        architecture = aliases.get(str(self.architecture).strip().lower())
        if architecture is None:
            raise ValueError(
                f"Unsupported architecture={self.architecture!r}; expected "
                "'tribe_static_transformer'."
            )
        self.architecture = architecture
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.transformer_d_model < 1 or self.transformer_layers < 1:
            raise ValueError("TRIBE transformer dimensions and layer count must be positive.")
        if self.transformer_heads < 1:
            raise ValueError("TRIBE transformer_heads must be positive.")
        if self.transformer_d_model % self.transformer_heads != 0:
            raise ValueError("TRIBE transformer_d_model must be divisible by transformer_heads.")
        if self.transformer_ff_multiplier <= 0:
            raise ValueError("TRIBE transformer_ff_multiplier must be positive.")

    @classmethod
    def from_config(cls, cfg: dict | None, seed: int = 42) -> "StaticTransformerConfig":
        cfg = cfg or {}
        transformer = cfg.get("transformer", {}) or {}
        return cls(
            architecture=str(cfg.get("architecture", "tribe_static_transformer")),
            transformer_d_model=int(
                cfg.get("transformer_d_model", transformer.get("d_model", 384))
            ),
            transformer_layers=int(
                cfg.get("transformer_layers", transformer.get("n_layers", 8))
            ),
            transformer_heads=int(
                cfg.get("transformer_heads", transformer.get("n_heads", 8))
            ),
            transformer_ff_multiplier=float(
                cfg.get("transformer_ff_multiplier", transformer.get("ff_multiplier", 4.0))
            ),
            dropout=float(cfg.get("dropout", 0.10)),
            learning_rate=float(cfg.get("learning_rate", 1.0e-3)),
            weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
            batch_size=int(cfg.get("batch_size", 512)),
            max_epochs=int(cfg.get("max_epochs", 200)),
            patience=int(cfg.get("patience", 20)),
            val_fraction=float(cfg.get("val_fraction", 0.10)),
            val_metric=str(cfg.get("val_metric", "pearson")),
            validation_mode=str(cfg.get("validation_mode", "stimulus_grouped")),
            standardize_targets=bool(cfg.get("standardize_targets", True)),
            device=str(cfg.get("device", "cuda")),
            seed=int(cfg.get("seed", seed)),
        )


class _StaticTribeTransformer:
    """Static-image analogue of TRIBE's projected-stream transformer encoder.

    TRIBE v2 exchanges information over temporal tokens. Static NSD examples
    have no temporal axis, so named feature streams become tokens and a learned
    CLS token aggregates them before the shared-space readout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        feature_slices: dict[str, tuple[int, int]],
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_multiplier: float,
        dropout: float,
    ):
        torch, nn, _, _ = _require_torch()
        stream_specs = tuple(
            (str(name), int(bounds[0]), int(bounds[1]))
            for name, bounds in feature_slices.items()
        )
        if not stream_specs:
            stream_specs = (("features", 0, int(input_dim)),)
        expected_start = 0
        for name, start, end in stream_specs:
            if start != expected_start or end <= start:
                raise ValueError(
                    f"TRIBE feature slices must be contiguous; {name!r} has ({start}, {end}) "
                    f"after expected start {expected_start}."
                )
            expected_start = end
        if expected_start != int(input_dim):
            raise ValueError(
                f"TRIBE feature slices cover {expected_start} columns, expected {input_dim}."
            )

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.stream_names = [name for name, _, _ in stream_specs]
                self.stream_slices = [(start, end) for _, start, end in stream_specs]
                self.stream_projections = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(end - start, d_model, bias=False),
                            nn.LayerNorm(d_model),
                        )
                        for _, start, end in stream_specs
                    ]
                )
                self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
                self.stream_embeddings = nn.Parameter(
                    torch.zeros(1, len(stream_specs) + 1, d_model)
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
                self.backbone = nn.TransformerEncoder(
                    layer,
                    num_layers=n_layers,
                    norm=nn.LayerNorm(d_model),
                    enable_nested_tensor=False,
                )
                for encoder_layer in self.backbone.layers:
                    for parameter in encoder_layer.parameters():
                        if parameter.ndim > 1:
                            nn.init.xavier_uniform_(parameter)
                nn.init.normal_(self.cls_token, std=0.02)
                nn.init.normal_(self.stream_embeddings, std=0.02)

                self.subject_head = nn.Linear(d_model, output_dim)

            def encode(self, x):
                stream_tokens = [
                    projection(x[:, start:end])
                    for projection, (start, end) in zip(
                        self.stream_projections, self.stream_slices
                    )
                ]
                tokens = torch.stack(stream_tokens, dim=1)
                cls = self.cls_token.expand(x.shape[0], -1, -1)
                tokens = torch.cat([cls, tokens], dim=1) + self.stream_embeddings
                return self.backbone(tokens)[:, 0]

            def forward(self, x):
                return self.subject_head(self.encode(x))

        self.module = Module()


class StaticTransformerEncoder:
    """Frozen static TRIBE transformer from CLIP features to shared responses."""

    encoder_type = "tribe_static_transformer"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: StaticTransformerConfig | None = None,
        feature_slices: dict[str, tuple[int, int]] | None = None,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.config = config or StaticTransformerConfig()
        self.feature_slices = dict(feature_slices or {})
        self.architecture = self.config.architecture
        self.encoder_type = self.architecture
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.z_mean: np.ndarray | None = None
        self.z_std: np.ndarray | None = None
        self.validation_summary: dict[str, int | str] = {}
        self.history: list[dict[str, float | int]] = []
        self._model = None

    def _device(self):
        torch, _, _, _ = _require_torch()
        if self.config.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _init_model(self):
        if self._model is None:
            wrapper = _StaticTribeTransformer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                feature_slices=self.feature_slices,
                d_model=self.config.transformer_d_model,
                n_layers=self.config.transformer_layers,
                n_heads=self.config.transformer_heads,
                ff_multiplier=self.config.transformer_ff_multiplier,
                dropout=self.config.dropout,
            )
            self._model = wrapper.module
        return self._model

    def _standardize_fit(self, X: np.ndarray, fit_rows: np.ndarray | None = None) -> np.ndarray:
        fit_x = X if fit_rows is None else X[fit_rows]
        self.x_mean = fit_x.mean(axis=0).astype(np.float32)
        self.x_std = fit_x.std(axis=0).astype(np.float32)
        self.x_std[self.x_std < 1e-8] = 1e-8
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def _standardize_x(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Encoder is not fitted.")
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def _standardize_targets_fit(self, Z: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
        if self.config.standardize_targets:
            self.z_mean = Z[fit_rows].mean(axis=0).astype(np.float32)
            self.z_std = Z[fit_rows].std(axis=0).astype(np.float32)
            self.z_std[self.z_std < 1e-8] = 1.0
        else:
            self.z_mean = np.zeros((Z.shape[1],), dtype=np.float32)
            self.z_std = np.ones((Z.shape[1],), dtype=np.float32)
        return ((Z - self.z_mean) / self.z_std).astype(np.float32)

    def _standardize_targets(self, Z: np.ndarray) -> np.ndarray:
        if self.z_mean is None or self.z_std is None:
            raise RuntimeError("Target standardizer is not fitted.")
        return ((_as_float32(Z) - self.z_mean) / self.z_std).astype(np.float32)

    def _inverse_targets(self, Zs: np.ndarray) -> np.ndarray:
        if self.z_mean is None or self.z_std is None:
            raise RuntimeError("Target standardizer is not fitted.")
        return (_as_float32(Zs) * self.z_std + self.z_mean).astype(np.float32)

    def _validation_indices(
        self,
        n_rows: int,
        sample_groups: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.config.seed)
        if n_rows <= 1:
            only = np.arange(n_rows, dtype=np.int64)
            return only, np.empty((0,), dtype=np.int64)

        if self.config.validation_mode == "stimulus_grouped" and sample_groups is not None:
            groups = np.asarray(sample_groups).ravel()
            if groups.shape[0] != n_rows:
                raise ValueError(
                    f"sample_groups has {groups.shape[0]} rows, expected {n_rows}."
                )
            unique_groups = np.unique(groups)
            if unique_groups.size < 2:
                raise ValueError("stimulus_grouped validation requires at least two unique groups.")
            shuffled_groups = unique_groups[rng.permutation(unique_groups.size)]
            n_val_groups = int(round(unique_groups.size * self.config.val_fraction))
            n_val_groups = max(1, min(n_val_groups, unique_groups.size - 1))
            val_groups = shuffled_groups[:n_val_groups]
            val_mask = np.isin(groups, val_groups, assume_unique=False)
            val_idx = np.flatnonzero(val_mask).astype(np.int64, copy=False)
            train_idx = np.flatnonzero(~val_mask).astype(np.int64, copy=False)
            self.validation_summary = {
                "mode": "stimulus_grouped",
                "n_train_rows": int(train_idx.size),
                "n_val_rows": int(val_idx.size),
                "n_train_groups": int(unique_groups.size - n_val_groups),
                "n_val_groups": int(n_val_groups),
            }
            return train_idx, val_idx

        if self.config.validation_mode == "stimulus_grouped":
            logger.warning(
                "stimulus_grouped validation requested without sample_groups; using random rows."
            )
        order = rng.permutation(n_rows)
        n_val = int(round(n_rows * self.config.val_fraction))
        n_val = max(1, min(n_val, n_rows - 1))
        val_idx = order[:n_val].astype(np.int64, copy=False)
        train_idx = order[n_val:].astype(np.int64, copy=False)
        self.validation_summary = {
            "mode": "random_rows",
            "n_train_rows": int(train_idx.size),
            "n_val_rows": int(val_idx.size),
            "n_train_groups": 0,
            "n_val_groups": 0,
        }
        return train_idx, val_idx

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        sample_groups: np.ndarray | None = None,
    ) -> "StaticTransformerEncoder":
        torch, nn, DataLoader, TensorDataset = _require_torch()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        X = _as_float32(X)
        Z = _as_float32(Z)
        if X.ndim != 2 or Z.ndim != 2:
            raise ValueError(f"X and Z must be 2D, got {X.shape} and {Z.shape}.")
        if X.shape[0] != Z.shape[0]:
            raise ValueError(f"X/Z row mismatch: {X.shape[0]} vs {Z.shape[0]}.")

        train_idx, val_idx = self._validation_indices(X.shape[0], sample_groups)
        Xs = self._standardize_fit(X, fit_rows=train_idx)
        Zs = self._standardize_targets_fit(Z, fit_rows=train_idx)
        n_val = int(val_idx.size)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xs[train_idx]), torch.from_numpy(Zs[train_idx])),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        device = self._device()
        model = self._init_model().to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        best_score = -np.inf
        best_state = None
        bad_epochs = 0

        for epoch in range(1, self.config.max_epochs + 1):
            model.train()
            train_loss = 0.0
            n_train = 0
            for xb, zb in train_loader:
                xb = xb.to(device)
                zb = zb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, zb)
                loss.backward()
                opt.step()
                train_loss += float(loss.item()) * int(xb.shape[0])
                n_train += int(xb.shape[0])

            val_mse = np.nan
            val_pearson = np.nan
            if n_val:
                pred_val = self._predict_standardized(
                    Xs[val_idx],
                    batch_size=self.config.batch_size,
                )
                val_mse = float(np.mean((Zs[val_idx] - pred_val) ** 2))
                val_pearson = _pearson_np(Zs[val_idx], pred_val)
                score = val_pearson if self.config.val_metric == "pearson" else -val_mse
            else:
                score = -train_loss / max(1, n_train)

            row = {
                "epoch": epoch,
                "train_loss": train_loss / max(1, n_train),
                "val_mse": val_mse,
                "val_pearson": val_pearson,
            }
            self.history.append(row)
            if score > best_score:
                best_score = float(score)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.patience:
                    logger.info("%s early stopping at epoch %d", self.architecture, epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        logger.info(
            "%s encoder fitted: epochs=%d best_%s=%.4f",
            self.architecture,
            len(self.history),
            self.config.val_metric,
            best_score,
        )
        return self

    def _predict_standardized(
        self,
        Xs: np.ndarray,
        batch_size: int = 4096,
    ) -> np.ndarray:
        torch, _, DataLoader, TensorDataset = _require_torch()
        model = self._init_model().to(self._device())
        model.eval()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(_as_float32(Xs))),
            batch_size=batch_size,
            shuffle=False,
        )
        chunks = []
        with torch.no_grad():
            for batch in loader:
                xb = batch[0].to(self._device())
                chunks.append(model(xb).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def predict(
        self,
        X_new: np.ndarray,
    ) -> np.ndarray:
        Xs = self._standardize_x(_as_float32(X_new))
        pred_s = self._predict_standardized(Xs, batch_size=4096)
        return self._inverse_targets(pred_s)

    def predict_voxels(
        self,
        X_new: np.ndarray,
        P: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        Z_hat = self.predict(X_new)
        return (Z_hat @ R.T @ P.T).astype(np.float32)

    def save(self, path_or_dir: str) -> None:
        torch, _, _, _ = _require_torch()
        os.makedirs(path_or_dir, exist_ok=True)
        model = self._init_model().to("cpu")
        torch.save(model.state_dict(), os.path.join(path_or_dir, "model.pt"))
        np.savez(
            os.path.join(path_or_dir, "feature_standardizer.npz"),
            x_mean=self.x_mean,
            x_std=self.x_std,
        )
        if self.z_mean is None or self.z_std is None:
            raise RuntimeError("Cannot save nonlinear encoder without a fitted target standardizer.")
        np.savez(
            os.path.join(path_or_dir, "target_standardizer.npz"),
            z_mean=self.z_mean,
            z_std=self.z_std,
        )
        metadata = {
            "artifact_version": 4,
            "encoder_type": self.encoder_type,
            "architecture": self.architecture,
            "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "feature_slices": self.feature_slices,
            "validation_summary": self.validation_summary,
            "config": self.config.__dict__,
        }
        with open(os.path.join(path_or_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        if self.history:
            with open(os.path.join(path_or_dir, "train_history.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
                writer.writeheader()
                writer.writerows(self.history)
        logger.info("Saved %s encoder to %s", self.architecture, path_or_dir)

    @classmethod
    def load(cls, path_or_dir: str) -> "StaticTransformerEncoder":
        torch, _, _, _ = _require_torch()
        with open(os.path.join(path_or_dir, "metadata.json")) as f:
            metadata = json.load(f)
        cfg = StaticTransformerConfig.from_config(metadata.get("config", {}))
        encoder = cls(
            input_dim=int(metadata["input_dim"]),
            output_dim=int(metadata["output_dim"]),
            config=cfg,
            feature_slices={
                str(k): tuple(int(x) for x in v)
                for k, v in (metadata.get("feature_slices") or {}).items()
            },
        )
        encoder.validation_summary = dict(metadata.get("validation_summary", {}) or {})
        std = np.load(os.path.join(path_or_dir, "feature_standardizer.npz"))
        encoder.x_mean = std["x_mean"]
        encoder.x_std = std["x_std"]
        target_std_path = os.path.join(path_or_dir, "target_standardizer.npz")
        if os.path.exists(target_std_path):
            target_std = np.load(target_std_path)
            encoder.z_mean = target_std["z_mean"]
            encoder.z_std = target_std["z_std"]
        else:
            encoder.z_mean = np.zeros((encoder.output_dim,), dtype=np.float32)
            encoder.z_std = np.ones((encoder.output_dim,), dtype=np.float32)
        model = encoder._init_model()
        model.load_state_dict(torch.load(os.path.join(path_or_dir, "model.pt"), map_location="cpu"))
        model.eval()
        return encoder
