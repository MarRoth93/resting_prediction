"""Leakage-controlled NSD validation for the shared Schaefer-400 VDVAE decoder."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import cho_factor, cho_solve

from src.pipelines.multiexpert_artifacts import json_fingerprint
from src.pipelines.reconstruct_for_schaefer400_vdvae import (
    N_PARCELS,
    _load_target_stimulus_ids,
    decode_vdvae_images,
    standardize_parcel_patterns,
)
from src.pipelines.benchmark_reconstructions_vdvae_vd import _load_vdvae_model


logger = logging.getLogger(__name__)
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class SubjectRows:
    subject: int
    responses: np.ndarray
    stimulus_ids: np.ndarray
    target_rows: np.ndarray


class _MetricAccumulator:
    def __init__(self, n_rows: int):
        self.n_rows = int(n_rows)
        self.width = 0
        self.row_true_sum = np.zeros(n_rows, dtype=np.float64)
        self.row_pred_sum = np.zeros(n_rows, dtype=np.float64)
        self.row_true_sq = np.zeros(n_rows, dtype=np.float64)
        self.row_pred_sq = np.zeros(n_rows, dtype=np.float64)
        self.row_cross = np.zeros(n_rows, dtype=np.float64)
        self.target_r: list[np.ndarray] = []
        self.target_r2: list[np.ndarray] = []
        self.true_sum = 0.0
        self.pred_sum = 0.0
        self.true_sq_sum = 0.0
        self.pred_sq_sum = 0.0

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        truth = np.asarray(truth, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if truth.shape != prediction.shape or truth.shape[0] != self.n_rows:
            raise ValueError("Validation truth and prediction chunks do not align.")
        self.width += int(truth.shape[1])
        self.row_true_sum += truth.sum(axis=1)
        self.row_pred_sum += prediction.sum(axis=1)
        self.row_true_sq += np.square(truth).sum(axis=1)
        self.row_pred_sq += np.square(prediction).sum(axis=1)
        self.row_cross += (truth * prediction).sum(axis=1)
        self.true_sum += float(truth.sum())
        self.pred_sum += float(prediction.sum())
        self.true_sq_sum += float(np.square(truth).sum())
        self.pred_sq_sum += float(np.square(prediction).sum())

        truth_centered = truth - truth.mean(axis=0, keepdims=True)
        pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
        numerator = (truth_centered * pred_centered).sum(axis=0)
        denominator = np.sqrt(
            np.square(truth_centered).sum(axis=0)
            * np.square(pred_centered).sum(axis=0)
        )
        correlation = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-10,
        )
        total_ss = np.square(truth_centered).sum(axis=0)
        residual_ss = np.square(truth - prediction).sum(axis=0)
        r2 = np.divide(
            residual_ss,
            total_ss,
            out=np.zeros_like(residual_ss),
            where=total_ss > 1e-10,
        )
        r2 = 1.0 - r2
        r2[(total_ss <= 1e-10) & (residual_ss > 1e-10)] = 0.0
        self.target_r.append(correlation)
        self.target_r2.append(r2)

    def finalize(self) -> dict[str, float]:
        if self.width < 1:
            raise ValueError("No validation dimensions were accumulated.")
        width = float(self.width)
        row_cov = self.row_cross - self.row_true_sum * self.row_pred_sum / width
        row_true_var = self.row_true_sq - np.square(self.row_true_sum) / width
        row_pred_var = self.row_pred_sq - np.square(self.row_pred_sum) / width
        row_denominator = np.sqrt(np.maximum(row_true_var * row_pred_var, 0.0))
        row_r = np.divide(
            row_cov,
            row_denominator,
            out=np.zeros_like(row_cov),
            where=row_denominator > 1e-10,
        )
        target_r = np.concatenate(self.target_r)
        target_r2 = np.concatenate(self.target_r2)
        count = float(self.n_rows * self.width)
        true_mean = self.true_sum / count
        pred_mean = self.pred_sum / count
        return {
            "r2_vs_true_eval": float(target_r2.mean()),
            "mean_target_r_vs_true_eval": float(target_r.mean()),
            "median_target_r_vs_true_eval": float(np.median(target_r)),
            "mean_row_r_vs_true_eval": float(row_r.mean()),
            "median_row_r_vs_true_eval": float(np.median(row_r)),
            "true_eval_mean": true_mean,
            "pred_eval_mean": pred_mean,
            "true_eval_std": float(
                np.sqrt(max(self.true_sq_sum / count - true_mean**2, 0.0))
            ),
            "pred_eval_std": float(
                np.sqrt(max(self.pred_sq_sum / count - pred_mean**2, 0.0))
            ),
        }


def build_validation_split(
    stimulus_ids: np.ndarray,
    *,
    subjects: list[int],
    seed: int,
    final_fraction: float,
) -> dict:
    stimulus_ids = np.asarray(stimulus_ids, dtype=np.int64)
    if stimulus_ids.ndim != 1 or np.unique(stimulus_ids).size != stimulus_ids.size:
        raise ValueError("Validation stimulus IDs must be a unique vector.")
    if len(subjects) < 2 or not 0.05 <= float(final_fraction) <= 0.5:
        raise ValueError("Validation requires at least two subjects and a sensible final fraction.")
    rng = np.random.RandomState(int(seed))
    shuffled = stimulus_ids[rng.permutation(stimulus_ids.size)]
    n_final = max(1, int(round(stimulus_ids.size * float(final_fraction))))
    final_ids = np.sort(shuffled[:n_final])
    dev_ids = shuffled[n_final:]
    tuning_parts = [np.sort(part) for part in np.array_split(dev_ids, len(subjects))]
    return {
        "seed": int(seed),
        "subjects": [int(value) for value in subjects],
        "final_fraction": float(final_fraction),
        "final_stimulus_ids": final_ids.astype(int).tolist(),
        "tuning_folds": {
            str(subject): part.astype(int).tolist()
            for subject, part in zip(subjects, tuning_parts, strict=True)
        },
        "development_stimulus_ids": np.sort(dev_ids).astype(int).tolist(),
    }


def _load_subject_rows(
    *,
    nsd_data_root: Path,
    feature_dir: Path,
    subjects: list[int],
) -> tuple[dict[int, SubjectRows], np.ndarray]:
    target_ids = _load_target_stimulus_ids(feature_dir)
    target_lookup = {int(value): row for row, value in enumerate(target_ids.tolist())}
    result: dict[int, SubjectRows] = {}
    for subject in subjects:
        subject_dir = nsd_data_root / f"subj{subject:02d}"
        responses = np.load(subject_dir / "test_fmri.npy", mmap_mode="r")
        stimulus_ids = np.asarray(np.load(subject_dir / "test_stim_idx.npy"), dtype=np.int64)
        if responses.shape != (stimulus_ids.size, N_PARCELS):
            raise ValueError(f"NSD subject {subject} response/ID shape mismatch.")
        mapped = np.asarray([target_lookup.get(int(value), -1) for value in stimulus_ids])
        keep = mapped >= 0
        result[subject] = SubjectRows(
            subject=subject,
            responses=np.asarray(responses[keep], dtype=np.float32),
            stimulus_ids=stimulus_ids[keep],
            target_rows=mapped[keep].astype(np.int64),
        )
    return result, target_ids


def _select_design(rows: SubjectRows, allowed_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(rows.stimulus_ids, np.asarray(allowed_ids, dtype=np.int64))
    if int(keep.sum()) < 2:
        raise ValueError(f"NSD subject {rows.subject} has too few selected validation rows.")
    return standardize_parcel_patterns(rows.responses[keep]), rows.target_rows[keep]


def _fold_matrices(
    subject_rows: dict[int, SubjectRows],
    *,
    heldout_subject: int,
    training_ids: np.ndarray,
    evaluation_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_design: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    for subject, rows in sorted(subject_rows.items()):
        if subject == heldout_subject:
            continue
        design, target_rows = _select_design(rows, training_ids)
        train_design.append(design)
        train_targets.append(target_rows)
    val_design, val_targets = _select_design(subject_rows[heldout_subject], evaluation_ids)
    val_keep = np.isin(subject_rows[heldout_subject].stimulus_ids, evaluation_ids)
    val_stimulus_ids = subject_rows[heldout_subject].stimulus_ids[val_keep]
    return (
        np.concatenate(train_design, axis=0),
        np.concatenate(train_targets, axis=0),
        val_design,
        val_targets,
        val_stimulus_ids,
    )


def evaluate_ridge_fold(
    *,
    train_design: np.ndarray,
    train_target_rows: np.ndarray,
    val_design: np.ndarray,
    val_target_rows: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    chunk_size: int,
    prediction_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    train_design = np.asarray(train_design, dtype=np.float64)
    val_design = np.asarray(val_design, dtype=np.float64)
    x_mean = train_design.mean(axis=0)
    centered = train_design - x_mean
    gram = centered.T @ centered
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    factor = cho_factor(gram, lower=True, check_finite=False)
    ridge_metrics = _MetricAccumulator(val_design.shape[0])
    mean_metrics = _MetricAccumulator(val_design.shape[0])

    temporary: Path | None = None
    prediction = None
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_path.with_name(f".{prediction_path.name}.part")
        temporary.unlink(missing_ok=True)
        prediction = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(val_design.shape[0], targets.shape[1]),
        )
    try:
        for start in range(0, targets.shape[1], int(chunk_size)):
            stop = min(start + int(chunk_size), targets.shape[1])
            y_train = np.asarray(targets[train_target_rows, start:stop], dtype=np.float64)
            y_mean = y_train.mean(axis=0)
            weights = cho_solve(
                factor,
                centered.T @ (y_train - y_mean),
                check_finite=False,
            )
            predicted = (val_design - x_mean) @ weights + y_mean
            truth = np.asarray(targets[val_target_rows, start:stop], dtype=np.float64)
            baseline = np.broadcast_to(y_mean, truth.shape)
            ridge_metrics.update(truth, predicted)
            mean_metrics.update(truth, baseline)
            if prediction is not None:
                prediction[:, start:stop] = predicted.astype(np.float32)
        if prediction is not None and temporary is not None and prediction_path is not None:
            prediction.flush()
            del prediction
            os.replace(temporary, prediction_path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return ridge_metrics.finalize(), mean_metrics.finalize()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validation_contract(
    *,
    nsd_data_root: Path,
    feature_dir: Path,
    subjects: list[int],
    alphas: list[float],
    seed: int,
    final_fraction: float,
    target_ids: np.ndarray,
) -> dict:
    files = {}
    for subject in subjects:
        for name in ("test_fmri.npy", "test_stim_idx.npy"):
            path = nsd_data_root / f"subj{subject:02d}" / name
            stat = path.stat()
            files[f"subj{subject:02d}/{name}"] = {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
    feature_bundle = feature_dir / "vdvae_features.npz"
    feature_stat = feature_bundle.stat()
    contract = {
        "artifact_version": ARTIFACT_VERSION,
        "subjects": subjects,
        "alphas": alphas,
        "seed": int(seed),
        "final_fraction": float(final_fraction),
        "target_stimulus_ids": np.asarray(target_ids, dtype=int).tolist(),
        "feature_bundle": {
            "path": str(feature_bundle.resolve()),
            "size": int(feature_stat.st_size),
            "mtime_ns": int(feature_stat.st_mtime_ns),
        },
        "files": files,
    }
    contract["fingerprint"] = json_fingerprint(contract)
    return contract


def validate_decoder(
    *,
    nsd_data_root: Path,
    feature_dir: Path,
    output_dir: Path,
    subjects: list[int],
    alphas: list[float],
    seed: int,
    final_fraction: float,
    chunk_size: int,
) -> dict:
    subject_rows, target_ids = _load_subject_rows(
        nsd_data_root=nsd_data_root,
        feature_dir=feature_dir,
        subjects=subjects,
    )
    contract = _validation_contract(
        nsd_data_root=nsd_data_root,
        feature_dir=feature_dir,
        subjects=subjects,
        alphas=alphas,
        seed=seed,
        final_fraction=final_fraction,
        target_ids=target_ids,
    )
    split = build_validation_split(
        target_ids,
        subjects=subjects,
        seed=seed,
        final_fraction=final_fraction,
    )
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("contract", {}).get("fingerprint") != contract["fingerprint"]:
            raise RuntimeError("Existing decoder validation uses a different input contract.")
    else:
        _write_json_atomic(manifest_path, {"contract": contract, "split": split})

    vdvae_path = feature_dir / "vdvae_features.npz"
    with np.load(vdvae_path) as bundle:
        targets = np.asarray(bundle["test_latents"], dtype=np.float32)
        development_ids = np.asarray(split["development_stimulus_ids"], dtype=np.int64)
        tuning_rows = []
        for alpha in alphas:
            for subject in subjects:
                metric_path = output_dir / "tuning" / f"alpha_{alpha:g}_subj{subject:02d}.json"
                if metric_path.exists():
                    tuning_rows.append(json.loads(metric_path.read_text()))
                    continue
                evaluation_ids = np.asarray(
                    split["tuning_folds"][str(subject)], dtype=np.int64
                )
                training_ids = np.setdiff1d(development_ids, evaluation_ids)
                matrices = _fold_matrices(
                    subject_rows,
                    heldout_subject=subject,
                    training_ids=training_ids,
                    evaluation_ids=evaluation_ids,
                )
                logger.info(
                    "Tuning alpha=%g with held-out NSD subject %d (%d rows)",
                    alpha,
                    subject,
                    matrices[2].shape[0],
                )
                ridge, baseline = evaluate_ridge_fold(
                    train_design=matrices[0],
                    train_target_rows=matrices[1],
                    val_design=matrices[2],
                    val_target_rows=matrices[3],
                    targets=targets,
                    alpha=alpha,
                    chunk_size=chunk_size,
                )
                row = {
                    "alpha": float(alpha),
                    "heldout_subject": int(subject),
                    "evaluation_rows": int(matrices[2].shape[0]),
                    "ridge": ridge,
                    "mean_latent_baseline": baseline,
                }
                _write_json_atomic(metric_path, row)
                tuning_rows.append(row)

        alpha_scores = {
            float(alpha): float(
                np.average(
                    [
                        row["ridge"]["mean_row_r_vs_true_eval"]
                        for row in tuning_rows
                        if float(row["alpha"]) == float(alpha)
                    ],
                    weights=[
                        row["evaluation_rows"]
                        for row in tuning_rows
                        if float(row["alpha"]) == float(alpha)
                    ],
                )
            )
            for alpha in alphas
        }
        selected_alpha = max(alpha_scores, key=lambda value: (alpha_scores[value], -value))
        selection = {
            "selection_metric": "weighted_mean_row_r_vs_true_eval",
            "alpha_scores": {str(key): value for key, value in alpha_scores.items()},
            "selected_alpha": float(selected_alpha),
        }
        _write_json_atomic(output_dir / "selected_alpha.json", selection)

        final_ids = np.asarray(split["final_stimulus_ids"], dtype=np.int64)
        final_rows = []
        for subject in subjects:
            subject_dir = output_dir / "final" / f"subj{subject:02d}"
            prediction_path = subject_dir / "predicted_vdvae_latents.npy"
            stimulus_path = subject_dir / "stimulus_ids.npy"
            metrics_path = subject_dir / "metrics.json"
            existing_count = sum(path.exists() for path in (prediction_path, stimulus_path, metrics_path))
            if existing_count == 3:
                final_rows.append(json.loads(metrics_path.read_text()))
                continue
            if existing_count:
                raise RuntimeError(f"Partial final validation output exists: {subject_dir}")
            matrices = _fold_matrices(
                subject_rows,
                heldout_subject=subject,
                training_ids=development_ids,
                evaluation_ids=final_ids,
            )
            logger.info(
                "Final test for held-out NSD subject %d (%d untouched rows)",
                subject,
                matrices[2].shape[0],
            )
            ridge, baseline = evaluate_ridge_fold(
                train_design=matrices[0],
                train_target_rows=matrices[1],
                val_design=matrices[2],
                val_target_rows=matrices[3],
                targets=targets,
                alpha=selected_alpha,
                chunk_size=chunk_size,
                prediction_path=prediction_path,
            )
            np.save(stimulus_path, matrices[4])
            row = {
                "heldout_subject": int(subject),
                "selected_alpha": float(selected_alpha),
                "evaluation_rows": int(matrices[2].shape[0]),
                "ridge": ridge,
                "mean_latent_baseline": baseline,
            }
            _write_json_atomic(metrics_path, row)
            final_rows.append(row)

    summary = {
        "status": "complete",
        "contract_fingerprint": contract["fingerprint"],
        "selected_alpha": float(selected_alpha),
        "tuning": selection,
        "final_subjects": final_rows,
        "final_weighted_mean_row_r": float(
            np.average(
                [row["ridge"]["mean_row_r_vs_true_eval"] for row in final_rows],
                weights=[row["evaluation_rows"] for row in final_rows],
            )
        ),
        "final_weighted_baseline_mean_row_r": float(
            np.average(
                [
                    row["mean_latent_baseline"]["mean_row_r_vs_true_eval"]
                    for row in final_rows
                ],
                weights=[row["evaluation_rows"] for row in final_rows],
            )
        ),
    }
    _write_json_atomic(output_dir / "validation_summary.json", summary)
    return summary


def decode_validation(
    *,
    output_dir: Path,
    feature_dir: Path,
    recon_model_root: Path,
    subjects: list[int],
    batch_size: int,
    device: str,
) -> None:
    if device != "cuda":
        raise ValueError("The retained VDVAE implementation is CUDA-only.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to decode validation images.")
    summary_path = output_dir / "validation_summary.json"
    if not summary_path.exists() or json.loads(summary_path.read_text()).get("status") != "complete":
        raise RuntimeError("Complete decoder validation is required before image decoding.")
    ref_latent = np.load(feature_dir / "ref_latents.npz", allow_pickle=True)["ref_latent"]
    ema_vae = _load_vdvae_model(recon_model_root)
    for subject in subjects:
        subject_label = f"subj{subject:02d}"
        subject_dir = output_dir / "final" / subject_label
        decode_vdvae_images(
            ema_vae=ema_vae,
            latent_path=subject_dir / "predicted_vdvae_latents.npy",
            ref_latent=ref_latent,
            subject_label=subject_label,
            output_root=output_dir / "final",
            stimulus_ids=np.load(subject_dir / "stimulus_ids.npy"),
            batch_size=batch_size,
            device=device,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "validate", "decode", "selected-alpha"])
    parser.add_argument("--study-config", default="config_for_assessor_study.yaml")
    parser.add_argument("--nsd-data-root")
    parser.add_argument("--feature-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--recon-model-root", default="third_party")
    parser.add_argument("--subjects", nargs="+", type=int)
    parser.add_argument("--alphas", nargs="+", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--final-fraction", type=float)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--decode-batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.study_config).resolve()
    config = yaml.safe_load(config_path.read_text())
    config_root = config_path.parent

    def resolved(raw_value: str) -> Path:
        value = Path(raw_value).expanduser()
        return (value if value.is_absolute() else config_root / value).resolve()

    validation_config = config["decoder_validation"]
    nsd_data_root = resolved(args.nsd_data_root or config["inputs"]["nsd_data_root"])
    feature_dir = resolved(
        args.feature_dir or config["inputs"]["reconstruction_feature_dir"]
    )
    output_dir = resolved(args.output_dir or validation_config["output_dir"])
    recon_model_root = Path(args.recon_model_root).resolve()
    subjects = [int(value) for value in (args.subjects or validation_config["subjects"])]
    alphas = [float(value) for value in (args.alphas or validation_config["alphas"])]
    seed = int(args.seed if args.seed is not None else config["study"]["seed"])
    final_fraction = float(
        args.final_fraction
        if args.final_fraction is not None
        else validation_config["final_fraction"]
    )
    chunk_size = int(args.chunk_size or validation_config["chunk_size"])
    decode_batch_size = int(
        args.decode_batch_size or validation_config["decode_batch_size"]
    )

    if args.command == "selected-alpha":
        selection = json.loads((output_dir / "selected_alpha.json").read_text())
        print(selection["selected_alpha"])
        return

    if args.command == "check":
        rows, target_ids = _load_subject_rows(
            nsd_data_root=nsd_data_root,
            feature_dir=feature_dir,
            subjects=subjects,
        )
        split = build_validation_split(
            target_ids,
            subjects=subjects,
            seed=seed,
            final_fraction=final_fraction,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "subjects": subjects,
                    "aligned_rows_by_subject": {
                        str(subject): int(value.stimulus_ids.size)
                        for subject, value in rows.items()
                    },
                    "development_images": len(split["development_stimulus_ids"]),
                    "final_unseen_images": len(split["final_stimulus_ids"]),
                    "candidate_alphas": alphas,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "validate":
        summary = validate_decoder(
            nsd_data_root=nsd_data_root,
            feature_dir=feature_dir,
            output_dir=output_dir,
            subjects=subjects,
            alphas=alphas,
            seed=seed,
            final_fraction=final_fraction,
            chunk_size=chunk_size,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    decode_validation(
        output_dir=output_dir,
        feature_dir=feature_dir,
        recon_model_root=recon_model_root,
        subjects=subjects,
        batch_size=decode_batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
