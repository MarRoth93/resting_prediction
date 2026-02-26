"""
Optuna + TensorBoard sweep for shared-space ridge/alignment hyperparameters.

Targets:
- encoding.ridge_alpha
- alignment.n_components

Objective:
- LOSO (leave-one-subject-out) mean zero-shot median_r across subjects.train
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.pipelines.predict_subject import predict_zero_shot
from src.pipelines.train_shared_space import apply_config_overrides, load_config, train_pipeline

logger = logging.getLogger(__name__)

_DEFAULT_SHARED_SPACE_SWEEP = {
    "study_name": "shared_space_optuna",
    "output_dir": "outputs/hparam_sweeps/shared_space",
    "n_trials": 40,
    "timeout": None,
    "sampler_seed": 42,
    "cleanup_trial_artifacts": True,
    "retrain_best": True,
    "pruner": {
        "type": "median",
        "n_startup_trials": 5,
        "n_warmup_steps": 2,
        "interval_steps": 1,
    },
    "search_space": {
        "ridge_alpha": {"min": 1.0e-2, "max": 1.0e5, "log": True},
        "n_components": {"min": 20, "max": 120, "step": 5},
    },
}


TrainFn = Callable[..., Any]
PredictFn = Callable[..., dict]


def build_loso_folds(train_subjects: list[int]) -> list[tuple[list[int], int]]:
    """Generate LOSO folds as (train_subjects_without_val, val_subject)."""
    subs = [int(s) for s in train_subjects]
    if len(subs) < 2:
        raise ValueError("LOSO requires at least 2 training subjects.")
    folds: list[tuple[list[int], int]] = []
    for idx, val_sub in enumerate(subs):
        fold_train = [s for j, s in enumerate(subs) if j != idx]
        folds.append((fold_train, val_sub))
    return folds


def _import_optuna():
    try:
        import optuna  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when missing dep.
        raise ImportError(
            "Optuna sweep requires `optuna`. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return optuna


def _get_summary_writer_cls():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - exercised only when missing dep.
        raise ImportError(
            "TensorBoard logging requires `tensorboard` (and torch). "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return SummaryWriter


def _resolve_storage_url(storage: str | None, study_dir: Path) -> str:
    if not storage:
        default_db = (study_dir / "study.db").resolve()
        return f"sqlite:///{default_db}"
    if "://" in storage:
        return storage
    return f"sqlite:///{Path(storage).resolve()}"


def _build_pruner(optuna_mod, pruner_cfg: dict):
    kind = str(pruner_cfg.get("type", "median")).strip().lower()
    if kind in {"none", "nop", "disabled"}:
        return optuna_mod.pruners.NopPruner()
    if kind == "median":
        return optuna_mod.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 2)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )
    raise ValueError(f"Unsupported pruner type: {kind}")


def _validate_search_space(
    alpha_min: float,
    alpha_max: float,
    ncomp_min: int,
    ncomp_max: int,
    ncomp_step: int,
) -> None:
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValueError("ridge_alpha bounds must be > 0.")
    if alpha_min >= alpha_max:
        raise ValueError(f"ridge_alpha min must be < max, got min={alpha_min}, max={alpha_max}.")
    if ncomp_min < 1 or ncomp_max < 1:
        raise ValueError("n_components bounds must be >= 1.")
    if ncomp_min > ncomp_max:
        raise ValueError(
            f"n_components min must be <= max, got min={ncomp_min}, max={ncomp_max}."
        )
    if ncomp_step < 1:
        raise ValueError(f"n_components step must be >= 1, got {ncomp_step}.")


def _serialize_trial_state(trial) -> str:
    state = getattr(trial, "state", None)
    return str(getattr(state, "name", state))


def _trial_duration_seconds(trial) -> float | None:
    dt_start = getattr(trial, "datetime_start", None)
    dt_end = getattr(trial, "datetime_complete", None)
    if dt_start is None or dt_end is None:
        return None
    return float((dt_end - dt_start).total_seconds())


def save_trials_csv(study, path: str) -> None:
    """Save study trials to a CSV without requiring pandas."""
    trials = list(study.trials)
    param_keys = sorted({k for t in trials for k in t.params.keys()})
    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_seconds",
    ] + [f"param_{k}" for k in param_keys] + ["user_attrs_json"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row = {
                "number": int(trial.number),
                "state": _serialize_trial_state(trial),
                "value": "" if trial.value is None else float(trial.value),
                "datetime_start": (
                    trial.datetime_start.isoformat() if trial.datetime_start is not None else ""
                ),
                "datetime_complete": (
                    trial.datetime_complete.isoformat() if trial.datetime_complete is not None else ""
                ),
                "duration_seconds": (
                    "" if _trial_duration_seconds(trial) is None else _trial_duration_seconds(trial)
                ),
                "user_attrs_json": json.dumps(trial.user_attrs, sort_keys=True, default=str),
            }
            for key in param_keys:
                val = trial.params.get(key)
                row[f"param_{key}"] = "" if val is None else val
            writer.writerow(row)


def _count_trial_states(study) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in study.trials:
        state = _serialize_trial_state(trial)
        counts[state] = counts.get(state, 0) + 1
    return counts


def _maybe_add_scalar(writer, tag: str, value: float, step: int) -> None:
    if writer is not None:
        writer.add_scalar(tag, float(value), int(step))


def make_loso_objective(
    *,
    optuna_mod,
    config_path: str,
    data_root: str,
    raw_data_root: str,
    feature_type: str,
    folds: list[tuple[list[int], int]],
    trial_artifacts_root: str,
    fixed_eval_size: int,
    eval_split_seed: int,
    reliability_thresholds: list[float] | None,
    cleanup_trial_artifacts: bool,
    writer,
    best_tracker: dict[str, float],
    alpha_min: float,
    alpha_max: float,
    ncomp_min: int,
    ncomp_max: int,
    ncomp_step: int,
    train_fn: TrainFn = train_pipeline,
    predict_fn: PredictFn = predict_zero_shot,
):
    """Factory for Optuna trial objective with fold-level reporting/pruning."""

    def objective(trial) -> float:
        trial_dir = os.path.join(trial_artifacts_root, f"trial_{int(trial.number):05d}")
        os.makedirs(trial_dir, exist_ok=True)
        started = time.time()

        ridge_alpha = float(
            trial.suggest_float(
                "encoding.ridge_alpha",
                float(alpha_min),
                float(alpha_max),
                log=True,
            )
        )
        n_components = int(
            trial.suggest_int(
                "alignment.n_components",
                int(ncomp_min),
                int(ncomp_max),
                step=int(ncomp_step),
            )
        )

        fold_scores: list[float] = []
        fold_scores_by_sub: dict[str, float] = {}
        try:
            for fold_step, (fold_train_subs, fold_val_sub) in enumerate(folds, start=1):
                fold_root = os.path.join(trial_dir, f"fold_sub{int(fold_val_sub):02d}")
                fold_model_dir = os.path.join(fold_root, "model")
                fold_pred_dir = os.path.join(fold_root, "predictions")

                fold_overrides = {
                    "subjects": {
                        "train": [int(s) for s in fold_train_subs],
                        "test": [int(fold_val_sub)],
                    },
                    "alignment": {"n_components": int(n_components)},
                    "encoding": {"ridge_alpha": float(ridge_alpha)},
                }

                train_fn(
                    config_path=config_path,
                    data_root=data_root,
                    raw_data_root=raw_data_root,
                    output_dir=fold_model_dir,
                    feature_type_override=feature_type,
                    config_overrides=fold_overrides,
                )
                pred = predict_fn(
                    test_sub=int(fold_val_sub),
                    model_dir=fold_model_dir,
                    data_root=data_root,
                    feature_type=feature_type,
                    output_dir=fold_pred_dir,
                    use_fixed_eval_split=True,
                    fixed_eval_size=int(fixed_eval_size),
                    eval_split_seed=int(eval_split_seed),
                    reliability_thresholds=reliability_thresholds,
                )
                score = float(pred["metrics"]["median_r"])
                fold_scores.append(score)
                fold_scores_by_sub[str(int(fold_val_sub))] = score

                interim = float(np.mean(fold_scores))
                trial.report(interim, step=fold_step)
                _maybe_add_scalar(writer, f"fold/sub{int(fold_val_sub):02d}_median_r", score, trial.number)
                _maybe_add_scalar(writer, "trial/intermediate_objective", interim, trial.number)

                if trial.should_prune():
                    trial.set_user_attr("fold_scores", fold_scores_by_sub)
                    trial.set_user_attr("pruned_after_fold", int(fold_step))
                    _maybe_add_scalar(writer, "trial/pruned", 1.0, trial.number)
                    raise optuna_mod.TrialPruned(f"Pruned after fold {fold_step}.")

            objective_value = float(np.mean(fold_scores))
            elapsed = float(time.time() - started)
            trial.set_user_attr("fold_scores", fold_scores_by_sub)
            trial.set_user_attr("trial_seconds", elapsed)
            trial.set_user_attr("objective_metric", "median_r")

            _maybe_add_scalar(writer, "trial/objective", objective_value, trial.number)
            _maybe_add_scalar(writer, "timing/trial_seconds", elapsed, trial.number)
            best_tracker["value"] = max(float(best_tracker["value"]), objective_value)
            _maybe_add_scalar(writer, "trial/best_so_far", best_tracker["value"], trial.number)

            if writer is not None:
                writer.add_hparams(
                    {
                        "ridge_alpha": float(ridge_alpha),
                        "n_components": int(n_components),
                    },
                    {"hparam/objective": float(objective_value)},
                    run_name=f"trial_{int(trial.number):05d}",
                )

            return objective_value
        finally:
            if cleanup_trial_artifacts:
                shutil.rmtree(trial_dir, ignore_errors=True)

    return objective


def _resolve_sweep_settings(
    config: dict,
    *,
    n_trials: int | None,
    timeout: int | None,
    study_name: str | None,
    output_root: str | None,
    seed: int | None,
    alpha_min: float | None,
    alpha_max: float | None,
    ncomp_min: int | None,
    ncomp_max: int | None,
    ncomp_step: int | None,
    fixed_eval_size: int | None,
    eval_split_seed: int | None,
    cleanup_trial_artifacts: bool | None,
    retrain_best: bool | None,
) -> dict:
    sweep_cfg = (
        config.get("sweep", {}).get("shared_space", {})
        if isinstance(config.get("sweep", {}), dict)
        else {}
    )
    resolved = apply_config_overrides(_DEFAULT_SHARED_SPACE_SWEEP, sweep_cfg)

    if n_trials is not None:
        resolved["n_trials"] = int(n_trials)
    if timeout is not None:
        resolved["timeout"] = int(timeout)
    if study_name:
        resolved["study_name"] = str(study_name)
    if output_root:
        resolved["output_dir"] = str(output_root)
    if seed is not None:
        resolved["sampler_seed"] = int(seed)
    if cleanup_trial_artifacts is not None:
        resolved["cleanup_trial_artifacts"] = bool(cleanup_trial_artifacts)
    if retrain_best is not None:
        resolved["retrain_best"] = bool(retrain_best)

    search = resolved["search_space"]
    if alpha_min is not None:
        search["ridge_alpha"]["min"] = float(alpha_min)
    if alpha_max is not None:
        search["ridge_alpha"]["max"] = float(alpha_max)
    if ncomp_min is not None:
        search["n_components"]["min"] = int(ncomp_min)
    if ncomp_max is not None:
        search["n_components"]["max"] = int(ncomp_max)
    if ncomp_step is not None:
        search["n_components"]["step"] = int(ncomp_step)

    eval_cfg = config.get("evaluation", {}) if isinstance(config.get("evaluation", {}), dict) else {}
    resolved["fixed_eval_size"] = (
        int(fixed_eval_size) if fixed_eval_size is not None else int(eval_cfg.get("fixed_eval_size", 250))
    )
    resolved["eval_split_seed"] = (
        int(eval_split_seed)
        if eval_split_seed is not None
        else int(eval_cfg.get("eval_split_seed", config.get("random_seed", 42)))
    )
    resolved["reliability_thresholds"] = [
        float(x) for x in eval_cfg.get("reliability_thresholds", [0.0, 0.1, 0.3])
    ]
    if int(resolved["n_trials"]) < 1:
        raise ValueError(f"n_trials must be >= 1, got {resolved['n_trials']}.")
    if resolved["timeout"] is not None and int(resolved["timeout"]) < 1:
        raise ValueError(f"timeout must be >= 1 when provided, got {resolved['timeout']}.")
    if int(resolved["fixed_eval_size"]) < 1:
        raise ValueError(
            f"fixed_eval_size must be >= 1, got {resolved['fixed_eval_size']}."
        )

    return resolved


def run_shared_space_sweep(
    *,
    config_path: str = "config.yaml",
    data_root: str = "processed_data",
    raw_data_root: str = ".",
    feature_type_override: str | None = None,
    n_trials: int | None = None,
    timeout: int | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    seed: int | None = None,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    ncomp_min: int | None = None,
    ncomp_max: int | None = None,
    ncomp_step: int | None = None,
    fixed_eval_size: int | None = None,
    eval_split_seed: int | None = None,
    output_root: str | None = None,
    cleanup_trial_artifacts: bool | None = None,
    retrain_best: bool | None = None,
    dry_run: bool = False,
    train_fn: TrainFn = train_pipeline,
    predict_fn: PredictFn = predict_zero_shot,
) -> dict:
    """Run single-process Optuna LOSO sweep for shared-space ridge/alignment."""
    config = load_config(config_path)
    train_subjects = [int(s) for s in config["subjects"]["train"]]
    folds = build_loso_folds(train_subjects)

    feature_type = (
        str(feature_type_override).strip()
        if feature_type_override is not None and str(feature_type_override).strip()
        else str(config["features"]["type"]).strip()
    )
    if feature_type == "":
        raise ValueError("Resolved feature type is empty.")

    settings = _resolve_sweep_settings(
        config,
        n_trials=n_trials,
        timeout=timeout,
        study_name=study_name,
        output_root=output_root,
        seed=seed,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        ncomp_min=ncomp_min,
        ncomp_max=ncomp_max,
        ncomp_step=ncomp_step,
        fixed_eval_size=fixed_eval_size,
        eval_split_seed=eval_split_seed,
        cleanup_trial_artifacts=cleanup_trial_artifacts,
        retrain_best=retrain_best,
    )

    alpha_cfg = settings["search_space"]["ridge_alpha"]
    ncomp_cfg = settings["search_space"]["n_components"]
    alpha_min_v = float(alpha_cfg["min"])
    alpha_max_v = float(alpha_cfg["max"])
    ncomp_min_v = int(ncomp_cfg["min"])
    ncomp_max_v = int(ncomp_cfg["max"])
    ncomp_step_v = int(ncomp_cfg.get("step", 1))
    _validate_search_space(
        alpha_min=alpha_min_v,
        alpha_max=alpha_max_v,
        ncomp_min=ncomp_min_v,
        ncomp_max=ncomp_max_v,
        ncomp_step=ncomp_step_v,
    )

    study_name_v = str(settings["study_name"]).strip()
    output_root_v = Path(str(settings["output_dir"]))
    study_dir = output_root_v / study_name_v
    trial_artifacts_root = study_dir / "trial_artifacts"
    tb_dir = output_root_v / "tensorboard" / study_name_v

    storage_url = _resolve_storage_url(storage, study_dir=study_dir)
    dry_summary = {
        "mode": "dry_run" if dry_run else "run",
        "study_name": study_name_v,
        "feature_type": feature_type,
        "output_root": str(output_root_v),
        "study_dir": str(study_dir),
        "storage": storage_url,
        "n_trials": int(settings["n_trials"]),
        "timeout": settings["timeout"],
        "fixed_eval_size": int(settings["fixed_eval_size"]),
        "eval_split_seed": int(settings["eval_split_seed"]),
        "cleanup_trial_artifacts": bool(settings["cleanup_trial_artifacts"]),
        "retrain_best": bool(settings["retrain_best"]),
        "search_space": {
            "ridge_alpha": {"min": alpha_min_v, "max": alpha_max_v, "log": True},
            "n_components": {
                "min": ncomp_min_v,
                "max": ncomp_max_v,
                "step": ncomp_step_v,
            },
        },
        "folds": [
            {"train_subjects": [int(s) for s in fold_train], "val_subject": int(val_sub)}
            for fold_train, val_sub in folds
        ],
    }
    if dry_run:
        logger.info("Dry run config:\n%s", json.dumps(dry_summary, indent=2))
        return dry_summary

    for d in [study_dir, trial_artifacts_root, tb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    optuna_mod = _import_optuna()
    summary_writer_cls = _get_summary_writer_cls()
    writer = summary_writer_cls(log_dir=str(tb_dir))

    sampler = optuna_mod.samplers.TPESampler(seed=int(settings["sampler_seed"]))
    pruner = _build_pruner(optuna_mod, pruner_cfg=settings.get("pruner", {}))
    study = optuna_mod.create_study(
        study_name=study_name_v,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )
    complete_values = [float(t.value) for t in study.trials if t.value is not None and _serialize_trial_state(t) == "COMPLETE"]
    best_tracker = {"value": max(complete_values) if complete_values else float("-inf")}

    objective = make_loso_objective(
        optuna_mod=optuna_mod,
        config_path=config_path,
        data_root=data_root,
        raw_data_root=raw_data_root,
        feature_type=feature_type,
        folds=folds,
        trial_artifacts_root=str(trial_artifacts_root),
        fixed_eval_size=int(settings["fixed_eval_size"]),
        eval_split_seed=int(settings["eval_split_seed"]),
        reliability_thresholds=list(settings.get("reliability_thresholds", [0.0, 0.1, 0.3])),
        cleanup_trial_artifacts=bool(settings["cleanup_trial_artifacts"]),
        writer=writer,
        best_tracker=best_tracker,
        alpha_min=alpha_min_v,
        alpha_max=alpha_max_v,
        ncomp_min=ncomp_min_v,
        ncomp_max=ncomp_max_v,
        ncomp_step=ncomp_step_v,
        train_fn=train_fn,
        predict_fn=predict_fn,
    )

    try:
        study.optimize(
            objective,
            n_trials=int(settings["n_trials"]),
            timeout=None if settings["timeout"] is None else int(settings["timeout"]),
        )
    finally:
        writer.flush()
        writer.close()

    trials_csv_path = study_dir / "trials.csv"
    save_trials_csv(study, str(trials_csv_path))

    state_counts = _count_trial_states(study)
    complete_trials = [t for t in study.trials if _serialize_trial_state(t) == "COMPLETE" and t.value is not None]
    if not complete_trials:
        raise RuntimeError("No completed trials found; cannot select best hyperparameters.")
    best_trial = max(complete_trials, key=lambda t: float(t.value))
    best_ridge_alpha = float(best_trial.params["encoding.ridge_alpha"])
    best_n_components = int(best_trial.params["alignment.n_components"])
    best_params = {
        "encoding": {"ridge_alpha": best_ridge_alpha},
        "alignment": {"n_components": best_n_components},
    }

    best_params_path = study_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    best_metrics = {
        "trial_number": int(best_trial.number),
        "value": float(best_trial.value),
        "fold_scores": best_trial.user_attrs.get("fold_scores", {}),
        "trial_seconds": best_trial.user_attrs.get("trial_seconds"),
        "params": best_params,
    }
    best_trial_metrics_path = study_dir / "best_trial_metrics.json"
    with open(best_trial_metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    final_model_dir = os.path.join(
        str(config.get("output_root", "outputs")),
        f"shared_space_best_{study_name_v}",
    )
    retrained = False
    if bool(settings["retrain_best"]):
        train_fn(
            config_path=config_path,
            data_root=data_root,
            raw_data_root=raw_data_root,
            output_dir=final_model_dir,
            feature_type_override=feature_type,
            config_overrides=best_params,
        )
        retrained = True

    study_summary = {
        "study_name": study_name_v,
        "feature_type": feature_type,
        "storage": storage_url,
        "n_trials_total": len(study.trials),
        "state_counts": state_counts,
        "best_trial": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_params": best_params,
        "paths": {
            "study_dir": str(study_dir),
            "tensorboard_dir": str(tb_dir),
            "trials_csv": str(trials_csv_path),
            "best_params_json": str(best_params_path),
            "best_trial_metrics_json": str(best_trial_metrics_path),
            "best_model_dir": final_model_dir,
        },
        "retrained_best_model": retrained,
    }
    study_summary_path = study_dir / "study_summary.json"
    with open(study_summary_path, "w") as f:
        json.dump(study_summary, f, indent=2)

    return study_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna LOSO sweep for shared-space model "
            "(encoding.ridge_alpha + alignment.n_components)."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--raw-data-root", default=".")
    parser.add_argument("--feature-type", default="")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", default="")
    parser.add_argument("--storage", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha-min", type=float, default=None)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--ncomp-min", type=int, default=None)
    parser.add_argument("--ncomp-max", type=int, default=None)
    parser.add_argument("--ncomp-step", type=int, default=None)
    parser.add_argument("--fixed-eval-size", type=int, default=None)
    parser.add_argument("--eval-split-seed", type=int, default=None)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-cleanup-trial-artifacts",
        action="store_true",
        help="Retain per-trial fold artifacts under the sweep study directory.",
    )
    parser.add_argument(
        "--no-retrain-best",
        action="store_true",
        help="Skip final retraining on full train subjects using best params.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = run_shared_space_sweep(
        config_path=args.config,
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
        feature_type_override=(args.feature_type.strip() or None),
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=(args.study_name.strip() or None),
        storage=(args.storage.strip() or None),
        seed=args.seed,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        ncomp_min=args.ncomp_min,
        ncomp_max=args.ncomp_max,
        ncomp_step=args.ncomp_step,
        fixed_eval_size=args.fixed_eval_size,
        eval_split_seed=args.eval_split_seed,
        output_root=(args.output_root.strip() or None),
        cleanup_trial_artifacts=(False if args.no_cleanup_trial_artifacts else None),
        retrain_best=(False if args.no_retrain_best else None),
        dry_run=bool(args.dry_run),
    )
    logger.info("Sweep summary:\n%s", json.dumps(summary, indent=2))
