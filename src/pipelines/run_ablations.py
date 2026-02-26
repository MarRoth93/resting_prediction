"""
Ablation experiments: few-shot sweep, n_components, feature types.
"""

import csv
import json
import logging
import os

import numpy as np
import yaml

from src.evaluation.statistics import build_condition_summary, save_summary_csv
from src.pipelines.predict_subject import predict_few_shot, predict_zero_shot

logger = logging.getLogger(__name__)


def _resolve_model_dirs(
    feature_types: list[str],
    default_feature: str,
    base_model_dir: str,
    eval_cfg: dict,
) -> dict[str, str]:
    """Resolve a model directory per feature backbone."""
    resolved: dict[str, str] = {}
    configured = eval_cfg.get("feature_model_dirs", {})
    if isinstance(configured, dict):
        for key, value in configured.items():
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                resolved[str(key)] = value_str

    for feature_type in feature_types:
        if feature_type in resolved:
            continue
        if feature_type == default_feature:
            resolved[feature_type] = base_model_dir
        else:
            resolved[feature_type] = f"{base_model_dir}_{feature_type}"
    return resolved


def _assert_model_artifacts(feature_type: str, model_dir: str) -> None:
    """Ensure a feature backbone points at a trained shared-space model directory."""
    required = ["builder.npz", "encoder.npz", "shared_stim_idx.npy"]
    missing = [name for name in required if not os.path.exists(os.path.join(model_dir, name))]
    if missing:
        missing_joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing model artifacts for feature_type={feature_type!r} in {model_dir!r}: {missing_joined}. "
            f"Train this backbone first (e.g., train_shared_space --feature-type {feature_type})."
        )


def run_fewshot_ablation(
    test_sub: int = 7,
    shots_list: list[int] | None = None,
    n_repeats: int = 5,
    base_seed: int = 42,
    model_dir: str = "outputs/shared_space",
    data_root: str = "processed_data",
    feature_type: str = "clip",
    output_dir: str = "outputs/ablations/fewshot",
    n_bootstrap: int = 5000,
    ci_level: float = 95.0,
    n_permutations: int = 10000,
    permutation_metric: str = "median_r",
    permutation_reference_condition: int = 0,
    fdr_alpha: float = 0.05,
    use_fixed_eval_split: bool = True,
    fixed_eval_size: int = 250,
    eval_split_seed: int = 42,
    reliability_thresholds: list[float] | None = None,
) -> dict:
    """
    Run few-shot ablation with varying N and repeated random splits.

    N=0 uses zero-shot (no random split needed).
    """
    if shots_list is None:
        shots_list = [0, 10, 25, 50, 100, 250, 500, 750]

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for n_shots in shots_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"Few-shot ablation: N = {n_shots}")

        if n_shots == 0:
            # Zero-shot: single run, no random split
            result = predict_zero_shot(
                test_sub=test_sub,
                model_dir=model_dir,
                data_root=data_root,
                feature_type=feature_type,
                output_dir=output_dir,
                use_fixed_eval_split=use_fixed_eval_split,
                fixed_eval_size=fixed_eval_size,
                eval_split_seed=eval_split_seed,
                reliability_thresholds=reliability_thresholds,
            )
            all_results[n_shots] = {
                "median_r": result["metrics"]["median_r"],
                "std": 0.0,
                "repeats": [result["metrics"]],
            }
        else:
            # Multiple repeats with different seeds
            repeat_metrics = []
            for repeat in range(n_repeats):
                seed = base_seed + repeat
                result = predict_few_shot(
                    test_sub=test_sub,
                    n_shots=n_shots,
                    model_dir=model_dir,
                    data_root=data_root,
                    feature_type=feature_type,
                    seed=seed,
                    output_dir=output_dir,
                    use_fixed_eval_split=use_fixed_eval_split,
                    fixed_eval_size=fixed_eval_size,
                    eval_split_seed=eval_split_seed,
                    reliability_thresholds=reliability_thresholds,
                )
                repeat_metrics.append(result["metrics"])

            median_rs = [m["median_r"] for m in repeat_metrics]
            all_results[n_shots] = {
                "median_r": float(np.mean(median_rs)),
                "std": float(np.std(median_rs)),
                "repeats": repeat_metrics,
            }

        logger.info(f"N={n_shots}: median_r = {all_results[n_shots]['median_r']:.4f} "
                     f"+/- {all_results[n_shots]['std']:.4f}")

    # Save summary
    # Convert keys to strings for JSON
    json_results = {str(k): v for k, v in all_results.items()}
    with open(os.path.join(output_dir, "fewshot_ablation.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    metric_names = ["median_r", "mean_r", "median_pattern_r", "two_vs_two"]
    first_condition = sorted(all_results.keys())[0] if all_results else None
    first_repeats = all_results[first_condition]["repeats"] if first_condition is not None else []
    if first_repeats:
        sample = first_repeats[0]
        reliability_metric_names = sorted(
            k for k in sample
            if k.startswith("median_r_nc_ge_") or k.startswith("mean_r_nc_ge_")
        )
        reliability_count_names = sorted(k for k in sample if k.startswith("n_voxels_nc_ge_"))
        metric_names.extend(reliability_metric_names)
        metric_names.extend(reliability_count_names)
        if "noise_ceiling_median" in sample:
            metric_names.append("noise_ceiling_median")
        if "noise_ceiling_mean" in sample:
            metric_names.append("noise_ceiling_mean")

    summary_rows = build_condition_summary(
        condition_results=all_results,
        metric_names=metric_names,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        permutation_metric=permutation_metric,
        n_permutations=n_permutations,
        permutation_reference_condition=permutation_reference_condition,
        fdr_alpha=fdr_alpha,
        seed=base_seed,
    )
    summary_csv_path = os.path.join(output_dir, "fewshot_summary.csv")
    save_summary_csv(summary_rows, summary_csv_path)

    stats_json_path = os.path.join(output_dir, "fewshot_statistics.json")
    with open(stats_json_path, "w") as f:
        json.dump(
            {
                "ci_level": ci_level,
                "n_bootstrap": n_bootstrap,
                "n_permutations": n_permutations,
                "permutation_metric": permutation_metric,
                "permutation_reference_condition": permutation_reference_condition,
                "fdr_alpha": fdr_alpha,
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved summary table: {summary_csv_path}")
    logger.info(f"Saved statistics: {stats_json_path}")
    logger.info(f"\nFew-shot ablation complete. Saved to {output_dir}")
    return all_results


def run_feature_backbone_sweep(
    feature_types: list[str],
    test_sub: int,
    shots_list: list[int],
    n_repeats: int,
    base_seed: int,
    model_dirs: dict[str, str],
    data_root: str,
    output_dir: str,
    n_bootstrap: int,
    ci_level: float,
    n_permutations: int,
    permutation_metric: str,
    permutation_reference_condition: int,
    fdr_alpha: float,
    use_fixed_eval_split: bool = True,
    fixed_eval_size: int = 250,
    eval_split_seed: int = 42,
    reliability_thresholds: list[float] | None = None,
    precomputed_results: dict[str, dict] | None = None,
    require_parcellation_artifacts: bool = False,
) -> dict:
    """
    Run few-shot ablation for multiple feature backbones with matched eval splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    sweep_results: dict[str, dict] = {}
    precomputed_results = precomputed_results or {}
    summary_rows: list[dict[str, float | int | str]] = []

    for feature_type in feature_types:
        feature_model_dir = model_dirs[feature_type]
        _assert_model_artifacts(feature_type, feature_model_dir)
        if require_parcellation_artifacts:
            _assert_model_artifacts_parcellation(feature_type, feature_model_dir, test_sub)
        if feature_type in precomputed_results:
            logger.info(
                "Feature backbone sweep: %s (using precomputed results from %s)",
                feature_type,
                feature_model_dir,
            )
            result = precomputed_results[feature_type]
        else:
            logger.info(f"\n{'='*60}")
            logger.info(
                "Feature backbone sweep: %s (model_dir=%s)",
                feature_type,
                feature_model_dir,
            )
            feature_dir = os.path.join(output_dir, f"feature_{feature_type}")
            result = run_fewshot_ablation(
                test_sub=test_sub,
                shots_list=shots_list,
                n_repeats=n_repeats,
                base_seed=base_seed,
                model_dir=feature_model_dir,
                data_root=data_root,
                feature_type=feature_type,
                output_dir=feature_dir,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                n_permutations=n_permutations,
                permutation_metric=permutation_metric,
                permutation_reference_condition=permutation_reference_condition,
                fdr_alpha=fdr_alpha,
                use_fixed_eval_split=use_fixed_eval_split,
                fixed_eval_size=fixed_eval_size,
                eval_split_seed=eval_split_seed,
                reliability_thresholds=reliability_thresholds,
            )
        sweep_results[feature_type] = result

        zero_metrics = result.get(0, {})
        best_n = None
        best_median = -np.inf
        for n_shots, vals in result.items():
            if int(n_shots) <= 0:
                continue
            score = float(vals.get("median_r", np.nan))
            if np.isfinite(score) and score > best_median:
                best_median = score
                best_n = int(n_shots)

        row = {
            "feature_type": feature_type,
            "model_dir": feature_model_dir,
            "zero_shot_median_r": float(zero_metrics.get("median_r", np.nan)),
            "best_fewshot_n": int(best_n) if best_n is not None else -1,
            "best_fewshot_median_r": float(best_median) if np.isfinite(best_median) else float("nan"),
            "delta_best_minus_zero": (
                float(best_median - float(zero_metrics.get("median_r", np.nan)))
                if best_n is not None and np.isfinite(best_median)
                else float("nan")
            ),
        }
        if zero_metrics.get("repeats"):
            zero_sample = zero_metrics["repeats"][0]
            for key in sorted(zero_sample):
                if key.startswith("median_r_nc_ge_"):
                    row[f"zero_{key}"] = float(zero_sample[key])
        if best_n is not None and result[best_n].get("repeats"):
            best_vals = [float(m["median_r"]) for m in result[best_n]["repeats"] if "median_r" in m]
            if best_vals:
                row["best_fewshot_median_r_across_repeats"] = float(np.median(best_vals))
            best_sample = result[best_n]["repeats"][0]
            for key in sorted(best_sample):
                if key.startswith("median_r_nc_ge_"):
                    row[f"best_{key}"] = float(best_sample[key])
        summary_rows.append(row)

    with open(os.path.join(output_dir, "feature_backbone_sweep.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)

    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row.keys()})
        csv_path = os.path.join(output_dir, "feature_backbone_sweep.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Saved feature sweep table: {csv_path}")

    logger.info(f"Feature backbone sweep complete. Saved to {output_dir}")
    return sweep_results


def _assert_model_artifacts_parcellation(
    feature_type: str,
    model_dir: str,
    test_sub: int,
) -> None:
    """Require parcellation-specific artifacts used by prediction-time alignment."""
    required = ["atlas_info.npz", f"atlas_masked_{test_sub}.npy"]
    missing = [name for name in required if not os.path.exists(os.path.join(model_dir, name))]
    if missing:
        missing_joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing parcellation artifacts for feature_type={feature_type!r} in {model_dir!r}: "
            f"{missing_joined}. Re-run shared-space training for this backbone."
        )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default="outputs/shared_space")
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--output-dir", default="outputs/ablations")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--ci-level", type=float, default=95.0)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--permutation-metric", default="median_r")
    parser.add_argument("--permutation-reference-condition", type=int, default=0)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument(
        "--feature-types",
        default="",
        help="Comma-separated feature backbones to sweep (e.g., clip,dinov2,clip_dinov2).",
    )
    parser.add_argument("--disable-feature-sweep", action="store_true")
    parser.add_argument("--disable-fixed-eval-split", action="store_true")
    parser.add_argument("--fixed-eval-size", type=int, default=250)
    parser.add_argument("--eval-split-seed", type=int, default=42)
    parser.add_argument(
        "--reliability-thresholds",
        default="0.0,0.1,0.3",
        help="Comma-separated NC thresholds for stratified metrics.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    stats_cfg = config.get("evaluation", {}).get("statistics", {})
    eval_cfg = config.get("evaluation", {})
    use_fixed_eval_split = not args.disable_fixed_eval_split
    fixed_eval_size = (
        args.fixed_eval_size
        if args.fixed_eval_size != 250
        else int(eval_cfg.get("fixed_eval_size", 250))
    )
    eval_split_seed = (
        args.eval_split_seed
        if args.eval_split_seed != 42
        else int(eval_cfg.get("eval_split_seed", config.get("random_seed", 42)))
    )
    if args.reliability_thresholds != "0.0,0.1,0.3":
        reliability_thresholds = [
            float(x.strip()) for x in args.reliability_thresholds.split(",") if x.strip() != ""
        ]
    else:
        reliability_thresholds = [float(x) for x in eval_cfg.get("reliability_thresholds", [0.0, 0.1, 0.3])]

    default_feature = str(config["features"]["type"])
    config_feature_types = eval_cfg.get("feature_backbones")
    cli_feature_types = [s.strip() for s in args.feature_types.split(",") if s.strip()]
    feature_types = cli_feature_types
    if not feature_types and isinstance(config_feature_types, list) and config_feature_types:
        feature_types = [str(x) for x in config_feature_types]
    if not feature_types:
        feature_types = [default_feature]

    run_feature_sweep = (
        len(feature_types) > 1
        and not args.disable_feature_sweep
        and bool(eval_cfg.get("run_feature_sweep", True))
    )
    require_parcellation_artifacts = (
        str(config.get("alignment", {}).get("connectivity_mode", "")).strip() == "parcellation"
    )

    sweep_types = feature_types if default_feature in feature_types else [default_feature] + feature_types
    model_dirs = _resolve_model_dirs(
        feature_types=sweep_types,
        default_feature=default_feature,
        base_model_dir=args.model_dir,
        eval_cfg=eval_cfg,
    )

    common_kwargs = dict(
        test_sub=config["subjects"]["test"][0],
        shots_list=config["evaluation"]["fewshot_shots"],
        n_repeats=config.get("fewshot_n_repeats", 5),
        base_seed=config.get("random_seed", 42),
        data_root=args.data_root,
        n_bootstrap=args.n_bootstrap if args.n_bootstrap != 5000 else int(stats_cfg.get("n_bootstrap", 5000)),
        ci_level=args.ci_level if args.ci_level != 95.0 else float(stats_cfg.get("ci_level", 95.0)),
        n_permutations=args.n_permutations if args.n_permutations != 10000 else int(stats_cfg.get("n_permutations", 10000)),
        permutation_metric=args.permutation_metric if args.permutation_metric != "median_r" else str(stats_cfg.get("permutation_metric", "median_r")),
        permutation_reference_condition=args.permutation_reference_condition if args.permutation_reference_condition != 0 else int(stats_cfg.get("permutation_reference_condition", 0)),
        fdr_alpha=args.fdr_alpha if args.fdr_alpha != 0.05 else float(stats_cfg.get("fdr_alpha", 0.05)),
        use_fixed_eval_split=use_fixed_eval_split,
        fixed_eval_size=fixed_eval_size,
        eval_split_seed=eval_split_seed,
        reliability_thresholds=reliability_thresholds,
    )

    primary_feature = default_feature
    primary_model_dir = model_dirs[primary_feature]
    _assert_model_artifacts(primary_feature, primary_model_dir)
    if require_parcellation_artifacts:
        _assert_model_artifacts_parcellation(primary_feature, primary_model_dir, config["subjects"]["test"][0])
    baseline_results = run_fewshot_ablation(
        model_dir=primary_model_dir,
        feature_type=primary_feature,
        output_dir=os.path.join(args.output_dir, "fewshot"),
        **common_kwargs,
    )

    if run_feature_sweep:
        run_feature_backbone_sweep(
            feature_types=sweep_types,
            model_dirs=model_dirs,
            output_dir=os.path.join(args.output_dir, "feature_backbone_sweep"),
            precomputed_results={primary_feature: baseline_results},
            require_parcellation_artifacts=require_parcellation_artifacts,
            **common_kwargs,
        )
