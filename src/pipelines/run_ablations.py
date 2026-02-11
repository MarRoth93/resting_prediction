"""
Ablation experiments: few-shot sweep, n_components, feature types.
"""

import json
import logging
import os

import numpy as np
import yaml

from src.evaluation.statistics import build_condition_summary, save_summary_csv
from src.pipelines.predict_subject import predict_few_shot, predict_zero_shot

logger = logging.getLogger(__name__)


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

    summary_rows = build_condition_summary(
        condition_results=all_results,
        metric_names=["median_r", "mean_r", "median_pattern_r", "two_vs_two"],
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
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    stats_cfg = config.get("evaluation", {}).get("statistics", {})
    run_fewshot_ablation(
        test_sub=config["subjects"]["test"][0],
        shots_list=config["evaluation"]["fewshot_shots"],
        n_repeats=config.get("fewshot_n_repeats", 5),
        base_seed=config.get("random_seed", 42),
        model_dir=args.model_dir,
        data_root=args.data_root,
        feature_type=config["features"]["type"],
        output_dir=os.path.join(args.output_dir, "fewshot"),
        n_bootstrap=args.n_bootstrap if args.n_bootstrap != 5000 else int(stats_cfg.get("n_bootstrap", 5000)),
        ci_level=args.ci_level if args.ci_level != 95.0 else float(stats_cfg.get("ci_level", 95.0)),
        n_permutations=args.n_permutations if args.n_permutations != 10000 else int(stats_cfg.get("n_permutations", 10000)),
        permutation_metric=args.permutation_metric if args.permutation_metric != "median_r" else str(stats_cfg.get("permutation_metric", "median_r")),
        permutation_reference_condition=args.permutation_reference_condition if args.permutation_reference_condition != 0 else int(stats_cfg.get("permutation_reference_condition", 0)),
        fdr_alpha=args.fdr_alpha if args.fdr_alpha != 0.05 else float(stats_cfg.get("fdr_alpha", 0.05)),
    )
