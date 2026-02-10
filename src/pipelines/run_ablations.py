"""
Ablation experiments: few-shot sweep, n_components, feature types.
"""

import json
import logging
import os

import numpy as np
import yaml

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
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_fewshot_ablation(
        test_sub=config["subjects"]["test"][0],
        shots_list=config["evaluation"]["fewshot_shots"],
        n_repeats=config.get("fewshot_n_repeats", 5),
        base_seed=config.get("random_seed", 42),
        model_dir=args.model_dir,
        data_root=args.data_root,
        feature_type=config["features"]["type"],
        output_dir=os.path.join(args.output_dir, "fewshot"),
    )
