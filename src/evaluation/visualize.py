"""
Visualization functions for prediction results.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_correlation_histogram(
    corrs: np.ndarray,
    nc: np.ndarray | None = None,
    title: str = "Voxelwise Prediction Correlation",
    save_path: str | None = None,
):
    """Histogram of voxelwise correlations with optional noise ceiling."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(corrs, bins=100, alpha=0.7, color="steelblue", label="Prediction r")
    if nc is not None:
        ax.hist(nc, bins=100, alpha=0.5, color="coral", label="Noise ceiling")

    ax.axvline(np.median(corrs), color="navy", linestyle="--",
               label=f"Median r = {np.median(corrs):.3f}")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Number of voxels")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    return fig


def plot_fewshot_curve(
    results: dict[int, dict],
    metric: str = "median_r",
    title: str = "Few-shot Performance Curve",
    save_path: str | None = None,
):
    """
    Performance vs number of shots.

    Args:
        results: n_shots -> {'median_r': float, 'std': float, ...}
        metric: key in results dict
        title: plot title
        save_path: optional save path
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    shots = sorted(results.keys())
    means = [results[n].get(metric, 0) for n in shots]
    stds = [results[n].get("std", 0) for n in shots]

    ax.errorbar(shots, means, yerr=stds, marker="o", capsize=4,
                color="steelblue", linewidth=2)
    ax.set_xlabel("Number of shots (N)")
    ax.set_ylabel(f"{metric}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Mark zero-shot
    if 0 in results:
        ax.axhline(results[0].get(metric, 0), color="coral",
                    linestyle="--", label="Zero-shot")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roi_comparison(
    roi_results: dict[str, dict[str, dict]],
    conditions: list[str] | None = None,
    metric: str = "median_r",
    title: str = "ROI Performance Comparison",
    save_path: str | None = None,
):
    """
    Bar chart comparing ROI performance across conditions.

    Args:
        roi_results: condition_name -> {roi_name -> {metric: value}}
        conditions: ordered list of conditions (default: all keys)
    """
    if conditions is None:
        conditions = list(roi_results.keys())

    # Collect all ROIs
    all_rois = set()
    for cond in conditions:
        all_rois.update(roi_results[cond].keys())
    roi_names = sorted(all_rois)

    fig, ax = plt.subplots(figsize=(max(10, len(roi_names)), 6))
    x = np.arange(len(roi_names))
    width = 0.8 / len(conditions)

    for i, cond in enumerate(conditions):
        values = [roi_results[cond].get(roi, {}).get(metric, 0) for roi in roi_names]
        offset = (i - len(conditions) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=cond, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_predicted_vs_actual_patterns(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    stim_indices: np.ndarray,
    n_examples: int = 5,
    save_path: str | None = None,
):
    """
    Show predicted vs actual activation patterns for example stimuli.

    Args:
        Y_true: (N, V)
        Y_pred: (N, V)
        stim_indices: (N,) NSD image IDs
        n_examples: number of examples to show
    """
    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))

    # Select examples with highest pattern correlation
    from src.evaluation.metrics import pattern_correlation
    pat_corrs = pattern_correlation(Y_true, Y_pred)
    best_idx = np.argsort(pat_corrs)[::-1][:n_examples]

    for row, idx in enumerate(best_idx):
        axes[row, 0].plot(Y_true[idx], alpha=0.7)
        axes[row, 0].set_title(f"Actual (stim {stim_indices[idx]})")
        axes[row, 0].set_ylabel("Beta")

        axes[row, 1].plot(Y_pred[idx], alpha=0.7, color="coral")
        axes[row, 1].set_title(f"Predicted (r={pat_corrs[idx]:.3f})")

    for ax in axes[-1]:
        ax.set_xlabel("Voxel index")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
