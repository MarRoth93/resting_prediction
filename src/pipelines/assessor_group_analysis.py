"""Exploratory motion-uncorrected analysis of frozen-assessor score pickles."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import theilslopes


logger = logging.getLogger(__name__)

DIMENSIONS = (
    "valence",
    "arousal",
    "approach",
    "attention",
    "control",
    "dominance",
)
GROUPS = ("healthy", "depressed")
BOOTSTRAP_SAMPLES = 10_000
FIGURE_SUFFIX = "EXPLORATORY — motion-uncorrected"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _write_pickle_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    os.replace(temporary, path)


def _load_groups(path: Path) -> dict[str, str]:
    groups = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [value.strip() for value in next(reader)]
        except StopIteration:
            raise ValueError(f"Empty FOR group CSV: {path}") from None
        if "subject" not in header or "group" not in header:
            raise ValueError("FOR group CSV must contain subject and group columns.")
        subject_index = header.index("subject")
        group_index = header.index("group")
        for row in reader:
            if not row or not any(value.strip() for value in row):
                continue
            if max(subject_index, group_index) >= len(row):
                raise ValueError(f"Invalid FOR group row: {row}")
            subject = row[subject_index].strip()
            group = row[group_index].strip().lower()
            if (
                not subject.startswith("sub-")
                or group not in GROUPS
                or subject in groups
            ):
                raise ValueError(f"Invalid or duplicate FOR group row: {row}")
            groups[subject] = group
    counts = Counter(groups.values())
    if counts != Counter({"healthy": 25, "depressed": 25}):
        raise ValueError(
            "FOR group CSV must contain exactly 25 healthy and 25 depressed "
            f"subjects; found {dict(counts)}."
        )
    return groups


def _load_score_pickle(
    path: Path,
    *,
    expected_subject: str,
    require_ood: bool,
) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid score pickle for {expected_subject}: {path}")
    if payload.get("subject") != expected_subject:
        raise ValueError(
            f"Score pickle subject mismatch for {expected_subject}: {path}"
        )

    stimulus_ids = np.asarray(payload.get("stimulus_ids"))
    if stimulus_ids.ndim != 1 or stimulus_ids.size == 0:
        raise ValueError(f"Incomplete stimulus_ids for {expected_subject}.")
    stimulus_ids = np.asarray(stimulus_ids, dtype=np.int64)
    if np.unique(stimulus_ids).size != stimulus_ids.size:
        raise ValueError(f"Duplicate stimulus_ids for {expected_subject}.")

    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError(f"Incomplete scores for {expected_subject}.")
    scores = {}
    for dimension in DIMENSIONS:
        if dimension not in raw_scores:
            raise ValueError(f"Incomplete scores for {expected_subject}/{dimension}.")
        values = np.asarray(raw_scores[dimension], dtype=np.float64)
        if values.shape != stimulus_ids.shape:
            raise ValueError(
                f"Incomplete scores for {expected_subject}/{dimension}: "
                f"expected shape {stimulus_ids.shape}, got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{expected_subject}/{dimension} contains NaN/Inf.")
        scores[dimension] = values

    extras = payload.get("va_extras", {})
    normalized_extras = {}
    for dimension in ("valence", "arousal"):
        raw_dimension = extras.get(dimension, {}) if isinstance(extras, dict) else {}
        normalized_extras[dimension] = {}
        for name, raw_values in raw_dimension.items():
            values = np.asarray(raw_values, dtype=np.float64)
            if values.shape != stimulus_ids.shape:
                raise ValueError(
                    f"Incomplete {name} for {expected_subject}/{dimension}: "
                    f"expected shape {stimulus_ids.shape}, got {values.shape}."
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{expected_subject}/{dimension}/{name} contains NaN/Inf."
                )
            normalized_extras[dimension][name] = values
    if require_ood and "ood_percentile" not in normalized_extras["valence"]:
        raise ValueError(f"Missing VA OOD percentiles for {expected_subject}.")
    return {
        "subject": expected_subject,
        "stimulus_ids": stimulus_ids,
        "scores": scores,
        "va_extras": normalized_extras,
    }


def _align_record(record: dict, stimulus_ids: np.ndarray) -> dict:
    subject = record["subject"]
    own_ids = record["stimulus_ids"]
    if set(own_ids.tolist()) != set(stimulus_ids.tolist()):
        raise ValueError(
            f"Incomplete scores for {subject}: stimulus IDs do not match originals."
        )
    positions = {int(stimulus): index for index, stimulus in enumerate(own_ids)}
    order = np.asarray([positions[int(stimulus)] for stimulus in stimulus_ids])
    return {
        "subject": subject,
        "stimulus_ids": np.asarray(stimulus_ids, dtype=np.int64),
        "scores": {
            dimension: values[order] for dimension, values in record["scores"].items()
        },
        "va_extras": {
            dimension: {
                name: values[order] for name, values in dimension_values.items()
            }
            for dimension, dimension_values in record["va_extras"].items()
        },
    }


def _bh_qvalues(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("BH p-values must be a finite one-dimensional vector.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("BH p-values must be between zero and one.")
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def _cohens_d(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    pooled_variance = (
        (first.size - 1) * first.var(ddof=1)
        + (second.size - 1) * second.var(ddof=1)
    ) / (first.size + second.size - 2)
    difference = float(first.mean() - second.mean())
    if pooled_variance <= 0.0:
        return 0.0 if difference == 0.0 else None
    return float(difference / np.sqrt(pooled_variance))


def _subject_contrast(
    depressed: np.ndarray,
    healthy: np.ndarray,
    *,
    permutations: int,
    rng: np.random.RandomState,
) -> dict:
    depressed = np.asarray(depressed, dtype=float)
    healthy = np.asarray(healthy, dtype=float)
    observed = float(depressed.mean() - healthy.mean())
    combined = np.concatenate([depressed, healthy])
    exceed = 0
    for _ in range(int(permutations)):
        shuffled = combined[rng.permutation(combined.size)]
        difference = (
            shuffled[: depressed.size].mean() - shuffled[depressed.size :].mean()
        )
        exceed += int(abs(difference) >= abs(observed))

    depressed_indices = rng.randint(
        0, depressed.size, size=(BOOTSTRAP_SAMPLES, depressed.size)
    )
    healthy_indices = rng.randint(
        0, healthy.size, size=(BOOTSTRAP_SAMPLES, healthy.size)
    )
    bootstrap = (
        depressed[depressed_indices].mean(axis=1)
        - healthy[healthy_indices].mean(axis=1)
    )
    return {
        "n_depressed": int(depressed.size),
        "n_healthy": int(healthy.size),
        "mean_depressed_delta": float(depressed.mean()),
        "mean_healthy_delta": float(healthy.mean()),
        "difference_depressed_minus_healthy": observed,
        "cohens_d": _cohens_d(depressed, healthy),
        "permutation_p": float((exceed + 1) / (int(permutations) + 1)),
        "bootstrap_ci95_low": float(np.percentile(bootstrap, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(bootstrap, 97.5)),
    }


def _per_image_profile(
    depressed: np.ndarray,
    healthy: np.ndarray,
    *,
    permutations: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, float, float, float]:
    depressed = np.asarray(depressed, dtype=float)
    healthy = np.asarray(healthy, dtype=float)
    observed = depressed.mean(axis=0) - healthy.mean(axis=0)
    combined = np.concatenate([depressed, healthy], axis=0)
    null = np.empty((int(permutations), observed.size), dtype=np.float32)
    for index in range(int(permutations)):
        order = rng.permutation(combined.shape[0])
        null[index] = (
            combined[order[: depressed.shape[0]]].mean(axis=0)
            - combined[order[depressed.shape[0] :]].mean(axis=0)
        )
    threshold = float(np.percentile(np.abs(null), 95.0))
    share = float(np.mean(np.abs(observed) > threshold))
    return np.asarray(observed, dtype=np.float32), -threshold, threshold, share


def _robust_trend(x: np.ndarray, y: np.ndarray) -> dict | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or float(np.ptp(x)) == 0.0:
        return None
    slope, intercept, _, _ = theilslopes(y, x)
    return {"slope": float(slope), "intercept": float(intercept)}


def _finish_figure(fig, path: Path, title: str) -> None:
    fig.suptitle(f"{title}\n{FIGURE_SUFFIX}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _strip_x(center: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([center])
    return center + np.linspace(-0.08, 0.08, count)


def _plot_subject_deltas(
    subject_rows: list[dict],
    dimension_stats: dict,
    reference_rows: dict[str, dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, dimension in zip(axes.ravel(), DIMENSIONS, strict=True):
        healthy = np.asarray(
            [
                row["mean_delta"]
                for row in subject_rows
                if row["dimension"] == dimension and row["group"] == "healthy"
            ]
        )
        depressed = np.asarray(
            [
                row["mean_delta"]
                for row in subject_rows
                if row["dimension"] == dimension and row["group"] == "depressed"
            ]
        )
        boxes = axis.boxplot(
            [healthy, depressed],
            positions=[0, 1],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(boxes["boxes"], ("#4C78A8", "#E45756"), strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        axis.scatter(_strip_x(0, healthy.size), healthy, s=24, color="#4C78A8")
        axis.scatter(_strip_x(1, depressed.size), depressed, s=24, color="#E45756")
        ticks = [0, 1]
        labels = ["Healthy", "Depressed"]
        if dimension in reference_rows:
            axis.scatter(
                [2],
                [reference_rows[dimension]["mean_delta"]],
                marker="*",
                s=130,
                color="#54A24B",
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
            )
            ticks.append(2)
            labels.append("subj07\nreference")
        axis.set_xticks(ticks, labels)
        axis.set_title(
            dimension.title()
            + (" (exploratory)" if dimension == "dominance" else "")
        )
        axis.set_ylabel("Mean reconstruction − original")
        stats = dimension_stats[dimension]
        d_text = "undefined" if stats["cohens_d"] is None else f"{stats['cohens_d']:.2f}"
        axis.text(
            0.03,
            0.97,
            f"permutation p={stats['permutation_p']:.4g}\nd={d_text}",
            transform=axis.transAxes,
            va="top",
        )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    _finish_figure(fig, output_path, "Subject-level mean-delta distributions")


def _plot_per_image_histograms(
    profiles: dict[str, dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, dimension in zip(axes.ravel(), DIMENSIONS, strict=True):
        profile = profiles[dimension]
        values = profile["differences"]
        bins = min(30, max(5, int(np.sqrt(values.size))))
        axis.hist(values, bins=bins, color="#7A5195", alpha=0.8)
        axis.axvspan(
            profile["null_band_low"],
            profile["null_band_high"],
            color="#B8DE29",
            alpha=0.2,
            label="Permutation-null 95% band",
        )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(
            dimension.title()
            + (" (exploratory)" if dimension == "dominance" else "")
        )
        axis.set_xlabel("Depressed − healthy score")
        axis.set_ylabel("Images")
        axis.text(
            0.03,
            0.97,
            f"Outside band: {profile['share_outside_null_95']:.1%}",
            transform=axis.transAxes,
            va="top",
        )
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    _finish_figure(fig, output_path, "Per-image group-difference profiles")


def _plot_content_dependence(
    originals: dict,
    profiles: dict[str, dict],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, dimension in zip(axes.ravel(), DIMENSIONS, strict=True):
        original = originals["scores"][dimension]
        difference = profiles[dimension]["differences"]
        axis.scatter(original, difference, s=15, alpha=0.55, color="#4C78A8")
        trend = profiles[dimension]["robust_trend"]
        if trend is not None:
            endpoints = np.asarray([original.min(), original.max()])
            axis.plot(
                endpoints,
                trend["intercept"] + trend["slope"] * endpoints,
                color="#E45756",
                linewidth=2,
                label="Theil–Sen trend",
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(
            dimension.title()
            + (" (exploratory)" if dimension == "dominance" else "")
        )
        axis.set_xlabel("Original image score")
        axis.set_ylabel("Depressed − healthy")
    if any(profiles[dimension]["robust_trend"] is not None for dimension in DIMENSIONS):
        axes.ravel()[0].legend(loc="best", fontsize=8)
    _finish_figure(
        fig,
        output_path,
        "Image-content dependence of group differences",
    )


def _plot_ood_distributions(
    ood_by_group: dict[str, np.ndarray],
    reference_ood: float | None,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    healthy = ood_by_group["healthy"]
    depressed = ood_by_group["depressed"]
    boxes = axis.boxplot(
        [healthy, depressed],
        positions=[0, 1],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(boxes["boxes"], ("#4C78A8", "#E45756"), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    axis.scatter(_strip_x(0, healthy.size), healthy, s=28, color="#4C78A8")
    axis.scatter(_strip_x(1, depressed.size), depressed, s=28, color="#E45756")
    ticks = [0, 1]
    labels = ["Healthy", "Depressed"]
    if reference_ood is not None:
        axis.scatter(
            [2],
            [reference_ood],
            marker="*",
            s=150,
            color="#54A24B",
            edgecolor="black",
            linewidth=0.5,
        )
        ticks.append(2)
        labels.append("subj07\nreference")
    axis.set_xticks(ticks, labels)
    axis.set_ylabel("Subject mean VA OOD percentile")
    _finish_figure(fig, output_path, "VA out-of-distribution quality control")


def analyze_scores(
    *,
    scores_root: Path,
    groups_csv: Path,
    output_root: Path,
    permutations: int,
    seed: int,
) -> dict:
    if permutations <= 0:
        raise ValueError(f"permutations must be positive, got {permutations}.")
    groups = _load_groups(groups_csv)
    originals_path = scores_root / "originals.pkl"
    if not originals_path.is_file():
        raise FileNotFoundError(
            f"Missing original-image scores: {originals_path}. Run "
            "`python -m src.pipelines.assessor_score_reconstructions "
            "--include-originals` first."
        )
    originals = _load_score_pickle(
        originals_path, expected_subject="originals", require_ood=False
    )
    stimulus_ids = originals["stimulus_ids"]

    score_paths = {path.stem: path for path in sorted(scores_root.glob("sub-*.pkl"))}
    extra_subjects = sorted(set(score_paths) - set(groups))
    if extra_subjects:
        raise ValueError(f"Score pickles have no FOR group row: {extra_subjects}")
    missing_for_subjects = sorted(set(groups) - set(score_paths))
    for subject in missing_for_subjects:
        logger.warning("Excluding %s: score pickle is missing.", subject)

    records = {}
    for subject, path in score_paths.items():
        record = _load_score_pickle(
            path, expected_subject=subject, require_ood=True
        )
        records[subject] = _align_record(record, stimulus_ids)
    included_counts = Counter(groups[subject] for subject in records)
    if any(included_counts[group] < 2 for group in GROUPS):
        raise ValueError(
            "At least two scored subjects per FOR group are required; "
            f"found {dict(included_counts)}."
        )

    reference_path = scores_root / "subj07.pkl"
    reference = None
    missing_subjects = list(missing_for_subjects)
    if reference_path.is_file():
        reference = _align_record(
            _load_score_pickle(
                reference_path, expected_subject="subj07", require_ood=True
            ),
            stimulus_ids,
        )
    else:
        logger.warning("Excluding subj07 reference: score pickle is missing.")
        missing_subjects.append("subj07")

    subject_rows = []
    for subject in sorted(records):
        group = groups[subject]
        for dimension in DIMENSIONS:
            raw = records[subject]["scores"][dimension]
            original = originals["scores"][dimension]
            subject_rows.append(
                {
                    "subject": subject,
                    "group": group,
                    "dimension": dimension,
                    "n_images": int(stimulus_ids.size),
                    "mean_raw_score": float(raw.mean()),
                    "mean_delta": float((raw - original).mean()),
                    "dominance_exploratory": dimension == "dominance",
                }
            )

    reference_rows = {}
    if reference is not None:
        for dimension in DIMENSIONS:
            raw = reference["scores"][dimension]
            original = originals["scores"][dimension]
            reference_rows[dimension] = {
                "subject": "subj07",
                "group": "nsd_reference",
                "dimension": dimension,
                "n_images": int(stimulus_ids.size),
                "mean_raw_score": float(raw.mean()),
                "mean_delta": float((raw - original).mean()),
                "dominance_exploratory": dimension == "dominance",
            }

    dimension_stats = {}
    profiles = {}
    for dimension_index, dimension in enumerate(DIMENSIONS):
        rng = np.random.RandomState(int(seed) + dimension_index)
        depressed_rows = [
            row
            for row in subject_rows
            if row["dimension"] == dimension and row["group"] == "depressed"
        ]
        healthy_rows = [
            row
            for row in subject_rows
            if row["dimension"] == dimension and row["group"] == "healthy"
        ]
        depressed_delta = np.asarray([row["mean_delta"] for row in depressed_rows])
        healthy_delta = np.asarray([row["mean_delta"] for row in healthy_rows])
        stats = _subject_contrast(
            depressed_delta,
            healthy_delta,
            permutations=permutations,
            rng=rng,
        )
        stats.update(
            {
                "mean_depressed_raw_score": float(
                    np.mean([row["mean_raw_score"] for row in depressed_rows])
                ),
                "mean_healthy_raw_score": float(
                    np.mean([row["mean_raw_score"] for row in healthy_rows])
                ),
                "dominance_exploratory": dimension == "dominance",
            }
        )

        depressed_images = np.stack(
            [
                records[row["subject"]]["scores"][dimension]
                for row in depressed_rows
            ]
        )
        healthy_images = np.stack(
            [records[row["subject"]]["scores"][dimension] for row in healthy_rows]
        )
        differences, band_low, band_high, share = _per_image_profile(
            depressed_images,
            healthy_images,
            permutations=permutations,
            rng=rng,
        )
        trend = _robust_trend(originals["scores"][dimension], differences)
        profiles[dimension] = {
            "differences": differences,
            "null_band_low": band_low,
            "null_band_high": band_high,
            "share_outside_null_95": share,
            "robust_trend": trend,
        }
        stats["per_image_profile"] = {
            "n_images": int(differences.size),
            "null_band_low": band_low,
            "null_band_high": band_high,
            "share_outside_null_95": share,
            "robust_trend": trend,
        }
        dimension_stats[dimension] = stats

    qvalues = _bh_qvalues(
        [dimension_stats[dimension]["permutation_p"] for dimension in DIMENSIONS]
    )
    for dimension, qvalue in zip(DIMENSIONS, qvalues, strict=True):
        dimension_stats[dimension]["bh_adjusted_p"] = float(qvalue)

    ood_by_group = {
        group: np.asarray(
            [
                records[subject]["va_extras"]["valence"]["ood_percentile"].mean()
                for subject in sorted(records)
                if groups[subject] == group
            ]
        )
        for group in GROUPS
    }
    reference_ood = (
        None
        if reference is None
        else float(
            reference["va_extras"]["valence"]["ood_percentile"].mean()
        )
    )

    output_root.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "fig1": output_root / "fig1_subject_mean_deltas.png",
        "fig2": output_root / "fig2_per_image_differences.png",
        "fig3": output_root / "fig3_content_dependence.png",
        "fig4": output_root / "fig4_ood_percentiles.png",
    }
    _plot_subject_deltas(
        subject_rows, dimension_stats, reference_rows, figure_paths["fig1"]
    )
    _plot_per_image_histograms(profiles, figure_paths["fig2"])
    _plot_content_dependence(originals, profiles, figure_paths["fig3"])
    _plot_ood_distributions(ood_by_group, reference_ood, figure_paths["fig4"])

    per_image_path = output_root / "per_image_differences.pkl"
    _write_pickle_atomic(
        per_image_path,
        {
            "status": "exploratory",
            "motion_corrected": False,
            "outside_frozen_endpoint_families": True,
            "owner_requested": "2026-08-14",
            "stimulus_ids": np.asarray(stimulus_ids, dtype=np.int64),
            "differences": {
                dimension: profiles[dimension]["differences"]
                for dimension in DIMENSIONS
            },
            "permutation_null_95_bands": {
                dimension: {
                    "low": profiles[dimension]["null_band_low"],
                    "high": profiles[dimension]["null_band_high"],
                }
                for dimension in DIMENSIONS
            },
            "permutations": int(permutations),
            "seed": int(seed),
        },
    )

    summary = {
        "status": "exploratory",
        "motion_corrected": False,
        "outside_frozen_endpoint_families": True,
        "owner_requested": "2026-08-14",
        "dominance_note": (
            "Dominance is exploratory because the frozen assessor documents it as "
            "unreliable."
        ),
        "group_contrast": "depressed_minus_healthy_on_subject_mean_delta",
        "group_counts_csv": {group: 25 for group in GROUPS},
        "ns": {
            "healthy_subjects_included": int(included_counts["healthy"]),
            "depressed_subjects_included": int(included_counts["depressed"]),
            "images_per_subject": int(stimulus_ids.size),
            "nsd_reference_subjects": int(reference is not None),
        },
        "seeds": {
            "seed": int(seed),
            "permutations": int(permutations),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "dimension_seed_offsets": {
                dimension: index for index, dimension in enumerate(DIMENSIONS)
            },
        },
        "missing_subjects": sorted(missing_subjects),
        "missing_for_subjects": missing_for_subjects,
        "dimensions": dimension_stats,
        "subject_level": subject_rows,
        "nsd_reference": reference_rows,
        "ood_qc": {
            "unit": "subject_mean_va_ood_percentile",
            "mean_healthy": float(ood_by_group["healthy"].mean()),
            "mean_depressed": float(ood_by_group["depressed"].mean()),
            "subj07_reference": reference_ood,
        },
        "outputs": {
            "figures": {name: str(path) for name, path in figure_paths.items()},
            "per_image_differences": str(per_image_path),
            "summary": str(output_root / "summary.json"),
        },
    }
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-root",
        default="artifacts/recon_from_predictions/seed42/assessor_scores",
    )
    parser.add_argument("--groups-csv", default="for_groups.csv")
    parser.add_argument(
        "--output-root",
        default=(
            "artifacts/recon_from_predictions/seed42/analysis_exploratory"
        ),
    )
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = analyze_scores(
        scores_root=Path(args.scores_root).resolve(),
        groups_csv=Path(args.groups_csv).resolve(),
        output_root=Path(args.output_root).resolve(),
        permutations=int(args.permutations),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
