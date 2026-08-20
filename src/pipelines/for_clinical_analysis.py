"""Frozen Phase-5 label-blind endpoints and clinical inference pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest, spearmanr

from src.alignment.rest_preprocessing import compute_rest_connectivity
from src.alignment.shared_space import SharedSpaceBuilder
from src.analysis.permutation import (
    bh_adjust,
    build_permutation_schedule,
    permutation_pvalue_omnibus,
    permutation_pvalue_two_sided,
    stratified_bootstrap,
)
from src.pipelines.multiexpert_artifacts import file_sha256


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FOR_INFERENCE_ROOT = REPO_ROOT / "artifacts/for_inference/seed42"
CONTRACT_FOR_ROOT = REPO_ROOT / "data/processed_voxel_contract/for"
FINAL_MODEL_DIR = REPO_ROOT / "artifacts/voxel_contract_final/seed42/model"
OUTPUT_ROOT = REPO_ROOT / "artifacts/for_clinical_analysis/seed42"
FREEZE_PATH = REPO_ROOT / "analysis_registry/phase5_freeze.json"
LABELS_PATH = REPO_ROOT / "for_groups.csv"
ANNOT_PATHS = (
    REPO_ROOT
    / "data/processed_schaefer400/_registration_assets/annotations/"
    "lh.Schaefer2018_400Parcels_7Networks_order.annot",
    REPO_ROOT
    / "data/processed_schaefer400/_registration_assets/annotations/"
    "rh.Schaefer2018_400Parcels_7Networks_order.annot",
)
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 42
BOOTSTRAP_SEED = 1042
N_BOOTSTRAP = 10_000
YEO_NETWORKS = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)
EXPECTED_NETWORK_COUNTS = {
    "Vis": 61,
    "SomMot": 77,
    "DorsAttn": 46,
    "SalVentAttn": 47,
    "Limbic": 26,
    "Cont": 52,
    "Default": 91,
}
FIGURE_SUFFIX = "EXPLORATORY — motion-uncorrected"
DVARS_DEFINITION = (
    "mean over t of sqrt(mean over voxels of (x_t − x_{t−1})²) computed "
    "on per-voxel z-scored rest_targets"
)
ENDPOINT_DEFINITIONS = {
    "fingerprint_residual_global": (
        "||F_s − T||_F / ||T||_F. NO Procrustes refit."
    ),
    "fingerprint_residual_network": (
        "||F_net − T_net||_F / ||T_net||_F."
    ),
    "offset_share": (
        "D = F_s − T; O = row-broadcast of column means of D (mean over the "
        "400 rows per latent column); offset_share = ||O||_F² / ||D||_F²."
    ),
    "fingerprint_residual_centered": "||D − O||_F / ||T||_F.",
    "connectivity_blocks": (
        "Pearson correlation of rest_seeds columns, clipped to ±0.999999, "
        "Fisher-z transformed; self-edges excluded, strict upper triangle "
        "within networks, all cross edges between networks"
    ),
}


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing file: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _require_finite(subject: str, name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{subject}: {name} contains NaN/Inf.")


def _canonical_sha(payload: dict) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _git_output(*args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _git_commit() -> str | None:
    output = _git_output("rev-parse", "HEAD")
    return None if output is None else output.decode("utf-8").strip()


def _block_names() -> list[str]:
    return [
        f"{first}|{second}"
        for first_index, first in enumerate(YEO_NETWORKS)
        for second in YEO_NETWORKS[first_index:]
    ]


def _build_yeo7_mapping(
    annot_paths: Sequence[Path] | None = None,
) -> dict:
    """Read and validate the frozen Schaefer-400 to Yeo-7 parcel mapping."""
    from nibabel.freesurfer import read_annot

    paths = tuple(Path(path) for path in (ANNOT_PATHS if annot_paths is None else annot_paths))
    if len(paths) != 2:
        raise ValueError(f"Expected two annotation files, got {len(paths)}.")

    parcel_to_network: dict[str, str] = {}
    for path, hemisphere, offset in zip(
        paths, ("LH", "RH"), (0, 200), strict=True
    ):
        _, _, names = read_annot(path)
        if len(names) != 201:
            raise ValueError(
                f"{hemisphere} annotation must contain background plus 200 "
                f"parcel names, got {len(names)} names."
            )
        for local_id, raw_name in enumerate(names[1:], start=1):
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            tokens = name.split("_")
            if len(tokens) < 3:
                raise ValueError(f"Invalid annotation parcel name: {name}")
            network = tokens[2]
            if network not in YEO_NETWORKS:
                raise ValueError(
                    f"Unknown Yeo-7 network {network!r} in annotation name {name!r}."
                )
            parcel_to_network[str(offset + local_id)] = network

    if len(parcel_to_network) != 400:
        raise ValueError(
            f"Yeo-7 mapping must contain exactly 400 parcels, got "
            f"{len(parcel_to_network)}."
        )
    counts = Counter(parcel_to_network.values())
    observed_counts = {network: int(counts[network]) for network in YEO_NETWORKS}
    if observed_counts != EXPECTED_NETWORK_COUNTS:
        raise ValueError(
            "Yeo-7 network counts do not match the frozen counts: "
            f"expected {EXPECTED_NETWORK_COUNTS}, got {observed_counts}."
        )

    block_unique_edge_counts = {}
    for first_index, first in enumerate(YEO_NETWORKS):
        for second in YEO_NETWORKS[first_index:]:
            first_count = observed_counts[first]
            second_count = observed_counts[second]
            block_unique_edge_counts[f"{first}|{second}"] = (
                first_count * (first_count - 1) // 2
                if first == second
                else first_count * second_count
            )
    return {
        "parcel_to_network": parcel_to_network,
        "networks": list(YEO_NETWORKS),
        "network_counts": observed_counts,
        "block_names": _block_names(),
        "block_unique_edge_counts": block_unique_edge_counts,
        "annotation_sha256": {
            "lh": file_sha256(paths[0]),
            "rh": file_sha256(paths[1]),
        },
    }


def _parcel_networks(mapping: dict) -> np.ndarray:
    parcel_to_network = mapping["parcel_to_network"]
    return np.asarray(
        [parcel_to_network[str(parcel_id)] for parcel_id in range(1, 401)]
    )


def _average_connectivity_blocks(
    fisher_z: np.ndarray,
    parcel_networks: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fisher_z, dtype=np.float64)
    networks = np.asarray(parcel_networks)
    if values.shape != (400, 400):
        raise ValueError(
            f"Fisher-z connectivity must have shape (400, 400), got {values.shape}."
        )
    if networks.shape != (400,):
        raise ValueError(
            f"parcel_networks must have shape (400,), got {networks.shape}."
        )
    blocks = []
    for first_index, first in enumerate(YEO_NETWORKS):
        first_rows = np.flatnonzero(networks == first)
        for second in YEO_NETWORKS[first_index:]:
            second_rows = np.flatnonzero(networks == second)
            if first == second:
                block = values[np.ix_(first_rows, first_rows)]
                edges = block[np.triu_indices(first_rows.size, k=1)]
            else:
                edges = values[np.ix_(first_rows, second_rows)].ravel()
            if edges.size == 0 or not np.all(np.isfinite(edges)):
                raise ValueError(f"Invalid or empty connectivity block {first}|{second}.")
            blocks.append(float(edges.mean()))
    result = np.asarray(blocks, dtype=np.float64)
    if result.shape != (28,) or not np.all(np.isfinite(result)):
        raise ValueError("Connectivity block averaging did not produce 28 finite values.")
    return result


def _compute_connectivity_blocks(
    rest_seeds: np.ndarray,
    parcel_networks: np.ndarray,
) -> np.ndarray:
    seeds = np.asarray(rest_seeds, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 400:
        raise ValueError(f"rest_seeds must have shape (T, 400), got {seeds.shape}.")
    correlation = np.corrcoef(seeds, rowvar=False)
    if not np.all(np.isfinite(correlation)):
        raise ValueError("rest_seeds correlation contains NaN/Inf.")
    fisher_z = np.arctanh(np.clip(correlation, -0.999999, 0.999999))
    if not np.all(np.isfinite(fisher_z)):
        raise ValueError("Fisher-z connectivity contains NaN/Inf.")
    return _average_connectivity_blocks(fisher_z, parcel_networks)


def _offset_decomposition(
    fingerprint: np.ndarray,
    template: np.ndarray,
) -> tuple[float, float]:
    fingerprint = np.asarray(fingerprint, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if fingerprint.shape != template.shape or fingerprint.ndim != 2:
        raise ValueError(
            "fingerprint and template must have the same two-dimensional shape."
        )
    difference = fingerprint - template
    offset = np.broadcast_to(difference.mean(axis=0), difference.shape)
    denominator = float(np.sum(difference**2))
    template_norm = float(np.linalg.norm(template))
    if denominator <= 0.0:
        raise ValueError("Fingerprint offset share is undefined for zero residual.")
    if template_norm <= 0.0:
        raise ValueError("Fingerprint residual is undefined for a zero template.")
    offset_share = float(np.sum(offset**2) / denominator)
    centered_residual = float(np.linalg.norm(difference - offset) / template_norm)
    if not np.isfinite(offset_share) or not np.isfinite(centered_residual):
        raise ValueError("Fingerprint offset decomposition produced NaN/Inf.")
    return offset_share, centered_residual


def _fingerprint_residual(
    fingerprint: np.ndarray,
    template: np.ndarray,
) -> float:
    denominator = float(np.linalg.norm(template))
    if denominator <= 0.0:
        raise ValueError("Fingerprint residual is undefined for a zero template.")
    value = float(np.linalg.norm(fingerprint - template) / denominator)
    if not np.isfinite(value):
        raise ValueError("Fingerprint residual produced NaN/Inf.")
    return value


def _fingerprint_network_residuals(
    fingerprint: np.ndarray,
    template: np.ndarray,
    parcel_networks: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            _fingerprint_residual(
                fingerprint[parcel_networks == network],
                template[parcel_networks == network],
            )
            for network in YEO_NETWORKS
        ],
        dtype=np.float64,
    )


def _mean_dvars(rest_targets: np.ndarray) -> float:
    values = np.asarray(rest_targets, dtype=np.float64)
    standard_deviation = values.std(axis=0)
    if np.any(standard_deviation <= 0.0):
        raise ValueError("rest_targets contains a constant voxel column.")
    standardized = (values - values.mean(axis=0)) / standard_deviation
    value = float(np.sqrt(np.mean(np.diff(standardized, axis=0) ** 2, axis=1)).mean())
    if not np.isfinite(value):
        raise ValueError("mean_dvars produced NaN/Inf.")
    return value


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


def _spearman(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return {"rho": None, "pvalue": None}
    result = spearmanr(first, second)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": rho if np.isfinite(rho) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
    }


def _subject_paths(
    subject: str,
    for_inference_root: Path,
    contract_for_root: Path,
) -> dict[str, Path]:
    return {
        "fingerprint": for_inference_root / subject / "fingerprint.npy",
        "alignment_P": for_inference_root / subject / "alignment_P.npy",
        "alignment_R": for_inference_root / subject / "alignment_R.npy",
        "rest_seeds": contract_for_root / subject / "rest_seeds.npy",
        "rest_targets": contract_for_root / subject / "rest_targets.npy",
    }


def build_endpoints(
    *,
    for_inference_root: Path = FOR_INFERENCE_ROOT,
    contract_for_root: Path = CONTRACT_FOR_ROOT,
    final_model_dir: Path = FINAL_MODEL_DIR,
    output_root: Path = OUTPUT_ROOT,
    annot_paths: Sequence[Path] | None = None,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """Construct and save all frozen endpoints without accessing labels."""
    for_inference_root = Path(for_inference_root)
    contract_for_root = Path(contract_for_root)
    final_model_dir = Path(final_model_dir)
    output_root = Path(output_root)
    subjects = sorted(
        path.name
        for path in for_inference_root.glob("sub-*")
        if path.is_dir()
    )
    if len(subjects) != 50:
        raise ValueError(
            f"Expected exactly 50 FOR inference subjects, found {len(subjects)}."
        )

    mapping = _build_yeo7_mapping(annot_paths)
    mapping_path = output_root / "yeo7_mapping.json"
    _write_json_atomic(mapping_path, mapping)
    parcel_networks = _parcel_networks(mapping)

    builder = SharedSpaceBuilder.load(str(final_model_dir))
    template = np.asarray(builder.template_fingerprint, dtype=np.float64)
    if template.shape != (400, 100):
        raise ValueError(
            "Final-model template_fingerprint must have shape (400, 100), "
            f"got {template.shape}."
        )
    _require_finite("template", "template_fingerprint", template)

    global_residuals = []
    network_residuals = []
    offset_shares = []
    centered_residuals = []
    connectivity_blocks = []
    connectivity_half_1 = []
    connectivity_half_2 = []
    fingerprint_half_1 = []
    fingerprint_half_2 = []
    target_widths = []
    mean_dvars = []
    input_hashes = {}

    for subject in subjects:
        paths = _subject_paths(subject, for_inference_root, contract_for_root)
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{subject}: missing {name}: {path}")
        fingerprint = np.asarray(np.load(paths["fingerprint"]), dtype=np.float64)
        alignment_p = np.asarray(np.load(paths["alignment_P"]), dtype=np.float64)
        alignment_r = np.asarray(np.load(paths["alignment_R"]), dtype=np.float64)
        rest_seeds = np.asarray(np.load(paths["rest_seeds"]), dtype=np.float64)
        rest_targets = np.asarray(np.load(paths["rest_targets"]), dtype=np.float64)

        expected_shapes = {
            "fingerprint": (400, 100),
            "rest_seeds": (235, 400),
        }
        observed = {"fingerprint": fingerprint.shape, "rest_seeds": rest_seeds.shape}
        for name, expected_shape in expected_shapes.items():
            if observed[name] != expected_shape:
                raise ValueError(
                    f"{subject}: {name} must have shape {expected_shape}, "
                    f"got {observed[name]}."
                )
        if rest_targets.ndim != 2 or rest_targets.shape[0] != 235 or rest_targets.shape[1] < 1:
            raise ValueError(
                f"{subject}: rest_targets must have shape (235, V_t), "
                f"got {rest_targets.shape}."
            )
        expected_p_shape = (rest_targets.shape[1], 100)
        if alignment_p.shape != expected_p_shape:
            raise ValueError(
                f"{subject}: alignment_P must have shape {expected_p_shape}, "
                f"got {alignment_p.shape}."
            )
        if alignment_r.shape != (100, 100):
            raise ValueError(
                f"{subject}: alignment_R must have shape (100, 100), "
                f"got {alignment_r.shape}."
            )
        for name, values in (
            ("fingerprint", fingerprint),
            ("alignment_P", alignment_p),
            ("alignment_R", alignment_r),
            ("rest_seeds", rest_seeds),
            ("rest_targets", rest_targets),
        ):
            _require_finite(subject, name, values)

        global_residuals.append(_fingerprint_residual(fingerprint, template))
        network_residuals.append(
            _fingerprint_network_residuals(fingerprint, template, parcel_networks)
        )
        offset_share, centered_residual = _offset_decomposition(fingerprint, template)
        offset_shares.append(offset_share)
        centered_residuals.append(centered_residual)
        connectivity_blocks.append(
            _compute_connectivity_blocks(rest_seeds, parcel_networks)
        )

        half_values = []
        half_fingerprints = []
        for start, stop in ((0, 117), (117, 235)):
            half_seeds = rest_seeds[start:stop]
            half_targets = rest_targets[start:stop]
            half_values.append(
                _compute_connectivity_blocks(half_seeds, parcel_networks)
            )
            half_connectivity = compute_rest_connectivity(
                [half_targets],
                seed_runs=[half_seeds],
                ensemble=builder.ensemble_method,
            )
            half_fingerprint = half_connectivity @ alignment_p @ alignment_r
            _require_finite(subject, "half fingerprint", half_fingerprint)
            half_fingerprints.append(
                _fingerprint_residual(half_fingerprint, template)
            )
        connectivity_half_1.append(half_values[0])
        connectivity_half_2.append(half_values[1])
        fingerprint_half_1.append(half_fingerprints[0])
        fingerprint_half_2.append(half_fingerprints[1])
        target_widths.append(rest_targets.shape[1])
        mean_dvars.append(_mean_dvars(rest_targets))
        input_hashes[subject] = {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        }

    arrays = {
        "subjects": np.asarray(subjects),
        "fingerprint_residual_global": np.asarray(global_residuals, dtype=np.float64),
        "fingerprint_residual_network": np.asarray(network_residuals, dtype=np.float64),
        "offset_share": np.asarray(offset_shares, dtype=np.float64),
        "fingerprint_residual_centered": np.asarray(centered_residuals, dtype=np.float64),
        "connectivity_blocks": np.asarray(connectivity_blocks, dtype=np.float64),
        "connectivity_blocks_half1": np.asarray(connectivity_half_1, dtype=np.float64),
        "connectivity_blocks_half2": np.asarray(connectivity_half_2, dtype=np.float64),
        "fingerprint_residual_half1": np.asarray(fingerprint_half_1, dtype=np.float64),
        "fingerprint_residual_half2": np.asarray(fingerprint_half_2, dtype=np.float64),
        "v_t": np.asarray(target_widths, dtype=np.int64),
        "mean_dvars": np.asarray(mean_dvars, dtype=np.float64),
    }
    for name, values in arrays.items():
        if name != "subjects":
            _require_finite("all subjects", name, values)

    endpoints_path = output_root / "endpoints.npz"
    _write_npz_atomic(endpoints_path, **arrays)
    schedule = build_permutation_schedule(
        50, int(n_permutations), PERMUTATION_SEED
    )
    schedule_path = output_root / "permutation_schedule.npy"
    _write_npy_atomic(schedule_path, schedule)

    block_reliability = {
        block_name: _pearson(
            arrays["connectivity_blocks_half1"][:, index],
            arrays["connectivity_blocks_half2"][:, index],
        )
        for index, block_name in enumerate(_block_names())
    }
    coverage = {
        "fingerprint_residual_global_vs_v_t": _spearman(
            arrays["fingerprint_residual_global"], arrays["v_t"]
        ),
        "fingerprint_residual_global_vs_mean_dvars": _spearman(
            arrays["fingerprint_residual_global"], arrays["mean_dvars"]
        ),
        "fingerprint_residual_centered_vs_v_t": _spearman(
            arrays["fingerprint_residual_centered"], arrays["v_t"]
        ),
        "fingerprint_residual_centered_vs_mean_dvars": _spearman(
            arrays["fingerprint_residual_centered"], arrays["mean_dvars"]
        ),
        "connectivity_blocks_vs_mean_dvars": {
            block_name: _spearman(
                arrays["connectivity_blocks"][:, index], arrays["mean_dvars"]
            )
            for index, block_name in enumerate(_block_names())
        },
    }
    builder_path = final_model_dir / "builder.npz"
    manifest = {
        "subjects": subjects,
        "n_subjects": len(subjects),
        "input_files": input_hashes,
        "template": {
            "source": str(builder_path),
            "builder_npz_sha256": file_sha256(builder_path),
            "shape": list(template.shape),
        },
        "yeo7_mapping_sha256": file_sha256(mapping_path),
        "permutation_schedule_sha256": file_sha256(schedule_path),
        "n_permutations": int(n_permutations),
        "permutation_seed": PERMUTATION_SEED,
        "endpoint_definitions": ENDPOINT_DEFINITIONS,
        "dvars_definition": DVARS_DEFINITION,
        "reliability": {
            "contiguous_halves": ["rows 0:117", "rows 117:235"],
            "connectivity_blocks_pearson": block_reliability,
            "fingerprint_residual_pearson": _pearson(
                arrays["fingerprint_residual_half1"],
                arrays["fingerprint_residual_half2"],
            ),
        },
        "coverage_spearman": coverage,
        "code_hint_git_rev_parse_head": _git_commit(),
    }
    manifest_path = output_root / "endpoints_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    logger.info("Wrote frozen label-free endpoints to %s", output_root)
    return manifest


def _load_endpoints(output_root: Path) -> dict[str, np.ndarray]:
    path = Path(output_root) / "endpoints.npz"
    try:
        with np.load(path) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing file: {path}") from None
    required_shapes = {
        "subjects": (50,),
        "fingerprint_residual_global": (50,),
        "fingerprint_residual_network": (50, 7),
        "offset_share": (50,),
        "fingerprint_residual_centered": (50,),
        "connectivity_blocks": (50, 28),
        "connectivity_blocks_half1": (50, 28),
        "connectivity_blocks_half2": (50, 28),
        "fingerprint_residual_half1": (50,),
        "fingerprint_residual_half2": (50,),
        "v_t": (50,),
        "mean_dvars": (50,),
    }
    missing = sorted(set(required_shapes) - set(arrays))
    if missing:
        raise ValueError(f"endpoints.npz is missing arrays: {missing}")
    for name, expected_shape in required_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"endpoints.npz {name} must have shape {expected_shape}, "
                f"got {arrays[name].shape}."
            )
        if name != "subjects":
            _require_finite("endpoints.npz", name, arrays[name])
    subjects = [str(value) for value in arrays["subjects"].tolist()]
    if subjects != sorted(subjects) or len(set(subjects)) != 50:
        raise ValueError("endpoints.npz subjects must be 50 sorted unique IDs.")
    arrays["subjects"] = np.asarray(subjects)
    return arrays


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    output_root = Path(output_root)
    return {
        "endpoints.npz": output_root / "endpoints.npz",
        "endpoints_manifest.json": output_root / "endpoints_manifest.json",
        "calibration.json": output_root / "calibration.json",
        "permutation_schedule.npy": output_root / "permutation_schedule.npy",
        "yeo7_mapping.json": output_root / "yeo7_mapping.json",
    }


def _hash_paths(paths: dict[str, Path]) -> dict[str, str]:
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")
        hashes[name] = file_sha256(path)
    return hashes


def calibrate_endpoints(
    *,
    output_root: Path = OUTPUT_ROOT,
    calibration_repeats: int = 200,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """Run label-free smoke and injection calibration on frozen endpoints."""
    if int(calibration_repeats) < 1:
        raise ValueError("calibration_repeats must be positive.")
    output_root = Path(output_root)
    endpoints = _load_endpoints(output_root)
    schedule_path = output_root / "permutation_schedule.npy"
    try:
        schedule = np.asarray(np.load(schedule_path), dtype=np.int64)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing file: {schedule_path}") from None
    if schedule.shape != (int(n_permutations), 50):
        raise ValueError(
            "permutation_schedule.npy must have shape "
            f"({int(n_permutations)}, 50), got {schedule.shape}."
        )

    smoke_rng = np.random.default_rng(4242)
    primary_pvalues = []
    omnibus_pvalues = []
    for _ in range(int(calibration_repeats)):
        mask = np.zeros(50, dtype=bool)
        mask[smoke_rng.permutation(50)[:25]] = True
        _, primary_p = permutation_pvalue_two_sided(
            endpoints["fingerprint_residual_global"], mask, schedule
        )
        _, omnibus_p = permutation_pvalue_omnibus(
            endpoints["connectivity_blocks"], mask, schedule
        )
        primary_pvalues.append(float(primary_p[0]))
        omnibus_pvalues.append(float(omnibus_p))

    bins = np.linspace(0.0, 1.0, 11)

    def uniformity(values: list[float]) -> dict:
        histogram, _ = np.histogram(values, bins=bins)
        ks_result = kstest(values, "uniform")
        return {
            "pvalues": values,
            "decile_histogram": histogram.astype(int).tolist(),
            "ks_statistic": float(ks_result.statistic),
            "ks_pvalue": float(ks_result.pvalue),
            "descriptive_only": True,
        }

    injection_rng = np.random.default_rng(777)
    global_values = np.asarray(
        endpoints["fingerprint_residual_global"], dtype=np.float64
    )
    injection_shift = float(global_values.std(ddof=1))
    injection_pvalues = []
    for _ in range(int(calibration_repeats)):
        mask = np.zeros(50, dtype=bool)
        mask[injection_rng.permutation(50)[:25]] = True
        injected = global_values.copy()
        injected[mask] += injection_shift
        _, pvalue = permutation_pvalue_two_sided(injected, mask, schedule)
        injection_pvalues.append(float(pvalue[0]))

    bound_paths = _artifact_paths(output_root)
    del bound_paths["calibration.json"]
    calibration = {
        "calibration_repeats": int(calibration_repeats),
        "n_permutations": int(n_permutations),
        "permutation_seed": PERMUTATION_SEED,
        "bound_sha256": _hash_paths(bound_paths),
        "smoke_uniformity": {
            "split": "K random 25/25 splits",
            "rng_seed": 4242,
            "primary_1": uniformity(primary_pvalues),
            "primary_2_omnibus": uniformity(omnibus_pvalues),
            "dependence_note": (
                "pseudo-replicates are dependent; descriptive only; no gate"
            ),
        },
        "injection": {
            "split": "K random 25/25 splits",
            "rng_seed": 777,
            "effect": "add 1.0 * std(global residual) to group A",
            "pvalues": injection_pvalues,
            "rejection_rate_alpha05": float(
                np.mean(np.asarray(injection_pvalues) < 0.05)
            ),
            "note": (
                "expected ≥ 0.85 for d≈1.0 at n=25/25; informational"
            ),
        },
    }
    _write_json_atomic(output_root / "calibration.json", calibration)
    logger.info("Wrote label-free calibration to %s", output_root / "calibration.json")
    return calibration


def freeze_analysis(
    *,
    output_root: Path = OUTPUT_ROOT,
    freeze_path: Path = FREEZE_PATH,
    force: bool = False,
) -> dict:
    """Write the git-trackable pre-label freeze anchor."""
    output_root = Path(output_root)
    freeze_path = Path(freeze_path)
    if freeze_path.exists() and not force:
        raise FileExistsError(
            f"Freeze anchor already exists (use --force to replace it): {freeze_path}"
        )
    endpoints = _load_endpoints(output_root)
    artifact_hashes = _hash_paths(_artifact_paths(output_root))
    status = _git_output("status", "--porcelain")
    payload = {
        "artifact_sha256": artifact_hashes,
        "subjects": [str(value) for value in endpoints["subjects"].tolist()],
        "git_commit": _git_commit(),
        "git_status_porcelain_sha256": (
            None if status is None else hashlib.sha256(status).hexdigest()
        ),
        "production_constants": {
            "n_permutations": N_PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_bootstrap": N_BOOTSTRAP,
        },
    }
    payload["freeze_digest"] = _canonical_sha(payload)
    _write_json_atomic(freeze_path, payload)
    logger.info("Wrote Phase-5 freeze anchor to %s", freeze_path)
    return payload


def _validate_freeze_before_labels(
    *,
    output_root: Path,
    freeze_path: Path,
) -> tuple[dict, dict]:
    """Complete every frozen-artifact check before label bytes are opened."""
    output_root = Path(output_root)
    freeze_path = Path(freeze_path)
    freeze = _read_json(freeze_path)
    recorded_digest = freeze.get("freeze_digest")
    digest_payload = dict(freeze)
    digest_payload.pop("freeze_digest", None)
    if recorded_digest != _canonical_sha(digest_payload):
        raise ValueError("Freeze digest mismatch.")
    expected_hashes = freeze.get("artifact_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Freeze anchor has no artifact_sha256 object.")
    paths = _artifact_paths(output_root)
    if set(expected_hashes) != set(paths):
        raise ValueError("Freeze anchor artifact set does not match the frozen schema.")
    current_hashes = _hash_paths(paths)
    for name, expected in expected_hashes.items():
        if current_hashes[name] != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")

    results_dir = output_root / "results"
    if results_dir.exists():
        raise FileExistsError(f"Results directory already exists: {results_dir}")

    calibration = _read_json(paths["calibration.json"])
    bound_hashes = calibration.get("bound_sha256")
    expected_bound_names = {
        "endpoints.npz",
        "endpoints_manifest.json",
        "permutation_schedule.npy",
        "yeo7_mapping.json",
    }
    if not isinstance(bound_hashes, dict) or set(bound_hashes) != expected_bound_names:
        raise ValueError("Calibration bound hashes do not match the frozen schema.")
    for name in sorted(expected_bound_names):
        if bound_hashes[name] != current_hashes[name]:
            raise ValueError(f"Calibration bound hash mismatch: {name}")
    return freeze, calibration


def _read_label_bytes(path: Path) -> bytes:
    """Open the clinical label CSV; this is the pipeline's sole label read."""
    return Path(path).read_bytes()


def _parse_labels(label_bytes: bytes, expected_subjects: list[str]) -> dict[str, str]:
    try:
        text = label_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("FOR group CSV must be UTF-8.") from exc
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("Empty FOR group CSV.") from None
    if header != ["subject", "group"]:
        raise ValueError("FOR group CSV header must be exactly subject,group.")
    labels = {}
    for row in reader:
        if not row or not any(row):
            continue
        if len(row) != 2:
            raise ValueError(f"Invalid FOR group row: {row}")
        subject, group = row
        if group not in {"healthy", "depressed"} or not subject or subject in labels:
            raise ValueError(f"Invalid or duplicate FOR group row: {row}")
        labels[subject] = group
    counts = Counter(labels.values())
    if counts != Counter({"healthy": 25, "depressed": 25}):
        raise ValueError(
            "FOR group CSV must contain exactly 25 healthy and 25 depressed "
            f"subjects; found {dict(counts)}."
        )
    if set(labels) != set(expected_subjects):
        missing = sorted(set(expected_subjects) - set(labels))
        extra = sorted(set(labels) - set(expected_subjects))
        raise ValueError(
            "FOR group CSV subject set does not match frozen endpoints; "
            f"missing={missing}, extra={extra}."
        )
    return labels


def _float_ci(values: np.ndarray, index: int) -> dict[str, float]:
    return {
        "low": float(values[index, 0]),
        "high": float(values[index, 1]),
    }


def _effect_rows(
    *,
    values: np.ndarray,
    depressed_mask: np.ndarray,
    schedule: np.ndarray,
    names: Sequence[str],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict]:
    statistic, pvalues = permutation_pvalue_two_sided(
        values, depressed_mask, schedule
    )
    bootstrap = stratified_bootstrap(
        values, depressed_mask, n_bootstrap, bootstrap_seed
    )
    adjusted = bh_adjust(pvalues)
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "name": name,
                "welch_t": float(statistic[index]),
                "permutation_p_raw": float(pvalues[index]),
                "bh_adjusted_p": float(adjusted[index]),
                "mean_depressed": float(np.mean(values[depressed_mask, index])),
                "mean_healthy": float(np.mean(values[~depressed_mask, index])),
                "raw_mean_difference_depressed_minus_healthy": float(
                    bootstrap["raw_mean_difference"][index]
                ),
                "raw_mean_difference_ci95": _float_ci(
                    bootstrap["raw_mean_difference_ci95"], index
                ),
                "cohens_d": float(bootstrap["cohens_d"][index]),
                "cohens_d_ci95": _float_ci(
                    bootstrap["cohens_d_ci95"], index
                ),
            }
        )
    return rows


def _finish_figure(fig, path: Path, title: str) -> None:
    fig.suptitle(f"{title}\n{FIGURE_SUFFIX}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _strip_x(center: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([center])
    return center + np.linspace(-0.08, 0.08, count)


def _plot_residual_groups(
    values: np.ndarray,
    depressed_mask: np.ndarray,
    path: Path,
) -> None:
    healthy = values[~depressed_mask]
    depressed = values[depressed_mask]
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(_strip_x(0.0, healthy.size), healthy, alpha=0.75, label="healthy")
    axis.scatter(
        _strip_x(1.0, depressed.size), depressed, alpha=0.75, label="depressed"
    )
    axis.scatter(
        [0.0, 1.0],
        [healthy.mean(), depressed.mean()],
        marker="_",
        s=500,
        linewidths=4,
        color="black",
        label="group mean",
    )
    axis.set_xticks([0.0, 1.0], ["healthy", "depressed"])
    axis.set_ylabel("Global fingerprint residual")
    axis.legend(frameon=False)
    _finish_figure(fig, path, "Saved-fingerprint residual by group")


def _plot_blocks_forest(rows: list[dict], path: Path) -> None:
    differences = np.asarray(
        [row["raw_mean_difference_depressed_minus_healthy"] for row in rows]
    )
    lows = np.asarray([row["raw_mean_difference_ci95"]["low"] for row in rows])
    highs = np.asarray([row["raw_mean_difference_ci95"]["high"] for row in rows])
    significant = np.asarray([row["bh_adjusted_p"] <= 0.05 for row in rows])
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(10, 12))
    axis.axvline(0.0, color="0.6", linewidth=1)
    for index in positions:
        axis.errorbar(
            differences[index],
            index,
            xerr=np.asarray(
                [
                    [differences[index] - lows[index]],
                    [highs[index] - differences[index]],
                ]
            ),
            fmt="*" if significant[index] else "o",
            color="tab:red" if significant[index] else "tab:blue",
            capsize=3,
        )
    axis.set_yticks(positions, [row["name"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Raw mean difference (depressed − healthy), bootstrap 95% CI")
    _finish_figure(fig, path, "Yeo-7 connectivity-block differences")


def analyze_frozen(
    *,
    output_root: Path = OUTPUT_ROOT,
    freeze_path: Path = FREEZE_PATH,
    labels_path: Path = LABELS_PATH,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Validate the freeze, read labels once, and run the frozen inference."""
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be positive.")
    output_root = Path(output_root)
    freeze, _ = _validate_freeze_before_labels(
        output_root=output_root, freeze_path=Path(freeze_path)
    )
    endpoints = _load_endpoints(output_root)
    frozen_subjects = freeze.get("subjects")
    endpoint_subjects = [str(value) for value in endpoints["subjects"].tolist()]
    if frozen_subjects != endpoint_subjects:
        raise ValueError("Frozen subject list does not match endpoints.npz.")
    schedule = np.asarray(
        np.load(output_root / "permutation_schedule.npy"), dtype=np.int64
    )

    label_bytes = _read_label_bytes(Path(labels_path))
    labels_sha256 = hashlib.sha256(label_bytes).hexdigest()
    labels = _parse_labels(label_bytes, endpoint_subjects)
    depressed_mask = np.asarray(
        [labels[subject] == "depressed" for subject in endpoint_subjects],
        dtype=bool,
    )

    global_values = np.asarray(
        endpoints["fingerprint_residual_global"], dtype=np.float64
    )
    global_t, global_p = permutation_pvalue_two_sided(
        global_values, depressed_mask, schedule
    )
    global_bootstrap = stratified_bootstrap(
        global_values, depressed_mask, int(n_bootstrap), BOOTSTRAP_SEED
    )
    primary_1 = {
        "name": "fingerprint_residual_global",
        "contrast": "depressed_minus_healthy",
        "welch_t": float(global_t[0]),
        "permutation_p_raw": float(global_p[0]),
        "mean_depressed": float(global_values[depressed_mask].mean()),
        "mean_healthy": float(global_values[~depressed_mask].mean()),
        "raw_mean_difference_depressed_minus_healthy": float(
            global_bootstrap["raw_mean_difference"][0]
        ),
        "raw_mean_difference_ci95": _float_ci(
            global_bootstrap["raw_mean_difference_ci95"], 0
        ),
        "cohens_d": float(global_bootstrap["cohens_d"][0]),
        "cohens_d_ci95": _float_ci(global_bootstrap["cohens_d_ci95"], 0),
    }
    omnibus_statistic, omnibus_p = permutation_pvalue_omnibus(
        endpoints["connectivity_blocks"], depressed_mask, schedule
    )
    primary_2 = {
        "name": "connectivity_blocks_omnibus_sum_t2",
        "statistic": float(omnibus_statistic),
        "permutation_p_raw": float(omnibus_p),
    }
    network_rows = _effect_rows(
        values=np.asarray(endpoints["fingerprint_residual_network"], dtype=float),
        depressed_mask=depressed_mask,
        schedule=schedule,
        names=YEO_NETWORKS,
        n_bootstrap=int(n_bootstrap),
        bootstrap_seed=BOOTSTRAP_SEED + 1,
    )
    block_rows = _effect_rows(
        values=np.asarray(endpoints["connectivity_blocks"], dtype=float),
        depressed_mask=depressed_mask,
        schedule=schedule,
        names=_block_names(),
        n_bootstrap=int(n_bootstrap),
        bootstrap_seed=BOOTSTRAP_SEED + 2,
    )

    results_dir = output_root / "results"
    results_dir.mkdir(parents=True)
    residual_figure = results_dir / "fig_residual_groups.png"
    blocks_figure = results_dir / "fig_blocks_forest.png"
    _plot_residual_groups(global_values, depressed_mask, residual_figure)
    _plot_blocks_forest(block_rows, blocks_figure)

    semantics = {
        "primary_pvalue_adjustment": (
            "no adjustment across the two primary families"
        ),
        "family_wise_error": "family-wise error across families uncontrolled",
        "overall_decision": "no overall positive/negative decision derived",
        "claim_status": "all claims exploratory (motion-uncorrected)",
        "prior_label_exposure_disclosure": (
            "prospectively locked endpoint analysis after prior label exposure; "
            "endpoints and tests specified before THESE endpoints ever met labels, "
            "not before any label access by the project"
        ),
    }
    results = {
        "status": "exploratory",
        "motion_corrected": False,
        "contrast": "depressed_minus_healthy",
        "group_counts": {"healthy": 25, "depressed": 25},
        "primary_1": primary_1,
        "primary_2": primary_2,
        "raw_primary_pvalues": {
            "fingerprint_residual_global": primary_1["permutation_p_raw"],
            "connectivity_blocks_omnibus_sum_t2": primary_2["permutation_p_raw"],
        },
        "semantics": semantics,
        "exploratory": {
            "fingerprint_residual_networks": network_rows,
            "connectivity_blocks": block_rows,
        },
        "bootstrap": {
            "n_bootstrap": int(n_bootstrap),
            "named_rng_streams": {
                "primary_1": BOOTSTRAP_SEED,
                "fingerprint_residual_networks": BOOTSTRAP_SEED + 1,
                "connectivity_blocks": BOOTSTRAP_SEED + 2,
            },
            "method": "stratified subject bootstrap, percentile 95% CI",
        },
        "freeze_digest": freeze["freeze_digest"],
        "label_csv_sha256": labels_sha256,
        "git_commit": _git_commit(),
        "figures": {
            "fig_residual_groups": str(residual_figure),
            "fig_blocks_forest": str(blocks_figure),
        },
    }
    _write_json_atomic(results_dir / "results.json", results)
    logger.info("Wrote frozen exploratory analysis to %s", results_dir)
    return results


def _testing_help(description: str) -> str:
    return f"{description} (testing override; production value is frozen)"


def _add_endpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--for-inference-root",
        type=Path,
        default=FOR_INFERENCE_ROOT,
        help=_testing_help("FOR inference root"),
    )
    parser.add_argument(
        "--contract-for-root",
        type=Path,
        default=CONTRACT_FOR_ROOT,
        help=_testing_help("FOR contract root"),
    )
    parser.add_argument(
        "--final-model-dir",
        type=Path,
        default=FINAL_MODEL_DIR,
        help=_testing_help("final model directory"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=_testing_help("analysis output root"),
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=ANNOT_PATHS[0].parent,
        help=_testing_help("annotation directory"),
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help=_testing_help("number of permutations"),
    )


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=_testing_help("analysis output root"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    endpoints_parser = subparsers.add_parser("endpoints")
    _add_endpoint_arguments(endpoints_parser)
    calibrate_parser = subparsers.add_parser("calibrate")
    _add_output_root(calibrate_parser)
    calibrate_parser.add_argument(
        "--calibration-repeats",
        type=int,
        default=200,
        help=_testing_help("number of calibration repeats"),
    )
    calibrate_parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help=_testing_help("number of frozen schedule rows"),
    )
    freeze_parser = subparsers.add_parser("freeze")
    _add_output_root(freeze_parser)
    freeze_parser.add_argument(
        "--freeze-path",
        type=Path,
        default=FREEZE_PATH,
        help=_testing_help("freeze anchor path"),
    )
    freeze_parser.add_argument("--force", action="store_true")
    analyze_parser = subparsers.add_parser("analyze")
    _add_output_root(analyze_parser)
    analyze_parser.add_argument(
        "--freeze-path",
        type=Path,
        default=FREEZE_PATH,
        help=_testing_help("freeze anchor path"),
    )
    analyze_parser.add_argument(
        "--labels-path",
        type=Path,
        default=LABELS_PATH,
        help=_testing_help("clinical label CSV path"),
    )
    analyze_parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=_testing_help("number of bootstrap resamples"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _build_parser().parse_args(argv)
    if args.command == "endpoints":
        annotation_paths = (
            args.annotations_root
            / "lh.Schaefer2018_400Parcels_7Networks_order.annot",
            args.annotations_root
            / "rh.Schaefer2018_400Parcels_7Networks_order.annot",
        )
        build_endpoints(
            for_inference_root=args.for_inference_root,
            contract_for_root=args.contract_for_root,
            final_model_dir=args.final_model_dir,
            output_root=args.output_root,
            annot_paths=annotation_paths,
            n_permutations=args.n_permutations,
        )
        return 0
    if args.command == "calibrate":
        calibrate_endpoints(
            output_root=args.output_root,
            calibration_repeats=args.calibration_repeats,
            n_permutations=args.n_permutations,
        )
        return 0
    if args.command == "freeze":
        freeze_analysis(
            output_root=args.output_root,
            freeze_path=args.freeze_path,
            force=args.force,
        )
        return 0
    if args.command == "analyze":
        analyze_frozen(
            output_root=args.output_root,
            freeze_path=args.freeze_path,
            labels_path=args.labels_path,
            n_bootstrap=args.n_bootstrap,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
