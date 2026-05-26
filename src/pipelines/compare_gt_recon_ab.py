"""
Compare two ground-truth VDVAE+VD reconstruction runs.

Outputs:
- comparison_summary.json
- shared_stimuli.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _relative_or_abs(path: Path) -> str:
    return str(path.resolve())


def compare_runs(
    label_a: str,
    label_b: str,
    data_root_a: Path,
    data_root_b: Path,
    recon_dir_a: Path,
    recon_dir_b: Path,
    output_dir: Path,
    subject: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    subj_tag = f"subj{subject:02d}"
    test_idx_a = np.load(data_root_a / subj_tag / "test_stim_idx.npy").astype(np.int64)
    test_idx_b = np.load(data_root_b / subj_tag / "test_stim_idx.npy").astype(np.int64)

    summary_a = _load_json(recon_dir_a / "summary.json")
    summary_b = _load_json(recon_dir_b / "summary.json")
    task_summary_a = _load_json(data_root_a / subj_tag / "task_data_summary.json")
    task_summary_b = _load_json(data_root_b / subj_tag / "task_data_summary.json")

    row_by_stim_a = {int(stim): idx for idx, stim in enumerate(test_idx_a.tolist())}
    row_by_stim_b = {int(stim): idx for idx, stim in enumerate(test_idx_b.tolist())}
    shared_stim = np.array(sorted(set(row_by_stim_a) & set(row_by_stim_b)), dtype=np.int64)

    manifest_path = output_dir / "shared_stimuli.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stim_id",
                f"row_{label_a}",
                f"row_{label_b}",
                f"img_{label_a}",
                f"img_{label_b}",
                f"vdvae_{label_a}",
                f"vdvae_{label_b}",
            ],
        )
        writer.writeheader()
        for stim in shared_stim.tolist():
            row_a = row_by_stim_a[int(stim)]
            row_b = row_by_stim_b[int(stim)]
            writer.writerow(
                {
                    "stim_id": int(stim),
                    f"row_{label_a}": int(row_a),
                    f"row_{label_b}": int(row_b),
                    f"img_{label_a}": _relative_or_abs(
                        recon_dir_a / "reconstructions" / "gt_fmri" / f"row{row_a:05d}_stim{int(stim)}.png"
                    ),
                    f"img_{label_b}": _relative_or_abs(
                        recon_dir_b / "reconstructions" / "gt_fmri" / f"row{row_b:05d}_stim{int(stim)}.png"
                    ),
                    f"vdvae_{label_a}": _relative_or_abs(
                        recon_dir_a / "reconstructions_vdvae" / "gt_fmri" / f"row{row_a:05d}_stim{int(stim)}.png"
                    ),
                    f"vdvae_{label_b}": _relative_or_abs(
                        recon_dir_b / "reconstructions_vdvae" / "gt_fmri" / f"row{row_b:05d}_stim{int(stim)}.png"
                    ),
                }
            )

    comparison = {
        "subject": int(subject),
        "labels": {"a": label_a, "b": label_b},
        "task_data": {
            label_a: task_summary_a,
            label_b: task_summary_b,
        },
        "reconstruction": {
            label_a: summary_a,
            label_b: summary_b,
        },
        "shared_test_stimuli": {
            "count": int(shared_stim.shape[0]),
            "manifest_csv": str(manifest_path.resolve()),
            "first_20": [int(v) for v in shared_stim[:20].tolist()],
        },
        "metric_delta_b_minus_a": {
            key: float(summary_b["metrics"][key] - summary_a["metrics"][key])
            for key in summary_a["metrics"].keys()
        },
    }
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compare two GT reconstruction runs.")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--label-a", default="current")
    parser.add_argument("--label-b", default="brain_diffuser_compat")
    parser.add_argument("--data-root-a", required=True)
    parser.add_argument("--data-root-b", required=True)
    parser.add_argument("--recon-dir-a", required=True)
    parser.add_argument("--recon-dir-b", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    compare_runs(
        label_a=str(args.label_a),
        label_b=str(args.label_b),
        data_root_a=Path(args.data_root_a),
        data_root_b=Path(args.data_root_b),
        recon_dir_a=Path(args.recon_dir_a),
        recon_dir_b=Path(args.recon_dir_b),
        output_dir=Path(args.output_dir),
        subject=int(args.subject),
    )


if __name__ == "__main__":
    main()
