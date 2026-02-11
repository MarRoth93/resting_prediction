"""
Download all required NSD data from AWS S3.

Downloads:
1. Experiment design (nsd_expdesign.mat, stim info)
2. Stimuli (nsd_stimuli.hdf5 — 73k images)
3. Task betas (betas_fithrf_GLMdenoise_RR — dynamically discovered sessions)
4. ROI masks (nsdgeneral + all visual area atlases)
5. Resting-state timeseries (REST runs only)
6. Noise ceiling maps (ncsnr)

Usage:
    python download_nsddata.py                    # download everything
    python download_nsddata.py --skip-stimuli     # skip 26GB stimuli file
    python download_nsddata.py --skip-rest        # skip REST timeseries
    python download_nsddata.py --only-rest        # download only REST data
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUBJECTS = [1, 2, 5, 7]
S3_BASE = "s3://natural-scenes-dataset"
AWS_SIGNED_REQUEST = False


def run_cmd(cmd: str, description: str = "") -> bool:
    """Run a shell command, return True if successful."""
    if description:
        logger.info(description)
    if cmd.strip().startswith("aws ") and not AWS_SIGNED_REQUEST and "--no-sign-request" not in cmd:
        cmd = f"{cmd} --no-sign-request"
    result = os.system(cmd)
    if result != 0:
        logger.warning(f"Command failed (exit {result}): {cmd}")
        return False
    return True


def local_file_ready(path: str, min_bytes: int = 1) -> bool:
    """Return True if local file exists and has at least min_bytes."""
    try:
        return os.path.exists(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def s3_list(path: str) -> list[str]:
    """List files at an S3 path. Returns list of filenames."""
    cmd = ["aws", "s3", "ls", path]
    if not AWS_SIGNED_REQUEST:
        cmd.append("--no-sign-request")
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Failed to list {path}: {result.stderr}")
        return []
    files = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            # Format: "2020-01-01 00:00:00     12345 filename"
            parts = line.strip().split()
            if len(parts) >= 4:
                files.append(parts[-1])
    return files


def extract_session_run_key(filename: str) -> tuple[int | None, int] | None:
    """
    Extract (session_id, run_id) from filenames.

    Supports patterns like:
    - ...session40...run01...
    - ...run01...session40...
    - ...run01... (session unknown)
    """
    m = re.search(r"session(\d+).*run(\d+)", filename)
    if m is not None:
        return int(m.group(1)), int(m.group(2))

    m = re.search(r"run(\d+).*session(\d+)", filename)
    if m is not None:
        return int(m.group(2)), int(m.group(1))

    m = re.search(r"run(\d+)", filename)
    if m is not None:
        return None, int(m.group(1))

    return None


def is_rest_design_file(s3_path: str) -> bool:
    """
    Determine if a design TSV corresponds to REST.

    Heuristic:
    - Parse numeric tokens per row.
    - If a row has >1 numeric column, ignore first column (often index/time).
    - REST if all inspected numeric values are ~0.
    """
    cmd = ["aws", "s3", "cp", s3_path, "-"]
    if not AWS_SIGNED_REQUEST:
        cmd.append("--no-sign-request")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        logger.warning(f"Timed out reading design file {s3_path}")
        return False
    if result.returncode != 0:
        logger.warning(f"Failed to read design file {s3_path}: {result.stderr.strip()}")
        return False

    inspected_values = []
    for line in result.stdout.splitlines():
        nums = []
        for tok in line.strip().split():
            try:
                nums.append(float(tok))
            except ValueError:
                continue
        if not nums:
            continue
        vals = nums[1:] if len(nums) > 1 else nums
        inspected_values.extend(vals)

    if not inspected_values:
        return False

    return all(abs(v) < 1e-12 for v in inspected_values)


# ── 1. Experiment design ─────────────────────────────────────────────────────

def download_experiment_info():
    """Download experiment design files."""
    logger.info("=" * 60)
    logger.info("Downloading experiment info...")
    os.makedirs("nsddata/experiments/nsd", exist_ok=True)
    targets = [
        "nsd_expdesign.mat",
        "nsd_stim_info_merged.pkl",
    ]
    for fname in targets:
        local_path = os.path.join("nsddata/experiments/nsd", fname)
        if local_file_ready(local_path):
            logger.info(f"  {fname}: already exists, skipping")
            continue
        run_cmd(
            f"aws s3 cp {S3_BASE}/nsddata/experiments/nsd/{fname} "
            f"nsddata/experiments/nsd/",
            f"  {fname}",
        )


# ── 2. Stimuli ───────────────────────────────────────────────────────────────

def download_stimuli():
    """Download NSD stimuli (~26 GB)."""
    logger.info("=" * 60)
    logger.info("Downloading stimuli (~26 GB)...")
    os.makedirs("nsddata_stimuli/stimuli/nsd", exist_ok=True)
    local_path = "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    if local_file_ready(local_path):
        logger.info("  nsd_stimuli.hdf5: already exists, skipping")
        return
    run_cmd(
        f"aws s3 cp {S3_BASE}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 "
        f"nsddata_stimuli/stimuli/nsd/",
        "  nsd_stimuli.hdf5",
    )


# ── 3. Task betas ────────────────────────────────────────────────────────────

def discover_beta_sessions(sub: int) -> list[int]:
    """Discover available beta sessions for a subject from S3."""
    s3_path = (
        f"{S3_BASE}/nsddata_betas/ppdata/subj{sub:02d}/"
        f"func1pt8mm/betas_fithrf_GLMdenoise_RR/"
    )
    files = s3_list(s3_path)
    sessions = []
    for f in files:
        if f.startswith("betas_session") and f.endswith(".nii.gz"):
            num = f.replace("betas_session", "").replace(".nii.gz", "")
            try:
                sessions.append(int(num))
            except ValueError:
                continue
    return sorted(sessions)


def download_betas():
    """Download task betas for all subjects (dynamic session discovery)."""
    logger.info("=" * 60)
    logger.info("Downloading task betas...")
    for sub in SUBJECTS:
        # Discover sessions from S3
        sessions = discover_beta_sessions(sub)
        if not sessions:
            # Fallback: try 1-40
            logger.warning(f"  Subject {sub}: could not list S3, trying sessions 1-40")
            sessions = list(range(1, 41))

        logger.info(f"  Subject {sub}: {len(sessions)} sessions ({sessions[0]}-{sessions[-1]})")
        out_dir = f"nsddata_betas/ppdata/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/"
        os.makedirs(out_dir, exist_ok=True)

        for sess in sessions:
            fname = f"betas_session{sess:02d}.nii.gz"
            s3_path = (
                f"{S3_BASE}/nsddata_betas/ppdata/subj{sub:02d}/"
                f"func1pt8mm/betas_fithrf_GLMdenoise_RR/{fname}"
            )
            local_path = os.path.join(out_dir, fname)
            if local_file_ready(local_path):
                logger.info(f"    Session {sess}: already exists, skipping")
                continue
            run_cmd(f"aws s3 cp {s3_path} {out_dir}", f"    Session {sess}")


# ── 4. ROI masks and atlases ─────────────────────────────────────────────────

def download_rois():
    """Download ROI masks and visual area atlases."""
    logger.info("=" * 60)
    logger.info("Downloading ROIs and atlases...")

    # Specific atlas files we need
    roi_files = [
        "nsdgeneral.nii.gz",
        "prf-visualrois.nii.gz",
        "prf-eccrois.nii.gz",
        "Kastner2015.nii.gz",
        "floc-bodies.nii.gz",
        "floc-faces.nii.gz",
        "floc-places.nii.gz",
        "floc-words.nii.gz",
    ]

    for sub in SUBJECTS:
        out_dir = f"nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/"
        os.makedirs(out_dir, exist_ok=True)

        for roi_file in roi_files:
            local_path = os.path.join(out_dir, roi_file)
            if local_file_ready(local_path):
                logger.info(f"  Subject {sub}/{roi_file}: already exists, skipping")
                continue
            s3_path = (
                f"{S3_BASE}/nsddata/ppdata/subj{sub:02d}/func1pt8mm/roi/{roi_file}"
            )
            run_cmd(f"aws s3 cp {s3_path} {out_dir}", f"  Subject {sub}: {roi_file}")


# ── 5. Noise ceiling maps ────────────────────────────────────────────────────

def download_ncsnr():
    """Download noise ceiling SNR maps (for reliability masking)."""
    logger.info("=" * 60)
    logger.info("Downloading ncsnr maps...")
    for sub in SUBJECTS:
        out_dir = f"nsddata_betas/ppdata/subj{sub:02d}/func1pt8mm/"
        os.makedirs(out_dir, exist_ok=True)
        s3_path = (
            f"{S3_BASE}/nsddata_betas/ppdata/subj{sub:02d}/"
            f"func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr.nii.gz"
        )
        local_path = os.path.join(out_dir, "ncsnr.nii.gz")
        if local_file_ready(local_path):
            logger.info(f"  Subject {sub}: already exists, skipping")
            continue
        run_cmd(f"aws s3 cp {s3_path} {out_dir}", f"  Subject {sub}: ncsnr.nii.gz")


# ── 6. Resting-state timeseries ──────────────────────────────────────────────

def discover_rest_runs(sub: int) -> list[str]:
    """
    Discover REST run filenames for a subject via S3 listing.

    Strategy:
    1. List all timeseries files
    2. Filter for REST-related files (contain 'rest' in name)
    3. If no 'rest' pattern found, check NSD documentation patterns
    """
    s3_path = (
        f"{S3_BASE}/nsddata_timeseries/ppdata/subj{sub:02d}/"
        f"func1pt8mm/timeseries/"
    )
    all_files = s3_list(s3_path)

    if not all_files:
        logger.warning(f"  Subject {sub}: no timeseries files found at {s3_path}")
        return []

    logger.info(f"  Subject {sub}: found {len(all_files)} timeseries files total")

    # Build lookup maps for timeseries files
    ts_by_key: dict[tuple[int | None, int], str] = {}
    ts_by_run: dict[int, list[str]] = {}
    for f in all_files:
        if not f.endswith(".nii.gz"):
            continue
        key = extract_session_run_key(f)
        if key is None:
            continue
        ts_by_key[key] = f
        _, rid = key
        ts_by_run.setdefault(rid, []).append(f)

    # Primary method: design-based REST detection
    design_s3_path = (
        f"{S3_BASE}/nsddata_timeseries/ppdata/subj{sub:02d}/"
        f"func1pt8mm/design/"
    )
    design_files = s3_list(design_s3_path)

    if design_files:
        logger.info(f"  Subject {sub}: found {len(design_files)} design files")
        rest_files = []
        ambiguous_matches = 0
        unmatched_design = 0
        sorted_design_files = sorted(design_files)
        for i, dfile in enumerate(sorted_design_files, start=1):
            if i == 1 or i % 25 == 0 or i == len(sorted_design_files):
                logger.info(f"  Subject {sub}: scanning design files {i}/{len(sorted_design_files)}")
            if not dfile.endswith(".tsv"):
                continue
            dkey = extract_session_run_key(dfile)
            if dkey is None:
                continue
            sess_id, rid = dkey
            candidate = None

            # Prefer exact session+run match when available
            if dkey in ts_by_key:
                candidate = ts_by_key[dkey]
            else:
                candidates = sorted(set(ts_by_run.get(rid, [])))
                if len(candidates) == 1:
                    candidate = candidates[0]
                elif len(candidates) > 1:
                    ambiguous_matches += 1
                    continue
                else:
                    unmatched_design += 1
                    continue

            dfile_s3 = f"{design_s3_path}{dfile}"
            if is_rest_design_file(dfile_s3):
                rest_files.append(candidate)

        rest_files = sorted(set(rest_files))
        if rest_files:
            logger.info(
                f"  Subject {sub}: {len(rest_files)} REST files (detected from design/*.tsv)"
            )
            if ambiguous_matches > 0:
                logger.warning(
                    f"  Subject {sub}: skipped {ambiguous_matches} ambiguous design->timeseries matches"
                )
            if unmatched_design > 0:
                logger.info(
                    f"  Subject {sub}: {unmatched_design} design files had no matching timeseries file"
                )
            return rest_files
        logger.warning(
            f"  Subject {sub}: design files found, but no REST runs detected by zero-design heuristic"
        )

    # Fallback method: filename matching
    rest_files = [f for f in all_files if "rest" in f.lower()]
    if rest_files:
        logger.info(f"  Subject {sub}: {len(rest_files)} REST files (matched filename pattern)")
        return sorted(rest_files)

    logger.warning(
        f"  Subject {sub}: no REST runs found by design-based detection or filename matching. "
        f"Available files: {all_files[:10]}{'...' if len(all_files) > 10 else ''}"
    )
    return []


def download_rest_timeseries():
    """Download resting-state timeseries for all subjects."""
    logger.info("=" * 60)
    logger.info("Downloading REST timeseries...")

    for sub in SUBJECTS:
        out_dir = (
            f"nsddata_timeseries/ppdata/subj{sub:02d}/func1pt8mm/timeseries/"
        )
        os.makedirs(out_dir, exist_ok=True)

        manifest_dir = f"processed_data/subj{sub:02d}"
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, "rest_run_manifest.json")

        rest_files = []
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                manifest_runs = manifest.get("rest_runs", [])
                if manifest_runs:
                    missing = [
                        rf for rf in manifest_runs
                        if not local_file_ready(os.path.join(out_dir, rf))
                    ]
                    if not missing:
                        logger.info(
                            f"  Subject {sub}: REST files already present from manifest "
                            f"({len(manifest_runs)} runs), skipping download"
                        )
                        continue
                    logger.info(
                        f"  Subject {sub}: manifest found with {len(manifest_runs)} runs, "
                        f"{len(missing)} missing locally"
                    )
                    rest_files = manifest_runs
            except Exception as e:
                logger.warning(f"  Subject {sub}: failed to read manifest {manifest_path}: {e}")

        if not rest_files:
            rest_files = discover_rest_runs(sub)

        if len(rest_files) < 2:
            logger.warning(
                f"  Subject {sub}: only {len(rest_files)} REST runs found (need >= 2). "
                f"Zero-shot CHA will not be available for this subject."
            )
            if not rest_files:
                continue

        for rest_file in rest_files:
            local_path = os.path.join(out_dir, rest_file)
            if local_file_ready(local_path):
                logger.info(f"    {rest_file}: already exists, skipping")
                continue
            s3_path = (
                f"{S3_BASE}/nsddata_timeseries/ppdata/subj{sub:02d}/"
                f"func1pt8mm/timeseries/{rest_file}"
            )
            run_cmd(f"aws s3 cp {s3_path} {out_dir}", f"    {rest_file}")

        # Save manifest
        with open(manifest_path, "w") as f:
            json.dump({"rest_runs": rest_files, "subject": sub}, f, indent=2)
        logger.info(f"  Subject {sub}: saved REST manifest ({len(rest_files)} runs)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global AWS_SIGNED_REQUEST

    parser = argparse.ArgumentParser(description="Download NSD data from AWS S3")
    parser.add_argument("--skip-stimuli", action="store_true", help="Skip 26GB stimuli download")
    parser.add_argument("--skip-rest", action="store_true", help="Skip REST timeseries")
    parser.add_argument("--skip-betas", action="store_true", help="Skip task betas")
    parser.add_argument("--only-rest", action="store_true", help="Download only REST data")
    parser.add_argument(
        "--signed-request",
        action="store_true",
        help="Use signed AWS requests (default is unsigned --no-sign-request)",
    )
    args = parser.parse_args()
    AWS_SIGNED_REQUEST = args.signed_request
    logger.info(
        "AWS request mode: %s",
        "signed" if AWS_SIGNED_REQUEST else "unsigned (--no-sign-request)",
    )

    if args.only_rest:
        download_rest_timeseries()
        return

    # Always download experiment info and ROIs
    download_experiment_info()
    download_rois()
    download_ncsnr()

    if not args.skip_stimuli:
        download_stimuli()

    if not args.skip_betas:
        download_betas()

    if not args.skip_rest:
        download_rest_timeseries()

    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info("Next steps:")
    logger.info("  1. python -m src.data.prepare_task_data -sub 1  (repeat for 2, 5, 7)")
    logger.info("  2. python -m src.data.prepare_rest_data -sub 1  (repeat for 2, 5, 7)")
    logger.info("  3. python -m src.data.prepare_features")
    logger.info("  4. python -m src.pipelines.train_shared_space")


if __name__ == "__main__":
    main()
