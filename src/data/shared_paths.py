from __future__ import annotations

import os
from pathlib import Path


DEFAULT_SHARED_NSD_DATA_ROOT = Path("/scratch_shared/rothermm/brain-diffuser/data")
LOCAL_SHARED_NSD_DATA_ROOT = Path("/media/psycontrol/HDD/Datasets/brain-diffuser/data")


def default_raw_data_root() -> str:
    root = os.environ.get("NSD_SHARED_DATA_ROOT")
    if root:
        return str(Path(root).expanduser())
    if LOCAL_SHARED_NSD_DATA_ROOT.exists():
        return str(LOCAL_SHARED_NSD_DATA_ROOT)
    return str(DEFAULT_SHARED_NSD_DATA_ROOT)


def default_stimuli_hdf5() -> str:
    return str(
        Path(default_raw_data_root()) / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    )


def default_annotation_candidates() -> list[Path]:
    shared_root = Path(default_raw_data_root())
    return [
        Path("data/annots/COCO_73k_annots_curated.npy"),
        shared_root / "annots" / "COCO_73k_annots_curated.npy",
        shared_root / "nsddata" / "experiments" / "nsd" / "COCO_73k_annots_curated.npy",
    ]
