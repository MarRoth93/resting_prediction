import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import src.pipelines.for_clinical_analysis as clinical
from src.alignment.shared_space import SharedSpaceBuilder


def _synthetic_annotation_names() -> tuple[list[bytes], list[str]]:
    networks = []
    for network in clinical.YEO_NETWORKS:
        networks.extend([network] * clinical.EXPECTED_NETWORK_COUNTS[network])
    names = {}
    for hemisphere, selected in (("lh", networks[:200]), ("rh", networks[200:])):
        within_network = {network: 0 for network in clinical.YEO_NETWORKS}
        hemisphere_names: list[bytes | str] = [b"unknown"]
        for index, network in enumerate(selected):
            within_network[network] += 1
            name = f"7Networks_{hemisphere.upper()}_{network}_{within_network[network]}"
            hemisphere_names.append(name.encode() if index % 2 == 0 else name)
        names[hemisphere] = hemisphere_names
    return names["lh"], names["rh"]


def _write_annotation_files(tmp_path: Path) -> tuple[Path, Path]:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    paths = (
        annotations / "lh.Schaefer2018_400Parcels_7Networks_order.annot",
        annotations / "rh.Schaefer2018_400Parcels_7Networks_order.annot",
    )
    paths[0].write_bytes(b"synthetic lh")
    paths[1].write_bytes(b"synthetic rh")
    return paths


def _patch_read_annot(monkeypatch, paths: tuple[Path, Path]) -> None:
    left_names, right_names = _synthetic_annotation_names()

    def fake_read_annot(path):
        names = left_names if Path(path) == paths[0] else right_names
        return np.zeros(1), np.zeros((len(names), 4)), names

    monkeypatch.setattr("nibabel.freesurfer.read_annot", fake_read_annot)


def test_yeo7_mapping_validates_counts_names_and_edge_counts(tmp_path, monkeypatch):
    paths = _write_annotation_files(tmp_path)
    _patch_read_annot(monkeypatch, paths)
    mapping = clinical._build_yeo7_mapping(paths)
    assert len(mapping["parcel_to_network"]) == 400
    assert mapping["network_counts"] == clinical.EXPECTED_NETWORK_COUNTS
    assert mapping["block_names"] == clinical._block_names()
    assert len(mapping["block_names"]) == 28
    assert mapping["block_unique_edge_counts"]["Vis|Vis"] == 61 * 60 // 2
    assert mapping["block_unique_edge_counts"]["Vis|SomMot"] == 61 * 77


def test_yeo7_mapping_rejects_wrong_frozen_count(tmp_path, monkeypatch):
    paths = _write_annotation_files(tmp_path)
    left_names, right_names = _synthetic_annotation_names()
    left_names[1] = b"7Networks_LH_Default_999"

    def fake_read_annot(path):
        names = left_names if Path(path) == paths[0] else right_names
        return np.zeros(1), np.zeros((len(names), 4)), names

    monkeypatch.setattr("nibabel.freesurfer.read_annot", fake_read_annot)
    with pytest.raises(ValueError, match="frozen counts"):
        clinical._build_yeo7_mapping(paths)


def test_connectivity_block_averaging_excludes_diagonal():
    networks = []
    for network in clinical.YEO_NETWORKS:
        networks.extend([network] * clinical.EXPECTED_NETWORK_COUNTS[network])
    parcel_networks = np.asarray(networks)
    fisher_z = np.zeros((400, 400), dtype=float)
    vis = np.flatnonzero(parcel_networks == "Vis")
    sommot = np.flatnonzero(parcel_networks == "SomMot")
    fisher_z[np.ix_(vis, vis)] = 2.0
    fisher_z[vis, vis] = 100.0
    fisher_z[np.ix_(vis, sommot)] = 3.0
    blocks = clinical._average_connectivity_blocks(fisher_z, parcel_networks)
    assert blocks[clinical._block_names().index("Vis|Vis")] == pytest.approx(2.0)
    assert blocks[clinical._block_names().index("Vis|SomMot")] == pytest.approx(3.0)


def test_offset_decomposition_separates_column_mean():
    template = np.ones((4, 2))
    fingerprint = template + np.asarray([2.0, -3.0])
    offset_share, centered = clinical._offset_decomposition(fingerprint, template)
    assert offset_share == pytest.approx(1.0)
    assert centered == pytest.approx(0.0)


@pytest.fixture(scope="module")
def frozen_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase5")
    inference_root = root / "for_inference"
    contract_root = root / "contract_for"
    model_dir = root / "model"
    output_root = root / "analysis"
    freeze_path = root / "analysis_registry" / "phase5_freeze.json"
    annotation_paths = _write_annotation_files(root)

    rng = np.random.default_rng(123)
    template = rng.normal(size=(400, 100)).astype(np.float32)
    builder = SharedSpaceBuilder(n_components=100, min_k=10)
    builder.k_global = 100
    builder.template_Z = np.zeros((1, 100), dtype=np.float32)
    builder.template_fingerprint = template
    builder.save(str(model_dir))

    subjects = []
    for subject_index in range(50):
        subject = f"sub-{subject_index:04d}"
        subjects.append(subject)
        inference_dir = inference_root / subject
        contract_dir = contract_root / subject
        inference_dir.mkdir(parents=True)
        contract_dir.mkdir(parents=True)
        subject_rng = np.random.default_rng(10_000 + subject_index)
        fingerprint = template + subject_rng.normal(
            scale=0.04 + subject_index * 0.0005, size=template.shape
        ).astype(np.float32)
        rest_seeds = subject_rng.normal(size=(235, 400)).astype(np.float32)
        rest_targets = subject_rng.normal(size=(235, 30)).astype(np.float32)
        alignment_p = subject_rng.normal(scale=0.05, size=(30, 100)).astype(
            np.float32
        )
        alignment_r = np.eye(100, dtype=np.float32)
        np.save(inference_dir / "fingerprint.npy", fingerprint)
        np.save(inference_dir / "alignment_P.npy", alignment_p)
        np.save(inference_dir / "alignment_R.npy", alignment_r)
        np.save(contract_dir / "rest_seeds.npy", rest_seeds)
        np.save(contract_dir / "rest_targets.npy", rest_targets)

    labels_path = root / "for_groups.csv"
    label_rows = ["subject,group"] + [
        f"{subject},{'healthy' if index < 25 else 'depressed'}"
        for index, subject in enumerate(subjects)
    ]
    labels_path.write_text("\n".join(label_rows) + "\n")

    left_names, right_names = _synthetic_annotation_names()
    patcher = pytest.MonkeyPatch()

    def fake_read_annot(path):
        names = left_names if Path(path) == annotation_paths[0] else right_names
        return np.zeros(1), np.zeros((len(names), 4)), names

    patcher.setattr("nibabel.freesurfer.read_annot", fake_read_annot)
    label_reads = 0
    original_read = clinical._read_label_bytes

    def prelabel_spy(path):
        nonlocal label_reads
        label_reads += 1
        return original_read(path)

    patcher.setattr(clinical, "_read_label_bytes", prelabel_spy)
    clinical.main(
        [
            "endpoints",
            "--for-inference-root",
            str(inference_root),
            "--contract-for-root",
            str(contract_root),
            "--final-model-dir",
            str(model_dir),
            "--output-root",
            str(output_root),
            "--annotations-root",
            str(annotation_paths[0].parent),
            "--n-permutations",
            "200",
        ]
    )
    clinical.main(
        [
            "calibrate",
            "--output-root",
            str(output_root),
            "--calibration-repeats",
            "5",
            "--n-permutations",
            "200",
        ]
    )
    clinical.main(
        [
            "freeze",
            "--output-root",
            str(output_root),
            "--freeze-path",
            str(freeze_path),
        ]
    )
    patcher.undo()
    assert label_reads == 0
    return {
        "root": root,
        "inference_root": inference_root,
        "contract_root": contract_root,
        "model_dir": model_dir,
        "output_root": output_root,
        "freeze_path": freeze_path,
        "labels_path": labels_path,
        "annotation_paths": annotation_paths,
        "subjects": subjects,
        "prelabel_reads": label_reads,
    }


def _copy_frozen_fixture(frozen_fixture, tmp_path):
    output_root = tmp_path / "analysis"
    shutil.copytree(frozen_fixture["output_root"], output_root)
    freeze_path = tmp_path / "analysis_registry" / "phase5_freeze.json"
    freeze_path.parent.mkdir(parents=True)
    shutil.copy2(frozen_fixture["freeze_path"], freeze_path)
    labels_path = tmp_path / "for_groups.csv"
    shutil.copy2(frozen_fixture["labels_path"], labels_path)
    return output_root, freeze_path, labels_path


def test_cli_end_to_end_schema_and_single_label_read(
    frozen_fixture, tmp_path, monkeypatch
):
    output_root, freeze_path, labels_path = _copy_frozen_fixture(
        frozen_fixture, tmp_path
    )
    calls = []
    original = clinical._read_label_bytes

    def spy(path):
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(clinical, "_read_label_bytes", spy)
    assert clinical.main(
        [
            "analyze",
            "--output-root",
            str(output_root),
            "--freeze-path",
            str(freeze_path),
            "--labels-path",
            str(labels_path),
            "--n-bootstrap",
            "200",
        ]
    ) == 0
    assert calls == [labels_path]

    results_path = output_root / "results" / "results.json"
    results = json.loads(results_path.read_text())
    assert {
        "primary_1",
        "primary_2",
        "raw_primary_pvalues",
        "semantics",
        "exploratory",
        "freeze_digest",
        "label_csv_sha256",
        "git_commit",
    }.issubset(results)
    assert results["semantics"] == {
        "claim_status": "all claims exploratory (motion-uncorrected)",
        "family_wise_error": "family-wise error across families uncontrolled",
        "overall_decision": "no overall positive/negative decision derived",
        "primary_pvalue_adjustment": (
            "no adjustment across the two primary families"
        ),
        "prior_label_exposure_disclosure": (
            "prospectively locked endpoint analysis after prior label exposure; "
            "endpoints and tests specified before THESE endpoints ever met labels, "
            "not before any label access by the project"
        ),
    }
    assert "cohens_d" not in results["primary_2"]
    assert len(results["exploratory"]["fingerprint_residual_networks"]) == 7
    assert len(results["exploratory"]["connectivity_blocks"]) == 28
    assert (output_root / "results" / "fig_residual_groups.png").is_file()
    assert (output_root / "results" / "fig_blocks_forest.png").is_file()
    assert "subjects" not in results


def test_prelabel_commands_never_read_labels(frozen_fixture):
    assert frozen_fixture["prelabel_reads"] == 0


def test_analyze_refuses_missing_freeze_before_label_read(
    frozen_fixture, tmp_path, monkeypatch
):
    output_root, _, labels_path = _copy_frozen_fixture(frozen_fixture, tmp_path)
    calls = []
    monkeypatch.setattr(
        clinical, "_read_label_bytes", lambda path: calls.append(path) or b""
    )
    with pytest.raises(FileNotFoundError, match="Missing file"):
        clinical.analyze_frozen(
            output_root=output_root,
            freeze_path=tmp_path / "missing-freeze.json",
            labels_path=labels_path,
            n_bootstrap=20,
        )
    assert calls == []


def test_analyze_refuses_tampered_endpoint_before_label_read(
    frozen_fixture, tmp_path, monkeypatch
):
    output_root, freeze_path, labels_path = _copy_frozen_fixture(
        frozen_fixture, tmp_path
    )
    with (output_root / "endpoints.npz").open("ab") as handle:
        handle.write(b"tampered")
    calls = []
    monkeypatch.setattr(
        clinical, "_read_label_bytes", lambda path: calls.append(path) or b""
    )
    with pytest.raises(ValueError, match="Frozen artifact hash mismatch: endpoints.npz"):
        clinical.analyze_frozen(
            output_root=output_root,
            freeze_path=freeze_path,
            labels_path=labels_path,
            n_bootstrap=20,
        )
    assert calls == []


def test_analyze_refuses_existing_results_before_label_read(
    frozen_fixture, tmp_path, monkeypatch
):
    output_root, freeze_path, labels_path = _copy_frozen_fixture(
        frozen_fixture, tmp_path
    )
    (output_root / "results").mkdir()
    calls = []
    monkeypatch.setattr(
        clinical, "_read_label_bytes", lambda path: calls.append(path) or b""
    )
    with pytest.raises(FileExistsError, match="Results directory already exists"):
        clinical.analyze_frozen(
            output_root=output_root,
            freeze_path=freeze_path,
            labels_path=labels_path,
            n_bootstrap=20,
        )
    assert calls == []


def test_analyze_rejects_wrong_subject_set_after_one_label_read(
    frozen_fixture, tmp_path, monkeypatch
):
    output_root, freeze_path, labels_path = _copy_frozen_fixture(
        frozen_fixture, tmp_path
    )
    rows = labels_path.read_text().splitlines()
    rows[1] = "sub-9999,healthy"
    labels_path.write_text("\n".join(rows) + "\n")
    calls = []
    original = clinical._read_label_bytes

    def spy(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(clinical, "_read_label_bytes", spy)
    with pytest.raises(ValueError, match="subject set does not match"):
        clinical.analyze_frozen(
            output_root=output_root,
            freeze_path=freeze_path,
            labels_path=labels_path,
            n_bootstrap=20,
        )
    assert len(calls) == 1


def test_analyze_rejects_non_25_25_labels_after_one_label_read(
    frozen_fixture, tmp_path, monkeypatch
):
    output_root, freeze_path, labels_path = _copy_frozen_fixture(
        frozen_fixture, tmp_path
    )
    rows = labels_path.read_text().splitlines()
    rows[25] = rows[25].replace(",healthy", ",depressed")
    labels_path.write_text("\n".join(rows) + "\n")
    calls = []
    original = clinical._read_label_bytes

    def spy(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(clinical, "_read_label_bytes", spy)
    with pytest.raises(ValueError, match="exactly 25 healthy and 25 depressed"):
        clinical.analyze_frozen(
            output_root=output_root,
            freeze_path=freeze_path,
            labels_path=labels_path,
            n_bootstrap=20,
        )
    assert len(calls) == 1


def test_freeze_refuses_existing_anchor(frozen_fixture):
    with pytest.raises(FileExistsError, match="already exists"):
        clinical.freeze_analysis(
            output_root=frozen_fixture["output_root"],
            freeze_path=frozen_fixture["freeze_path"],
        )


def test_endpoints_hard_fail_on_234_seed_rows_without_label_read(
    frozen_fixture, tmp_path, monkeypatch
):
    inference_root = tmp_path / "inference"
    contract_root = tmp_path / "contract"
    for subject in frozen_fixture["subjects"]:
        (inference_root / subject).mkdir(parents=True)
        (contract_root / subject).mkdir(parents=True)
    first = frozen_fixture["subjects"][0]
    for filename in ("fingerprint.npy", "alignment_P.npy", "alignment_R.npy"):
        (inference_root / first / filename).symlink_to(
            frozen_fixture["inference_root"] / first / filename
        )
    (contract_root / first / "rest_targets.npy").symlink_to(
        frozen_fixture["contract_root"] / first / "rest_targets.npy"
    )
    np.save(contract_root / first / "rest_seeds.npy", np.zeros((234, 400)))
    _patch_read_annot(monkeypatch, frozen_fixture["annotation_paths"])
    calls = []
    monkeypatch.setattr(
        clinical, "_read_label_bytes", lambda path: calls.append(path) or b""
    )
    with pytest.raises(ValueError, match=r"rest_seeds must have shape \(235, 400\)"):
        clinical.build_endpoints(
            for_inference_root=inference_root,
            contract_for_root=contract_root,
            final_model_dir=frozen_fixture["model_dir"],
            output_root=tmp_path / "output",
            annot_paths=frozen_fixture["annotation_paths"],
            n_permutations=20,
        )
    assert calls == []
