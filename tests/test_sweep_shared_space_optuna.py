import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.pipelines.sweep_shared_space_optuna import (
    build_loso_folds,
    make_loso_objective,
    run_shared_space_sweep,
)
from src.pipelines.train_shared_space import apply_config_overrides


def _require_sweep_deps():
    pytest.importorskip("optuna", reason="optuna not installed")
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("tensorboard", reason="tensorboard not installed")


class _DummyWriter:
    def __init__(self):
        self.scalars = []
        self.hparams = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))

    def add_hparams(self, hparam_dict, metric_dict, run_name=None):
        self.hparams.append((hparam_dict, metric_dict, run_name))


class _DummyTrial:
    def __init__(self, *, number=0, ridge_alpha=10.0, n_components=30, prune_after_step=None):
        self.number = number
        self._ridge_alpha = float(ridge_alpha)
        self._n_components = int(n_components)
        self._prune_after_step = prune_after_step
        self.reports = []
        self.user_attrs = {}

    def suggest_float(self, *_args, **_kwargs):
        return self._ridge_alpha

    def suggest_int(self, *_args, **_kwargs):
        return self._n_components

    def report(self, value, step):
        self.reports.append((int(step), float(value)))

    def should_prune(self):
        if self._prune_after_step is None:
            return False
        return len(self.reports) >= int(self._prune_after_step)

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _OptunaStub:
    class TrialPruned(Exception):
        pass


def _write_sweep_config(tmp_path: Path, *, train_subjects=None, n_trials=2, retrain_best=True) -> Path:
    if train_subjects is None:
        train_subjects = [1, 2]
    cfg = {
        "subjects": {"train": train_subjects, "test": [7]},
        "features": {"type": "clip"},
        "random_seed": 42,
        "evaluation": {
            "fixed_eval_size": 12,
            "eval_split_seed": 7,
            "reliability_thresholds": [0.0, 0.1, 0.3],
        },
        "output_root": str(tmp_path / "outputs"),
        "sweep": {
            "shared_space": {
                "study_name": "unit_sweep",
                "output_dir": str(tmp_path / "sweeps"),
                "n_trials": n_trials,
                "sampler_seed": 13,
                "cleanup_trial_artifacts": True,
                "retrain_best": retrain_best,
                "pruner": {
                    "type": "median",
                    "n_startup_trials": 0,
                    "n_warmup_steps": 0,
                    "interval_steps": 1,
                },
                "search_space": {
                    "ridge_alpha": {"min": 1.0e-2, "max": 10.0, "log": True},
                    "n_components": {"min": 20, "max": 30, "step": 10},
                },
            }
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _build_fake_train_predict():
    train_calls = []

    def fake_train_fn(
        *,
        config_path,
        data_root,
        raw_data_root,
        output_dir,
        feature_type_override=None,
        config_overrides=None,
    ):
        del config_path, data_root, raw_data_root, feature_type_override
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "override.json"), "w") as f:
            json.dump(config_overrides or {}, f)
        np.savez(
            os.path.join(output_dir, "builder.npz"),
            k_global=np.int64(1),
            template_Z=np.zeros((1, 1), dtype=np.float32),
            template_fingerprint=np.zeros((1, 1), dtype=np.float32),
            connectivity_mode=np.array("parcellation"),
            experiment_mode=np.array("hybrid_cha"),
            n_components=np.int64(1),
            min_k=np.int64(1),
            ensemble_method=np.array("average"),
            max_iters=np.int64(1),
            tol=np.float64(1e-5),
        )
        np.savez(
            os.path.join(output_dir, "encoder.npz"),
            W=np.zeros((1, 1), dtype=np.float32),
            b=np.zeros((1,), dtype=np.float32),
            x_mean=np.zeros((1,), dtype=np.float32),
            x_std=np.ones((1,), dtype=np.float32),
            alpha=np.float32(1.0),
        )
        np.save(os.path.join(output_dir, "shared_stim_idx.npy"), np.array([0], dtype=np.int64))
        train_calls.append({"output_dir": output_dir, "config_overrides": config_overrides or {}})
        return None, None

    def fake_predict_fn(
        *,
        test_sub,
        model_dir,
        data_root,
        feature_type,
        output_dir,
        use_fixed_eval_split,
        fixed_eval_size,
        eval_split_seed,
        reliability_thresholds=None,
    ):
        del data_root, feature_type, output_dir, use_fixed_eval_split, fixed_eval_size, eval_split_seed, reliability_thresholds
        with open(os.path.join(model_dir, "override.json")) as f:
            overrides = json.load(f)
        n_components = int(overrides.get("alignment", {}).get("n_components", 20))
        ridge_alpha = float(overrides.get("encoding", {}).get("ridge_alpha", 1.0))
        # Deterministic score, lightly favoring n_components=30.
        score = float(1.0 / (1.0 + abs(n_components - 30)) + 1e-3 * np.log10(max(ridge_alpha, 1e-8)) + 0.01 * int(test_sub))
        return {"metrics": {"median_r": score}}

    return fake_train_fn, fake_predict_fn, train_calls


def test_apply_config_overrides_deep_merge_preserves_base():
    base = {
        "subjects": {"train": [1, 2, 3], "test": [7]},
        "alignment": {"n_components": 50, "min_k": 10},
        "encoding": {"ridge_alpha": 100.0},
    }
    overrides = {
        "subjects": {"train": [1, 2]},
        "alignment": {"n_components": 30},
        "encoding": {"ridge_alpha": 10.0},
    }
    merged = apply_config_overrides(base, overrides)

    assert base["subjects"]["train"] == [1, 2, 3]
    assert base["alignment"]["n_components"] == 50
    assert merged["subjects"]["train"] == [1, 2]
    assert merged["subjects"]["test"] == [7]
    assert merged["alignment"]["n_components"] == 30
    assert merged["alignment"]["min_k"] == 10
    assert merged["encoding"]["ridge_alpha"] == 10.0


def test_build_loso_folds_covers_each_subject_once():
    folds = build_loso_folds([1, 2, 3, 4])
    assert len(folds) == 4
    assert sorted(val for _train, val in folds) == [1, 2, 3, 4]
    for train_subs, val_sub in folds:
        assert val_sub not in train_subs
        assert sorted(train_subs + [val_sub]) == [1, 2, 3, 4]


def test_objective_aggregates_mean_and_reports_intermediate(tmp_path):
    writer = _DummyWriter()
    folds = [([2, 3], 1), ([1, 3], 2), ([1, 2], 3)]
    best_tracker = {"value": float("-inf")}

    def fake_train_fn(**kwargs):
        os.makedirs(kwargs["output_dir"], exist_ok=True)
        return None, None

    def fake_predict_fn(**kwargs):
        val = int(kwargs["test_sub"])
        return {"metrics": {"median_r": val / 10.0}}

    objective = make_loso_objective(
        optuna_mod=_OptunaStub,
        config_path="unused.yaml",
        data_root="unused",
        raw_data_root="unused",
        feature_type="clip",
        folds=folds,
        trial_artifacts_root=str(tmp_path / "trial_artifacts"),
        fixed_eval_size=10,
        eval_split_seed=42,
        reliability_thresholds=[0.0, 0.1],
        cleanup_trial_artifacts=True,
        writer=writer,
        best_tracker=best_tracker,
        alpha_min=1e-2,
        alpha_max=1e2,
        ncomp_min=20,
        ncomp_max=40,
        ncomp_step=5,
        train_fn=fake_train_fn,
        predict_fn=fake_predict_fn,
    )
    trial = _DummyTrial(number=0, ridge_alpha=1.0, n_components=30)
    value = objective(trial)

    assert np.isclose(value, 0.2)
    assert trial.reports == [(1, 0.1), (2, 0.15), (3, 0.2)]
    assert np.isclose(trial.user_attrs["fold_scores"]["1"], 0.1)
    assert np.isclose(trial.user_attrs["fold_scores"]["2"], 0.2)
    assert np.isclose(trial.user_attrs["fold_scores"]["3"], 0.3)
    assert any(tag == "trial/objective" for tag, _value, _step in writer.scalars)


def test_objective_pruning_raises_trial_pruned(tmp_path):
    writer = _DummyWriter()
    folds = [([2, 3], 1), ([1, 3], 2)]
    best_tracker = {"value": float("-inf")}

    def fake_train_fn(**kwargs):
        os.makedirs(kwargs["output_dir"], exist_ok=True)
        return None, None

    def fake_predict_fn(**kwargs):
        return {"metrics": {"median_r": 0.01}}

    objective = make_loso_objective(
        optuna_mod=_OptunaStub,
        config_path="unused.yaml",
        data_root="unused",
        raw_data_root="unused",
        feature_type="clip",
        folds=folds,
        trial_artifacts_root=str(tmp_path / "trial_artifacts"),
        fixed_eval_size=10,
        eval_split_seed=42,
        reliability_thresholds=[0.0],
        cleanup_trial_artifacts=True,
        writer=writer,
        best_tracker=best_tracker,
        alpha_min=1e-2,
        alpha_max=1e2,
        ncomp_min=20,
        ncomp_max=40,
        ncomp_step=5,
        train_fn=fake_train_fn,
        predict_fn=fake_predict_fn,
    )
    trial = _DummyTrial(number=1, ridge_alpha=1.0, n_components=30, prune_after_step=1)
    with pytest.raises(_OptunaStub.TrialPruned):
        objective(trial)
    assert trial.user_attrs["pruned_after_fold"] == 1


def test_study_resume_appends_trials(tmp_path):
    _require_sweep_deps()
    config_path = _write_sweep_config(tmp_path, train_subjects=[1, 2], n_trials=1, retrain_best=False)
    fake_train, fake_predict, _calls = _build_fake_train_predict()
    db_path = tmp_path / "resume.db"

    first = run_shared_space_sweep(
        config_path=str(config_path),
        data_root="unused",
        raw_data_root="unused",
        n_trials=1,
        study_name="resume_test",
        storage=str(db_path),
        output_root=str(tmp_path / "sweeps"),
        train_fn=fake_train,
        predict_fn=fake_predict,
        retrain_best=False,
    )
    second = run_shared_space_sweep(
        config_path=str(config_path),
        data_root="unused",
        raw_data_root="unused",
        n_trials=1,
        study_name="resume_test",
        storage=str(db_path),
        output_root=str(tmp_path / "sweeps"),
        train_fn=fake_train,
        predict_fn=fake_predict,
        retrain_best=False,
    )

    assert int(first["n_trials_total"]) >= 1
    assert int(second["n_trials_total"]) >= int(first["n_trials_total"]) + 1


def test_best_model_retrain_writes_artifacts(tmp_path):
    _require_sweep_deps()
    config_path = _write_sweep_config(tmp_path, train_subjects=[1, 2], n_trials=1, retrain_best=True)
    fake_train, fake_predict, _calls = _build_fake_train_predict()

    summary = run_shared_space_sweep(
        config_path=str(config_path),
        data_root="unused",
        raw_data_root="unused",
        n_trials=1,
        study_name="best_model_test",
        output_root=str(tmp_path / "sweeps"),
        train_fn=fake_train,
        predict_fn=fake_predict,
        retrain_best=True,
    )
    best_model_dir = Path(summary["paths"]["best_model_dir"])
    assert best_model_dir.exists()
    assert (best_model_dir / "builder.npz").exists()
    assert (best_model_dir / "encoder.npz").exists()
    assert (best_model_dir / "shared_stim_idx.npy").exists()


def test_tensorboard_events_include_scalar_and_hparams(tmp_path):
    _require_sweep_deps()
    from tensorboard.backend.event_processing import event_accumulator

    config_path = _write_sweep_config(tmp_path, train_subjects=[1, 2], n_trials=1, retrain_best=False)
    fake_train, fake_predict, _calls = _build_fake_train_predict()
    summary = run_shared_space_sweep(
        config_path=str(config_path),
        data_root="unused",
        raw_data_root="unused",
        n_trials=1,
        study_name="tb_events_test",
        output_root=str(tmp_path / "sweeps"),
        train_fn=fake_train,
        predict_fn=fake_predict,
        retrain_best=False,
    )

    tb_dir = Path(summary["paths"]["tensorboard_dir"])
    event_files = list(tb_dir.rglob("events.out.tfevents.*"))
    assert event_files, "No TensorBoard event files created."

    found_trial_objective = False
    found_hparam_signal = False
    for event_file in event_files:
        acc = event_accumulator.EventAccumulator(str(event_file))
        acc.Reload()
        tags = acc.Tags()
        scalar_tags = set(tags.get("scalars", []))
        tensor_tags = set(tags.get("tensors", []))
        if "trial/objective" in scalar_tags:
            found_trial_objective = True
        if "hparam/objective" in scalar_tags or any(t.startswith("_hparams_/") for t in tensor_tags):
            found_hparam_signal = True

    assert found_trial_objective
    assert found_hparam_signal


def test_smoke_run_two_trials_with_mocks(tmp_path):
    _require_sweep_deps()
    config_path = _write_sweep_config(tmp_path, train_subjects=[1, 2], n_trials=2, retrain_best=False)
    fake_train, fake_predict, _calls = _build_fake_train_predict()
    summary = run_shared_space_sweep(
        config_path=str(config_path),
        data_root="unused",
        raw_data_root="unused",
        n_trials=2,
        study_name="smoke_two_trials",
        output_root=str(tmp_path / "sweeps"),
        train_fn=fake_train,
        predict_fn=fake_predict,
        retrain_best=False,
    )

    assert int(summary["n_trials_total"]) >= 2
    assert Path(summary["paths"]["trials_csv"]).exists()
    assert Path(summary["paths"]["best_params_json"]).exists()
