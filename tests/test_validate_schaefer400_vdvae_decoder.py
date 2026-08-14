import numpy as np
from sklearn.linear_model import Ridge

from src.pipelines.validate_schaefer400_vdvae_decoder import (
    build_validation_split,
    evaluate_ridge_fold,
)


def test_validation_split_keeps_final_images_out_of_all_tuning_folds():
    stimulus_ids = np.arange(100, dtype=np.int64)
    split = build_validation_split(
        stimulus_ids,
        subjects=[1, 2, 3, 4],
        seed=7,
        final_fraction=0.2,
    )

    final_ids = set(split["final_stimulus_ids"])
    development_ids = set(split["development_stimulus_ids"])
    tuning_ids = [value for fold in split["tuning_folds"].values() for value in fold]

    assert len(final_ids) == 20
    assert final_ids.isdisjoint(development_ids)
    assert set(tuning_ids) == development_ids
    assert len(tuning_ids) == len(set(tuning_ids))
    assert final_ids | development_ids == set(stimulus_ids.tolist())


def test_validation_ridge_predictions_match_sklearn(tmp_path):
    rng = np.random.RandomState(11)
    train_design = rng.randn(30, 400).astype(np.float32)
    val_design = rng.randn(9, 400).astype(np.float32)
    targets = rng.randn(24, 17).astype(np.float32)
    train_rows = rng.randint(0, targets.shape[0], size=train_design.shape[0])
    val_rows = rng.randint(0, targets.shape[0], size=val_design.shape[0])
    prediction_path = tmp_path / "prediction.npy"

    ridge_metrics, baseline_metrics = evaluate_ridge_fold(
        train_design=train_design,
        train_target_rows=train_rows,
        val_design=val_design,
        val_target_rows=val_rows,
        targets=targets,
        alpha=13.0,
        chunk_size=5,
        prediction_path=prediction_path,
    )

    expected = Ridge(alpha=13.0, fit_intercept=True).fit(
        train_design, targets[train_rows]
    )
    np.testing.assert_allclose(
        np.load(prediction_path),
        expected.predict(val_design),
        atol=2e-5,
        rtol=2e-5,
    )
    assert set(ridge_metrics) == set(baseline_metrics)
    assert np.isfinite(list(ridge_metrics.values())).all()

