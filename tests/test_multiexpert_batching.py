import numpy as np
import pytest

from src.data.multiexpert_batching import (
    SubjectHomogeneousBatchSampler,
    stimulus_disjoint_split,
)


def _stimulus_memberships(split, ids_by_subject):
    train = set()
    val = set()
    for subject, stimulus_ids in ids_by_subject.items():
        train.update(stimulus_ids[split.by_subject[subject].train_indices].tolist())
        val.update(stimulus_ids[split.by_subject[subject].val_indices].tolist())
    return train, val


def test_stimulus_split_is_global_disjoint_and_deterministic():
    ids_by_subject = {
        1: np.array([10, 11, 12, 13, 14, 15], dtype=np.int64),
        2: np.array([10, 20, 21, 22, 23, 24], dtype=np.int64),
    }
    first = stimulus_disjoint_split(ids_by_subject, val_fraction=0.4, seed=42)
    second = stimulus_disjoint_split(ids_by_subject, val_fraction=0.4, seed=42)

    np.testing.assert_array_equal(first.train_stimuli, second.train_stimuli)
    np.testing.assert_array_equal(first.val_stimuli, second.val_stimuli)
    for subject in ids_by_subject:
        np.testing.assert_array_equal(
            first.by_subject[subject].train_indices,
            second.by_subject[subject].train_indices,
        )
        np.testing.assert_array_equal(
            first.by_subject[subject].val_indices,
            second.by_subject[subject].val_indices,
        )

        all_rows = np.sort(
            np.concatenate(
                [
                    first.by_subject[subject].train_indices,
                    first.by_subject[subject].val_indices,
                ]
            )
        )
        np.testing.assert_array_equal(all_rows, np.arange(ids_by_subject[subject].size))

    train_stimuli, val_stimuli = _stimulus_memberships(first, ids_by_subject)
    assert train_stimuli.isdisjoint(val_stimuli)
    assert train_stimuli | val_stimuli == set(np.concatenate(list(ids_by_subject.values())))


def test_stimulus_split_keeps_duplicate_rows_together():
    ids_by_subject = {
        1: np.array([1, 1, 2, 3, 4]),
        2: np.array([1, 5, 6, 7, 8]),
    }
    split = stimulus_disjoint_split(ids_by_subject, val_fraction=0.5, seed=3)
    train_stimuli, val_stimuli = _stimulus_memberships(split, ids_by_subject)
    assert train_stimuli.isdisjoint(val_stimuli)
    assert (1 in train_stimuli) != (1 in val_stimuli)


def test_stimulus_split_fails_when_per_subject_validation_is_impossible():
    with pytest.raises(ValueError, match="at least two unique stimuli"):
        stimulus_disjoint_split(
            {1: np.array([5, 5]), 2: np.array([6, 7])},
            val_fraction=0.5,
        )


def test_sampler_batches_are_subject_homogeneous_and_cover_rows():
    sampler = SubjectHomogeneousBatchSampler(
        {1: np.arange(5), 2: np.arange(7)},
        batch_size=3,
        shuffle=True,
        seed=9,
    )
    batches = list(sampler)

    assert len(sampler) == 5
    assert len(batches) == 5
    assert all(1 <= len(batch) <= 3 for batch in batches)
    assert all(len({subject for subject, _ in batch}) == 1 for batch in batches)
    assert sorted(key for batch in batches for key in batch) == [
        *( (1, row) for row in range(5) ),
        *( (2, row) for row in range(7) ),
    ]

    assert list(sampler) == batches
    sampler.set_epoch(1)
    assert list(sampler) != batches


def test_sampler_drop_last_and_order_without_shuffle():
    sampler = SubjectHomogeneousBatchSampler(
        {2: np.arange(7), 1: np.arange(5)},
        batch_size=3,
        shuffle=False,
        drop_last=True,
    )
    assert len(sampler) == 3
    assert list(sampler) == [
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        [(2, 3), (2, 4), (2, 5)],
    ]


def test_sampler_rejects_duplicate_or_negative_indices():
    with pytest.raises(ValueError, match="unique"):
        SubjectHomogeneousBatchSampler({1: [0, 0]}, batch_size=2)
    with pytest.raises(ValueError, match="non-negative"):
        SubjectHomogeneousBatchSampler({1: [-1, 0]}, batch_size=2)
