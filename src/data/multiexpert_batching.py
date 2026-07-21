"""Leakage-safe splits and subject-homogeneous batches for multi-expert fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SubjectSplit:
    """Row indices for one subject after a global stimulus-grouped split."""

    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass(frozen=True)
class StimulusDisjointSplit:
    """A split where a stimulus ID appears on exactly one side globally."""

    by_subject: dict[int, SubjectSplit]
    train_stimuli: np.ndarray
    val_stimuli: np.ndarray

    def indices(self, subject: int, partition: str) -> np.ndarray:
        subject = int(subject)
        if subject not in self.by_subject:
            raise KeyError(f"Subject {subject} is not present in this split.")
        if partition == "train":
            return self.by_subject[subject].train_indices
        if partition in {"val", "validation"}:
            return self.by_subject[subject].val_indices
        raise ValueError("partition must be 'train' or 'val'.")

    def indices_by_subject(self, partition: str) -> dict[int, np.ndarray]:
        return {
            subject: self.indices(subject, partition)
            for subject in sorted(self.by_subject)
        }


def _split_for_val_stimuli(
    stimulus_ids_by_subject: Mapping[int, np.ndarray],
    val_stimuli: np.ndarray,
) -> dict[int, SubjectSplit]:
    result: dict[int, SubjectSplit] = {}
    for subject in sorted(stimulus_ids_by_subject):
        stimulus_ids = stimulus_ids_by_subject[subject]
        val_mask = np.isin(stimulus_ids, val_stimuli, assume_unique=False)
        result[int(subject)] = SubjectSplit(
            train_indices=np.flatnonzero(~val_mask).astype(np.int64, copy=False),
            val_indices=np.flatnonzero(val_mask).astype(np.int64, copy=False),
        )
    return result


def stimulus_disjoint_split(
    stimulus_ids_by_subject: Mapping[int, Sequence[int] | np.ndarray],
    *,
    val_fraction: float = 0.10,
    seed: int = 42,
    require_each_subject: bool = True,
    max_attempts: int = 1024,
) -> StimulusDisjointSplit:
    """Split rows by global stimulus identity, never by independent subject rows.

    Shared stimulus IDs are assigned together even when they occur in multiple
    subjects.  The deterministic retry loop only matters for very small data:
    it seeks a split with both train and validation rows for every subject.
    """
    if not stimulus_ids_by_subject:
        raise ValueError("stimulus_ids_by_subject must not be empty.")
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be positive.")

    clean: dict[int, np.ndarray] = {}
    for raw_subject, raw_ids in stimulus_ids_by_subject.items():
        subject = int(raw_subject)
        if subject in clean:
            raise ValueError(f"Duplicate subject key after integer conversion: {subject}.")
        ids = np.asarray(raw_ids)
        if ids.ndim != 1:
            raise ValueError(
                f"Subject {subject}: stimulus IDs must be 1D, got shape {ids.shape}."
            )
        if ids.size == 0:
            raise ValueError(f"Subject {subject}: stimulus IDs are empty.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(
                f"Subject {subject}: stimulus IDs must be integers, got {ids.dtype}."
            )
        ids = ids.astype(np.int64, copy=False)
        if require_each_subject and np.unique(ids).size < 2:
            raise ValueError(
                f"Subject {subject} needs at least two unique stimuli to populate "
                "both train and validation partitions."
            )
        clean[subject] = ids

    all_stimuli = np.unique(np.concatenate(list(clean.values())))
    if all_stimuli.size < 2:
        raise ValueError("At least two unique stimuli are required for a split.")
    n_val = int(round(all_stimuli.size * float(val_fraction)))
    n_val = max(1, min(n_val, int(all_stimuli.size) - 1))

    rng = np.random.RandomState(int(seed))
    for _ in range(int(max_attempts)):
        shuffled = all_stimuli[rng.permutation(all_stimuli.size)]
        val_stimuli = np.sort(shuffled[:n_val]).astype(np.int64, copy=False)
        by_subject = _split_for_val_stimuli(clean, val_stimuli)
        if not require_each_subject or all(
            split.train_indices.size > 0 and split.val_indices.size > 0
            for split in by_subject.values()
        ):
            train_stimuli = np.setdiff1d(
                all_stimuli,
                val_stimuli,
                assume_unique=True,
            ).astype(np.int64, copy=False)
            return StimulusDisjointSplit(
                by_subject=by_subject,
                train_stimuli=train_stimuli,
                val_stimuli=val_stimuli,
            )

    raise ValueError(
        "Could not create a global stimulus-disjoint split with train and validation "
        f"rows for every subject after {int(max_attempts)} deterministic attempts. "
        "Increase val_fraction, provide more stimuli, or set require_each_subject=False."
    )


class SubjectHomogeneousBatchSampler:
    """Yield batches of ``(subject, row_index)`` keys from exactly one subject.

    This follows the small part of PyTorch's batch-sampler protocol needed by
    ``DataLoader``: it is iterable, has a stable length, and supports
    ``set_epoch`` for deterministic epoch-specific shuffling.
    """

    def __init__(
        self,
        indices_by_subject: Mapping[int, Sequence[int] | np.ndarray],
        *,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive.")
        if not indices_by_subject:
            raise ValueError("indices_by_subject must not be empty.")

        clean: dict[int, np.ndarray] = {}
        for raw_subject, raw_indices in indices_by_subject.items():
            subject = int(raw_subject)
            if subject in clean:
                raise ValueError(f"Duplicate subject key after integer conversion: {subject}.")
            indices = np.asarray(raw_indices)
            if indices.ndim != 1:
                raise ValueError(
                    f"Subject {subject}: row indices must be 1D, got {indices.shape}."
                )
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(
                    f"Subject {subject}: row indices must be integers, got {indices.dtype}."
                )
            indices = indices.astype(np.int64, copy=False)
            if np.any(indices < 0):
                raise ValueError(f"Subject {subject}: row indices must be non-negative.")
            if np.unique(indices).size != indices.size:
                raise ValueError(f"Subject {subject}: row indices must be unique.")
            clean[subject] = indices.copy()

        self.indices_by_subject = {
            subject: clean[subject]
            for subject in sorted(clean)
        }
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                int(indices.size) // self.batch_size
                for indices in self.indices_by_subject.values()
            )
        return sum(
            (int(indices.size) + self.batch_size - 1) // self.batch_size
            for indices in self.indices_by_subject.values()
        )

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        subject_batches: list[tuple[int, np.ndarray]] = []
        for subject, original_indices in self.indices_by_subject.items():
            indices = original_indices.copy()
            if self.shuffle and indices.size > 1:
                indices = indices[rng.permutation(indices.size)]
            for start in range(0, int(indices.size), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                if self.drop_last and batch_indices.size < self.batch_size:
                    continue
                subject_batches.append((subject, batch_indices))

        if self.shuffle and len(subject_batches) > 1:
            order = rng.permutation(len(subject_batches))
            subject_batches = [subject_batches[int(index)] for index in order]

        for subject, indices in subject_batches:
            yield [(int(subject), int(index)) for index in indices]
