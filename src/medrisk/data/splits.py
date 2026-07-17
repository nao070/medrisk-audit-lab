"""Patient-level splitting.

Splits are made on unique ``patient_nbr`` values, never on rows, so every
encounter of a patient falls into exactly one split
(docs/DATA_AND_MODEL_PLAN.md). Split ID lists are versioned, hashed, and
committed; raw data is not.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "calibration", "validation", "test")


class SplitOverlapError(RuntimeError):
    """Raised when a patient appears in more than one split."""


def split_patients(
    patient_ids: Iterable[int], fractions: Mapping[str, float], seed: int
) -> dict[str, list[int]]:
    """Deterministically partition unique patient IDs by the given fractions."""
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"fractions must define exactly {SPLIT_NAMES}")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    unique_ids = np.array(sorted(set(int(i) for i in patient_ids)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    splits: dict[str, list[int]] = {}
    start = 0
    for name in SPLIT_NAMES[:-1]:
        size = round(n * fractions[name])
        splits[name] = sorted(int(i) for i in unique_ids[start : start + size])
        start += size
    splits[SPLIT_NAMES[-1]] = sorted(int(i) for i in unique_ids[start:])
    assert_no_overlap(splits)
    return splits


def assert_no_overlap(splits: Mapping[str, list[int]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                raise SplitOverlapError(f"{len(overlap)} patient(s) appear in both '{a}' and '{b}'")


def split_hash(splits: Mapping[str, list[int]]) -> str:
    canonical = "\n".join(
        f"{name}:{','.join(str(i) for i in splits[name])}" for name in SPLIT_NAMES
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_splits(
    splits: Mapping[str, list[int]], out_dir: Path, seed: int, fractions: Mapping[str, float]
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        (out_dir / f"{name}.txt").write_text("\n".join(str(i) for i in splits[name]) + "\n")
    digest = split_hash(splits)
    manifest = {
        "seed": seed,
        "fractions": dict(fractions),
        "patients_per_split": {name: len(splits[name]) for name in SPLIT_NAMES},
        "sha256": digest,
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Split lists written to %s (sha256=%s)", out_dir, digest)
    return digest


def load_splits(out_dir: Path) -> dict[str, list[int]]:
    splits = {
        name: [int(line) for line in (out_dir / f"{name}.txt").read_text().split()]
        for name in SPLIT_NAMES
    }
    assert_no_overlap(splits)
    return splits


def assign_rows(df: pd.DataFrame, splits: Mapping[str, list[int]]) -> pd.Series:
    """Return, per row, the split its patient belongs to."""
    mapping: dict[int, str] = {}
    for name in SPLIT_NAMES:
        for pid in splits[name]:
            mapping[pid] = name
    assigned = df["patient_nbr"].map(mapping)
    if assigned.isna().any():
        missing = int(assigned.isna().sum())
        raise ValueError(f"{missing} row(s) belong to patients absent from the split lists")
    return assigned
