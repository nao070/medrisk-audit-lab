"""Configuration loading.

This module is the only place allowed to read configuration sources.
No secrets exist in v0.1 configuration; when Live LLM Mode arrives (v0.4),
environment access will also be confined to this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    url: str
    sha256: str
    raw_dir: Path
    csv_name: str


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    fractions: dict[str, float]
    out_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    seed: int
    max_iter: int


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    preprocessing_version: str
    processed_dir: Path
    split: SplitConfig
    model: ModelConfig
    reports_dir: Path
    config_sha256: str


def load_config(path: Path) -> AppConfig:
    raw_bytes = path.read_bytes()
    data: dict[str, Any] = yaml.safe_load(raw_bytes)
    dataset = DatasetConfig(
        url=data["dataset"]["url"],
        sha256=data["dataset"]["sha256"],
        raw_dir=Path(data["dataset"]["raw_dir"]),
        csv_name=data["dataset"]["csv_name"],
    )
    split = SplitConfig(
        seed=int(data["split"]["seed"]),
        fractions={k: float(v) for k, v in data["split"]["fractions"].items()},
        out_dir=Path(data["split"]["out_dir"]),
    )
    model = ModelConfig(
        seed=int(data["model"]["seed"]),
        max_iter=int(data["model"]["max_iter"]),
    )
    return AppConfig(
        dataset=dataset,
        preprocessing_version=str(data["preprocessing"]["version"]),
        processed_dir=Path(data["preprocessing"]["processed_dir"]),
        split=split,
        model=model,
        reports_dir=Path(data["reports_dir"]),
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
