"""Run manifests: the reproducibility record of every pipeline run.

A manifest contains hashes, versions, and metrics — never raw data rows and
never any secret (docs/SECURITY_DESIGN.md).
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "pyyaml")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def package_versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in TRACKED_PACKAGES}


def build_manifest(
    *,
    stage: str,
    dataset_sha256: str,
    preprocessing_version: str,
    split_sha256: str,
    config_sha256: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "run_id": f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": stage,
        "git_commit": _git_commit(),
        "dataset_sha256": dataset_sha256,
        "preprocessing_version": preprocessing_version,
        "split_sha256": split_sha256,
        "config_sha256": config_sha256,
        "seed": seed,
        "package_versions": package_versions(),
        "final_test_evaluations": 0,  # must remain 0 until the v1.0 one-shot evaluation
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict[str, Any], reports_dir: Path) -> Path:
    run_dir = reports_dir / "runs" / str(manifest["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
