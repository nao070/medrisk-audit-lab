"""Dataset acquisition with mandatory SHA-256 verification.

The raw archive and its extracted contents live under the gitignored
``data/raw`` directory and are never committed (docs/DATA_AND_MODEL_PLAN.md).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

ARCHIVE_NAME = "diabetes_130_us_hospitals_1999_2008.zip"


class ChecksumMismatchError(RuntimeError):
    """Raised when the downloaded archive does not match the pinned SHA-256."""


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset archive from %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "medrisk-audit-lab/0.1"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    logger.info("Saved archive to %s", dest)


def verify_archive(archive: Path, expected_sha256: str) -> str:
    actual = sha256_of_file(archive)
    if actual != expected_sha256:
        raise ChecksumMismatchError(
            f"Dataset archive checksum mismatch: expected {expected_sha256}, got {actual}. "
            "Refusing to proceed. Delete the file and re-run the download, or update the "
            "pinned checksum only after verifying the source."
        )
    return actual


def ensure_dataset(url: str, expected_sha256: str, raw_dir: Path, csv_name: str) -> Path:
    """Download (if absent), verify, and extract the dataset CSV.

    Returns the path to the extracted CSV. Aborts on checksum mismatch.
    """
    archive = raw_dir / ARCHIVE_NAME
    if not archive.exists():
        download_archive(url, archive)
    else:
        logger.info("Archive already present at %s; skipping download", archive)
    verify_archive(archive, expected_sha256)

    csv_path = raw_dir / csv_name
    if not csv_path.exists():
        with zipfile.ZipFile(archive) as zf:
            member = next((name for name in zf.namelist() if Path(name).name == csv_name), None)
            if member is None:
                raise FileNotFoundError(f"{csv_name} not found inside {archive}")
            zf.extract(member, raw_dir)
            extracted = raw_dir / member
            if extracted != csv_path:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                extracted.replace(csv_path)
    logger.info("Dataset CSV available at %s", csv_path)
    return csv_path
