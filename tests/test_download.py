from pathlib import Path

import pytest

from medrisk.data.download import ChecksumMismatchError, sha256_of_file, verify_archive


def test_sha256_of_file_matches_known_digest(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"medrisk")
    assert sha256_of_file(f) == ("6a1bfbeff24c0b355bcf7996d30970333e9d5da3b727b76a4bab8c78ba7b21e2")


def test_verify_archive_aborts_on_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "archive.zip"
    f.write_bytes(b"tampered content")
    expected = "0" * 64
    with pytest.raises(ChecksumMismatchError) as excinfo:
        verify_archive(f, expected)
    # The message names both hashes so the user can diagnose (hashes are not secrets).
    assert expected in str(excinfo.value)
    assert sha256_of_file(f) in str(excinfo.value)


def test_verify_archive_accepts_correct_hash(tmp_path: Path) -> None:
    f = tmp_path / "archive.zip"
    f.write_bytes(b"good content")
    assert verify_archive(f, sha256_of_file(f)) == sha256_of_file(f)
