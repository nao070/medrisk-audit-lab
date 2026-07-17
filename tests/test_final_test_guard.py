"""The one-shot final-test guard must refuse a second run."""

import pytest

from medrisk.stages import final_test


def test_final_test_refuses_second_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    marker = tmp_path / "final_test_metrics.json"
    marker.write_text("{}")
    monkeypatch.setattr(final_test, "FINAL_METRICS", marker)
    with pytest.raises(final_test.FinalTestAlreadyRunError):
        final_test.run_final_test(config=None)  # guard fires before config is used


def test_only_final_test_module_loads_the_test_split_for_evaluation() -> None:
    """Grep-style guard: no other stage evaluates assignment == 'test'."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "medrisk"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        if '== "test"' in text and path.name != "final_test.py":
            offenders.append(str(path))
    assert not offenders, offenders
