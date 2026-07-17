"""Smoke tests for the Streamlit demo app (key-free, offline)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "apps" / "streamlit_app" / "app.py")

PAGES = (
    "Overview",
    "Model results",
    "Calibration & thresholds",
    "Subgroup audit",
    "Abstention",
    "LLM explanations (Demo)",
    "LLM evaluation",
)


def _run(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exception(page: str) -> None:
    at = _run(page)
    assert not at.exception, at.exception


def test_disclaimer_is_always_visible() -> None:
    for page in PAGES:
        at = _run(page)
        sidebar_text = " ".join(w.value for w in at.sidebar.warning)
        assert "NOT a medical device" in sidebar_text, page


def test_no_free_text_input_exists() -> None:
    for page in PAGES:
        at = _run(page)
        assert len(at.text_input) == 0, page
        assert len(at.text_area) == 0, page


def test_demo_page_shows_stub_provenance() -> None:
    at = _run("LLM explanations (Demo)")
    captions = " ".join(c.value for c in at.caption)
    assert "StubClient" in captions
    assert "no live LLM call" in captions
