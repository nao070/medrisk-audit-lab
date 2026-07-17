"""Mode-aware client selection.

Demo is the default and needs no key. Live requires MEDRISK_LLM_MODE=live
plus a locally configured key; anything missing falls back to Demo with a
clear message — missing keys never crash any feature
(docs/PRODUCT_REQUIREMENTS.md).
"""

from __future__ import annotations

import logging
from pathlib import Path

from medrisk.config import LLM_MODE_ENV, anthropic_api_key, resolve_llm_mode
from medrisk.llm.clients import AnthropicClient, LLMClient, ReplayClient

logger = logging.getLogger(__name__)

STORED_OUTPUTS = Path("reports/llm/stored_outputs.json")

LIVE_DISABLED_MESSAGE = (
    "Live LLM Mode is not active (set {env}=live and configure a local API key "
    "to enable it). Falling back to Demo LLM Mode with stored outputs."
)


def make_client(stored_outputs_path: Path = STORED_OUTPUTS) -> tuple[LLMClient, str]:
    """Return (client, mode_actually_used)."""
    mode = resolve_llm_mode()
    if mode == "off":
        raise RuntimeError("LLM features are disabled (MEDRISK_LLM_MODE=off)")
    if mode == "live":
        key = anthropic_api_key()
        if key is not None:
            return AnthropicClient(api_key=key), "live"
        logger.warning(LIVE_DISABLED_MESSAGE.format(env=LLM_MODE_ENV))
    return ReplayClient.from_file(stored_outputs_path), "demo"
