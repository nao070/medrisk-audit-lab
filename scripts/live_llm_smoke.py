"""Manual live-LLM smoke test (paid API; never run in normal CI).

Sends ONE stored evaluation case through the live client and runs the same
validators used for stored outputs. Inputs/outputs are not persisted; only
validator verdicts are printed. Requires MEDRISK_LLM_MODE=live and a local
key; refuses to run otherwise.
"""

from __future__ import annotations

import sys

from medrisk.config import anthropic_api_key, resolve_llm_mode
from medrisk.llm.evalset import load_cases
from medrisk.llm.validators import validate_case
from medrisk.stages.llm_eval import CASES_PATH


def main() -> int:
    if resolve_llm_mode() != "live" or anthropic_api_key() is None:
        print("Refusing: MEDRISK_LLM_MODE=live and a local API key are required.")
        return 1
    from medrisk.llm.clients import AnthropicClient

    client = AnthropicClient(api_key=anthropic_api_key() or "")
    case = next(c for c in load_cases(CASES_PATH) if c.corruption is None)
    raw = client.generate("full_contract", case)
    verdicts = validate_case(case, [raw])
    for dimension, verdict in verdicts.items():
        print(f"{dimension}: {'PASS' if verdict['passed'] else 'FAIL'}")
    return 0 if all(v["passed"] for v in verdicts.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
