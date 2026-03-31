"""Load static chatbot assets from external files."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "chatbot_config.json"
PROMPT_PATH = PROJECT_ROOT / "faq_assistant_prompt.txt"
REQUIRED_CONFIG_KEYS = {
    "messages",
    "supported_keywords",
    "intent_priority",
    "query_filler_words",
    "security_keyword_groups",
    "follow_up_phrases",
    "referential_terms",
    "code_markers",
}


@lru_cache(maxsize=1)
def load_chatbot_config() -> dict[str, Any]:
    """Read and validate the static chatbot config."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    missing_keys = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise KeyError(f"chatbot_config.json is missing required keys: {missing_text}")
    return config


@lru_cache(maxsize=1)
def load_prompt_template() -> str:
    """Read the external prompt template text."""
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt_template:
        raise ValueError("faq_assistant_prompt.txt is empty.")
    return prompt_template
