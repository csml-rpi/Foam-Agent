"""Shared string helpers for interpreter (from cfd-scientist)."""

from __future__ import annotations

import re

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def strip_json_fences(text: str) -> str:
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text
