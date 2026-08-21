"""Credential diagnostics must identify the selected LLM authentication path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import LLMService  # noqa: E402


def test_missing_codex_oauth_explains_how_to_select_api_key_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bare API key must not look like a broken Codex login."""
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    service = object.__new__(LLMService)
    with pytest.raises(FileNotFoundError) as exc_info:
        service._load_codex_oauth()

    message = str(exc_info.value)
    assert "model_provider='openai-codex'" in message
    assert "FOAMAGENT_MODEL_PROVIDER=openai" in message
