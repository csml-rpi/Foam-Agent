"""Regression tests for transient structured LLM response recovery."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel
import requests


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import LLMService  # noqa: E402


class _ResponseSchema(BaseModel):
    value: int


class _StructuredModel:
    def __init__(self) -> None:
        self.calls = 0

    def get_num_tokens(self, _text: str) -> int:
        return 1

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("Empty response; expected JSON")
        return _ResponseSchema(value=7)


class _TransportThenSuccessModel:
    def __init__(self) -> None:
        self.calls = 0

    def get_num_tokens(self, _text: str) -> int:
        return 1

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise requests.exceptions.SSLError("unexpected EOF while reading")
        return type("Response", (), {"content": "recovered"})()


def _service_with(model: _StructuredModel) -> LLMService:
    """Construct only the public invoke state, avoiding provider credentials."""
    service = object.__new__(LLMService)
    service.llm = model
    service.model_provider = "openai"
    service.total_calls = 0
    service.failed_calls = 0
    service.retry_count = 0
    service.total_prompt_tokens = 0
    service.total_completion_tokens = 0
    service.total_tokens = 0
    return service


def test_structured_empty_response_is_retried(monkeypatch) -> None:
    model = _StructuredModel()
    service = _service_with(model)
    monkeypatch.setattr("utils.time.sleep", lambda _delay: None)

    result = service.invoke("request", pydantic_obj=_ResponseSchema)

    assert result == _ResponseSchema(value=7)
    assert model.calls == 2
    assert service.retry_count == 1
    assert service.failed_calls == 0


def test_only_empty_or_missing_json_errors_are_retryable() -> None:
    assert LLMService._is_retryable_structured_response_error(
        ValueError("Empty response; expected JSON")
    )
    assert LLMService._is_retryable_structured_response_error(
        ValueError("Could not find a JSON object in response: <truncated>")
    )
    assert not LLMService._is_retryable_structured_response_error(
        ValueError("field required")
    )


def test_transient_transport_error_is_retried(monkeypatch) -> None:
    model = _TransportThenSuccessModel()
    service = _service_with(model)
    monkeypatch.setattr("utils.time.sleep", lambda _delay: None)

    result = service.invoke("request")

    assert result == "recovered"
    assert model.calls == 2
    assert service.retry_count == 1
    assert service.failed_calls == 0


def test_only_transport_errors_are_retryable() -> None:
    assert LLMService._is_retryable_transport_error(
        requests.exceptions.SSLError("unexpected EOF while reading")
    )
    assert LLMService._is_retryable_transport_error(
        requests.exceptions.ConnectTimeout("connection timed out")
    )
    assert not LLMService._is_retryable_transport_error(
        requests.exceptions.HTTPError("401 unauthorized")
    )
