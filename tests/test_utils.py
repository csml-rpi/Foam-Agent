"""Unit tests for deterministic helpers in foamagent.utils.

Everything here is offline: no network, no credentials, no docker, no real LLM call.
Fake LLM clients are hand-rolled objects presenting only the interface LLMService.invoke()
actually uses (get_num_tokens / with_structured_output / invoke).
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import BaseModel, ValidationError

from foamagent.config import Config
from foamagent.services import get_llm_service, set_llm_service
from foamagent.utils import (
    LLMService,
    _botocore_client_error,
    _build_openfoam_argv,
    _lfs_pointer_reason,
)


# ---------------------------------------------------------------------------
# _build_openfoam_argv
# ---------------------------------------------------------------------------


def _make_openfoam_root(tmp_path):
    """Create a fake $WM_PROJECT_DIR with an etc/bashrc so native runs succeed."""
    root = tmp_path / "openfoam_root"
    (root / "etc").mkdir(parents=True)
    (root / "etc" / "bashrc").write_text("# fake bashrc\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("runtime_env", [None, "native"])
def test_build_openfoam_argv_native(tmp_path, monkeypatch, runtime_env):
    if runtime_env is None:
        monkeypatch.delenv("FOAMAGENT_OPENFOAM_RUNTIME", raising=False)
    else:
        monkeypatch.setenv("FOAMAGENT_OPENFOAM_RUNTIME", runtime_env)

    root = _make_openfoam_root(tmp_path)
    monkeypatch.setenv("WM_PROJECT_DIR", str(root))

    work_dir = tmp_path / "case"
    work_dir.mkdir()
    script = work_dir / "Allrun"
    script.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")

    argv, container_name = _build_openfoam_argv(str(script), str(work_dir))

    assert container_name is None
    assert argv[0] == "bash"
    assert argv[1] == "-c"
    # The single shell command must source the bashrc and run the script under bash.
    bashrc_path = str(root / "etc" / "bashrc")
    command = argv[2]
    assert f"source {bashrc_path}" in command
    assert "&& bash " in command
    assert str(script) in command


def test_build_openfoam_argv_native_relative_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("FOAMAGENT_OPENFOAM_RUNTIME", raising=False)
    root = _make_openfoam_root(tmp_path)
    monkeypatch.setenv("WM_PROJECT_DIR", str(root))

    work_dir = tmp_path / "case"
    work_dir.mkdir()
    (work_dir / "Allrun").write_text("echo hi\n", encoding="utf-8")

    monkeypatch.chdir(work_dir)
    argv, _ = _build_openfoam_argv("Allrun", ".")

    # Relative inputs must be resolved to absolute paths in the resulting command.
    command = argv[2]
    assert os.path.isabs(str(work_dir / "Allrun"))
    assert str(work_dir / "Allrun") in command


def test_build_openfoam_argv_native_missing_wm_project_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("FOAMAGENT_OPENFOAM_RUNTIME", raising=False)
    monkeypatch.delenv("WM_PROJECT_DIR", raising=False)

    with pytest.raises(RuntimeError, match="WM_PROJECT_DIR"):
        _build_openfoam_argv(str(tmp_path / "Allrun"), str(tmp_path))


def test_build_openfoam_argv_docker(tmp_path, monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_RUNTIME", "docker")
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_IMAGE", "my-openfoam-image:test")
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_BASHRC", "/opt/openfoam10/etc/bashrc")

    work_dir = tmp_path / "case"
    work_dir.mkdir()
    script = work_dir / "Allrun"
    script.write_text("echo hi\n", encoding="utf-8")

    argv, container_name = _build_openfoam_argv(str(script), str(work_dir))

    abs_work_dir = str(work_dir.resolve())

    assert container_name
    assert argv[0] == "docker"
    assert argv[1] == "run"
    assert argv[2] == "--rm"
    assert "my-openfoam-image:test" in argv

    # The working dir must be mounted at the SAME absolute path on both sides.
    v_index = argv.index("-v")
    assert argv[v_index + 1] == f"{abs_work_dir}:{abs_work_dir}"

    w_index = argv.index("-w")
    assert argv[w_index + 1] == abs_work_dir

    user_index = argv.index("--user")
    assert argv[user_index + 1] == f"{os.getuid()}:{os.getgid()}"

    # The inner bash command sources the configured bashrc.
    c_index = argv.index("-c")
    inner_command = argv[c_index + 1]
    assert "source /opt/openfoam10/etc/bashrc" in inner_command
    assert str(script.resolve()) in inner_command


def test_build_openfoam_argv_docker_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_RUNTIME", "docker")
    monkeypatch.delenv("FOAMAGENT_OPENFOAM_IMAGE", raising=False)
    monkeypatch.delenv("FOAMAGENT_OPENFOAM_BASHRC", raising=False)

    work_dir = tmp_path / "case"
    work_dir.mkdir()
    (work_dir / "Allrun").write_text("echo hi\n", encoding="utf-8")

    monkeypatch.chdir(work_dir)
    argv, container_name = _build_openfoam_argv("Allrun", ".")

    abs_work_dir = str(work_dir.resolve())
    v_index = argv.index("-v")
    assert argv[v_index + 1] == f"{abs_work_dir}:{abs_work_dir}"
    assert container_name


# ---------------------------------------------------------------------------
# _lfs_pointer_reason
# ---------------------------------------------------------------------------


def test_lfs_pointer_reason_detects_pointer(tmp_path):
    pointer_file = tmp_path / "index.faiss"
    pointer_file.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"a" * 64 + b"\n"
        b"size 1234\n"
    )

    reason = _lfs_pointer_reason(pointer_file)
    assert reason is not None
    assert "git lfs pull" in reason
    assert str(pointer_file) in reason


def test_lfs_pointer_reason_normal_binary_file(tmp_path):
    binary_file = tmp_path / "index.faiss"
    binary_file.write_bytes(bytes(range(256)) * 4)

    assert _lfs_pointer_reason(binary_file) is None


def test_lfs_pointer_reason_non_pointer_text_file(tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("just some ordinary text content\n", encoding="utf-8")

    assert _lfs_pointer_reason(text_file) is None


def test_lfs_pointer_reason_nonexistent_path(tmp_path):
    missing = tmp_path / "does-not-exist.faiss"
    assert _lfs_pointer_reason(missing) is None


# ---------------------------------------------------------------------------
# _botocore_client_error
# ---------------------------------------------------------------------------


def test_botocore_client_error_does_not_raise():
    result = _botocore_client_error()
    # Either botocore isn't installed (None) or its ClientError type is returned.
    assert result is None or isinstance(result, type)


# ---------------------------------------------------------------------------
# LLMService: laziness, error classification, statistics
# ---------------------------------------------------------------------------


def test_llm_service_construction_is_lazy(monkeypatch):
    # No credentials of any kind are set, and the provider name is bogus. Construction
    # must still succeed because the client is only built on first access to `.llm`.
    monkeypatch.delenv("FOAMAGENT_MODEL_PROVIDER", raising=False)
    cfg = Config(model_provider="totally-unsupported-provider")
    service = LLMService(cfg)

    assert service._llm is None  # nothing was built yet


def test_llm_service_llm_property_triggers_build_and_raises(monkeypatch):
    monkeypatch.delenv("FOAMAGENT_MODEL_PROVIDER", raising=False)
    cfg = Config(model_provider="totally-unsupported-provider")
    service = LLMService(cfg)

    # Construction succeeded above; the ValueError only happens on attribute access.
    with pytest.raises(ValueError, match="not a supported model_provider"):
        service.llm


def test_is_structured_output_error_validation_error():
    class Dummy(BaseModel):
        value: str

    service = LLMService(Config())
    try:
        Dummy.model_validate({})
    except ValidationError as exc:
        assert service._is_structured_output_error(exc) is True
    else:
        pytest.fail("expected ValidationError")


def test_is_structured_output_error_json_decode_error():
    service = LLMService(Config())
    try:
        json.loads("not json")
    except json.JSONDecodeError as exc:
        assert service._is_structured_output_error(exc) is True
    else:
        pytest.fail("expected JSONDecodeError")


def test_is_structured_output_error_false_for_plain_error():
    service = LLMService(Config())
    assert service._is_structured_output_error(RuntimeError("boom")) is False


def test_get_statistics_fresh_instance_has_no_zero_division():
    service = LLMService(Config())
    stats = service.get_statistics()

    assert stats["total_calls"] == 0
    assert stats["failed_calls"] == 0
    assert stats["retry_count"] == 0
    assert stats["total_prompt_tokens"] == 0
    assert stats["total_completion_tokens"] == 0
    assert stats["total_tokens"] == 0
    assert stats["average_prompt_tokens"] == 0
    assert stats["average_completion_tokens"] == 0
    assert stats["average_tokens"] == 0


class _DummyModel(BaseModel):
    value: str


class _FakeStructuredLLM:
    """Mimics `<llm>.with_structured_output(model)`'s return value.

    Fails validation on the first call, then succeeds -- matching what LLMService.invoke()
    expects from `structured_llm.invoke(messages)`.
    """

    def __init__(self, outer):
        self._outer = outer

    def invoke(self, messages):
        self._outer.attempts += 1
        if self._outer.attempts == 1:
            raise self._outer.validation_error
        return self._outer.valid_response


class _FakeLLM:
    """Mimics the minimal LangChain chat-model interface LLMService.invoke() relies on."""

    def __init__(self, validation_error, valid_response):
        self.attempts = 0
        self.validation_error = validation_error
        self.valid_response = valid_response

    def get_num_tokens(self, text):
        return len(text.split())

    def with_structured_output(self, pydantic_obj):
        return _FakeStructuredLLM(self)


def test_invoke_retries_once_on_structured_output_error():
    try:
        _DummyModel.model_validate({})
        validation_error = None
    except ValidationError as exc:
        validation_error = exc
    assert validation_error is not None

    valid_response = _DummyModel(value="ok")
    fake_llm = _FakeLLM(validation_error, valid_response)

    service = LLMService(Config())
    service.llm = fake_llm  # uses the property setter; bypasses _build_llm entirely

    result = service.invoke("please answer", pydantic_obj=_DummyModel)

    assert result is valid_response
    assert fake_llm.attempts == 2  # first attempt failed validation, second succeeded
    assert service.retry_count == 1
    assert service.total_calls == 1
    assert service.failed_calls == 0


# ---------------------------------------------------------------------------
# foamagent.services.get_llm_service / set_llm_service
# ---------------------------------------------------------------------------


def test_get_llm_service_singleton_and_reset():
    set_llm_service(None)  # start from a clean slate regardless of test order
    try:
        first = get_llm_service(Config(model_provider="totally-unsupported-provider"))
        second = get_llm_service()
        assert first is second

        set_llm_service(None)
        third = get_llm_service(Config(model_provider="totally-unsupported-provider"))
        assert third is not first
    finally:
        set_llm_service(None)  # do not leak the module-level singleton to other tests
