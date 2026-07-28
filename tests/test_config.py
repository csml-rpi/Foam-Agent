"""Unit tests for foamagent.config.Config and foamagent.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from foamagent import paths
from foamagent.config import Config

# All FOAMAGENT_* env vars that Config()/paths read. Cleared before every test so
# ambient shell state (or leakage from a previous test) can't affect the assertions.
_ENV_KEYS = [
    "FOAMAGENT_MODEL_PROVIDER",
    "FOAMAGENT_MODEL_VERSION",
    "FOAMAGENT_EMBEDDING_PROVIDER",
    "FOAMAGENT_EMBEDDING_MODEL",
    "FOAMAGENT_OPENAI_BASE_URL",
    "FOAMAGENT_OPENFOAM_RUNTIME",
    "FOAMAGENT_OPENFOAM_IMAGE",
    "FOAMAGENT_OPENFOAM_BASHRC",
    "FOAMAGENT_OPENFOAM_FORK",
    "FOAMAGENT_MAX_LOOP",
    "FOAMAGENT_MAX_TIME_LIMIT",
    "FOAMAGENT_ROOT",
    "FOAMAGENT_DATABASE_PATH",
    "FOAMAGENT_RUN_DIRECTORY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no FOAMAGENT_* env var leaks in from the ambient shell or a prior test."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------


def test_defaults_no_env():
    cfg = Config()
    assert cfg.model_provider == "openai"
    assert cfg.openfoam_runtime == "native"
    assert cfg.openfoam_fork == "foundation"


def test_default_does_not_read_codex_credentials(monkeypatch, tmp_path):
    """Regression guard: the old default was "openai-codex", which reads a Codex OAuth
    cache from disk on construction. Point HOME somewhere with no such file and confirm
    Config() still builds fine and never picks the codex provider by default."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    # No credential file exists anywhere under fake_home.
    codex_auth = fake_home / ".codex" / "auth.json"
    assert not codex_auth.exists()

    cfg = Config()

    assert cfg.model_provider == "openai"
    assert cfg.model_provider != "openai-codex"
    # Still no credential file was created or expected to exist afterwards.
    assert not codex_auth.exists()


# ---------------------------------------------------------------------------
# 2. Each override actually applies
# ---------------------------------------------------------------------------


def test_override_model_provider(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_MODEL_PROVIDER", "anthropic")
    assert Config().model_provider == "anthropic"


def test_override_model_version(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_MODEL_VERSION", "gpt-5.3-codex")
    assert Config().model_version == "gpt-5.3-codex"


def test_override_embedding_provider(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_EMBEDDING_PROVIDER", "openai")
    assert Config().embedding_provider == "openai"


def test_override_embedding_model(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_EMBEDDING_MODEL", "text-embedding-3-large")
    assert Config().embedding_model == "text-embedding-3-large"


def test_override_openai_base_url(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    assert Config().openai_base_url == "https://openrouter.ai/api/v1"


def test_override_openfoam_runtime(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_RUNTIME", "docker")
    assert Config().openfoam_runtime == "docker"


def test_override_openfoam_image(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_IMAGE", "custom-foam:v1")
    assert Config().openfoam_image == "custom-foam:v1"


def test_override_openfoam_bashrc(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_BASHRC", "/opt/openfoam11/etc/bashrc")
    assert Config().openfoam_bashrc == "/opt/openfoam11/etc/bashrc"


def test_override_openfoam_fork(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_FORK", "esi")
    assert Config().openfoam_fork == "esi"


def test_override_max_loop(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_MAX_LOOP", "42")
    assert Config().max_loop == 42


def test_override_max_time_limit(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_MAX_TIME_LIMIT", "7200")
    assert Config().max_time_limit == 7200


# ---------------------------------------------------------------------------
# 3. Invalid values fall back to default instead of raising
# ---------------------------------------------------------------------------


def test_invalid_model_provider_falls_back(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_MODEL_PROVIDER", "not-a-real-provider")
    cfg = Config()
    assert cfg.model_provider == "openai"


def test_invalid_embedding_provider_falls_back(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_EMBEDDING_PROVIDER", "not-a-real-provider")
    cfg = Config()
    assert cfg.embedding_provider == "huggingface"


def test_invalid_openfoam_runtime_falls_back(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_RUNTIME", "kubernetes")
    cfg = Config()
    assert cfg.openfoam_runtime == "native"


def test_invalid_openfoam_fork_falls_back(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_FORK", "extend")
    cfg = Config()
    assert cfg.openfoam_fork == "foundation"


def test_invalid_max_loop_falls_back(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_MAX_LOOP", "not-an-int")
    cfg = Config()
    assert cfg.max_loop == 25


# ---------------------------------------------------------------------------
# 4. Empty-string / whitespace-only env values treated as unset
# ---------------------------------------------------------------------------


def test_empty_string_env_treated_as_unset(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_MODEL_PROVIDER", "")
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_FORK", "")
    monkeypatch.setenv("FOAMAGENT_MAX_LOOP", "")
    cfg = Config()
    assert cfg.model_provider == "openai"
    assert cfg.openfoam_fork == "foundation"
    assert cfg.max_loop == 25


def test_whitespace_only_env_treated_as_unset(monkeypatch):
    monkeypatch.setenv("FOAMAGENT_MODEL_PROVIDER", "   ")
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_FORK", "\t")
    monkeypatch.setenv("FOAMAGENT_MAX_LOOP", "  \n")
    cfg = Config()
    assert cfg.model_provider == "openai"
    assert cfg.openfoam_fork == "foundation"
    assert cfg.max_loop == 25


def test_whitespace_padded_valid_value_is_stripped(monkeypatch):
    # _env_nonempty() strips before comparing/assigning, so padded-but-valid values apply.
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_FORK", "  esi  ")
    cfg = Config()
    assert cfg.openfoam_fork == "esi"


# ---------------------------------------------------------------------------
# 5. Config() must never write to stdout (stdout is the MCP stdio protocol channel)
# ---------------------------------------------------------------------------


def test_config_writes_nothing_to_stdout(capsys):
    Config()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_config_writes_nothing_to_stdout_with_overrides(monkeypatch, capsys):
    # Exercise every branch (valid override, invalid override, unset) to make sure none
    # of them leak a print() to stdout.
    monkeypatch.setenv("FOAMAGENT_MODEL_PROVIDER", "bogus")
    monkeypatch.setenv("FOAMAGENT_OPENFOAM_RUNTIME", "docker")
    monkeypatch.setenv("FOAMAGENT_MAX_LOOP", "not-an-int")
    Config()
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# 6. foamagent.paths
# ---------------------------------------------------------------------------


def test_repo_root_default():
    # <root>/src/foamagent/paths.py -> repo_root() is two levels above src/
    expected = Path(paths.__file__).resolve().parent.parent.parent
    assert paths.repo_root() == expected


def test_repo_root_honours_foamagent_root(monkeypatch, tmp_path):
    monkeypatch.setenv("FOAMAGENT_ROOT", str(tmp_path))
    assert paths.repo_root() == tmp_path.resolve()


def test_database_dir_default_is_repo_root_slash_database():
    assert paths.database_dir() == paths.repo_root() / "database"


def test_runs_dir_default_is_repo_root_slash_runs():
    assert paths.runs_dir() == paths.repo_root() / "runs"


def test_database_dir_honours_foamagent_database_path(monkeypatch, tmp_path):
    custom = tmp_path / "custom_db"
    monkeypatch.setenv("FOAMAGENT_DATABASE_PATH", str(custom))
    assert paths.database_dir() == custom.resolve()


def test_runs_dir_honours_foamagent_run_directory(monkeypatch, tmp_path):
    custom = tmp_path / "custom_runs"
    monkeypatch.setenv("FOAMAGENT_RUN_DIRECTORY", str(custom))
    assert paths.runs_dir() == custom.resolve()


def test_database_dir_ignores_foamagent_root_when_database_path_set(monkeypatch, tmp_path):
    # FOAMAGENT_DATABASE_PATH takes precedence over FOAMAGENT_ROOT for database_dir().
    monkeypatch.setenv("FOAMAGENT_ROOT", str(tmp_path / "root"))
    custom = tmp_path / "custom_db"
    monkeypatch.setenv("FOAMAGENT_DATABASE_PATH", str(custom))
    assert paths.database_dir() == custom.resolve()


def test_config_picks_up_foamagent_root_for_database_and_run_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("FOAMAGENT_ROOT", str(tmp_path))
    cfg = Config()
    # NOTE: database_path/run_directory are annotated `str` on the dataclass, but their
    # default_factory (paths.database_dir/runs_dir) actually returns a pathlib.Path, so
    # the runtime value is a Path, not a str. Compare with Path() to match reality.
    assert Path(cfg.database_path) == tmp_path.resolve() / "database"
    assert Path(cfg.run_directory) == tmp_path.resolve() / "runs"
