"""Lightweight boundary tests for the benchmark CLI wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "foambench_main_under_test", ROOT / "foambench_main.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_wrapper_uses_default_prompt_without_precreating_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    output = tmp_path / "not-created-by-wrapper"
    captured: list[list[str]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["foambench_main.py", "--output", str(output)],
    )
    monkeypatch.setattr(cli, "run_command", lambda command: captured.append(command))

    cli.main()

    assert not output.exists()
    assert captured == [
        [
            sys.executable,
            "src/main.py",
            "--output_dir",
            str(output),
            "--prompt_path",
            str(ROOT / "user_requirement.txt"),
        ]
    ]


def test_case_wrapper_forwards_case_arguments_without_creating_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    output = tmp_path / "not-created-by-wrapper"
    case_path = tmp_path / "input case.zip"
    captured: list[list[str]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "foambench_main.py",
            "--output",
            str(output),
            "--case_path",
            str(case_path),
            "--case_subdir",
            "nested/cavity",
            "--visualize",
            "--overwrite_output",
        ],
    )
    monkeypatch.setattr(cli, "run_command", lambda command: captured.append(command))

    cli.main()

    assert not output.exists()
    assert captured == [
        [
            sys.executable,
            "src/main.py",
            "--output_dir",
            str(output),
            "--case_path",
            str(case_path),
            "--case_subdir",
            "nested/cavity",
            "--visualize",
            "--overwrite_output",
        ]
    ]


def test_openfoam_path_sources_the_environment_for_the_child_workflow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cli = _load_cli_module()
    openfoam_root = tmp_path / "openfoam10"
    (openfoam_root / "etc").mkdir(parents=True)
    (openfoam_root / "etc" / "bashrc").write_text("# test", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        sys,
        "argv",
        ["foambench_main.py", "--openfoam_path", str(openfoam_root)],
    )
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda command, *, env=None, openfoam_bashrc=None: captured.update(
            command=command,
            env=env,
            openfoam_bashrc=openfoam_bashrc,
        ),
    )

    cli.main()

    assert captured["env"]["WM_PROJECT_DIR"] == str(openfoam_root)
    assert captured["openfoam_bashrc"] == str(openfoam_root / "etc" / "bashrc")


def test_run_command_sources_bashrc_before_executing_the_workflow(monkeypatch) -> None:
    cli = _load_cli_module()
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli.run_command(
        ["python", "src/main.py"],
        env={"WM_PROJECT_DIR": "/opt/openfoam10"},
        openfoam_bashrc="/opt/openfoam10/etc/bashrc",
    )

    assert captured["command"] == [
        "bash",
        "-c",
        'source "$1" && shift && exec "$@"',
        "foamagent-benchmark",
        "/opt/openfoam10/etc/bashrc",
        "python",
        "src/main.py",
    ]
    assert captured["kwargs"]["env"] == {"WM_PROJECT_DIR": "/opt/openfoam10"}
