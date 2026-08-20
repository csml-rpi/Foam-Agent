"""Restricted Allrun parser and renderer for imported OpenFOAM cases.

Unlike generated cases, an imported ``Allrun`` is treated as untrusted input.
This module turns a small Foundation-v10 subset into a data-only execution
plan, then renders the controlled equivalent used by the runner.
"""

from __future__ import annotations

from pathlib import PurePosixPath
import re
import shlex
from typing import Iterable

from .allrun_commands import (
    is_standard_application_variable,
    normalise_shell_lines,
    strip_shell_comment,
)
from .case_import_models import CaseImportError, ExecutionStep
from .openfoam_commands import MESH_MUTATING_COMMANDS


_SHELL_META_RE = re.compile(r"[;&|`<>]")
_SAFE_COMMAND_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_SAFE_ARGUMENT_RE = re.compile(r"^[A-Za-z0-9_./:=+@%$,-]+$")
_ALLOWED_UTILITY_COMMANDS = frozenset(
    {
        "blockMesh",
        "checkMesh",
        "createBaffles",
        "createNonConformalCouples",
        "decomposePar",
        "fluentMeshToFoam",
        "gmshToFoam",
        "reconstructPar",
        "setFields",
        "snappyHexMesh",
        "splitBaffles",
        "splitMeshRegions",
        "topoSet",
    }
)


def _safe_tokens(command: str, args: Iterable[str]) -> None:
    if not _SAFE_COMMAND_RE.fullmatch(command):
        raise CaseImportError(f"Unsafe command in Allrun: {command!r}")
    for arg in args:
        if arg == "-case" or arg.startswith("-case="):
            raise CaseImportError(
                "Allrun -case arguments are not supported in safe case-import mode."
            )
        if not _SAFE_ARGUMENT_RE.fullmatch(arg) or _SHELL_META_RE.search(arg):
            raise CaseImportError(f"Unsafe argument in Allrun: {arg!r}")
        path_values = (arg.split("=", 1)[-1],) if "=" in arg else (arg,)
        if any(value.startswith("/") for value in path_values):
            raise CaseImportError(
                f"Absolute paths are not allowed in imported Allrun: {arg!r}"
            )
        if any(".." in PurePosixPath(value).parts for value in path_values):
            raise CaseImportError(
                f"Parent-directory traversal is not allowed in Allrun: {arg!r}"
            )
        if "$" in arg:
            raise CaseImportError(f"Unsupported shell expansion in Allrun: {arg!r}")


def _resolve_application(token: str, application: str) -> str:
    return application if is_standard_application_variable(token) else token


def _shell_split(line: str) -> list[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError as exc:
        raise CaseImportError(f"Unable to parse Allrun line {line!r}: {exc}") from exc


def _is_supported_setup_line(line: str) -> bool:
    """Allow only Foundation v10's normal RunFunctions setup boilerplate."""
    normalised = re.sub(r"\s+", " ", line).strip()
    return normalised in {
        ". $WM_PROJECT_DIR/bin/tools/RunFunctions",
        '. "$WM_PROJECT_DIR/bin/tools/RunFunctions"',
        "source $WM_PROJECT_DIR/bin/tools/RunFunctions",
        'source "$WM_PROJECT_DIR/bin/tools/RunFunctions"',
        "cd ${0%/*}",
        'cd "${0%/*}"',
        "cd ${0%/*} || exit 1",
        'cd "${0%/*}" || exit 1',
    }


def _parse_execution_line(line: str, application: str) -> ExecutionStep:
    """Parse one non-setup Allrun line into an allow-listed execution step."""
    tokens = _shell_split(line)
    if not tokens:
        raise CaseImportError(f"Unable to parse Allrun command: {line!r}")
    launcher = tokens.pop(0)
    parallel = launcher == "runParallel"
    if launcher in {"runApplication", "runParallel"}:
        if not tokens:
            raise CaseImportError(f"Allrun command lacks an application: {line!r}")
        command = _resolve_application(tokens.pop(0), application)
    else:
        command = _resolve_application(launcher, application)
    _safe_tokens(command, tokens)
    if command != application and command not in _ALLOWED_UTILITY_COMMANDS:
        raise CaseImportError(
            f"Unsupported command in user Allrun: {command}. "
            "Only Foundation v10 utilities and the controlDict application are allowed."
        )
    if parallel and command != application:
        raise CaseImportError(
            f"runParallel may only execute the controlDict application, not {command}."
        )
    return ExecutionStep(command, tuple(tokens), parallel)


def _parse_allrun_line(line: str, application: str) -> ExecutionStep | None:
    """Ignore known setup lines and parse an allowed OpenFOAM command."""
    if not line or line.startswith("#!"):
        return None
    if line.startswith("application="):
        if not re.fullmatch(r"application\s*=\s*\$?\(?getApplication\)?", line):
            raise CaseImportError(f"Unsupported application assignment in Allrun: {line!r}")
        return None
    if line.startswith((".", "source ", "cd ")):
        if not _is_supported_setup_line(line):
            raise CaseImportError(
                "Unsupported shell setup in Allrun. Only the standard "
                "Foundation v10 RunFunctions source and case-directory cd are allowed."
            )
        return None
    return _parse_execution_line(line, application)


def parse_allrun(allrun_content: str, application: str) -> list[ExecutionStep]:
    """Parse the supported, non-Turing-complete subset of an imported Allrun."""
    try:
        lines = normalise_shell_lines(allrun_content)
    except ValueError as exc:
        raise CaseImportError(str(exc)) from exc

    steps = [
        step
        for raw_line in lines
        if (step := _parse_allrun_line(strip_shell_comment(raw_line), application))
        is not None
    ]
    if not steps:
        raise CaseImportError("Allrun contains no supported OpenFOAM execution commands.")
    return steps


def synthesise_execution_plan(application: str, mesh_state: str) -> list[ExecutionStep]:
    """Infer only the minimal execution plan for a self-contained case."""
    if mesh_state == "existing-polyMesh":
        steps = [ExecutionStep("checkMesh", origin="generated_missing_allrun")]
        if application != "checkMesh":
            steps.append(ExecutionStep(application, origin="generated_missing_allrun"))
        return steps
    if mesh_state == "blockMesh":
        steps = [
            ExecutionStep("blockMesh", origin="generated_missing_allrun"),
            ExecutionStep("checkMesh", origin="generated_missing_allrun"),
        ]
        if application not in {"blockMesh", "checkMesh"}:
            steps.append(ExecutionStep(application, origin="generated_missing_allrun"))
        return steps
    raise CaseImportError(
        "Allrun is missing and the case has neither constant/polyMesh nor "
        "system/blockMeshDict. A safe execution plan cannot be inferred."
    )


def ensure_mesh_check(
    steps: list[ExecutionStep],
    application: str,
) -> list[ExecutionStep]:
    """Ensure the selected application only sees a post-mesh ``checkMesh``."""
    if not any(step.command == application for step in steps):
        raise CaseImportError("Allrun never runs the application declared in system/controlDict.")

    mesh_indexes = [
        index for index, step in enumerate(steps) if step.command in MESH_MUTATING_COMMANDS
    ]
    if mesh_indexes:
        last_mesh_index = max(mesh_indexes)
        steps = [
            step
            for index, step in enumerate(steps)
            if not (step.command == "checkMesh" and index <= last_mesh_index)
        ]
        mesh_indexes = [
            index for index, step in enumerate(steps) if step.command in MESH_MUTATING_COMMANDS
        ]

    check_indexes = [
        index for index, step in enumerate(steps) if step.command == "checkMesh"
    ]
    required_after = max(mesh_indexes) if mesh_indexes else -1
    # ``checkMesh`` can itself be the controlDict application for a mesh-only
    # tutorial.  Its existing invocation is already the required quality gate;
    # inserting another one would run it twice.  If a mesh operation preceded
    # it, the filtering above has removed stale pre-mesh checks and this branch
    # inserts exactly one final check instead.
    if application == "checkMesh":
        if any(index > required_after for index in check_indexes):
            return steps
        repaired = list(steps)
        repaired.insert(required_after + 1, ExecutionStep("checkMesh", origin="foamagent_mesh_gate"))
        return repaired
    if application in MESH_MUTATING_COMMANDS:
        if any(index > required_after for index in check_indexes):
            return steps
        repaired = list(steps)
        repaired.insert(required_after + 1, ExecutionStep("checkMesh", origin="foamagent_mesh_gate"))
        return repaired

    solver_index = next(index for index, step in enumerate(steps) if step.command == application)
    if any(index > solver_index for index in mesh_indexes):
        raise CaseImportError(
            "Allrun modifies the mesh after the configured solver starts; "
            "safe case-import mode requires mesh validation before the solver."
        )
    if any(required_after < index < solver_index for index in check_indexes):
        return steps

    repaired = list(steps)
    repaired.insert(
        required_after + 1 if mesh_indexes else solver_index,
        ExecutionStep("checkMesh", origin="foamagent_mesh_gate"),
    )
    return repaired


def render_controlled_allrun(steps: list[ExecutionStep]) -> str:
    """Render a fail-closed equivalent of the validated execution plan."""
    lines = [
        "#!/bin/sh",
        'cd "${0%/*}/.." || exit 1',
        '. "$WM_PROJECT_DIR/bin/tools/RunFunctions"',
        'if [ "${WM_PROJECT_VERSION:-}" != "10" ]; then',
        '    echo "Foam-Agent case-import requires Foundation OpenFOAM v10 (WM_PROJECT_VERSION=10)." >&2',
        "    exit 64",
        "fi",
        "foamagent_require_stock_command() {",
        '    foamagent_command_path=$(command -v "$1") || {',
        '        echo "Required OpenFOAM command is unavailable: $1" >&2',
        "        return 64",
        "    }",
        '    case "$foamagent_command_path" in',
        '        "$FOAM_APPBIN"/*) return 0 ;;',
        '        *) echo "Custom or non-Foundation command is not allowed: $1 ($foamagent_command_path)" >&2; return 64 ;;',
        "    esac",
        "}",
        "foamagent_require_mesh_ok() {",
        '    foamagent_mesh_log="log.checkMesh"',
        '    if [ ! -f "$foamagent_mesh_log" ]; then',
        '        echo "checkMesh did not produce $foamagent_mesh_log" >&2',
        "        return 65",
        "    fi",
        '    if grep -Eiq "Failed[[:space:]]+[1-9][0-9]*[[:space:]]+mesh checks?" "$foamagent_mesh_log" || ! grep -Eiq "Mesh[[:space:]]+OK" "$foamagent_mesh_log"; then',
        '        echo "checkMesh did not establish a usable mesh; refusing to start a solver." >&2',
        "        return 65",
        "    fi",
        "}",
        "",
    ]
    for step in steps:
        invocation = " ".join((step.command, *step.args)).strip()
        lines.append(f"foamagent_require_stock_command {step.command} || exit $?")
        runner = "runParallel" if step.parallel else "runApplication"
        lines.append(f"{runner} {invocation} || exit $?")
        if step.command == "checkMesh":
            lines.append("foamagent_require_mesh_ok || exit $?")
    return "\n".join(lines) + "\n"
