"""Small, shared helpers for inspecting OpenFOAM ``Allrun`` scripts.

The project has two deliberately different execution policies:

* generated cases may contain normal OpenFOAM shell scripts;
* imported cases accept only a tightly restricted command subset.

Both policies still need to answer the same non-security question: which
OpenFOAM command does a line invoke?  Keeping that recognition in one place
avoids the generated-run, imported-run, and writer paths disagreeing about
the ordering of mesh generation, ``checkMesh``, and the solver.
"""

from __future__ import annotations

import shlex
from typing import Iterable


_NON_COMMAND_PREFIXES = frozenset(
    {
        ".",
        "cd",
        "do",
        "done",
        "echo",
        "else",
        "fi",
        "for",
        "if",
        "source",
        "then",
    }
)
_RUN_WRAPPERS = frozenset({"runApplication", "runParallel"})
_APPLICATION_VARIABLES = frozenset(
    {"$(getApplication)", "$application", "${application}", "getApplication"}
)


def strip_shell_comment(line: str) -> str:
    """Remove a shell comment for the restricted Allrun grammar we inspect."""
    return line.split("#", 1)[0].strip()


def normalise_shell_lines(script: str) -> list[str]:
    """Join trailing-backslash continuations without executing shell syntax."""
    lines: list[str] = []
    pending = ""
    for raw_line in script.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        lines.append(pending + line)
        pending = ""
    if pending:
        raise ValueError("Allrun ends with an unfinished line continuation.")
    return lines


def shell_tokens(line: str) -> list[str]:
    """Return shell words for one line, or an empty list when it is malformed.

    This is intentionally inspection-only.  It neither expands variables nor
    accepts a line as safe to execute; import mode applies its stricter policy
    separately.
    """
    code = strip_shell_comment(line)
    if not code:
        return []
    try:
        return shlex.split(code, posix=True)
    except ValueError:
        return []


def invoked_command_from_tokens(tokens: Iterable[str]) -> str:
    """Return the OpenFOAM command represented by simple Allrun tokens."""
    words = list(tokens)
    if not words:
        return ""
    first = words[0]
    if first in _RUN_WRAPPERS:
        return words[1] if len(words) > 1 else ""
    if "=" in first or first in _NON_COMMAND_PREFIXES:
        return ""
    return first


def invoked_command(line: str) -> str:
    """Return the command invoked by a simple Allrun line, if any."""
    return invoked_command_from_tokens(shell_tokens(line))


def line_invokes_command(line: str, command: str) -> bool:
    tokens = shell_tokens(line)
    if tokens and tokens[0] in {"mpirun", "mpiexec"}:
        return command in tokens[1:]
    return invoked_command_from_tokens(tokens) == command


def line_invokes_solver(line: str, solver: str) -> bool:
    """Recognise literal or standard ``getApplication`` solver invocations."""
    tokens = shell_tokens(line)
    if not tokens:
        return False
    command = invoked_command_from_tokens(tokens)
    if solver and command == solver:
        return True
    return (
        tokens[0] in _RUN_WRAPPERS
        and len(tokens) > 1
        and tokens[1] in _APPLICATION_VARIABLES
    )


def command_positions(script: str, command: str) -> list[int]:
    """Return source offsets where a command is actually invoked."""
    positions: list[int] = []
    offset = 0
    for raw_line in script.splitlines(keepends=True):
        if line_invokes_command(raw_line, command):
            positions.append(offset + raw_line.find(command))
        offset += len(raw_line)
    return positions


def application_positions(script: str, application: str | None) -> list[int]:
    """Return literal and standard variable-based solver invocation offsets."""
    if not application:
        return []
    positions: list[int] = []
    offset = 0
    for raw_line in script.splitlines(keepends=True):
        if line_invokes_solver(raw_line, application):
            command = invoked_command(raw_line)
            positions.append(offset + raw_line.find(command or "run"))
        offset += len(raw_line)
    return sorted(set(positions))


def script_without_comments(script: str) -> str:
    """Remove comments while preserving one line per source line."""
    return "\n".join(strip_shell_comment(line) for line in script.splitlines())


def is_standard_application_variable(token: str) -> bool:
    return token in _APPLICATION_VARIABLES


def allrun_uses_runfunctions(lines: Iterable[str]) -> bool:
    return any("RunFunctions" in strip_shell_comment(line) for line in lines)


def is_run_wrapper(line: str) -> bool:
    tokens = shell_tokens(line)
    return bool(tokens and tokens[0] in _RUN_WRAPPERS)
