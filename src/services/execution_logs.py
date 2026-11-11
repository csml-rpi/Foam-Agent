"""Current-attempt log evidence, bounded diagnostics, and historical log retention."""

import os
import gzip
import zlib
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from collections import deque

from utils import is_solver_output_dir, reject_symbolic_links
from .case_inputs import is_declared_input, native_foamfile_header, dictionary_primitive_entry, dictionary_code
from .review_history import ReviewHistory
from logger import active_log_paths

_STREAMS = {"Allrun.out", "Allrun.err", "Allmesh.out", "Allmesh.err"}
_ERROR = re.compile(r"FOAM FATAL(?: IO)? ERROR|\bERROR\s*:|Floating point exception\s*\(core dumped\)|Segmentation fault|already run.*(?:remove|re-run)|Failed\s+[1-9]\d*\s+mesh checks", re.I)
_EXEC = re.compile(r"^Exec\s*:\s*(\S+)")


def declared_startup_inputs(operations):
    return [required["path"] for operation in operations for required in operation["required_inputs"]]


def check_execution_outputs(outputs, protected_inputs):
    """Fixed executor outputs cannot replace explicitly declared startup inputs."""
    return [
        {"file": required, "error_content":
         f"Declared startup input {required} conflicts with execution output {output}. "
         "Reconcile the input path and its planned operations before execution."}
        for output in outputs for required in protected_inputs
        if is_declared_input(output, [required])
    ]


def diagnostic_outputs(case_dir):
    """Only diagnostics actually owned inside this case can conflict with its inputs."""
    root = Path(case_dir).expanduser().resolve()
    return [path.relative_to(root).as_posix() for output in active_log_paths()
            if (path := output.resolve()).is_relative_to(root)]


def check_diagnostic_outputs(case_dir, protected_inputs):
    return check_execution_outputs(diagnostic_outputs(case_dir), protected_inputs)


def execution_log_candidates(case):
    root = Path(case)
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [n for n in names if not n.startswith('.') and not is_solver_output_dir(n)
                    and not (Path(directory) / n).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                yield path


def _signature(path):
    value = path.stat()
    return value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def snapshot_execution_logs(case):
    root = Path(case)
    return {p.relative_to(root).as_posix(): _signature(p) for p in execution_log_candidates(root)}


def is_native_log(path):
    """Native headers establish log ownership; an arbitrary name does not."""
    if path.name in {"Allrun", "Allmesh", "generate_mesh.py"} or path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as stream:
        head = stream.read(64_000).decode("utf-8", errors="replace")
        if head.startswith('#!'):
            return False
        stream.seek(max(0, path.stat().st_size - 4096))
        tail = stream.read(4096).decode("utf-8", errors="replace")
    def log_framing(text):
        # Source programs can quote complete native examples in multiline
        # strings. Their contents do not establish historical log ownership.
        text = re.sub(r"('''|\"\"\").*?(?:\1|\Z)",
                      lambda match: re.sub(r'[^\n]', ' ', match[0]), text, flags=re.S)
        code = dictionary_code(text, mask_strings=True)
        code = re.sub(r"(?<![\w'\\])'[^']*'", lambda match: re.sub(r'[^\n]', ' ', match[0]), code)
        # Only literal here-document boundaries are needed, not shell execution.
        pending, framed = [], []
        for line, visible in zip(text.splitlines(keepends=True), code.splitlines(keepends=True)):
            if pending:
                delimiter, strip_tabs = pending[0]
                if (line.lstrip('\t') if strip_tabs else line).rstrip('\r\n') == delimiter:
                    pending.pop(0)
                framed.append(re.sub(r'[^\n]', ' ', line))
                continue
            visible = re.sub(r'(?<!\S)#.*', '', visible)
            for opener in re.finditer(r'(?<!<)<<(-?)(?!<)', visible):
                word = re.match(
                    r'''[ \t]*((?:'[^'\n]*'|"(?:\\.|[^"\\\n])*"|\\[^\n]|[^\s'"\\;|&<>()])+)(?![^\s;|&<>()])''',
                    line[opener.end():])
                if word is None:
                    return ''  # Unresolved source syntax cannot establish log ownership.
                pending.append((shlex.split(word[1])[0], bool(opener[1])))
            framed.append(visible)
        return ''.join(framed)
    framed_head, framed_tail = log_framing(head), log_framing(tail)
    if not (re.search(r"^Exec\s*:\s*\S+", framed_head, re.M) and (
            re.search(r"^Build\s*:|OpenFOAM:|^End\s*$|FOAM FATAL", framed_head + "\n" + framed_tail, re.M))):
        return False
    input_head = head
    if path.suffix == '.gz':
        try:
            with gzip.open(path, 'rb') as stream:
                input_head = stream.read(64_000).decode('utf-8', errors='replace')
        except (gzip.BadGzipFile, EOFError, zlib.error):
            return False
    if native_foamfile_header(input_head) is not None:
        return False
    return True


def archive_native_logs(case, *, protected_inputs=()):
    root = Path(case)
    logs = [p for p in execution_log_candidates(root) if p.name not in {"workflow.log", "review.log"}
            and not is_declared_input(p.relative_to(root), protected_inputs) and is_native_log(p)]
    if not logs:
        return
    parent = root / ".foamagent" / "log-history"
    history = ReviewHistory(str(root))
    reject_symbolic_links(parent)
    reject_symbolic_links(history.path)
    parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="attempt-", dir=parent))
    archived = {}
    for path in logs:
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), target)
        archived[relative.as_posix()] = target.relative_to(root).as_posix()
    history.append("native_logs_archived", files=archived)


def current_execution_logs(case, before, *, protected_inputs=(), include_unverified=False):
    """Select current evidence; unverified log names are for failure context only."""
    root = Path(case)
    for path in execution_log_candidates(root):
        name = path.relative_to(root).as_posix()
        if is_declared_input(name, protected_inputs) or before.get(name) == _signature(path):
            continue
        if (name in _STREAMS or is_native_log(path)
                or include_unverified and path.name.startswith('log.')):
            yield path


def native_job_completion(job_info_dir, solver, host, pid):
    """Match a process in this attempt to its native termination record."""
    if job_info_dir is None:
        return None
    name = f"{host}.{pid}"
    if Path(name).name != name:
        return None
    root = Path(job_info_dir)
    path = root / 'finishedJobs' / name
    reject_symbolic_links(path)
    if not path.is_file():
        return False if (root / 'runningJobs' / name).is_file() else None
    with path.open(encoding='utf-8') as stream:
        record = stream.read(64_000)
    code = dictionary_primitive_entry(record, 'code')
    termination = dictionary_primitive_entry(record, 'termination')
    return (code is not None and Path(code[1].strip().strip('"')).name == solver
            and termination is not None and termination[1].strip() == 'normal')


def scan_execution_log(path, *, solver=None, job_info_dir=None):
    """Read fixed-size line fragments; retain at most eight 4 KB errors and a tail."""
    runs, errors, tail = [], [], deque()
    tail_size = 0
    depth, solver_depth = 0, None
    solver_header, host = False, None
    pending = ""
    at_start = True
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in iter(lambda: stream.readline(8192), ""):
            tail.append(line)
            tail_size += len(line)
            while tail_size - len(tail[0]) >= 16_000:
                tail_size -= len(tail.popleft())
            header = _EXEC.match(line) if at_start else None
            if header:
                depth += 1
                solver_header, host = bool(solver and Path(header[1]).name == solver), None
                if solver_depth is None and solver and Path(header[1]).name == solver:
                    solver_depth = depth
            if at_start and solver_header:
                host_entry = re.fullmatch(r'Host\s*:\s*"?([^"\n]+?)"?\s*\n?', line)
                if host_entry:
                    host = host_entry[1]
                pid_entry = re.fullmatch(r'PID\s*:\s*([0-9]+)\s*\n?', line)
                if pid_entry and host is not None:
                    completion = native_job_completion(job_info_dir, solver, host, pid_entry[1])
                    if completion is not None:
                        runs.append(completion)
                        runs[:] = [all(runs)]
                        if solver_depth == depth:
                            solver_depth = None
                    solver_header = False
            if at_start and line.strip() == "End" and depth:
                # Function objects can run native utilities within the solver's
                # output. Their footer closes the child, not the pending solver.
                if solver_depth == depth:
                    runs.append(True)
                    runs[:] = [all(runs)]
                    solver_depth = None
                depth -= 1
            if _ERROR.search(line) and not pending and len(errors) < 8:
                pending = line[:4096]
            elif pending:
                pending += line[:4096 - len(pending)]
            if len(pending) >= 4096:
                errors.append(pending)
                pending = ""
            at_start = line.endswith("\n")
    if pending:
        errors.append(pending)
    if solver_depth is not None:
        runs.append(False)
        runs[:] = [all(runs)]
    return {"runs": runs, "errors": errors, "tail": "".join(tail)[-16_000:]}
