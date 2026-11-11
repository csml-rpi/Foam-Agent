"""Bounded current case inputs shared by review and repair, independent of execution mode."""

import hashlib
import gzip
import json
import os
import re
import zlib
from pathlib import Path

from utils import FoamfilePydantic, FoamPydantic, is_solver_output_dir, case_file_key
from .output_safety import OUTPUT_MARKER_NAME

_METADATA_NAMES = {".foamagent", OUTPUT_MARKER_NAME}


def dictionary_code(content: str, *, mask_strings: bool = False) -> str:
    """Mask comments/optional strings while preserving native entry offsets."""
    pattern = r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/'
    return re.sub(pattern, lambda match: match[0] if match[0].startswith('"') and not mask_strings
                  else re.sub(r'[^\n]', ' ', match[0]), content, flags=re.S)


def dictionary_entries(content: str, *, recursive: bool = False):
    """Yield native key/value boundaries without interpreting values or includes.

    A primitive value ends at its semicolon, not at a newline or a word that
    resembles another key. Lists and dictionaries within that value are data.
    Matches and offsets refer to the original content.
    """
    code = dictionary_code(content)
    tokens = list(re.finditer(r'"(?:\\.|[^"\\])*"|[{}();\[\]]|[^\s{}();\[\]"]+', code))
    closing = {'{': '}', '(': ')', '[': ']'}

    def balanced_end(index, limit):
        stack = [closing[tokens[index][0]]]
        for end in range(index + 1, limit):
            word = tokens[end][0]
            if word in closing:
                stack.append(closing[word])
            elif word == stack[-1]:
                stack.pop()
                if not stack:
                    return end
        return None

    def entries(index, limit):
        while index < limit:
            key = tokens[index]
            if key[0] == ';':
                index += 1
                continue
            value = index + 1
            if (key[0] == '$' and value < limit and tokens[value][0] == '{'
                    and tokens[value].start() == key.end()):
                variable_end = balanced_end(value, limit)
                if variable_end is not None:
                    key = re.compile(r'[\s\S]+').match(code, key.start(), tokens[variable_end].end())
                    value = variable_end + 1
            if key[0].startswith('$') and (value == limit or tokens[value][0] != '{'):
                # Dictionary substitution consumes its variable word, without
                # requiring a semicolon. A following block uses a variable key.
                yield key, key.end(), key.end()
                index = value
                continue
            start = tokens[value].start() if value < limit else key.end()
            if key[0].startswith('#'):
                # Native includes/inputMode consume one literal argument; a
                # following inline entry remains visible. Other bodies end at their
                # line/semicolon, outside any grouped argument or code body.
                end = value
                if key[0] in {'#include', '#includeIfPresent', '#sinclude', '#includeEtc', '#inputMode'}:
                    end = min(value + 1, limit)
                else:
                    while end < limit:
                        word = tokens[end][0]
                        previous = tokens[end - 1]
                        if '\n' in code[previous.end():tokens[end].start()] or word == '}':
                            break
                        if word in closing:
                            group_end = balanced_end(end, limit)
                            if group_end is None:
                                end = limit
                                break
                            end = group_end + 1
                        else:
                            end += 1
                        if word == ';':
                            break
                yield key, start, tokens[end - 1].end()
                index = end
            elif value < limit and tokens[value][0] == '{':
                end = balanced_end(value, limit)
                yield key, start, tokens[end].end() if end is not None else None
                if end is None:
                    return
                if recursive:
                    yield from entries(value + 1, end)
                index = end + 1
            else:
                end = value
                while end < limit and tokens[end][0] != ';':
                    if tokens[end][0] in closing:
                        group_end = balanced_end(end, limit)
                        if group_end is None:
                            end = limit
                            break
                        end = group_end + 1
                    else:
                        end += 1
                yield key, start, tokens[end].end() if end < limit else None
                index = end + 1

    yield from entries(0, len(tokens))


def dictionary_primitive_entry(content: str, key: str):
    """Locate a direct primitive value without resolving directives or macros."""
    code = dictionary_code(content)
    matches = [(entry, start, end) for entry, start, end in dictionary_entries(content)
               if entry[0] == key]
    if not matches:
        return None
    entry, start, end = matches[-1]
    value = (re.compile(r'\s+([\s\S]*);').match(code, entry.end(), end)
             if end is not None and code[start:start + 1] != '{' else None)
    if value is None:
        raise ValueError(f'{key} has no terminated primitive value')
    return value


def native_foamfile_header(content: str) -> tuple[str, int] | None:
    """Return the actual leading FoamFile body and its end offset, if complete."""
    code = dictionary_code(content)
    masked = dictionary_code(code, mask_strings=True)
    opening = re.match(r'\s*FoamFile\s*\{', code)
    if opening is None:
        return None
    depth = 1
    for index in range(opening.end(), len(masked)):
        if masked[index] == '{':
            depth += 1
        elif masked[index] == '}':
            depth -= 1
            if depth == 0:
                return code[opening.end():index], index + 1
    return None


def format_case_inputs(foamfiles) -> str:
    """Give model prompts explicit case-relative paths instead of object reprs."""
    value = foamfiles.model_dump() if hasattr(foamfiles, "model_dump") else foamfiles
    if isinstance(value, dict) and "list_foamfile" in value:
        entries = [entry.model_dump() if hasattr(entry, "model_dump") else entry for entry in value["list_foamfile"]]
        value = {
            "/".join(part for part in (entry["folder_name"], entry["file_name"]) if part): entry["content"]
            for entry in entries
        }
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def is_declared_input(name, inputs) -> bool:
    return any(case_file_key(name) == case_file_key(path) or
               Path(name) in Path(path).parents for path in inputs)


def is_case_artifact(relative: Path, *, directory: bool = False, protected_inputs=()) -> bool:
    """Identify run outputs and Foam-Agent metadata, shared with input snapshots."""
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in _METADATA_NAMES:
        return True
    if is_declared_input(relative, protected_inputs):
        return False
    if (
        is_solver_output_dir(parts[0]) and (directory or len(parts) > 1)
    ):
        return True
    if directory or len(parts) > 1:
        return False
    # These root files are produced by orchestration. An arbitrary .log/.out
    # file may instead be a user input, particularly inside an asset directory.
    return (
        relative.name in {
            "Allrun.out", "Allrun.err", "workflow.log", "review.log", "similar_case.txt",
            "Allmesh.out", "Allmesh.err", "submit_job.slurm",
        }
    )


def canonical_input_protections(source_dir, protected_inputs=()) -> list[str]:
    """Keep stored inputs that would otherwise be classified as run artifacts."""
    protected = list(protected_inputs)
    if source_dir:
        root = Path(source_dir)
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if (relative.parts[0] not in _METADATA_NAMES and path.is_file()
                    and is_case_artifact(relative)):
                protected.append(relative.as_posix())
    return list(dict.fromkeys(protected))


def _input_paths(case_dir: str, *, protected_inputs=()) -> tuple[Path, list[Path], dict[str, str]]:
    """Discover inputs once for both model context and complete content identity."""
    from .execution_logs import is_native_log
    if not case_dir or not str(case_dir).strip():
        raise ValueError("Case directory must not be empty or whitespace")
    root = Path(case_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {root}")
    paths: list[Path] = []
    omitted: dict[str, str] = {}

    def walk_error(error):
        raise error

    for directory, names, files in os.walk(root, followlinks=False, onerror=walk_error):
        parent = Path(directory)
        kept = []
        for name in sorted(names):
            path = parent / name
            relative = path.relative_to(root)
            if is_case_artifact(relative, directory=True, protected_inputs=protected_inputs):
                continue
            if name.startswith(".") and not is_declared_input(relative, protected_inputs):
                omitted[path.relative_to(root).as_posix()] = "Hidden input not inspected"
            elif path.is_symlink():
                omitted[path.relative_to(root).as_posix()] = "Symbolic link not inspected"
            else:
                kept.append(name)
        names[:] = kept
        for name in files:
            if name.startswith(".") and not is_declared_input((parent / name).relative_to(root), protected_inputs):
                if parent != root or name not in _METADATA_NAMES:
                    omitted[(parent / name).relative_to(root).as_posix()] = "Hidden input not inspected"
                continue
            if is_case_artifact((parent / name).relative_to(root), protected_inputs=protected_inputs):
                continue
            if (not is_declared_input((parent / name).relative_to(root), protected_inputs)
                    and is_native_log(parent / name)):
                continue
            paths.append(parent / name)
    return root, paths, omitted


def case_input_fingerprint(case_dir: str, *, protected_inputs=()) -> str | None:
    """Hash complete input payloads, excluding run artifacts and gzip metadata.

    Unknown or linked inputs cannot prove equality; leave those trials uncached.
    """
    if not case_dir:
        return None
    try:
        root, paths, omitted = _input_paths(case_dir, protected_inputs=protected_inputs)
        if omitted:
            return None
        digests = {}
        for path in paths:
            if path.is_symlink() or not path.is_file():
                return None
            digest = hashlib.sha256()
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests[path.relative_to(root).as_posix()] = digest.hexdigest()
        return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    except (OSError, ValueError, EOFError, zlib.error):
        return None


def read_case_inputs(case_dir: str, *, protected_inputs=(), selected_paths=None) -> tuple[FoamPydantic, dict[str, str]]:
    """Read bounded current inputs, disclosing anything not inspected in full."""
    root, paths, omitted = _input_paths(case_dir, protected_inputs=protected_inputs)
    if selected_paths is not None:
        selected = {case_file_key(path) for path in selected_paths}
        paths = [path for path in paths if case_file_key(path.relative_to(root)) in selected]
        omitted = {name: reason for name, reason in omitted.items() if is_declared_input(name, selected)}
    # Native plain-file precedence applies before context limits, even when the
    # selected plain file cannot be inspected. Its gzip shadow is not a fallback.
    plain = {path.relative_to(root).as_posix() for path in paths if path.suffix != '.gz'}
    paths = [path for path in paths if not (
        path.suffix == '.gz' and case_file_key(path.relative_to(root)) in plain)]

    remaining = 512_000
    inputs = []
    # Root scripts first, then shallow dictionaries before bulk nested data.
    for path in sorted(paths, key=lambda path: (len(path.relative_to(root).parts), path.as_posix())):
        relative = path.relative_to(root)
        name = relative.as_posix()
        if path.is_symlink():
            omitted[name] = "Symbolic link not inspected"
            continue
        if not path.is_file():
            omitted[name] = "Non-regular input not inspected"
            continue
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as handle:
                data = handle.read(min(128_000, remaining) + 1)
        except (EOFError, gzip.BadGzipFile, zlib.error):
            omitted[name] = "Unreadable gzip input not inspected"
            continue
        if len(data) > min(128_000, remaining):
            omitted[name] = "File exceeds the per-file or total review context limit"
            continue
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            omitted[name] = "Non-UTF-8 input not inspected"
            continue
        if "\x00" in content:
            omitted[name] = "Binary input not inspected"
            continue
        remaining -= len(data)
        inputs.append(FoamfilePydantic(
            folder_name="" if relative.parent == Path(".") else relative.parent.as_posix(),
            file_name=relative.name,
            content=content,
        ))
    return FoamPydantic(list_foamfile=inputs), omitted
