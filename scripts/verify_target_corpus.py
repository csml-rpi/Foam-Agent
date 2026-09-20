#!/usr/bin/env python3
"""Validate that a native target corpus is complete and hydrated from Git LFS.

This intentionally has no Foam-Agent runtime dependencies so it can run in a
Docker build before Conda is created and in CI immediately after ``git lfs
pull``. It validates the data contract shared by the Foundation v10 and ESI
v2006 native targets rather than merely checking that placeholder files exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
RAW_FILES = (
    "openfoam_case_stats.json",
    "openfoam_commands.txt",
    "openfoam_command_help.txt",
    "openfoam_allrun_scripts.txt",
    "openfoam_tutorials_structure.txt",
    "openfoam_tutorials_details.txt",
)
INDICES = (
    "openfoam_command_help",
    "openfoam_allrun_scripts",
    "openfoam_tutorials_structure",
    "openfoam_tutorials_details",
)
TARGET_RUNTIME_VERSIONS = {
    "foundation-v10": "10",
    "esi-v2006": "2006",
}


def _model_dir(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return False


def _validate_file(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing: {path}")
    elif _is_lfs_pointer(path):
        errors.append(f"Git LFS object was not hydrated: {path}")


def corpus_root(target: str, explicit_path: str) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return ROOT / "database" / target


def validate(target: str, root: Path, models: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    raw = root / "raw"
    manifest_path = raw / "foamagent_target.json"
    _validate_file(manifest_path, errors)
    if manifest_path.is_file() and not _is_lfs_pointer(manifest_path):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid manifest {manifest_path}: {exc}")
        else:
            if manifest.get("openfoam_target") != target:
                errors.append(
                    f"{target} corpus manifest does not declare openfoam_target={target}"
                )
            expected_version = TARGET_RUNTIME_VERSIONS[target]
            if str(manifest.get("wm_project_version", "")).lstrip("vV") != expected_version:
                errors.append(
                    f"{target} corpus manifest does not declare "
                    f"WM_PROJECT_VERSION={expected_version}"
                )
            if not manifest.get("source"):
                errors.append(f"{target} corpus manifest is missing source metadata")
    for filename in RAW_FILES:
        _validate_file(raw / filename, errors)
    for model in models:
        for index in INDICES:
            index_dir = root / "faiss" / _model_dir(model) / index
            _validate_file(index_dir / "index.faiss", errors)
            _validate_file(index_dir / "index.pkl", errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("foundation-v10", "esi-v2006"), required=True)
    parser.add_argument("--database-path", default="")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    root = corpus_root(args.target, args.database_path)
    errors = validate(args.target, root, (args.embedding_model,))
    if errors:
        print("Native target corpus validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 2
    print(f"Validated {args.target} corpus at {root} for {args.embedding_model!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
