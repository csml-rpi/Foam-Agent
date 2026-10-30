"""Explicit OpenFOAM target selection and ESI v2006 compatibility boundaries.

The historical ``openfoam_fork`` flag deliberately remains untouched: its
``foundation`` and generic ``esi`` values keep their established behaviour.
``esi-v2006`` is opt-in through ``openfoam_target`` and never falls back to
the Foundation tutorial corpus or translation middleware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ESI_V2006 = "esi-v2006"
FOUNDATION_V10 = "foundation-v10"
_EXPLICIT_TARGETS = frozenset({FOUNDATION_V10, ESI_V2006})
_REQUIRED_RAW_CORPUS_FILES = frozenset(
    {
        "openfoam_case_stats.json",
        "openfoam_commands.txt",
        "openfoam_command_help.txt",
        "openfoam_allrun_scripts.txt",
        "openfoam_tutorials_structure.txt",
        "openfoam_tutorials_details.txt",
    }
)
_REQUIRED_FAISS_INDICES = frozenset(
    {
        "openfoam_command_help",
        "openfoam_allrun_scripts",
        "openfoam_tutorials_structure",
        "openfoam_tutorials_details",
    }
)


def normalise_openfoam_target(value: str | None) -> str:
    """Return a validated explicit target, or an empty legacy-routing value."""
    target = (value or "").strip().lower()
    if not target:
        return ""
    if target not in _EXPLICIT_TARGETS:
        supported = ", ".join(sorted(_EXPLICIT_TARGETS))
        raise ValueError(
            f"Unsupported FOAMAGENT_OPENFOAM_TARGET={value!r}. "
            f"Supported explicit targets: {supported}."
        )
    return target


def configured_openfoam_target(config: Any) -> str:
    """Read the opt-in target without changing legacy configurations."""
    return normalise_openfoam_target(getattr(config, "openfoam_target", ""))


def runtime_openfoam_target(config: Any) -> str:
    """Guard the default v10 runtime while retaining generic ESI compatibility."""
    target = configured_openfoam_target(config)
    if target:
        return target
    fork = (getattr(config, "openfoam_fork", "foundation") or "foundation").strip().lower()
    return "" if fork == "esi" else FOUNDATION_V10


def generation_convention(config: Any) -> str:
    """Return the convention supplied to the writer and reviewer.

    Legacy values are intentionally returned verbatim so the pre-existing
    Foundation and generic ESI paths remain source- and behaviour-compatible.
    """
    target = configured_openfoam_target(config)
    if target == ESI_V2006:
        return target
    if target == FOUNDATION_V10:
        return "foundation"
    return (getattr(config, "openfoam_fork", "foundation") or "foundation").strip().lower()


def uses_legacy_esi_translation(config: Any) -> bool:
    """Whether this configuration deliberately selects the old ESI middleware."""
    return (
        not configured_openfoam_target(config)
        and (getattr(config, "openfoam_fork", "foundation") or "").strip().lower()
        == "esi"
    )


def database_path_for_config(config: Any) -> Path:
    """Select the target subdirectory under the configured database root."""
    configured_path = getattr(config, "database_path", None)
    if not configured_path:
        configured_path = Path(__file__).resolve().parent.parent / "database"
    base_path = Path(configured_path).expanduser().resolve()
    target = configured_openfoam_target(config) or FOUNDATION_V10
    return base_path / target


def require_target_corpus(config: Any) -> None:
    """Check the selected target's manifest, raw files, and embedding indices."""
    target = configured_openfoam_target(config) or FOUNDATION_V10
    expected_version = "2006" if target == ESI_V2006 else "10"
    database_path = database_path_for_config(config)
    manifest_path = database_path / "raw" / "foamagent_target.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{target} requires a target-scoped corpus manifest at "
            f"{manifest_path}. Build it with: python init_database.py "
            f"--openfoam_target {target} --openfoam_path <OpenFOAM-root> "
            "--embedding_provider <provider> --embedding_model <model>."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {target} corpus manifest {manifest_path}: {exc}") from exc
    if manifest.get("openfoam_target") != target:
        raise ValueError(
            f"Corpus manifest {manifest_path} declares "
            f"{manifest.get('openfoam_target')!r}, not {target!r}."
        )
    version = str(manifest.get("wm_project_version", "")).lstrip("vV")
    if version != expected_version:
        raise ValueError(
            f"Corpus manifest {manifest_path} is not sourced from {target} "
            f"(wm_project_version={manifest.get('wm_project_version')!r})."
        )

    raw_dir = database_path / "raw"
    missing_raw = sorted(
        filename for filename in _REQUIRED_RAW_CORPUS_FILES if not (raw_dir / filename).is_file()
    )
    model = str(
        getattr(config, "embedding_model", "Qwen/Qwen3-Embedding-0.6B")
        or "Qwen/Qwen3-Embedding-0.6B"
    )
    model_dir = model.replace("/", "_").replace(":", "_")
    index_root = database_path / "faiss" / model_dir
    missing_indices = sorted(
        index
        for index in _REQUIRED_FAISS_INDICES
        if not (index_root / index / "index.faiss").is_file()
        or not (index_root / index / "index.pkl").is_file()
    )
    if missing_raw or missing_indices:
        details = []
        if missing_raw:
            details.append(f"raw files: {', '.join(missing_raw)}")
        if missing_indices:
            details.append(
                f"FAISS indices for {model!r}: {', '.join(missing_indices)}"
            )
        raise FileNotFoundError(
            f"{target} corpus is incomplete (" + "; ".join(details) + "). "
            f"Build it with: python init_database.py --openfoam_target {target} "
            "--openfoam_path <OpenFOAM-root> and select the matching "
            "--embedding_provider and --embedding_model."
        )
