#!/usr/bin/env python3
"""Check installation locally; model/embedding probes are explicit opt-ins."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import Config
from openfoam_target import configured_openfoam_target

# This sibling verifier uses only the Python standard library.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_target_corpus import validate


def check_model_files(root: Path) -> list[str]:
    """Check local config and weights without importing Torch or downloading."""
    errors = []
    config = root / "config.json"
    try:
        if not isinstance(json.loads(config.read_text()), dict):
            errors.append(f"invalid model config: {config}")
    except (OSError, ValueError):
        errors.append(f"missing or invalid model config: {config}")
    weights = [p for name in ("model.safetensors", "pytorch_model.bin") if (p := root / name).is_file()]
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / name
        if not index.is_file():
            continue
        try:
            names = set(json.loads(index.read_text())["weight_map"].values())
            for item in names:
                path = (root / item).resolve()
                path.relative_to(root.resolve())
                weights.append(path)
        except (OSError, ValueError, KeyError, TypeError):
            errors.append(f"invalid weight index: {index}")
    if not weights:
        errors.append(f"no model weights under {root}")
    for path in weights:
        try:
            with path.open("rb") as stream:
                head = stream.read(128)
            if not head or head.startswith(b"version https://git-lfs.github.com/spec/"):
                errors.append(f"empty or Git LFS pointer weight: {path}")
        except OSError:
            errors.append(f"missing weight: {path}")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-embedding", action="store_true", help="Probe embeddings and index dimensions (may download weights or call an embedding API)")
    parser.add_argument("--check-llm", action="store_true", help="Send one small structured request to the selected LLM (uses credentials/quota)")
    args = parser.parse_args(argv)
    failed = False

    def report(ok: bool, message: str):
        nonlocal failed
        print(f"{'OK' if ok else 'FAIL'}  {message}")
        failed |= not ok

    try:
        with contextlib.redirect_stdout(sys.stderr):
            cfg = Config()
    except ValueError as exc:
        report(False, str(exc))
        return 1

    target = configured_openfoam_target(cfg) or "foundation-v10"
    report(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (release environment: 3.12)")
    for package in ("langchain", "langchain-openai", "langchain-anthropic", "langchain-community", "langchain-huggingface", "langgraph", "fastmcp", "pydantic", "faiss-cpu", "sentence-transformers", "pyvista", "requests", "tiktoken", "Pillow"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report(False, f"missing {package}; install with python -m pip install -c constraints.txt -e '.[mesh]'")
        else:
            report(True, f"{package} {version}")

    expected = {"foundation-v10": {"10"}, "esi-v2006": {"2006", "v2006"}}[target]
    report(bool(os.getenv("WM_PROJECT_DIR")) and bool(shutil.which("blockMesh")) and os.getenv("WM_PROJECT_VERSION") in expected,
           f"OpenFOAM {target}: WM_PROJECT_VERSION={os.getenv('WM_PROJECT_VERSION', 'unset')}; source the matching etc/bashrc")
    corpus = Path(cfg.database_path) / target
    corpus_errors = validate(target, corpus, (cfg.embedding_model,))
    report(not corpus_errors, "tutorial corpus and FAISS files" + ("; run git lfs pull and select matching model/target" if corpus_errors else ""))
    for error in corpus_errors:
        print(f"      {error}")

    if cfg.embedding_provider == "huggingface":
        model_path = cfg.embedding_model_path
        if not model_path:
            try:
                from huggingface_hub import try_to_load_from_cache
                cached = try_to_load_from_cache(cfg.embedding_model, "config.json")
                if isinstance(cached, str):
                    model_path = str(Path(cached).parent)
            except ImportError:
                pass
        if model_path:
            errors = check_model_files(Path(model_path))
            report(not errors, "local embedding weights (file checks; use --check-embedding to verify loading)")
            for error in errors:
                print(f"      {error}")
        elif args.check_embedding:
            print("INFO  embedding weights are not cached; the requested probe may download them")
        else:
            report(False, "embedding weights not cached; run with --check-embedding to prepare them, or set FOAMAGENT_EMBEDDING_MODEL_PATH")

    key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}.get(cfg.model_provider)
    if key:
        report(bool(os.getenv(key)), f"{cfg.model_provider} credentials: {key} {'set' if os.getenv(key) else 'missing'} (not sent)")
    elif cfg.model_provider == "openai-codex":
        caches = [Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser() / "auth.json", Path.home() / ".codex/auth.json", Path.home() / ".clawdbot/agents/main/agent/auth-profiles.json"]
        report(any(p.is_file() for p in caches), "Codex OAuth cache exists (validity not tested)")
    else:
        print(f"INFO  {cfg.model_provider} authentication is checked only with --check-llm")

    if args.check_embedding and not corpus_errors:
        try:
            from utils import get_embedding_model
            import faiss
            vector = get_embedding_model(cfg).embed_query("OpenFOAM installation check")
            model_dir = cfg.embedding_model.replace("/", "_").replace(":", "_")
            for index in (corpus / "faiss" / model_dir).glob("*/index.faiss"):
                dimension = faiss.read_index(str(index)).d
                if dimension != len(vector):
                    raise ValueError(f"dimension mismatch: {index}: index={dimension}, model={len(vector)}")
            report(True, "embedding model loads and matches index dimensions; use the same model revision as the indices")
        except Exception as exc:
            report(False, f"embedding probe failed ({type(exc).__name__}); check weights and model identity")

    if args.check_llm:
        try:
            from typing import Literal
            from pydantic import BaseModel
            from utils import LLMService
            class Probe(BaseModel):
                status: Literal["ok"]
            LLMService(cfg).invoke('Return {"status":"ok"}.', pydantic_obj=Probe, max_retries=0)
            report(True, f"LLM authentication, model and structured output ({cfg.structured_output_method})")
        except Exception as exc:
            report(False, f"LLM probe failed ({type(exc).__name__}); verify provider/model/credentials and FOAMAGENT_STRUCTURED_OUTPUT_METHOD")

    print("Installation checks passed." if not failed else "Installation needs attention; see FAIL lines.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
