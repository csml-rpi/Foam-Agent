"""Regression tests for configuration-scoped FAISS loading and caching."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402
import utils  # noqa: E402


_INDEX_NAMES = (
    "openfoam_allrun_scripts",
    "openfoam_tutorials_structure",
    "openfoam_tutorials_details",
    "openfoam_command_help",
)


def _config(
    database_path: Path,
    *,
    embedding_provider: str = "huggingface",
    embedding_model: str = "test/model-a",
) -> Config:
    return Config(
        database_path=database_path,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )


def test_load_faiss_dbs_uses_the_configured_database_root(monkeypatch, tmp_path: Path) -> None:
    """A programmatic Config must not silently read the repository database."""
    monkeypatch.delenv("FOAMAGENT_DATABASE_PATH", raising=False)
    monkeypatch.delenv("FOAMAGENT_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("FOAMAGENT_EMBEDDING_MODEL", raising=False)

    database_root = tmp_path / "custom-database"
    config = _config(database_root)
    model_dir = database_root / "faiss" / "test_model-a"
    for index_name in _INDEX_NAMES:
        (model_dir / index_name).mkdir(parents=True)

    embedding = object()
    loaded_paths: list[tuple[Path, object, bool]] = []

    class FakeFAISS:
        @staticmethod
        def load_local(path: str, embedding_model: object, *, allow_dangerous_deserialization: bool):
            loaded_paths.append(
                (Path(path), embedding_model, allow_dangerous_deserialization)
            )
            return {"loaded_from": path}

    monkeypatch.setattr(utils, "get_embedding_model", lambda cfg: embedding)
    monkeypatch.setattr(utils, "FAISS", FakeFAISS)

    databases = utils.load_faiss_dbs(config)

    assert set(databases) == set(_INDEX_NAMES)
    assert {path for path, _, _ in loaded_paths} == {
        model_dir / index_name for index_name in _INDEX_NAMES
    }
    assert all(loaded_embedding is embedding for _, loaded_embedding, _ in loaded_paths)
    assert all(allow for _, _, allow in loaded_paths)


def test_retrieve_faiss_cache_isolated_by_database_provider_and_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """One process must not reuse an incompatible vector store for another Config."""
    monkeypatch.delenv("FOAMAGENT_DATABASE_PATH", raising=False)
    monkeypatch.delenv("FOAMAGENT_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("FOAMAGENT_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(utils, "FAISS_DB_CACHE", {})

    class FakeDocument:
        def __init__(self, label: str) -> None:
            self.page_content = label
            self.metadata = {
                "command": label,
                "full_content": f"help for {label}",
            }

    class FakeVectorStore:
        def __init__(self, label: str) -> None:
            self.label = label

        def similarity_search_with_score(self, _query: str, *, k: int):
            assert k == 1
            return [(FakeDocument(self.label), 0.25)]

    loaded_keys: list[utils.FAISSCacheKey] = []

    def fake_load(config: Config):
        key = utils._faiss_cache_key(config)
        loaded_keys.append(key)
        return {"openfoam_command_help": FakeVectorStore("|".join(key))}

    monkeypatch.setattr(utils, "load_faiss_dbs", fake_load)

    base = _config(tmp_path / "database-a")
    provider_changed = _config(
        tmp_path / "database-a",
        embedding_provider="ollama",
        embedding_model="test/model-a",
    )
    model_changed = _config(
        tmp_path / "database-a",
        embedding_model="test/model-b",
    )
    path_changed = _config(tmp_path / "database-b")
    configs = (base, provider_changed, model_changed, path_changed)
    expected_keys = [utils._faiss_cache_key(config) for config in configs]

    results = [
        utils.retrieve_faiss("openfoam_command_help", "blockMesh", config=config)
        for config in configs
    ]
    repeated_result = utils.retrieve_faiss(
        "openfoam_command_help", "blockMesh", config=base
    )

    assert len(set(expected_keys)) == 4
    assert loaded_keys == expected_keys
    assert [result[0]["command"] for result in results] == [
        "|".join(key) for key in expected_keys
    ]
    assert repeated_result[0]["command"] == "|".join(expected_keys[0])
    assert len(utils.FAISS_DB_CACHE) == 4
