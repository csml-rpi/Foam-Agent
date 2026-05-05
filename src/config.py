# config.py
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    max_loop: int = 25
    batchsize: int = 10
    searchdocs: int = 10 # max(10, searchdocs)
    run_times: int = 1   # current run number (for directory naming)
    database_path: str = Path(__file__).resolve().parent.parent / "database"
    run_directory: str = Path(__file__).resolve().parent.parent / "runs"
    case_dir: str = ""
    max_time_limit: int = 3600  # Max time limit after which the openfoam run will be terminated, in seconds
    recursion_limit: int = 100  # LangGraph recursion limit
    # Input writer generation mode:
    # - "sequential_dependency": generate files sequentially; use already-generated files as context to enforce consistency.
    # - "parallel_no_context": generate files in parallel without cross-file context (faster, may need more reviewer iterations).
    input_writer_generation_mode: str = "sequential_dependency"
    # Optional: reuse previously generated files by copying from this directory.
    # If set, InputWriter will check <reuse_generated_dir>/<folder>/<file> first.
    # When present, it will copy into the current case_dir and skip LLM generation.
    reuse_generated_dir: str = ""
    dataset_log_path: str = ""
    case_id: str = ""
    # LLM backend:
    # - "openai": OpenAI Platform usage-based (API key)
    # - "openai-codex": ChatGPT/Codex subscription sign-in (Codex auth cache)
    # - "ollama": local models
    # - "bedrock": AWS Bedrock
    # - "anthropic": Anthropic Claude API (requires ANTHROPIC_API_KEY)
    model_provider: str = "openai-codex"  # [openai, openai-codex, ollama, bedrock, anthropic]
    # model_version examples:
    # - OpenAI: "gpt-5-mini"
    # - OpenAI Codex subscription: "gpt-5.3-codex" (or whichever Codex model you have access to)
    # - Ollama: "qwen2.5:32b-instruct"
    # - Bedrock: application inference profile ARN
    # - Anthropic: claude-3-5-sonnet-latest
    model_version: str = "gpt-5.3-codex"
    temperature: float = 1

    # Embedding Configuration
    embedding_provider: str = "huggingface"  # [openai, huggingface, ollama]
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # e.g. "text-embedding-3-small", "text-embedding-3-large", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-8B"

    def __post_init__(self) -> None:
        """Load config overrides from environment variables.

        Priority: env var (if set & non-empty) > default value.
        Always prints what value is used to make runs reproducible.
        """

        def _env_nonempty(key: str) -> str | None:
            v = os.getenv(key)
            if v is None:
                return None
            v = v.strip()
            return v if v else None

        # LLM provider/model overrides
        provider_key = "FOAMAGENT_MODEL_PROVIDER"
        version_key = "FOAMAGENT_MODEL_VERSION"

        provider_env = _env_nonempty(provider_key)
        if provider_env is not None:
            allowed = {"openai", "openai-codex", "ollama", "bedrock", "anthropic"}
            if provider_env in allowed:
                self.model_provider = provider_env
                print(f"<config>model_provider={self.model_provider} (env:{provider_key})</config>")
            else:
                print(
                    f"<config>model_provider={self.model_provider} (default; invalid env:{provider_key}={provider_env!r})</config>"
                )
        else:
            print(f"<config>model_provider={self.model_provider} (default)</config>")

        version_env = _env_nonempty(version_key)
        if version_env is not None:
            self.model_version = version_env
            print(f"<config>model_version={self.model_version} (env:{version_key})</config>")
        else:
            print(f"<config>model_version={self.model_version} (default)</config>")

        # Embedding provider/model overrides
        emb_provider_key = "FOAMAGENT_EMBEDDING_PROVIDER"
        emb_model_key = "FOAMAGENT_EMBEDDING_MODEL"

        emb_provider_env = _env_nonempty(emb_provider_key)
        if emb_provider_env is not None:
            allowed_emb = {"openai", "huggingface", "ollama"}
            if emb_provider_env in allowed_emb:
                self.embedding_provider = emb_provider_env
                print(f"<config>embedding_provider={self.embedding_provider} (env:{emb_provider_key})</config>")
            else:
                print(
                    f"<config>embedding_provider={self.embedding_provider} (default; invalid env:{emb_provider_key}={emb_provider_env!r})</config>"
                )
        else:
            print(f"<config>embedding_provider={self.embedding_provider} (default)</config>")

        emb_model_env = _env_nonempty(emb_model_key)
        if emb_model_env is not None:
            self.embedding_model = emb_model_env
            print(f"<config>embedding_model={self.embedding_model} (env:{emb_model_key})</config>")
        else:
            print(f"<config>embedding_model={self.embedding_model} (default)</config>")
