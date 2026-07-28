# config.py
import os
from dataclasses import dataclass, field

from foamagent import paths
from pathlib import Path

from foamagent.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Config:
    max_loop: int = 25
    batchsize: int = 10
    searchdocs: int = 10 # max(10, searchdocs)
    run_times: int = 1   # current run number (for directory naming)
    database_path: Path = field(default_factory=paths.database_dir)
    run_directory: Path = field(default_factory=paths.runs_dir)
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
    # LLM backend:
    # - "openai": OpenAI Platform usage-based (API key)
    # - "openai-codex": ChatGPT/Codex subscription sign-in (Codex auth cache)
    # - "ollama": local models
    # - "bedrock": AWS Bedrock
    # - "anthropic": Anthropic Claude API (requires ANTHROPIC_API_KEY)
    # Default to "openai": it is the only provider that reaches an arbitrary endpoint
    # (set openai_base_url for OpenRouter, vLLM, LiteLLM, ...) and it takes credentials
    # the user chose to supply. "openai-codex" is deliberately not the default because it
    # reads another tool's OAuth cache from disk, which no one should opt into silently.
    model_provider: str = "openai"  # [openai, openai-codex, ollama, bedrock, anthropic, deepseek]
    # model_version examples:
    # - OpenAI: "gpt-5-mini"
    # - OpenAI Codex subscription: "gpt-5.3-codex" (or whichever Codex model you have access to)
    # - Ollama: "qwen2.5:32b-instruct"
    # - Bedrock: application inference profile ARN
    # - Anthropic: claude-3-5-sonnet-latest
    model_version: str = "gpt-5-mini"
    temperature: float = 1
    # Optional base URL for OpenAI-compatible endpoints (OpenRouter, vLLM, LiteLLM, ...).
    # Only used when model_provider == "openai". Empty means the official OpenAI endpoint.
    openai_base_url: str = ""
    openfoam_fork: str = "foundation"  # Default to Foundation v10

    # OpenFOAM execution runtime:
    # - "native": source $WM_PROJECT_DIR/etc/bashrc in the current machine
    # - "docker": run inside openfoam_image, mounting the case at the same absolute path
    openfoam_runtime: str = "native"
    openfoam_image: str = "foam-bench:latest"
    openfoam_bashrc: str = "/opt/openfoam10/etc/bashrc"  # bashrc path inside the image

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
            allowed = {"openai", "openai-codex", "ollama", "bedrock", "anthropic", "deepseek"}
            if provider_env in allowed:
                self.model_provider = provider_env
                logger.info(f"<config>model_provider={self.model_provider} (env:{provider_key})</config>")
            else:
                logger.info(
                    f"<config>model_provider={self.model_provider} (default; invalid env:{provider_key}={provider_env!r})</config>"
                )
        else:
            logger.info(f"<config>model_provider={self.model_provider} (default)</config>")

        version_env = _env_nonempty(version_key)
        if version_env is not None:
            self.model_version = version_env
            logger.info(f"<config>model_version={self.model_version} (env:{version_key})</config>")
        else:
            logger.info(f"<config>model_version={self.model_version} (default)</config>")

        # Embedding provider/model overrides
        emb_provider_key = "FOAMAGENT_EMBEDDING_PROVIDER"
        emb_model_key = "FOAMAGENT_EMBEDDING_MODEL"

        emb_provider_env = _env_nonempty(emb_provider_key)
        if emb_provider_env is not None:
            allowed_emb = {"openai", "huggingface", "ollama"}
            if emb_provider_env in allowed_emb:
                self.embedding_provider = emb_provider_env
                logger.info(f"<config>embedding_provider={self.embedding_provider} (env:{emb_provider_key})</config>")
            else:
                logger.info(
                    f"<config>embedding_provider={self.embedding_provider} (default; invalid env:{emb_provider_key}={emb_provider_env!r})</config>"
                )
        else:
            logger.info(f"<config>embedding_provider={self.embedding_provider} (default)</config>")

        emb_model_env = _env_nonempty(emb_model_key)
        if emb_model_env is not None:
            self.embedding_model = emb_model_env
            logger.info(f"<config>embedding_model={self.embedding_model} (env:{emb_model_key})</config>")
        else:
            logger.info(f"<config>embedding_model={self.embedding_model} (default)</config>")

        # Integer overrides (loop / time limits)
        for env_key, attr in (
            ("FOAMAGENT_MAX_LOOP", "max_loop"),
            ("FOAMAGENT_MAX_TIME_LIMIT", "max_time_limit"),
        ):
            raw = _env_nonempty(env_key)
            if raw is None:
                continue
            try:
                setattr(self, attr, int(raw))
                logger.info(f"<config>{attr}={getattr(self, attr)} (env:{env_key})</config>")
            except ValueError:
                logger.info(f"<config>{attr}={getattr(self, attr)} (default; invalid env:{env_key}={raw!r})</config>")

        # OpenAI-compatible base URL override
        base_url_key = "FOAMAGENT_OPENAI_BASE_URL"
        base_url_env = _env_nonempty(base_url_key)
        if base_url_env is not None:
            self.openai_base_url = base_url_env
            logger.info(f"<config>openai_base_url={self.openai_base_url} (env:{base_url_key})</config>")

        # OpenFOAM execution runtime overrides
        runtime_key = "FOAMAGENT_OPENFOAM_RUNTIME"
        runtime_env = _env_nonempty(runtime_key)
        if runtime_env is not None:
            allowed_runtimes = {"native", "docker"}
            if runtime_env.lower() in allowed_runtimes:
                self.openfoam_runtime = runtime_env.lower()
                logger.info(f"<config>openfoam_runtime={self.openfoam_runtime} (env:{runtime_key})</config>")
            else:
                logger.info(
                    f"<config>openfoam_runtime={self.openfoam_runtime} (default; invalid env:{runtime_key}={runtime_env!r})</config>"
                )
        else:
            logger.info(f"<config>openfoam_runtime={self.openfoam_runtime} (default)</config>")

        image_env = _env_nonempty("FOAMAGENT_OPENFOAM_IMAGE")
        if image_env is not None:
            self.openfoam_image = image_env
            logger.info(f"<config>openfoam_image={self.openfoam_image} (env:FOAMAGENT_OPENFOAM_IMAGE)</config>")

        bashrc_env = _env_nonempty("FOAMAGENT_OPENFOAM_BASHRC")
        if bashrc_env is not None:
            self.openfoam_bashrc = bashrc_env
            logger.info(f"<config>openfoam_bashrc={self.openfoam_bashrc} (env:FOAMAGENT_OPENFOAM_BASHRC)</config>")

        # OpenFOAM Fork Override
        fork_key = "FOAMAGENT_OPENFOAM_FORK"
        fork_env = _env_nonempty(fork_key)
        if fork_env is not None:
            allowed_forks = {"foundation", "esi"}
            if fork_env.lower() in allowed_forks:
                self.openfoam_fork = fork_env.lower()
                logger.info(f"<config>openfoam_fork={self.openfoam_fork} (env:{fork_key})</config>")
            else:
                self.openfoam_fork = "foundation"  # Safe fallback assignment
                logger.info(f"<config>openfoam_fork={self.openfoam_fork} (default; invalid env:{fork_key}={fork_env!r})</config>")
        else:
            logger.info(f"<config>openfoam_fork={self.openfoam_fork} (default)</config>")
