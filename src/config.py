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

    # Post-run interpreter (viz + VLM); optional requirement revision loop back through planner
    enable_post_run_interpreter: bool = True
    interpreter_max_reruns: int = 10
    # If set, overrides model_version for interpreter/viz LLM only (vision-capable model recommended)
    interpreter_model_version: str = ""
    # After interpreter marks case OK: extra viz + narrative (cfd-scientist-style analysis)
    enable_flow_field_analysis: bool = True

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
                print(f"[Config] model_provider={self.model_provider} (env:{provider_key})")
            else:
                print(
                    f"[Config] model_provider={self.model_provider} (default; invalid env:{provider_key}={provider_env!r})"
                )
        else:
            print(f"[Config] model_provider={self.model_provider} (default)")

        version_env = _env_nonempty(version_key)
        if version_env is not None:
            self.model_version = version_env
            print(f"[Config] model_version={self.model_version} (env:{version_key})")
        else:
            print(f"[Config] model_version={self.model_version} (default)")

        interp_en = _env_nonempty("FOAMAGENT_ENABLE_INTERPRETER")
        if interp_en is not None:
            self.enable_post_run_interpreter = interp_en.lower() in ("1", "true", "yes", "on")
            print(f"[Config] enable_post_run_interpreter={self.enable_post_run_interpreter} (env:FOAMAGENT_ENABLE_INTERPRETER)")

        interp_max = _env_nonempty("FOAMAGENT_INTERPRETER_MAX_RERUNS")
        if interp_max is not None:
            try:
                self.interpreter_max_reruns = max(0, int(interp_max))
                print(f"[Config] interpreter_max_reruns={self.interpreter_max_reruns} (env:FOAMAGENT_INTERPRETER_MAX_RERUNS)")
            except ValueError:
                print(f"[Config] interpreter_max_reruns={self.interpreter_max_reruns} (default; invalid env)")

        interp_mv = _env_nonempty("FOAMAGENT_INTERPRETER_MODEL_VERSION")
        if interp_mv is not None:
            self.interpreter_model_version = interp_mv
            print(f"[Config] interpreter_model_version={self.interpreter_model_version!r} (env:FOAMAGENT_INTERPRETER_MODEL_VERSION)")

        fa = _env_nonempty("FOAMAGENT_ENABLE_FLOW_ANALYSIS")
        if fa is not None:
            self.enable_flow_field_analysis = fa.lower() in ("1", "true", "yes", "on")
            print(f"[Config] enable_flow_field_analysis={self.enable_flow_field_analysis} (env:FOAMAGENT_ENABLE_FLOW_ANALYSIS)")
