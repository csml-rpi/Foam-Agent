"""Vision-capable LangChain chat model for interpreter / viz_creator (aligned with Foam-Agent Config)."""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import tracking_aws


def create_interpreter_llm(config: Any, temperature: float = 0.1) -> Any:
    """Return a LangChain BaseChatModel suitable for text + image (vision) calls.

    For ``openai-codex``, Codex's Responses wrapper is not used here; set
    ``OPENAI_API_KEY`` and optionally ``FOAMAGENT_INTERPRETER_MODEL`` (default ``gpt-4o``),
    or set ``config.interpreter_model_version`` to a vision-capable OpenAI model name.
    """
    provider = (getattr(config, "model_provider", "openai") or "openai").lower()
    mv = (getattr(config, "interpreter_model_version", "") or "").strip()
    if not mv:
        mv = getattr(config, "model_version", "gpt-4o")

    if provider == "openai":
        return init_chat_model(mv, model_provider="openai", temperature=temperature)

    if provider == "anthropic":
        return ChatAnthropic(model=mv, temperature=temperature)

    if provider == "bedrock":
        bedrock_runtime = tracking_aws.new_default_client()
        return ChatBedrockConverse(
            client=bedrock_runtime,
            model_id=mv,
            temperature=temperature,
            max_tokens=8192,
        )

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=mv,
            temperature=temperature,
            base_url=base_url,
        )

    if provider in {"openai-codex", "codex", "chatgpt-oauth"}:
        platform_mv = (
            getattr(config, "interpreter_model_version", "") or ""
        ).strip() or os.environ.get(
            "FOAMAGENT_INTERPRETER_MODEL_VERSION",
            os.environ.get("FOAMAGENT_INTERPRETER_MODEL", "gpt-4o"),
        ).strip()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Post-run interpreter needs a vision-capable OpenAI API model. "
                "With model_provider=openai-codex, set OPENAI_API_KEY and optionally "
                "FOAMAGENT_INTERPRETER_MODEL (default gpt-4o), or set "
                "config.interpreter_model_version, or disable with "
                "enable_post_run_interpreter=False."
            )
        return ChatOpenAI(model=platform_mv, temperature=temperature)

    raise ValueError(f"Unsupported model_provider for interpreter: {provider}")
