# Namespace package for service-layer wrappers
from utils import LLMService
from config import Config


class _LazyLLMService:
    """Create the configured LLM only when a workflow actually invokes it.

    Service modules are also imported by deterministic preflight checks and
    unit tests.  Constructing an OAuth-backed client at import time made those
    local operations depend on credentials that they never use.
    """

    def __init__(self) -> None:
        self._instance = None

    def _get_instance(self) -> LLMService:
        if self._instance is None:
            self._instance = LLMService(Config())
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._get_instance(), name)


# Legacy callers retain the ``global_llm_service.invoke(...)`` interface,
# while import-only code remains free of network and credential side effects.
global_llm_service = _LazyLLMService()
