"""Service entry points do not construct models during import."""


def get_llm_service(llm_service=None, config=None):
    if llm_service is not None:
        return llm_service
    from utils import LLMService
    from config import Config
    return LLMService(config if config is not None else Config())
