# Namespace package for service-layer wrappers
from utils import LLMService
from config import Config

# Global LLM service instance for services
global_llm_service = LLMService(Config())