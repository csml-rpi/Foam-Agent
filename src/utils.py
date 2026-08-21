# utils.py
import re
import subprocess
import os
import signal
from typing import Optional, Any, Type, TypedDict, List, Dict
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import tiktoken
from langchain_aws import ChatBedrockConverse
from langchain_anthropic import ChatAnthropic
from pathlib import Path
import tracking_aws
import requests
import time
import random
import threading
from botocore.exceptions import ClientError
import shutil
from config import Config
from langchain_ollama import ChatOllama
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None


FAISSCacheKey = tuple[str, str, str]

# Cache FAISS indices by their full retrieval configuration.  A process can
# serve more than one workflow (notably through MCP), so a single global index
# set would otherwise mix embeddings or tutorial corpora from unrelated
# Config instances.
FAISS_DB_CACHE: Dict[FAISSCacheKey, Dict[str, Any]] = {}
_FAISS_DB_CACHE_LOCK = threading.RLock()


def _configured_database_path(config: Config) -> Path:
    """Resolve the configured database root without silently using the repo one."""
    configured_path = getattr(config, "database_path", None)
    if not configured_path:
        configured_path = Path(__file__).resolve().parent.parent / "database"
    return Path(configured_path).expanduser().resolve()


def _faiss_cache_key(config: Config) -> FAISSCacheKey:
    """Return the cache identity for one embedding/index configuration."""
    provider = str(getattr(config, "embedding_provider", "openai") or "openai").casefold()
    model = str(getattr(config, "embedding_model", "") or "")
    return (str(_configured_database_path(config)), provider, model)

def get_embedding_model(config: Optional[Config] = None):
    """Return an embedding model based on the provided config.

    Note: historically this module accessed Config.* class attributes at import time.
    That works for defaults but breaks when callers pass a customized Config instance.
    """
    cfg = config or Config()

    provider = (cfg.embedding_provider or "openai").lower()
    model = cfg.embedding_model

    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    if provider == "huggingface":
        if HuggingFaceEmbeddings is None:
            raise ImportError(
                "langchain-huggingface is not installed. Please install it to use HuggingFace embeddings."
            )
        return HuggingFaceEmbeddings(model_name=model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)

    raise ValueError(f"Unsupported embedding provider: {provider}")


def load_faiss_dbs(config: Optional[Config] = None):
    cfg = config or Config()
    embedding_model = get_embedding_model(cfg)

    base_dir = _configured_database_path(cfg) / "faiss"

    # Sanitize model name for directory usage
    model_dir_name = (cfg.embedding_model or "").replace("/", "_").replace(":", "_")
    db_path = base_dir / model_dir_name

    print(f"Loading FAISS indices from: {db_path} with model: {cfg.embedding_model}")

    dbs = {}
    indices = [
        "openfoam_allrun_scripts",
        "openfoam_tutorials_structure",
        "openfoam_tutorials_details",
        "openfoam_command_help",
    ]

    for index in indices:
        index_path = db_path / index
        if index_path.exists():
            try:
                dbs[index] = FAISS.load_local(
                    str(index_path), embedding_model, allow_dangerous_deserialization=True
                )
            except (OSError, RuntimeError, ValueError) as e:
                print(f"Failed to load index {index}: {e}")
        else:
            print(f"Warning: Index path does not exist: {index_path}")

    return dbs


# The embedding model can be large and may require a network/cache lookup.
# Defer it until a workflow actually performs retrieval; importing a service
# that does not use RAG (for example preflight validation) must stay local and
# deterministic.
def _ensure_faiss_dbs_loaded(config: Optional[Config] = None) -> Dict[str, Any]:
    """Load and return the index set matching ``config`` exactly once."""
    cfg = config or Config()
    cache_key = _faiss_cache_key(cfg)
    with _FAISS_DB_CACHE_LOCK:
        # An empty mapping is also a cached result: repeatedly attempting to
        # load a missing index set is expensive and obscures the original
        # configuration error.
        if cache_key not in FAISS_DB_CACHE:
            FAISS_DB_CACHE[cache_key] = load_faiss_dbs(cfg)
        return FAISS_DB_CACHE[cache_key]

class FoamfilePydantic(BaseModel):
    file_name: str = Field(description="Name of the OpenFOAM input file")
    folder_name: str = Field(description="Folder where the foamfile should be stored")
    content: str = Field(description="Content of the OpenFOAM file, written in OpenFOAM dictionary format")

class FoamPydantic(BaseModel):
    list_foamfile: List[FoamfilePydantic] = Field(description="List of OpenFOAM configuration files")

class ResponseWithThinkPydantic(BaseModel):
    think: str = Field(description="Thought process of the LLM")
    response: str = Field(description="Response of the LLM")

class _CodexResponsesWrapper:
    """Wrapper for an OpenAI Responses-compatible endpoint.

    This mimics the minimal interface LLMService expects from LangChain chat models:
    - invoke(messages) -> object with .content
    - get_num_tokens(text) -> int

    We support two wire endpoints:
    - OpenAI Platform: https://api.openai.com/v1/responses (API key / some OAuth tokens)
    - ChatGPT/Codex subscription backend: https://chatgpt.com/backend-api/codex/responses

    The ChatGPT backend requires a non-empty `instructions` field that matches the Codex harness
    expectations. We ship a default copy in `src/codex_instructions_default.txt`.
    """

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def __init__(
        self,
        token: str,
        model: str,
        temperature: float = 0.0,
        *,
        base_url: str = "https://api.openai.com/v1",
        account_id: Optional[str] = None,
        instructions: Optional[str] = None,
        stream: bool = False,
    ):
        self._token = token
        self._model = model
        self._temperature = temperature
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._instructions = instructions
        self._stream = stream
        # Token counting (best-effort). Exact tokenization may differ by model.
        # We default to a modern tokenizer; adjust if you need model-specific counting.
        try:
            self._enc = tiktoken.get_encoding("o200k_base")
        except (KeyError, ValueError):
            self._enc = tiktoken.get_encoding("cl100k_base")

    def get_num_tokens(self, text: str) -> int:
        return len(self._enc.encode(text or ""))

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Best-effort extraction of a JSON object from a model response."""
        if not text:
            raise ValueError("Empty response; expected JSON")

        s = text.strip()
        # Strip fenced code blocks.
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
            s = re.sub(r"\n```\s*$", "", s).strip()

        # If it's already a JSON object, return it.
        if s.startswith("{") and s.endswith("}"):
            return s

        # Otherwise, find the first {...} block.
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            raise ValueError(f"Could not find a JSON object in response: {s[:200]}")
        return m.group(0)

    def with_structured_output(self, pydantic_obj: Type[BaseModel]):
        """Return a wrapper that parses the model output into a Pydantic object.

        This is a minimal compatibility shim for LangChain's `.with_structured_output()`.
        """

        parent = self

        class _StructuredWrapper:
            def get_num_tokens(self, text: str) -> int:
                return parent.get_num_tokens(text)

            def invoke(self, messages):
                schema = pydantic_obj.model_json_schema()
                schema_hint = (
                    "Return ONLY valid JSON (no markdown) that matches this JSON Schema:\n"
                    + str(schema)
                )

                patched = list(messages)
                # Prepend a system constraint for JSON output.
                patched.insert(0, {"role": "system", "content": schema_hint})

                resp = parent.invoke(patched)
                raw = getattr(resp, "content", "")
                json_text = parent._extract_json_object(raw)
                return pydantic_obj.model_validate_json(json_text)

        return _StructuredWrapper()

    @staticmethod
    def _to_responses_input(messages):
        out = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            # Responses API supports rich content; we use simple input_text.
            out.append({"role": role, "content": [{"type": "input_text", "text": content}]})
        return out

    @staticmethod
    def _extract_output_text(resp_json: dict) -> str:
        # Newer APIs often include output_text; fall back to traversing.
        if isinstance(resp_json, dict) and isinstance(resp_json.get("output_text"), str):
            return resp_json["output_text"]

        texts = []
        for item in resp_json.get("output", []) if isinstance(resp_json, dict) else []:
            for c in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(c, dict):
                    if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                        texts.append(c["text"])
        return "\n".join(texts).strip()

    def _build_payload(self, messages):
        payload = {
            "model": self._model,
            "input": self._to_responses_input(messages),
        }

        # OpenAI Platform supports temperature.
        if "chatgpt.com" not in self._base_url:
            payload["temperature"] = self._temperature

        # ChatGPT/Codex subscription backend expects these extra keys.
        if "chatgpt.com" in self._base_url:
            payload.update(
                {
                    "instructions": self._instructions or "",
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "reasoning": {"summary": "auto"},
                    "store": False,
                    "stream": bool(self._stream),
                    "include": ["reasoning.encrypted_content"],
                }
            )
        return payload

    @staticmethod
    def _iter_sse_text(resp: requests.Response):
        """Yield decoded SSE 'data:' payloads as strings."""
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            line = str(raw).strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            yield data

    def _response_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not self._stream else "text/event-stream",
            "User-Agent": "Foam-Agent",
        }
        if self._account_id:
            headers["ChatGPT-Account-Id"] = self._account_id
        return headers

    def _post_responses_request(self, messages) -> requests.Response:
        """Send one Responses request and preserve useful HTTP failure detail."""
        url = f"{self._base_url}/responses"
        payload = self._build_payload(messages)
        timeout = int(os.environ.get("FOAMAGENT_HTTP_TIMEOUT", "300"))
        response = requests.post(
            url,
            headers=self._response_headers(),
            json=payload,
            timeout=timeout,
            stream=bool(self._stream),
        )
        if not response.ok:
            try:
                detail = response.text[:2000]
            except requests.RequestException:
                detail = ""
            raise requests.HTTPError(
                f"HTTP {response.status_code} for {url}. Body: {detail}",
                response=response,
            )
        return response

    def _stream_output_text(self, response: requests.Response) -> str:
        """Accumulate supported Responses SSE text events with a safe fallback."""
        import json

        chunks: list[str] = []
        for text in self._iter_sse_text(response):
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                event_type = event.get("type")
                if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
                    chunks.append(event["delta"])
                    continue
                if event_type == "response.output_text.done" and isinstance(event.get("text"), str):
                    if not chunks:
                        chunks.append(event["text"])
                    continue
            fallback = self._extract_output_text(event)
            if fallback:
                chunks.append(fallback)
        return "".join(chunks).strip()

    def invoke(self, messages):
        response = self._post_responses_request(messages)
        if self._stream:
            return self._Resp(self._stream_output_text(response))
        return self._Resp(self._extract_output_text(response.json()))


class LLMService:
    @staticmethod
    def _load_codex_oauth_from_auth_json(auth_json_path: Path) -> tuple[str, Optional[str]]:
        import json

        data = json.loads(auth_json_path.read_text(encoding="utf-8"))

        # Be permissive: different Codex versions may store different shapes.
        # Common patterns we try:
        #   {"access_token": "..."}
        #   {"token": "..."}
        #   {"tokens": {"access_token": "...", "account_id": "..."}}
        #   {"auth": {"access_token": "..."}}
        #   {"credentials": {"access_token": "..."}}
        candidates = []
        account_id = None

        def maybe_add(v):
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())

        if isinstance(data, dict):
            maybe_add(data.get("access_token"))
            maybe_add(data.get("token"))

            for k in ("auth", "credentials", "session"):
                v = data.get(k)
                if isinstance(v, dict):
                    maybe_add(v.get("access_token"))
                    maybe_add(v.get("token"))

            tokens = data.get("tokens")
            if isinstance(tokens, dict):
                maybe_add(tokens.get("access_token"))
                maybe_add(tokens.get("token"))
                if isinstance(tokens.get("account_id"), str) and tokens["account_id"].strip():
                    account_id = tokens["account_id"].strip()

        if not candidates:
            raise ValueError(
                f"Could not find an access token in {auth_json_path}. "
                "Expected keys like access_token/token or tokens.access_token."
            )

        # Prefer access_token-like strings first (we already appended in that order)
        return candidates[0], account_id

    @staticmethod
    def _load_codex_oauth_from_clawdbot_auth_profiles(auth_profiles_path: Path) -> tuple[str, Optional[str]]:
        """Load (access token, account id) from Clawdbot's OpenAI-Codex OAuth cache.

        Expected shape (v1):
          {"profiles": {"openai-codex:default": {"access": "...", "accountId": "...", ...}}}

        We also fall back to "openai-codex" or any first profile that looks usable.
        """
        import json

        data = json.loads(auth_profiles_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected JSON in {auth_profiles_path}")

        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError(f"Missing 'profiles' in {auth_profiles_path}")

        preferred_keys = ["openai-codex:default", "openai-codex"]
        for k in preferred_keys:
            v = profiles.get(k)
            if isinstance(v, dict):
                token = v.get("access")
                account_id = v.get("accountId")
                if isinstance(token, str) and token.strip():
                    return token.strip(), account_id if isinstance(account_id, str) else None

        # Fallback: scan any profile entry that has an 'access' string
        for _, v in profiles.items():
            if isinstance(v, dict):
                token = v.get("access")
                account_id = v.get("accountId")
                if isinstance(token, str) and token.strip():
                    return token.strip(), account_id if isinstance(account_id, str) else None

        raise ValueError(
            f"Could not find an 'access' token in {auth_profiles_path}. "
            "Expected profiles[*].access"
        )

    def _load_codex_oauth(self) -> tuple[str, Optional[str]]:
        """Load the Codex/ChatGPT OAuth token from a local auth cache.

        Supported locations (first match wins):
        1) $CODEX_HOME/auth.json (Codex CLI file-based cache)
        2) ~/.codex/auth.json (Codex CLI default)
        3) ~/.clawdbot/agents/main/agent/auth-profiles.json (Clawdbot OpenAI-Codex OAuth cache)

        Note: These files contain access/refresh tokens. Treat them like passwords.
        """
        candidates: list[Path] = []

        codex_home = os.getenv("CODEX_HOME")
        if codex_home:
            candidates.append(Path(codex_home) / "auth.json")

        candidates.append(Path.home() / ".codex" / "auth.json")

        # Clawdbot stores the OpenAI-Codex OAuth profile here.
        candidates.append(
            Path.home()
            / ".clawdbot"
            / "agents"
            / "main"
            / "agent"
            / "auth-profiles.json"
        )

        for p in candidates:
            if not p.exists():
                continue

            # Codex CLI cache
            if p.name == "auth.json":
                return self._load_codex_oauth_from_auth_json(p)

            # Clawdbot cache
            if p.name == "auth-profiles.json":
                return self._load_codex_oauth_from_clawdbot_auth_profiles(p)

        raise FileNotFoundError(
            "model_provider='openai-codex' requires a Codex/ChatGPT OAuth cache. "
            "Looked for: "
            + ", ".join(str(x) for x in candidates)
            + ". "
            "If you used the Codex CLI, run `codex login` and ensure file-based credential storage. "
            "If you used Clawdbot, make sure you completed OpenAI Codex OAuth in onboarding. "
            "To use an OpenAI Platform API key instead, set "
            "FOAMAGENT_MODEL_PROVIDER=openai along with OPENAI_API_KEY."
        )

    def __init__(self, config: object):
        self.model_version = getattr(config, "model_version", "gpt-4o")
        self.temperature = getattr(config, "temperature", 0)
        self.model_provider = getattr(config, "model_provider", "openai")
        self._config = config
        
        # Initialize statistics
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.failed_calls = 0
        self.retry_count = 0
        
        # Initialize the LLM
        if self.model_provider.lower() == "bedrock":
            bedrock_runtime = tracking_aws.new_default_client()
            self.llm = ChatBedrockConverse(
                client=bedrock_runtime, 
                model_id=self.model_version, 
                temperature=self.temperature, 
                max_tokens=8192
            )
        elif self.model_provider.lower() == "anthropic":
            self.llm = ChatAnthropic(
                model=self.model_version,
                temperature=self.temperature
            )
        elif self.model_provider.lower() == "openai":
            # Usage-based API access (requires OPENAI_API_KEY or equivalent OpenAI SDK config)
            self.llm = init_chat_model(
                self.model_version,
                model_provider=self.model_provider,
                temperature=self.temperature,
            )
        elif self.model_provider.lower() in {"openai-codex", "codex", "chatgpt-oauth"}:
            # Subscription-based access via "Sign in with ChatGPT" (Codex auth cache).
            # We use the OpenAI Responses API, which is the typical surface for Codex subscription access.
            token, account_id = self._load_codex_oauth()

            # ChatGPT/Codex subscription route: use the same endpoint as Codex CLI.
            # This avoids requiring Platform API scopes like api.responses.write.
            instructions_path = Path(__file__).resolve().parent / "codex_instructions_default.txt"
            try:
                instructions = instructions_path.read_text(encoding="utf-8")
            except OSError:
                instructions = "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer."

            self.llm = _CodexResponsesWrapper(
                token=token,
                account_id=account_id,
                model=self.model_version,
                temperature=self.temperature,
                base_url="https://chatgpt.com/backend-api/codex",
                instructions=instructions,
                stream=True,
            )
        elif self.model_provider.lower() == "ollama":
            try:
                requests.get("http://localhost:11434/api/version", timeout=2)
                # If request successful, service is running
            except requests.exceptions.RequestException:
                print("Ollama is not running, starting it...")
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                # Wait for service to start
                time.sleep(5)  # Give it 3 seconds to initialize

            self.llm = ChatOllama(
                model=self.model_version, 
                temperature=self.temperature,
                num_predict=-1,
                num_ctx=131072,
                base_url="http://localhost:11434"
            )
        elif self.model_provider.lower() == "deepseek":
            from langchain_openai import ChatOpenAI
            reasoning = os.getenv("FOAMAGENT_REASONING_EFFORT", "max")
            if reasoning not in ("low", "medium", "high", "max"):
                reasoning = "max"
            # Note: temperature is ignored by DeepSeek in thinking mode.
            self.llm = ChatOpenAI(
                model=self.model_version,
                temperature=self.temperature,
                base_url="https://api.deepseek.com/v1",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                reasoning_effort=reasoning,
                extra_body={"thinking": {"type": "enabled"}},
            )
        else:
            raise ValueError(f"{self.model_provider} is not a supported model_provider")
    
    def _is_throttling_error(self, error: Exception) -> bool:
        """
        Check if an exception is a throttling-related error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if it's a throttling error, False otherwise
        """
        # Check ClientError with specific error codes
        if isinstance(error, ClientError):
            error_code = error.response.get('Error', {}).get('Code', '')
            return error_code in ('Throttling', 'TooManyRequestsException', 'ThrottlingException')
        
        # Check for ThrottlingException and throttling-related error messages
        error_type = type(error).__name__
        error_str = str(error)
        
        throttling_indicators = (
            error_type == 'ThrottlingException',
            'ThrottlingException' in error_str,
            'Too many tokens' in error_str,
            'reached max retries' in error_str
        )
        
        return any(throttling_indicators)
    
    def _handle_throttling_retry(self, error: Exception, retry_count: int, max_retries: int) -> Optional[int]:
        """
        Handle throttling error by implementing exponential backoff retry logic.
        
        Args:
            error: The throttling exception
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries allowed
            
        Returns:
            The updated retry count if retry should continue, None if max retries exceeded
        """
        retry_count += 1
        self.retry_count += 1
        
        if retry_count > max_retries:

            print(f"Maximum retries ({max_retries}) exceeded: {str(error)}")
            return None
        
        # Exponential backoff with jitter
        base_delay = 1.0
        max_delay = 60.0
        delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
        jitter = random.uniform(0, 0.1 * delay)
        sleep_time = delay + jitter
        
        print(f"ThrottlingException occurred: {str(error)}. Retrying in {sleep_time:.2f} seconds (attempt {retry_count}/{max_retries})")
        time.sleep(sleep_time)
        
        return retry_count

    @staticmethod
    def _is_retryable_transport_error(error: Exception) -> bool:
        """Recognise short-lived network failures from an LLM provider.

        Only connection, timeout, TLS and proxy transport errors are retried.
        HTTP response errors (for example authentication or invalid-request
        errors) are deliberately left to the caller because retrying them
        cannot make a malformed request valid.
        """
        return isinstance(
            error,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.SSLError,
            ),
        )

    def _handle_transport_retry(
        self,
        error: Exception,
        retry_count: int,
        max_retries: int,
    ) -> Optional[int]:
        """Retry a transient LLM transport error with bounded backoff."""
        retry_count += 1
        self.retry_count += 1
        if retry_count > max_retries:
            print(f"Maximum transport retries ({max_retries}) exceeded: {error}")
            return None

        delay = min(8.0, 1.0 * (2 ** (retry_count - 1)))
        print(
            "Transient LLM transport error; retrying in "
            f"{delay:.1f} seconds (attempt {retry_count}/{max_retries}): {error}"
        )
        time.sleep(delay)
        return retry_count

    @staticmethod
    def _is_retryable_structured_response_error(error: Exception) -> bool:
        """Recognise transient empty/truncated structured-output responses.

        This deliberately does not retry arbitrary Pydantic validation errors:
        those usually indicate that the prompt or schema needs a real repair.
        It covers only the two failures emitted by the local structured-output
        wrapper before any JSON has been received.
        """
        message = str(error).casefold()
        return (
            "empty response; expected json" in message
            or "could not find a json object in response" in message
        )

    def _handle_structured_response_retry(
        self,
        error: Exception,
        retry_count: int,
        max_retries: int,
    ) -> Optional[int]:
        """Retry a transient empty structured response with bounded backoff."""
        retry_count += 1
        self.retry_count += 1
        if retry_count > max_retries:
            print(
                "Maximum structured-response retries "
                f"({max_retries}) exceeded: {error}"
            )
            return None

        delay = min(4.0, 0.5 * (2 ** (retry_count - 1)))
        print(
            "Structured LLM response was empty or truncated; retrying in "
            f"{delay:.1f} seconds (attempt {retry_count}/{max_retries})."
        )
        time.sleep(delay)
        return retry_count

    @staticmethod
    def _messages_for_request(user_prompt: str, system_prompt: Optional[str]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _invoke_deepseek_structured(
        self,
        messages: list[dict[str, str]],
        schema: Type[BaseModel],
    ) -> BaseModel:
        """Use the JSON prompt fallback required by DeepSeek thinking mode."""
        json_instruction = (
            "Return ONLY valid JSON (no markdown, no extra text) matching this schema:\n"
            + str(schema.model_json_schema())
        )
        raw_response = self.llm.invoke(
            [*messages, {"role": "user", "content": json_instruction}]
        )
        text = raw_response.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text).strip()
        return schema.model_validate_json(text)

    def _invoke_model(
        self,
        messages: list[dict[str, str]],
        pydantic_obj: Optional[Type[BaseModel]],
    ) -> Any:
        """Call the configured provider, including its structured-output variant."""
        if pydantic_obj is None:
            return self.llm.invoke(messages).content
        if self.model_provider.lower() == "deepseek":
            return self._invoke_deepseek_structured(messages, pydantic_obj)
        return self.llm.with_structured_output(pydantic_obj).invoke(messages)

    def _record_successful_response(self, response: Any, prompt_tokens: int) -> None:
        completion_tokens = self.llm.get_num_tokens(str(response))
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens

    def _retry_counts_after_error(
        self,
        error: Exception,
        *,
        retry_count: int,
        transport_retry_count: int,
        structured_retry_count: int,
        max_retries: int,
        has_structured_output: bool,
    ) -> tuple[int, int, int]:
        """Sleep and return updated counters for retryable failures, else raise."""
        if self._is_throttling_error(error):
            updated = self._handle_throttling_retry(error, retry_count, max_retries)
            if updated is not None:
                return updated, transport_retry_count, structured_retry_count
            self.failed_calls += 1
            raise RuntimeError(
                f"Maximum retries ({max_retries}) exceeded for throttling error: {error}"
            ) from error
        if self._is_retryable_transport_error(error):
            updated = self._handle_transport_retry(error, transport_retry_count, 3)
            if updated is not None:
                return retry_count, updated, structured_retry_count
            self.failed_calls += 1
            raise error
        if has_structured_output and self._is_retryable_structured_response_error(error):
            updated = self._handle_structured_response_retry(error, structured_retry_count, 3)
            if updated is not None:
                return retry_count, transport_retry_count, updated
            self.failed_calls += 1
            raise error
        print(f"Non-throttling error occurred: {error}.")
        if isinstance(error, ClientError):
            print(error.response)
        self.failed_calls += 1
        raise error

    def invoke(self,
              user_prompt: str, 
              system_prompt: Optional[str] = None, 
              pydantic_obj: Optional[Type[BaseModel]] = None,
              max_retries: int = 10) -> Any:
        """
        Invoke the LLM with the given prompts and return the response.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: Optional system prompt
            pydantic_obj: Optional Pydantic model for structured output
            max_retries: Maximum number of retries for throttling errors
            
        Returns:
            The LLM response with token usage statistics
        """
        self.total_calls += 1
        
        messages = self._messages_for_request(user_prompt, system_prompt)
        prompt_tokens = sum(self.llm.get_num_tokens(message["content"]) for message in messages)
        retry_count = 0
        transport_retry_count = 0
        structured_response_retry_count = 0
        while True:
            try:
                response = self._invoke_model(messages, pydantic_obj)
                self._record_successful_response(response, prompt_tokens)
                return response
            except Exception as error:  # noqa: BLE001 - provider exceptions have no common base class
                retry_count, transport_retry_count, structured_response_retry_count = (
                    self._retry_counts_after_error(
                        error,
                        retry_count=retry_count,
                        transport_retry_count=transport_retry_count,
                        structured_retry_count=structured_response_retry_count,
                        max_retries=max_retries,
                        has_structured_output=pydantic_obj is not None,
                    )
                )
    
    def get_statistics(self) -> dict:
        """
        Get the current statistics of the LLM service.
        
        Returns:
            Dictionary containing various statistics
        """
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "retry_count": self.retry_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "average_prompt_tokens": self.total_prompt_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_completion_tokens": self.total_completion_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_tokens": self.total_tokens / self.total_calls if self.total_calls > 0 else 0
        }
    
    def print_statistics(self) -> None:
        """
        Print the current statistics of the LLM service.
        """
        stats = self.get_statistics()
        print("\n<LLM Service Statistics>")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Total retries: {stats['retry_count']}")
        print(f"Total prompt tokens: {stats['total_prompt_tokens']}")
        print(f"Total completion tokens: {stats['total_completion_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average prompt tokens per call: {stats['average_prompt_tokens']:.2f}")
        print(f"Average completion tokens per call: {stats['average_completion_tokens']:.2f}")
        print(f"Average tokens per call: {stats['average_tokens']:.2f}\n")
        print("</LLM Service Statistics>")

class GraphState(TypedDict):
    user_requirement: str
    config: Config
    case_dir: str
    tutorial: str
    case_name: str
    subtasks: List[dict]
    current_subtask_index: int
    error_command: Optional[str]
    error_content: Optional[str]
    loop_count: int
    # Additional state fields that will be added during execution
    llm_service: Optional['LLMService']
    case_stats: Optional[dict]
    tutorial_reference: Optional[str]
    case_path_reference: Optional[str]
    dir_structure_reference: Optional[str]
    case_info: Optional[str]
    allrun_reference: Optional[str]
    dir_structure: Optional[dict]
    commands: Optional[List[str]]
    foamfiles: Optional[dict]
    error_logs: Optional[List[str]]
    history_text: Optional[List[str]]
    case_domain: Optional[str]
    case_category: Optional[str]
    case_solver: Optional[str]
    # Mesh-related state fields
    mesh_info: Optional[dict]
    mesh_commands: Optional[List[str]]
    custom_mesh_used: Optional[bool]
    mesh_type: Optional[str]
    custom_mesh_path: Optional[str]
    # Review and rewrite related fields
    review_analysis: Optional[str]
    rewrite_plan: Optional[dict]
    input_writer_mode: Optional[str]
    similar_case_advice: Optional[dict]
    # Routing decision cache
    requires_hpc: Optional[bool]
    requires_visualization: Optional[bool]
    # HPC-related fields
    job_id: Optional[str]
    cluster_info: Optional[dict]
    slurm_script_path: Optional[str]
    termination_reason: Optional[str]
    # Existing-case import branch.  These fields keep the imported source and
    # its controlled retry history separate from generated-case dictionaries.
    workflow_mode: str
    execution_policy: str
    repair_policy: str
    case_import_path: Optional[str]
    case_import_subdir: Optional[str]
    case_import_manifest: Optional[Any]
    case_import_original_dir: Optional[str]
    case_import_report_dir: Optional[str]
    case_import_attempts: Optional[List[dict]]
    case_import_overrides: Optional[Dict[str, bytes]]
    case_import_error_fingerprints: Optional[List[str]]
    case_import_status: Optional[str]

def tokenize(text: str) -> str:
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Insert a space between a lowercase letter and an uppercase letter (global match)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.lower()

def save_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Saved file at {path}")

def read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""

def list_case_files(case_dir: str) -> str:
    files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    return ", ".join(files)

def remove_files(directory: str, prefix: str) -> None:
    for file in os.listdir(directory):
        if file.startswith(prefix):
            os.remove(os.path.join(directory, file))
    print(f"Removed files with prefix '{prefix}' in {directory}")

def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed file {path}")

def remove_numeric_folders(case_dir: str) -> None:
    """
    Remove all folders in case_dir that represent numeric values, including those with decimal points,
    except for the "0" folder.
    
    Args:
        case_dir (str): The directory path to process
    """
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path) and item != "0":
            try:
                # Try to convert to float to check if it's a numeric value
                float(item)
                # If conversion succeeds, it's a numeric folder
                try:
                    shutil.rmtree(item_path)
                    print(f"Removed numeric folder: {item_path}")
                except OSError as e:
                    print(f"Error removing folder {item_path}: {str(e)}")
            except ValueError:
                # Not a numeric value, so we keep this folder
                pass


def scan_case_directory(case_dir: str) -> Dict[str, List[str]]:
    """
    Scan an OpenFOAM case directory and return the directory structure.
    
    This function traverses the case directory one level deep and collects
    the files in each subdirectory (typically 'system', 'constant', '0', etc.).
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping folder names to lists of file names
            Example: {"system": ["controlDict", "fvSchemes"], "constant": ["transportProperties"]}
    
    Raises:
        FileNotFoundError: If case_dir does not exist
        PermissionError: If directory cannot be accessed
    
    Example:
        >>> structure = scan_case_directory("/path/to/case")
        >>> print(structure["system"])  # ["controlDict", "fvSchemes", "fvSolution"]
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    
    dir_structure = {}
    base_depth = case_dir.rstrip(os.sep).count(os.sep)
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(case_dir):
        # Only process directories one level below case_dir
        current_depth = root.rstrip(os.sep).count(os.sep)
        if current_depth == base_depth + 1:
            folder_name = os.path.relpath(root, case_dir)
            # Filter out hidden files and only include regular files
            regular_files = [f for f in files if not f.startswith('.') and os.path.isfile(os.path.join(root, f))]
            if regular_files:
                dir_structure[folder_name] = regular_files
    
    return dir_structure


def read_case_foamfiles(case_dir: str, dir_structure: Optional[Dict[str, List[str]]] = None) -> 'FoamPydantic':
    """
    Read OpenFOAM files from a case directory and convert to FoamPydantic format.
    
    This function reads all OpenFOAM configuration files from the case directory
    (typically from 'system', 'constant', '0' folders) and creates a FoamPydantic
    object containing the file contents.
    
    Args:
        case_dir (str): Path to the OpenFOAM case directory
        dir_structure (Optional[Dict[str, List[str]]]): Pre-scanned directory structure.
            If None, will scan the directory automatically.
    
    Returns:
        FoamPydantic: Object containing list of FoamfilePydantic objects with file metadata
    
    Raises:
        FileNotFoundError: If case_dir does not exist
        UnicodeDecodeError: If files contain invalid encoding (will skip those files)
    
    Example:
        >>> foamfiles = read_case_foamfiles("/path/to/case")
        >>> print(len(foamfiles.list_foamfile))  # Number of files read
        >>> print(foamfiles.list_foamfile[0].file_name)  # "controlDict"
    """
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    
    # Scan directory structure if not provided
    if dir_structure is None:
        dir_structure = scan_case_directory(case_dir)
    
    foamfile_list = []
    
    # Read files from each folder
    for folder_name, file_names in dir_structure.items():
        for file_name in file_names:
            file_path = os.path.join(case_dir, folder_name, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                foamfile_list.append(FoamfilePydantic(
                    file_name=file_name,
                    folder_name=folder_name,
                    content=content
                ))
            except UnicodeDecodeError:
                print(f"Warning: Skipping file due to encoding error: {file_path}")
            except OSError as e:
                print(f"Warning: Error reading file {file_path}: {e}")
    
    return FoamPydantic(list_foamfile=foamfile_list)

def run_command(
    script_path: str,
    out_file: str,
    err_file: str,
    working_dir: str,
    max_time_limit: int,
) -> Dict[str, Any]:
    """Run an OpenFOAM shell script and return its process outcome.

    Existing callers may continue to ignore the return value.  Callers which
    need reliable execution status can inspect ``returncode`` and
    ``timed_out`` instead of inferring success solely from log contents.
    """
    print(f"Executing script {script_path} in {working_dir}")
    script = Path(script_path).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"OpenFOAM script does not exist: {script}")
    openfoam_dir = os.getenv("WM_PROJECT_DIR")
    if not openfoam_dir:
        raise RuntimeError(
            "WM_PROJECT_DIR is not set. Please source OpenFOAM environment before running Foam-Agent "
            "(e.g., source env/common.sh and env/foamagent.sh)."
        )

    bashrc_path = Path(openfoam_dir).expanduser() / "etc" / "bashrc"
    if not bashrc_path.is_file():
        raise RuntimeError(f"OpenFOAM bashrc not found at: {bashrc_path}")

    # Never interpolate paths into shell source.  Script and bashrc paths can
    # legitimately contain spaces and, for imported cases, originate outside
    # Foam-Agent.  Passing them as positional parameters prevents shell
    # metacharacters from being interpreted as code.
    shell_command = 'source "$1" && exec bash "$2"'

    timed_out = False

    with open(out_file, 'w') as out, open(err_file, 'w') as err:
        process = subprocess.Popen(
            ["bash", "-c", shell_command, "foamagent-runner", str(bashrc_path), str(script)],
            cwd=working_dir,
            # Stream child output directly to disk.  ``communicate()`` buffers
            # all solver output in memory and can OOM on a legitimate long CFD
            # run with verbose residual logs.
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

        try:
            process.wait(timeout=max_time_limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                # The process can exit between ``wait`` timing out and the
                # group lookup; its return code is still collected below.
                pass
            process.wait()
            timeout_message = (
                f"OpenFOAM execution exceeded the {max_time_limit} second timeout.\n"
            )
            out.write(timeout_message)
            err.write(timeout_message)
            print(f"Execution timed out: {script_path}")

    print(f"Executed script {script_path}")
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
    }

_EXPLICIT_FOAM_ERROR_RE = re.compile(r"ERROR:(.*)", re.DOTALL)
_SEMANTIC_FOAM_FAILURES = (
    (
        re.compile(r"Failed\s+\d+\s+mesh\s+checks?", re.IGNORECASE),
        "checkMesh reported failed mesh checks",
    ),
)
_FOAM_END_RE = re.compile(r"^\s*End\s*$", re.MULTILINE)


def _read_foam_log(directory: str, file_name: str) -> tuple[str | None, dict | None]:
    filepath = os.path.join(directory, file_name)
    try:
        with open(filepath, encoding="utf-8") as log_file:
            return log_file.read(), None
    except OSError:
        return None, {"file": file_name, "error_content": f"Could not read log file: {filepath}"}


def _foam_log_failure(file_name: str, content: str) -> dict | None:
    """Return an explicit or semantic OpenFOAM error found in one log."""
    for pattern, reason in _SEMANTIC_FOAM_FAILURES:
        match = pattern.search(content)
        if match:
            line = content[match.start() :].splitlines()[0].strip()
            return {
                "file": file_name,
                "error_content": f"Semantic simulation failure: {reason}. OpenFOAM reported: {line}",
            }
    match = _EXPLICIT_FOAM_ERROR_RE.search(content)
    if match:
        return {"file": file_name, "error_content": match.group(0).strip()}
    return None


def _unfinished_foam_log(file_name: str, content: str) -> dict | None:
    if _FOAM_END_RE.search(content):
        return None
    last_lines = "\n".join(content.strip().split("\n")[-30:])
    return {
        "file": file_name,
        "error_content": (
            "Solver did not complete (no 'End' marker found). "
            f"Last 30 lines:\n{last_lines}"
        ),
    }


def check_foam_errors(directory: str) -> list:
    """Check OpenFOAM log files for errors.

    Tier 1 (existing): Match explicit ``ERROR:`` lines.
    Tier 2 (safety-net): If no explicit error is found, verify that **every**
    log file contains the ``End`` marker that OpenFOAM prints on successful
    completion.  Any log missing ``End`` is reported with the last 30 lines
    as error context so the caller can diagnose the crash.
    """
    error_logs: list[dict] = []
    log_contents: dict[str, str] = {}
    for file_name in os.listdir(directory):
        if not file_name.startswith("log"):
            continue
        content, read_error = _read_foam_log(directory, file_name)
        if read_error:
            error_logs.append(read_error)
            continue
        if content is None:
            continue
        log_contents[file_name] = content
        failure = _foam_log_failure(file_name, content)
        if failure:
            error_logs.append(failure)
    if error_logs or not log_contents:
        return error_logs
    return [
        failure
        for file_name, content in log_contents.items()
        if (failure := _unfinished_foam_log(file_name, content)) is not None
    ]

def extract_commands_from_allrun_out(out_file: str) -> list:
    commands = []
    if not os.path.exists(out_file):
        return commands
    with open(out_file, 'r') as f:
        for line in f:
            if line.startswith("Running "):
                parts = line.split(" ")
                if len(parts) > 1:
                    commands.append(parts[1].strip())
    return commands

def parse_case_name(text: str) -> str:
    match = re.search(r'case name:\s*(.+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "default_case"

def split_subtasks(text: str) -> list:
    header_match = re.search(r'splits into (\d+) subtasks:', text, re.IGNORECASE)
    if not header_match:
        print("Warning: No subtasks header found in the response.")
        return []
    num_subtasks = int(header_match.group(1))
    subtasks = re.findall(r'subtask\d+:\s*(.*)', text, re.IGNORECASE)
    if len(subtasks) != num_subtasks:
        print(f"Warning: Expected {num_subtasks} subtasks but found {len(subtasks)}.")
    return subtasks

def parse_context(text: str) -> str:
    match = re.search(r'FoamFile\s*\{.*?(?=```|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    print("Warning: Could not parse context; returning original text.")
    return text


def parse_file_name(subtask: str) -> str:
    match = re.search(r'openfoam\s+(.*?)\s+foamfile', subtask, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def parse_folder_name(subtask: str) -> str:
    match = re.search(r'foamfile in\s+(.*?)\s+folder', subtask, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def find_similar_file(description: str, tutorial: str) -> str:
    start_pos = tutorial.find(description)
    if start_pos == -1:
        return "None"
    end_marker = "input_file_end."
    end_pos = tutorial.find(end_marker, start_pos)
    if end_pos == -1:
        return "None"
    return tutorial[start_pos:end_pos + len(end_marker)]

def read_commands(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Commands file not found: {file_path}")
    with open(file_path, 'r') as f:
        # join non-empty lines with a comma
        return ", ".join(line.strip() for line in f if line.strip())

def find_input_file(case_dir: str, command: str) -> str:
    for root, _, files in os.walk(case_dir):
        for file in files:
            if command in file:
                return os.path.join(root, file)
    return ""

def retrieve_faiss(
    database_name: str,
    query: str,
    topk: int = 1,
    *,
    config: Optional[Config] = None,
) -> dict:
    """
    Retrieve a similar case from a FAISS database selected by ``config``.

    Omitting ``config`` retains the environment/default-backed behaviour used
    by legacy callers.  Supplying it is required when one process serves
    multiple configured workflows, because each embedding model needs its own
    compatible FAISS index and query vector.
    """

    dbs = _ensure_faiss_dbs_loaded(config)
    if database_name not in dbs:
        raise ValueError(f"Database '{database_name}' is not loaded.")

    # Tokenize the query
    query = tokenize(query)

    vectordb = dbs[database_name]
    try:
        docs_and_scores = vectordb.similarity_search_with_score(query, k=topk)
        docs = [d for d, _ in docs_and_scores]
        scores = [s for _, s in docs_and_scores]
    except (AttributeError, NotImplementedError):
        docs = vectordb.similarity_search(query, k=topk)
        scores = [None] * len(docs)

    if not docs:
        raise ValueError(f"No documents found for query: {query}")

    formatted_results = []
    for doc, score in zip(docs, scores):
        metadata = doc.metadata or {}

        if database_name == "openfoam_allrun_scripts":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "allrun_script": metadata.get("allrun_script", "N/A"),
                "score": score,
            })
        elif database_name == "openfoam_command_help":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "command": metadata.get("command", "unknown"),
                "help_text": metadata.get("help_text", "unknown"),
                "score": score,
            })
        elif database_name == "openfoam_tutorials_structure":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "score": score,
            })
        elif database_name == "openfoam_tutorials_details":
            formatted_results.append({
                "index": doc.page_content,
                "full_content": metadata.get("full_content", "unknown"),
                "case_name": metadata.get("case_name", "unknown"),
                "case_domain": metadata.get("case_domain", "unknown"),
                "case_category": metadata.get("case_category", "unknown"),
                "case_solver": metadata.get("case_solver", "unknown"),
                "dir_structure": metadata.get("dir_structure", "unknown"),
                "tutorials": metadata.get("tutorials", "N/A"),
                "score": score,
            })
        else:
            raise ValueError(f"Unknown database name: {database_name}")

    return formatted_results
        

def parse_directory_structure(data: str) -> dict:
    """
    Parses the directory structure string and returns a dictionary where:
      - Keys: directory names
      - Values: count of files in that directory.
    """
    directory_file_counts = {}

    # Find all <dir>...</dir> blocks in the input string.
    dir_blocks = re.findall(r'<dir>(.*?)</dir>', data, re.DOTALL)

    for block in dir_blocks:
        # Extract the directory name (everything after "directory name:" until the first period)
        dir_name_match = re.search(r'directory name:\s*(.*?)\.', block)
        # Extract the list of file names within square brackets
        files_match = re.search(r'File names in this directory:\s*\[(.*?)\]', block)
        
        if dir_name_match and files_match:
            dir_name = dir_name_match.group(1).strip()
            files_str = files_match.group(1)
            # Split the file names by comma, removing any surrounding whitespace
            file_list = [filename.strip() for filename in files_str.split(',')]
            directory_file_counts[dir_name] = len(file_list)

    return directory_file_counts
