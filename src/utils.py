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
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_anthropic import ChatAnthropic
from pathlib import Path
import tracking_aws
import requests
import time
import random
from botocore.exceptions import ClientError
import shutil
from config import Config
from langchain_ollama import ChatOllama
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None


# Global dictionary to store loaded FAISS databases
FAISS_DB_CACHE = {}

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

    base_dir = Path(__file__).resolve().parent.parent / "database" / "faiss"

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
            except Exception as e:
                print(f"Failed to load index {index}: {e}")
        else:
            print(f"Warning: Index path does not exist: {index_path}")

    return dbs


# Default DB cache (uses default Config()). If you change embedding settings at runtime,
# call load_faiss_dbs(custom_config) and replace FAISS_DB_CACHE.
FAISS_DB_CACHE = load_faiss_dbs()

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
        except Exception:
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

    def invoke(self, messages):
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not self._stream else "text/event-stream",
            "User-Agent": "Foam-Agent",
        }
        if self._account_id:
            headers["ChatGPT-Account-Id"] = self._account_id

        payload = self._build_payload(messages)

        r = requests.post(url, headers=headers, json=payload, timeout=60, stream=bool(self._stream))

        # If we get an error, surface the response body to aid debugging.
        if not r.ok:
            try:
                detail = r.text[:2000]
            except Exception:
                detail = ""
            raise requests.HTTPError(
                f"HTTP {r.status_code} for {url}. Body: {detail}", response=r
            )

        if not self._stream:
            data = r.json()
            return self._Resp(self._extract_output_text(data))

        # Streaming: accumulate text deltas.
        import json

        chunks: list[str] = []
        for s in self._iter_sse_text(r):
            try:
                j = json.loads(s)
            except Exception:
                continue

            # Codex backend streams OpenAI Responses-style events.
            if isinstance(j, dict):
                t = j.get("type")
                if t == "response.output_text.delta" and isinstance(j.get("delta"), str):
                    chunks.append(j["delta"])
                    continue
                if t == "response.output_text.done" and isinstance(j.get("text"), str):
                    # Some clients rely on done; we already collected deltas but keep as fallback.
                    if not chunks:
                        chunks.append(j["text"])
                    continue

            # Fallback: try generic extraction.
            t2 = self._extract_output_text(j)
            if t2:
                chunks.append(t2)

        return self._Resp("".join(chunks).strip())


class LLMService:
    @staticmethod
    def _load_codex_access_token_from_auth_json(auth_json_path: Path) -> str:
        import json

        data = json.loads(auth_json_path.read_text(encoding="utf-8"))

        # Be permissive: different Codex versions may store different shapes.
        # Common patterns we try:
        #   {"access_token": "..."}
        #   {"token": "..."}
        #   {"auth": {"access_token": "..."}}
        #   {"credentials": {"access_token": "..."}}
        candidates = []

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

        if not candidates:
            raise ValueError(
                f"Could not find an access token in {auth_json_path}. "
                "Expected keys like access_token/token/id_token."
            )

        # Prefer access_token-like strings first (we already appended in that order)
        return candidates[0]

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
                return self._load_codex_access_token_from_auth_json(p), None

            # Clawdbot cache
            if p.name == "auth-profiles.json":
                return self._load_codex_oauth_from_clawdbot_auth_profiles(p)

        raise FileNotFoundError(
            "Could not find a Codex/ChatGPT OAuth cache. Looked for: "
            + ", ".join(str(x) for x in candidates)
            + ". "
            "If you used the Codex CLI, run `codex login` and ensure file-based credential storage. "
            "If you used Clawdbot, make sure you completed OpenAI Codex OAuth in onboarding."
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
            except Exception:
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
                response = requests.get("http://localhost:11434/api/version", timeout=2)
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
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Calculate prompt tokens
        prompt_tokens = 0
        for message in messages:
            prompt_tokens += self.llm.get_num_tokens(message["content"])
        
        retry_count = 0
        while True:
            try:
                if pydantic_obj:
                    structured_llm = self.llm.with_structured_output(pydantic_obj)
                    response = structured_llm.invoke(messages)
                else:
                    if self.model_version.startswith("deepseek"):
                        structured_llm = self.llm.with_structured_output(ResponseWithThinkPydantic)
                        response = structured_llm.invoke(messages)

                        # Extract the resposne without the think
                        response = response.response
                    else:
                        response = self.llm.invoke(messages)
                        response = response.content

                # Calculate completion tokens
                response_content = str(response)
                completion_tokens = self.llm.get_num_tokens(response_content)
                total_tokens = prompt_tokens + completion_tokens
                
                # Update statistics
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                return response
                
            except Exception as e:
                if self._is_throttling_error(e):
                    print(f"ThrottlingException occurred: {str(e)}.")
                    print(f"Retrying: {retry_count + 1}/{max_retries}")
                    retry_count = self._handle_throttling_retry(e, retry_count, max_retries)
                    if retry_count is None:
                        # Max retries exceeded
                        self.failed_calls += 1
                        raise Exception(f"Maximum retries ({max_retries}) exceeded for throttling error: {str(e)}")
                    continue  # Retry the request
                else:
                    print(f"Non-throttling error occurred: {str(e)}.")

                    # Non-throttling error: log and raise
                    print(f"Error occurred in LLM service: {str(e)}")
                    if isinstance(e, ClientError):
                        print(e.response)
                    self.failed_calls += 1
                    raise e
    
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
                except Exception as e:
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
            except Exception as e:
                print(f"Warning: Error reading file {file_path}: {e}")
    
    return FoamPydantic(list_foamfile=foamfile_list)

def run_command(script_path: str, out_file: str, err_file: str, working_dir: str, max_time_limit: int) -> None:
    print(f"Executing script {script_path} in {working_dir}")
    os.chmod(script_path, 0o777)
    openfoam_dir = os.getenv("WM_PROJECT_DIR")
    if not openfoam_dir:
        raise RuntimeError(
            "WM_PROJECT_DIR is not set. Please source OpenFOAM environment before running Foam-Agent "
            "(e.g., source env/common.sh and env/foamagent.sh)."
        )

    bashrc_path = os.path.join(openfoam_dir, "etc", "bashrc")
    if not os.path.exists(bashrc_path):
        raise RuntimeError(f"OpenFOAM bashrc not found at: {bashrc_path}")

    command = f"source {bashrc_path} && bash {os.path.abspath(script_path)}"

    with open(out_file, 'w') as out, open(err_file, 'w') as err:
        process = subprocess.Popen(
            ['bash', "-c", command],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=max_time_limit)
            out.write(stdout)
            err.write(stderr)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL) 
            
            stdout, stderr = process.communicate()
            timeout_message = (
                "OpenFOAM execution took too long. "
                "This case, if set up right, does not require such large execution times.\n"
            )
            out.write(timeout_message + stdout)
            err.write(timeout_message + stderr)
            print(f"Execution timed out: {script_path}")

    

    print(f"Executed script {script_path}")

def check_foam_errors(directory: str) -> list:
    """Check OpenFOAM log files for errors.

    Tier 1 (existing): Match explicit ``ERROR:`` lines.
    Tier 2 (safety-net): If no explicit error is found, verify that **every**
    log file contains the ``End`` marker that OpenFOAM prints on successful
    completion.  Any log missing ``End`` is reported with the last 30 lines
    as error context so the caller can diagnose the crash.
    """
    error_logs = []
    log_contents = {}  # filename -> content

    # DOTALL mode allows '.' to match newline characters
    pattern = re.compile(r"ERROR:(.*)", re.DOTALL)

    for file in os.listdir(directory):
        if file.startswith("log"):
            filepath = os.path.join(directory, file)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
            except (IOError, OSError):
                error_logs.append({"file": file, "error_content": f"Could not read log file: {filepath}"})
                continue

            log_contents[file] = content

            match = pattern.search(content)
            if match:
                error_content = match.group(0).strip()
                error_logs.append({"file": file, "error_content": error_content})
            elif "error" in content.lower():
                print(f"Warning: file {file} contains 'error' but does not match expected format.")

    # Safety-net: if no explicit ERROR was found, check for missing 'End' marker
    # Check EACH log individually – a successful blockMesh should not mask a
    # crashed solver (e.g. pimpleFoam).
    if not error_logs and log_contents:
        end_pattern = re.compile(r"^\s*End\s*$", re.MULTILINE)

        for file, content in log_contents.items():
            if not end_pattern.search(content):
                last_lines = "\n".join(content.strip().split("\n")[-30:])
                error_logs.append({
                    "file": file,
                    "error_content": (
                        f"Solver did not complete (no 'End' marker found). "
                        f"Last 30 lines:\n{last_lines}"
                    ),
                })

    return error_logs

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

def retrieve_faiss(database_name: str, query: str, topk: int = 1) -> dict:
    """
    Retrieve a similar case from a FAISS database.
    """

    if database_name not in FAISS_DB_CACHE:
        raise ValueError(f"Database '{database_name}' is not loaded.")

    # Tokenize the query
    query = tokenize(query)

    vectordb = FAISS_DB_CACHE[database_name]
    try:
        docs_and_scores = vectordb.similarity_search_with_score(query, k=topk)
        docs = [d for d, _ in docs_and_scores]
        scores = [s for _, s in docs_and_scores]
    except Exception:
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
