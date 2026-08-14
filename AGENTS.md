# AGENTS.md

> This file helps AI agents (Codex, Cursor, Claude Code, Copilot, etc.) understand and work with this codebase.

## What is Foam-Agent?

Foam-Agent is a multi-agent framework that automates CFD (Computational Fluid Dynamics) simulations in **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) from natural language prompts. It uses LangChain/LangGraph for orchestration, FAISS for RAG-based tutorial retrieval, and supports multiple LLM providers (OpenAI, Anthropic, Bedrock, Ollama).

> **Important:** All generated case files, dictionary names, and solver binaries follow Foundation OpenFOAM v10 conventions. ESI OpenFOAM (openfoam.com, e.g., v2312, v2406, v2512) is **not compatible**.

## Build and Run

```bash
# Environment setup (uv). Core is intentionally lightweight; add the extras you need.
git lfs install --local && git lfs pull        # database/ is stored with Git LFS
uv sync --extra rag-local --extra direct-api --extra viz

# Run a simulation
uv run python foambench_main.py --output ./output --prompt_path ./user_requirement.txt

# Run with custom mesh
uv run python foambench_main.py --output ./output --prompt_path ./user_requirement.txt --custom_mesh_path ./mesh.msh

# Run tests. Unit tests need no credentials, network, Docker, or LFS content.
uv run pytest -m "not integration" -q
uv run ruff check .

# Start MCP server
uv run python -m foamagent.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

Requires **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) at runtime. Either source it natively (`$WM_PROJECT_DIR` must be set) or set `FOAMAGENT_OPENFOAM_RUNTIME=docker` to run solvers inside a container. ESI OpenFOAM (openfoam.com) file generation is best-effort translation only.

## Architecture

### Workflow Pipeline (LangGraph StateGraph)

Defined in `src/foamagent/main.py`:

```
PLANNER -> [mesh routing] -> MESHING (if needed) -> INPUT_WRITER -> [HPC/local routing]
-> RUNNER -> [error check] -> REVIEWER -> INPUT_WRITER (retry loop, max 25 iterations)
-> VISUALIZATION (if requested) -> END
```

All routing decisions (mesh type, HPC vs local, visualization) are LLM calls in `src/foamagent/router_func.py`.

### Directory Structure

```
src/foamagent/          # the importable package (`import foamagent`)
  main.py              # LangGraph workflow definition and entry point
  config.py            # Config dataclass with env var overrides
  utils.py             # GraphState (TypedDict), LLMService (unified LLM interface)
  models.py            # Pydantic models for generated files and plans
  router_func.py       # LLM-based routing decisions
  logger.py            # Structured XML-tagged logging
  nodes/               # LangGraph node functions (thin wrappers calling services)
    planner_node.py
    input_writer_node.py
    meshing_node.py
    local_runner_node.py
    hpc_runner_node.py
    reviewer_node.py
    visualization_node.py
  services/            # Business logic (where the real work happens)
    plan.py            # Case planning and analysis
    input_writer.py    # OpenFOAM file generation via LLM + RAG
    mesh.py            # Mesh generation (blockMesh / Gmsh conversion)
    run_local.py       # Local OpenFOAM execution
    run_hpc.py         # HPC job submission
    review.py          # Error diagnosis and fix planning
    visualization.py   # PyVista-based post-processing
  paths.py             # Resolves database/ and runs/ (FOAMAGENT_ROOT overrides)
  mcp/                 # FastMCP server exposing workflow as tools
database/
  faiss/               # Pre-built FAISS vector indices (do NOT regenerate unless necessary)
  raw/                 # Raw OpenFOAM tutorial data
tests/                 # unit tests: no credentials, network, Docker, or LFS content
scripts/manual/        # end-to-end scripts that DO need credentials; run by hand
docker/                # Dockerfile for containerized deployment
```

### Key Abstractions

- **`GraphState`** (`src/foamagent/utils.py`): TypedDict threaded through all workflow nodes. Contains user requirement, case metadata, generated files, error logs, loop count.
- **`LLMService`** (`src/foamagent/utils.py`): Unified LLM interface supporting OpenAI, Anthropic, Bedrock, Ollama. Provides `invoke()` and `structure_output()` (Pydantic-validated).
- **`Config`** (`src/foamagent/config.py`): Global config dataclass. Every field can be overridden via `FOAMAGENT_*` env vars.
- **Pydantic models** (`src/foamagent/models.py`): `FoamPydantic`/`FoamfilePydantic` for generated files, `RewritePlan` for error fixes, `CaseSummaryModel` for case metadata.

### Design Patterns

1. **Service-oriented**: Nodes in `src/foamagent/nodes/` are thin orchestration wrappers. All logic lives in `src/foamagent/services/`.
2. **Error correction loop**: Runner detects errors -> Reviewer diagnoses via LLM -> Input Writer rewrites targeted files -> re-run (up to `max_loop` iterations).
3. **RAG retrieval**: FAISS indices built from OpenFOAM tutorials provide reference cases to the input writer.
4. **Two generation modes** (`config.input_writer_generation_mode`):
   - `sequential_dependency` (default): Files generated in order with cross-file context.
   - `parallel_no_context`: All files generated independently (faster, relies on retry loop).

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `FOAMAGENT_MODEL_PROVIDER` | LLM provider: `openai`, `openai-codex`, `anthropic`, `bedrock`, `ollama` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier (e.g., `claude-opus-4-6`, `gpt-5.3-codex`) |
| `FOAMAGENT_EMBEDDING_PROVIDER` | Embedding backend: `openai`, `huggingface`, `ollama` |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model (default: `Qwen/Qwen3-Embedding-0.6B`) |
| `OPENAI_API_KEY` | Required for `openai` provider |
| `ANTHROPIC_API_KEY` | Required for `anthropic` provider |
| `WM_PROJECT_DIR` | OpenFOAM installation path (required for `native` runtime) |
| `FOAMAGENT_OPENAI_BASE_URL` | OpenAI-compatible endpoint (OpenRouter, vLLM, LiteLLM, ...) |
| `FOAMAGENT_OPENFOAM_RUNTIME` | `native` (default) or `docker` |
| `FOAMAGENT_OPENFOAM_IMAGE` / `_BASHRC` | Image and bashrc path for the `docker` runtime |
| `FOAMAGENT_ROOT` | Overrides where `database/` and `runs/` are looked up |
| `FOAMAGENT_LOG_LEVEL` | Log verbosity (default `INFO`). Logs go to stderr |

## Common Tasks

### Adding a new LLM provider
Extend `LLMService` in `src/foamagent/utils.py`. Follow the pattern of existing providers (each has an `if` branch in the constructor).

### Adding a new workflow node
1. Create service logic in `src/foamagent/services/`.
2. Create a thin node wrapper in `src/foamagent/nodes/`.
3. Wire it into the StateGraph in `src/foamagent/main.py`.

### Modifying file generation
The input writer logic is in `src/foamagent/services/input_writer.py`. It uses RAG context from FAISS indices and LLM calls to generate OpenFOAM configuration files.

### Rebuilding FAISS indices
Only needed if OpenFOAM tutorials change:
```bash
python init_database.py --openfoam_path $WM_PROJECT_DIR --force
```

## Things to Watch Out For

- **Do not regenerate FAISS indices** unless you have a specific reason. The pre-built indices in `database/faiss/` are correct and ready to use.
- **Foundation OpenFOAM v10 must be sourced** for any simulation execution. Without `$WM_PROJECT_DIR`, the runner nodes will fail. ESI OpenFOAM is not compatible.
- **The error correction loop** can run up to 25 iterations. When modifying the reviewer or input writer, consider the impact on convergence.
- **`GraphState` is mutable** and passed by reference through the entire pipeline. Be careful about unintended side effects when modifying state fields.
