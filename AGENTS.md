# AGENTS.md

> This file helps AI agents (Codex, Cursor, Claude Code, Copilot, etc.) understand and work with this codebase.

## What is Foam-Agent?

Foam-Agent is a multi-agent framework that automates CFD (Computational Fluid Dynamics) simulations in **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) from natural language prompts. It uses LangChain/LangGraph for orchestration, FAISS for RAG-based tutorial retrieval, and supports multiple LLM providers (OpenAI, Anthropic, Bedrock, Ollama).

> **Important:** Foundation v10 remains the default. `FOAMAGENT_OPENFOAM_TARGET=esi-v2006` selects a separate native ESI/OpenCFD v2006 path with its own tutorials and runtime checks. The historical `FOAMAGENT_OPENFOAM_FORK=esi` path remains a best-effort v10-to-ESI translator and is not the native v2006 target.

## Build and Run

```bash
# Environment setup
conda env create -n FoamAgent -f environment.yml
conda activate FoamAgent

# Run a simulation
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt

# Run with custom mesh
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt --custom_mesh_path ./mesh.msh

# Run tests
pytest tests/ -v

# Start MCP server
python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

Requires a sourced OpenFOAM runtime (`$WM_PROJECT_DIR` must be set): Foundation v10 for the default path, or ESI/OpenCFD v2006 when `FOAMAGENT_OPENFOAM_TARGET=esi-v2006`. Python 3.12.9 via Conda.

## Architecture

### Workflow Pipeline (LangGraph StateGraph)

Defined in `src/main.py`. All non-CAD inputs enter `preparation`, followed by
`meshing_planner`, an ordinary `meshing` node, and `solver_planner`. Node adapters
remain thin service wrappers; mesh backends are not graph nodes or child graphs.

```
START -> PREPARATION -> MESHING_PLANNER -> MESHING -> SOLVER_PLANNER
      -> INPUT_WRITER -> LOCAL/HPC RUNNER -> [VISUALIZATION] -> END

Solver errors -> REVIEWER -> INPUT_WRITER for solver repairs
                         -> MESHING_PLANNER -> MESHING for mesh repairs -> SOLVER_PLANNER or INPUT_WRITER
```

Preparation installs working copies of prompt, external material and imported
case inputs and records actual file facts. It does not select a simulation
solver, retrieve solver cases, or force mesh reuse/regeneration. Existing case
import services retain target detection, original/work/report copies and target
mismatch failures. Imported cases enter through Preparation; the old imported-case
node adapter has been removed.

Meshing Planner selects a backend, source material, files and ordered commands
from the requirement and available files. No uploaded path or polyMesh directory
forces a backend or reuse policy. Standard OpenFOAM utilities, generated Gmsh,
uploaded Gmsh .msh files and meshes inside imported cases use the same graph.
External .geo scripts and standalone polyMesh directories are outside this update;
SpaceClaim/Fluent input is deferred. Planners use prepared input paths, case file
listings and contents; mesh handoff includes actual patches and boundary text.

Meshing generates/edits mesh inputs, executes Allmesh or converter tools and
reads final patches. Mesh failures return to the main-graph Reviewer, then
Meshing Planner and Meshing, using the existing shared repair budget. Meshing Planner
decides which utility startup dictionaries to create or modify through normal file tasks;
execution does not inject defaults. Converter staging files stay separate from simulation settings.

Solver selection and solver-reference retrieval happen in Solver Planner after
mesh handoff. It selects physical/solver file changes and extracts a solver-only
Allrun from an imported script, preserving field preparation and solver-parallel
setup. Allmesh owns mesh operations. The actual mesh path, patches
and available conversion/check evidence are passed to the solver
planner, writer and reviewer; unknown values are not filled by name heuristics.

Planner and Reviewer declare file stages and create/modify instructions. Filename
heuristics no longer classify or silently discard tasks. After mesh repair, the mesh
planner assesses actual results and decides whether to revisit Solver Planner or
continue to Writer; pending physical tasks survive either route.
Retain the three supported input modes and the standalone MCP planning service.
Imported-case planning only runs after Preparation and Meshing; target checks belong
to Preparation, with no legacy all-stage planning branch.
No new safety or boundary-condition policies or quality thresholds are added.

### Directory Structure

```
src/
  main.py              # LangGraph workflow definition and entry point
  config.py            # Config dataclass with env var overrides
  utils.py             # GraphState (TypedDict), LLMService (unified LLM interface)
  models.py            # Pydantic models for generated files and plans
  router_func.py       # LLM-based routing decisions
  logger.py            # Structured XML-tagged logging
  nodes/               # LangGraph node functions (thin wrappers calling services)
    meshing_planner_node.py
    solver_planner_node.py
    input_writer_node.py
    meshing_node.py     # Ordinary node selecting one of three backend services
    local_runner_node.py
    hpc_runner_node.py
    reviewer_node.py
    visualization_node.py
    preparation_node.py
  services/            # Business logic (where the real work happens)
    preparation.py     # Shared input preparation and imported-case materialisation
    plan.py            # Solver metadata, references and file planning
    input_writer.py    # OpenFOAM file generation via LLM + RAG
    meshing.py         # Mesh planning and single-pass standard execution
    mesh.py            # Gmsh generation and uploaded mesh conversion backends
    run_local.py       # Local OpenFOAM execution
    run_hpc.py         # HPC job submission
    review.py          # Error diagnosis and fix planning
    visualization.py   # PyVista-based post-processing
    case_import.py     # Existing-case copy, target detection, and context scan
    case_paths.py      # Shared generated path validation
    output_safety.py   # Output ownership and overwrite protection
  mcp/                 # FastMCP server exposing workflow as tools
  translation/         # Legacy Foundation-to-ESI translation compatibility
  openfoam_target.py   # Native target selection and runtime/corpus checks
database/
  foundation-v10/      # Foundation v10 raw data and FAISS indices
  esi-v2006/           # Native ESI/OpenCFD v2006 raw data and FAISS indices
  script/               # Shared corpus parsers and FAISS builders
tests/                 # Focused pytest regressions and manual integration scripts
docker/                # Dockerfile for containerized deployment
```

### Key Abstractions

- **`GraphState`** (`src/utils.py`): TypedDict threaded through all workflow nodes. Contains user requirement, case metadata, generated files, error logs, loop count.
- **`LLMService`** (`src/utils.py`): Unified LLM interface supporting OpenAI, Anthropic, Bedrock, Ollama. Provides `invoke()` and `structure_output()` (Pydantic-validated).
- **`Config`** (`src/config.py`): Config dataclass. Environment overrides are implemented for LLM provider/model, embedding provider/model, OpenFOAM fork, and native target. Other fields require Python configuration or an exposed CLI/API argument.
- **Pydantic models** (`src/models.py`): `FoamPydantic`/`FoamfilePydantic` for generated files, `RewritePlan` for error fixes, `CaseSummaryModel` for case metadata.
- **Native targets** (`src/openfoam_target.py`): Foundation v10 is the default; native ESI/OpenCFD v2006 is opt-in and uses an isolated corpus and runtime guard.
- **Existing-case workflow** (`src/services/case_import.py`, `src/services/plan.py`): Materialises read-only `original/` and writable `work/` trees, builds case context, plans targeted changes through graph routes, uses the shared file-rewrite repair loop.

### Design Patterns

1. **Service-oriented**: Nodes in `src/nodes/` are thin orchestration wrappers. All logic lives in `src/services/`.
2. **Error correction loop**: Runner detects errors -> Reviewer diagnoses via LLM -> Input Writer rewrites solver files, or Meshing rewrites/regenerates mesh -> re-run. Meshing-stage errors return to the main-graph Reviewer and then Meshing Planner. Existing review limits are retained. Solver-review repeated-error stopping is unchanged; mesh review receives repetition evidence and decides whether to continue within the shared budget. No new safety or boundary checks are added.
3. **RAG retrieval**: FAISS indices built from OpenFOAM tutorials provide reference cases to the input writer.
4. **Two generation modes** (`config.input_writer_generation_mode`):
   - `sequential_dependency` (default): Files generated in order with cross-file context.
   - `parallel_no_context`: All files generated independently (faster, relies on retry loop).
5. **Target-scoped corpora**: Foundation v10 and native ESI v2006 use separate raw and FAISS data under `database/<target>/`.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `FOAMAGENT_MODEL_PROVIDER` | LLM provider: `openai`, `openai-codex` (default), `anthropic`, `bedrock`, `ollama`, `deepseek` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier (e.g., `claude-opus-4-6`, `gpt-5.3-codex`) |
| `FOAMAGENT_EMBEDDING_PROVIDER` | Embedding backend: `openai`, `huggingface`, `ollama` |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model (default: `Qwen/Qwen3-Embedding-0.6B`) |
| `FOAMAGENT_OPENFOAM_FORK` | Legacy fork routing: `foundation` or generic translated `esi` |
| `FOAMAGENT_OPENFOAM_TARGET` | Explicit native target: `foundation-v10` or `esi-v2006` |
| `OPENAI_API_KEY` | Required for `openai` provider |
| `ANTHROPIC_API_KEY` | Required for `anthropic` provider |
| `DEEPSEEK_API_KEY` | Required for `deepseek` provider |
| `WM_PROJECT_DIR` | OpenFOAM installation path (required at runtime) |

## Common Tasks

### Adding a new LLM provider
Extend `LLMService` in `src/utils.py`. Follow the pattern of existing providers (each has an `if` branch in the constructor).

### Adding a new workflow node
1. Create service logic in `src/services/`.
2. Create a thin node wrapper in `src/nodes/`.
3. Wire it into the StateGraph in `src/main.py`.

### Modifying file generation
The input writer logic is in `src/services/input_writer.py`. It uses RAG context from FAISS indices and LLM calls to generate OpenFOAM configuration files.

### Rebuilding FAISS indices
Rebuild when the corpus changes or the desired embedding model's indices are missing. Match the builder arguments to the runtime embedding configuration:
```bash
python init_database.py --openfoam_path "$WM_PROJECT_DIR" \
  --openfoam_target foundation-v10 \
  --embedding_provider huggingface --embedding_model Qwen/Qwen3-Embedding-0.6B --force
```

Use `esi-v2006` with a matching runtime to build that corpus. Runtime defaults are Hugging Face/Qwen 0.6B, but the standalone FAISS builders default to OpenAI/`text-embedding-3-small`; `init_database.py` checks Qwen 0.6B completeness when no model is specified. Always specify both embedding arguments. Its `--database_path` is the target directory, whereas `Config.database_path` is the parent containing target directories.

## Things to Watch Out For

- **Do not regenerate FAISS indices** unless you have a specific reason. Hydrate Git LFS assets and select indices matching the configured embedding model. The current ESI corpus includes Qwen 0.6B and OpenAI small indices, not Qwen 8B.
- **Foundation OpenFOAM v10 must be sourced** for the default path. Native `esi-v2006` requires an ESI/OpenCFD v2006 environment (`WM_PROJECT_VERSION=v2006`). Both targets use and validate their own corpus under `<database_path>/<target>/`.
- **The error correction loop** can run up to 25 iterations. When modifying the reviewer or input writer, consider the impact on convergence.
- **`GraphState`** defines the state schema. LangGraph merges node-returned updates; some nodes also mutate nested objects such as `Config`. Do not assume every node shares one unchanged dictionary instance.
- **Visualization failure after simulation success** produces `partial_success`, `termination_reason=visualization_failed`, and the English message `Simulation completed successfully, but visualization failed.` The CLI exits nonzero; MCP `run_case` includes the summary and visualization error. A standalone MCP `visualization` failure raises an error.
- **HPC monitoring is basic**: it runs Slurm commands in the agent's environment, treats an empty `squeue` result as completed, and does not query `sacct` or validate the final exit code. Monitoring timeout returns the last observed state and may enter the existing repair/resubmission loop. It is not an SSH deployment or job-resume service.
