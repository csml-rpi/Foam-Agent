# AGENTS.md

## 本项目的 feature 开发规则

- 采用满足当前需求的最小代码和功能实现。
- 仅修改与本次 feature 直接相关的代码，不附带无关重构或功能扩展。
- 不额外增加安全测试、边界条件检查或限制，不扩展现有检查规则。
- 每个 feature 只使用一个核心测试，可以在同一个测试函数中参数化相关场景。
- 实现过程中出现具体问题时，可以增加临时诊断测试，并修复对应实现代码。
- 问题修复并验证后，删除额外的临时诊断测试及其临时文件；仅保留该 feature 的核心测试，不删除原有测试。
- 实现前先将已确定的开发规则写入本文件，再修改功能代码。

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

Defined in `src/main.py`. Split the original Planner into two main-graph nodes,
`meshing_planner` and `solver_planner`, with an ordinary `meshing` node between
them. Do not create a Meshing child graph or register mesh backends as nodes.
Node adapters remain thin wrappers around service functions. First entry into
`meshing_planner` prepares generated-case metadata, the output directory and
references through shared preparation functions; imported cases reuse their
existing context. Preparation is not a separate graph node and is not repeated
on mesh repair. Input remains a single user requirement.

```
START -> [CASE_IMPORT for imported cases] -> MESHING_PLANNER -> MESHING
      -> SOLVER_PLANNER -> INPUT_WRITER -> LOCAL/HPC RUNNER -> [VISUALIZATION] -> END

Solver errors -> REVIEWER -> INPUT_WRITER for solver repairs
                         -> MESHING_PLANNER -> MESHING for mesh repairs -> INPUT_WRITER
```

Routing decisions are implemented in `src/router_func.py`. A complete imported
case with `Allrun`, no detected physical issues, and no explicit prompt or custom
mesh skips physical file planning. It still visits Meshing Planner to plan mesh reuse and
split its script; Input Writer preserves physical files when no edits are needed.
Other imported cases use targeted physical planning; missing physical information
that cannot be inferred produces a planning failure. Explicit target conflicts
end with `case_target_mismatch`, preserving the existing behavior.

`solver_planner` consumes the final mesh metadata and plans physical/solver files
after successful Meshing. The `solver_planned` state flag records successful
initial planning; subsequent mesh repairs return directly to Input Writer and
preserve the Reviewer's pending file changes. Existing imported-target checks
run before Meshing. CLI arguments and standalone planning-service calls retain
their current behavior; no new safety or boundary-condition checks are added.

`meshing_planner` selects a backend, plans mesh files and ordered commands, and
extracts a solver-only remainder from an imported `Allrun` using the existing LLM
approach. Standard mesh dictionaries, geometry, generation, conversion, checks,
mesh modifications and repair run inside Meshing. Gmsh-specific checks stay in
the Gmsh backend; external-mesh conversion/checking and case `polyMesh` reuse
share `imported_meshing`. Existing `polyMesh` is a reuse condition, not a fourth
backend. Case import copies the whole case into `original/` and `work/`; it does
not split the case's directories by purpose.

The ordinary Meshing node selects a backend service with `if`/`elif` and calls
the shared execution service. Internal mesh retries stay in that service loop,
which reuses `plan_meshing()` directly without re-entering a graph node.

Mesh shell operations run through `Allmesh`; solver-only `Allrun` preserves solver
arguments, physical field preparation and solver-parallel decomposition. Mesh
repairs replan the backend and ordered commands from the Reviewer's feedback,
including switching from mesh reuse to regeneration when no file edits are needed.
The pending solver repair plan is preserved across these internal mesh retries.
Mesh utilities can create a minimal bootstrap `controlDict` when missing; existing
case solver settings are preserved. Mesh metadata contains the final `polyMesh`
path and actual patch names/types/face counts for subsequent file generation.
The physical boundary conditions in field files such as `0/U` remain the solver
stage's responsibility. Shared file-writing/review helpers and metadata readers
may be used by other nodes; mesh execution and mesh file repairs belong to Meshing.

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
    imported_case_node.py
  services/            # Business logic (where the real work happens)
    plan.py            # Case planning and analysis
    input_writer.py    # OpenFOAM file generation via LLM + RAG
    meshing.py         # Mesh planning, standard execution and internal repair
    mesh.py            # Existing Gmsh/external-mesh backend implementations
    mesh_context.py    # Mesh classification and final patch metadata
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
2. **Error correction loop**: Runner detects errors -> Reviewer diagnoses via LLM -> Input Writer rewrites solver files, or Meshing rewrites/regenerates mesh -> re-run. Meshing-stage errors use the same review service internally. Existing review limits and repeated-error fingerprints are retained; no new safety or boundary checks are added.
3. **RAG retrieval**: FAISS indices built from OpenFOAM tutorials provide reference cases to the input writer.
4. **Two generation modes** (`config.input_writer_generation_mode`):
   - `sequential_dependency` (default): Files generated in order with cross-file context.
   - `parallel_no_context`: All files generated independently (faster, relies on retry loop).
5. **Target-scoped corpora**: Foundation v10 and native ESI v2006 use separate raw and FAISS data under `database/<target>/`.

### Meshing and solver-planning PR verification

Treat Feature1 and Feature2 as one PR with one core parameterized test function,
`test_mesh_before_solver_workflow` in `tests/test_meshing_workflow.py`.
Keep nine scenarios: standard mesh, Gmsh, uploaded mesh, imported polyMesh reuse,
mesh-file repair, reuse-to-regeneration repair, solver-only repair, mixed
mesh/solver repair, and the existing target-mismatch behavior.
Check mesh-before-solver ordering, final patch context, one-time preparation,
Allmesh/Allrun separation, imported-file preservation and pending solver repairs.
Run `python -m pytest tests/test_meshing_workflow.py -q`. The test runs the actual
main graph and services, with external LLM, retrieval, Gmsh subprocess
and OpenFOAM execution mocked. It does not establish real CFD convergence or live
LLM behavior. Consolidate the added `tests/test_solver_planner.py` into this test;
retain pre-existing repository tests. Remove temporary diagnostic tests and files
after fixing the corresponding implementation.

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
