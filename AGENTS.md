# Working on Foam-Agent

Foam-Agent automates CFD simulation through LangGraph, OpenFOAM tutorial retrieval, and model-backed services. See [README](README.md) for setup, [usage](docs/usage.md) for configuration, and [MCP](docs/mcp.md) for tool contracts.

## Development Commands

```bash
conda env create -f environment.yml
conda activate FoamAgent
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints.txt -e '.[mesh]'
python scripts/doctor.py
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
pytest tests/ -v
foamagent-mcp --transport http --host 0.0.0.0 --port 7860
```

The Conda file bootstraps Python 3.12.9 and pip; application dependencies are installed from pyproject.toml using constraints.txt. Docker builds install CPU Torch and bundle the default embedding weights. Source the intended OpenFOAM runtime before local execution.

## Code Map

| Location | Responsibility |
|---|---|
| `src/main.py`, `src/router_func.py` | Graph construction, initialization, and routing |
| `src/nodes/` | Thin orchestration wrappers |
| `src/services/` | Planning, file generation, meshing, execution, review, visualization, and case import |
| `src/config.py`, `src/openfoam_target.py` | Configuration and target/runtime/corpus contracts |
| `src/utils.py` | LLM clients, RAG, graph state, and shared file/process helpers |
| `src/models.py` | Import manifests and structured existing-case plans |
| `src/mcp/` | Tool schemas, server, and shared startup logic |
| `src/translation/` | Legacy Foundation-to-ESI compatibility |
| `database/<target>/` | Target-specific raw tutorials and FAISS indices |
| `database/script/`, `scripts/` | Corpus builders and verification tools |
| `examples/`, `docs/`, `tests/` | Sample inputs, user guides, and automated regressions |

Generated-file models live in `utils.py`; planning/rewrite models are defined in their services. `GraphState` is the schema merged by LangGraph. Nodes may mutate nested `Config` objects, so do not assume state dictionaries are shared unchanged.

## Maintenance Rules

- Keep business logic in services and graph coordination in nodes/routes. Extend `LLMService` for providers; wire new nodes into `src/main.py`.
- Keep generated cases, evaluation data, logs, reports, and one-off debugging checks out of tracked source. Retain reusable tests for user-visible behavior. Put examples in `examples/` and user instructions in `docs/`; avoid PR-specific documents.
- Preserve public CLI arguments and MCP request/response schemas. `foambench_main.py` is the public CLI wrapper. MCP payloads use a top-level `request` object.
- Model/provider, embedding, fork, and target have explicit environment overrides. Other fields require Python configuration or exposed CLI/API arguments; do not assume generic environment mapping.
- Optional provider dependencies must not be required to import or run other providers.

## OpenFOAM and Corpus Contracts

Foundation OpenFOAM v10 is the default. `FOAMAGENT_OPENFOAM_TARGET=esi-v2006` selects native ESI/OpenCFD v2006 with isolated tutorials, native conventions, runtime checks, and a matching Docker image. `FOAMAGENT_OPENFOAM_FORK=esi` is the separate legacy translation path; native v2006 must bypass it.

Hydrate Git LFS assets and select indices matching the embedding model. Do not regenerate FAISS indices without a specific need. Foundation includes Qwen 0.6B/8B and OpenAI small indices; ESI includes Qwen 0.6B and OpenAI small. When rebuilding, always pass both embedding provider and model; see [corpus instructions](docs/usage.md#openfoam-targets-and-corpora). The initializer's `--database_path` is a target directory, while `Config.database_path` is its parent.

## Workflow Contracts

- Generated and imported cases share Preparation, Meshing Planner, Standard/Gmsh/Custom Meshing, Solver Planner, Input Writer, local/HPC Runner, separate Mesh/Solver Reviewers, and visualization. Solver selection follows mesh handoff. Allmesh owns mesh operations; Allrun owns physical preparation and solving.
- `execution_mode=careful` (default) generates in dependency order and statically reviews before initial mesh and solver execution. `fast` generates independently in parallel without static review; an explicit generation mode overrides generation only. Both stages share 10 static repair rounds, independent of runtime `max_loop`; each stage still reviews after the allowance is exhausted. Reaching the limit or making no edits proceeds to execution checks. Review service failures stop; runtime repairs do not restart initial review. Persist review/edit/runtime history; standalone MCP tools require explicit `static_review`.
- Full workflows edit canonical inputs and restore them before runs and mesh retries. Synchronize successful mesh handoffs; do not capture execution outputs as new source inputs. Preserve complete assets, diagnostics and final case paths; standalone execution remains direct. Keep workspace logic in services and source paths in graph state.
- Detect target conflicts deterministically and stop with `case_target_mismatch`. Missing files can enter planning; missing physical information that cannot be inferred produces a planning failure.
- Preserve read-only `original/` and writable `work/` imports and output ownership checks. Local execution clears old runtime outputs and nonzero times in `work/` after successful mesh handoff; it does not resume prior results.
- Mesh/Solver Reviewers return structured decisions with file tasks and operations. Mesh repairs return to Meshing Planner while retaining pending physical tasks. Default runtime `max_loop=25` is also bounded by graph recursion and repeated error/input stopping. A solver repair with no changes must stop before Runner while retaining errors and outputs. Local/HPC success requires current native completion and result-field evidence.
- Visualization failure after execution success produces `partial_success`, `termination_reason=visualization_failed`, and a nonzero CLI exit. MCP `run_case` returns its summary and visualization error; standalone visualization raises on failure.
- HPC uses local Slurm commands and shared case paths. Empty `squeue` also requires the current attempt's zero exit-code record. Monitoring failure/timeout triggers cancellation and stop confirmation; an unconfirmed stop ends with `hpc_job_state_unconfirmed`. Uncertain submissions stop without resubmission.
