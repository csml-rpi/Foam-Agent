# Working on Foam-Agent

Foam-Agent automates CFD simulation through LangGraph, OpenFOAM tutorial retrieval, and model-backed services. See [README](README.md) for setup, [usage](docs/usage.md) for configuration, and [MCP](docs/mcp.md) for tool contracts.

## Development Commands

```bash
conda env create -f environment.yml
conda activate FoamAgent
pip install --no-deps -e .
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
pytest tests/ -v
foamagent-mcp --transport http --host 0.0.0.0 --port 7860
```

The Conda environment uses Python 3.12.9. Source the intended OpenFOAM runtime before execution.

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

- Generated and imported cases share Planner, Meshing, Input Writer, local/HPC Runner, Reviewer, and visualization nodes. Complete imports with `Allrun`, no detected issues, and no explicit prompt/custom mesh skip file planning; LLM resources and routing calls remain required.
- Detect target conflicts deterministically, then route them to Reviewer. Missing files can enter planning; missing physical information that cannot be inferred produces a planning failure.
- Preserve read-only `original/` and writable `work/` imports and output ownership checks. Local execution clears old runtime outputs and nonzero times in `work/`; it does not resume prior results.
- Repairs use one error analysis followed by direct file rewriting. Mesh repairs return to Meshing before execution. Default `max_loop=25` is also bounded by graph recursion and repeated error/input fingerprints. A repair with no changes and no pending mesh action must stop before Runner while retaining errors and outputs.
- Visualization failure after execution success produces `partial_success`, `termination_reason=visualization_failed`, and a nonzero CLI exit. MCP `run_case` returns its summary and visualization error; standalone visualization raises on failure.
- HPC uses local Slurm commands and shared case paths. Empty `squeue` means completion without `sacct`/exit-code validation; a monitoring timeout can enter repair while the original job is active. Preserve these documented limits unless explicitly changing that behavior.
