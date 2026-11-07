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

- Non-CAD inputs enter Preparation, Meshing Planner, one of the explicit Standard/Gmsh/Custom Meshing nodes, Solver Planner, Input Writer, local/HPC Runner, separate Mesh/Solver Reviewers, and optional visualization. Keep node adapters thin and mesh backends as services rather than child graphs.
- `careful` (default) runs initial mesh and solver static reviews after their final inputs/scripts are prepared and before native execution or Slurm submission. Both reviews share `max_static_review_retries=10`; a later stage still reviews after the repair budget is exhausted. Passing, no edits, or reaching the limit proceeds to normal execution checks; review-service failures stop. Initial reviews do not restart after runtime repairs. Keep mesh and solver review scopes distinct and reuse the existing task/operation repair services.
- Full workflows retain clean editable inputs, restore them before execution, and synchronize explicit edits plus successful mesh handoffs. Do not capture runtime-expanded inputs as new sources, overwrite a newly handed-off mesh with an older source, or clear imported mesh sources before handoff. Persist phase-labelled review/edit/execution history. Standalone tools retain direct execution and explicit static review.
- Preparation records supplied input facts and preserves read-only `original/` and writable `work/` imports. Target mismatches fail during preparation. Preparation does not select solvers or retrieve solver cases.
- Meshing Planner chooses the backend, sources, file stages and ordered operations from the requirement and actual files. No uploaded path forces a backend or reuse policy. Do not inject startup dictionaries, mesh commands or boundary-condition defaults in execution.
- Mesh planning retrieves target-specific tutorial guidance once per workflow and carries proposed boundary roles with their evidence separately from actual mesh facts. Planner explicitly selects no additional script, existing Allmesh execution, or command-based Allmesh generation on every attempt; file edits do not imply execution. Conversion backends retain their native checks.
- Solver selection and solver-reference retrieval happen after mesh handoff. Solver Planner checks reference/physics compatibility against actual boundaries and the mesh plan, requesting mesh repairs when needed. It adapts mesh-tool startup dictionaries to the physical problem. Allmesh owns mesh operations; Allrun retains physical preparation and solver-parallel setup.
- Handoff requires complete `constant/polyMesh` for a single-region case, or `constant/<region>/polyMesh` for every active region declared by native `constant/regionProperties`, with actual patch/boundary evidence retaining region names. Unresolved time/parallel meshes return to repair without deleting initial fields. Explicit final reconstruction validates all sources before replacing the serial target. Ordinary serial edits retain their mesh.
- Planner and Reviewer declare file stages, create/modify instructions and ordered operations with startup dependencies. Check only declared startup files before execution or HPC submission; intermediate outputs remain native execution responsibilities. Mesh and Solver Reviewers use one structured diagnosis/decision each, separate histories, and the shared repair budget. Keep pending physical tasks across mesh repairs; Meshing Planner decides whether to revisit Solver Planner or continue to Writer. Do not classify work or infer boundary roles from filenames or patch names.
- Repairs share the existing budget (`max_loop=25` by default), graph recursion limit and solver repeated-error/input stopping. Mesh repetition evidence is reviewed within the shared budget. A solver repair with no changes must stop before Runner while retaining errors and outputs.
- Allrun generation and explicit repairs share a script service and failure/logging contract. Local and HPC solver success requires current native completion and result-field evidence, not only a zero script exit. Preserve unrelated imported script content.
- Preserve public CLI/MCP inputs, standalone planning, and output ownership checks. Local execution clears prior runtime outputs only after successful mesh handoff; it does not resume prior results.
- Visualization failure after execution success produces `partial_success`, `termination_reason=visualization_failed`, and a nonzero CLI exit. MCP `run_case` returns its summary and visualization error; standalone visualization raises on failure.
- HPC uses local Slurm commands and shared paths, including native mesh jobs before solver jobs. Empty `squeue` also requires the current attempt's zero exit-code record. Monitoring failure/timeout triggers cancellation and confirmation; an unconfirmed stop terminates with `hpc_job_state_unconfirmed`. Uncertain submissions stop without resubmission. This is not an SSH deployment or job-resume service.
