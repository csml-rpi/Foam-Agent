# Usage Guide

## Local Installation

Use this path inside an existing container or on a Linux host with a matching
OpenFOAM installation. Python 3.12 is the release environment; the package supports
Python 3.10 and later. A GPU is optional.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
git lfs install
git lfs pull --include="database/foundation-v10/raw/*,database/foundation-v10/faiss/Qwen_Qwen3-Embedding-0.6B/**,examples/tandem-wing/*.msh"
python3 scripts/verify_target_corpus.py --target foundation-v10
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Linux amd64, install the CPU Torch wheel first to avoid downloading CUDA libraries:

```bash
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints.txt -e '.[mesh]'
python -m pip check
```

For another platform or GPU acceleration, install the appropriate Torch build,
then install the project with the same constraints. Omit `[mesh]` if you do not
use generated/custom Gmsh meshes; that extra needs system libraries such as
`libglu1-mesa`. Docker includes them. Optional providers use `.[bedrock]`,
`.[ollama]`, or `.[all]`; pass `-c constraints.txt` when installing extras too.

If you prefer Conda, replace the venv step with `conda env create -f environment.yml`
and `conda activate FoamAgent`, then run the same pip commands. The Conda file
only creates Python and pip; `pyproject.toml` is the application dependency source.
`constraints.txt` pins compatible versions, rather than exporting an entire workstation.

Install and source Foundation OpenFOAM v10 for
the default target, or ESI/OpenCFD v2006 with its matching corpus. Adapt the path:

```bash
source /opt/openfoam10/etc/bashrc
cp .env.example .env
# Edit .env with your credentials before loading it.
set -a
source .env
set +a
python scripts/doctor.py --check-embedding --check-llm
bash examples/quickstart/run.sh "$PWD/output/quickstart"
```

Commands assume the repository root. `.env` is loaded by Docker's `--env-file` or
explicitly by your shell, not automatically by the Python CLI. The editable install
registers `foamagent-mcp`. Use `python foambench_main.py --help` for options.
Each `--output` is a dedicated task directory; use a new directory for a new task.
`--overwrite_output` replaces an existing Foam-Agent-owned output.

## Configuration

| Environment variable | Purpose | Default |
|---|---|---|
| `FOAMAGENT_MODEL_PROVIDER` | `openai`, `openai-codex`, `anthropic`, `bedrock`, `ollama`, or `deepseek` | `openai-codex` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier for the selected provider | `gpt-5.3-codex` |
| `FOAMAGENT_EMBEDDING_PROVIDER` | `openai`, `huggingface`, or `ollama` | `huggingface` |
| `FOAMAGENT_EMBEDDING_MODEL` | Model identity matching the selected FAISS indices | `Qwen/Qwen3-Embedding-0.6B` |
| `FOAMAGENT_EMBEDDING_MODEL_PATH` | Local directory containing that same Hugging Face model | Unset |
| `FOAMAGENT_STRUCTURED_OUTPUT_METHOD` | `auto`, `json_schema`, `function_calling`, `json_mode`, or `json_prompt` | `auto` |
| `FOAMAGENT_OPENFOAM_TARGET` | Explicit native target: `foundation-v10` or `esi-v2006` | Unset; Foundation path |
| `FOAMAGENT_OPENFOAM_FORK` | Legacy fork routing: `foundation` or translated `esi` | `foundation` |

Other `Config` fields use Python configuration or an exposed CLI/API parameter. Defaults include `max_loop=25`, `recursion_limit=158`, and `max_time_limit=3600` seconds.

### Execution modes and static review

`execution_mode="careful"` is the default for generated and imported cases. It
shares generated files as context and uses the configured LLM to review inputs
before initial mesh and solver execution or HPC submission. `fast` generates files independently
in parallel and skips static review. Both modes retain runtime review and repair.

```bash
# Default: dependency-aware generation and initial static review
python foambench_main.py --prompt_path requirements.txt --output output/careful

# For inexpensive runs
python foambench_main.py --prompt_path requirements.txt --output output/fast --execution_mode fast
```

`input_writer_generation_mode` defaults to `None`, following the preset. Set it to
`sequential_dependency` or `parallel_no_context` through the corresponding CLI flag
or Python `Config` to override generation independently:

```python
import sys
sys.path.insert(0, "src")  # Run from the repository root.
from config import Config

config = Config(execution_mode="careful", input_writer_generation_mode="parallel_no_context")
```

Static review requests only definite, necessary corrections to syntax,
consistency, or explicit user requirements. It reuses the retrieved references;
tutorial differences alone do not justify edits. Unconfirmed claims and questions
requiring execution are warnings. Review does not establish convergence or
physical correctness; omitted binary, oversized, hidden, or linked files are listed.

`max_static_review_retries` defaults to `10`, shared by initial mesh and solver
reviews and independent of runtime `max_loop`. Execution proceeds when review passes, no inputs or operations change, or the repair
limit is reached; unresolved findings remain in the report. Each stage still
reviews after the budget is exhausted; `0` reviews both stages without repairing. A failed service call or invalid review stops the workflow.
After each stage's first Run, repairs proceed without repeating static review. The initial
static phase receives its own bounded allowance above `recursion_limit`.

Reports live in `CASE/.foamagent/` for generated cases or `OUTPUT/report/` for
imports. `mesh_static_review.json` and `solver_static_review.json` retain each
stage's last initial review; `static_review.json` aggregates them;
`review_history.jsonl` stores findings, verification decisions, edit diffs and
runtime feedback; `static_review_references.json` stores the references used.
Standalone MCP tools require an explicit `static_review` call;
`run_case` follows the complete workflow.

### Authentication

Set credentials before model-backed operations; services initialize model clients only when needed.

| Provider | Credentials |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `bedrock` | AWS credentials and a supported model/inference profile |
| `ollama` | Optional Python packages, a local Ollama service and model |
| `openai-codex` | Codex subscription OAuth cache |

For API-key usage, select both provider and model:

```bash
export FOAMAGENT_MODEL_PROVIDER=anthropic
export FOAMAGENT_MODEL_VERSION=claude-sonnet-4-6
export ANTHROPIC_API_KEY=your-key-here
```

For a Codex subscription, run `codex login` on the host and select **Sign in with ChatGPT**. Foam-Agent searches `$CODEX_HOME/auth.json`, `~/.codex/auth.json`, then `~/.clawdbot/agents/main/agent/auth-profiles.json`. Mount the cache read-only when using Docker:

```bash
docker run -it \
  -e FOAMAGENT_MODEL_PROVIDER=openai-codex \
  -e FOAMAGENT_MODEL_VERSION=gpt-5.3-codex \
  -v ~/.codex/auth.json:/root/.codex/auth.json:ro \
  foamagent:local
```

### Ollama

Install the Ollama extra in the same Python environment and select your installed model:

```bash
python -m pip install -c constraints.txt -e '.[ollama]'
export FOAMAGENT_MODEL_PROVIDER=ollama
export FOAMAGENT_MODEL_VERSION=your-installed-model
```

Install Ollama separately, start its service at `http://localhost:11434`, and
download the selected model before running Foam-Agent. The Python extra installs
client packages only. Default Docker builds include the core and mesh dependencies;
use the local Python route for an Ollama service running on the host.

## Existing Cases

`--case_path` accepts a directory or ZIP. Add a prompt to request changes, and use `--case_subdir` to select a case inside an archive containing multiple cases:

```bash
python foambench_main.py --output ./output/modified-case \
  --case_path /path/to/case --prompt_path ./requirements.txt

python foambench_main.py --output ./output/imported-case \
  --case_path ./tutorials.zip \
  --case_subdir multiphase/interFoam/laminar/damBreak/damBreak
```

The output contains read-only `original/`, writable `work/`, and `report/` directories. The report records initial case context and workflow logs; execution logs remain in `work/`. Source symlinks are rejected. Replacing a non-empty output requires Foam-Agent's ownership marker and `--overwrite_output` (API: `overwrite_output=true`). Keep output separate from the source and additional mesh inputs.

In Docker, mount host case directories or ZIPs into the container and pass their container paths. Mount the output directory separately to retain results on the host.

Platform detection uses OpenFOAM headers. Specify `--openfoam_target` when detection is inconclusive. A configured target conflict stops the workflow. Both generated and imported cases plan and validate the mesh before solver selection and physical input preparation. Meshing Planner chooses Standard, Gmsh or Custom meshing from the request and files; supplied meshes do not force reuse. Planning fails if required physical information cannot be inferred.

Full workflows retain editable source inputs under the report directory's
`inputs/workflow-*/`. Writer edits and successful mesh handoffs update these sources, which are restored into the
case directory before each execution. Supplied meshes and binary/private assets
are retained. Standalone `run` executes the supplied case directly.

After mesh handoff, local execution clears previous run artifacts and nonzero time directories;
it does not resume prior results. Use `original/` to recover imported inputs.
`Allrun` executes from `work/` for imports or the generated case directory.
HPC executes the same case through Slurm. Gzip inputs and multi-region cases are supported.

Local/HPC success requires zero exit, normal native solver completion and a valid
result field created or rewritten during this attempt, including time zero.

Generated and imported cases share runtime repair: Mesh/Solver Reviewers diagnose
current errors and logs, Writer edits inputs, and mesh repairs return to Meshing Planner.
Repairs are bounded by retry/recursion limits; solver repairs also stop on no progress
or repeated failed inputs, retaining diagnostics. Static repair follows the
separate policy above. If planning cannot proceed, correct the reported inputs
and start a new run.

Request visualization in the prompt. If execution succeeds but visualization fails, the status is `partial_success`, the termination reason is `visualization_failed`, and the CLI exits nonzero. MCP `run_case` returns the execution result and visualization error separately.

## Custom Meshes

External meshes use Gmsh ASCII 2.2 format. Describe the boundary conditions in the prompt:

```bash
python foambench_main.py --output ./output/tandem-wing \
  --prompt_path ./examples/tandem-wing/requirement.txt \
  --custom_mesh_path ./examples/tandem-wing/tandem_wing.msh
```

`--custom_mesh_path` can also be combined with an existing case and a modification prompt. For Docker, mount host inputs under `/home/openfoam/Foam-Agent/` and pass their container paths.

## OpenFOAM Targets and Corpora

| Native target | Runtime | Corpus |
|---|---|---|
| `foundation-v10` | Foundation OpenFOAM v10 | `database/foundation-v10/` |
| `esi-v2006` | ESI/OpenCFD v2006 | `database/esi-v2006/` |

Both targets share planning, generation, meshing, execution, repair, visualization, and import services while retaining release-native solver names and dictionaries. Set `FOAMAGENT_OPENFOAM_TARGET=esi-v2006` or pass `--openfoam_target esi-v2006` with its matching runtime. Legacy `FOAMAGENT_OPENFOAM_FORK=esi` translates Foundation-generated files, stopping and preserving inputs if a required conversion cannot be established; it neither selects nor runs for the native v2006 target.

Hydrate Git LFS files before use. Raw tutorials and FAISS indices are scoped to each target. Foundation includes Qwen 0.6B, Qwen 8B, and OpenAI small indices; ESI includes Qwen 0.6B and OpenAI small. Select matching runtime embedding settings. Prebuilt indices do not include embedding model weights; Hugging Face models may download those on first use.

Maintainers can rebuild a corpus from the matching sourced OpenFOAM installation:

```bash
python init_database.py --openfoam_path "$WM_PROJECT_DIR" \
  --openfoam_target foundation-v10 \
  --embedding_provider huggingface --embedding_model Qwen/Qwen3-Embedding-0.6B --force
```

Use `esi-v2006` for its corpus. If tutorials are omitted from an installation, pass `--tutorials_path` pointing to the matching release's extracted tutorials. Always supply both embedding arguments: the initializer checks Qwen 0.6B completeness by default, while standalone builders default to OpenAI/`text-embedding-3-small`. OpenAI embeddings need `OPENAI_API_KEY`. Omit `--force` to reuse complete assets. Existing assets require a matching target manifest. `init_database.py --database_path` is a target directory; `Config.database_path` is the parent containing the targets.

## HPC Execution and Monitoring

HPC execution uses locally available `sbatch` and `squeue`. Case files must be accessible to compute nodes, where the matching OpenFOAM environment must already be available. The generated job validates the runtime version; this is not an SSH or file-upload service.

Mesh and solver use separate jobs. Monitoring polls every 30 seconds within `Config.max_time_limit` (default 3600 seconds, including queue time). Empty `squeue` also requires the current attempt's zero exit-code record; `sacct` is not queried. Monitoring failure/timeout triggers cancellation and up to 60 seconds of stop confirmation. Unconfirmed stopping ends with `hpc_job_state_unconfirmed`; check Slurm before retrying. Uncertain submissions stop with `hpc_submission_unconfirmed` without automatic resubmission.

## Installation Checks and Offline Use

`python scripts/doctor.py` checks installed packages, the sourced OpenFOAM runtime,
LFS hydration, index files, cached/local model files and credential presence.
It does not contact an LLM or download model weights. Missing credentials/weights
produce a nonzero exit and a suggested next step.

- Add `--check-embedding` to load the embedding model, downloading it if needed,
  and compare its vector dimension with each selected FAISS index. Remote
  embedding providers use their API and may consume quota.
- Add `--check-llm` for a small structured request to the configured model.
  This tests authentication, model access and output compatibility, and consumes quota.
  Client retry policies can cause additional requests after transient errors.

File checks cannot prove weight integrity or model identity. Use a trusted copy
of the same model revision as your indices; the embedding probe also verifies
that weights can actually be loaded. Matching dimensions alone is insufficient.
Selected Bedrock/Ollama backends also require their corresponding optional Python
packages; the diagnostic reports missing extras before any remote probe.

For an offline copy, download on a connected machine and transfer the complete directory:

```bash
hf download Qwen/Qwen3-Embedding-0.6B \
  --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
  --local-dir /path/to/models/Qwen3-Embedding-0.6B
```

Then set the **identity and location separately** on the destination:

```bash
export FOAMAGENT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
export FOAMAGENT_EMBEDDING_MODEL_PATH=/path/to/models/Qwen3-Embedding-0.6B
export HF_HUB_OFFLINE=1
python scripts/doctor.py --check-embedding
```

Local model loading uses `local_files_only=True`; it continues using the existing
`Qwen_Qwen3-Embedding-0.6B` index directory. No symlink or index rebuild is needed.
With Docker, mount the model directory read-only and use its **container** path.
Alternative online models can share a persistent Hugging Face cache with
`-v foamagent-hf:/opt/foamagent-hf-cache`.

For an OpenAI-compatible endpoint, set `OPENAI_BASE_URL` together with the provider,
model and credentials. Keep `auto` for the existing provider behavior. If your
backend lacks native structured output, set
`FOAMAGENT_STRUCTURED_OUTPUT_METHOD=json_prompt` for schema-in-prompt output and
Pydantic validation, or explicitly select another method supported by that backend.
`json_mode` also puts the schema in the prompt. These are explicit settings;
authentication errors are not silently downgraded. Test with `--check-llm` before
starting a workflow. JSON validation checks response structure, not CFD correctness.

The Codex subscription backend uses native strict JSON Schema with `auto` or
`json_schema`, followed by local Pydantic validation. Use `json_prompt` explicitly
for schema-in-prompt compatibility. `function_calling` and `json_mode` are not
supported by this client. Authentication errors are not silently downgraded.

## Docker Builds

Build from the repository root. These Dockerfiles target Linux amd64 and bundle
the selected OpenFOAM runtime/corpus, CPU Torch, Gmsh and pinned default Qwen 0.6B
weights and client tokenizer caches. Build-time downloads require a connection; starting the built image does
not update code, resolve dependencies, or download default embedding weights.
LLM requests still require access to your selected provider.

```bash
git lfs pull --include="database/foundation-v10/raw/*,database/foundation-v10/faiss/Qwen_Qwen3-Embedding-0.6B/**,examples/tandem-wing/*.msh"
docker build --platform linux/amd64 -f docker/Dockerfile -t foamagent:local .

# Native ESI v2006 is compiled from source and can take substantially longer.
git lfs pull --include="database/esi-v2006/raw/*,database/esi-v2006/faiss/Qwen_Qwen3-Embedding-0.6B/**,examples/tandem-wing/*.msh"
docker build --platform linux/amd64 -f docker/Dockerfile.esi-v2006 -t foamagent:esi-v2006 .
```

Both images work in `/home/openfoam/Foam-Agent`. Mount input directories read-only
and outputs separately.
The default command opens a shell; append any CLI/MCP command to run it directly.
Container entrypoint messages use stderr.

Older images that auto-update at startup require `FOAMAGENT_SKIP_UPDATE=1` to
retain their bundled code. New builds never auto-update; `FOAMAGENT_VERSION`
no longer changes code inside the container. A full offline transfer can use
`docker save` and `docker load` on a Docker host.

## Troubleshooting

| Problem | Check |
|---|---|
| OpenFOAM not found or wrong version | Source the matching runtime and inspect `WM_PROJECT_DIR` / `WM_PROJECT_VERSION` |
| Missing corpus or index | Run `git lfs pull` and the corpus verifier; keep the model ID separate from its local path |
| Structured-output HTTP 400 | Select a supported `FOAMAGENT_STRUCTURED_OUTPUT_METHOD`, then use `doctor.py --check-llm` |
| Missing runtime dependencies | Install with constraints and run `python scripts/doctor.py`; add the extra for an optional provider |
| Authentication failure | Check the selected provider/model and its credentials before startup |
| MCP connection failure | Check server transport, host, port, and client settings; the HTTP endpoint is `/mcp` |
