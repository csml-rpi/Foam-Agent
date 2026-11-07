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

Install and source [Foundation OpenFOAM v10](https://openfoam.org/version/10/) for
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

Other `Config` fields use Python configuration or an exposed CLI/API parameter. Defaults include `max_loop=25` and `max_time_limit=3600` seconds. Unless explicitly set, `recursion_limit` is `6 * max(1, max_loop) + 8` (158 with the default repair budget). `input_writer_generation_mode=None` follows the execution preset; explicit `sequential_dependency` shares file context and `parallel_no_context` generates independently within each stage.

### Execution modes and static review

`execution_mode="careful"` is the default. Each stage generates files in the
Planner's dependency order and reviews its final inputs before its first execution.
Mesh review sees mesh dictionaries, generated Gmsh code, Allmesh and declared
operations before any mesh command or Slurm submission. Solver review follows
successful mesh handoff and physical input generation, using actual regional
boundary evidence before local/HPC Allrun execution. Mesh review does not require
future solver files. Both modes retain native checks and runtime repair.

```bash
python foambench_main.py --prompt_path requirements.txt --output output/careful
python foambench_main.py --prompt_path requirements.txt --output output/fast --execution_mode fast
```

`fast` generates independently in parallel within each stage and skips static
review. Mesh handoff still precedes solver planning. Set
`--input_writer_generation_mode sequential_dependency|parallel_no_context` or the
same Python `Config` field to override generation independently of review.

Static review uses the configured LLM to find definite, necessary input defects,
then verifies proposed errors in a second call. References provide target-version
syntax evidence, not replacement physical values. Unverified claims become
warnings for execution. Binary, oversized, hidden or linked inputs not inspected
are disclosed; gzip text inputs are decoded within the same context limits.
Passing review does not establish mesh quality, convergence or physical correctness.

`max_static_review_retries` defaults to `10`, shared by the two initial stages and
separate from runtime `max_loop`. Each requested repair uses one allowance, not
one per model call. A stage proceeds to normal execution checks when review passes,
no inputs/operations change, or the shared limit is reached. A later stage still
reviews and retains findings when no repair allowance remains. `0` reviews each
stage without static repairs. Service/response failures stop with
`static_review_failed`. Once a stage's initial review ends, native failures go to
its Mesh/Solver Reviewer without restarting static review. The graph adds one
initial solver-review node and budgets up to five node visits per static repair
for the longest mesh-detour route, in addition to the configured base recursion limit.

Reports live in `CASE/.foamagent/` for generated cases and `OUTPUT/report/` for
imports. `mesh_static_review.json` and `solver_static_review.json` retain each
stage's last initial review; `static_review.json` aggregates them in the existing
report format. Corresponding stage reference files and
`static_review_references.json` retain the evidence used. `review_history.jsonl`
stores phase-labelled proposals, verification decisions, repairs and runtime results.
Standalone MCP tools use an explicit [`static_review`](mcp.md#workflows) call;
full `run_case` follows both initial stage reviews in careful mode.

### Authentication

Set credentials before running the CLI or starting MCP; importing the services initializes a model client.

| Provider | Credentials |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `bedrock` | AWS credentials and a supported model/inference profile |
| `ollama` | [Optional Python packages, a local Ollama service and model](#ollama) |
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

Follow the [local installation](#local-installation), then install the Ollama
extra in the same Python environment and select your installed model:

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

The output contains read-only `original/`, writable `work/`, and `report/` directories. The report records initial case context and workflow logs; execution logs remain in `work/`. Source symlinks are rejected. Replacing a non-empty output requires Foam-Agent's ownership marker and `--overwrite_output` (API: `overwrite_output=true`). The output must not contain either the imported source or an additional mesh input, and must not be inside the source case directory; these relationships are checked before clearing an existing output.

In Docker, mount host case directories or ZIPs into the container and pass their container paths. Mount the output directory separately to retain results on the host.

Platform detection uses OpenFOAM headers. Specify `--openfoam_target` when detection is inconclusive. A configured target conflict stops preparation. Generated and imported cases enter Preparation, Meshing Planner, one of the Standard/Gmsh/Custom Meshing nodes in the main graph, then Solver Planner. Gmsh generates a mesh; Custom uses an existing `polyMesh` or converts a selected `.msh` file. All three execution nodes share file preparation, dependency checks, logging and actual mesh handoff services. The planner explicitly selects the backend; supplied meshes do not force reuse or a particular backend.

Mesh planning retrieves tutorial guidance from the selected target's existing corpus before generating mesh inputs. It records proposed boundary roles and their basis in the user request, supplied files, tutorial or inference. These plans travel alongside the actual mesh boundary and patch coordinate bounds read from the generated mesh; a proposed role or position is not proof of what was generated. Multi-region cases validate every active `constant/<region>/polyMesh` declared in native `constant/regionProperties` and preserve region names in boundary evidence, dictionaries and initial fields. An unused root mesh does not replace a missing regional mesh. Solver selection and solver-reference retrieval follow mesh handoff and check compatibility with the physical request. Incompatible layouts return to mesh planning.

Each mesh plan explicitly selects whether to run an existing Allmesh, build and run it from planned commands, or perform no additional script. Editing a utility input does not itself rerun the utility. Gmsh and Custom mesh conversion retain their own conversion/check operations. Final parallel reconstruction validates the declared partition sources against the actual ordinary or collated layout and retains the previous serial mesh until handoff succeeds. A confirmed reconstruction failure restores that mesh; an unconfirmed HPC submission or job state preserves the files for inspection and stops the workflow. Allrun contains solver execution, physical preparation and solver-parallel setup.

Gmsh executes one generation/conversion/check attempt per mesh node visit. Failures return to the main Mesh Reviewer and Meshing Planner within the shared repair budget. A Gmsh script successfully written by a declared file task is executed directly. Other attempts generate one program using the current script and repair feedback. Backend execution retains actual commands, return codes and output for handoff; Python generation, conversion and checking use the configured execution timeout. Boundary names and types are planned from supplied evidence and reread from the resulting mesh.

Workflow planners declare literal shell operations, including arguments, setup, helpers and retained script content. Allmesh and Allrun creation and repair render that approved program without a second command-selection call. Running an existing script requires its complete program in the plan; execution reports any difference before cleanup or submission. Script changes require an updated operation plan and startup dependencies. Standalone script-generation tools retain their existing usage. Required command failures propagate in both local and HPC execution, including failed pipelines; scripts must explicitly handle any intentionally nonzero command. When using `runApplication` or `runParallel`, let the helper own its log rather than redirecting to the same filename.

Solver planning receives current case files, actual mesh execution evidence and the separately retrieved solver script references. References remain guidance to reconcile with the request and actual files. Workflow runners activate the selected native runtime and execute Bash from the case directory with failure propagation enabled; operation planners receive these execution facts when deciding what shell setup is needed.

A zero script exit alone is not simulation success. Local and HPC runners check the current attempt for explicit errors, skipped RunFunctions commands, normal native completion (`Exec` and `End`) of the application declared in `system/controlDict`, and a valid result field. Result fields must be newly created or rewritten during the current Allrun attempt, including at time zero and when rewritten contents are identical. Existing unchanged initial fields do not establish success. This check covers the whole script and does not attribute writes to individual commands. Standalone MCP `run` executes the supplied Allrun directly without a separate model call to identify solver commands. Serial, processor-local, collated, compressed and region fields are supported. Keep native solver output in root-level `log*` files or standard output. These checks establish execution completion; they do not establish numerical accuracy or enforce a benchmark's prescribed comparison time.

Full workflows keep editable clean inputs under the report directory's
`inputs/workflow-*/`. Writer repairs these sources and restores them to the working
case. Mesh retries restore clean inputs, synchronize explicit file edits and
commit successful mesh handoffs before solver execution. Failed execution outputs
are not recaptured as clean inputs. Supplied binary/private assets are preserved;
review context limits do not limit the stored inputs. Standalone execution remains direct.

Local and HPC solver execution clear previous run artifacts, nonzero time directories and processor directories from `work/` before each attempt, after successful mesh handoff. Initial directories representing time zero, such as `0/` and `0.0/`, and the final `constant/polyMesh` are retained. Compressed `.gz` dictionaries, initial fields and mesh boundaries are read into planning and repair context; edits preserve the selected file's compression. Use `original/` to recover the imported input; this workflow does not resume prior results. The case's solver-only `Allrun` executes from `work/`, locally or through Slurm.

Generated and imported cases share the repair budget but have separate Mesh Reviewer and Solver Reviewer nodes. Each reviewer returns one structured diagnosis, decision, file tasks and operation adjustments; its own history records subsequent execution outcomes. A surrounding JSON code fence is accepted. Other response validation failures allow one correction request in the same review round; a second invalid response stops with `invalid_review_response` and preserves both raw responses in the review log. Mesh failures return to Meshing Planner. Solver Reviewer sends physical repairs to Input Writer, or explicitly requests mesh work while retaining pending physical tasks. Successful file writes consume tasks but do not establish execution success. Solver repairs stop when no progress is made; changes to initial time-zero fields count as progress, while runtime results and archived attempts do not. Repeated mesh failures are supplied to Mesh Reviewer within the shared budget.

Stage plans declare each operation's stage, intended effect and required startup files. Mesh planning and Mesh Reviewer can defer physical operations for Solver Planner to reconcile after mesh handoff; only declared mesh operations enter Allmesh. Missing physical inputs do not make physical initialization part of meshing. Deferred physical operations also return an already planned case to Solver Planner before solver execution. After writing and before local execution or Slurm submission, missing declared inputs are reported to the corresponding reviewer. Intermediate files produced by preceding operations are not required at startup. This checks declared dependencies only; it cannot establish that the model has identified every implicit dependency. Allmesh and Allrun generation follows these plans. Mesh tutorial retrieval caches the matching tutorial script separately from solver references; scripts are matched by tutorial identity and directory structure and used only as planning guidance. Files and logs from the last attempt remain available. If planning cannot proceed, correct the reported inputs and start a new run.

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

Both targets share planning, generation, meshing, execution, repair, visualization, and import services while retaining release-native solver names and dictionaries. Set `FOAMAGENT_OPENFOAM_TARGET=esi-v2006` or pass `--openfoam_target esi-v2006` with its matching runtime. Legacy `FOAMAGENT_OPENFOAM_FORK=esi` translates Foundation-generated files on a best-effort basis; it neither selects nor runs for the native v2006 target.

Hydrate Git LFS files before use. Raw tutorials and FAISS indices are scoped to each target. Foundation includes Qwen 0.6B, Qwen 8B, and OpenAI small indices; ESI includes Qwen 0.6B and OpenAI small. Select matching runtime embedding settings. Prebuilt indices do not include embedding model weights; Hugging Face models may download those on first use.

Maintainers can rebuild a corpus from the matching sourced OpenFOAM installation:

```bash
python init_database.py --openfoam_path "$WM_PROJECT_DIR" \
  --openfoam_target foundation-v10 \
  --embedding_provider huggingface --embedding_model Qwen/Qwen3-Embedding-0.6B --force
```

Use `esi-v2006` for its corpus. If tutorials are omitted from an installation, pass `--tutorials_path` pointing to the matching release's extracted tutorials. Always supply both embedding arguments: the initializer checks Qwen 0.6B completeness by default, while standalone builders default to OpenAI/`text-embedding-3-small`. OpenAI embeddings need `OPENAI_API_KEY`. Omit `--force` to reuse complete assets. `init_database.py --database_path` is a target directory; `Config.database_path` is the parent containing the targets.

## HPC Execution and Monitoring

HPC execution uses locally available `sbatch` and `squeue`. Case files must be accessible to compute nodes, where the matching OpenFOAM environment must already be available. The generated job validates the runtime version; this is not an SSH or file-upload service.

Mesh and solver stages use separate Slurm jobs. Monitoring polls every 30 seconds by default; the deadline is `Config.max_time_limit` (3600 seconds by default), including queue time. An empty `squeue` result ends monitoring, but success also requires the current attempt's zero exit-code record. Solver execution additionally requires normal native completion and newly created or rewritten result fields as described above. The exit-code record comes from the stage execution wrapper; `sacct` is not queried.

On monitoring failure or timeout, Foam-Agent attempts to cancel the job, then waits up to 60 seconds to confirm that it stopped. If stopping cannot be confirmed, the workflow terminates with `hpc_job_state_unconfirmed`; check the job in Slurm before starting another run.

A zero-exit `sbatch` response without a job ID, or an unexpected submission exception, stops the workflow with `hpc_submission_unconfirmed`. Nonzero `sbatch` exits enter the bounded submission-repair loop. Communication failures reported through a nonzero exit are not distinguished from confirmed submission rejections.

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
and outputs separately, as in the [Quick Start](../README.md#quick-start-recommended-docker).
The default command opens a shell; append any CLI/MCP command to run it directly.
Container entrypoint messages use stderr.

For published images such as `v2.2.0`, follow that release's documentation.
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
| MCP connection failure | Check server transport, host, port, and client settings in the [MCP guide](mcp.md) |
