# Usage Guide

## Local Installation

Install Conda, Git, and Git LFS first. The supplied Conda environment targets Linux with Python 3.12.9.

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
git lfs install
git lfs pull
conda env create -f environment.yml
conda activate FoamAgent
pip install --no-deps -e .
```

Install and source [Foundation OpenFOAM v10](https://openfoam.org/version/10/) for the default target, or ESI/OpenCFD v2006 for the native `esi-v2006` target. Verify that `WM_PROJECT_DIR` and `WM_PROJECT_VERSION` identify the intended installation. The Conda environment supplies Python dependencies; the editable install registers `foamagent-mcp`.

Configure [authentication](#authentication) before running. Adapt the bashrc path to your installation:

```bash
source /opt/openfoam10/etc/bashrc
python foambench_main.py --output ./output/my-case --prompt_path ./user_requirement.txt
```

For an existing Python environment, `pip install -e .` installs the base Python dependencies. Add `.[bedrock]` or `.[ollama]` for those providers, or `.[all]` for both. Generated Gmsh scripts require the `gmsh` Python package and its system libraries; the Docker images include it.

Commands assume the repository root. Use `python foambench_main.py --help` for CLI options. Each `--output` is a dedicated task directory: generated cases contain OpenFOAM files, results, `Allrun.out`/`Allrun.err`, and workflow/review logs. Use a new directory for a new task; `--overwrite_output` clears an existing Foam-Agent-owned output.

## Configuration

| Environment variable | Purpose | Default |
|---|---|---|
| `FOAMAGENT_MODEL_PROVIDER` | `openai`, `openai-codex`, `anthropic`, `bedrock`, `ollama`, or `deepseek` | `openai-codex` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier for the selected provider | `gpt-5.3-codex` |
| `FOAMAGENT_EMBEDDING_PROVIDER` | `openai`, `huggingface`, or `ollama` | `huggingface` |
| `FOAMAGENT_EMBEDDING_MODEL` | Model matching the selected FAISS indices | `Qwen/Qwen3-Embedding-0.6B` |
| `FOAMAGENT_OPENFOAM_TARGET` | Explicit native target: `foundation-v10` or `esi-v2006` | Unset; Foundation path |
| `FOAMAGENT_OPENFOAM_FORK` | Legacy fork routing: `foundation` or translated `esi` | `foundation` |

Other `Config` fields use Python configuration or an exposed CLI/API parameter. Defaults include `max_loop=25`, `recursion_limit=100`, and `max_time_limit=3600` seconds. `input_writer_generation_mode` selects `sequential_dependency` (default, files share context) or `parallel_no_context` (independent parallel generation).

### Authentication

Set credentials before running the CLI or starting MCP; importing the services initializes a model client.

| Provider | Credentials |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `bedrock` | AWS credentials and a supported model/inference profile |
| `ollama` | A local Ollama installation and model |
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
  leoyue123/foamagent
```

## Existing Cases

`--case_path` accepts a directory or ZIP. Add a prompt to request changes, and use `--case_subdir` to select a case inside an archive containing multiple cases:

```bash
python foambench_main.py --output ./output/modified-case \
  --case_path /path/to/case --prompt_path ./requirements.txt

python foambench_main.py --output ./output/imported-case \
  --case_path ./tutorials.zip \
  --case_subdir multiphase/interFoam/laminar/damBreak/damBreak
```

The output contains read-only `original/`, writable `work/`, and `report/` directories. The report records initial case context and workflow logs; execution logs remain in `work/`. Source symlinks are rejected. Replacing a non-empty output requires Foam-Agent's ownership marker and `--overwrite_output` (API: `overwrite_output=true`).

In Docker, mount host case directories or ZIPs into the container and pass their container paths. Mount the output directory separately to retain results on the host.

Platform detection uses OpenFOAM headers. Specify `--openfoam_target` when detection is inconclusive. A configured target conflict is routed to Reviewer for repair. With `Allrun`, no detected issues, and no explicit prompt or custom mesh, a case skips file planning and routes to Runner; LLM resources and routing calls are still required. Other cases use planning to select file changes or meshing, and fail if required physical information cannot be inferred.

Local execution clears previous run artifacts and nonzero time directories from `work/` before each run. Use `original/` to recover the input; this workflow does not resume prior results. The case's `Allrun` executes from `work/` unless the repair workflow changes it. HPC execution follows its Slurm script and `Allrun` instead.

Generated and imported cases share the repair loop. Reviewer diagnoses errors, Input Writer rewrites files, and meshing errors return to Meshing before running again. Repair stops at the retry/recursion limit or when no progress is made, retaining the last attempt's files and logs. If planning cannot proceed, correct the reported inputs and start a new run.

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

Monitoring polls every 30 seconds for up to 3600 seconds by default. An empty `squeue` result is treated as completion, then case logs are checked; `sacct` and the Slurm exit code are not checked. A monitoring timeout may enter the repair/resubmission loop while the original job remains active.

## Docker Builds

Build from the repository root with hydrated target corpora:

```bash
docker build -f docker/Dockerfile -t foamagent:foundation-v10 .
docker build -f docker/Dockerfile.esi-v2006 -t foamagent:esi-v2006 .
```

Each image contains its own OpenFOAM runtime; select the matching image. ESI v2006 is compiled from source on Ubuntu 20.04. Both use `/home/openfoam/Foam-Agent` and update code from GitHub at startup by default. Set `FOAMAGENT_SKIP_UPDATE=1` to use bundled code, including local changes; `FOAMAGENT_VERSION` selects a tag during startup updates.

An image tag alone does not disable startup updates. For host access to CLI results, bind-mount an output directory as shown in [Quick Start](../README.md#quick-start); input paths passed to the CLI must also exist inside the container.

## Troubleshooting

| Problem | Check |
|---|---|
| OpenFOAM not found or wrong version | Source the matching runtime and inspect `WM_PROJECT_DIR` / `WM_PROJECT_VERSION` |
| Missing corpus or index | Run `git lfs pull`; select an embedding model with indices for that target |
| Missing runtime dependencies | Use the Conda environment; install the extra for an optional provider |
| Authentication failure | Check the selected provider/model and its credentials before startup |
| MCP connection failure | Check server transport, host, port, and client settings in the [MCP guide](mcp.md) |
