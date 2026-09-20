# Foam-Agent MCP Server

Expose OpenFOAM CFD simulation as tools for any AI coding assistant via [MCP (Model Context Protocol)](https://modelcontextprotocol.io/).

> **OpenFOAM version:** This server targets **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) by default. `FOAMAGENT_OPENFOAM_FORK=esi` remains the legacy best-effort translation path for generated files. Set `FOAMAGENT_OPENFOAM_TARGET=esi-v2006` to use the supported native ESI/OpenCFD v2006 target with its isolated corpus, native conventions, and target-specific runtime checks. Native v2006 requires a matching runtime and corpus.

## Quick Start

### 1. Install

```bash
# Clone and install
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
pip install -e .
```

Or with conda (full environment including PyTorch, FAISS, etc.):

```bash
conda env create -f environment.yml
conda activate FoamAgent
pip install -e .
```

### 2. Register with your AI tool (one command)

**Claude Code:**
```bash
claude mcp add foamagent -- foamagent-mcp
```

**Cursor:**
Add to `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "foamagent": {
      "command": "foamagent-mcp"
    }
  }
}
```

**Windsurf / Other MCP-compatible tools:**
```json
{
  "mcpServers": {
    "foamagent": {
      "command": "foamagent-mcp"
    }
  }
}
```

**HTTP mode** (for web clients or remote access):
```bash
foamagent-mcp --transport http --host 0.0.0.0 --port 7860
```

### 3. Configure LLM authentication and provider

Set environment variables to choose your LLM backend:

```bash
export FOAMAGENT_MODEL_PROVIDER=anthropic          # also openai, openai-codex, bedrock, ollama, deepseek
export FOAMAGENT_MODEL_VERSION=claude-sonnet-4-6   # model identifier
export ANTHROPIC_API_KEY=sk-ant-...                # API key for your provider
```

The default is `openai-codex`/`gpt-5.3-codex`, using the token cache read by `src/utils.py`. An OpenAI API key alone does not select the `openai` backend. Configure credentials before starting the server: importing its services initializes an LLM client. Use the full Conda environment for the repository's runtime dependencies; `pip install -e .` registers the console command but is not equivalent to that environment.

## Available MCP Tools

Foam-Agent generates output following **Foundation OpenFOAM v10** conventions by default.
Native ESI/OpenCFD v2006 is selected with `FOAMAGENT_OPENFOAM_TARGET=esi-v2006` and uses
the isolated v2006 corpus and native conventions. The legacy
`FOAMAGENT_OPENFOAM_FORK=esi` setting remains a separate best-effort translation path.

| Tool | Description |
|------|-------------|
| `plan` | Analyze requirements and plan a case using references for the active OpenFOAM target |
| `input_writer` | Generate OpenFOAM configuration files using the active target; legacy `fork=esi` translation remains best-effort |
| `run` | Execute the local case with target-specific runtime selection and error collection |
| `review` | Analyze simulation errors using references for the active OpenFOAM target |
| `apply_fixes` | Rewrite OpenFOAM files using the active target convention |
| `run_case` | Run or modify an existing case/ZIP through the shared Planner, Writer, Meshing, Runner, and Reviewer graph |
| `visualization` | Generate PyVista visualization of simulation results |

## Typical Workflow

Once registered, ask your AI assistant naturally:

> "Simulate lid-driven cavity flow at Re=1000"

The assistant will call the tools in sequence:
1. **plan** - Parse requirements, select solver, generate subtasks
2. **input_writer** - Generate all OpenFOAM files
3. **run** - Execute the simulation
4. **review + apply_fixes** - Analyze and rewrite errors; the client decides whether to rerun and how many times
5. **visualization** - Render results

For an existing case, call **run_case** with `case_path`, `output_dir`, and an
optional `user_requirement`, `custom_mesh_path`, or `openfoam_target`. With no explicit requirement or custom mesh, an imported case with `Allrun` and no detected issues skips file planning; it still loads LLM resources and makes routing calls. Missing files can enter LLM planning rather than failing immediately. Explicit target conflicts are recorded deterministically and routed to Reviewer for repair. If planning fails, correct the inputs described in the failure reason and start a new run.

The individual tools do not automatically run the complete graph. In particular, `run` is local-only, and there is no standalone meshing or HPC tool. Use the CLI graph for generated-case mesh/HPC routing. `run_case` executes the shared imported-case graph, including mesh repair and its configured retry limit.

Standalone `review` uses the case directory's basename and fixed `simpleFoam`/`fluid`/`tutorial` metadata for reference retrieval; it does not infer the actual solver. It does read the current case files and supplied errors. `apply_fixes` requires `case_dir`, `error_logs`, `review_analysis`, and `user_requirement` and builds a file-scoped rewrite plan.

`run_case` returns `status`, `message`, `visualization_error`, `case_dir`, `report_dir`, `errors`, and `termination_reason`. If simulation succeeds but visualization fails, its status is `partial_success` and its message is `Simulation completed successfully, but visualization failed.` The standalone `visualization` tool instead raises an error on failure; it uses PyVista regardless of the currently unused `visualization_type` field.

HPC jobs use locally available Slurm commands, not an SSH/file-upload service. Monitoring currently treats an empty `squeue` result as completion without checking `sacct` or the exit code. A monitoring timeout may enter the repair/resubmission loop while the original job remains active. See the root [README](../../README.md#hpc-execution-and-monitoring).

## Prerequisites

- **Python 3.10+** with dependencies installed
- **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) installed and available in PATH for the default runtime path, or a matching **ESI/OpenCFD v2006** runtime when `FOAMAGENT_OPENFOAM_TARGET=esi-v2006` is selected.
- Authentication for the selected backend (OAuth cache, provider API key, AWS credentials, or a reachable local Ollama service).

## Architecture

```
AI Tool (Claude Code / Cursor / ...)
    ↓ MCP protocol (stdio or HTTP)
foamagent-mcp (this server)
    ↓
Service Layer (src/services/*.py)
    ↓
OpenFOAM + LLM Services
```

## Advanced Configuration

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `FOAMAGENT_MODEL_PROVIDER` | LLM backend | `openai-codex` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier | `gpt-5.3-codex` |
| `FOAMAGENT_EMBEDDING_PROVIDER` | Embedding backend | `huggingface` |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model | `Qwen/Qwen3-Embedding-0.6B` |
| `FOAMAGENT_OPENFOAM_FORK` | OpenFOAM target fork for generated files: `foundation` or `esi` | `foundation` |
| `FOAMAGENT_OPENFOAM_TARGET` | Explicit native target: `foundation-v10` or `esi-v2006`; v2006 bypasses legacy ESI translation | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — |

These are explicit environment overrides, not a generic mapping for all `Config` fields. `foamagent-mcp` defaults to stdio; invoking `python -m src.mcp.fastmcp_server` defaults to HTTP. Individual tools use the server's configured target. `run_case.openfoam_target` overrides the target only for that workflow invocation.

## Troubleshooting

**Import errors:** Ensure you ran `pip install -e .` from the repo root.

**Database errors:** Target-scoped FAISS indices ship pre-built in
`database/foundation-v10/` and `database/esi-v2006/`. If a target corpus is
missing, rebuild the selected corpus with:
```bash
python init_database.py --openfoam_path "$WM_PROJECT_DIR" --openfoam_target foundation-v10 \
  --embedding_provider huggingface --embedding_model Qwen/Qwen3-Embedding-0.6B --force
```

For native v2006, select `esi-v2006` and its matching installation. Hydrate Git LFS assets before loading. Always specify the embedding provider/model: runtime defaults are Qwen 0.6B, while standalone builders default to OpenAI small. The ESI corpus currently has no prebuilt Qwen 8B indices.

**OpenFOAM not found:** The default validated runtime path requires Foundation OpenFOAM v10 ([openfoam.org](https://openfoam.org)). Legacy generic ESI uses `FOAMAGENT_OPENFOAM_FORK=esi`; native v2006 uses `FOAMAGENT_OPENFOAM_TARGET=esi-v2006`, a v2006 target corpus, and `WM_PROJECT_VERSION=v2006`. HPC jobs use the existing compute-node OpenFOAM environment and validate its version. Build the matching image with:
```bash
docker build -f docker/Dockerfile -t foamagent:foundation-v10 .
docker build -f docker/Dockerfile.esi-v2006 -t foamagent:esi-v2006 .
docker run -it -p 7860:7860 foamagent:foundation-v10 foamagent-mcp --transport http
```
