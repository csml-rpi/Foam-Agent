# Foam-Agent    <a href="https://arxiv.org/abs/2505.04997"><img src="https://img.shields.io/badge/arXiv-2505.04997-b31b1b.svg" alt="Paper"></a>
<p align="center">
  <img src="overview.png" alt="Foam-Agent System Architecture" width="800">
</p>

<p align="center">
    <em>An End-to-End Composable Multi-Agent Framework for Automating CFD Simulation in OpenFOAM</em>
</p>

**Foam-Agent** automates the entire **OpenFOAM**-based CFD simulation workflow from a single natural language prompt. It manages meshing, case setup, execution, error correction, and post-processing — dramatically lowering the expertise barrier for Computational Fluid Dynamics. Evaluated on [FoamBench](https://arxiv.org/abs/2509.20374) with 110 simulation tasks, our framework achieves an **100% success rate** with Claude Opus 4.6.

Visit [deepwiki.com/csml-rpi/Foam-Agent](https://deepwiki.com/csml-rpi/Foam-Agent) for a comprehensive introduction and to ask questions interactively.

## Key Features

- **End-to-End Automation**: From meshing (including external Gmsh `.msh` files) to HPC job submission to ParaView/PyVista visualization — one prompt does it all.
- **Multi-Agent Workflow**: Architect, Input Writer, Runner, and Reviewer agents collaborate through a LangGraph pipeline with automatic error correction (up to 25 iterations).
- **RAG-Enhanced Generation**: Hierarchical FAISS indices built from OpenFOAM tutorials provide context-specific retrieval for accurate configuration file generation.
- **Composable Service Architecture**: Core functions are exposed as MCP tools, enabling integration with Claude Code, Cursor, and other agentic systems.

## Quick Start

### 1. Pull and run the Docker image

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  --name foamagent \
  leoyue123/foamagent
```

The container comes with OpenFOAM v10, Conda, and all dependencies pre-installed.

> For a specific release: `docker pull leoyue123/foamagent:v2.0.0`

### 2. Write your prompt

Edit `user_requirement.txt` inside the container:

```text
do a Reynolds-Averaged Simulation (RAS) pitzdaily simulation. Use PIMPLE algorithm.
The domain is a 2D millimeter-scale channel geometry. Boundary conditions specify a
fixed velocity of 10m/s at the inlet (left), zero gradient pressure at the outlet
(right), and no-slip conditions for walls. Use timestep of 0.0001 and output every
0.01. Finaltime is 0.3. use nu value of 1e-5.
```

### 3. Run

```bash
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
```

That's it. Foam-Agent will plan the case, generate all OpenFOAM files, run the simulation, and fix errors automatically.

### 4. Run an Existing Case

An existing case can be supplied instead of a natural-language prompt. This
mode does not invoke Planner, Meshing, or Input Writer. Instead it enters the
workflow's protected existing-case branch (`case_import` → shared local runner
with a controlled-import policy → shared reviewer with a deterministic
safe-repair policy), preserving the uploaded dictionaries and running a
validated, controlled equivalent of its `Allrun`.

```bash
python foambench_main.py \
  --output ./imported-dam-break \
  --case_path /path/to/damBreak \
  --visualize
```

`--visualize` is optional. When supplied, a successful imported case uses the
same read-only PyVista visualization node as a prompt-generated case.

`--case_path` accepts either a case directory or a ZIP archive. If an archive
contains multiple cases, select one explicitly:

```bash
python foambench_main.py \
  --output ./imported-case \
  --case_path ./tutorials.zip \
  --case_subdir multiphase/interFoam/laminar/damBreak/damBreak
```

The output directory contains `original/` (a hashed source copy with read-only files),
`work/` (the only directory executed or repaired), and `report/` (the manifest,
attempt history, and any diffs). User-provided numeric tokens are never changed
automatically. The importer only permits a small set of Foundation v10
utilities/solver commands; custom compilation, dynamic code, ESI cases, and
unsafe shell setup are reported as blockers rather than executed. If `Allrun`
is absent, Foam-Agent can infer only the minimal `blockMesh`/`checkMesh`/solver
plan for a case that already provides enough files.

## Configuration

All settings live in `src/config.py` with sensible defaults. Every setting can be overridden via environment variables — no need to edit files, especially useful for Docker and CI.

### LLM Provider and Model

| Environment Variable | Purpose | Allowed Values |
|---|---|---|
| `FOAMAGENT_MODEL_PROVIDER` | LLM backend | `openai`, `openai-codex`, `anthropic`, `bedrock`, `ollama` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier | e.g., `gpt-5-mini`, `gpt-5.6-terra`, `claude-opus-4-6` |

Example:
```bash
docker run -it \
  -e FOAMAGENT_MODEL_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY=your-key-here \
  -e FOAMAGENT_MODEL_VERSION=claude-opus-4-6 \
  -p 7860:7860 \
  leoyue123/foamagent
```

### Embedding Provider and Model

| Environment Variable | Purpose | Allowed Values |
|---|---|---|
| `FOAMAGENT_EMBEDDING_PROVIDER` | Embedding backend | `openai`, `huggingface`, `ollama` |
| `FOAMAGENT_EMBEDDING_MODEL` | Embedding model | e.g., `Qwen/Qwen3-Embedding-0.6B`, `text-embedding-3-small` |

Defaults to `huggingface` with `Qwen/Qwen3-Embedding-0.6B` (runs locally, no API key needed).

### API Keys

| Variable | When needed |
|---|---|
| `OPENAI_API_KEY` | Using `openai` provider |
| `ANTHROPIC_API_KEY` | Using `anthropic` provider |
| AWS credentials | Using `bedrock` provider |

### Input Writer Generation Mode

Set in `src/config.py` via `input_writer_generation_mode`:

| Mode | Behavior | Best for |
|---|---|---|
| `sequential_dependency` | Files generated in order with cross-file context | Expensive runs (HPC, long simulations) |
| `parallel_no_context` | Files generated in parallel, no cross-file context | Fast local runs where retry is cheap |

### Recommended Models

| Framework | Model | Basic | Advanced |
|---|---|---:|---:|
| FoamAgent 2.0.0 (10 loops) | Opus 4.6 | 85.45% | 100% |
| FoamAgent 2.0.0 (25 loops) | Opus 4.6 | 100% | 100% |
| FoamAgent 2.0.0 (25 loops) | Sonnet 4.6 | 87.88% | 75.00% |
| FoamAgent 2.0.0 (25 loops) | Haiku 4.6 | 54.55% | 37.50% |
| FoamAgent 2.0.0 (25 loops) | gpt-5.4 | 45.45% | 75.00% |
| FoamAgent 2.0.0 (25 loops) | gpt-5.3-codex | 54.55% | 62.50% |

We recommend **Anthropic Claude Opus 4.6** for best results.

## Advanced Usage

### Custom Mesh Files

Foam-Agent supports external Gmsh `.msh` files (ASCII 2.2 format). Describe boundary conditions in your prompt and pass the mesh:

```bash
python foambench_main.py \
  --output ./output \
  --prompt_path ./user_req_tandem_wing.txt \
  --custom_mesh_path ./tandem_wing.msh
```

To mount a mesh file from the host into Docker:

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -v /path/to/my_mesh.msh:/home/openfoam/Foam-Agent/my_mesh.msh \
  -p 7860:7860 \
  leoyue123/foamagent
```

### Skill / MCP Integration (Claude Code, Cursor, Windsurf, etc.)

Foam-Agent exposes its full CFD workflow as an **MCP server** — the universal protocol supported by Claude Code, Cursor, Windsurf, and other AI-powered tools. It also ships with a **Claude Code skill** (`/foam`) for one-command simulation runs.

#### Quick Setup (Local Install)

```bash
# 1. Install (adds the foamagent-mcp command)
pip install -e .

# 2. Register with your AI tool
claude mcp add foamagent -- foamagent-mcp                # Claude Code
```

For **Cursor**: open Settings > Features > MCP > Edit MCP Settings, and add:

```json
{
  "mcpServers": {
    "foamagent": {
      "command": "foamagent-mcp"
    }
  }
}
```

For **Windsurf / other MCP-compatible tools**, use the same JSON config above.

#### Quick Setup (Docker)

If running in Docker, start the HTTP server and point your MCP client at it:

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  leoyue123/foamagent \
  foamagent-mcp --transport http --host 0.0.0.0 --port 7860
```

Then configure your MCP client:

```json
{
  "mcpServers": {
    "foamagent": {
      "url": "http://localhost:7860/mcp"
    }
  }
}
```

> If running Docker on a remote server, ensure port 7860 is reachable (e.g., via SSH port forwarding or `-p 7860:7860`).

#### Available MCP Tools

Foam-Agent generates output following **Foundation OpenFOAM v10** conventions by default. If
`FOAMAGENT_OPENFOAM_FORK=esi` is set, generated input files are translated to ESI OpenFOAM
(`openfoam.com`) naming and dictionary conventions on a best-effort basis before they are returned.
The run/review/fix workflow is still primarily validated with Foundation OpenFOAM v10.

| Tool | Description |
|------|-------------|
| `plan` | Analyze requirements and plan simulation structure using Foundation v10 references |
| `input_writer` | Generate OpenFOAM configuration files; optionally translate generated files when `FOAMAGENT_OPENFOAM_FORK=esi` |
| `run` | Execute Allrun script locally with error collection; primarily validated with Foundation OpenFOAM v10 |
| `review` | Analyze simulation errors and suggest fixes via LLM using Foundation v10 references |
| `apply_fixes` | Rewrite OpenFOAM files based on review analysis; ESI cases remain best-effort |
| `visualization` | Generate PyVista visualization of simulation results |

#### Claude Code Skill

For Claude Code users who clone this repo, a `/foam` skill is included in `.claude/skills/foam.md`. It orchestrates the MCP tools into a complete workflow:

```
/foam Simulate lid-driven cavity flow at Re=1000
```

This triggers the full pipeline: plan -> generate files -> run -> review/fix loop -> visualize.

### Codex OAuth Sign-in (No API Key)

If you have a ChatGPT/Codex subscription, you can authenticate via OAuth instead of an API key:

1. Install the [Codex CLI](https://github.com/openai/codex) on your host machine.
2. Run `codex login` and choose **"Sign in with ChatGPT"**.
3. Verify the token cache exists: `ls ~/.codex/auth.json`
4. Mount it into the container:

```bash
docker run -it \
  -e FOAMAGENT_MODEL_PROVIDER=openai-codex \
  -e FOAMAGENT_MODEL_VERSION=gpt-5.6-terra \
  -v ~/.codex/auth.json:/root/.codex/auth.json:ro \
  -p 7860:7860 \
  leoyue123/foamagent
```

Foam-Agent searches for OAuth tokens at (first match wins):
- `$CODEX_HOME/auth.json`
- `~/.codex/auth.json`
- `~/.clawdbot/agents/main/agent/auth-profiles.json`

> Security note: `auth.json` contains access tokens. Treat it like a password.

### Manual Installation (Without Docker)

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
conda env create -n FoamAgent -f environment.yml
conda activate FoamAgent
```

You also need **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) installed and sourced for the default, fully validated runtime path. ESI OpenFOAM (`openfoam.com`) file generation is available as best-effort translation by setting `FOAMAGENT_OPENFOAM_FORK=esi`, but ESI execution and repair loops should be verified per case. Follow the [official Foundation v10 installation guide](https://openfoam.org/version/10/) and verify with:

```bash
echo $WM_PROJECT_DIR   # should print e.g. /opt/openfoam10
```

Then run:

```bash
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
```

### Building the Docker Image from Source

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
docker build -f docker/Dockerfile -t foamagent:latest .
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  foamagent:latest
```

## Troubleshooting

| Problem | Solution |
|---|---|
| OpenFOAM environment not found | Ensure the intended OpenFOAM bashrc is sourced. The default validated path is Foundation OpenFOAM v10 ([openfoam.org](https://openfoam.org)); ESI OpenFOAM requires `FOAMAGENT_OPENFOAM_FORK=esi` and per-case verification |
| Database files missing | Ensure the full repo is cloned including `database/`. Docker image has these pre-built |
| Missing dependencies | `conda env update -n FoamAgent -f environment.yml --prune` |
| API key errors | Ensure the appropriate key is set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) |
| MCP connection errors | Verify the container is running and port 7860 is accessible |

> **OpenFOAM version:** Foam-Agent targets **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) by default. With `FOAMAGENT_OPENFOAM_FORK=esi`, generated files are translated to ESI OpenFOAM ([openfoam.com](https://openfoam.com), e.g., v2312, v2406, v2512) conventions on a best-effort basis. The Docker image includes Foundation OpenFOAM v10 pre-installed.

## Community

### Join the WeChat community

Chinese-speaking users can join the Foam-Agent WeChat community by adding the volunteer's WeChat account: **ZDSJTUCFD**. The volunteer will invite you to the group.

## Citation
If you use Foam-Agent in your research, please cite our paper:
```bibtex
@article{yue2025foam,
    title = {Foam-Agent: A large language model-based multi-agent framework for automating computational fluid dynamics workflows},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume = {461},
    pages = {119271},
    year = {2026},
    issn = {0045-7825},
    author = {Ling Yue and Nithin Somasekharan and Tingwen Zhang and Yadi Cao and Zhangze Chen and Shimin Di and Shaowu Pan}
}

```

