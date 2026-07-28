# Foam-Agent    <a href="https://arxiv.org/abs/2505.04997"><img src="https://img.shields.io/badge/arXiv-2505.04997-b31b1b.svg" alt="Paper"></a>
<p align="center">
  <img src="overview.png" alt="Foam-Agent System Architecture" width="800">
</p>

<p align="center">
    <em>An End-to-End Composable Multi-Agent Framework for Automating CFD Simulation in OpenFOAM</em>
</p>

<p align="center">
  <b>English</b> | <a href="README_ja.md">日本語</a>
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

The container comes with OpenFOAM v10, Python, and all dependencies pre-installed.

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
uv run python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
```

That's it. Foam-Agent will plan the case, generate all OpenFOAM files, run the simulation, and fix errors automatically.

## Configuration

All settings live in `src/foamagent/config.py` with sensible defaults. Every setting can be overridden via environment variables — no need to edit files, especially useful for Docker and CI.

### LLM Provider and Model

| Environment Variable | Purpose | Allowed Values |
|---|---|---|
| `FOAMAGENT_MODEL_PROVIDER` | LLM backend (default `openai`) | `openai`, `openai-codex`, `anthropic`, `bedrock`, `ollama` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier (default `gpt-5-mini`) | e.g., `gpt-5-mini`, `gpt-5.3-codex`, `claude-opus-4-6` |
| `FOAMAGENT_OPENAI_BASE_URL` | OpenAI-compatible endpoint (OpenRouter, vLLM, LiteLLM, ...) | a base URL; empty means the official OpenAI endpoint |

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

### OpenFOAM Execution Runtime

By default Foam-Agent runs solvers with the OpenFOAM installation on the current machine. It can
instead run them inside a container, which is useful when the host has no OpenFOAM.

| Environment Variable | Purpose | Default |
|---|---|---|
| `FOAMAGENT_OPENFOAM_RUNTIME` | `native` sources the local installation; `docker` runs inside an image | `native` |
| `FOAMAGENT_OPENFOAM_IMAGE` | Image used by the `docker` runtime | `foam-bench:latest` |
| `FOAMAGENT_OPENFOAM_BASHRC` | Path to the OpenFOAM bashrc inside that image | `/opt/openfoam10/etc/bashrc` |

The `docker` runtime mounts the case directory at the same absolute path inside the container, so
paths in the logs match the host, and passes your UID/GID so the generated files are not owned by
root.

### Other Settings

| Environment Variable | Purpose | Default |
|---|---|---|
| `FOAMAGENT_MAX_LOOP` | Maximum error-correction iterations | `25` |
| `FOAMAGENT_MAX_TIME_LIMIT` | Seconds before a solver run is terminated | `3600` |
| `FOAMAGENT_LOG_LEVEL` | Log verbosity. Logs go to stderr; stdout carries only the CLI's own markers | `INFO` |
| `FOAMAGENT_ROOT` | Where `database/` and `runs/` are looked up | the repository root |

### Input Writer Generation Mode

Set in `src/foamagent/config.py` via `input_writer_generation_mode`:

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
uv run python foambench_main.py \
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
uv sync --extra rag-local --extra direct-api --extra viz

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

If you have a ChatGPT/Codex subscription, you can authenticate via OAuth instead of an API key.
This provider is opt-in: it reads another tool's token cache from disk, so it is never selected
unless you ask for it explicitly.

1. Install the [Codex CLI](https://github.com/openai/codex) on your host machine.
2. Run `codex login` and choose **"Sign in with ChatGPT"**.
3. Verify the token cache exists: `ls ~/.codex/auth.json`
4. Mount it into the container:

```bash
docker run -it \
  -e FOAMAGENT_MODEL_PROVIDER=openai-codex \
  -e FOAMAGENT_MODEL_VERSION=gpt-5.3-codex \
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

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent

# The FAISS indices under database/ are stored with Git LFS. Without this step they are
# ~130-byte pointer files and retrieval fails.
git lfs install --local && git lfs pull

# Core only. Add the extras you need (see the table below).
uv sync --extra rag-local --extra direct-api --extra viz
```

The core install deliberately excludes anything heavy. Pick the extras for what you use:

| Extra | Provides | Needed when |
|---|---|---|
| `rag-local` | FAISS, sentence-transformers, torch (CPU) | Retrieval over the bundled tutorial database (the default) |
| `direct-api` | langchain-openai / -anthropic, openai, anthropic | Inference from inside this process |
| `viz` | PyVista | Post-processing images |
| `web` | FastAPI, uvicorn | The `app.py` web UI |
| `hpc` | boto3 | SLURM/HPC submission |
| `ollama`, `bedrock` | provider SDKs | Those providers |
| `all` | everything above | |

You also need **Foundation OpenFOAM v10** ([openfoam.org](https://openfoam.org)) installed and sourced for the default, fully validated runtime path. ESI OpenFOAM (`openfoam.com`) file generation is available as best-effort translation by setting `FOAMAGENT_OPENFOAM_FORK=esi`, but ESI execution and repair loops should be verified per case. Follow the [official Foundation v10 installation guide](https://openfoam.org/version/10/) and verify with:

```bash
echo $WM_PROJECT_DIR   # should print e.g. /opt/openfoam10
```

Then run:

```bash
uv run python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
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

## Development

```bash
uv sync                          # core + dev tools
uv run pytest -m "not integration" -q
uv run ruff check .
```

Unit tests require no API credentials, no network, no Docker, no Git LFS content, and no torch.
That constraint is what keeps `import foamagent` free of side effects, so please keep new unit
tests within it. Tests that need the bundled database are marked `integration` and are skipped by
default.

CI runs lint, the unit tests on Python 3.10 and 3.12, and a wheel build on every push and pull
request. It checks out without Git LFS content on purpose.

## Troubleshooting

| Problem | Solution |
|---|---|
| OpenFOAM environment not found | Ensure the intended OpenFOAM bashrc is sourced. The default validated path is Foundation OpenFOAM v10 ([openfoam.org](https://openfoam.org)); ESI OpenFOAM requires `FOAMAGENT_OPENFOAM_FORK=esi` and per-case verification |
| Database files missing | Ensure the full repo is cloned including `database/`. Docker image has these pre-built |
| `Index type 0x73726576 ("vers") not recognized` | The FAISS indices are unfetched Git LFS pointers. Run `git lfs install --local && git lfs pull` |
| Missing dependencies | `uv sync --all-extras` |
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
  title={Foam-Agent: Towards Automated Intelligent CFD Workflows},
  author={Yue, Ling and Somasekharan, Nithin and Zhang, Tingwen and Cao, Yadi and Chen, Zhangze and Di, Shimin and Pan, Shaowu},
  journal={arXiv preprint arXiv:2505.04997},
  year={2025}
}

@article{somasekharan2026cfdllmbench,
    title={CFDLLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics},
    author={Somasekharan, Nithin and Yue, Ling and Cao, Yadi and Li, Weichao and Emami, Patrick and Bhargav, Pochinapeddi Sai and Acharya, Anurag and Xie, Xingyu and Pan, Shaowu},
    journal={Journal of Data-centric Machine Learning Research},
    year={2026},
    url={https://openreview.net/forum?id=kTcH1MnkjY},
    note={}
}

```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=csml-rpi/Foam-Agent&type=timeline&legend=top-left)](https://www.star-history.com/#csml-rpi/Foam-Agent&type=timeline&legend=top-left)
