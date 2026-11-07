# Foam-Agent <a href="https://arxiv.org/abs/2505.04997"><img src="https://img.shields.io/badge/arXiv-2505.04997-b31b1b.svg" alt="Paper"></a>

Foam-Agent automates OpenFOAM simulations from a natural language prompt or an existing case. Its agents plan the case, prepare meshes and input files, review them before execution, run locally or through Slurm, repair errors, and optionally visualize results.

- **OpenFOAM targets:** Foundation v10 by default, with a separate native ESI/OpenCFD v2006 target.
- **Tutorial retrieval:** FAISS indices provide references from the selected target's tutorials.
- **Model choice:** OpenAI, Codex subscription, Anthropic, Bedrock, Ollama, and DeepSeek backends.
- **MCP integration:** Tools for Claude Code, Cursor, and other MCP clients.

<a id="quick-start"></a>

## Quick Start (Recommended: Docker)

Docker bundles Foundation OpenFOAM v10, Python dependencies, tutorial indices,
and the default Qwen 0.6B embedding weights. No host Conda, OpenFOAM, or GPU is
needed. Configure an API-key provider or a Codex subscription using the
[authentication guide](docs/usage.md#authentication). For Ollama, follow the
[local setup](docs/usage.md#ollama): it requires optional Python packages and
a running Ollama service, which the default images do not include.

| Your environment | Installation |
|---|---|
| A host with Docker | Use the Docker steps below |
| An existing container or OpenFOAM installation | [Local Python installation](docs/usage.md#local-installation) |
| Native ESI/OpenCFD v2006 | [Build its matching image](docs/usage.md#docker-builds) |

Build the current source on Linux amd64 (or an amd64 Docker VM):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
git lfs install
git lfs pull --include="database/foundation-v10/raw/*,database/foundation-v10/faiss/Qwen_Qwen3-Embedding-0.6B/**,examples/tandem-wing/*.msh"
docker build --platform linux/amd64 -f docker/Dockerfile -t foamagent:local .
cp .env.example .env
```

Edit `.env` with your provider, model and API key. Then check the installation
and run the bundled example, keeping results on the host:

```bash
mkdir -p output
docker run --rm --env-file .env foamagent:local \
  python scripts/doctor.py --check-embedding --check-llm
docker run --rm --env-file .env \
  -v "$PWD/output:/home/openfoam/Foam-Agent/output" \
  foamagent:local bash examples/quickstart/run.sh
```

The diagnostic checks LLM access with a small request; the example uses the existing
`user_requirement.txt`. Files and logs are saved to `output/quickstart/`.
Use a fresh output directory for another run. See [authentication](docs/usage.md#authentication)
for a Codex subscription or another provider.

Locally built images run their bundled code and assets without startup updates.
See [Docker builds](docs/usage.md#docker-builds) for native targets and older image
compatibility.

To use your own prompt, mount its directory read-only and pass the container path:

```bash
docker run --rm --env-file .env \
  -v "$PWD/inputs:/inputs:ro" \
  -v "$PWD/output:/home/openfoam/Foam-Agent/output" \
  foamagent:local python foambench_main.py \
  --prompt_path /inputs/requirements.txt --output ./output/my-case
```

An [existing directory or ZIP](docs/usage.md#existing-cases) is also supported with
`--case_path`. Foam-Agent keeps an `original/` copy and runs in `work/`.

## Configuration

| Environment variable | Purpose | Default |
|---|---|---|
| `FOAMAGENT_MODEL_PROVIDER` | LLM backend | `openai-codex` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier for that backend | `gpt-5.3-codex` |
| `FOAMAGENT_EMBEDDING_PROVIDER` | Tutorial embedding backend | `huggingface` |
| `FOAMAGENT_EMBEDDING_MODEL` | Model identity matching the FAISS indices | `Qwen/Qwen3-Embedding-0.6B` |
| `FOAMAGENT_EMBEDDING_MODEL_PATH` | Optional local copy of that Hugging Face model | Unset |
| `FOAMAGENT_OPENFOAM_TARGET` | Native target: `foundation-v10` or `esi-v2006` | Unset; Foundation v10 |

The default LLM uses a Codex OAuth cache. An API key requires selecting its provider and model explicitly, as in Quick Start. Docker builds bundle the default embedding weights. Local Python installations may download them on first use; prebuilt tutorial indices come through Git LFS and must match the embedding model. See [installation checks and offline setup](docs/usage.md#installation-checks-and-offline-use).

Choose an execution mode with `--execution_mode` (or `Config.execution_mode`):

| Mode | File generation | Before the first execution |
|---|---|---|
| `careful` (default) | Sequential, sharing generated files as context | LLM static review |
| `fast` | Independent parallel generation | Run directly |

Both modes repair runtime errors. Static review requests only definite, necessary
corrections, with up to `--max_static_review_retries 10` rounds, separate from
runtime `max_loop`. Reaching the limit or making no edits proceeds to Run;
review service failures stop the workflow. Later runs use runtime review only.
`--input_writer_generation_mode sequential_dependency|parallel_no_context`
overrides generation independently. See [configuration and reports](docs/usage.md#execution-modes-and-static-review).

Settings live in `src/config.py`; see [configuration](docs/usage.md#configuration) for credentials, the legacy ESI translation switch, and retry limits. Native ESI v2006 requires its matching runtime/image and corpus.

## MCP Integration

After [local installation](docs/usage.md#local-installation), register the server with Claude Code:

```bash
claude mcp add foamagent -- foamagent-mcp
```

The repository also includes the `/foam` Claude Code skill. See the [MCP guide](docs/mcp.md) for other clients, HTTP/Docker setup, tool arguments, and workflow limits.

## Usage and Project Layout

- [Usage guide](docs/usage.md): configuration, existing cases, custom meshes, native targets, Slurm, and Docker builds.
- [MCP guide](docs/mcp.md): client setup and available tools.
- [DeepWiki](https://deepwiki.com/csml-rpi/Foam-Agent): project introduction and interactive questions.

| Directory | Contents |
|---|---|
| `src/` | Workflow, services, model clients, and MCP server |
| `database/` | Target-specific tutorial corpora, FAISS indices, and builders |
| `examples/` | Sample inputs, including the tandem-wing mesh and prompt |
| `docs/` | User guides and illustrations |
| `scripts/` | Corpus checks and translation verification tools |
| `tests/` | Automated regression tests |
| `docker/` | Runtime images and entrypoints |

<p align="center">
  <img src="docs/assets/overview.png" alt="Foam-Agent System Architecture" width="800">
</p>

## Reported Benchmark Results

These are the project's recorded [FoamBench](https://arxiv.org/abs/2509.20374) results, not guarantees for the current working tree, native ESI v2006, or arbitrary imported cases.

| Framework | Model | Basic | Advanced |
|---|---|---:|---:|
| FoamAgent 2.0.0 (10 loops) | Opus 4.6 | 85.45% | 100% |
| FoamAgent 2.0.0 (25 loops) | Opus 4.6 | 100% | 100% |
| FoamAgent 2.0.0 (25 loops) | Sonnet 4.6 | 87.88% | 75.00% |
| FoamAgent 2.0.0 (25 loops) | Haiku 4.6 | 54.55% | 37.50% |
| FoamAgent 2.0.0 (25 loops) | gpt-5.4 | 45.45% | 75.00% |
| FoamAgent 2.0.0 (25 loops) | gpt-5.3-codex | 54.55% | 62.50% |

The highest reported scores in this table use **Anthropic Claude Opus 4.6**; model selection remains configurable.

## Community

Chinese-speaking users can add **ZDSJTUCFD** on WeChat to request an invitation to the Foam-Agent community.

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
