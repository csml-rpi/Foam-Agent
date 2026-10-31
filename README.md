# Foam-Agent <a href="https://arxiv.org/abs/2505.04997"><img src="https://img.shields.io/badge/arXiv-2505.04997-b31b1b.svg" alt="Paper"></a>

<p align="center">
  <img src="docs/assets/overview.png" alt="Foam-Agent System Architecture" width="800">
</p>

Foam-Agent automates OpenFOAM simulations from a natural language prompt or an existing case. Its agents plan the case, prepare meshes and input files, run locally or through Slurm, repair errors, and optionally visualize results.

- **OpenFOAM targets:** Foundation v10 by default, with a separate native ESI/OpenCFD v2006 target.
- **Tutorial retrieval:** FAISS indices provide references from the selected target's tutorials.
- **Model choice:** OpenAI, Codex subscription, Anthropic, Bedrock, Ollama, and DeepSeek backends.
- **MCP integration:** Tools for Claude Code, Cursor, and other MCP clients.

## Quick Start

Use Docker for the bundled OpenFOAM/Python runtime, or follow [local installation](docs/usage.md#local-installation) with Conda, Git LFS, and a matching OpenFOAM installation. Both paths require credentials for the selected LLM provider.

Run the Docker image with an explicitly selected model and save results on the host:

```bash
mkdir -p output
docker run -it \
  -e FOAMAGENT_MODEL_PROVIDER=openai \
  -e FOAMAGENT_MODEL_VERSION=gpt-5-mini \
  -e OPENAI_API_KEY=your-key-here \
  -v "$PWD/output:/home/openfoam/Foam-Agent/output" \
  --name foamagent \
  leoyue123/foamagent
```

Inside the container, edit `user_requirement.txt` to describe the geometry, boundary conditions, physical properties, and time settings, then run:

```bash
python foambench_main.py --output ./output/my-case --prompt_path ./user_requirement.txt
```

Input files, meshes, solver results, and logs are written to the host's `./output/my-case/`. Choose a fresh task directory for each run; replacing an existing managed output requires `--overwrite_output`.

The image includes Foundation OpenFOAM v10. It updates code at startup by default; to keep the code bundled in a release such as `leoyue123/foamagent:v2.0.0`, add `-e FOAMAGENT_SKIP_UPDATE=1`. See [authentication](docs/usage.md#authentication) for a Codex subscription or another provider.

An existing directory or ZIP can be used as input:

```bash
python foambench_main.py --output ./output/imported-case --case_path /path/to/case
```

Foam-Agent keeps an `original/` copy and runs in `work/`. Local execution clears previous results in `work/`; see [existing cases](docs/usage.md#existing-cases) for modification, archive selection, and output handling.

## Configuration

| Environment variable | Purpose | Default |
|---|---|---|
| `FOAMAGENT_MODEL_PROVIDER` | LLM backend | `openai-codex` |
| `FOAMAGENT_MODEL_VERSION` | Model identifier for that backend | `gpt-5.3-codex` |
| `FOAMAGENT_EMBEDDING_PROVIDER` | Tutorial embedding backend | `huggingface` |
| `FOAMAGENT_EMBEDDING_MODEL` | Model matching the FAISS indices | `Qwen/Qwen3-Embedding-0.6B` |
| `FOAMAGENT_OPENFOAM_TARGET` | Native target: `foundation-v10` or `esi-v2006` | Unset; Foundation v10 |

The default LLM uses a Codex OAuth cache. An API key requires selecting its provider and model explicitly, as in Quick Start. Default embeddings run locally without an API key; the first use may download model weights. Prebuilt tutorial indices come through Git LFS and must match the embedding model.

Settings live in `src/config.py`; see [configuration](docs/usage.md#configuration) for credentials, the legacy ESI translation switch, generation modes, and retry limits. Native ESI v2006 requires its matching runtime/image and corpus.

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
