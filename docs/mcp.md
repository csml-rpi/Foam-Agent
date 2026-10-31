# MCP Guide

Foam-Agent exposes planning, file generation, execution, repair, and visualization as MCP tools. Complete [local installation](usage.md#local-installation) and [authentication](usage.md#authentication) before starting the server. Select an [OpenFOAM target](usage.md#openfoam-targets-and-corpora) with its matching runtime and corpus.

## Client Setup

Register with Claude Code:

```bash
claude mcp add foamagent -- foamagent-mcp
```

For Cursor, use `.cursor/mcp.json`; other MCP clients accept the same server configuration:

```json
{
  "mcpServers": {
    "foamagent": { "command": "foamagent-mcp" }
  }
}
```

The console command defaults to stdio. To serve HTTP:

```bash
foamagent-mcp --transport http --host 0.0.0.0 --port 7860
```

For Docker:

```bash
mkdir -p runs
docker run -it \
  -e FOAMAGENT_MODEL_PROVIDER=openai \
  -e FOAMAGENT_MODEL_VERSION=gpt-5-mini \
  -e OPENAI_API_KEY=your-key-here \
  -v "$PWD/runs:/home/openfoam/Foam-Agent/runs" \
  -p 7860:7860 \
  leoyue123/foamagent \
  python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

Point the HTTP client at the server:

```json
{
  "mcpServers": {
    "foamagent": { "url": "http://localhost:7860/mcp" }
  }
}
```

For a remote host, use a reachable address or SSH port forwarding. The module entry point `python -m src.mcp.fastmcp_server` retains its HTTP/localhost defaults. The compatibility entry point `python -m src.mcp.start_mcp` reads `MCP_TRANSPORT`, `MCP_HOST`, and `MCP_PORT` (defaults: stdio, `0.0.0.0`, 7860).

## Tools and Arguments

| Tool | Purpose |
|---|---|
| `plan` | Analyze requirements and select case metadata and files |
| `input_writer` | Generate input files and `Allrun` using the selected target |
| `run` | Execute a local case and return errors and log paths |
| `review` | Diagnose supplied errors using case files and references |
| `apply_fixes` | Build a file-scoped repair plan and rewrite files |
| `run_case` | Run or modify an existing case/ZIP through the shared workflow graph |
| `visualization` | Generate PyVista plots |

Every tool takes its payload under `request`. For example:

```json
{
  "request": {
    "user_requirement": "Simulate lid-driven cavity flow at Re=1000"
  }
}
```

For an existing case, call `run_case` with:

```json
{
  "request": {
    "case_path": "/path/to/case",
    "output_dir": "/path/to/new-output",
    "user_requirement": "Set the final time to 5"
  }
}
```

`user_requirement`, `case_subdir`, `custom_mesh_path`, `openfoam_target`, and `overwrite_output` are optional for `run_case`. The target override applies to that invocation; individual tools use the server's configuration. See [existing cases](usage.md#existing-cases) for import, cleanup, and planning behavior.

All paths refer to the server's filesystem. Generated `input_writer` cases default to `runs/<case_name>/` in the repository; the Docker mount above exposes those results on the host. For `run_case`, mount its inputs and use an `output_dir` inside a mounted writable directory to retain the results.

## Workflows

For generated cases, the client calls `plan` → `input_writer` → `run`, then `review` → `apply_fixes` → `run` as needed, with optional visualization. The client controls the repair limit. `apply_fixes` requires `case_dir`, `error_logs`, `review_analysis`, and `user_requirement`; `no_changes` indicates that no files were updated.

Individual tools do not execute the complete graph. `run` is local-only, and there is no standalone meshing or HPC tool. Use the CLI graph for generated-case Gmsh/custom-mesh and HPC routing; use `run_case` for the imported-case graph, including its configured repair limit. [HPC monitoring limits](usage.md#hpc-execution-and-monitoring) apply to that graph.

Standalone `review` reads the case and supplied errors, but uses fixed `simpleFoam`/`fluid`/`tutorial` metadata and the directory basename for retrieval rather than inferring the solver.

`run_case` returns `status`, `message`, `visualization_error`, `case_dir`, `report_dir`, `errors`, and `termination_reason`. Simulation success with failed visualization is `partial_success` with message `Simulation completed successfully, but visualization failed.` Standalone `visualization` raises an error on failure.

## Claude Code Skill

Cloned repositories include the [`/foam` skill](../.claude/skills/foam.md):

```text
/foam Simulate lid-driven cavity flow at Re=1000
```

The skill orchestrates individual tools with up to five repair attempts and optional visualization. It uses the server's selected target; it does not add meshing or HPC tools.
