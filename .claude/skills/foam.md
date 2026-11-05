---
name: foam
description: Run a local OpenFOAM CFD simulation from a natural language prompt using Foam-Agent's MCP tools and configured native target
user_invocable: true
---

# Foam-Agent: Automated CFD Simulation

You are orchestrating a complete OpenFOAM CFD simulation workflow using the Foam-Agent MCP tools. Follow these steps in order:

## Input

The user provides: `$ARGUMENTS`

If no arguments are provided, ask the user to describe their CFD simulation requirements (e.g., "Simulate lid-driven cavity flow at Re=1000" or "Simulate flow over a backward-facing step").

## Workflow

### Step 1: Plan
Call the `plan` MCP tool with the user's requirement:
```
plan({ request: { user_requirement: "<user's description>" } })
```
Show the user the planned case name, solver, domain, category, and subtask list. Ask for confirmation before proceeding.

### Step 2: Generate Files
Call the `input_writer` MCP tool with all fields from the plan response:
```
input_writer({
  request: {
    case_name: "<from plan>",
    subtasks: <from plan>,
    user_requirement: "<user's description>",
    case_solver: "<from plan>",
    case_domain: "<from plan>",
    case_category: "<from plan>"
  }
})
```
Report the case directory and number of generated files.

### Step 3: Run Simulation
Call the `run` MCP tool:
```
run({ request: { case_dir: "<from input_writer>", timeout: 3600 } })
```
Report the status (success/failed) and any errors.

### Step 4: Error Correction Loop (if errors)
If the simulation failed, enter the review-fix-rerun loop (max 5 iterations):

1. Call `review` with `request: { case_dir, errors, user_requirement }`
2. Show the user the error analysis
3. Call `apply_fixes` with `request: { case_dir, error_logs: <run errors>, review_analysis: <review analysis>, user_requirement }`
4. Call `run` again
5. Repeat until success or max iterations reached

### Step 5: Visualization (optional)
If the simulation succeeded and the user wants visualization, call:
```
visualization({ request: { case_dir: "<case_dir>", quantity: "<e.g., pressure, velocity>" } })
```

## Guidelines
- Use the server's configured target: Foundation OpenFOAM v10 by default, or native ESI/OpenCFD v2006 with `FOAMAGENT_OPENFOAM_TARGET=esi-v2006` and its matching runtime/corpus. Legacy `FOAMAGENT_OPENFOAM_FORK=esi` is a separate best-effort translation mode.
- These standalone tools run locally and do not expose meshing or HPC. Use the CLI graph for generated-case mesh/HPC routing, and `run_case` for an existing case or ZIP. See [MCP usage](../../docs/mcp.md).
- Always show the plan to the user and get confirmation before generating files
- Report progress at each step
- If any step fails after retries, explain the error clearly and suggest manual intervention
- The case directory contains all OpenFOAM files - the user can inspect/modify them directly
