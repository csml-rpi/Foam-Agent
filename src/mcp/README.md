# OpenFOAM Agent - FastMCP Server

This directory contains a modern MCP (Model Context Protocol) server implementation for the OpenFOAM Agent using FastMCP.

## Overview

The FastMCP-based server provides a clean, well-typed interface to OpenFOAM simulation capabilities, exposing the following core functions:

### Core Workflow Functions
- `create_openfoam_case` - Create a new OpenFOAM case from user requirements
- `plan_simulation` - Plan simulation structure and generate subtasks
- `generate_openfoam_files` - Generate OpenFOAM input files
- `prepare_mesh` - Prepare mesh for simulation
- `run_simulation` - Run simulation locally or on HPC
- `monitor_simulation` - Monitor simulation progress
- `review_results` - Review simulation results and suggest fixes
- `apply_fixes` - Apply suggested fixes to the case
- `generate_visualization` - Generate visualization artifacts

### Utility Functions
- `get_case_logs` - Retrieve case logs
- `check_job_status` - Check HPC job status
- `list_available_cases` - List all available cases

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure FastMCP is installed:
```bash
pip install fastmcp>=2.0.0
```

## Usage

### Running the MCP Server

The server can be run in different modes:

#### Standard I/O Mode (for MCP clients)
```bash
python -m src.mcp.fastmcp_server
```

#### HTTP Mode (for web clients)
```bash
python -m src.mcp.fastmcp_server --transport http --port 8080
```

### MCP Client Configuration

Add the following to your MCP client configuration:

```json
{
  "mcpServers": {
    "openfoam-agent": {
      "command": "python",
      "args": ["-m", "src.mcp.fastmcp_server"],
      "cwd": "/path/to/Foam-Agent",
      "env": {
        "PYTHONPATH": "/path/to/Foam-Agent/src"
      }
    }
  }
}
```

### Example Usage

Here's how to use the MCP server programmatically:

```python
from fastmcp import FastMCPClient

# Connect to the server
client = FastMCPClient("http://localhost:8080")

# Create a new case
case_response = await client.call_tool(
    "create_openfoam_case",
    {
        "user_requirement": "Create a simple fluid flow simulation around a cylinder",
        "output_dir": "/path/to/output"
    }
)

# Plan the simulation
plan_response = await client.call_tool(
    "plan_simulation",
    {
        "case_id": case_response["case_id"],
        "user_requirement": "Create a simple fluid flow simulation around a cylinder"
    }
)

# Generate OpenFOAM files
files_response = await client.call_tool(
    "generate_openfoam_files",
    {
        "case_id": case_response["case_id"],
        "subtasks": plan_response["subtasks"],
        "user_requirement": "Create a simple fluid flow simulation around a cylinder",
        "case_solver": "simpleFoam"
    }
)
```

## Architecture

The FastMCP server is built on top of the existing service layer:

```
MCP Client
    ↓
FastMCP Server (fastmcp_server.py)
    ↓
Service Layer (services/*.py)
    ↓
OpenFOAM Tools & LLM Services
```

### Key Benefits

1. **Type Safety**: All inputs and outputs are validated using Pydantic models
2. **Error Handling**: Comprehensive error handling with detailed logging
3. **Progress Reporting**: Real-time progress updates through MCP context
4. **Clean Interface**: Well-defined API that's easy to understand and use
5. **Extensibility**: Easy to add new tools and capabilities

### Input/Output Models

The server uses structured Pydantic models for all inputs and outputs:

- `CreateCaseRequest/Response` - Case creation
- `PlanRequest/Response` - Simulation planning
- `GenerateFilesRequest/Response` - File generation
- `MeshRequest/Response` - Mesh preparation
- `RunSimulationRequest/Response` - Simulation execution
- `MonitorRequest/Response` - Simulation monitoring
- `ReviewRequest/Response` - Result review
- `ApplyFixRequest/Response` - Fix application
- `VisualizationRequest/Response` - Visualization generation

## Configuration

The server uses the existing `Config` class from `config.py`. Key configuration options:

- `database_path`: Path to FAISS database
- `run_directory`: Directory for case outputs
- `max_loop`: Maximum retry loops
- `searchdocs`: Number of similar documents to retrieve
- `model_provider`: LLM provider (bedrock, openai, ollama)
- `model_version`: Specific model version

## Error Handling

The server provides comprehensive error handling:

1. **Validation Errors**: Input validation using Pydantic
2. **Service Errors**: Wrapped service layer exceptions
3. **File System Errors**: Proper handling of file operations
4. **LLM Errors**: Graceful handling of LLM service failures

All errors are logged through the MCP context and returned to the client with detailed information.

## Logging

The server uses FastMCP's built-in logging capabilities:

- `ctx.info()` - Informational messages
- `ctx.warning()` - Warning messages
- `ctx.error()` - Error messages
- `ctx.debug()` - Debug messages

## Development

### Adding New Tools

To add a new tool to the MCP server:

1. Define input/output models using Pydantic
2. Create the tool function with `@mcp.tool()` decorator
3. Add proper error handling and logging
4. Update documentation

### Testing

Run tests with:
```bash
pytest tests/test_fastmcp_server.py
```

### Code Style

The code follows Python best practices:
- Type hints for all functions
- Comprehensive docstrings
- Error handling with proper exceptions
- Clean separation of concerns

## Migration from Legacy Adapter

The new FastMCP server replaces the legacy `adapter.py` implementation with several improvements:

1. **Better Type Safety**: Pydantic models instead of raw dictionaries
2. **Cleaner API**: Well-defined request/response models
3. **Better Error Handling**: Comprehensive error handling and logging
4. **Progress Reporting**: Real-time progress updates
5. **Modern Architecture**: Built on FastMCP framework

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes the src directory
2. **Database Errors**: Check that the database_path exists and is accessible
3. **Permission Errors**: Ensure write permissions for output directories
4. **LLM Errors**: Verify LLM service configuration and credentials

### Debug Mode

Run with debug logging:
```bash
PYTHONPATH=/path/to/Foam-Agent/src python -m src.mcp.fastmcp_server --log-level debug
```

## Contributing

When contributing to the MCP server:

1. Follow the existing code style
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility where possible
5. Use type hints and proper error handling
