# Foam-Agent

<p align="center">
  <img src="overview.png" alt="Foam-Agent System Architecture" width="600">
</p>

<p align="center">
    <em>An End-to-End Composable Multi-Agent Framework for Automating CFD Simulation in OpenFOAM</em>
</p>

You can visit https://deepwiki.com/csml-rpi/Foam-Agent for a comprehensive introduction and to ask any questions interactively.

**Foam-Agent** is a multi-agent framework that automates the entire **OpenFOAM**-based CFD simulation workflow from a single natural language prompt. By managing the full pipeline—from meshing and case setup to execution and post-processing—Foam-Agent dramatically lowers the expertise barrier for Computational Fluid Dynamics. Evaluated on [FoamBench](https://arxiv.org/abs/2509.20374) of 110 simulation tasks, our framework achieves an **88.2% success rate**, demonstrating how specialized multi-agent systems can democratize complex scientific computing.

## Key Innovations

Our framework introduces three key innovations:

* **End-to-End Simulation Automation**: Foam-Agent manages the full simulation pipeline, including advanced pre-processing with a versatile Meshing Agent that handles external mesh files and generates new geometries via **Gmsh**, automatic generation of HPC submission scripts, and post-simulation visualization via **ParaView/PyVista**.
* **High-Fidelity Configuration**: We use a Retrieval-Augmented Generation (RAG) system based on a hierarchical index of case metadata. Generation proceeds in a dependency-aware order, ensuring consistency and accuracy across all configuration files.
* **Composable Service Architecture**: The framework exposes its core functions as discrete, callable tools using a Model Context Protocol (MCP). This allows for flexible integration with other agentic systems for more complex or exploratory workflows. Code will be released soon.

## Features
### 🔍 **Enhanced Retrieval System**
- **Hierarchical retrieval** covering case files, directory structures, and dependencies
- **Specialized vector index architecture** for improved information retrieval
- **Context-specific knowledge retrieval** at different simulation stages

### 🤖 **Multi-Agent Workflow Optimization**
- **Architect Agent** interprets requirements and plans file structures
- **Input Writer Agent** generates configuration files with consistency management
- **Runner Agent** executes simulations and captures outputs
- **Reviewer Agent** analyzes errors and proposes corrections

### 🛠️ **Intelligent Error Correction**
- **Error pattern recognition** for common simulation failures
- **Automatic diagnosis and resolution** of configuration issues
- **Iterative refinement process** that progressively improves simulation configurations

### 📐 **External Mesh File Support**
- **Custom mesh integration** with GMSH `.msh` files
- **Boundary condition specification** through natural language requirements
- **Currently supports** GMSH ASCII 2.2 format mesh files
- **Seamless workflow** from mesh import to simulation execution

**Example Usage:**
```bash
python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path ./user_requirement.txt --custom_mesh_path ./tandem_wing.msh
```

**Example Mesh File:** The `geometry.msh` file in this repository is taken from the [tandem wing tutorial](https://github.com/openfoamtutorials/tandem_wing) and demonstrates a 3D tandem wing simulation with NACA 0012 airfoils.

**Requirements Format:** In your `user_req_tandem_wing.txt`, describe the boundary conditions and physical parameters for your custom mesh. The agent will automatically detect the mesh type and generate appropriate OpenFOAM configuration files.

## Getting Started

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
```
If you prefer a stable version, please check the tags, and do
```bash
git checkout v1.1.0
```
Otherwise, FoamAgent will be at the latest version.

#### Foam-Agent Docker
You can skip manual installation steps [1](#1-clone-the-repository-and-install-dependencies) and [2](#2-install-and-configure-openfoam-v10) by using Docker, which provides a complete Foam-Agent environment with OpenFOAM-v10, the FoamAgent conda environment, and a pre-initialized database. **You must build the image from the Dockerfile** - no pre-built images are provided.

**Features:**
- **Fully automated setup**: Conda environment initialized and activated automatically
- **Pre-initialized database**: Database is built during image construction
- **Local code copy**: Foam-Agent code copied from your local directory (no GitHub access needed)
- **Auto-configured**: OpenFOAM and conda environments automatically sourced
- **Optimized build**: Large files (like `runs/` directory) excluded via `.dockerignore`

**Building the Docker image:**

From the repository root directory:
```bash
docker build -f docker/Dockerfile -t foamagent:latest .
```

**Build Notes:**
- Build time: ~15-20 minutes (includes database initialization)
- Image size: ~7-8 GB
- Your local code is copied (excluding `runs/`, `__pycache__/`, `.git/` per `.dockerignore`)

The Dockerfile automatically installs Miniconda, creates the conda environment, **pre-initializes the database**, and configures all necessary environment variables.

**Running the Container:**

```bash
docker run -it -e OPENAI_API_KEY=your-key-here -p 7860:7860 --name foamagent foamagent:latest
```

**Note:** If the container already exists (from a previous run), restart it using `docker start -i foamagent` instead of running this command again. To remove the existing container first, use `docker rm foamagent`.

When the container starts, you'll automatically get:
- ✅ OpenFOAM environment sourced
- ✅ Conda initialized
- ✅ FoamAgent conda environment activated
- ✅ Working directory set to `/home/openfoam/Foam-Agent`
- ✅ Database pre-initialized (done during build)
- ✅ Welcome message with usage instructions

**Run Foam-Agent:**
Once inside the container (everything is pre-configured):
```bash
python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path ./user_requirement.txt
```

**Updating Your Code:**
Since code is copied during build, update by rebuilding the image:
```bash
docker build -f docker/Dockerfile -t foamagent:latest .
docker rm foamagent  # if container exists
docker run -it -e OPENAI_API_KEY=your-key-here -p 7860:7860 --name foamagent foamagent:latest
```

**Restarting the Container:**
```bash
docker start -i foamagent
```

**Note:** In the Dockerfile (around lines 99-107), there is an option to exclude root access. However, the image size will increase to around 10-15 GB.

**Manual Installation (if not using Docker):**
```bash
conda env create -n FoamAgent -f environment.yml
conda activate FoamAgent
```

### 2. Install and configure OpenFOAM v10

Foam-Agent requires OpenFOAM v10. Please follow the official installation guide for your operating system:

- Official installation: [https://openfoam.org/version/10/](https://openfoam.org/version/10/)

Verify your installation with:

```bash
echo $WM_PROJECT_DIR
```
and the result should be
```
/opt/openfoam10
```
or something similar.

`WM_PROJECT_DIR` is an environment variable that comes with your OpenFOAM installation, indicating the location of OpenFOAM on your computer.

### 3. Database preprocessing (first-time setup)

Before running any workflow, you must initialize the OpenFOAM tutorial and command database. Run:

```bash
python init_database.py --openfoam_path $WM_PROJECT_DIR
```

This script automatically checks and runs all necessary preprocessing scripts in `database/script/` if the database files don't exist. It's safe to run multiple times as it skips already-generated files.

### 4. Run a demo workflow

```bash
python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path ./user_requirement.txt
```

You can also specify a custom mesh:

```bash
python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path ./user_requirement.txt --custom_mesh_path ./my_mesh.msh
```

#### Example user_requirement.txt

```
do a Reynolds-Averaged Simulation (RAS) pitzdaily simulation. Use PIMPLE algorithm. The domain is a 2D millimeter-scale channel geometry. Boundary conditions specify a fixed velocity of 10m/s at the inlet (left), zero gradient pressure at the outlet (right), and no-slip conditions for walls. Use timestep of 0.0001 and output every 0.01. Finaltime is 0.3. use nu value of 1e-5.
```

### 5. Configuration and environment variables

- Default configuration is in `src/config.py`. You can modify model provider, database path, and other parameters there.
- You must set the `OPENAI_API_KEY` environment variable if using OpenAI/Bedrock models.

### 6. Using Foam-Agent via MCP (Model Context Protocol)

Foam-Agent exposes its capabilities as an MCP server, allowing integration with AI coding assistants like Claude Code and Cursor.

#### Starting the MCP Server

**Option 1: Standard I/O Mode (for MCP clients)**
```bash
python -m src.mcp.fastmcp_server
```

**Option 2: HTTP Mode (for web clients)**
```bash
python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

#### Configuring Claude Code

1. Open Claude Code settings and navigate to MCP configuration
2. Copy the following configuration and paste it into your MCP settings file:

```json
{
  "mcpServers": {
    "openfoam-agent": {
      "command": "python",
      "args": ["-m", "src.mcp.fastmcp_server"],
      "cwd": "/path/to/Foam-Agent",
      "env": {
        "PYTHONPATH": "/path/to/Foam-Agent/src",
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

**Important:** Replace `/path/to/Foam-Agent` with your actual Foam-Agent installation path. If using conda, you may need to use the full path to the conda Python interpreter, e.g., `"/opt/conda/envs/FoamAgent/bin/python"` or `"$CONDA_PREFIX/bin/python"`.

#### Configuring Cursor

1. Open Cursor settings (Cmd/Ctrl + ,)
2. Search for "MCP" or navigate to **Settings → Features → MCP**
3. Click "Edit MCP Settings" or open the MCP configuration file
4. Copy and paste the same JSON configuration as shown above
5. Replace `/path/to/Foam-Agent` with your actual installation path
6. Save the configuration and restart Cursor to apply the changes

**Tip:** You can also use the example configuration file `mcp_config.json` in the repository root as a reference. Just update the paths to match your installation.

**Note:** Ensure that:
- The conda environment `FoamAgent` is activated, or
- The Python interpreter used has all required dependencies installed
- `OPENAI_API_KEY` is set in the environment or in the MCP configuration
- The database has been initialized (run `init_database.py`)

Once configured, you can use Foam-Agent tools directly in Claude Code or Cursor to create, run, and manage OpenFOAM simulations through natural language prompts.

### 7. Troubleshooting

- **OpenFOAM environment not found**: Ensure you have sourced the OpenFOAM bashrc and restarted your terminal.
- **Database not initialized**: Run `python init_database.py --openfoam_path $WM_PROJECT_DIR` to initialize the database.
- **Missing dependencies**: Recreate the environment: `conda env update -n FoamAgent -f environment.yml --prune` or `conda env remove -n FoamAgent && conda env create -n FoamAgent -f environment.yml`.
- **API key errors**: Ensure `OPENAI_API_KEY` is set in your environment.
- **MCP connection errors**: Verify the Python path in MCP configuration matches your installation, ensure the conda environment is accessible, and check that all dependencies are installed.

## Citation
If you use Foam-Agent in your research, please cite our paper:
```bibtex
@article{yue2025foam,
  title={Foam-Agent: Towards Automated Intelligent CFD Workflows},
  author={Yue, Ling and Somasekharan, Nithin and Cao, Yadi and Pan, Shaowu},
  journal={arXiv preprint arXiv:2505.04997},
  year={2025}
}

@article{yue2025foamagent,
  title={Foam-Agent 2.0: An End-to-End Composable Multi-Agent Framework for Automating CFD Simulation in OpenFOAM},
  author={Yue, Ling and Somasekharan, Nithin and Zhang, Tingwen and Cao, Yadi and Pan, Shaowu},
  journal={arXiv preprint arXiv:2509.18178},
  year={2025}
}

@article{somasekharan2025cfdllmbench,
  title={CFD-LLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics},
  author={Nithin Somasekharan, Ling Yue, Yadi Cao, Weichao Li, Patrick Emami, Pochinapeddi Sai Bhargav, Anurag Acharya, Xingyu Xie, Shaowu Pan},
  journal={arXiv preprint arXiv:2509.20374},
  year={2025}
}

```
