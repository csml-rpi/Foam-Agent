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
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt --custom_mesh_path ./tandem_wing.msh
```

**Example Mesh File:** The `geometry.msh` file in this repository is taken from the [tandem wing tutorial](https://github.com/openfoamtutorials/tandem_wing) and demonstrates a 3D tandem wing simulation with NACA 0012 airfoils.

**Requirements Format:** In your `user_req_tandem_wing.txt`, describe the boundary conditions and physical parameters for your custom mesh. The agent will automatically detect the mesh type and generate appropriate OpenFOAM configuration files.

## Getting Started

### 1. Quick Start with Docker (Recommended)

Foam-Agent is fully pre-installed in the Docker image `leoyue123/foamagent`. This is the easiest way to get an end-to-end OpenFOAM + Foam-Agent environment.

#### 1.1 Pull the image

```bash
docker pull leoyue123/foamagent
```

#### 1.2 Start a container

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  --name foamagent \
  leoyue123/foamagent
```

Inside the container you automatically get:
- **OpenFOAM v10** installed and sourced
- **Conda** initialized and the `FoamAgent` environment activated
- **Working directory** set to `/home/openfoam/Foam-Agent`
- **Database files** pre-initialized and ready to use

#### 1.3 Prepare your `user_requirement.txt`

- **Default location inside Docker**: `/home/openfoam/Foam-Agent/user_requirement.txt`
- **Edit directly in the container** (example):
  ```bash
  nano user_requirement.txt
  ```
- **Or mount a prompt file from the host**:
  ```bash
  docker run -it \
    -e OPENAI_API_KEY=your-key-here \
    -p 7860:7860 \
    -v /absolute/path/to/my_requirement.txt:/home/openfoam/Foam-Agent/user_requirement.txt \
    --name foamagent \
    leoyue123/foamagent
  ```

**Example content of `user_requirement.txt`:**

```text
do a Reynolds-Averaged Simulation (RAS) pitzdaily simulation. Use PIMPLE algorithm. The domain is a 2D millimeter-scale channel geometry. Boundary conditions specify a fixed velocity of 10m/s at the inlet (left), zero gradient pressure at the outlet (right), and no-slip conditions for walls. Use timestep of 0.0001 and output every 0.01. Finaltime is 0.3. use nu value of 1e-5.
```

#### 1.4 (Optional) Provide a custom mesh

If you have a Gmsh `.msh` file on the host, mount it into the container and point Foam-Agent to it:

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  -v /absolute/path/to/my_mesh.msh:/home/openfoam/Foam-Agent/my_mesh.msh \
  --name foamagent \
  leoyue123/foamagent
```

Then, inside the container, call:

```bash
python foambench_main.py \
  --output ./output \
  --prompt_path ./user_requirement.txt \
  --custom_mesh_path ./my_mesh.msh
```

#### 1.5 Run a simulation inside Docker

From `/home/openfoam/Foam-Agent` in the container:

```bash
# Basic run
python foambench_main.py \
  --output ./output \
  --prompt_path ./user_requirement.txt

# With a custom mesh (if provided)
python foambench_main.py \
  --output ./output \
  --prompt_path ./user_requirement.txt \
  --custom_mesh_path ./my_mesh.msh
```

To restart and reuse the same container later:

```bash
docker start -i foamagent
```

#### 1.6 (Optional) Build the image from source

If you prefer to build the Docker image yourself from this repository:

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
docker build -f docker/Dockerfile -t foamagent:latest .
```

Run the locally built image:

```bash
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -p 7860:7860 \
  --name foamagent \
  foamagent:latest
```

### 2. Configuring LLM provider and model

Foam-Agent selects the LLM backend and model from `src/config.py`. Inside the container, this file is at `/home/openfoam/Foam-Agent/src/config.py`.

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    ...
    model_provider: str = "openai"  # ["openai", "ollama", "bedrock"]
    # model_version can be e.g. "gpt-5-mini", "deepseek-r1:32b-qwen-distill-fp16", "qwen2.5:32b-instruct"
    model_version: str = "gpt-5-mini"
    temperature: float = 1.0
```

To change the LLM configuration inside Docker:

```bash
docker exec -it foamagent bash
cd /home/openfoam/Foam-Agent
nano src/config.py
```

- **OpenAI (via `OPENAI_API_KEY`)**:
  - **model_provider**: `"openai"`
  - **model_version**: e.g. `"gpt-5-mini"` or another supported OpenAI-compatible model name
- **AWS Bedrock**:
  - **model_provider**: `"bedrock"`
  - **model_version**: your Bedrock application ARN
- **Ollama (local models)**:
  - **model_provider**: `"ollama"`
  - **model_version**: the local model name, e.g. `"qwen2.5:32b-instruct"`


### 3. Using Foam-Agent via MCP (with Docker)

Foam-Agent exposes its capabilities as an MCP server. The recommended workflow is:
1. Run Foam-Agent in Docker  
2. Start the MCP server inside the container  
3. Point Claude Code or Cursor to that server

#### 3.1 Start the MCP server inside the container

Make sure the container is running:

```bash
docker start -i foamagent
```

In a separate terminal, attach and start the MCP server:

```bash
docker exec -it foamagent bash
cd /home/openfoam/Foam-Agent

# HTTP mode (if your MCP client supports HTTP transport)
python -m src.mcp.fastmcp_server --transport http --host 0.0.0.0 --port 7860
```

If you are running Docker on a remote server, make sure port `7860` is reachable from your local machine
(for example, by using SSH port forwarding or a proper port mapping such as `-p 7860:7860` when starting the container).

#### 3.2 Configure Claude Code / Cursor (HTTP mode)

In your MCP configuration file, use a simple HTTP-based entry like:

```json
{
  "mcpServers": {
    "foam-agent": {
      "url": "http://localhost:7860"
    }
  }
}
```

Adjust `localhost` and the port if your server is running on a different host or port.

#### 3.3 (Optional) stdio mode via Docker

If your MCP client prefers stdio instead of HTTP, you can still use the original `docker exec` style configuration.
Refer to the Foam-Agent repository documentation for the stdio example.

#### 3.4 Configure Cursor (same MCP config)

1. Open Cursor settings (Cmd/Ctrl + ,)
2. Search for **"MCP"** or navigate to **Settings → Features → MCP**
3. Click **"Edit MCP Settings"** or open the MCP configuration file
4. Paste the JSON configuration from section **3.2**
5. Save and restart Cursor

Once configured, you can call Foam-Agent tools directly from Claude Code or Cursor to plan cases, write input files, run simulations, and visualize results through natural-language commands.

### 4. Manual Installation (without Docker, optional)

If you prefer not to use Docker, you can install Foam-Agent and its dependencies manually.

#### 4.1 Clone the repository and create the environment

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
conda env create -n FoamAgent -f environment.yml
conda activate FoamAgent
```

#### 4.2 Install and configure OpenFOAM v10

Foam-Agent requires OpenFOAM v10. Please follow the official installation guide for your operating system:

- Official installation: [https://openfoam.org/version/10/](https://openfoam.org/version/10/)

Verify your installation with:

```bash
echo $WM_PROJECT_DIR
```

The result should be something like:

```text
/opt/openfoam10
```

`WM_PROJECT_DIR` is an environment variable that comes with your OpenFOAM installation, indicating the location of OpenFOAM on your computer.

#### 4.3 Run a demo workflow (manual setup)

From the repository root:

```bash
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt
```

You can also specify a custom mesh:

```bash
python foambench_main.py \
  --output ./output \
  --prompt_path ./user_requirement.txt \
  --custom_mesh_path ./my_mesh.msh
```

### 5. Configuration and environment variables (summary)

- Default configuration (including LLM provider and model) is in `src/config.py`.
- You must set the `OPENAI_API_KEY` environment variable if using OpenAI/Bedrock-style models.
- For AWS Bedrock or other cloud providers, ensure their credentials are configured in your environment.

### 6. Troubleshooting

- **OpenFOAM environment not found**: Ensure you have sourced the OpenFOAM bashrc and restarted your terminal (for manual installations), or use the provided Docker image where this is pre-configured.
- **Database files missing**: Database files are included in the repository (and in the Docker image). If they are missing, ensure you have cloned the complete repository including the `database/` directory.
- **Missing dependencies**: Recreate the environment: `conda env update -n FoamAgent -f environment.yml --prune` or `conda env remove -n FoamAgent && conda env create -n FoamAgent -f environment.yml`.
- **API key errors**: Ensure `OPENAI_API_KEY` is set in your environment or in the MCP configuration.
- **MCP connection errors**: Verify that the Docker container is running, the MCP command in your configuration matches your setup, and that all dependencies are installed.

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
