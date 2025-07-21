# Foam-Agent

<p align="center">
  <img src="overview.png" alt="Foam-Agent System Architecture" width="600">
</p>

You can visit https://deepwiki.com/csml-rpi/Foam-Agent for a comprehensive introduction and to ask any questions interactively.

## Introduction
**Foam-Agent** is a multi-agent framework that automates complex OpenFOAM-based CFD simulation workflows from natural language inputs. By leveraging advanced AI techniques, Foam-Agent significantly lowers the expertise barrier for Computational Fluid Dynamics while maintaining modeling accuracy.

Our framework offers three key innovations:
- **Hierarchical multi-index retrieval system** with specialized indices for different simulation aspects
- **Dependency-aware file generation system** ensuring consistency across configuration files
- **Iterative error correction mechanism** that diagnoses and resolves simulation failures without human intervention

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

## Getting Started

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
git checkout v1.0.0
conda env create -f environment.yml
conda activate openfoamAgent
```

### 2. Install and configure OpenFOAM v10

Foam-Agent requires OpenFOAM v10. Please follow the official installation guide for your operating system:

- Official installation: [https://openfoam.org/download/10/](https://openfoam.org/download/10/)

After installation, make sure to source the OpenFOAM environment before running any workflow:

```bash
source /path/to/OpenFOAM-10/etc/bashrc
```

Verify your installation with:

```bash
foamInstallationTest
```

### 3. Database preprocessing (first-time setup)

Before running any workflow, you must preprocess the OpenFOAM tutorial and command database. This can be done automatically or manually.

#### Recommended: Automatic preprocessing

```bash
python foambench_main.py --openfoam_path /path/to/OpenFOAM-10 --output ./output --prompt_path ./user_requirement.txt
```

This script will automatically run all necessary preprocessing scripts in `database/script/` and then launch the main workflow.

#### Manual preprocessing (advanced)

If you prefer to run preprocessing scripts manually, execute the following:

```bash
python database/script/tutorial_parser.py --output_dir=./database/raw --wm_project_dir=/path/to/OpenFOAM-10
python database/script/faiss_command_help.py --database_path=./database
python database/script/faiss_allrun_scripts.py --database_path=./database
python database/script/faiss_tutorials_structure.py --database_path=./database
python database/script/faiss_tutorials_details.py --database_path=./database
```

### 4. Run a demo workflow

#### Option 1: Automated benchmark (recommended)

```bash
python foambench_main.py --openfoam_path /path/to/OpenFOAM-10 --output ./output --prompt_path ./user_requirement.txt
```

#### Option 2: Directly run the main agent

```bash
python src/main.py --prompt_path ./user_requirement.txt --output_dir ./output
```

- You can also specify a custom mesh:

```bash
python src/main.py --prompt_path ./user_requirement.txt --output_dir ./output --custom_mesh_path ./my_mesh.msh
```

#### Example user_requirement.txt

```
do a Reynolds-Averaged Simulation (RAS) pitzdaily simulation. Use PIMPLE algorithm. The domain is a 2D millimeter-scale channel geometry. Boundary conditions specify a fixed velocity of 10m/s at the inlet (left), zero gradient pressure at the outlet (right), and no-slip conditions for walls. Use timestep of 0.0001 and output every 0.01. Finaltime is 0.3. use nu value of 1e-5.
```

### 5. Configuration and environment variables

- Default configuration is in `src/config.py`. You can modify model provider, database path, and other parameters there.
- You must set the `OPENAI_API_KEY` environment variable if using OpenAI/Bedrock models.

### 6. Troubleshooting

- **OpenFOAM environment not found**: Ensure you have sourced the OpenFOAM bashrc and restarted your terminal.
- **Database not initialized**: Make sure you have run `foambench_main.py` or all scripts in `database/script/`.
- **Missing dependencies**: After activating the environment, run `pip install -r requirements.txt` if needed.
- **API key errors**: Ensure `OPENAI_API_KEY` is set in your environment.

## Citation
If you use Foam-Agent in your research, please cite our paper:
```bibtex
@article{yue2025foam,
  title={Foam-Agent: Towards Automated Intelligent CFD Workflows},
  author={Yue, Ling and Somasekharan, Nithin and Cao, Yadi and Pan, Shaowu},
  journal={arXiv preprint arXiv:2505.04997},
  year={2025}
}
```
