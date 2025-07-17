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
Clone the repository and install dependencies:
```bash
git clone https://github.com/csml-rpi/Foam-Agent.git
cd Foam-Agent
conda env create -f environment.yml
```

Run the benchmark with logging:
```bash
python foambench_main.py --openfoam_path /opt/openfoam10/ --prompt_path ./user_requirement.txt --output ./output 2>&1 | tee foambench.log
```

We will release our dataset as soon as the relevant license review is completed. This project will be continuously updated to integrate more advanced large language models for improved performance and to support additional functionalities. We encourage everyone to follow this GitHub repository and our team's progress. We firmly believe in the inevitable arrival of AGI and remain committed to advancing automation in the CFD domain. We welcome collaboration and communication from researchers and practitioners in the field.

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
