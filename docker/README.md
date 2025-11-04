## Foam-Agent Docker

This Docker setup provides a complete Foam-Agent environment with OpenFOAM-v10 and the FoamAgent conda environment. **You must build the image from the Dockerfile** - no pre-built images are provided.

### Features
- **Fully automated setup**: Conda environment is automatically initialized and activated
- **Direct git clone**: Foam-Agent is cloned directly from GitHub during build
- **Easy updates**: Simply run `git pull` inside the container to get the latest code
- **Auto-configured**: OpenFOAM and conda environments are automatically sourced

### Building the Docker image

**Required**: Build the image from the Dockerfile:

1. Navigate to the directory containing the Dockerfile:
```bash
cd Foam-Agent/docker
```

2. Build the image (everything is automated - no need to download Miniconda or copy files):
```bash
docker build --platform linux/amd64 --tag foamagent:latest .
```

**Note**: The building process should take around 15 minutes, and the final image size should be between 7-8 GB. The Dockerfile will automatically:
- Download and install Miniconda
- Clone the latest Foam-Agent code from GitHub
- Create and configure the conda environment
- Set up all necessary environment variables

### Running the Container
```bash
docker run -it -e OPENAI_API_KEY=your-key-here --name foamagent foamagent:latest
```

When the container starts, you'll automatically get:
- ✅ OpenFOAM environment sourced
- ✅ Conda initialized
- ✅ FoamAgent conda environment activated
- ✅ Working directory set to `/home/openfoam/Foam-Agent`
- ✅ Welcome message with usage instructions


#### Run Foam-Agent
Once inside the container (everything is already set up):
```bash
python foambench_main.py --openfoam_path $WM_PROJECT_DIR --output ./output --prompt_path ./user_requirement.txt
```

#### Update to Latest Foam-Agent Code
To get the latest code from GitHub:
```bash
cd /home/openfoam/Foam-Agent
git pull
```

### Restarting the Container
```bash
docker start -i foamagent
```

