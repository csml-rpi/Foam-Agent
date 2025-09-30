## Foam-Agent Docker
A Foam-Agent image is available at [https://hub.docker.com/r/zhangt20/foamagent](https://hub.docker.com/r/zhangt20/foamagent). This image includes Foam-Agent, OpenFOAM-v10, and the openfoamAgent conda environment. 

### Building the Docker image
If you prefer do build the Docker image yourself, do the following steps.

1. In your terminal, navigate to one level above `Foam-Agent/`, and then do
```bash
cp Foam-Agent/docker/Dockerfile .
```

You file structure should now be
```
./
├── Dockerfile
└── Foam-Agent
```

2. Download the MiniConda installation script
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./Miniconda3-latest-Linux-x86_64.sh
```

3. Build the image
```
docker build --tag foamagent:1.0.1 .
```

Afterwards, follow the instruction at [https://hub.docker.com/r/zhangt20/foamagent](https://hub.docker.com/r/zhangt20/foamagent) to create a container.

The building process should take around 15 minutes, and the image size should be between 7-8 GB.

## Note
In [Dockerfile](Dockerfile) line 72-80, there is an option to exclude root access. However, the image size will increase to around 10-15 GB.

