# --- Stage 1: Builder ---
FROM openfoam/openfoam10-paraview510:latest AS builder
USER root

# Install base utilities (including ca-certificates for SSL)
RUN apt-get update \
    && apt-get install -y \
       build-essential \
       wget \
       git \
       python3 \
       python3-pip \
       libboost-all-dev \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the Miniconda installer into the image
COPY Miniconda3-latest-Linux-x86_64.sh /tmp/

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR" \
    && rm -f /tmp/Miniconda3-latest-Linux-x86_64.sh \
    && ln -s "$CONDA_DIR/bin/conda" /usr/bin/conda \
    && conda --version

# Copy FoamAgent sources into working dir
ENV FoamAgent_PATH="/home/openfoam/Foam-Agent"
WORKDIR $FoamAgent_PATH
COPY Foam-Agent/ ./

# Switch to bash to source OpenFOAM environment
SHELL ["/bin/bash", "-c"]

# Create the Foam-Agent environment
RUN conda config --add channels conda-forge && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create --file environment.yml --yes

# Source OpenFOAM and verify
RUN source /opt/openfoam10/etc/bashrc && \
    echo "WM_PROJECT_DIR: $WM_PROJECT_DIR"

RUN conda clean --all --yes && \
    rm -rf "$CONDA_DIR/pkgs" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Stage 2: Runtime ---
FROM openfoam/openfoam10-paraview510:latest
USER root

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y \
       git \
       libboost-all-dev \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Environment variables in a FoamAgent container
ENV CONDA_DIR="/opt/conda"
ENV PATH=$CONDA_DIR/bin:$PATH
ENV FoamAgent_PATH="/home/openfoam/Foam-Agent"

# Copy Conda runtime and FoamBench from builder
COPY --from=builder $CONDA_DIR $CONDA_DIR
COPY --from=builder $FoamAgent_PATH $FoamAgent_PATH

# ============================================================================
# Uncomment to exclude root access but create a larger image
# ============================================================================
# # Ensure openfoam user owns entire home directory
# RUN chown -R openfoam:openfoam /home/openfoam

# # Switch to non-root user
# USER openfoam
# ============================================================================

# Set default working directory
WORKDIR /home/openfoam/

ENTRYPOINT ["/bin/bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && exec bash"]
