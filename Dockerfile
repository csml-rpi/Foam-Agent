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

# Download and install Miniconda directly
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR" \
    && rm -f /tmp/Miniconda3-latest-Linux-x86_64.sh \
    && ln -s "$CONDA_DIR/bin/conda" /usr/bin/conda \
    && conda --version

# Clone Foam-Agent repository directly
ENV FoamAgent_PATH="/home/openfoam/Foam-Agent"
WORKDIR /home/openfoam
RUN git clone https://github.com/csml-rpi/Foam-Agent.git

# Switch to bash to source OpenFOAM environment
SHELL ["/bin/bash", "-c"]

# Create the Foam-Agent environment
WORKDIR $FoamAgent_PATH
RUN conda config --add channels conda-forge && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create --file environment.yml --yes

# Initialize conda for bash (automatic setup)
RUN conda init bash && \
    echo "conda activate FoamAgent" >> ~/.bashrc

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

# Copy Conda runtime and Foam-Agent from builder
COPY --from=builder $CONDA_DIR $CONDA_DIR
COPY --from=builder $FoamAgent_PATH $FoamAgent_PATH
COPY --from=builder /root/.bashrc /root/.bashrc

# Ensure conda is properly initialized in .bashrc for interactive shells
RUN echo '' >> /root/.bashrc && \
    echo '# Auto-activate FoamAgent environment' >> /root/.bashrc && \
    echo 'if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then' >> /root/.bashrc && \
    echo '    source "$CONDA_DIR/etc/profile.d/conda.sh"' >> /root/.bashrc && \
    echo '    conda activate FoamAgent 2>/dev/null || true' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc && \
    echo '' >> /root/.bashrc && \
    echo '# Auto-source OpenFOAM' >> /root/.bashrc && \
    echo 'if [ -f /opt/openfoam10/etc/bashrc ]; then' >> /root/.bashrc && \
    echo '    source /opt/openfoam10/etc/bashrc' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc && \
    echo '' >> /root/.bashrc && \
    echo '# Auto-change to Foam-Agent directory' >> /root/.bashrc && \
    echo 'cd "$FoamAgent_PATH" 2>/dev/null || true' >> /root/.bashrc

# Create a startup script that automatically sets up everything
RUN cat > /usr/local/bin/foamagent-entrypoint.sh << 'EOFSCRIPT'
#!/bin/bash
set -e

# Source OpenFOAM environment in a controlled way: allow non-zero RC, then validate
set +e
source /opt/openfoam10/etc/bashrc
openfoam_rc=$?
set -e

# Strict validation: must have WM_PROJECT_DIR and blockMesh in PATH
if [ -z "$WM_PROJECT_DIR" ] || ! command -v blockMesh >/dev/null 2>&1; then
    echo "ERROR: OpenFOAM environment failed to load (rc=$openfoam_rc)." >&2
    echo "Diag: WM_PROJECT_DIR='${WM_PROJECT_DIR:-unset}', blockMesh=$(command -v blockMesh || echo 'NOT-IN-PATH')" >&2
    exit 1
fi

# Initialize conda
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Activate FoamAgent environment
conda activate FoamAgent

# Change to Foam-Agent directory
cd "$FoamAgent_PATH"

# Display welcome message
echo "=========================================="
echo "Foam-Agent Docker Container Ready!"
echo "=========================================="
echo "OpenFOAM: $WM_PROJECT_DIR"
echo "Conda Env: FoamAgent (activated)"
echo "Working Dir: $FoamAgent_PATH"
echo ""
echo "To update to latest Foam-Agent:"
echo "  cd $FoamAgent_PATH && git pull"
echo ""
echo "To run Foam-Agent:"
echo "  python foambench_main.py --openfoam_path \$WM_PROJECT_DIR --output ./output --prompt_path ./user_requirement.txt"
echo ""
echo "Note: Make sure OPENAI_API_KEY is set before running!"
echo "=========================================="

# Execute the command passed to the container
if [ "$1" = "/bin/bash" ] || [ "$1" = "bash" ] || [ -z "$1" ]; then
    exec /bin/bash -i
else
    exec "$@"
fi
EOFSCRIPT
RUN chmod +x /usr/local/bin/foamagent-entrypoint.sh

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

# Use the custom entrypoint script
ENTRYPOINT ["/usr/local/bin/foamagent-entrypoint.sh"]
CMD ["/bin/bash"]
