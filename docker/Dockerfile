# --- Stage 1: Builder ---
FROM openfoam/openfoam10-paraview510:latest AS builder
USER root

# Install base utilities (including ca-certificates for SSL)
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        wget \
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

# Copy Foam-Agent from local directory (build context)
ENV FoamAgent_PATH="/home/openfoam/Foam-Agent"
WORKDIR /home/openfoam
# Copy the entire project directory (excluding .dockerignore patterns)
# Note: Build context must be the project root directory
COPY . Foam-Agent

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
        libboost-all-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Environment variables in a FoamAgent container
ENV CONDA_DIR="/opt/conda"
ENV PATH=$CONDA_DIR/bin:$PATH
ENV FoamAgent_PATH="/home/openfoam/Foam-Agent"

# Copy Conda runtime and Foam-Agent from builder
COPY --from=builder --chown=root:root $CONDA_DIR $CONDA_DIR
COPY --from=builder --chown=root:root $FoamAgent_PATH $FoamAgent_PATH
COPY --from=builder --chown=root:root /root/.bashrc /root/.bashrc

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

# Copy entrypoint script from docker directory
# Note: Build context should be the project root directory
# Build command: docker build -f docker/Dockerfile .
COPY docker/foamagent-entrypoint.sh /usr/local/bin/foamagent-entrypoint.sh
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