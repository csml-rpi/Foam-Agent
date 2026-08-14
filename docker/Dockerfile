# --- Stage 1: Builder ---
FROM openfoam/openfoam10-paraview510:latest AS builder
USER root

# Install base utilities (including ca-certificates for SSL)
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        wget \
        ca-certificates \
        libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# uv manages both the Python toolchain and the virtual environment, so this image needs
# neither a system Python nor conda.
ENV UV_INSTALL_DIR=/usr/local/bin
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && uv --version

# Copy Foam-Agent from local directory (build context)
ENV FoamAgent_PATH="/home/openfoam/Foam-Agent"
WORKDIR /home/openfoam
# Copy the entire project directory (excluding .dockerignore patterns)
# Note: Build context must be the project root directory
COPY . Foam-Agent

# Switch to bash to source OpenFOAM environment
SHELL ["/bin/bash", "-c"]

# Build the virtual environment at a fixed path so the runtime stage can copy it whole.
WORKDIR $FoamAgent_PATH
ENV UV_PROJECT_ENVIRONMENT=/opt/foamagent-venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev \
        --extra rag-local --extra direct-api --extra viz --extra web --extra hpc \
    && uv cache clean

# --- Stage 2: Runtime ---
FROM openfoam/openfoam10-paraview510:latest
USER root

# Install runtime dependencies

RUN apt-get update \
    && apt-get install -y \
        libboost-all-dev \
        ca-certificates \
        git \
        git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Environment variables in a FoamAgent container
ENV FoamAgent_PATH="/home/openfoam/Foam-Agent"
ENV VIRTUAL_ENV=/opt/foamagent-venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the Python runtime, the virtual environment, and Foam-Agent from the builder
COPY --from=builder --chown=root:root /opt/uv-python /opt/uv-python
COPY --from=builder --chown=root:root /opt/foamagent-venv /opt/foamagent-venv
COPY --from=builder --chown=root:root $FoamAgent_PATH $FoamAgent_PATH

RUN echo '' >> /root/.bashrc && \
    echo '# Foam-Agent virtual environment' >> /root/.bashrc && \
    echo 'export PATH="/opt/foamagent-venv/bin:$PATH"' >> /root/.bashrc && \
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
