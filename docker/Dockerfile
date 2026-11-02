# Build from the repository root with hydrated target corpus files.
FROM openfoam/openfoam10-paraview510:latest AS builder
USER root
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates python3 && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR=/opt/conda
ENV FoamAgent_PATH=/home/openfoam/Foam-Agent
WORKDIR /home/openfoam/Foam-Agent
COPY scripts/verify_target_corpus.py /tmp/verify_target_corpus.py
COPY database/foundation-v10/ database/foundation-v10/
RUN python3 /tmp/verify_target_corpus.py --target foundation-v10 --database-path /home/openfoam/Foam-Agent/database/foundation-v10
COPY environment.yml pyproject.toml constraints.txt README.md ./
COPY docker/install-python.sh /tmp/install-python.sh
RUN bash /tmp/install-python.sh
COPY src/ src/
COPY scripts/ scripts/
COPY examples/ examples/
COPY foambench_main.py user_requirement.txt LICENSE ./
RUN /opt/conda/envs/FoamAgent/bin/python -m pip install --no-cache-dir --no-deps -e .

# Pin model assets independently of the location used to load them.
ARG EMBEDDING_REVISION=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
RUN /opt/conda/envs/FoamAgent/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', revision='$EMBEDDING_REVISION', local_dir='/opt/foamagent-model', allow_patterns=['*.json', '*.safetensors', '*.txt', 'README.md'])" && \
    rm -rf /opt/foamagent-model/.cache

FROM openfoam/openfoam10-paraview510:latest
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates xvfb libgl1 libglu1-mesa libgomp1 && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR=/opt/conda \
    FoamAgent_PATH=/home/openfoam/Foam-Agent \
    PATH=/opt/conda/envs/FoamAgent/bin:/opt/conda/bin:$PATH \
    FOAMAGENT_IMAGE_OPENFOAM_TARGET=foundation-v10 \
    FOAMAGENT_BUNDLED_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
    FOAMAGENT_BUNDLED_EMBEDDING_MODEL_PATH=/opt/foamagent-model
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/foamagent-model /opt/foamagent-model
COPY --from=builder /home/openfoam/Foam-Agent /home/openfoam/Foam-Agent
COPY docker/foamagent-entrypoint.sh /usr/local/bin/foamagent-entrypoint.sh
ARG VCS_REF=local
ARG VERSION=local
ENV FOAMAGENT_IMAGE_REVISION=$VCS_REF
LABEL org.opencontainers.image.source="https://github.com/csml-rpi/Foam-Agent" \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION
RUN chmod +x /usr/local/bin/foamagent-entrypoint.sh && \
    printf '\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate FoamAgent\nsource /opt/openfoam10/etc/bashrc\ncd /home/openfoam/Foam-Agent\n' >> /root/.bashrc
WORKDIR /home/openfoam/Foam-Agent
ENTRYPOINT ["/usr/local/bin/foamagent-entrypoint.sh"]
CMD ["/bin/bash"]
