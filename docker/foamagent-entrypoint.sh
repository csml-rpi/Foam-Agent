#!/usr/bin/env bash
set -e

image_target="${FOAMAGENT_IMAGE_OPENFOAM_TARGET:-foundation-v10}"
if [[ -n "${FOAMAGENT_OPENFOAM_TARGET:-}" && "$FOAMAGENT_OPENFOAM_TARGET" != "$image_target" ]]; then
    echo "ERROR: image target is $image_target; run the image matching FOAMAGENT_OPENFOAM_TARGET." >&2
    exit 64
fi
case "$image_target" in
    foundation-v10) foam_bashrc=/opt/openfoam10/etc/bashrc; expected_version=10 ;;
    esi-v2006) foam_bashrc=/opt/OpenFOAM/OpenFOAM-v2006/etc/bashrc; expected_version=2006 ;;
    *) echo "ERROR: unsupported image target: $image_target" >&2; exit 64 ;;
esac
if [[ ! -f "$foam_bashrc" ]]; then
    echo "ERROR: missing $image_target runtime: $foam_bashrc" >&2
    exit 64
fi

# OpenFOAM setup interprets positional arguments and probes optional programs.
command_args=("$@")
set --
set +e +u
source "$foam_bashrc"
set -e
set -- "${command_args[@]}"
if [[ -z "${WM_PROJECT_DIR:-}" || "${WM_PROJECT_VERSION#v}" != "$expected_version" ]] || ! command -v blockMesh >/dev/null; then
    echo "ERROR: failed to load the $image_target runtime from $foam_bashrc." >&2
    exit 64
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate FoamAgent
cd "$FoamAgent_PATH"

# Only use bundled weights for the matching model; other models use their own cache.
if [[ "${FOAMAGENT_EMBEDDING_PROVIDER:-huggingface}" == huggingface &&
      "${FOAMAGENT_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}" == "${FOAMAGENT_BUNDLED_EMBEDDING_MODEL:-}" &&
      -z "${FOAMAGENT_EMBEDDING_MODEL_PATH:-}" ]]; then
    export FOAMAGENT_EMBEDDING_MODEL_PATH="$FOAMAGENT_BUNDLED_EMBEDDING_MODEL_PATH"
fi

if [[ -n "${FOAMAGENT_VERSION:-}" || "${FOAMAGENT_SKIP_UPDATE:-1}" == 0 ]]; then
    echo "[Foam-Agent] Runtime code updates are disabled. Pull/build the desired image version instead." >&2
fi
# Keep stdout available to stdio MCP clients; do not display credentials.
echo "[Foam-Agent] $image_target; bundled code ${FOAMAGENT_IMAGE_REVISION:-local}; no startup downloads or updates." >&2
if [[ $# -eq 0 ]] || { [[ $# -eq 1 ]] && [[ "$1" == bash || "$1" == /bin/bash ]]; }; then
    exec /bin/bash -i
fi
exec "$@"
