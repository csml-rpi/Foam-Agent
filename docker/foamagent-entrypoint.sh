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
echo "  python foambench_main.py --output ./output --prompt_path ./user_requirement.txt"
echo ""
if [ -n "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY: ${OPENAI_API_KEY:0:20}... (set)"
else
    echo "OPENAI_API_KEY: (not set)"
fi
echo ""
echo "Note: Make sure OPENAI_API_KEY is set before running!"
echo "      To change the model provider, edit src/config.py"
echo "=========================================="

# Execute the command passed to the container
if [ "$1" = "/bin/bash" ] || [ "$1" = "bash" ] || [ -z "$1" ]; then
    exec /bin/bash -i
else
    exec "$@"
fi

