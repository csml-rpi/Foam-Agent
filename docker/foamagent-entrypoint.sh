#!/bin/bash
set -e

if [ "${FOAMAGENT_OPENFOAM_TARGET:-}" = "esi-v2006" ]; then
    echo "ERROR: this image contains Foundation OpenFOAM v10. Build/run foamagent:esi-v2006 instead." >&2
    exit 64
fi

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
if [ "${WM_PROJECT_VERSION:-}" != "10" ]; then
    echo "ERROR: expected Foundation OpenFOAM v10, found ${WM_PROJECT_VERSION:-unset}." >&2
    exit 64
fi

# Initialize conda
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Activate FoamAgent environment
conda activate FoamAgent

# Change to Foam-Agent directory
cd "$FoamAgent_PATH"

# ---------------------------------------------------------------------------
# Auto-update Foam-Agent code from GitHub
# Skip with: docker run -e FOAMAGENT_SKIP_UPDATE=1 ...
# Pin a version: docker run -e FOAMAGENT_VERSION=v2.0.0 ...
# ---------------------------------------------------------------------------
FOAMAGENT_REPO="https://github.com/csml-rpi/Foam-Agent.git"

if [ "${FOAMAGENT_SKIP_UPDATE:-0}" != "1" ]; then
    echo "[entrypoint] Updating Foam-Agent from GitHub ..."

    # First-time: the image ships code via COPY but without .git.
    # Clone fresh so that future pulls work.
    if [ ! -d "$FoamAgent_PATH/.git" ]; then
        echo "[entrypoint] No .git found — cloning repository ..."
        # Clone into a temp dir, then swap in place (keeps any user-added files)
        tmp_dir=$(mktemp -d)
        if git clone --depth 1 "$FOAMAGENT_REPO" "$tmp_dir"; then
            # Move .git into the existing directory
            mv "$tmp_dir/.git" "$FoamAgent_PATH/.git"
            cd "$FoamAgent_PATH"
            # Reset working tree to match the cloned HEAD
            git checkout -- .
            # Pull LFS objects
            git lfs install --local 2>/dev/null || true
            git lfs pull 2>/dev/null || true
            echo "[entrypoint] Clone complete."
        else
            echo "[entrypoint] WARNING: Clone failed (no network?). Using bundled code." >&2
        fi
        rm -rf "$tmp_dir"
    else
        # .git exists (persistent volume or previous clone) — just pull
        cd "$FoamAgent_PATH"
        if git pull --ff-only 2>/dev/null; then
            git lfs pull 2>/dev/null || true
            echo "[entrypoint] Code updated to latest."
        else
            echo "[entrypoint] WARNING: git pull failed. Using current code." >&2
        fi
    fi

    # Checkout a specific version/tag if requested
    if [ -n "$FOAMAGENT_VERSION" ]; then
        echo "[entrypoint] Checking out version: $FOAMAGENT_VERSION"
        git fetch --tags 2>/dev/null || true
        git checkout "$FOAMAGENT_VERSION" 2>/dev/null || \
            echo "[entrypoint] WARNING: Could not checkout $FOAMAGENT_VERSION" >&2
    fi

    # If environment.yml changed, update conda env (best-effort)
    if [ -d "$FoamAgent_PATH/.git" ]; then
        # Compare bundled environment snapshot with current
        if ! git diff --quiet HEAD@{1} -- environment.yml 2>/dev/null; then
            echo "[entrypoint] environment.yml changed — updating conda env (this may take a while) ..."
            conda env update --file environment.yml --prune 2>&1 | tail -5 || \
                echo "[entrypoint] WARNING: conda env update failed. Some new dependencies may be missing." >&2
        fi
    fi
else
    echo "[entrypoint] FOAMAGENT_SKIP_UPDATE=1 — skipping code update."
fi

cd "$FoamAgent_PATH"

# Display welcome message
echo "=========================================="
echo "Foam-Agent Docker Container Ready!"
echo "=========================================="
echo "OpenFOAM: $WM_PROJECT_DIR"
echo "Conda Env: FoamAgent (activated)"
echo "Working Dir: $FoamAgent_PATH"
if [ -d "$FoamAgent_PATH/.git" ]; then
    echo "Git:        $(git log --oneline -1 2>/dev/null || echo 'unknown')"
fi
echo ""
echo "To run Foam-Agent:"
echo "  python foambench_main.py --output ./output --prompt_path ./user_requirement.txt"
echo ""
echo "Environment variables:"
if [ -n "$OPENAI_API_KEY" ]; then
    echo "  OPENAI_API_KEY:         (set)"
else
    echo "  OPENAI_API_KEY:         (not set)"
fi
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  ANTHROPIC_API_KEY:      (set)"
else
    echo "  ANTHROPIC_API_KEY:      (not set)"
fi
echo "  FOAMAGENT_SKIP_UPDATE:  ${FOAMAGENT_SKIP_UPDATE:-0}"
echo "  FOAMAGENT_VERSION:      ${FOAMAGENT_VERSION:-(latest)}"
echo "=========================================="

# Execute the command passed to the container
if [ "$1" = "/bin/bash" ] || [ "$1" = "bash" ] || [ -z "$1" ]; then
    exec /bin/bash -i
else
    exec "$@"
fi

