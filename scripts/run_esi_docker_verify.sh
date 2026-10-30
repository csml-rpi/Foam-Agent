#!/usr/bin/env bash
# Run extract -> translate -> blockMesh -> solver inside OpenCFD Docker image.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${FOAM_ESI_DOCKER_IMAGE:-opencfd/openfoam-default}"
CASE_DIR="${1:-/tmp/foamagent_cavity_esi}"
PRESET="${FOAM_VERIFY_PRESET:-cavity}"
TIMEOUT="${FOAM_VERIFY_TIMEOUT:-600}"

echo "[1/4] Extract + translate (${PRESET}) -> ${CASE_DIR}"
python3 "${ROOT}/scripts/verify_esi_translation.py" \
  --preset "${PRESET}" \
  -o "${CASE_DIR}" \
  --overwrite

echo "[2/4] Docker image: ${IMAGE}"
echo "[3/4] blockMesh + solver in container"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "${CASE_DIR}:${CASE_DIR}" \
  -v "${ROOT}/scripts/mock_sbatch.sh:/usr/local/bin/mock_sbatch.sh" \
  "${IMAGE}" \
  bash -lc "
    set -e
    cd '${CASE_DIR}'
    source /openfoam/bashrc 2>/dev/null || source /opt/openfoam*/etc/bashrc 2>/dev/null || true
    APP=\$(grep -E '^[[:space:]]*application[[:space:]]+' system/controlDict | awk '{print \$2}' | tr -d ';')
    echo \"Solver: \$APP\"
    blockMesh
    timeout ${TIMEOUT} \"\$APP\"
    echo '---'
    tail -5 \"log.\$APP\" 2>/dev/null || true
    grep -E 'End|FOAM exiting' \"log.\$APP\" 2>/dev/null || echo '(check log.\$APP for convergence)'
"

echo "[✓] Docker verification finished: ${CASE_DIR}"
