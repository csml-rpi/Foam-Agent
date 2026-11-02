#!/usr/bin/env bash
# Shared build-time Python dependencies; never run at container startup.
set -euo pipefail
installer=Miniconda3-py312_25.1.1-2-Linux-x86_64.sh
installer_sha256=4766d85b5f7d235ce250e998ebb5a8a8210cbd4f2b0fea4d2177b3ed9ea87884
wget -q "https://repo.anaconda.com/miniconda/$installer" -O /tmp/miniconda.sh
echo "$installer_sha256  /tmp/miniconda.sh" | sha256sum --check -
bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
rm /tmp/miniconda.sh
"$CONDA_DIR/bin/conda" env create -f environment.yml --yes
python_bin="$CONDA_DIR/envs/FoamAgent/bin/python"
"$python_bin" -m pip install --no-cache-dir 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
"$python_bin" - <<'PY'
import tomllib
from pathlib import Path
project = tomllib.loads(Path('pyproject.toml').read_text())['project']
deps = project['dependencies'] + project['optional-dependencies']['mesh']
Path('/tmp/foamagent-requirements.txt').write_text('\n'.join(deps) + '\n')
PY
"$python_bin" -m pip install --no-cache-dir -c constraints.txt -r /tmp/foamagent-requirements.txt
"$python_bin" -m pip check
"$CONDA_DIR/bin/conda" clean --all --yes
