#!/usr/bin/env bash
# Run the existing repository example without changing its physical parameters.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_dir="${1:-$repo_dir/output/quickstart}"
exec python "$repo_dir/foambench_main.py" --output "$output_dir" --prompt_path "$repo_dir/user_requirement.txt"
