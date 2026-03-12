#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="${SCRIPT_DIR}/figures_dirs_${TIMESTAMP}.tar.gz"

usage() {
    cat <<'EOF'
Usage:
  bash pack_figures_dirs.sh [output.tar.gz] [figures_dir ...]

Examples:
  bash pack_figures_dirs.sh
  bash pack_figures_dirs.sh /tmp/figures.tar.gz
  bash pack_figures_dirs.sh /tmp/mixtral_figures.tar.gz figures_mixtral

Behavior:
  - With no directory arguments, archives all top-level figures_* directories.
  - With directory arguments, archives only the specified directories.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

output_path="$DEFAULT_OUTPUT"
if [[ $# -gt 0 && "$1" == *.tar.gz ]]; then
    output_path="$1"
    shift
fi

declare -a dirs=()
if [[ $# -gt 0 ]]; then
    for input_path in "$@"; do
        if [[ "$input_path" != /* ]]; then
            input_path="${SCRIPT_DIR}/${input_path}"
        fi
        if [[ ! -d "$input_path" ]]; then
            echo "error: directory not found: $input_path" >&2
            exit 1
        fi
        dirs+=("$(basename "$input_path")")
    done
else
    shopt -s nullglob
    for dir_path in "${SCRIPT_DIR}"/figures_*; do
        [[ -d "$dir_path" ]] || continue
        dirs+=("$(basename "$dir_path")")
    done
    shopt -u nullglob
fi

if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "error: no figures_* directories found to archive" >&2
    exit 1
fi

mkdir -p "$(dirname "$output_path")"

echo "Packing ${#dirs[@]} figure director$( [[ ${#dirs[@]} -eq 1 ]] && echo y || echo ies ) into $output_path"
tar -czf "$output_path" -C "$SCRIPT_DIR" "${dirs[@]}"
echo "Created: $output_path"
du -sh "$output_path"
