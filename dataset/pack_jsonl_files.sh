#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="${SCRIPT_DIR}/jsonl_files_${TIMESTAMP}.tar.gz"

usage() {
    cat <<'EOF'
Usage:
  bash pack_jsonl_files.sh [output.tar.gz] [jsonl_file ...]

Examples:
  bash pack_jsonl_files.sh
  bash pack_jsonl_files.sh /tmp/all_jsonl.tar.gz
  bash pack_jsonl_files.sh /tmp/logs.tar.gz expert_log_mixtral.jsonl expert_log_qwen3.jsonl

Behavior:
  - With no file arguments, archives all top-level *.jsonl files in this directory.
  - With file arguments, archives only the specified files.
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

declare -a files=()
if [[ $# -gt 0 ]]; then
    for input_path in "$@"; do
        if [[ "$input_path" != /* ]]; then
            input_path="${SCRIPT_DIR}/${input_path}"
        fi
        if [[ ! -f "$input_path" ]]; then
            echo "error: file not found: $input_path" >&2
            exit 1
        fi
        if [[ "$input_path" != *.jsonl ]]; then
            echo "error: not a .jsonl file: $input_path" >&2
            exit 1
        fi
        files+=("$(basename "$input_path")")
    done
else
    shopt -s nullglob
    for file_path in "${SCRIPT_DIR}"/*.jsonl; do
        files+=("$(basename "$file_path")")
    done
    shopt -u nullglob
fi

if [[ ${#files[@]} -eq 0 ]]; then
    echo "error: no JSONL files found to archive" >&2
    exit 1
fi

mkdir -p "$(dirname "$output_path")"

echo "Packing ${#files[@]} JSONL file(s) into $output_path"
tar -czf "$output_path" -C "$SCRIPT_DIR" "${files[@]}"
echo "Created: $output_path"
du -sh "$output_path"
