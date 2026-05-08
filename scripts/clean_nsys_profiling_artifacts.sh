#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-results/dp_ep_vs_tp/nsys_profiling}"

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "Directory not found: $ROOT_DIR" >&2
    exit 1
fi

sqlite_count=$(find "$ROOT_DIR" -type f -name '*.sqlite' | wc -l)
stats_count=$(find "$ROOT_DIR" -path '*/nsys_stats/*' -type f | wc -l)

find "$ROOT_DIR" -type f -name '*.sqlite' -delete
find "$ROOT_DIR" -path '*/nsys_stats/*' -type f -delete

echo "Deleted $sqlite_count .sqlite files from $ROOT_DIR"
echo "Deleted $stats_count files under nsys_stats directories in $ROOT_DIR"