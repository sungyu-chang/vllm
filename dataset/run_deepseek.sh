#!/usr/bin/env bash
# run_deepseek.sh
# Run the DeepSeek-MoE inference job, optionally after another process finishes.
#
# Usage:
#   # Wait for a specific PID to exit first (e.g. the Qwen run):
#   bash run_deepseek.sh --wait-pid <PID>
#
#   # Sleep for N seconds before starting (default 18000 = 5 hours):
#   bash run_deepseek.sh --sleep 18000
#
#   # Run immediately:
#   bash run_deepseek.sh
#
# It is recommended to run this inside a screen/tmux session so it survives
# terminal disconnects:
#   screen -S deepseek bash run_deepseek.sh --wait-pid <PID>

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration – edit these as needed
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/usr/local/bin/python3
MODEL="deepseek-ai/DeepSeek-MoE-16B-Chat"
DATASET="${SCRIPT_DIR}/combined_dataset.jsonl"
OUTPUT="${SCRIPT_DIR}/expert_log_deepseek.jsonl"
TENSOR_PARALLEL_SIZE=1
MAX_INPUT_TOKENS=4096
MAX_OUTPUT_TOKENS=128
LOG_FILE="${SCRIPT_DIR}/deepseek_run.log"

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
WAIT_PID=""
SLEEP_SECS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait-pid)
            WAIT_PID="$2"; shift 2 ;;
        --sleep)
            SLEEP_SECS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Optionally wait for another process to finish
# --------------------------------------------------------------------------
if [[ -n "$WAIT_PID" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for PID $WAIT_PID to finish..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID $WAIT_PID has exited."
elif [[ "$SLEEP_SECS" -gt 0 ]]; then
    HOURS=$(( SLEEP_SECS / 3600 ))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sleeping for ${HOURS}h (${SLEEP_SECS}s) before starting..."
    sleep "$SLEEP_SECS"
fi

# --------------------------------------------------------------------------
# Run inference
# --------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting DeepSeek-MoE inference." | tee -a "$LOG_FILE"
echo "  model:   $MODEL"  | tee -a "$LOG_FILE"
echo "  dataset: $DATASET" | tee -a "$LOG_FILE"
echo "  output:  $OUTPUT"  | tee -a "$LOG_FILE"

"$PYTHON" "${SCRIPT_DIR}/02_run_inference.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_input_tokens "$MAX_INPUT_TOKENS" \
    --max_output_tokens "$MAX_OUTPUT_TOKENS" \
    2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DeepSeek-MoE inference finished." | tee -a "$LOG_FILE"
