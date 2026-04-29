#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-bench-model}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8100}"
TP_SIZES="${TP_SIZES:-2 4 8}"
DP_SIZES="${DP_SIZES:-1 2 3 4 5 6 7 8}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
ALL2ALL_BACKEND="${ALL2ALL_BACKEND:-allgather_reducescatter}"
RESULT_ROOT="${RESULT_ROOT:-benchmarks/dp_ep_vs_tp/results/$(date +%Y%m%d_%H%M%S)}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found. Activate/install the vLLM environment first." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl command not found. It is required for server health checks." >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT/server_logs" "$RESULT_ROOT/bench_logs" "$RESULT_ROOT/json"

server_pid=""

cleanup_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}

trap cleanup_server EXIT

gpu_list() {
  local count="$1"
  local ids=()
  local i
  for ((i = 0; i < count; i++)); do
    ids+=("$i")
  done
  local IFS=,
  echo "${ids[*]}"
}

wait_for_server() {
  local port="$1"
  local log_file="$2"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))

  while ((SECONDS < deadline)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before becoming ready. Log: $log_file" >&2
      return 1
    fi

    if curl -fsS "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done

  echo "Timed out waiting for server on ${HOST}:${port}. Log: $log_file" >&2
  return 1
}

run_case() {
  local case_name="$1"
  local gpu_count="$2"
  local port="$3"
  shift 3

  local server_log="$RESULT_ROOT/server_logs/${case_name}.log"
  local bench_log="$RESULT_ROOT/bench_logs/${case_name}.log"
  local result_json="$RESULT_ROOT/json/${case_name}.json"
  local cuda_devices
  cuda_devices="$(gpu_list "$gpu_count")"

  echo "=== ${case_name} on GPUs ${cuda_devices} port ${port} ==="

  cleanup_server

  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$cuda_devices" vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$port" \
    --max-model-len "$MAX_MODEL_LEN" \
    "$@" \
    $SERVER_EXTRA_ARGS \
    >"$server_log" 2>&1 &
  server_pid="$!"

  wait_for_server "$port" "$server_log"

  local bench_cmd=(
    vllm bench serve
    --backend openai
    --model "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$port"
    --dataset-name random
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$REQUEST_RATE"
    --save-result
    --result-dir "$RESULT_ROOT/json"
    --result-filename "${case_name}.json"
    --metadata
    "case=${case_name}"
    "model=${MODEL}"
    "gpu_count=${gpu_count}"
    "input_len=${INPUT_LEN}"
    "output_len=${OUTPUT_LEN}"
  )

  if [[ -n "$MAX_CONCURRENCY" ]]; then
    bench_cmd+=(--max-concurrency "$MAX_CONCURRENCY")
  fi

  # shellcheck disable=SC2206
  local bench_extra=( $BENCH_EXTRA_ARGS )
  bench_cmd+=("${bench_extra[@]}")

  "${bench_cmd[@]}" >"$bench_log" 2>&1

  cleanup_server

  if [[ ! -s "$result_json" ]]; then
    echo "Missing benchmark result JSON: $result_json" >&2
    return 1
  fi
}

case_index=0

for tp in $TP_SIZES; do
  port=$((BASE_PORT + case_index))
  run_case "tp${tp}" "$tp" "$port" \
    --tensor-parallel-size "$tp"
  case_index=$((case_index + 1))
done

for dp in $DP_SIZES; do
  port=$((BASE_PORT + case_index))
  run_case "dp${dp}_ep" "$dp" "$port" \
    --data-parallel-size "$dp" \
    --data-parallel-size-local "$dp" \
    --enable-expert-parallel \
    --all2all-backend "$ALL2ALL_BACKEND"
  case_index=$((case_index + 1))
done

if [[ -x "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" benchmarks/dp_ep_vs_tp/summarize_results.py "$RESULT_ROOT/json" \
    --output "$RESULT_ROOT/summary.csv"
  echo "Summary: $RESULT_ROOT/summary.csv"
else
  echo "Skipping summary: $PYTHON_BIN is not executable." >&2
fi

echo "Results: $RESULT_ROOT"
