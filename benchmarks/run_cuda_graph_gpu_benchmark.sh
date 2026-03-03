#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Orchestration script for the CUDA Graph GPU Benchmark.
#
# Runs the benchmark across all relevant configurations (eager vs cuda_graph,
# prefill vs decode) and optionally captures nsys profiles.
#
# Usage:
#   # Basic run (no nsys profiling)
#   bash run_cuda_graph_gpu_benchmark.sh --model meta-llama/Llama-2-7b-hf
#
#   # With nsys profiling
#   bash run_cuda_graph_gpu_benchmark.sh --model meta-llama/Llama-2-7b-hf --nsys
#
#   # Custom configuration
#   bash run_cuda_graph_gpu_benchmark.sh \
#       --model meta-llama/Llama-2-7b-hf \
#       --batch-sizes "1 8" \
#       --input-lens "128 512 1024" \
#       --output-len 128 \
#       --num-iters 10 \
#       --dtype float16 \
#       --nsys

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL=""
DTYPE="float16"
BATCH_SIZES="1"
INPUT_LENS="128 512 1024"
OUTPUT_LEN=128
NUM_ITERS=10
WARMUP_ITERS=3
NSYS=false
NSYS_ITERS=3          # fewer iters when profiling to keep trace size manageable
NSYS_WARMUP=2
OUTPUT_DIR="cuda_graph_bench_results"
EXTRA_ARGS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2";        shift 2 ;;
        --dtype)        DTYPE="$2";        shift 2 ;;
        --batch-sizes)  BATCH_SIZES="$2";  shift 2 ;;
        --input-lens)   INPUT_LENS="$2";   shift 2 ;;
        --output-len)   OUTPUT_LEN="$2";   shift 2 ;;
        --num-iters)    NUM_ITERS="$2";    shift 2 ;;
        --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
        --nsys)         NSYS=true;         shift ;;
        --nsys-iters)   NSYS_ITERS="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --extra-args)   EXTRA_ARGS="$2";   shift 2 ;;
        -h|--help)
            echo "Usage: $0 --model MODEL [options]"
            echo ""
            echo "Options:"
            echo "  --model MODEL         HuggingFace model name (required)"
            echo "  --dtype DTYPE         float16 | bfloat16 | auto  (default: float16)"
            echo "  --batch-sizes SIZES   Space-separated batch sizes (default: '1')"
            echo "  --input-lens LENS     Space-separated input lengths (default: '128 512 1024')"
            echo "  --output-len LEN      Output token count for decode (default: 128)"
            echo "  --num-iters N         Benchmark iterations (default: 10)"
            echo "  --warmup-iters N      Warmup iterations (default: 3)"
            echo "  --nsys                Enable nsys profiling"
            echo "  --nsys-iters N        Iterations when profiling (default: 3)"
            echo "  --output-dir DIR      Directory for results (default: cuda_graph_bench_results)"
            echo "  --extra-args ARGS     Extra arguments passed to the benchmark script"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/cuda_graph_gpu_benchmark.py"

# Detect GPU name for organizing results
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0).replace(' ', '_'))" 2>/dev/null || echo "unknown_gpu")

RESULT_DIR="${OUTPUT_DIR}/${GPU_NAME}"
mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo "CUDA Graph GPU Benchmark - Experiment Runner"
echo "============================================================"
echo "GPU             : ${GPU_NAME}"
echo "Model           : ${MODEL}"
echo "Dtype           : ${DTYPE}"
echo "Batch sizes     : ${BATCH_SIZES}"
echo "Input lengths   : ${INPUT_LENS}"
echo "Output length   : ${OUTPUT_LEN}"
echo "Iterations      : ${NUM_ITERS} (warmup: ${WARMUP_ITERS})"
echo "nsys profiling  : ${NSYS}"
echo "Output directory: ${RESULT_DIR}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: run a single benchmark configuration
# ---------------------------------------------------------------------------
run_bench() {
    local mode="$1"      # "eager" or "cudagraph"
    local phase="$2"     # "prefill" or "decode" or "all"
    local batch="$3"
    local input_len="$4"
    local output_len="$5"
    local iters="$6"
    local warmup="$7"

    local eager_flag=""
    if [[ "$mode" == "eager" ]]; then
        eager_flag="--enforce-eager"
    fi

    local tag="${mode}_${phase}_bs${batch}_in${input_len}_out${output_len}"
    local json_path="${RESULT_DIR}/${tag}.json"

    echo "------------------------------------------------------------"
    echo "Running: ${tag}"
    echo "------------------------------------------------------------"

    python3 "${BENCH_SCRIPT}" \
        --model "${MODEL}" \
        --dtype "${DTYPE}" \
        --batch-size "${batch}" \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --num-iters "${iters}" \
        --warmup-iters "${warmup}" \
        --phase "${phase}" \
        --output-json "${json_path}" \
        ${eager_flag} \
        ${EXTRA_ARGS}

    echo "  -> saved to ${json_path}"
    echo ""
}

# ---------------------------------------------------------------------------
# Helper: run a single nsys-profiled configuration
# ---------------------------------------------------------------------------
run_nsys() {
    local mode="$1"
    local phase="$2"
    local batch="$3"
    local input_len="$4"
    local output_len="$5"

    local eager_flag=""
    if [[ "$mode" == "eager" ]]; then
        eager_flag="--enforce-eager"
    fi

    local tag="${mode}_${phase}_bs${batch}_in${input_len}_out${output_len}"
    local nsys_output="${RESULT_DIR}/nsys_${tag}"
    local json_path="${RESULT_DIR}/nsys_${tag}.json"

    echo "------------------------------------------------------------"
    echo "nsys profiling: ${tag}"
    echo "------------------------------------------------------------"

    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace=graph \
        --output="${nsys_output}" \
        --force-overwrite=true \
        --stats=true \
        python3 "${BENCH_SCRIPT}" \
            --model "${MODEL}" \
            --dtype "${DTYPE}" \
            --batch-size "${batch}" \
            --input-len "${input_len}" \
            --output-len "${output_len}" \
            --num-iters "${NSYS_ITERS}" \
            --warmup-iters "${NSYS_WARMUP}" \
            --phase "${phase}" \
            --output-json "${json_path}" \
            ${eager_flag} \
            ${EXTRA_ARGS}

    echo "  -> nsys report: ${nsys_output}.nsys-rep"

    # Export kernel and API summaries
    if command -v nsys &>/dev/null; then
        echo "  Extracting kernel summary ..."
        nsys stats --report cuda_gpu_kern_sum \
            "${nsys_output}.nsys-rep" \
            > "${RESULT_DIR}/nsys_${tag}_kernel_summary.txt" 2>&1 || true

        echo "  Extracting CUDA API summary ..."
        nsys stats --report cuda_api_sum \
            "${nsys_output}.nsys-rep" \
            > "${RESULT_DIR}/nsys_${tag}_api_summary.txt" 2>&1 || true

        echo "  Extracting NVTX summary ..."
        nsys stats --report nvtx_sum \
            "${nsys_output}.nsys-rep" \
            > "${RESULT_DIR}/nsys_${tag}_nvtx_summary.txt" 2>&1 || true

        # Export to SQLite for custom analysis
        echo "  Exporting to SQLite ..."
        nsys export --type sqlite \
            "${nsys_output}.nsys-rep" \
            --output "${nsys_output}.sqlite" 2>/dev/null || true
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Run all configurations
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "Phase 1: Latency benchmarks (no nsys)"
echo "============================================================"
echo ""

for batch in ${BATCH_SIZES}; do
    for input_len in ${INPUT_LENS}; do
        # Prefill benchmark: both modes
        run_bench "cudagraph" "prefill" "${batch}" "${input_len}" "${OUTPUT_LEN}" "${NUM_ITERS}" "${WARMUP_ITERS}"
        run_bench "eager"     "prefill" "${batch}" "${input_len}" "${OUTPUT_LEN}" "${NUM_ITERS}" "${WARMUP_ITERS}"
    done

    # Decode benchmark: use first input_len only (decode is less sensitive to input_len)
    FIRST_INPUT_LEN=$(echo "${INPUT_LENS}" | awk '{print $1}')
    run_bench "cudagraph" "decode" "${batch}" "${FIRST_INPUT_LEN}" "${OUTPUT_LEN}" "${NUM_ITERS}" "${WARMUP_ITERS}"
    run_bench "eager"     "decode" "${batch}" "${FIRST_INPUT_LEN}" "${OUTPUT_LEN}" "${NUM_ITERS}" "${WARMUP_ITERS}"
done

# ---------------------------------------------------------------------------
# nsys profiling (optional, uses fewer iterations)
# ---------------------------------------------------------------------------

if [[ "${NSYS}" == "true" ]]; then
    echo ""
    echo "============================================================"
    echo "Phase 2: nsys profiling"
    echo "============================================================"
    echo ""

    FIRST_BATCH=$(echo "${BATCH_SIZES}" | awk '{print $1}')
    FIRST_INPUT_LEN=$(echo "${INPUT_LENS}" | awk '{print $1}')

    # Profile representative configurations
    run_nsys "cudagraph" "prefill" "${FIRST_BATCH}" "${FIRST_INPUT_LEN}" "${OUTPUT_LEN}"
    run_nsys "eager"     "prefill" "${FIRST_BATCH}" "${FIRST_INPUT_LEN}" "${OUTPUT_LEN}"
    run_nsys "cudagraph" "decode"  "${FIRST_BATCH}" "${FIRST_INPUT_LEN}" "${OUTPUT_LEN}"
    run_nsys "eager"     "decode"  "${FIRST_BATCH}" "${FIRST_INPUT_LEN}" "${OUTPUT_LEN}"
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "All experiments complete!"
echo "============================================================"
echo "Results directory: ${RESULT_DIR}"
echo ""
echo "To analyze results:"
echo "  python3 ${SCRIPT_DIR}/analyze_cuda_graph_gpu_results.py --results-dir ${OUTPUT_DIR}"
echo ""
echo "To view nsys traces (if profiled):"
echo "  nsys-ui ${RESULT_DIR}/nsys_*.nsys-rep"
echo ""
echo "To compare kernel summaries:"
echo "  diff ${RESULT_DIR}/nsys_cudagraph_*_kernel_summary.txt \\"
echo "       ${RESULT_DIR}/nsys_eager_*_kernel_summary.txt"
