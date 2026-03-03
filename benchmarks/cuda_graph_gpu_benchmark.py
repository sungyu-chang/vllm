#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUDA Graph GPU Benchmark for vLLM

Measures the latency speedup from CUDA graphs across different GPU
architectures (V100, A100, H100). Separates measurements into prefill
and decode phases with NVTX annotations for nsys profiling.

Hypothesis:
    As GPU hardware gets stronger, individual kernel compute time decreases,
    but kernel launch overhead (CPU-side) stays roughly constant. This makes
    launch overhead a larger *fraction* of total step time on faster GPUs.
    CUDA graphs eliminate per-kernel launch overhead by replaying a captured
    graph in a single launch, so the relative speedup should be larger on
    faster GPUs:  V100 < A100 < H100.

Phase separation strategy:
    - Prefill: Use long input_len with max_tokens=1. The single prefill step
      processes all input tokens and dominates the measured latency.
    - Decode: Use subtraction method. Run two configs with the same input_len
      but different output lengths (1 vs N). The difference isolates the
      per-token decode cost:
          per_token_decode = (T_long - T_short) / (output_len - 1)

NVTX annotations:
    High-level NVTX ranges bracket each phase and iteration so that nsys
    can correlate kernel-level traces to the correct benchmark phase.
    Within each range, nsys automatically captures every CUDA kernel launch,
    cudaGraphLaunch, memcpy, etc.

Usage:
    # Benchmark with CUDA graphs (default)
    python cuda_graph_gpu_benchmark.py --model meta-llama/Llama-2-7b-hf

    # Benchmark without CUDA graphs (eager mode)
    python cuda_graph_gpu_benchmark.py --model meta-llama/Llama-2-7b-hf \
        --enforce-eager

    # Profile with nsys (use fewer iters for manageable trace size)
    nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=graph \
        -o profile_cudagraph --force-overwrite=true \
        python cuda_graph_gpu_benchmark.py \
            --model meta-llama/Llama-2-7b-hf \
            --num-iters 3 --warmup-iters 2

    nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=graph \
        -o profile_eager --force-overwrite=true \
        python cuda_graph_gpu_benchmark.py \
            --model meta-llama/Llama-2-7b-hf \
            --enforce-eager --num-iters 3 --warmup-iters 2

    # Extract kernel statistics from nsys report
    nsys stats --report cuda_gpu_kern_sum profile_cudagraph.nsys-rep
    nsys stats --report cuda_api_sum profile_cudagraph.nsys-rep
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.cuda.nvtx as nvtx

from vllm import LLM, SamplingParams
from vllm.inputs import PromptType


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencySample:
    """A single latency measurement."""
    gpu_time_ms: float
    wall_time_ms: float


@dataclass
class PhaseResult:
    """Aggregated result for one benchmark phase."""
    phase: str  # "prefill" or "decode_per_token"
    samples: list[LatencySample] = field(default_factory=list)

    @property
    def gpu_times(self) -> list[float]:
        return [s.gpu_time_ms for s in self.samples]

    @property
    def wall_times(self) -> list[float]:
        return [s.wall_time_ms for s in self.samples]

    def summary(self) -> dict:
        gpu = np.array(self.gpu_times)
        wall = np.array(self.wall_times)
        return {
            "phase": self.phase,
            "num_samples": len(self.samples),
            "gpu_avg_ms": float(np.mean(gpu)),
            "gpu_median_ms": float(np.median(gpu)),
            "gpu_min_ms": float(np.min(gpu)),
            "gpu_max_ms": float(np.max(gpu)),
            "gpu_std_ms": float(np.std(gpu)),
            "wall_avg_ms": float(np.mean(wall)),
            "wall_median_ms": float(np.median(wall)),
            "samples": [asdict(s) for s in self.samples],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dummy_prompts(batch_size: int, input_len: int) -> list[PromptType]:
    """Create dummy prompts with exact token count using random token IDs."""
    rng = np.random.default_rng(seed=42)
    token_ids = rng.integers(low=10, high=10000, size=(batch_size, input_len))
    return [{"prompt_token_ids": row.tolist()} for row in token_ids]


def timed_generate(
    llm: LLM,
    prompts: list[PromptType],
    sampling_params: SamplingParams,
    nvtx_tag: str,
) -> LatencySample:
    """Run llm.generate() with CUDA event timing and an NVTX range."""
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    nvtx.range_push(nvtx_tag)
    wall_start = time.perf_counter()
    start_event.record()

    llm.generate(prompts, sampling_params, use_tqdm=False)

    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()
    nvtx.range_pop()

    return LatencySample(
        gpu_time_ms=start_event.elapsed_time(end_event),
        wall_time_ms=(wall_end - wall_start) * 1000.0,
    )


# ---------------------------------------------------------------------------
# Prefill benchmark
# ---------------------------------------------------------------------------

def benchmark_prefill(
    llm: LLM,
    batch_size: int,
    input_len: int,
    num_iters: int,
    warmup_iters: int,
) -> PhaseResult:
    """
    Benchmark prefill latency.

    Generates max_tokens=1 so the measurement is dominated by the single
    prefill step that processes all *input_len* tokens.
    """
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    prompts = make_dummy_prompts(batch_size, input_len)

    # -- warmup --
    print(f"  Prefill warmup ({warmup_iters} iterations) ...")
    for i in range(warmup_iters):
        timed_generate(llm, prompts, sampling_params,
                       f"prefill_warmup_{i}")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # -- benchmark --
    print(f"  Prefill benchmark ({num_iters} iterations, "
          f"input_len={input_len}, batch={batch_size}) ...")
    result = PhaseResult(phase="prefill")
    for i in range(num_iters):
        sample = timed_generate(
            llm, prompts, sampling_params, f"prefill_iter_{i}")
        result.samples.append(sample)
        print(f"    iter {i:3d}:  GPU {sample.gpu_time_ms:8.2f} ms  "
              f"Wall {sample.wall_time_ms:8.2f} ms")

    return result


# ---------------------------------------------------------------------------
# Decode benchmark (subtraction method)
# ---------------------------------------------------------------------------

def benchmark_decode(
    llm: LLM,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_iters: int,
    warmup_iters: int,
) -> PhaseResult:
    """
    Benchmark per-token decode latency using the subtraction method.

    Two generate() calls per iteration:
      short: input_len tokens in, 1 token out   → T_short
      long:  input_len tokens in, output_len out → T_long

    per_token_decode = (T_long - T_short) / (output_len - 1)

    This cancels prefill cost and most per-call overhead.
    """
    prompts = make_dummy_prompts(batch_size, input_len)
    short_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    long_params = SamplingParams(
        max_tokens=output_len,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )

    # -- warmup (both short and long to capture CUDA graphs for all sizes) --
    print(f"  Decode warmup ({warmup_iters} iterations) ...")
    for i in range(warmup_iters):
        timed_generate(llm, prompts, short_params,
                       f"decode_warmup_short_{i}")
        timed_generate(llm, prompts, long_params,
                       f"decode_warmup_long_{i}")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # -- benchmark --
    decode_steps = output_len - 1
    print(f"  Decode benchmark ({num_iters} iterations, "
          f"output_len={output_len}, batch={batch_size}) ...")
    print(f"  Per-token decode = (T_long - T_short) / {decode_steps}")

    result = PhaseResult(phase="decode_per_token")
    for i in range(num_iters):
        short = timed_generate(
            llm, prompts, short_params, f"decode_short_iter_{i}")
        long = timed_generate(
            llm, prompts, long_params, f"decode_long_iter_{i}")

        per_token_gpu = (long.gpu_time_ms - short.gpu_time_ms) / decode_steps
        per_token_wall = (long.wall_time_ms - short.wall_time_ms) / decode_steps

        result.samples.append(LatencySample(
            gpu_time_ms=per_token_gpu,
            wall_time_ms=per_token_wall,
        ))
        print(f"    iter {i:3d}:  per-token GPU {per_token_gpu:8.3f} ms  "
              f"Wall {per_token_wall:8.3f} ms  "
              f"(short={short.gpu_time_ms:.1f}  long={long.gpu_time_ms:.1f}  "
              f"diff={long.gpu_time_ms - short.gpu_time_ms:.1f} ms "
              f"over {decode_steps} steps)")

    return result


# ---------------------------------------------------------------------------
# Decode benchmark (end-to-end method, simpler alternative)
# ---------------------------------------------------------------------------

def benchmark_decode_e2e(
    llm: LLM,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_iters: int,
    warmup_iters: int,
) -> PhaseResult:
    """
    Benchmark end-to-end decode latency (prefill + all decode steps).

    Unlike the subtraction method, this measures total generate() time
    with output_len tokens. Useful as a cross-check and for nsys traces
    where you want a single contiguous NVTX range per iteration.
    """
    prompts = make_dummy_prompts(batch_size, input_len)
    sampling_params = SamplingParams(
        max_tokens=output_len,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )

    # -- warmup --
    print(f"  Decode E2E warmup ({warmup_iters} iterations) ...")
    for i in range(warmup_iters):
        timed_generate(llm, prompts, sampling_params,
                       f"decode_e2e_warmup_{i}")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # -- benchmark --
    print(f"  Decode E2E benchmark ({num_iters} iterations, "
          f"input_len={input_len}, output_len={output_len}, "
          f"batch={batch_size}) ...")
    result = PhaseResult(phase="decode_e2e")
    for i in range(num_iters):
        sample = timed_generate(
            llm, prompts, sampling_params, f"decode_e2e_iter_{i}")
        result.samples.append(sample)
        per_token = sample.gpu_time_ms / output_len
        print(f"    iter {i:3d}:  total GPU {sample.gpu_time_ms:8.2f} ms  "
              f"({per_token:.3f} ms/token)  "
              f"Wall {sample.wall_time_ms:8.2f} ms")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CUDA Graph GPU Benchmark for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or local path "
             "(e.g. meta-llama/Llama-2-7b-hf)")
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "auto"],
        help="Model dtype. Use float16 for V100 compatibility.")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Tensor parallel size.")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="Fraction of GPU memory to use for KV cache.")
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Override max model context length. "
             "Defaults to input_len + output_len + 256.")

    # Workload
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of requests per batch.")
    parser.add_argument(
        "--input-len", type=int, default=512,
        help="Input prompt length in tokens (controls prefill size).")
    parser.add_argument(
        "--output-len", type=int, default=128,
        help="Number of output tokens (controls decode steps).")

    # Benchmark control
    parser.add_argument(
        "--num-iters", type=int, default=10,
        help="Number of timed iterations per phase.")
    parser.add_argument(
        "--warmup-iters", type=int, default=3,
        help="Number of warmup iterations (ensures CUDA graphs are captured).")
    parser.add_argument(
        "--phase", choices=["prefill", "decode", "decode-e2e", "all"],
        default="all",
        help="Which phase(s) to benchmark. 'decode' uses subtraction method, "
             "'decode-e2e' measures total generate time.")

    # CUDA graph control
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="Disable CUDA graphs; run in pure eager mode.")

    # Output
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to write JSON results.")

    return parser


def main():
    args = build_parser().parse_args()

    # Derive max_model_len if not set
    if args.max_model_len is None:
        args.max_model_len = max(
            args.input_len + args.output_len + 256, 2048)

    gpu_name = torch.cuda.get_device_name(0)
    mode_tag = "eager" if args.enforce_eager else "cuda_graph"

    header = (
        f"{'=' * 70}\n"
        f"CUDA Graph GPU Benchmark\n"
        f"{'=' * 70}\n"
        f"GPU             : {gpu_name}\n"
        f"Model           : {args.model}\n"
        f"Dtype           : {args.dtype}\n"
        f"Mode            : {mode_tag}\n"
        f"Batch size      : {args.batch_size}\n"
        f"Input length    : {args.input_len}\n"
        f"Output length   : {args.output_len}\n"
        f"Iterations      : {args.num_iters} (warmup: {args.warmup_iters})\n"
        f"Phase           : {args.phase}\n"
        f"{'=' * 70}"
    )
    print(header)

    # ---- Initialize engine ----
    print("\nInitializing vLLM engine ...")
    nvtx.range_push(f"engine_init_{mode_tag}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
    )
    nvtx.range_pop()
    print("Engine initialized.\n")

    # ---- Run benchmarks ----
    results: dict = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_iters": args.num_iters,
            "warmup_iters": args.warmup_iters,
            "enforce_eager": args.enforce_eager,
            "mode": mode_tag,
            "gpu_name": gpu_name,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
    }

    run_prefill = args.phase in ("prefill", "all")
    run_decode = args.phase in ("decode", "all")
    run_decode_e2e = args.phase in ("decode-e2e", "all")

    if run_prefill:
        print("--- Prefill Benchmark ---")
        nvtx.range_push(f"PHASE_prefill_{mode_tag}")
        prefill = benchmark_prefill(
            llm, args.batch_size, args.input_len,
            args.num_iters, args.warmup_iters,
        )
        nvtx.range_pop()
        results["prefill"] = prefill.summary()
        print(f"\n  >>> Prefill avg GPU: {prefill.summary()['gpu_avg_ms']:.2f} ms\n")

    if run_decode:
        print("--- Decode Benchmark (subtraction method) ---")
        nvtx.range_push(f"PHASE_decode_{mode_tag}")
        decode = benchmark_decode(
            llm, args.batch_size, args.input_len, args.output_len,
            args.num_iters, args.warmup_iters,
        )
        nvtx.range_pop()
        results["decode"] = decode.summary()
        print(f"\n  >>> Decode per-token avg GPU: "
              f"{decode.summary()['gpu_avg_ms']:.3f} ms\n")

    if run_decode_e2e:
        print("--- Decode Benchmark (end-to-end) ---")
        nvtx.range_push(f"PHASE_decode_e2e_{mode_tag}")
        decode_e2e = benchmark_decode_e2e(
            llm, args.batch_size, args.input_len, args.output_len,
            args.num_iters, args.warmup_iters,
        )
        nvtx.range_pop()
        results["decode_e2e"] = decode_e2e.summary()
        e2e_avg = decode_e2e.summary()['gpu_avg_ms']
        print(f"\n  >>> Decode E2E avg GPU: {e2e_avg:.2f} ms "
              f"({e2e_avg / args.output_len:.3f} ms/token)\n")

    # ---- Summary ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"GPU  : {gpu_name}")
    print(f"Mode : {mode_tag}")
    if "prefill" in results:
        print(f"Prefill (input_len={args.input_len}, batch={args.batch_size}): "
              f"{results['prefill']['gpu_avg_ms']:.2f} ms (GPU avg)")
    if "decode" in results:
        print(f"Decode per-token (batch={args.batch_size}): "
              f"{results['decode']['gpu_avg_ms']:.3f} ms (GPU avg)")
    if "decode_e2e" in results:
        e2e = results['decode_e2e']['gpu_avg_ms']
        print(f"Decode E2E (input={args.input_len}, output={args.output_len}): "
              f"{e2e:.2f} ms total, {e2e / args.output_len:.3f} ms/token (GPU avg)")
    print("=" * 70)

    # ---- Save ----
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
