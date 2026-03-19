#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Triton fused MoE kernel vs native PyTorch per-expert matmul.

Usage examples:
    # Preset model, sweep token counts and distributions
    python expert_measurement/benchmark.py --model mixtral-8x7b \
        --num-tokens 32 128 512 --distribution uniform zipf --zipf-alpha 0.0 1.0

    # HuggingFace model (downloads config.json only, no weights)
    python expert_measurement/benchmark.py --hf-model mistralai/Mixtral-8x7B-v0.1 \
        --num-tokens 512

    # Manual config
    python expert_measurement/benchmark.py --num-experts 8 --top-k 2 \
        --hidden-size 4096 --intermediate-size 14336 --num-tokens 128 512

    # Expert parallelism: sweep EP=1,2,4 (simulates hosting fewer experts per GPU)
    python expert_measurement/benchmark.py --model mixtral-8x7b \
        --num-tokens 128 512 --ep 1 2 4

    # Single-expert comparison: multi-hosted (E_local allocated, 1 active)
    # vs single-hosted (1 allocated) at various batch sizes
    python expert_measurement/benchmark.py --model mixtral-8x7b \
        --single-expert --ep 1 2 4 8 --batch-sizes 1 4 16 64 256 1024

    # Profile mode for nsys/ncu
    python expert_measurement/benchmark.py --model mixtral-8x7b --num-tokens 512 \
        --profile --approach triton

    # List available presets
    python expert_measurement/benchmark.py --list-presets
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running as both `python benchmark.py` and `python -m expert_measurement`
_this_dir = Path(__file__).resolve().parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))

import numpy as np
import torch

from expert_measurement.expert_distribution import (
    generate_expert_assignments,
    get_distribution_stats,
    print_distribution_stats,
)
from expert_measurement.model_configs import (
    MoEModelConfig,
    get_config,
    list_presets,
)

# ---------------------------------------------------------------------------
# Native PyTorch reference (per-expert sequential matmul)
# ---------------------------------------------------------------------------


def native_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-expert torch.matmul loop — the 'native' baseline.

    Mirrors the reference implementation from tests/kernels/utils.py:torch_experts.
    """
    M, K = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_experts = w1.shape[0]

    a = hidden_states.view(M, 1, K).repeat(1, top_k, 1).reshape(-1, K)
    out = torch.zeros(M * top_k, w2.shape[1], dtype=a.dtype, device=a.device)

    flat_ids = topk_ids.view(-1)

    for i in range(num_experts):
        mask = flat_ids == i
        if mask.any():
            tmp1 = a[mask] @ w1[i].transpose(0, 1)
            tmp2 = torch.empty(
                tmp1.shape[0], tmp1.shape[1] // 2,
                dtype=tmp1.dtype, device=tmp1.device,
            )
            torch.ops._C.silu_and_mul(tmp2, tmp1)
            out[mask] = tmp2 @ w2[i].transpose(0, 1)

    return (
        (out.view(M, top_k, -1).to(torch.float32) * topk_weights.unsqueeze(-1))
        .sum(dim=1)
        .to(hidden_states.dtype)
    )


# ---------------------------------------------------------------------------
# Triton fused MoE (from vLLM)
# ---------------------------------------------------------------------------


def triton_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """vLLM's Triton fused_experts kernel."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
    )


# ---------------------------------------------------------------------------
# Padding waste estimation
# ---------------------------------------------------------------------------


def estimate_padding_waste(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int = 64,
) -> float:
    """Estimate fraction of compute wasted on padding rows for Triton kernel."""
    flat = topk_ids.view(-1).cpu()
    actual_total = 0
    padded_total = 0
    for eid in range(num_experts):
        count = int((flat == eid).sum().item())
        if count > 0:
            actual_total += count
            padded_total += ((count + block_size - 1) // block_size) * block_size
    if padded_total == 0:
        return 0.0
    return (padded_total - actual_total) / padded_total


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def benchmark_single(
    fn,
    args: tuple,
    num_warmup: int,
    num_iters: int,
    profile: bool = False,
    label: str = "",
) -> list[float]:
    """Run a function and return per-iteration GPU times in milliseconds."""
    # Warmup
    for _ in range(num_warmup):
        fn(*args)
    torch.cuda.synchronize()

    if profile:
        torch.cuda.cudart().cudaProfilerStart()

    times = []
    for i in range(num_iters):
        if profile:
            torch.cuda.nvtx.range_push(f"{label}_iter_{i}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

        if profile:
            torch.cuda.nvtx.range_pop()

    if profile:
        torch.cuda.cudart().cudaProfilerStop()

    return times


def run_benchmark(
    cfg: MoEModelConfig,
    num_tokens: int,
    distribution: str,
    zipf_alpha: float,
    dtype: torch.dtype,
    approaches: list[str],
    num_warmup: int,
    num_iters: int,
    profile: bool,
    tp: int,
    ep: int,
    seed: int,
) -> list[dict]:
    """Run benchmark for a single configuration, return result rows."""
    E = cfg.num_experts
    E_local = E // ep
    K = cfg.hidden_size
    N = cfg.intermediate_size // tp  # Simulate TP sharding
    top_k = min(cfg.top_k, E_local)

    # Generate routing directly among local experts
    topk_ids, topk_weights = generate_expert_assignments(
        num_tokens=num_tokens,
        num_experts=E_local,
        top_k=top_k,
        distribution=distribution,
        zipf_alpha=zipf_alpha,
        seed=seed,
    )

    dist_stats = get_distribution_stats(topk_ids, E_local)
    waste = estimate_padding_waste(topk_ids, E_local)

    # Generate data — weights are sized for local experts only
    hidden_states = torch.randn(num_tokens, K, dtype=dtype, device="cuda")
    w1 = torch.randn(E_local, 2 * N, K, dtype=dtype, device="cuda") * 0.01
    w2 = torch.randn(E_local, K, N, dtype=dtype, device="cuda") * 0.01

    dist_label = distribution
    if distribution == "zipf":
        dist_label = f"zipf(a={zipf_alpha})"

    ep_info = f", EP={ep}, local_E={E_local}" if ep > 1 else ""
    print(f"\n{'=' * 70}")
    print(
        f"Model: {cfg.model_name} (E={E}, top_k={top_k}, "
        f"H={K}, N={N}{f', TP={tp}' if tp > 1 else ''}{ep_info})"
    )
    print(f"Dtype: {dtype}, Tokens: {num_tokens}, Distribution: {dist_label}")
    print_distribution_stats(topk_ids, E_local)
    print(f"  Estimated padding waste (block=64): {waste:.1%}")
    print(f"{'=' * 70}")

    results = []

    timings = {}
    for approach in approaches:
        label, fn = FN_MAP[approach]

        if profile:
            torch.cuda.nvtx.range_push(f"benchmark_{approach}")

        times = benchmark_single(
            fn=fn,
            args=(hidden_states, w1, w2, topk_weights, topk_ids),
            num_warmup=num_warmup,
            num_iters=num_iters,
            profile=profile,
            label=approach,
        )

        if profile:
            torch.cuda.nvtx.range_pop()

        times_np = np.array(times)
        timings[approach] = times_np
        print(
            f"  {label:<16}: "
            f"mean={times_np.mean():.3f}ms, "
            f"median={np.median(times_np):.3f}ms, "
            f"min={times_np.min():.3f}ms, "
            f"max={times_np.max():.3f}ms, "
            f"std={times_np.std():.3f}ms"
        )

    # Compute speedup if both approaches were run
    if "triton" in timings and "native" in timings:
        speedup = np.median(timings["native"]) / np.median(timings["triton"])
        print(f"  Speedup (Triton vs Native): {speedup:.2f}x")
    else:
        speedup = None

    for approach in approaches:
        t = timings[approach]
        results.append({
            "model": cfg.model_name,
            "num_experts": E,
            "local_experts": E_local,
            "ep": ep,
            "top_k": top_k,
            "hidden_size": K,
            "intermediate_size": N,
            "num_tokens": num_tokens,
            "distribution": dist_label,
            "gini": f"{dist_stats['gini']:.3f}",
            "padding_waste": f"{waste:.3f}",
            "approach": approach,
            "mean_ms": f"{t.mean():.3f}",
            "median_ms": f"{np.median(t):.3f}",
            "min_ms": f"{t.min():.3f}",
            "max_ms": f"{t.max():.3f}",
            "std_ms": f"{t.std():.3f}",
            "speedup": f"{speedup:.2f}" if speedup else "",
        })

    # Free memory
    del hidden_states, w1, w2, topk_ids, topk_weights
    torch.cuda.empty_cache()

    return results


FN_MAP = {
    "triton": ("Triton Fused", triton_moe_forward),
    "native": ("Native PyTorch", native_moe_forward),
}


def run_single_expert_comparison(
    cfg: MoEModelConfig,
    batch_sizes: list[int],
    dtype: torch.dtype,
    approaches: list[str],
    num_warmup: int,
    num_iters: int,
    profile: bool,
    tp: int,
    ep_values: list[int],
) -> list[dict]:
    """Compare multi-expert-hosted (single activated) vs single-expert-hosted.

    For each batch size and EP value, runs two cases:
      - "multi_hosted":  Allocate E_local expert weights, route all tokens
                         to expert 0.  Simulates an EP rank where only one
                         expert receives traffic while others sit in memory.
      - "single_hosted": Allocate only 1 expert weight, route all tokens to
                         expert 0.  Pure efficiency baseline.

    The difference reveals overhead from unused expert weights occupying GPU
    memory (cache pollution, TLB pressure, etc.).
    """
    E = cfg.num_experts
    K = cfg.hidden_size
    N = cfg.intermediate_size // tp

    print(f"\n{'=' * 70}")
    print("SINGLE-EXPERT COMPARISON: multi-hosted vs single-hosted")
    print(f"  Model: {cfg.model_name} (E={E}, H={K}, N={N}"
          f"{f', TP={tp}' if tp > 1 else ''})")
    print(f"  EP values: {ep_values}  |  Batch sizes: {batch_sizes}")
    print(f"{'=' * 70}")

    results = []

    for ep_val in ep_values:
        E_local = E // ep_val

        print(f"\n--- EP={ep_val} → E_local={E_local} ---")

        for bs in batch_sizes:
            hidden_states = torch.randn(bs, K, dtype=dtype, device="cuda")
            topk_ids = torch.zeros(bs, 1, dtype=torch.int32, device="cuda")
            topk_weights = torch.ones(bs, 1, dtype=torch.float32, device="cuda")

            # Case A: multi-expert hosted, single activated
            w1_multi = torch.randn(
                E_local, 2 * N, K, dtype=dtype, device="cuda") * 0.01
            w2_multi = torch.randn(
                E_local, K, N, dtype=dtype, device="cuda") * 0.01

            # Case B: single expert hosted
            w1_single = torch.randn(
                1, 2 * N, K, dtype=dtype, device="cuda") * 0.01
            w2_single = torch.randn(
                1, K, N, dtype=dtype, device="cuda") * 0.01

            for approach in approaches:
                label, fn = FN_MAP[approach]

                # Run multi-hosted case
                times_multi = benchmark_single(
                    fn=fn,
                    args=(hidden_states, w1_multi, w2_multi,
                          topk_weights, topk_ids),
                    num_warmup=num_warmup,
                    num_iters=num_iters,
                    profile=profile,
                    label=f"multi_{approach}_ep{ep_val}_bs{bs}",
                )
                t_multi = np.array(times_multi)

                # Run single-hosted case
                times_single = benchmark_single(
                    fn=fn,
                    args=(hidden_states, w1_single, w2_single,
                          topk_weights, topk_ids),
                    num_warmup=num_warmup,
                    num_iters=num_iters,
                    profile=profile,
                    label=f"single_{approach}_bs{bs}",
                )
                t_single = np.array(times_single)

                overhead = (np.median(t_multi) / np.median(t_single) - 1) * 100
                print(
                    f"  bs={bs:<6} {label:<16}: "
                    f"multi={np.median(t_multi):.3f}ms, "
                    f"single={np.median(t_single):.3f}ms, "
                    f"overhead={overhead:+.1f}%"
                )

                # Record multi-hosted result
                results.append({
                    "model": cfg.model_name,
                    "num_experts": E,
                    "local_experts": E_local,
                    "ep": ep_val,
                    "top_k": 1,
                    "hidden_size": K,
                    "intermediate_size": N,
                    "num_tokens": bs,
                    "distribution": f"single_active(E_local={E_local})",
                    "gini": "1.000",
                    "padding_waste": "0.000",
                    "approach": approach,
                    "mean_ms": f"{t_multi.mean():.3f}",
                    "median_ms": f"{np.median(t_multi):.3f}",
                    "min_ms": f"{t_multi.min():.3f}",
                    "max_ms": f"{t_multi.max():.3f}",
                    "std_ms": f"{t_multi.std():.3f}",
                    "speedup": "",
                })

                # Record single-hosted result
                results.append({
                    "model": cfg.model_name,
                    "num_experts": E,
                    "local_experts": 1,
                    "ep": E,
                    "top_k": 1,
                    "hidden_size": K,
                    "intermediate_size": N,
                    "num_tokens": bs,
                    "distribution": "single_hosted",
                    "gini": "0.000",
                    "padding_waste": "0.000",
                    "approach": approach,
                    "mean_ms": f"{t_single.mean():.3f}",
                    "median_ms": f"{np.median(t_single):.3f}",
                    "min_ms": f"{t_single.min():.3f}",
                    "max_ms": f"{t_single.max():.3f}",
                    "std_ms": f"{t_single.std():.3f}",
                    "speedup": "",
                })

            del hidden_states, topk_ids, topk_weights
            del w1_multi, w2_multi, w1_single, w2_single

        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Triton fused MoE vs native PyTorch per-expert matmul",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model config source
    g = p.add_argument_group("Model configuration")
    g.add_argument("--model", type=str, help="Preset model name (see --list-presets)")
    g.add_argument("--hf-model", type=str, help="HuggingFace model name/path")
    g.add_argument("--num-experts", type=int, help="Override: number of experts")
    g.add_argument("--top-k", type=int, help="Override: experts per token")
    g.add_argument("--hidden-size", type=int, help="Override: hidden dimension")
    g.add_argument("--intermediate-size", type=int, help="Override: intermediate dim")
    g.add_argument("--tp", type=int, default=1, help="Simulate tensor parallelism (divides intermediate_size)")
    g.add_argument("--ep", type=int, nargs="+", default=[1], help="Expert parallelism degrees to sweep (e.g. 1 2 4)")
    g.add_argument("--list-presets", action="store_true", help="List presets and exit")

    # Workload
    g2 = p.add_argument_group("Workload")
    g2.add_argument("--num-tokens", type=int, nargs="+", default=[32, 128, 512], help="Token counts to benchmark")
    g2.add_argument("--distribution", type=str, nargs="+", default=["uniform"], help="Expert distributions: uniform, zipf, single_hot")
    g2.add_argument("--zipf-alpha", type=float, nargs="+", default=[1.0], help="Zipf exponents (used when distribution=zipf)")
    g2.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Compute dtype")
    g2.add_argument("--seed", type=int, default=42, help="Random seed")

    # Benchmark control
    g3 = p.add_argument_group("Benchmark control")
    g3.add_argument("--approach", type=str, nargs="+", default=["triton", "native"], choices=["triton", "native"], help="Which approaches to benchmark")
    g3.add_argument("--single-expert", action="store_true", help="Compare multi-expert-hosted (1 active) vs single-expert-hosted at various batch sizes")
    g3.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], help="Batch sizes for --single-expert comparison")
    g3.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    g3.add_argument("--num-iters", type=int, default=100, help="Timed iterations")

    # Profiler support
    g4 = p.add_argument_group("Profiler (nsys / ncu)")
    g4.add_argument("--profile", action="store_true", help="Enable profiling mode: adds NVTX markers, uses cudaProfilerApi, reduces iterations")
    g4.add_argument("--profile-iters", type=int, default=3, help="Number of iterations in profile mode (overrides --num-iters)")

    # Output
    g5 = p.add_argument_group("Output")
    g5.add_argument("--csv", type=str, help="Write results to CSV file")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list_presets:
        list_presets()
        sys.exit(0)

    cfg = get_config(
        model=args.model,
        hf_model=args.hf_model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Validate EP values
    for ep_val in args.ep:
        if cfg.num_experts % ep_val != 0:
            print(f"Error: --ep {ep_val} does not evenly divide "
                  f"num_experts={cfg.num_experts}")
            sys.exit(1)

    num_iters = args.profile_iters if args.profile else args.num_iters
    num_warmup = 2 if args.profile else args.num_warmup

    if args.profile:
        print("[Profile mode] NVTX markers enabled, cudaProfilerApi start/stop active")
        print(f"[Profile mode] Using {num_warmup} warmup + {num_iters} measured iterations")
        print("Run with: nsys profile --capture-range=cudaProfilerApi -o <output> python ...")
        print("      or: ncu --set full -o <output> python ... --approach triton --num-tokens <N>")

    all_results = []

    # ------------------------------------------------------------------
    # JIT warmup: run each kernel once per unique token count to trigger
    # Triton compilation *before* any timed benchmarking.
    # ------------------------------------------------------------------
    print("\n[Warmup] Pre-compiling kernels (Triton JIT) ...")
    K_w = cfg.hidden_size
    N_w = cfg.intermediate_size // args.tp

    # Collect all (E_local, token_count) pairs we'll benchmark
    warmup_configs: set[tuple[int, int]] = set()
    for ep_val in args.ep:
        E_local = cfg.num_experts // ep_val
        for nt in args.num_tokens:
            warmup_configs.add((E_local, nt))
    if args.single_expert:
        for ep_val in args.ep:
            E_local = cfg.num_experts // ep_val
            for bs in args.batch_sizes:
                warmup_configs.add((E_local, bs))
                warmup_configs.add((1, bs))

    for E_local, nt in sorted(warmup_configs):
        top_k_w = min(cfg.top_k, E_local)
        hs = torch.randn(nt, K_w, dtype=dtype, device="cuda")
        _w1 = torch.randn(E_local, 2 * N_w, K_w, dtype=dtype, device="cuda") * 0.01
        _w2 = torch.randn(E_local, K_w, N_w, dtype=dtype, device="cuda") * 0.01
        _ids, _weights = generate_expert_assignments(
            num_tokens=nt, num_experts=E_local, top_k=top_k_w,
            distribution="uniform", zipf_alpha=0.0, seed=0,
        )
        for approach in args.approach:
            fn = FN_MAP[approach][1]
            fn(hs, _w1, _w2, _weights, _ids)
        torch.cuda.synchronize()
        del hs, _w1, _w2, _ids, _weights
    torch.cuda.empty_cache()
    print("[Warmup] Done.\n")

    # Build the sweep: (distribution, zipf_alpha) pairs
    dist_configs = []
    for dist in args.distribution:
        if dist == "zipf":
            for alpha in args.zipf_alpha:
                dist_configs.append((dist, alpha))
        else:
            dist_configs.append((dist, 0.0))

    for ep_val in args.ep:
        for num_tokens in args.num_tokens:
            for dist, alpha in dist_configs:
                results = run_benchmark(
                    cfg=cfg,
                    num_tokens=num_tokens,
                    distribution=dist,
                    zipf_alpha=alpha,
                    dtype=dtype,
                    approaches=args.approach,
                    num_warmup=num_warmup,
                    num_iters=num_iters,
                    profile=args.profile,
                    tp=args.tp,
                    ep=ep_val,
                    seed=args.seed,
                )
                all_results.extend(results)

    # Single-expert comparison: multi-hosted vs single-hosted
    if args.single_expert:
        results = run_single_expert_comparison(
            cfg=cfg,
            batch_sizes=args.batch_sizes,
            dtype=dtype,
            approaches=args.approach,
            num_warmup=num_warmup,
            num_iters=num_iters,
            profile=args.profile,
            tp=args.tp,
            ep_values=args.ep,
        )
        all_results.extend(results)

    # Write CSV if requested
    if args.csv and all_results:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {path}")

    # Print summary table
    if len(all_results) > 2:
        print(f"\n{'=' * 90}")
        print("SUMMARY")
        print(f"{'=' * 90}")
        print(
            f"{'Tokens':>7} {'EP':>3} {'Local_E':>7} {'Distribution':<16} "
            f"{'Approach':<10} {'Median(ms)':>10} {'Waste':>7} {'Speedup':>8}"
        )
        print("-" * 80)
        for r in all_results:
            print(
                f"{r['num_tokens']:>7} {r['ep']:>3} {r['local_experts']:>7} "
                f"{r['distribution']:<16} {r['approach']:<10} "
                f"{r['median_ms']:>10} {r['padding_waste']:>7} {r['speedup']:>8}"
            )


if __name__ == "__main__":
    main()
