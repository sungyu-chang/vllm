#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Analyze results from the CUDA Graph GPU Benchmark.

Loads JSON result files produced by cuda_graph_gpu_benchmark.py, computes
speedup metrics, and prints comparison tables. Optionally parses nsys
SQLite exports for kernel-level analysis.

Usage:
    # Analyze all results in the output directory
    python analyze_cuda_graph_gpu_results.py --results-dir cuda_graph_bench_results

    # Analyze specific GPU results
    python analyze_cuda_graph_gpu_results.py \
        --results-dir cuda_graph_bench_results/NVIDIA_A100-SXM4-80GB

    # Include nsys kernel analysis (requires nsys SQLite exports)
    python analyze_cuda_graph_gpu_results.py \
        --results-dir cuda_graph_bench_results --nsys
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> list[dict]:
    """Recursively load all JSON result files from a directory."""
    results = []
    for json_file in sorted(Path(results_dir).rglob("*.json")):
        # Skip nsys-generated JSON files that don't have our format
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "config" in data and "gpu_name" in data.get("config", {}):
                data["_file"] = str(json_file)
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


# ---------------------------------------------------------------------------
# Organizing results into a comparison structure
# ---------------------------------------------------------------------------

def organize_results(
    results: list[dict],
) -> dict[str, dict[str, dict[str, dict]]]:
    """
    Organize results into a nested dict:
        gpu_name -> phase -> config_key -> result_data

    config_key encodes batch_size and input_len/output_len.
    """
    organized: dict = defaultdict(lambda: defaultdict(dict))

    for r in results:
        cfg = r["config"]
        gpu = cfg["gpu_name"]
        mode = cfg["mode"]
        bs = cfg["batch_size"]
        il = cfg["input_len"]
        ol = cfg["output_len"]

        for phase_key in ("prefill", "decode", "decode_e2e"):
            if phase_key not in r:
                continue
            config_tag = f"bs{bs}_in{il}_out{ol}"
            entry_key = f"{mode}_{config_tag}"
            organized[gpu][phase_key][entry_key] = {
                "mode": mode,
                "batch_size": bs,
                "input_len": il,
                "output_len": ol,
                "gpu_avg_ms": r[phase_key]["gpu_avg_ms"],
                "gpu_median_ms": r[phase_key]["gpu_median_ms"],
                "gpu_min_ms": r[phase_key]["gpu_min_ms"],
                "gpu_max_ms": r[phase_key]["gpu_max_ms"],
                "wall_avg_ms": r[phase_key]["wall_avg_ms"],
            }

    return dict(organized)


# ---------------------------------------------------------------------------
# Speedup computation
# ---------------------------------------------------------------------------

def compute_speedups(organized: dict) -> list[dict]:
    """
    For each (gpu, phase, config), compute:
        speedup = eager_time / cuda_graph_time
    """
    rows = []
    for gpu, phases in sorted(organized.items()):
        for phase, entries in sorted(phases.items()):
            # Group by config (batch_size, input_len, output_len)
            configs: dict[str, dict] = {}
            for key, data in entries.items():
                mode = data["mode"]
                config_tag = key.replace(f"{mode}_", "", 1)
                if config_tag not in configs:
                    configs[config_tag] = {}
                configs[config_tag][mode] = data

            for config_tag, modes in sorted(configs.items()):
                if "eager" in modes and "cuda_graph" in modes:
                    eager = modes["eager"]
                    cg = modes["cuda_graph"]
                    speedup_gpu = eager["gpu_avg_ms"] / cg["gpu_avg_ms"]
                    speedup_wall = eager["wall_avg_ms"] / cg["wall_avg_ms"]
                    rows.append({
                        "gpu": gpu,
                        "phase": phase,
                        "config": config_tag,
                        "batch_size": eager["batch_size"],
                        "input_len": eager["input_len"],
                        "output_len": eager["output_len"],
                        "eager_gpu_ms": eager["gpu_avg_ms"],
                        "cudagraph_gpu_ms": cg["gpu_avg_ms"],
                        "speedup_gpu": speedup_gpu,
                        "eager_wall_ms": eager["wall_avg_ms"],
                        "cudagraph_wall_ms": cg["wall_avg_ms"],
                        "speedup_wall": speedup_wall,
                    })
    return rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_comparison_table(rows: list[dict]):
    """Print a formatted comparison table."""
    if not rows:
        print("No matching eager/cuda_graph pairs found for comparison.")
        return

    # Group by phase
    phases = sorted(set(r["phase"] for r in rows))
    for phase in phases:
        phase_rows = [r for r in rows if r["phase"] == phase]
        if not phase_rows:
            continue

        unit = "ms" if phase != "decode" else "ms/token"
        print(f"\n{'=' * 90}")
        print(f"  Phase: {phase}")
        print(f"{'=' * 90}")

        # Header
        print(f"  {'GPU':<35} {'Config':<22} "
              f"{'Eager':>10} {'CUDAGraph':>10} {'Speedup':>8}")
        print(f"  {'':35} {'':22} "
              f"{'(' + unit + ')':>10} {'(' + unit + ')':>10} {'(x)':>8}")
        print(f"  {'-' * 87}")

        for r in phase_rows:
            gpu_short = r["gpu"]
            if len(gpu_short) > 33:
                gpu_short = gpu_short[:30] + "..."
            print(f"  {gpu_short:<35} {r['config']:<22} "
                  f"{r['eager_gpu_ms']:>10.3f} {r['cudagraph_gpu_ms']:>10.3f} "
                  f"{r['speedup_gpu']:>8.3f}")

    # Cross-GPU speedup comparison (the key hypothesis table)
    gpus = sorted(set(r["gpu"] for r in rows))
    if len(gpus) > 1:
        print(f"\n{'=' * 90}")
        print("  Cross-GPU CUDA Graph Speedup Comparison")
        print(f"  (Hypothesis: speedup increases with GPU generation)")
        print(f"{'=' * 90}")

        for phase in phases:
            phase_rows = [r for r in rows if r["phase"] == phase]
            if not phase_rows:
                continue

            configs = sorted(set(r["config"] for r in phase_rows))
            for config in configs:
                config_rows = [r for r in phase_rows if r["config"] == config]
                if len(config_rows) < 2:
                    continue

                print(f"\n  Phase: {phase}, Config: {config}")
                print(f"  {'GPU':<35} {'Eager (ms)':>12} "
                      f"{'CUDAGraph (ms)':>14} {'Speedup':>10}")
                print(f"  {'-' * 73}")

                for r in sorted(config_rows,
                                key=lambda x: x["eager_gpu_ms"],
                                reverse=True):
                    gpu_short = r["gpu"]
                    if len(gpu_short) > 33:
                        gpu_short = gpu_short[:30] + "..."
                    print(f"  {gpu_short:<35} {r['eager_gpu_ms']:>12.3f} "
                          f"{r['cudagraph_gpu_ms']:>14.3f} "
                          f"{r['speedup_gpu']:>9.3f}x")


def print_raw_results(organized: dict):
    """Print all raw results grouped by GPU and phase."""
    for gpu, phases in sorted(organized.items()):
        print(f"\n{'=' * 70}")
        print(f"  GPU: {gpu}")
        print(f"{'=' * 70}")

        for phase, entries in sorted(phases.items()):
            print(f"\n  Phase: {phase}")
            print(f"  {'Mode':<12} {'Config':<22} "
                  f"{'GPU avg':>10} {'GPU med':>10} "
                  f"{'GPU min':>10} {'GPU max':>10} {'Wall avg':>10}")
            print(f"  {'-' * 76}")

            for key in sorted(entries.keys()):
                data = entries[key]
                print(f"  {data['mode']:<12} "
                      f"bs{data['batch_size']}_in{data['input_len']}"
                      f"_out{data['output_len']:<6} "
                      f"{data['gpu_avg_ms']:>10.3f} "
                      f"{data['gpu_median_ms']:>10.3f} "
                      f"{data['gpu_min_ms']:>10.3f} "
                      f"{data['gpu_max_ms']:>10.3f} "
                      f"{data['wall_avg_ms']:>10.3f}")


# ---------------------------------------------------------------------------
# nsys SQLite analysis (optional)
# ---------------------------------------------------------------------------

def analyze_nsys_sqlite(sqlite_path: str) -> dict:
    """
    Extract key metrics from an nsys SQLite export.

    Returns dict with:
        - total_kernel_time_ms: sum of all GPU kernel durations
        - num_kernel_launches: count of kernel launches
        - avg_kernel_duration_us: average kernel duration
        - num_cuda_graph_launches: count of cudaGraphLaunch calls
        - num_cuda_launch_kernel: count of cudaLaunchKernel calls
    """
    conn = sqlite3.connect(sqlite_path)
    metrics = {}

    try:
        # GPU kernel execution time
        try:
            cur = conn.execute("""
                SELECT COUNT(*), SUM(end - start), AVG(end - start)
                FROM CUPTI_ACTIVITY_KIND_KERNEL
            """)
            row = cur.fetchone()
            if row and row[0]:
                metrics["num_kernels"] = row[0]
                metrics["total_kernel_time_ms"] = row[1] / 1e6  # ns -> ms
                metrics["avg_kernel_duration_us"] = row[2] / 1e3  # ns -> us
        except sqlite3.OperationalError:
            pass

        # CUDA API calls: cudaLaunchKernel vs cudaGraphLaunch
        try:
            cur = conn.execute("""
                SELECT
                    s.value as api_name,
                    COUNT(*) as call_count,
                    SUM(r.end - r.start) as total_time_ns
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON r.nameId = s.id
                WHERE s.value LIKE 'cudaLaunchKernel%'
                   OR s.value LIKE 'cudaGraphLaunch%'
                GROUP BY s.value
            """)
            for row in cur.fetchall():
                name = row[0]
                if "cudaGraphLaunch" in name:
                    metrics["num_cuda_graph_launches"] = row[1]
                    metrics["cuda_graph_launch_time_ms"] = row[2] / 1e6
                elif "cudaLaunchKernel" in name:
                    metrics["num_cuda_launch_kernel"] = row[1]
                    metrics["cuda_launch_kernel_time_ms"] = row[2] / 1e6
        except sqlite3.OperationalError:
            pass

    finally:
        conn.close()

    return metrics


def print_nsys_comparison(results_dir: str):
    """Find and compare nsys SQLite exports for eager vs cuda_graph."""
    sqlite_files = sorted(Path(results_dir).rglob("*.sqlite"))
    if not sqlite_files:
        print("\nNo nsys SQLite exports found. "
              "Run with --nsys flag to generate them.")
        return

    print(f"\n{'=' * 90}")
    print("  nsys Kernel-Level Analysis")
    print(f"{'=' * 90}")

    analyses = {}
    for sqlite_file in sqlite_files:
        tag = sqlite_file.stem  # e.g., nsys_cudagraph_prefill_bs1_in128_out128
        metrics = analyze_nsys_sqlite(str(sqlite_file))
        if metrics:
            analyses[tag] = metrics
            print(f"\n  {tag}:")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"    {k:<35} {v:>12.3f}")
                else:
                    print(f"    {k:<35} {v:>12}")

    # Compare eager vs cuda_graph pairs
    eager_tags = [t for t in analyses if "eager" in t]
    cg_tags = [t for t in analyses if "cudagraph" in t]

    for eager_tag in eager_tags:
        # Find matching cuda_graph tag
        config_part = eager_tag.replace("nsys_eager_", "")
        matching_cg = f"nsys_cudagraph_{config_part}"
        if matching_cg in analyses:
            eager = analyses[eager_tag]
            cg = analyses[matching_cg]

            print(f"\n  Comparison: {config_part}")
            print(f"  {'Metric':<35} {'Eager':>12} {'CUDAGraph':>12} {'Ratio':>8}")
            print(f"  {'-' * 69}")

            compare_keys = [
                ("num_kernels", "Kernel count"),
                ("num_cuda_launch_kernel", "cudaLaunchKernel calls"),
                ("num_cuda_graph_launches", "cudaGraphLaunch calls"),
                ("total_kernel_time_ms", "Total kernel time (ms)"),
                ("avg_kernel_duration_us", "Avg kernel duration (us)"),
                ("cuda_launch_kernel_time_ms", "cudaLaunchKernel time (ms)"),
            ]
            for key, label in compare_keys:
                e_val = eager.get(key, 0)
                c_val = cg.get(key, 0)
                ratio = e_val / c_val if c_val else float("inf")
                if isinstance(e_val, float):
                    print(f"  {label:<35} {e_val:>12.3f} {c_val:>12.3f} "
                          f"{ratio:>7.2f}x")
                else:
                    print(f"  {label:<35} {e_val:>12} {c_val:>12} "
                          f"{ratio:>7.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA Graph GPU Benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing JSON result files "
             "(from run_cuda_graph_gpu_benchmark.sh)")
    parser.add_argument(
        "--raw", action="store_true",
        help="Print raw results in addition to comparison tables.")
    parser.add_argument(
        "--nsys", action="store_true",
        help="Include nsys kernel-level analysis "
             "(requires nsys SQLite exports).")
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Write speedup table to CSV file.")

    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result files from {args.results_dir}")

    organized = organize_results(results)
    speedups = compute_speedups(organized)

    if args.raw:
        print_raw_results(organized)

    print_comparison_table(speedups)

    if args.nsys:
        print_nsys_comparison(args.results_dir)

    # Export CSV
    if args.output_csv and speedups:
        import csv
        fields = list(speedups[0].keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(speedups)
        print(f"\nSpeedup table saved to {args.output_csv}")


if __name__ == "__main__":
    main()
