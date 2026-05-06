#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one-node Qwen3-MoE DP+EP throughput and module-profile pipeline."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from matplotlib_plots import save_line_plot, save_stacked_bar_plot
from single_node_common import (
    SingleNodeBenchmarkConfig,
    SingleNodeBenchmarkRunner,
    configured_sizes,
    default_dp_sizes,
    default_result_root,
    detect_gpu_ids,
    env,
    env_bool,
    shlex_env,
    validate_sizes,
    with_default_flag,
)

CASE_RE = re.compile(r"^dp(?P<gpu_count>\d+)_ep$")
MODULE_LABELS = {
    "vllm:attention": "Attention",
    "vllm:fused_moe": "FusedMoE",
}


def skip_optional_plot(exc: SystemExit) -> bool:
    message = str(exc)
    if "matplotlib is required for benchmark figures" not in message:
        return False
    print(f"Skipping optional plot generation: {message}", flush=True)
    return True


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def case_gpu_count(case_name: str) -> int:
    match = CASE_RE.match(case_name)
    if match is None:
        raise ValueError(f"Unexpected DP+EP case name: {case_name}")
    return int(match.group("gpu_count"))


def dp_ep_server_args(dp_size: int) -> list[str]:
    return [
        "--data-parallel-size",
        str(dp_size),
        "--data-parallel-size-local",
        str(dp_size),
        "--enable-expert-parallel",
        "--all2all-backend",
        env("ALL2ALL_BACKEND", "allgather_reducescatter"),
    ]


def make_config(
    *,
    result_root: Path,
    gpu_ids: list[str],
    profile_modules: bool,
    base_port: int,
) -> SingleNodeBenchmarkConfig:
    bench_extra_args = with_default_flag(shlex_env("BENCH_EXTRA_ARGS"),
                                         "--ignore-eos")

    return SingleNodeBenchmarkConfig(
        model=env("MODEL", "Qwen/Qwen3-30B-A3B"),
        served_model_name=env("SERVED_MODEL_NAME", "qwen3-moe-bench"),
        host=env("HOST", "127.0.0.1"),
        base_port=base_port,
        gpu_ids=gpu_ids,
        num_prompts=env("NUM_PROMPTS", ""),
        prompts_per_gpu=int(env("PROMPTS_PER_GPU", "1000")),
        input_len=env("INPUT_LEN", "1"),
        output_len=env("OUTPUT_LEN", "256"),
        request_rate=env("REQUEST_RATE", "inf"),
        max_concurrency=env("MAX_CONCURRENCY", ""),
        max_concurrency_per_gpu=env("MAX_CONCURRENCY_PER_GPU", ""),
        max_model_len=env("MAX_MODEL_LEN", ""),
        result_root=result_root,
        server_start_timeout=int(env("SERVER_START_TIMEOUT", "900")),
        server_extra_args=shlex_env("SERVER_EXTRA_ARGS", "--dtype bfloat16"),
        bench_extra_args=bench_extra_args,
        python_bin=env("PYTHON_BIN", ".venv/bin/python"),
        profile_modules=profile_modules,
        profile_delay_iterations=int(env("PROFILE_DELAY_ITERATIONS", "5")),
        profile_max_iterations=int(env("PROFILE_MAX_ITERATIONS", "20")),
        profile_with_stack=env_bool("PROFILE_WITH_STACK"),
        profile_layer_scopes=env_bool("PROFILE_LAYER_SCOPES", True),
        disable_prefix_caching=True,
    )


def run_dp_ep_matrix(
    *,
    result_root: Path,
    gpu_ids: list[str],
    dp_sizes: list[int],
    profile_modules: bool,
    base_port: int,
) -> None:
    config = make_config(
        result_root=result_root,
        gpu_ids=gpu_ids,
        profile_modules=profile_modules,
        base_port=base_port,
    )
    runner = SingleNodeBenchmarkRunner(config)
    runner.require_command("vllm")
    runner.setup_dirs()
    runner.install_signal_handlers()

    try:
        for index, dp_size in enumerate(dp_sizes):
            runner.run_case(
                case_name=f"dp{dp_size}_ep",
                gpu_count=dp_size,
                port=config.base_port + index,
                server_args=dp_ep_server_args(dp_size),
                metadata={
                    "parallelism": "dp_ep",
                    "dp_size": dp_size,
                    "ep_size": dp_size,
                    "disable_prefix_caching": config.disable_prefix_caching,
                },
            )
        runner.summarize_results()
        if profile_modules:
            runner.summarize_module_profiles()
    finally:
        runner.cleanup_server()


def plot_throughput(summary_csv: Path, output_path: Path) -> None:
    metric = env("THROUGHPUT_METRIC", "total_token_throughput")
    rows = sorted(read_csv(summary_csv), key=lambda row: int(row["gpu_count"]))
    x_values = [int(row["gpu_count"]) for row in rows]
    y_values = [float(row[metric]) for row in rows]
    save_line_plot(
        output_path,
        x_values,
        y_values,
        title="Qwen3-MoE DP+EP Online Throughput",
        x_label="Number of GPUs (DP+EP size)",
        y_label=metric.replace("_", " "),
    )


def load_case_gpu_counts(summary_csv: Path) -> dict[str, int]:
    return {
        row["case"]: int(row["gpu_count"] or case_gpu_count(row["case"]))
        for row in read_csv(summary_csv)
    }


def plot_module_latency(
    module_summary_csv: Path,
    profile_summary_csv: Path,
    output_path: Path,
) -> None:
    case_gpus = load_case_gpu_counts(profile_summary_csv)
    values: dict[int, dict[str, float]] = {}
    for row in read_csv(module_summary_csv):
        if row["rank"] != "all" or row["module"] not in MODULE_LABELS:
            continue
        gpu_count = case_gpus.get(row["case"], case_gpu_count(row["case"]))
        values.setdefault(gpu_count, {})[row["module"]] = float(row["avg_cuda_ms"])

    x_values = sorted(values)
    colors = {
        "vllm:attention": "#356f8c",
        "vllm:fused_moe": "#d18f2f",
    }
    series = []
    for module, label in MODULE_LABELS.items():
        series.append(
            (
                label,
                colors[module],
                [values[gpu].get(module, 0.0) for gpu in x_values],
            )
        )
    save_stacked_bar_plot(
        output_path,
        x_values,
        series,
        title="Qwen3-MoE Attention vs FusedMoE Latency",
        x_label="Number of GPUs (DP+EP size)",
        y_label="Average CUDA latency per module call (ms)",
    )


def write_profile_breakdowns(profile_root: Path) -> None:
    module_summary = profile_root / "module_summary.csv"
    profile_summary = profile_root / "summary.csv"
    case_gpus = load_case_gpu_counts(profile_summary)

    layer_rows: list[dict[str, object]] = []
    comm_rows: list[dict[str, object]] = []
    for row in read_csv(module_summary):
        if row["rank"] != "all":
            continue
        module = row["module"]
        gpu_count = case_gpus.get(row["case"], case_gpu_count(row["case"]))
        common = {
            "case": row["case"],
            "gpu_count": gpu_count,
            "count": row["count"],
            "total_cuda_ms": row["total_cuda_ms"],
            "avg_cuda_ms": row["avg_cuda_ms"],
        }
        for prefix, kind in (
            ("vllm:attention:", "attention"),
            ("vllm:fused_moe:", "fused_moe"),
        ):
            if module.startswith(prefix):
                layer_rows.append(
                    {
                        **common,
                        "module_kind": kind,
                        "layer": module.removeprefix(prefix),
                    }
                )
        if module.startswith("vllm:moe_comm"):
            comm_rows.append({**common, "module": module})

    write_csv(
        profile_root / "per_layer_module_summary.csv",
        [
            "case",
            "gpu_count",
            "module_kind",
            "layer",
            "count",
            "total_cuda_ms",
            "avg_cuda_ms",
        ],
        layer_rows,
    )
    write_csv(
        profile_root / "moe_comm_summary.csv",
        [
            "case",
            "gpu_count",
            "module",
            "count",
            "total_cuda_ms",
            "avg_cuda_ms",
        ],
        comm_rows,
    )


def main() -> int:
    gpu_ids = detect_gpu_ids()
    gpu_count = len(gpu_ids)
    dp_sizes = configured_sizes("DP_SIZES", default_dp_sizes(gpu_count))
    validate_sizes("DP_SIZES", dp_sizes, gpu_count)
    run_throughput = env_bool("RUN_THROUGHPUT", True)
    run_profile = env_bool("RUN_PROFILE", True)
    if not run_throughput and not run_profile:
        raise SystemExit("At least one of RUN_THROUGHPUT or RUN_PROFILE must be 1.")

    result_root = Path(
        env("RESULT_ROOT", str(default_result_root("qwen3_moe_ep_pipeline")))
    )
    throughput_root = result_root / "throughput"
    profile_root = result_root / "profile"

    print(
        "Qwen3-MoE DP+EP pipeline: "
        f"GPU_IDS={gpu_ids}, DP_SIZES={dp_sizes}, "
        f"NUM_PROMPTS_PER_CASE=DP_SIZE*{env('PROMPTS_PER_GPU', '1000')}, "
        f"RUN_THROUGHPUT={run_throughput}, RUN_PROFILE={run_profile}",
        flush=True,
    )
    print(
        "DISABLE_PREFIX_CACHING disables prefix-cache reuse; vLLM serving still "
        "uses KV cache for autoregressive decoding.",
        flush=True,
    )
    print(
        "Qwen3-MoE pipeline always passes --no-enable-prefix-caching and "
        "--ignore-eos.",
        flush=True,
    )

    if run_throughput:
        run_dp_ep_matrix(
            result_root=throughput_root,
            gpu_ids=gpu_ids,
            dp_sizes=dp_sizes,
            profile_modules=False,
            base_port=int(env("BASE_PORT", "8100")),
        )
        try:
            plot_throughput(
                throughput_root / "summary.csv",
                result_root / "qwen3_moe_dp_ep_throughput.png",
            )
        except SystemExit as exc:
            if not skip_optional_plot(exc):
                raise

    if run_profile:
        run_dp_ep_matrix(
            result_root=profile_root,
            gpu_ids=gpu_ids,
            dp_sizes=dp_sizes,
            profile_modules=True,
            base_port=int(env("PROFILE_BASE_PORT", env("BASE_PORT", "8100"))),
        )
        write_profile_breakdowns(profile_root)
        try:
            plot_module_latency(
                profile_root / "module_summary.csv",
                profile_root / "summary.csv",
                result_root / "qwen3_moe_module_latency_stacked.png",
            )
        except SystemExit as exc:
            if not skip_optional_plot(exc):
                raise

    print(f"Results: {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
