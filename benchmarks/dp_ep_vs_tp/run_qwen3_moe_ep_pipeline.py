#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one-node Qwen3-MoE DP+EP throughput and module-profile pipeline."""

from __future__ import annotations

import csv
import re
from html import escape
from pathlib import Path

from single_node_common import (
    SingleNodeBenchmarkConfig,
    SingleNodeBenchmarkRunner,
    default_dp_sizes,
    default_result_root,
    detect_gpu_ids,
    env,
    env_bool,
    shlex_env,
    validate_sizes,
)

CASE_RE = re.compile(r"^dp(?P<gpu_count>\d+)_ep$")
MODULE_LABELS = {
    "vllm:attention": "Attention",
    "vllm:fused_moe": "FusedMoE",
}


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
    num_prompts: str,
    profile_modules: bool,
    base_port: int,
) -> SingleNodeBenchmarkConfig:
    bench_extra_args = ["--ignore-eos"]
    bench_extra_args.extend(shlex_env("BENCH_EXTRA_ARGS"))

    return SingleNodeBenchmarkConfig(
        model=env("MODEL", "Qwen/Qwen3-30B-A3B"),
        served_model_name=env("SERVED_MODEL_NAME", "qwen3-moe-bench"),
        host=env("HOST", "127.0.0.1"),
        base_port=base_port,
        gpu_ids=gpu_ids,
        num_prompts=num_prompts,
        input_len=env("INPUT_LEN", "1"),
        output_len=env("OUTPUT_LEN", "256"),
        request_rate=env("REQUEST_RATE", "inf"),
        max_concurrency=env("MAX_CONCURRENCY", ""),
        max_model_len=env("MAX_MODEL_LEN", "4096"),
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
        disable_prefix_caching=env_bool("DISABLE_PREFIX_CACHING", True),
    )


def run_dp_ep_matrix(
    *,
    result_root: Path,
    gpu_ids: list[str],
    dp_sizes: list[int],
    num_prompts: str,
    profile_modules: bool,
    base_port: int,
) -> None:
    config = make_config(
        result_root=result_root,
        gpu_ids=gpu_ids,
        num_prompts=num_prompts,
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
    write_line_svg(
        output_path,
        x_values,
        y_values,
        title="Qwen3-MoE DP+EP Online Throughput",
        y_label=metric.replace("_", " "),
    )


def load_case_gpu_counts(summary_csv: Path) -> dict[str, int]:
    return {
        row["case"]: int(row["gpu_count"] or case_gpu_count(row["case"]))
        for row in read_csv(summary_csv)
    }


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 13,
    anchor: str = "middle",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="serif" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(text)}</text>'
    )


def write_line_svg(
    output_path: Path,
    x_values: list[int],
    y_values: list[float],
    *,
    title: str,
    y_label: str,
) -> None:
    width, height = 820, 480
    left, right, top, bottom = 88, 32, 54, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(y_values) * 1.12 if y_values else 1.0
    x_min, x_max = min(x_values), max(x_values)
    x_span = max(x_max - x_min, 1)

    def sx(value: int) -> float:
        return left + (value - x_min) / x_span * plot_w

    def sy(value: float) -> float:
        return top + plot_h - value / y_max * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 28, title, size=18, weight="bold"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_h}" '
        f'x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
    ]
    for tick in range(5):
        y_value = y_max * tick / 4
        y = sy(y_value)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
            f'y2="{y:.1f}" stroke="#d0d0d0" stroke-width="0.8"/>'
        )
        elements.append(_svg_text(left - 10, y + 4, f"{y_value:.0f}", anchor="end"))
    points = " ".join(
        f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_values, y_values)
    )
    elements.append(
        f'<polyline points="{points}" fill="none" stroke="#1f5f5b" '
        'stroke-width="3"/>'
    )
    for x, y in zip(x_values, y_values):
        elements.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" '
            'fill="#1f5f5b"/>'
        )
        elements.append(_svg_text(sx(x), top + plot_h + 24, str(x)))
    elements.append(_svg_text(width / 2, height - 24, "Number of GPUs (DP+EP size)"))
    elements.append(
        f'<text x="24" y="{height / 2:.1f}" font-size="13" '
        'font-family="serif" text-anchor="middle" '
        f'transform="rotate(-90 24 {height / 2:.1f})">'
        f"{escape(y_label)}</text>"
    )
    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def write_stacked_bar_svg(
    output_path: Path,
    x_values: list[int],
    series: list[tuple[str, str, list[float]]],
    *,
    title: str,
    y_label: str,
) -> None:
    width, height = 820, 480
    left, right, top, bottom = 88, 140, 54, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    totals = [sum(values) for values in zip(*(item[2] for item in series))]
    y_max = max(totals) * 1.12 if totals else 1.0
    slot_w = plot_w / max(len(x_values), 1)
    bar_w = min(58.0, slot_w * 0.62)

    def sx(index: int) -> float:
        return left + slot_w * index + slot_w / 2

    def sy(value: float) -> float:
        return top + plot_h - value / y_max * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 28, title, size=18, weight="bold"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_h}" '
        f'x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
    ]
    for tick in range(5):
        y_value = y_max * tick / 4
        y = sy(y_value)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
            f'y2="{y:.1f}" stroke="#d0d0d0" stroke-width="0.8"/>'
        )
        elements.append(_svg_text(left - 10, y + 4, f"{y_value:.2f}", anchor="end"))

    bottoms = [0.0 for _ in x_values]
    for series_index, (label, color, values) in enumerate(series):
        for index, height_value in enumerate(values):
            x = sx(index) - bar_w / 2
            y = sy(bottoms[index] + height_value)
            rect_h = sy(bottoms[index]) - y
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{rect_h:.1f}" fill="{color}"/>'
            )
            bottoms[index] += height_value
        legend_y = top + 16 + 24 * series_index
        elements.append(
            f'<rect x="{left + plot_w + 28}" y="{legend_y - 11}" '
            f'width="14" height="14" fill="{color}"/>'
        )
        elements.append(
            _svg_text(left + plot_w + 50, legend_y, label, anchor="start")
        )

    for index, gpu_count in enumerate(x_values):
        elements.append(_svg_text(sx(index), top + plot_h + 24, str(gpu_count)))
    elements.append(_svg_text(width / 2, height - 24, "Number of GPUs (DP+EP size)"))
    elements.append(
        f'<text x="24" y="{height / 2:.1f}" font-size="13" '
        'font-family="serif" text-anchor="middle" '
        f'transform="rotate(-90 24 {height / 2:.1f})">'
        f"{escape(y_label)}</text>"
    )
    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


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
    write_stacked_bar_svg(
        output_path,
        x_values,
        series,
        title="Qwen3-MoE Attention vs FusedMoE Latency",
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
    dp_sizes = default_dp_sizes(gpu_count)
    validate_sizes("DP_SIZES", dp_sizes, gpu_count)

    num_prompts = env("NUM_PROMPTS", str(gpu_count * 1000))
    result_root = Path(
        env("RESULT_ROOT", str(default_result_root("qwen3_moe_ep_pipeline")))
    )
    throughput_root = result_root / "throughput"
    profile_root = result_root / "profile"

    print(
        "Qwen3-MoE DP+EP pipeline: "
        f"GPU_IDS={gpu_ids}, DP_SIZES={dp_sizes}, NUM_PROMPTS={num_prompts}",
        flush=True,
    )
    print(
        "DISABLE_PREFIX_CACHING disables prefix-cache reuse; vLLM serving still "
        "uses KV cache for autoregressive decoding.",
        flush=True,
    )

    run_dp_ep_matrix(
        result_root=throughput_root,
        gpu_ids=gpu_ids,
        dp_sizes=dp_sizes,
        num_prompts=num_prompts,
        profile_modules=False,
        base_port=int(env("BASE_PORT", "8100")),
    )
    plot_throughput(
        throughput_root / "summary.csv",
        result_root / "qwen3_moe_dp_ep_throughput.svg",
    )

    run_dp_ep_matrix(
        result_root=profile_root,
        gpu_ids=gpu_ids,
        dp_sizes=dp_sizes,
        num_prompts=num_prompts,
        profile_modules=True,
        base_port=int(env("PROFILE_BASE_PORT", env("BASE_PORT", "8100"))),
    )
    write_profile_breakdowns(profile_root)
    plot_module_latency(
        profile_root / "module_summary.csv",
        profile_root / "summary.csv",
        result_root / "qwen3_moe_module_latency_stacked.svg",
    )

    print(f"Results: {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
