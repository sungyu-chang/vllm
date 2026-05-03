#!/usr/bin/env python3
"""Plot total token throughput vs GPU count for TP and DP+EP runs.

This script reads one or more run directories or summary CSV files produced by
benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py and writes an SVG figure.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


SeriesMap = dict[str, list[tuple[int, float]]]
ModelMap = dict[str, SeriesMap]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw total token throughput vs GPU count from one or more "
            "dp_ep_vs_tp summary CSVs or run directories."
        ))
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Run directories containing summary.csv or explicit summary.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to results/dp_ep_vs_tp/total_token_throughput.svg",
    )
    parser.add_argument(
        "--title",
        default="Total Token Throughput vs GPU Count",
        help="Figure title.",
    )
    return parser.parse_args()


def resolve_summary_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            summary_path = input_path / "summary.csv"
            if not summary_path.is_file():
                raise SystemExit(f"Missing summary.csv under run directory: {input_path}")
            paths.append(summary_path)
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise SystemExit(f"Input path does not exist: {input_path}")
    return paths


def series_name(case_name: str) -> str | None:
    if case_name.startswith("tp"):
        return "TP"
    if case_name.startswith("dp") and case_name.endswith("_ep"):
        return "DP+EP"
    return None


def load_rows(summary_paths: list[Path]) -> ModelMap:
    model_series: ModelMap = defaultdict(lambda: defaultdict(list))
    for summary_path in summary_paths:
        with summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                case_name = (row.get("case") or "").strip()
                series = series_name(case_name)
                if series is None:
                    continue

                model_name = (row.get("model") or "unknown").strip()
                try:
                    gpu_count = int(float(row["gpu_count"]))
                    total_token_throughput = float(row["total_token_throughput"])
                except (TypeError, ValueError, KeyError) as exc:
                    raise SystemExit(
                        f"Invalid row in {summary_path}: {row}"
                    ) from exc

                model_series[model_name][series].append(
                    (gpu_count, total_token_throughput)
                )

    if not model_series:
        raise SystemExit("No TP or DP+EP rows found in the provided summaries.")

    for series_map in model_series.values():
        for points in series_map.values():
            points.sort(key=lambda item: item[0])

    return dict(model_series)


def format_tick(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def line_path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
        + [f"L {x:.1f} {y:.1f}" for x, y in points[1:]]
    )


def render_svg(title: str, model_series: ModelMap) -> str:
    colors = {"TP": "#0f766e", "DP+EP": "#b45309"}
    width = 1100
    panel_width = 480
    panel_height = 320
    margin_left = 70
    margin_right = 24
    margin_top = 70
    margin_bottom = 60
    panel_gap = 40
    top_gap = 40

    models = list(model_series.items())
    total_height = margin_top + top_gap + len(models) * panel_height + (len(models) - 1) * panel_gap + margin_bottom
    total_width = width

    all_gpu_counts = sorted({x for series_map in model_series.values() for points in series_map.values() for x, _ in points})
    max_y = max(y for series_map in model_series.values() for points in series_map.values() for _, y in points)
    y_max = math.ceil(max_y / 1000) * 1000 if max_y > 1000 else math.ceil(max_y / 100) * 100
    y_ticks = 5

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #111827; }',
        '.title { font-size: 24px; font-weight: 700; }',
        '.subtitle { font-size: 12px; fill: #4b5563; }',
        '.axis { stroke: #374151; stroke-width: 1.5; }',
        '.grid { stroke: #e5e7eb; stroke-width: 1; }',
        '.tick { font-size: 12px; fill: #4b5563; }',
        '.panel-title { font-size: 16px; font-weight: 600; }',
        '.legend { font-size: 13px; }',
        '</style>',
        f'<rect width="{total_width}" height="{total_height}" fill="#ffffff" />',
        f'<text x="{margin_left}" y="36" class="title">{escape(title)}</text>',
        '<text x="70" y="56" class="subtitle">x-axis: number of GPUs involved, y-axis: total token throughput</text>',
    ]

    legend_x = total_width - 220
    legend_y = 34
    for idx, (name, color) in enumerate(colors.items()):
        y = legend_y + idx * 20
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3" />')
        parts.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="4" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 38}" y="{y + 4}" class="legend">{escape(name)}</text>')

    plot_width = total_width - margin_left - margin_right

    for panel_idx, (model_name, series_map) in enumerate(models):
        top = margin_top + top_gap + panel_idx * (panel_height + panel_gap)
        left = margin_left
        right = left + plot_width
        bottom = top + panel_height

        parts.append(f'<text x="{left}" y="{top - 18}" class="panel-title">{escape(model_name)}</text>')

        for tick_idx in range(y_ticks + 1):
            y_value = y_max * tick_idx / y_ticks
            y = bottom - (panel_height * tick_idx / y_ticks)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid" />')
            parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{format_tick(y_value)}</text>')

        for gpu_count in all_gpu_counts:
            ratio = 0 if len(all_gpu_counts) == 1 else (all_gpu_counts.index(gpu_count) / (len(all_gpu_counts) - 1))
            x = left + ratio * plot_width
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid" />')
            parts.append(f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" class="tick">{gpu_count}</text>')

        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis" />')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis" />')
        parts.append(f'<text x="{(left + right) / 2:.1f}" y="{bottom + 46}" text-anchor="middle" class="tick">GPUs involved</text>')
        parts.append(
            f'<text x="{left - 54}" y="{(top + bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 {left - 54} {(top + bottom) / 2:.1f})" class="tick">Total token throughput</text>'
        )

        for series_name_value, color in colors.items():
            raw_points = series_map.get(series_name_value, [])
            if not raw_points:
                continue

            svg_points: list[tuple[float, float]] = []
            for gpu_count, throughput in raw_points:
                ratio_x = 0 if len(all_gpu_counts) == 1 else (all_gpu_counts.index(gpu_count) / (len(all_gpu_counts) - 1))
                ratio_y = 0 if y_max == 0 else throughput / y_max
                x = left + ratio_x * plot_width
                y = bottom - ratio_y * panel_height
                svg_points.append((x, y))

            parts.append(f'<path d="{line_path(svg_points)}" fill="none" stroke="{color}" stroke-width="3" />')
            for x, y in svg_points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" />')

    parts.append('</svg>')
    return "\n".join(parts)


def default_output_path() -> Path:
    return Path("results/dp_ep_vs_tp/total_token_throughput.svg")


def main() -> None:
    args = parse_args()
    summary_paths = resolve_summary_paths(args.inputs)
    model_series = load_rows(summary_paths)
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(args.title, model_series), encoding="utf-8")
    print(f"Wrote figure to {output_path}")


if __name__ == "__main__":
    main()