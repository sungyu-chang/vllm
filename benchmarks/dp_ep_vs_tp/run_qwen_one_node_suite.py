#!/usr/bin/env python3
"""Run one-node TP vs DP+EP benchmarks for Qwen MoE presets.

This wrapper launches separate one-node runs for Qwen1.5 MoE and Qwen3 MoE,
using fixed TP sizes of 1, 2, 4, and 8, DP+EP sizes of 1 through 8, and then
invokes the analysis script after both runs complete.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "results" / "dp_ep_vs_tp" / "one_node_online"
ANALYSIS_ROOT = REPO_ROOT / "results" / "dp_ep_vs_tp" / "analysis"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    run_notes: str


MODEL_PRESETS = {
    "qwen1.5": ModelSpec(
        name="qwen1.5",
        model_id="Qwen/Qwen1.5-MoE-A2.7B",
        run_notes=(
            "Single-node TP vs DP+EP comparison for Qwen1.5-MoE-A2.7B, "
            "input 128, output 256, 5000 prompts, TP=1/2/4/8, DP+EP=1..8"
        ),
    ),
    "qwen3": ModelSpec(
        name="qwen3",
        model_id="Qwen/Qwen3-30B-A3B",
        run_notes=(
            "Single-node TP vs DP+EP comparison for Qwen3-30B-A3B, "
            "input 128, output 256, 5000 prompts, TP=1/2/4/8, DP+EP=1..8"
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen one-node TP vs DP+EP experiments and analyze them.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_PRESETS),
        default=["qwen1.5", "qwen3"],
        help="Model presets to run. Defaults to both Qwen1.5 and Qwen3.",
    )
    parser.add_argument("--gpu-count", default="8")
    parser.add_argument("--tp-sizes", default="1 2 4 8")
    parser.add_argument("--dp-sizes", default="1 2 3 4 5 6 7 8")
    parser.add_argument("--input-len", default="128")
    parser.add_argument("--output-len", default="256")
    parser.add_argument("--num-prompts", default="5000")
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--server-extra-args", default="--dtype bfloat16")
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    return parser.parse_args()


def existing_run_dirs() -> set[Path]:
    if not RUN_ROOT.is_dir():
        return set()
    return {path.resolve() for path in RUN_ROOT.iterdir() if path.is_dir()}


def detect_new_run_dir(before: set[Path]) -> Path:
    after = existing_run_dirs()
    new_dirs = sorted(after - before)
    if len(new_dirs) == 1:
        return new_dirs[0]
    if not new_dirs:
        raise RuntimeError("Benchmark finished but no new run directory was created.")
    return max(new_dirs, key=lambda path: path.stat().st_mtime)


def run_model(spec: ModelSpec, args: argparse.Namespace) -> Path:
    env = os.environ.copy()
    env["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    env.update({
        "MODEL": spec.model_id,
        "SERVER_EXTRA_ARGS": args.server_extra_args,
        "GPU_COUNT": args.gpu_count,
        "TP_SIZES": args.tp_sizes,
        "DP_SIZES": args.dp_sizes,
        "INPUT_LEN": args.input_len,
        "OUTPUT_LEN": args.output_len,
        "NUM_PROMPTS": args.num_prompts,
        "REQUEST_RATE": args.request_rate,
        "ALL2ALL_BACKEND": args.all2all_backend,
        "RUN_NOTES": spec.run_notes,
        "PYTHON_BIN": str(REPO_ROOT / ".venv" / "bin" / "python"),
    })

    before = existing_run_dirs()
    subprocess.run(
        [sys.executable, "benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    return detect_new_run_dir(before)


def analyze_runs(run_dirs: list[Path]) -> Path:
    analysis_dir = ANALYSIS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    subprocess.run(
        [
            sys.executable,
            "benchmarks/dp_ep_vs_tp/analyze_one_node_runs.py",
            *[str(run_dir) for run_dir in run_dirs],
            "--output-dir",
            str(analysis_dir),
            "--title",
            "Qwen TP vs DP+EP Total Token Throughput",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return analysis_dir


def main() -> int:
    args = parse_args()
    run_dirs: list[Path] = []
    for model_key in args.models:
        spec = MODEL_PRESETS[model_key]
        print(f"=== Running {spec.model_id} ===", flush=True)
        run_dir = run_model(spec, args)
        run_dirs.append(run_dir)
        print(f"Run directory for {spec.model_id}: {run_dir}", flush=True)

    analysis_dir = analyze_runs(run_dirs)
    print(f"Analysis directory: {analysis_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())