#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one-node online TP vs DP+EP serving benchmarks."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_list(name: str, default: str) -> list[int]:
    raw = env(name, default)
    try:
        return [int(item) for item in raw.split()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a space-separated list of integers") from exc


MODEL = env("MODEL", "deepseek-ai/DeepSeek-V2-Lite")
SERVED_MODEL_NAME = env("SERVED_MODEL_NAME", "bench-model")
HOST = env("HOST", "127.0.0.1")
BASE_PORT = int(env("BASE_PORT", "8100"))
TP_SIZES = env_list("TP_SIZES", "2 4 8")
DP_SIZES = env_list("DP_SIZES", "1 2 3 4 5 6 7 8")
NUM_PROMPTS = env("NUM_PROMPTS", "1000")
INPUT_LEN = env("INPUT_LEN", "1024")
OUTPUT_LEN = env("OUTPUT_LEN", "128")
REQUEST_RATE = env("REQUEST_RATE", "inf")
MAX_CONCURRENCY = env("MAX_CONCURRENCY", "")
MAX_MODEL_LEN = env("MAX_MODEL_LEN", "4096")
ALL2ALL_BACKEND = env("ALL2ALL_BACKEND", "allgather_reducescatter")
RESULT_ROOT = Path(
    env(
        "RESULT_ROOT",
        f"benchmarks/dp_ep_vs_tp/results/{datetime.now():%Y%m%d_%H%M%S}",
    )
)
SERVER_START_TIMEOUT = int(env("SERVER_START_TIMEOUT", "900"))
SERVER_EXTRA_ARGS = shlex.split(env("SERVER_EXTRA_ARGS", ""))
BENCH_EXTRA_ARGS = shlex.split(env("BENCH_EXTRA_ARGS", ""))
PYTHON_BIN = env("PYTHON_BIN", sys.executable)

SERVER_LOG_DIR = RESULT_ROOT / "server_logs"
BENCH_LOG_DIR = RESULT_ROOT / "bench_logs"
JSON_DIR = RESULT_ROOT / "json"

server_proc: subprocess.Popen[bytes] | None = None


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(
            f"{command} command not found. Activate/install the vLLM environment first."
        )


def gpu_list(count: int) -> str:
    return ",".join(str(i) for i in range(count))


def cleanup_server() -> None:
    global server_proc
    proc = server_proc
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    server_proc = None


def handle_signal(_signum: int, _frame: object) -> None:
    cleanup_server()
    raise SystemExit(128 + int(signal.SIGTERM))


def wait_for_server(port: int, log_file: Path) -> None:
    assert server_proc is not None
    health_url = f"http://{HOST}:{port}/health"
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    while time.monotonic() < deadline:
        if server_proc.poll() is not None:
            raise RuntimeError(f"Server exited before becoming ready. Log: {log_file}")
        try:
            with urlopen(health_url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except URLError:
            pass
        except TimeoutError:
            pass
        time.sleep(5)
    raise TimeoutError(
        f"Timed out waiting for server on {HOST}:{port}. Log: {log_file}"
    )


def run_case(
    case_name: str,
    gpu_count: int,
    port: int,
    server_args: list[str],
) -> None:
    global server_proc

    server_log = SERVER_LOG_DIR / f"{case_name}.log"
    bench_log = BENCH_LOG_DIR / f"{case_name}.log"
    result_json = JSON_DIR / f"{case_name}.json"
    cuda_devices = gpu_list(gpu_count)

    print(f"=== {case_name} on GPUs {cuda_devices} port {port} ===", flush=True)
    cleanup_server()

    server_cmd = [
        "vllm",
        "serve",
        MODEL,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(port),
        "--max-model-len",
        MAX_MODEL_LEN,
        *server_args,
        *SERVER_EXTRA_ARGS,
    ]
    server_env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices}
    with server_log.open("wb") as log:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=server_env,
        )

    wait_for_server(port, server_log)

    bench_cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--model",
        SERVED_MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(port),
        "--dataset-name",
        "random",
        "--input-len",
        INPUT_LEN,
        "--output-len",
        OUTPUT_LEN,
        "--num-prompts",
        NUM_PROMPTS,
        "--request-rate",
        REQUEST_RATE,
        "--save-result",
        "--result-dir",
        str(JSON_DIR),
        "--result-filename",
        f"{case_name}.json",
        "--metadata",
        f"case={case_name}",
        f"model={MODEL}",
        f"gpu_count={gpu_count}",
        f"input_len={INPUT_LEN}",
        f"output_len={OUTPUT_LEN}",
        *BENCH_EXTRA_ARGS,
    ]
    if MAX_CONCURRENCY:
        bench_cmd.extend(["--max-concurrency", MAX_CONCURRENCY])

    try:
        with bench_log.open("wb") as log:
            subprocess.run(
                bench_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        cleanup_server()

    if not result_json.is_file() or result_json.stat().st_size == 0:
        raise RuntimeError(f"Missing benchmark result JSON: {result_json}")


def summarize_results() -> None:
    python_path = Path(PYTHON_BIN)
    if not python_path.is_file() and shutil.which(PYTHON_BIN) is None:
        print(f"Skipping summary: {PYTHON_BIN} is not executable.", file=sys.stderr)
        return
    summary_path = RESULT_ROOT / "summary.csv"
    subprocess.run(
        [
            PYTHON_BIN,
            "benchmarks/dp_ep_vs_tp/summarize_results.py",
            str(JSON_DIR),
            "--output",
            str(summary_path),
        ],
        check=True,
    )
    print(f"Summary: {summary_path}")


def main() -> int:
    require_command("vllm")
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, handle_signal)
    try:
        case_index = 0
        for tp_size in TP_SIZES:
            run_case(
                case_name=f"tp{tp_size}",
                gpu_count=tp_size,
                port=BASE_PORT + case_index,
                server_args=["--tensor-parallel-size", str(tp_size)],
            )
            case_index += 1

        for dp_size in DP_SIZES:
            run_case(
                case_name=f"dp{dp_size}_ep",
                gpu_count=dp_size,
                port=BASE_PORT + case_index,
                server_args=[
                    "--data-parallel-size",
                    str(dp_size),
                    "--data-parallel-size-local",
                    str(dp_size),
                    "--enable-expert-parallel",
                    "--all2all-backend",
                    ALL2ALL_BACKEND,
                ],
            )
            case_index += 1

        summarize_results()
        print(f"Results: {RESULT_ROOT}")
        return 0
    finally:
        cleanup_server()


if __name__ == "__main__":
    raise SystemExit(main())
