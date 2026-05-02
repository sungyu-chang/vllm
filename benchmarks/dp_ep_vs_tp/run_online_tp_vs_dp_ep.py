#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one-node online TP vs DP+EP serving benchmarks."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.dp_ep_vs_tp.results_layout import build_result_root, write_run_readme


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_list(name: str, default: str) -> list[int]:
    raw = env(name, default)
    try:
        return [int(item) for item in raw.split()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a space-separated list of integers") from exc


def parse_cuda_visible_devices() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() in ("", "-1", "NoDevFiles"):
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def detect_gpu_ids() -> list[str]:
    visible_ids = parse_cuda_visible_devices()
    if "GPU_COUNT" in os.environ:
        try:
            count = int(os.environ["GPU_COUNT"])
        except ValueError as exc:
            raise SystemExit("GPU_COUNT must be an integer") from exc
        if count < 1:
            raise SystemExit("GPU_COUNT must be >= 1")
        if visible_ids is not None:
            if count > len(visible_ids):
                raise SystemExit(
                    "GPU_COUNT exceeds CUDA_VISIBLE_DEVICES length: "
                    f"{count} > {len(visible_ids)}"
                )
            return visible_ids[:count]
        return [str(index) for index in range(count)]

    if visible_ids is not None:
        return visible_ids

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise SystemExit(
            "Unable to detect GPU count: set GPU_COUNT or install nvidia-smi."
        )

    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Unable to detect GPU count from nvidia-smi; set GPU_COUNT."
        ) from exc

    count = sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))
    if count < 1:
        raise SystemExit("No GPUs detected; set GPU_COUNT to override.")
    return [str(index) for index in range(count)]


def default_tp_sizes(gpu_count: int) -> list[int]:
    sizes: list[int] = []
    size = 1
    while size <= gpu_count:
        sizes.append(size)
        size *= 2
    return sizes


def default_dp_sizes(gpu_count: int) -> list[int]:
    return list(range(1, gpu_count + 1))


def configured_sizes(name: str, default_sizes: list[int]) -> list[int]:
    if name in os.environ:
        return env_list(name, "")
    return default_sizes


def validate_sizes(name: str, sizes: list[int], gpu_count: int) -> None:
    if not sizes:
        raise SystemExit(f"{name} must not be empty")
    invalid = [size for size in sizes if size < 1 or size > gpu_count]
    if invalid:
        raise SystemExit(
            f"{name} contains sizes outside available GPU count "
            f"{gpu_count}: {invalid}"
        )


MODEL = env("MODEL", "deepseek-ai/DeepSeek-V2-Lite")
SERVED_MODEL_NAME = env("SERVED_MODEL_NAME", "bench-model")
HOST = env("HOST", "127.0.0.1")
BASE_PORT = int(env("BASE_PORT", "8100"))
GPU_IDS = detect_gpu_ids()
GPU_COUNT = len(GPU_IDS)
TP_SIZES = configured_sizes("TP_SIZES", default_tp_sizes(GPU_COUNT))
DP_SIZES = configured_sizes("DP_SIZES", default_dp_sizes(GPU_COUNT))
validate_sizes("TP_SIZES", TP_SIZES, GPU_COUNT)
validate_sizes("DP_SIZES", DP_SIZES, GPU_COUNT)
NUM_PROMPTS = env("NUM_PROMPTS", "1000")
INPUT_LEN = env("INPUT_LEN", "1024")
OUTPUT_LEN = env("OUTPUT_LEN", "128")
REQUEST_RATE = env("REQUEST_RATE", "inf")
MAX_CONCURRENCY = env("MAX_CONCURRENCY", "")
MAX_MODEL_LEN = env("MAX_MODEL_LEN", "4096")
ALL2ALL_BACKEND = env("ALL2ALL_BACKEND", "allgather_reducescatter")
RESULT_ROOT = build_result_root("dp_ep_vs_tp", "one_node_online")
SERVER_START_TIMEOUT = int(env("SERVER_START_TIMEOUT", "900"))
PORT_RELEASE_TIMEOUT = int(env("PORT_RELEASE_TIMEOUT", "60"))
SERVER_EXTRA_ARGS = shlex.split(env("SERVER_EXTRA_ARGS", ""))
BENCH_EXTRA_ARGS = shlex.split(env("BENCH_EXTRA_ARGS", ""))
PYTHON_BIN = env("PYTHON_BIN", sys.executable)
RUN_NOTES = env("RUN_NOTES", "")
FIX_NOTES = env("FIX_NOTES", "")

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
    return ",".join(GPU_IDS[:count])


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


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((HOST, port)) == 0


def wait_for_port_to_close(port: int) -> None:
    deadline = time.monotonic() + PORT_RELEASE_TIMEOUT
    while time.monotonic() < deadline:
        if not is_port_open(port):
            return
        time.sleep(1)
    raise TimeoutError(
        f"Timed out waiting for port {port} on {HOST} to close. "
        "A stale API server may still be running."
    )


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
    wait_for_port_to_close(port)

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
        MODEL,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tokenizer",
        MODEL,
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


def write_run_summary(*, status: str, started_at: str,
                      completed_at: str | None = None,
                      failure_reason: str | None = None) -> None:
    planned_cases = [f"tp{size}" for size in TP_SIZES]
    planned_cases.extend(f"dp{size}_ep" for size in DP_SIZES)
    write_run_readme(
        RESULT_ROOT,
        title="One-node TP vs DP+EP Run",
        script_path="benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py",
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        setup={
            "model": MODEL,
            "served_model_name": SERVED_MODEL_NAME,
            "host": HOST,
            "base_port": str(BASE_PORT),
            "gpu_ids": ",".join(GPU_IDS),
            "gpu_count": str(GPU_COUNT),
            "tp_sizes": " ".join(str(size) for size in TP_SIZES),
            "dp_sizes": " ".join(str(size) for size in DP_SIZES),
            "num_prompts": NUM_PROMPTS,
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "request_rate": REQUEST_RATE,
            "max_concurrency": MAX_CONCURRENCY or "unset",
            "max_model_len": MAX_MODEL_LEN,
            "all2all_backend": ALL2ALL_BACKEND,
            "server_start_timeout": str(SERVER_START_TIMEOUT),
            "port_release_timeout": str(PORT_RELEASE_TIMEOUT),
            "server_extra_args": " ".join(SERVER_EXTRA_ARGS) or "(none)",
            "bench_extra_args": " ".join(BENCH_EXTRA_ARGS) or "(none)",
            "python_bin": PYTHON_BIN,
            "smoke_run": str(RESULT_ROOT.name.endswith("_smoke")).lower(),
        },
        planned_cases=planned_cases,
        artifact_paths={
            "server logs": "server_logs/",
            "bench logs": "bench_logs/",
            "json results": "json/",
            "summary": "summary.csv",
        },
        failure_reason=failure_reason,
        run_notes=RUN_NOTES,
        fix_notes=FIX_NOTES,
    )


def main() -> int:
    require_command("vllm")
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    write_run_summary(status="running", started_at=started_at)

    signal.signal(signal.SIGTERM, handle_signal)
    try:
        print(
            "Detected benchmark matrix: "
            f"GPU_IDS={GPU_IDS}, TP_SIZES={TP_SIZES}, DP_SIZES={DP_SIZES}",
            flush=True,
        )
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
        write_run_summary(
            status="completed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        print(f"Results: {RESULT_ROOT}")
        return 0
    except Exception:
        write_run_summary(
            status="failed",
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
            failure_reason=traceback.format_exc(),
        )
        raise
    finally:
        cleanup_server()


if __name__ == "__main__":
    raise SystemExit(main())
