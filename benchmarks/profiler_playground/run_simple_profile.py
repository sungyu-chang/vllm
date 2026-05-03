#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a minimal vLLM serving profile for profiler exploration."""

from __future__ import annotations

import sys
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parents[1] / "dp_ep_vs_tp"
sys.path.insert(0, str(COMMON_DIR))

from single_node_common import (  # noqa: E402
    SingleNodeBenchmarkConfig,
    SingleNodeBenchmarkRunner,
    default_result_root,
    detect_gpu_ids,
    env,
    env_bool,
    shlex_env,
)


def main() -> int:
    gpu_ids = detect_gpu_ids()
    profile_gpu_count = int(env("PROFILE_GPU_COUNT", "1"))
    if profile_gpu_count < 1 or profile_gpu_count > len(gpu_ids):
        raise SystemExit(
            "PROFILE_GPU_COUNT must be between 1 and the number of visible GPUs "
            f"({len(gpu_ids)})"
        )

    server_args: list[str] = []
    if profile_gpu_count > 1:
        server_args.extend(["--tensor-parallel-size", str(profile_gpu_count)])

    config = SingleNodeBenchmarkConfig(
        model=env("MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        served_model_name=env("SERVED_MODEL_NAME", "profiler-playground-model"),
        host=env("HOST", "127.0.0.1"),
        base_port=int(env("BASE_PORT", "8200")),
        gpu_ids=gpu_ids,
        num_prompts=env("NUM_PROMPTS", "16"),
        input_len=env("INPUT_LEN", "32"),
        output_len=env("OUTPUT_LEN", "32"),
        request_rate=env("REQUEST_RATE", "inf"),
        max_concurrency=env("MAX_CONCURRENCY", ""),
        max_model_len=env("MAX_MODEL_LEN", "2048"),
        result_root=Path(
            env("RESULT_ROOT", str(default_result_root("profiler_playground")))
        ),
        server_start_timeout=int(env("SERVER_START_TIMEOUT", "600")),
        server_extra_args=shlex_env("SERVER_EXTRA_ARGS", "--dtype bfloat16"),
        bench_extra_args=shlex_env("BENCH_EXTRA_ARGS", "--ignore-eos"),
        python_bin=env("PYTHON_BIN", ".venv/bin/python"),
        profile_modules=True,
        profile_delay_iterations=int(env("PROFILE_DELAY_ITERATIONS", "2")),
        profile_max_iterations=int(env("PROFILE_MAX_ITERATIONS", "8")),
        profile_with_stack=env_bool("PROFILE_WITH_STACK"),
        profile_layer_scopes=env_bool("PROFILE_LAYER_SCOPES", True),
        disable_prefix_caching=env_bool("DISABLE_PREFIX_CACHING"),
    )

    runner = SingleNodeBenchmarkRunner(config)
    runner.require_command("vllm")
    runner.setup_dirs()
    runner.install_signal_handlers()

    try:
        runner.run_case(
            case_name="simple_profile",
            gpu_count=profile_gpu_count,
            port=config.base_port,
            server_args=server_args,
            metadata={
                "purpose": "profiler_playground",
                "profile_gpu_count": profile_gpu_count,
            },
        )
        runner.summarize_results()
        runner.summarize_module_profiles()
        print(f"Results: {config.result_root}")
        return 0
    finally:
        runner.cleanup_server()


if __name__ == "__main__":
    raise SystemExit(main())
