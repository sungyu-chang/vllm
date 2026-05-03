# vLLM Profiler Playground

This folder contains a minimal one-node profiling run for inspecting what the
vLLM profiler collects.

Run from the repo root:

```bash
.venv/bin/python benchmarks/profiler_playground/run_simple_profile.py
```

or:

```bash
just vllm-profiler-playground
```

Defaults are intentionally small:

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct
PROFILE_GPU_COUNT=1
NUM_PROMPTS=16
INPUT_LEN=32
OUTPUT_LEN=32
REQUEST_RATE=inf
SERVER_EXTRA_ARGS="--dtype bfloat16"
PROFILE_LAYER_SCOPES=1
```

The script starts `vllm serve` with `--profiler-config`, runs
`vllm bench serve --profile`, and then shuts the server down.

Outputs are written under:

```text
benchmarks/profiler_playground/results/<timestamp>/
```

Important files:

- `server_logs/simple_profile.log`: server log and rank-0 profiler table
- `bench_logs/simple_profile.log`: `vllm bench serve` output
- `json/simple_profile.json`: online benchmark metrics
- `profiler_traces/simple_profile/profiler_out_<rank>.txt`: full torch profiler
  aggregate table
- `profiler_traces/simple_profile/module_profiler_out_<rank>.txt`: filtered
  `vllm:*` module timing table
- `module_summary.csv`: per-rank plus `rank=all` aggregate module timing rows
- TensorBoard trace files in `profiler_traces/simple_profile/`

Open TensorBoard traces with:

```bash
tensorboard --logdir benchmarks/profiler_playground/results/<timestamp>/profiler_traces
```

Notes:

- Profiling adds overhead; do not use these results as clean throughput
  numbers.
- Dense models will show attention scopes but no FusedMoE scopes.
- Use an MoE model, for example `MODEL=Qwen/Qwen3-30B-A3B`, if you want MoE
  and MoE communication rows.
