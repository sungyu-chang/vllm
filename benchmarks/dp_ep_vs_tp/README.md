# One-node TP vs DP+EP Online Benchmark

This folder contains a small harness for comparing online serving throughput on
one GPU node:

- TP baselines: powers of two up to the detected GPU count
- DP+EP cases: `DP=1..GPU_COUNT`, `TP=1`, `--enable-expert-parallel`

For example, a 4-GPU node runs `TP=1,2,4` and `DP+EP=1..4`. An
8-GPU node runs `TP=1,2,4,8` and `DP+EP=1..8`.

Use this only for MoE models when interpreting EP results. For dense models,
`--enable-expert-parallel` does not create useful expert sharding.

## Online vs Offline Benchmarks

`vllm bench throughput` is an offline benchmark. It constructs an `LLM` in the
benchmark process and measures batch inference throughput without the OpenAI API
server, frontend request routing, DP load balancing, or HTTP overhead.

Use `vllm bench serve` for online throughput. It sends requests to a running
`vllm serve` endpoint and reports request throughput, output-token throughput,
total-token throughput, TTFT, TPOT, and E2E latency.

## Quick Start

From the repo root, after setting up the vLLM environment:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
MODEL=deepseek-ai/DeepSeek-V2-Lite \
SERVER_EXTRA_ARGS="--trust-remote-code --dtype bfloat16" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py
```

For the one-node TP vs DP+EP online benchmark requested in this repo on an
8-GPU node with `input_len=128`, `output_len=256`, and `num_prompts=1000`:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
MODEL=Qwen/Qwen1.5-MoE-A2.7B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
TP_SIZES="1 2 4 8" \
DP_SIZES="1 2 4 8" \
INPUT_LEN=128 \
OUTPUT_LEN=256 \
NUM_PROMPTS=1000 \
REQUEST_RATE=inf \
RUN_NOTES="Single-node TP vs DP+EP comparison for Qwen1.5-MoE-A2.7B, input 128, output 256" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py
```

For the expanded one-node sweep requested for Qwen MoE models with
`TP_SIZES="1 2 4 8"`, `DP_SIZES="1 2 3 4 5 6 7 8"`, `input_len=128`,
`output_len=256`, and `num_prompts=5000`, use the suite wrapper. It runs the
Qwen1.5 MoE and Qwen3 MoE 30B experiments as separate run directories and then
generates a combined CSV, Markdown summary, and SVG figure without manual
polling:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen_one_node_suite.py
```

Artifacts from the analysis step are written under:

```text
results/dp_ep_vs_tp/analysis/<timestamp>/
```

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
MODEL=Qwen/Qwen3-30B-A3B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
TP_SIZES="1 2 4 8" \
DP_SIZES="1 2 4 8" \
INPUT_LEN=128 \
OUTPUT_LEN=256 \
NUM_PROMPTS=1000 \
REQUEST_RATE=inf \
RUN_NOTES="Single-node TP vs DP+EP comparison for Qwen3-30B-A3B, input 128, output 256" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py
```

For a two-node Ray cluster with one GPU per node, compare `TP=1 x 2` against
`DP=2 + EP` from the Ray head node:

```bash
VLLM_RAY_DP_PACK_STRATEGY=strict \
MODEL=allenai/OLMoE-1B-7B-0924-Instruct \
SERVER_EXTRA_ARGS="--dtype float16" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_two_node_ray_tp1_vs_dp_ep.py
```

The `tp1x2` case is one Ray-DP vLLM deployment with `DP=2`, `TP=1`, and EP
disabled. For MoE models, vLLM's default behavior shards MoE expert layers as
tensor parallel over `DP x TP`; this is not the same as two fully independent
HTTP servers behind an external load balancer.

The script creates a timestamped result directory under:

```text
results/dp_ep_vs_tp/one_node_online/
results/dp_ep_vs_tp/two_node_ray/
```

Each case gets:

- `server_logs/<case>.log`
- `bench_logs/<case>.log`
- `json/<case>.json`
- `summary.csv`
- `README.md`

Every run directory now includes a `README.md` with the experiment setup,
planned cases, artifact locations, run notes, fix notes, and failure details if
the run aborts.

## Module Timing Profile

The normal benchmark run reports online serving throughput and latency only. To
collect averaged server-side module timings for attention and FusedMoE, enable
the profiling pass:

```bash
PROFILE_MODULES=1 \
MODEL=deepseek-ai/DeepSeek-V2-Lite \
SERVER_EXTRA_ARGS="--trust-remote-code --dtype bfloat16" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py
```

or use:

```bash
just dp-ep-vs-tp-profile-modules
```

This starts `vllm serve` with the torch profiler, asks `vllm bench serve` to
call `/start_profile` and `/stop_profile`, and enables custom profiler scopes
inside the attention and FusedMoE module boundaries. Each case writes profiler
artifacts under:

```text
benchmarks/dp_ep_vs_tp/results/<run>/profiler_traces/<case>/
```

The most directly useful file is `module_profiler_out_<rank>.txt`, which
contains CSV-like rows:

```text
module,count,total_cuda_ms,avg_cuda_ms,total_cpu_ms,avg_cpu_ms
vllm:attention,...
vllm:fused_moe,...
```

The script also writes `module_summary.csv` at the run root. It includes one
row per case/rank/module plus `rank=all` aggregate rows for quick comparison
across EP sizes.

When `PROFILE_LAYER_SCOPES=1`, the profiler also emits nested rows such as
`vllm:attention:<layer_name>` and `vllm:fused_moe:<layer_name>`. The aggregate
`vllm:attention` and `vllm:fused_moe` rows are still present for stacked-bar
plots.

For the default `allgather_reducescatter` all2all backend, the profiler also
records `vllm:moe_comm` plus dispatch/combine detail rows. Other specialized
MoE all2all backends may need backend-specific scopes before their internal
communication can be separated cleanly from the fused MoE kernel.

Use these profiling runs to understand the relative module breakdown. Do not
mix them with clean throughput comparisons: torch profiling and custom scopes
add overhead. By default, the script skips the first 5 engine iterations and
profiles 20 iterations. Override with `PROFILE_DELAY_ITERATIONS` and
`PROFILE_MAX_ITERATIONS`.

## Qwen3-MoE DP+EP Pipeline

Run the all-in-one single-node Qwen3-MoE DP+EP pipeline with:

```bash
just qwen3-moe-ep-pipeline
```

By default, it uses `Qwen/Qwen3-30B-A3B`, runs `DP+EP=1..GPU_COUNT`, uses
`INPUT_LEN=1`, `OUTPUT_LEN=256`, `--ignore-eos`, and sets `NUM_PROMPTS` to
`GPU_COUNT * 1000`. It runs two passes:

- `throughput/`: clean online throughput run without torch profiling
- `profile/`: same benchmark shape with module profiling enabled

Pipeline outputs at the run root:

- `qwen3_moe_dp_ep_throughput.svg`: number of GPUs vs total token throughput
- `qwen3_moe_module_latency_stacked.svg`: stacked attention/FusedMoE average
  CUDA latency
- `profile/per_layer_module_summary.csv`: per-layer attention and FusedMoE
  latency rows when layer scopes are enabled
- `profile/moe_comm_summary.csv`: MoE communication rows for the profiled run

`DISABLE_PREFIX_CACHING=1` is enabled by default for this pipeline, implemented
as `--no-enable-prefix-caching`. This disables prefix-cache reuse, not the KV
cache itself. vLLM serving still uses KV cache for autoregressive decode; there
is no normal serving flag that disables KV cache while preserving efficient
generation.

## Common Knobs

Configure by environment variables:

```bash
MODEL=deepseek-ai/DeepSeek-V2-Lite
GPU_COUNT=8
TP_SIZES="1 2 4 8"
DP_SIZES="1 2 3 4 5 6 7 8"
NUM_PROMPTS=1000
INPUT_LEN=1024
OUTPUT_LEN=128
REQUEST_RATE=inf
MAX_CONCURRENCY=
MAX_MODEL_LEN=4096
ALL2ALL_BACKEND=allgather_reducescatter
BASE_PORT=8100
HOST=127.0.0.1
SERVER_EXTRA_ARGS="--trust-remote-code --dtype bfloat16"
BENCH_EXTRA_ARGS="--ignore-eos"
PYTHON_BIN=.venv/bin/python
VLLM_RAY_DP_PACK_STRATEGY=span
SMOKE_RUN=0
RUN_NOTES=
FIX_NOTES=
PROFILE_MODULES=0
PROFILE_DELAY_ITERATIONS=5
PROFILE_MAX_ITERATIONS=20
PROFILE_WITH_STACK=0
PROFILE_LAYER_SCOPES=0
DISABLE_PREFIX_CACHING=0
```

`GPU_COUNT` is optional. If it is unset, the script first honors
`CUDA_VISIBLE_DEVICES`, then falls back to `nvidia-smi -L`. If both
`CUDA_VISIBLE_DEVICES` and `GPU_COUNT` are set, the script uses the first
`GPU_COUNT` entries from `CUDA_VISIBLE_DEVICES`. Explicit `TP_SIZES` and
`DP_SIZES` override the detected defaults.

For the two-node Ray script, `VLLM_RAY_DP_PACK_STRATEGY` defaults to `strict`.
On the current DP placement logic, a two-node one-GPU-per-node cluster cannot
use `span`; use `strict` for this topology.

Set `SMOKE_RUN=1` to suffix the run folder name with `_smoke`.

Use `RUN_NOTES` to record experiment-specific context in the per-run
`README.md`, and `FIX_NOTES` to record what changed when rerunning after a
failure.

Use `REQUEST_RATE=inf` for saturation-style throughput. Use finite request
rates plus `MAX_CONCURRENCY` to study latency/throughput tradeoffs.

## Notes For Fair Comparisons

- DP+EP with `DP=N` uses the first `N` visible GPUs; TP uses the first `TP`
  visible GPUs. The largest TP and DP+EP cases are the full-node comparisons.
- `DP=1 + EP` is effectively a single-rank MoE run; it is included for scaling
  shape, not as a full-node baseline.
- Single-node internal DP defaults `api_server_count` to `data_parallel_size`,
  so DP+EP cases use multiple frontend API processes by default.
- Keep model, prompt/output lengths, dtype, quantization, and all server args
  fixed across cases.
- If you use DeepEP or another all2all backend, set `ALL2ALL_BACKEND` and ensure
  its dependencies are installed before running.
