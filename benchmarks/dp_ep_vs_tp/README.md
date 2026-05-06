# DP+EP vs TP Online Benchmarks

This directory contains online serving benchmark harnesses for comparing tensor
parallelism (TP) against data parallelism plus expert parallelism (DP+EP). The
main experiments are:

1. Single-node Qwen MoE experiments.
2. Two-node Ray TP1x2 vs DP+EP experiments.

Use these EP comparisons for MoE models. For dense models,
`--enable-expert-parallel` does not create useful expert sharding.

## Online vs Offline Benchmarks

`vllm bench throughput` is an offline benchmark. It constructs an `LLM` in the
benchmark process and measures batch inference throughput without the OpenAI API
server, frontend request routing, DP load balancing, or HTTP overhead.

Use `vllm bench serve` for these experiments. It sends requests to a running
`vllm serve` endpoint and reports request throughput, output-token throughput,
total-token throughput, TTFT, TPOT, and E2E latency.

## Setup

Run commands from the repo root after setting up the vLLM environment. Use the
repo virtual environment on `PATH` so the benchmark subprocesses use the same
installation:

```bash
PATH="$(pwd)/.venv/bin:$PATH"
```

Figure generation uses matplotlib and writes PNG files. Install the benchmark
extra before running plot-producing scripts:

```bash
uv pip install -e ".[bench]"
```

## Experiment 1: Single-Node Qwen Experiments

The single-node Qwen experiments are split into three cases:

1. TP vs DP+EP comparison.
2. DP+EP performance comparison.
3. DP+EP performance comparison with module profiling.

### Shared Single-Node Settings

Use the same serving and client settings across the three cases:

- Node shape: 8 GPUs.
- Models: `Qwen/Qwen1.5-MoE-A2.7B` and `Qwen/Qwen3-30B-A3B` for TP vs DP+EP;
  `Qwen/Qwen3-30B-A3B` for the DP+EP pipeline.
- Server dtype: `--dtype bfloat16`.
- Input length: `1`.
- Output length: `256`.
- Request rate: `inf`.
- Benchmark args: `--ignore-eos`.
- Prefix caching: disabled with `--no-enable-prefix-caching`.
- All2all backend: `allgather_reducescatter`.
- DP+EP sweep: `1..8`.
- TP sweep: `1 2 4 8` for the TP baselines.
- Request count: `1000 * GPUs used`.
- Max client concurrency: larger than the request count for every case. For an
  8-GPU node, use `MAX_CONCURRENCY_PER_GPU=1001` or `MAX_CONCURRENCY=10000`.

With the default `PROMPTS_PER_GPU=1000`, `dp1_ep` uses 1000 requests and
`dp8_ep` uses 8000 requests. TP baselines follow the same rule: `tp1` uses
1000 requests and `tp8` uses 8000 requests.

If `GPU_COUNT` is unset, the scripts first honor `CUDA_VISIBLE_DEVICES`, then
fall back to `nvidia-smi -L`. If both `CUDA_VISIBLE_DEVICES` and `GPU_COUNT`
are set, the scripts use the first `GPU_COUNT` entries from
`CUDA_VISIBLE_DEVICES`.

### Case 1: TP vs DP+EP

This case compares TP baselines against DP+EP on the same single node. The Qwen
suite runs Qwen1.5 MoE and Qwen3 MoE as separate run directories, then
generates combined analysis artifacts:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen_one_node_suite.py \
  --include-tp \
  --gpu-count 8 \
  --tp-sizes "1 2 4 8" \
  --dp-sizes "1 2 3 4 5 6 7 8" \
  --input-len 1 \
  --output-len 256 \
  --max-concurrency-per-gpu 1001 \
  --all2all-backend allgather_reducescatter
```

To run only one model preset:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen_one_node_suite.py \
  --models qwen3 \
  --include-tp \
  --gpu-count 8 \
  --tp-sizes "1 2 4 8" \
  --dp-sizes "1 2 3 4 5 6 7 8" \
  --input-len 1 \
  --output-len 256 \
  --max-concurrency-per-gpu 1001 \
  --all2all-backend allgather_reducescatter
```

The suite writes one run directory per model under:

```text
results/dp_ep_vs_tp/one_node_online/<timestamp>/
```

The combined analysis artifacts are written under:

```text
results/dp_ep_vs_tp/analysis/<timestamp>/
```

Each model run contains:

- `server_logs/<case>.log`
- `bench_logs/<case>.log`
- `json/<case>.json`
- `summary.csv`
- `README.md`

### Case 2: DP+EP Performance

This case runs the clean Qwen3 DP+EP throughput sweep without torch profiling.
It uses `DP+EP=1..8`, `INPUT_LEN=1`, `OUTPUT_LEN=256`, `--ignore-eos`, disabled
prefix caching, and per-case request counts of `DP_SIZE * 1000`.

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
RUN_PROFILE=0 \
MODEL=Qwen/Qwen3-30B-A3B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
DP_SIZES="1 2 3 4 5 6 7 8" \
PROMPTS_PER_GPU=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
ALL2ALL_BACKEND=allgather_reducescatter \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py
```

The clean run writes:

- `throughput/summary.csv`
- `throughput/json/<case>.json`
- `throughput/server_logs/<case>.log`
- `throughput/bench_logs/<case>.log`
- `qwen3_moe_dp_ep_throughput.png`

### Case 3: DP+EP Performance With Profiling

This case repeats the DP+EP sweep with module profiling enabled. Use it to
inspect attention, FusedMoE, and MoE communication timing. Do not mix these
numbers with clean throughput comparisons, because torch profiling and custom
scopes add overhead.

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
RUN_THROUGHPUT=0 \
RUN_PROFILE=1 \
MODEL=Qwen/Qwen3-30B-A3B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
DP_SIZES="1 2 3 4 5 6 7 8" \
PROMPTS_PER_GPU=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
ALL2ALL_BACKEND=allgather_reducescatter \
PROFILE_LAYER_SCOPES=1 \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py
```

The profiled run writes:

- `profile/summary.csv`
- `profile/module_summary.csv`
- `profile/per_layer_module_summary.csv`
- `profile/moe_comm_summary.csv`
- `profile/profiler_traces/<case>/`
- `qwen3_moe_module_latency_stacked.png`

Every run directory includes a `README.md` with the experiment setup, planned
cases, artifact locations, run notes, fix notes, and failure details if the run
aborts.

## Experiment 2: Two-Node Ray TP1x2 vs DP+EP

The two-node script assumes a Ray cluster is already running across two nodes
with one GPU per node, and that the command is launched from the Ray head node.
It compares:

- `tp1x2`: one Ray-DP vLLM deployment with `DP=2`, `TP=1`, and EP disabled.
- `dp2_ep`: the same Ray-DP shape with `--enable-expert-parallel`.

Run:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
VLLM_RAY_DP_PACK_STRATEGY=strict \
MODEL=allenai/OLMoE-1B-7B-0924-Instruct \
SERVER_EXTRA_ARGS="--dtype float16" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_two_node_ray_tp1_vs_dp_ep.py
```

The script writes results under:

```text
results/dp_ep_vs_tp/two_node_ray/<timestamp>/
```

Each two-node run contains:

- `server_logs/<case>.log`
- `bench_logs/<case>.log`
- `json/<case>.json`
- `summary.csv`
- `README.md`

For MoE models, vLLM's default behavior shards MoE expert layers as tensor
parallel over `DP x TP`; this is not the same as two fully independent HTTP
servers behind an external load balancer.

`VLLM_RAY_DP_PACK_STRATEGY` defaults to `strict` for this script. On the
current DP placement logic, a two-node one-GPU-per-node cluster cannot use
`span`; use `strict` for this topology.

## Module Timing Profile Details

The profiled single-node case starts `vllm serve` with the torch profiler, asks
`vllm bench serve` to call `/start_profile` and `/stop_profile`, and enables
custom profiler scopes inside the attention and FusedMoE module boundaries.
Each profiled case writes profiler artifacts under:

```text
<run-root>/profile/profiler_traces/<case>/
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

## Common Knobs

Configure runs with environment variables:

```bash
MODEL=deepseek-ai/DeepSeek-V2-Lite
GPU_COUNT=8
TP_SIZES="1 2 4 8"
DP_SIZES="1 2 3 4 5 6 7 8"
NUM_PROMPTS=
PROMPTS_PER_GPU=1000
INPUT_LEN=1
OUTPUT_LEN=256
REQUEST_RATE=inf
MAX_CONCURRENCY=
MAX_CONCURRENCY_PER_GPU=
MAX_MODEL_LEN=
ALL2ALL_BACKEND=allgather_reducescatter
BASE_PORT=8100
HOST=127.0.0.1
SERVER_EXTRA_ARGS="--trust-remote-code --dtype bfloat16"
BENCH_EXTRA_ARGS=
PYTHON_BIN=.venv/bin/python
VLLM_RAY_DP_PACK_STRATEGY=span
SMOKE_RUN=0
RUN_NOTES=
FIX_NOTES=
PROFILE_MODULES=0
RUN_THROUGHPUT=1
RUN_PROFILE=1
PROFILE_DELAY_ITERATIONS=5
PROFILE_MAX_ITERATIONS=20
PROFILE_WITH_STACK=0
PROFILE_LAYER_SCOPES=0
DISABLE_PREFIX_CACHING=1
```

By default, every case uses `INPUT_LEN=1`, `OUTPUT_LEN=256`, appends
`--ignore-eos` to `vllm bench serve`, and starts `vllm serve` with
`--no-enable-prefix-caching`. Unless `NUM_PROMPTS` is set explicitly, each case
uses `PROMPTS_PER_GPU * GPUs used`; the default `PROMPTS_PER_GPU` is `1000`.

Leave `MAX_MODEL_LEN` empty to use vLLM's model-derived default context length.
Set it only when you intentionally want to pass `--max-model-len`.

Set `SMOKE_RUN=1` to suffix the run folder name with `_smoke`.

Use `RUN_NOTES` to record experiment-specific context in the per-run
`README.md`, and `FIX_NOTES` to record what changed when rerunning after a
failure.

Use `REQUEST_RATE=inf` for saturation-style throughput. Use finite request
rates plus `MAX_CONCURRENCY` to study latency/throughput tradeoffs.
`MAX_CONCURRENCY` is an absolute client-side cap for the whole case. Set
`MAX_CONCURRENCY_PER_GPU` to scale that cap by the number of GPUs/DP ranks in
each case; for example, `MAX_CONCURRENCY_PER_GPU=1001` uses `6006` for DP6 and
`7007` for DP7 with the default `PROMPTS_PER_GPU=1000`. If both are set,
`MAX_CONCURRENCY` takes precedence. When a concurrency cap is configured, the
harness requires the computed cap to be greater than that case's `num_prompts`
so the benchmark is not accidentally throttled below the prompt count.

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
