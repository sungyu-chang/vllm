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

The single-node Qwen experiments are split into four cases:

1. TP vs DP+EP comparison.
2. DP+EP performance comparison.
3. DP+EP performance comparison with module profiling.
4. Fixed-request DP+EP profiling for the anatomy stacked bar figure.

### Shared Single-Node Settings

Use the same serving and client settings across the cases unless a case
explicitly overrides one of these values:

- Node shape: 8 GPUs.
- Models: `Qwen/Qwen1.5-MoE-A2.7B` and `Qwen/Qwen3-30B-A3B` for TP vs DP+EP;
  `Qwen/Qwen3-30B-A3B` for the DP+EP pipeline.
- Server dtype: `--dtype bfloat16`.
- Input length: `1`.
- Output length: `256`.
- Warmup requests: `100` per `vllm bench serve` run.
- Request rate: `inf`.
- Benchmark args: `--ignore-eos`.
- Prefix caching: disabled with `--no-enable-prefix-caching`.
- All2all backend: `allgather_reducescatter`.
- DP+EP sweep: defaults to `1..GPU_COUNT`.
- TP sweep: `1 2 4 8` for the TP baselines.
- Request count: `1000 * GPUs used`, except Case 4 which fixes every DP+EP
  case to `NUM_PROMPTS=1000`.
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
  --input-len 1 \
  --output-len 256 \
  --num-warmups 100 \
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
  --input-len 1 \
  --output-len 256 \
  --num-warmups 100 \
  --max-concurrency-per-gpu 1001 \
  --all2all-backend allgather_reducescatter
```

The suite writes one run directory per model under:

```text
results/dp_ep_vs_tp/one_node_online/YYYY-MM-DD_HH-MM-SS/
```

The combined analysis artifacts are written under:

```text
results/dp_ep_vs_tp/analysis/YYYY-MM-DD_HH-MM-SS/
```

Each model run contains:

- `server_logs/<case>.log`
- `bench_logs/<case>.log`
- `json/<case>.json`
- `summary.csv`
- `README.md`

### Case 2: DP+EP Performance

This case runs the clean Qwen3 DP+EP throughput sweep without torch profiling.
With `GPU_COUNT=8`, the default DP+EP sweep is `1..8`. It uses `INPUT_LEN=1`,
`OUTPUT_LEN=256`, `--ignore-eos`, disabled prefix caching, and per-case request
counts of `DP_SIZE * 1000`.

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
RUN_PROFILE=0 \
MODEL=Qwen/Qwen3-30B-A3B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
PROMPTS_PER_GPU=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
NUM_WARMUPS=100 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
ALL2ALL_BACKEND=allgather_reducescatter \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py
```

The clean run writes:

```text
results/dp_ep_vs_tp/qwen3_moe_ep_pipeline/YYYY-MM-DD_HH-MM-SS/
```

- `throughput/summary.csv`
- `throughput/json/<case>.json`
- `throughput/server_logs/<case>.log`
- `throughput/bench_logs/<case>.log`

To capture Nsight Systems traces for the same DP+EP execution shape, use the
dedicated nsys harness. It wraps each `vllm serve` case in `nsys profile`, uses
full server-lifetime capture by default, and writes one profile output base per
DP+EP case:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
MODEL=Qwen/Qwen3-30B-A3B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
PROMPTS_PER_GPU=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
NUM_WARMUPS=100 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
ALL2ALL_BACKEND=allgather_reducescatter \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_nsys.py
```

The nsys run writes:

```text
results/dp_ep_vs_tp/nsys_profiling/YYYY-MM-DD_HH-MM-SS/
```

- `nsys/<case>.nsys-rep`
- `nsys_stats/<case>.*_cuda_gpu_kern_sum.csv`
- `nsys_stats/<case>.*_cuda_gpu_trace.csv`
- `summary.csv`
- `json/<case>.json`
- `server_logs/<case>.log`
- `bench_logs/<case>.log`

Set `NSYS_BIN` when `nsys` is not on `PATH`. The script also checks the common
Nsight Systems CLI path `/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys`.
The `qwen15-moe-nsys-profile` just target sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` because both the server and benchmark client reuse the
cached model/tokenizer, and online Hugging Face checks can otherwise stall the
client before traffic begins. It also sets
`NSYS_CAPTURE_RANGE=cudaProfilerApi`, `NSYS_CAPTURE_RANGE_END=stop`,
`NSYS_CAPTURE_TIMEOUT=300`, and `NSYS_CUDA_GRAPH_TRACE=node` so the default
Qwen1.5-MoE nsys run captures only the benchmark range, expands CUDA graph nodes
in the timeline, and aborts the benchmark/profile capture if it runs longer than
five minutes.
Use `NSYS_TRACE` to override the default `cuda,nvtx` trace set, and
`NSYS_EXTRA_ARGS` to override the default extra `nsys profile` flags. By
default, the harness adds
`--sample=none --backtrace=none --resolve-symbols=false` so report import stays
focused on CUDA/NVTX data and does not block on CPU symbol resolution/downloads.
By default,
`NSYS_CAPTURE_RANGE=none`, so `vllm bench serve` does not call `/start_profile`
or `/stop_profile`; this avoids CUDA profiler API shutdown stalls while still
capturing CUDA kernels in the `.nsys-rep`. Set
`NSYS_CAPTURE_RANGE=cudaProfilerApi` to opt into bounded CUDA profiler API
capture for custom runs. Ranged capture starts `vllm serve` with
`--profiler-config '{"profiler":"cuda"}'` and adds `--profile` to
`vllm bench serve`, which makes the benchmark client call the server's
`/start_profile` endpoint before traffic and `/stop_profile` after traffic. If
the benchmark/profile subprocess exceeds `NSYS_CAPTURE_TIMEOUT`, the harness
kills that subprocess and then cleans up the server and nsys process groups.
CUDA kernel summaries require the default `NSYS_TRACE=cuda,nvtx`; CUDA graph
node tracing is controlled with `NSYS_CUDA_GRAPH_TRACE=node`. The nsys harness
runs the profiler and server in separate process groups and defaults to
`NSYS_WAIT=all` so multiprocess DP traces can finish writing all child process
data. On cleanup, it sends SIGINT to discovered server process groups without
interrupting the Nsight Systems process first; if reparented DP workers linger,
it terminates those server groups before sending SIGINT to the nsys root process
group to flush the report. `NSYS_SERVER_EXIT_TIMEOUT` controls the graceful
server-process wait before escalation. `NSYS_FLUSH_DELAY` controls the grace
period between server exit and nsys SIGINT; the default is 90 seconds because
interrupting nsys immediately after vLLM exits can leave an incomplete qdstrm
stream. It waits up to
`NSYS_SERVER_SHUTDOWN_TIMEOUT` seconds for nsys to write the report. The
default is 300 seconds because multiprocess DP traces can take several minutes
to finalize but should not hang indefinitely. It
keeps `--enforce-eager` disabled and fails early if it is included in
`SERVER_EXTRA_ARGS`. By default it also runs `nsys stats` after each case to
export `cuda_gpu_kern_sum` and `cuda_gpu_trace` CSV files. Override the report
list with `NSYS_STATS_REPORTS`.

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
PROMPTS_PER_GPU=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
NUM_WARMUPS=100 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
ALL2ALL_BACKEND=allgather_reducescatter \
PROFILE_LAYER_SCOPES=1 \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py
```

The profiled run writes:

```text
results/dp_ep_vs_tp/qwen3_moe_ep_pipeline/YYYY-MM-DD_HH-MM-SS/
```

- `profile/summary.csv`
- `profile/module_summary.csv`
- `profile/profiler_traces/<case>/`

### Case 4: Fixed-Request DP+EP Anatomy Profiling

This case keeps the same profiling settings as Case 3, but fixes each DP+EP
case to 1000 requests. Use it when comparing the attention, expert computation,
and MoE communication latency anatomy stacked bar figure across DP+EP sizes
without scaling the request count by the number of GPUs.

The only request-shape difference from Case 3 is `NUM_PROMPTS=1000`, which
overrides `PROMPTS_PER_GPU`. Keep `MAX_CONCURRENCY_PER_GPU=1001` so the client
concurrency remains greater than the 1000 total requests for every DP+EP size.

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
RUN_THROUGHPUT=0 \
RUN_PROFILE=1 \
MODEL=Qwen/Qwen3-30B-A3B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=8 \
NUM_PROMPTS=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
NUM_WARMUPS=100 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
ALL2ALL_BACKEND=allgather_reducescatter \
PROFILE_LAYER_SCOPES=1 \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py
```

For Qwen1.5-MoE on a 4-GPU single node:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
RUN_THROUGHPUT=0 \
RUN_PROFILE=1 \
MODEL=Qwen/Qwen1.5-MoE-A2.7B \
SERVER_EXTRA_ARGS="--dtype bfloat16" \
GPU_COUNT=4 \
NUM_PROMPTS=1000 \
INPUT_LEN=1 \
OUTPUT_LEN=256 \
NUM_WARMUPS=100 \
REQUEST_RATE=inf \
MAX_CONCURRENCY_PER_GPU=1001 \
ALL2ALL_BACKEND=allgather_reducescatter \
PROFILE_LAYER_SCOPES=1 \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py
```

The fixed-request profiled run writes benchmark artifacts under the generated
result directory:

- `profile/summary.csv`
- `profile/module_summary.csv`
- `profile/profiler_traces/<case>/`

To render or rerender Qwen3 DP+EP plots from an existing result directory, pass
the result directory as a positional argument to the dedicated plotting script:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmarks/dp_ep_vs_tp/plot_qwen3_moe_ep_pipeline.py \
  results/dp_ep_vs_tp/qwen3_moe_ep_pipeline/YYYY-MM-DD_HH-MM-SS
```

The Qwen plotting script writes or rewrites:

- `qwen3_moe_dp_ep_throughput.png`
- `qwen3_moe_module_latency_stacked.png`
- `profile/per_layer_module_summary.csv`
- `profile/moe_comm_summary.csv`

The module latency figure breaks the aggregate FusedMoE latency into expert
computation latency and MoE communication latency.

To rerender the TP vs DP+EP throughput figure from existing one-node run
directories, use the standalone plotting script:

```bash
PATH="$(pwd)/.venv/bin:$PATH" \
.venv/bin/python benchmarks/dp_ep_vs_tp/plot_total_token_throughput.py \
  results/dp_ep_vs_tp/one_node_online/YYYY-MM-DD_HH-MM-SS \
  --output results/dp_ep_vs_tp/analysis/YYYY-MM-DD_HH-MM-SS/total_token_throughput.png
```

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
NUM_WARMUPS=100 \
SERVER_EXTRA_ARGS="--dtype float16" \
.venv/bin/python benchmarks/dp_ep_vs_tp/run_two_node_ray_tp1_vs_dp_ep.py
```

The script writes results under:

```text
results/dp_ep_vs_tp/two_node_ray/YYYY-MM-DD_HH-MM-SS/
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
NUM_PROMPTS=
PROMPTS_PER_GPU=1000
INPUT_LEN=1
OUTPUT_LEN=256
NUM_WARMUPS=100
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

By default, every case uses `INPUT_LEN=1`, `OUTPUT_LEN=256`,
`NUM_WARMUPS=100`, appends `--ignore-eos` to `vllm bench serve`, and starts
`vllm serve` with
`--no-enable-prefix-caching`. Unless `NUM_PROMPTS` is set explicitly, each case
uses `PROMPTS_PER_GPU * GPUs used`; the default `PROMPTS_PER_GPU` is `1000`.
Leave `DP_SIZES` unset to sweep `1..GPU_COUNT`; set it only when you want a
subset or a custom order.

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
