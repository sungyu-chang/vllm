# One-node TP vs DP+EP Online Benchmark

This folder contains a small harness for comparing online serving throughput on
one 8-GPU node:

- TP baselines: `TP=2`, `TP=4`, `TP=8`
- DP+EP cases: `DP=1..8`, `TP=1`, `--enable-expert-parallel`

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
MODEL=deepseek-ai/DeepSeek-V2-Lite \
SERVER_EXTRA_ARGS="--trust-remote-code --dtype bfloat16" \
benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.sh
```

The script creates a timestamped result directory under:

```text
benchmarks/dp_ep_vs_tp/results/
```

Each case gets:

- `server_logs/<case>.log`
- `bench_logs/<case>.log`
- `json/<case>.json`
- `summary.csv`

## Common Knobs

Configure by environment variables:

```bash
MODEL=deepseek-ai/DeepSeek-V2-Lite
TP_SIZES="2 4 8"
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
```

Use `REQUEST_RATE=inf` for saturation-style throughput. Use finite request
rates plus `MAX_CONCURRENCY` to study latency/throughput tradeoffs.

## Notes For Fair Comparisons

- DP+EP with `DP=N` uses the first `N` visible GPUs; TP uses the first `TP`
  visible GPUs. On an 8-GPU node, `TP=8` and `DP=8 + EP` are the full-node
  comparisons.
- `DP=1 + EP` is effectively a single-rank MoE run; it is included for scaling
  shape, not as a full-node baseline.
- Single-node internal DP defaults `api_server_count` to `data_parallel_size`,
  so DP+EP cases use multiple frontend API processes by default.
- Keep model, prompt/output lengths, dtype, quantization, and all server args
  fixed across cases.
- If you use DeepEP or another all2all backend, set `ALL2ALL_BACKEND` and ensure
  its dependencies are installed before running.
