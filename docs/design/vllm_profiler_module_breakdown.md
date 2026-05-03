# vLLM Profiler Module Breakdown Notes

This document records the local source changes that add module-level latency
breakdown on top of vLLM's existing profiler path.

## What vLLM Already Provided

vLLM already supports server-side profiling through `ProfilerConfig`.

- The API server exposes `/start_profile` and `/stop_profile` only when
  `--profiler-config` enables a profiler.
- `vllm bench serve --profile` calls those endpoints before and after the
  benchmark traffic.
- GPU workers construct `TorchProfilerWrapper` when profiling starts.
- `TorchProfilerWrapper` uses `torch.profiler.profile` and writes TensorBoard
  traces plus rank-local text tables.
- Worker execution already has coarse scopes such as
  `gpu_model_runner: forward` when custom profiling scopes are enabled.

## Source Changes Added Here

The added behavior is opt-in and is only active when
`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` is set in the server environment.

### `vllm/profiler/scopes.py`

Added a small helper layer around `torch.autograd.profiler.record_function`.

- `module_profile_scope(name, detail=None)` emits a profiler scope when custom
  scopes are enabled.
- `VLLM_PROFILE_LAYER_SCOPES=1` adds nested detail scopes, for example
  `vllm:attention:model.layers.3.self_attn.attn`.
- `moe_profile_scope()` records the current MoE layer in a `ContextVar`.
- `moe_comm_detail()` lets communication scopes include the current MoE layer.

### `vllm/model_executor/layers/attention/attention.py`

Wrapped the attention custom op launch path in:

- `vllm:attention`
- `vllm:attention:<layer_name>` when layer scopes are enabled

This scope covers KV-cache update and attention kernel launch for both direct
and torch custom-op call paths.

### `vllm/model_executor/layers/fused_moe/layer.py`

Wrapped `FusedMoE.forward()` in:

- `vllm:fused_moe`
- `vllm:fused_moe:<layer_name>` when layer scopes are enabled

This scope covers the FusedMoE runner, including routing, dispatch/combine,
expert computation, shared experts, and output reduction that happen under the
FusedMoE call.

### `vllm/distributed/device_communicators/all2all.py`

For the default `allgather_reducescatter` all2all backend, wrapped the DP/EP
communication collectives in:

- `vllm:moe_comm`
- `vllm:moe_comm:<layer_name>:dispatch`
- `vllm:moe_comm:<layer_name>:dispatch_router_logits`
- `vllm:moe_comm:<layer_name>:combine`

This separates the all-gather dispatch and reduce-scatter combine costs from
the broader FusedMoE scope. Specialized backends such as DeepEP or FlashInfer
may still need backend-specific scopes to expose their internal communication
phases cleanly.

### `vllm/profiler/wrapper.py`

Extended `TorchProfilerWrapper._stop()` to write a filtered module summary.

Each profiled rank now writes:

```text
module_profiler_out_<rank>.txt
```

The file contains CSV-like rows:

```text
module,count,total_cuda_ms,avg_cuda_ms,total_cpu_ms,avg_cpu_ms
```

Only profiler keys beginning with `vllm:` are included. The original torch
profiler table is still written to `profiler_out_<rank>.txt`.

## Benchmark Scripts Added Around Profiling

### `benchmarks/dp_ep_vs_tp`

The single-node benchmark code was consolidated into
`benchmarks/dp_ep_vs_tp/single_node_common.py`.

The existing TP vs DP+EP benchmark now reuses this shared runner, and the
Qwen3-MoE pipeline uses the same runner for clean throughput and profiled runs.

### `benchmarks/profiler_playground`

This folder contains a minimal profiler playground for manually inspecting what
vLLM collects. It starts one `vllm serve`, runs a small `vllm bench serve`
profile pass, and writes:

- `profiler_traces/simple_profile/profiler_out_<rank>.txt`
- `profiler_traces/simple_profile/module_profiler_out_<rank>.txt`
- TensorBoard trace files
- `module_summary.csv`
- normal benchmark JSON and logs

## What You Can Learn From the Outputs

Use `profiler_out_<rank>.txt` for the full torch profiler table. It shows
operator and kernel-level timing sorted by CUDA time.

Use `module_profiler_out_<rank>.txt` or `module_summary.csv` for high-level
latency attribution:

- `vllm:attention`: average attention scope time per call.
- `vllm:fused_moe`: average FusedMoE scope time per call.
- `vllm:attention:<layer_name>`: per-layer attention time.
- `vllm:fused_moe:<layer_name>`: per-layer MoE time.
- `vllm:moe_comm...`: DP/EP communication time for AgRs all2all.

Use TensorBoard/Chrome trace viewers when you need timeline structure,
overlap, or kernel launch ordering rather than aggregate averages.

## Measurement Caveats

Profiling perturbs performance. Use profiled runs to understand relative module
breakdown and clean runs to report throughput.

The `avg_cuda_ms` value is averaged over profiler events, not requests. For
layer scopes, a call usually corresponds to one layer invocation in one worker
step.

Nested scopes are useful for readability, but parent and child times are not
additive. For example, `vllm:fused_moe` includes communication, expert compute,
and nested detail scopes.

For online serving, profiler windows are bounded by engine iterations, not a
fixed number of HTTP requests. Use `delay_iterations`, `max_iterations`,
`warmup_iterations`, and `active_iterations` to control the capture window.
