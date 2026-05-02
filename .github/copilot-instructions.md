# Project Instructions

## Benchmark Results Layout

When creating or updating benchmark harnesses in this repository:

- Write benchmark outputs under the top-level `results/` directory, not under a benchmark-local folder.
- Use the hierarchy `results/<benchmark-name>/<run-group>/<timestamp>`.
- For smoke runs, suffix the run folder with `_smoke`.
- Keep benchmark-specific grouping stable, for example:
  - `results/dp_ep_vs_tp/one_node_online/...`
  - `results/dp_ep_vs_tp/two_node_ray/...`

## Per-run Metadata

Every benchmark run directory must include a `README.md` that records:

- experiment status: running, completed, or failed
- script used to launch the run
- experiment setup: model, ports, lengths, request rate, relevant env vars, and other run parameters
- planned cases for the run
- artifact locations for logs, json outputs, and summaries
- failure reason when a run fails
- fix notes describing what changed before a successful rerun

Use `RUN_NOTES` for run-specific context and `FIX_NOTES` for rerun/fix context when a harness supports them.

## Git Tracking

- Treat the top-level `results/` directory as generated output.
- Do not add benchmark result artifacts under version control.
- Keep `results/` ignored in git.

## Benchmark Harness Conventions

- When using `vllm bench serve`, pass the real model identifier for `--model` and `--tokenizer`.
- Use `--served-model-name` only for the API alias exposed by `vllm serve`.
- On the current two-node one-GPU-per-node Ray setup, use `VLLM_RAY_DP_PACK_STRATEGY=strict` for the DP+EP benchmark harness.
