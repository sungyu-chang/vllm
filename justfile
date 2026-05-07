dp-ep-vs-tp-bench model="deepseek-ai/DeepSeek-V2-Lite" server_extra_args="--trust-remote-code --dtype bfloat16":
    MODEL='{{model}}' SERVER_EXTRA_ARGS='{{server_extra_args}}' .venv/bin/python benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py

dp-ep-vs-tp-profile-modules model="deepseek-ai/DeepSeek-V2-Lite" server_extra_args="--trust-remote-code --dtype bfloat16":
    PROFILE_MODULES=1 MODEL='{{model}}' SERVER_EXTRA_ARGS='{{server_extra_args}}' .venv/bin/python benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py

qwen3-moe-ep-pipeline model="Qwen/Qwen3-30B-A3B" server_extra_args="--dtype bfloat16":
    MODEL='{{model}}' SERVER_EXTRA_ARGS='{{server_extra_args}}' .venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py

qwen15-moe-case3-profile n_gpu:
    PATH="$(pwd)/.venv/bin:$PATH" RUN_THROUGHPUT=0 RUN_PROFILE=1 MODEL="Qwen/Qwen1.5-MoE-A2.7B" SERVER_EXTRA_ARGS="--dtype bfloat16" GPU_COUNT={{n_gpu}} PROMPTS_PER_GPU=1000 INPUT_LEN=1 OUTPUT_LEN=256 NUM_WARMUPS=100 REQUEST_RATE=inf MAX_CONCURRENCY_PER_GPU=1001 ALL2ALL_BACKEND=allgather_reducescatter PROFILE_LAYER_SCOPES=1 .venv/bin/python benchmarks/dp_ep_vs_tp/run_qwen3_moe_ep_pipeline.py

vllm-profiler-playground model="Qwen/Qwen2.5-0.5B-Instruct" server_extra_args="--dtype bfloat16":
    MODEL='{{model}}' SERVER_EXTRA_ARGS='{{server_extra_args}}' .venv/bin/python benchmarks/profiler_playground/run_simple_profile.py

two-node-ray-dp-ep-bench model="allenai/OLMoE-1B-7B-0924-Instruct" server_extra_args="--dtype float16":
    MODEL='{{model}}' SERVER_EXTRA_ARGS='{{server_extra_args}}' .venv/bin/python benchmarks/dp_ep_vs_tp/run_two_node_ray_tp1_vs_dp_ep.py
