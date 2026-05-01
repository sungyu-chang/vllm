dp-ep-vs-tp-bench model="deepseek-ai/DeepSeek-V2-Lite" server_extra_args="--trust-remote-code --dtype bfloat16":
    MODEL='{{model}}' SERVER_EXTRA_ARGS='{{server_extra_args}}' .venv/bin/python benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.py

two-node-ray-dp-ep-bench model="allenai/OLMoE-1B-7B-0924-Instruct" server_extra_args="--dtype float16":
    MODEL='{{model}}' SERVER_EXTRA_ARGS='{{server_extra_args}}' .venv/bin/python benchmarks/dp_ep_vs_tp/run_two_node_ray_tp1_vs_dp_ep.py
