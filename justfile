dp-ep-vs-tp-bench model="deepseek-ai/DeepSeek-V2-Lite" server_extra_args="--trust-remote-code --dtype bfloat16":
    MODEL='{{model}}' SERVER_EXTRA_ARGS='{{server_extra_args}}' benchmarks/dp_ep_vs_tp/run_online_tp_vs_dp_ep.sh
