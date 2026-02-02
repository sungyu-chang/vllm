prefill_qwen3_4b:
    vllm serve Qwen/Qwen3-4B-Thinking-2507 --max_model_len 88000
prefill_pp:
    vllm serve Qwen/Qwen3-4B-Thinking-2507 --max-model-len 113000 --pipeline-parallel-size 2 --distributed-executor-backend ray
prefill_tp:
prefill_tp:
    vllm serve Qwen/Qwen3-4B-Thinking-2507 --max-model-len 232000 --tensor-parallel-size 2 --distributed-executor-backend ray


    
