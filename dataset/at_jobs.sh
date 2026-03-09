#!/bin/bash
cd /mydata/songyu/vllm
PY=.venv/bin/python
DS=dataset/combined_dataset_1000.jsonl


$PY dataset/02_run_inference.py --model deepseek-ai/deepseek-moe-16b-chat --tensor_parallel_size 4 --output dataset/expert_log_deepseek-moe.jsonl --dataset $DS > /tmp/run_deepseek_moe.log 2>&1
$PY dataset/02_run_inference.py --model Qwen/Qwen3-30B-A3B --tensor_parallel_size 4 --output dataset/expert_log_qwen3.jsonl --dataset $DS > /tmp/run_qwen3.log 2>&1
$PY dataset/02_run_inference.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --tensor_parallel_size 4 --output dataset/expert_log_deepseek-v2.jsonl --dataset $DS > /tmp/run_deepseek_v2.log 2>&1
$PY dataset/02_run_inference.py --model openai/gpt-oss-20b --tensor_parallel_size 4 --output dataset/expert_log_gpt-oss.jsonl --dataset $DS > /tmp/run_gpt.log 2>&1
#$PY dataset/02_run_inference.py --model zai-org/GLM-4.7-Flash --tensor_parallel_size 4 --output dataset/expert_log_glm47flash.jsonl --dataset $DS > /tmp/run_glm.log 2>&1


$PY dataset/02_run_inference.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor_parallel_size 4 --output dataset/expert_log_mixtral.jsonl --dataset $DS > /tmp/run_mixtral.log 2>&1


# ── Package and upload ────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="/tmp/expert_logs_${TIMESTAMP}.tar.gz"

echo "Compressing dataset logs..."
tar -czf "$ARCHIVE" -C /mydata/songyu/vllm dataset/
echo "Archive: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"

echo "Uploading to cloud via rclone..."
rclone copy "$ARCHIVE" gdrive:expert_logs/ --progress
echo "Upload complete: gdrive:expert_logs/$(basename "$ARCHIVE")"
