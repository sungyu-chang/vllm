qwen:
	python dataset/02_run_inference.py \
	--model Qwen/Qwen3-30B-A3B \
	--tensor_parallel_size 4 \
	--output dataset/expert_log_qwen3.jsonl \
	--dataset dataset/combined_dataset.jsonl
glm:
	python dataset/02_run_inference.py \
	--model zai-org/GLM-4.7-Flash \
	--tensor_parallel_size 4 \
	--output dataset/expert_log_glm47flash.jsonl \
	--dataset dataset/combined_dataset.jsonl
deepseek:
	python dataset/02_run_inference.py \
	--model deepseek-ai/deepseek-moe-16b-chat \
	--tensor_parallel_size 4 \
	--output dataset/expert_log_deepseek.jsonl \
	--dataset dataset/combined_dataset.jsonl
mixtral:
	python dataset/02_run_inference.py \
	--model mistralai/Mixtral-8x7B-Instruct-v0.1 \
	--tensor_parallel_size 4 \
	--output dataset/expert_log_mixtral.jsonl \
	--dataset dataset/combined_dataset.jsonl

notebook:
	jupytext --to notebook dataset/03_analysis.py -o dataset/03_analysis.ipynb



