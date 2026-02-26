"""
02_run_inference.py
Run vLLM inference on the combined dataset and log expert selections per token per layer.

Requires a MoE model (e.g., THUDM/GLM-4.5-9B or similar) that uses FusedMoE layers.
The script leverages vLLM's built-in enable_return_routed_experts flag which captures
the topk expert IDs for every token at every MoE layer.

Output log format (JSONL, one line per sample):
  {
    "id":               str,   # sample ID from combined dataset
    "dataset_name":     str,   # source dataset name
    "num_prompt_tokens": int,  # number of prompt tokens (input)
    "num_output_tokens": int,  # number of generated tokens (output)
    "total_tokens":     int,   # num_prompt_tokens + num_output_tokens
    "num_layers":       int,   # number of MoE layers captured
    "topk":             int,   # top-k experts per token per layer
    "experts_shape":   [int, int, int],  # [total_tokens, num_layers, topk]
    "experts_b64":      str,   # base64-encoded int32 numpy array (C order)
  }

Decode experts_b64:
  import base64, numpy as np
  arr = np.frombuffer(base64.b64decode(rec["experts_b64"]), dtype=np.int32)
  arr = arr.reshape(rec["experts_shape"])
  # arr[token_pos, layer_idx, k] -> expert ID

Usage:
  python 02_run_inference.py \\
      --dataset combined_dataset.jsonl \\
      --model THUDM/GLM-4.5-9B \\
      --output expert_log.jsonl \\
      [--tensor_parallel_size 1] \\
      [--max_input_tokens 4096] \\
      [--max_output_tokens 128]

Dependencies:
  vllm (this repo), numpy
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams


def encode_experts(arr: np.ndarray) -> str:
    """Encode a numpy int32 array to a base64 string for compact storage."""
    return base64.b64encode(arr.astype(np.int32).tobytes()).decode("ascii")


def load_dataset(path: str) -> list[dict]:
    """Load the combined JSONL dataset produced by 01_prepare_datasets.py."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def truncate_input(text: str, max_chars: int) -> str:
    """Hard-truncate by character count as a pre-tokenisation safety guard."""
    return text[:max_chars]


def main():
    parser = argparse.ArgumentParser(description="Run inference and log expert selections.")
    parser.add_argument("--dataset", default="combined_dataset.jsonl",
                        help="Path to the combined JSONL dataset (from 01_prepare_datasets.py)")
    parser.add_argument("--model", default="THUDM/GLM-4.5-9B",
                        help="HuggingFace model name or local path (must be a MoE model)")
    parser.add_argument("--output", default="expert_log.jsonl",
                        help="Output JSONL log file path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_input_tokens", type=int, default=4096,
                        help="Approximate input token limit; input is truncated to max_input_tokens*4 chars")
    parser.add_argument("--max_output_tokens", type=int, default=128,
                        help="Maximum number of tokens to generate per sample")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples whose IDs are already in the output file")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Only process samples from these dataset names")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    records = load_dataset(args.dataset)
    if args.datasets:
        records = [r for r in records if r["dataset_name"] in args.datasets]
    print(f"Loaded {len(records)} records from {args.dataset}")

    # ------------------------------------------------------------------
    # Resume: collect IDs already processed
    # ------------------------------------------------------------------
    done_ids: set[str] = set()
    if args.resume and Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"Resuming: {len(done_ids)} samples already done, skipping them.")
        records = [r for r in records if r["id"] not in done_ids]
        print(f"Remaining: {len(records)} samples to process.")

    if not records:
        print("Nothing to process. Exiting.")
        return

    # ------------------------------------------------------------------
    # Initialise vLLM
    # enable_return_routed_experts=True makes vLLM capture and return
    # the topk expert indices for every token at every MoE layer.
    # Only TP rank 0 performs the logging (the capturer skips other ranks).
    # ------------------------------------------------------------------
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_return_routed_experts=True,
        trust_remote_code=True,
        # Enforce eager mode so every forward pass is visible to the capturer
        # without CUDA graph replay complications.
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_output_tokens,
        temperature=0.0,  # greedy for reproducibility
    )

    # ------------------------------------------------------------------
    # Inference loop  – one prompt at a time for simple, deterministic logs
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    with out_path.open("a", encoding="utf-8") as log_file:
        for rec in records:
            sample_id = rec["id"]
            dataset_name = rec["dataset_name"]
            prompt = truncate_input(rec["input"], args.max_input_tokens * 4)

            # Generate one sample; returns a list with one RequestOutput
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            request_output = outputs[0]
            completion = request_output.outputs[0]

            # routed_experts: np.ndarray of shape [total_seq_len, num_layers, topk]
            # total_seq_len = num_prompt_tokens + num_output_tokens
            # Returns None if the model has no MoE layers or capture failed.
            routed_experts: np.ndarray | None = completion.routed_experts

            num_prompt_tokens = len(request_output.prompt_token_ids or [])
            num_output_tokens = len(completion.token_ids)

            if routed_experts is None:
                print(
                    f"WARNING: routed_experts is None for sample '{sample_id}'. "
                    "Check that the model is a MoE model and vLLM supports it."
                )
                # Still write a record so the user knows which samples failed
                entry = {
                    "id": sample_id,
                    "dataset_name": dataset_name,
                    "num_prompt_tokens": num_prompt_tokens,
                    "num_output_tokens": num_output_tokens,
                    "total_tokens": num_prompt_tokens + num_output_tokens,
                    "num_layers": None,
                    "topk": None,
                    "experts_shape": None,
                    "experts_b64": None,
                }
            else:
                total_tokens, num_layers, topk = routed_experts.shape
                entry = {
                    "id": sample_id,
                    "dataset_name": dataset_name,
                    "num_prompt_tokens": num_prompt_tokens,
                    "num_output_tokens": num_output_tokens,
                    "total_tokens": total_tokens,
                    "num_layers": num_layers,
                    "topk": topk,
                    "experts_shape": list(routed_experts.shape),
                    # Compact binary encoding: base64(int32 bytes, C order)
                    "experts_b64": encode_experts(routed_experts),
                }

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

            print(
                f"[{sample_id}] prompt_tokens={entry['num_prompt_tokens']}, "
                f"output_tokens={entry['num_output_tokens']}, "
                f"experts_shape={entry['experts_shape']}"
            )

    print(f"\nDone. Logs written to {out_path}")


if __name__ == "__main__":
    main()
