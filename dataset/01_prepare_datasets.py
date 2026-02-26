"""
01_prepare_datasets.py
Download and combine five datasets into a single unified JSONL file.

Each output record has:
  id           - "<dataset_name>-<original_id_or_index>"
  input        - the text prompt to feed to the model
  dataset_name - which source dataset
  original_index - integer position inside the source dataset (or split)
  original_id  - the id string from the source dataset, or None

Usage:
  python 01_prepare_datasets.py --output combined_dataset.jsonl
  python 01_prepare_datasets.py --output combined_dataset.jsonl --max_samples 200

Dependencies:
  pip install datasets huggingface_hub tqdm
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import hf_hub_download
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-dataset processors
# ---------------------------------------------------------------------------

def _fmt_sharegpt(row: dict, index: int):
    """
    ShareGPT: conversations list of {from, value}.
    We concatenate every turn labelled human/gpt/system.
    """
    turns = []
    for turn in row.get("conversations", []):
        speaker = turn.get("from", "unknown").capitalize()
        text = turn.get("value", "").strip()
        if text:
            turns.append(f"{speaker}: {text}")

    original_id = str(row.get("id", ""))
    record_id = f"sharegpt-{original_id}" if original_id else f"sharegpt-{index}"
    return {
        "id": record_id,
        "input": "\n".join(turns),
        "dataset_name": "sharegpt",
        "original_index": index,
        "original_id": original_id or None,
    }


def _fmt_leval(row: dict, index: int, config_name: str = ""):
    """
    L-Eval: long document + one or more instruction questions.
    Fields: input (doc), instructions (list of questions), outputs.
    We use the first instruction as the representative question.
    """
    doc = row.get("input", "").strip()
    instructions = row.get("instructions", [])
    question = instructions[0].strip() if instructions else ""

    if question:
        combined = f"Document:\n{doc}\n\nQuestion:\n{question}"
    else:
        combined = doc

    suffix = f"-{config_name}" if config_name else ""
    record_id = f"leval{suffix}-{index}"
    return {
        "id": record_id,
        "input": combined,
        "dataset_name": "leval",
        "original_index": index,
        "original_id": None,
    }


def _fmt_longbench2(row: dict, index: int):
    """
    LongBench-v2: context + question + multiple-choice options.
    Fields: _id, context, question, choice_A/B/C/D, answer.
    """
    context = row.get("context", "").strip()
    question = row.get("question", "").strip()
    choices = "\n".join([
        f"A. {row.get('choice_A', '')}",
        f"B. {row.get('choice_B', '')}",
        f"C. {row.get('choice_C', '')}",
        f"D. {row.get('choice_D', '')}",
    ])

    combined = f"{context}\n\nQuestion: {question}\n{choices}"

    original_id = str(row.get("_id", ""))
    record_id = f"longbench2-{original_id}" if original_id else f"longbench2-{index}"
    return {
        "id": record_id,
        "input": combined,
        "dataset_name": "longbench2",
        "original_index": index,
        "original_id": original_id or None,
    }


def _fmt_lmsys(row: dict, index: int):
    """
    lmsys-chat-1m: conversation list of {role, content}.
    We concatenate every turn.
    """
    turns = []
    for turn in row.get("conversation", []):
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "").strip()
        if content:
            turns.append(f"{role}: {content}")

    original_id = str(row.get("conversation_id", ""))
    record_id = f"lmsys-{original_id}" if original_id else f"lmsys-{index}"
    return {
        "id": record_id,
        "input": "\n".join(turns),
        "dataset_name": "lmsys",
        "original_index": index,
        "original_id": original_id or None,
    }


def _fmt_loogle(row: dict, index: int, config_name: str = ""):
    """
    LooGLE: long document + optional question.
    Parquet fields: context (doc), question, answer, title.
    """
    doc = row.get("context", "").strip()
    question = row.get("question", "").strip()

    if question:
        combined = f"{doc}\n\nQuestion: {question}"
    else:
        combined = doc

    suffix = f"-{config_name}" if config_name else ""
    record_id = f"loogle{suffix}-{index}"
    return {
        "id": record_id,
        "input": combined,
        "dataset_name": "loogle",
        "original_index": index,
        "original_id": None,
    }


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _tqdm_total(ds, limit):
    """Return tqdm total: dataset length when no limit, else min(limit, len)."""
    return len(ds) if limit is None else min(limit, len(ds))


def load_sharegpt(max_samples: int | None) -> list[dict]:
    """Load ShareGPT Vicuna unfiltered dataset."""
    print("Loading ShareGPT...")
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
    )
    records = []
    for i, row in enumerate(tqdm(ds, desc="ShareGPT", total=_tqdm_total(ds, max_samples))):
        if max_samples is not None and i >= max_samples:
            break
        rec = _fmt_sharegpt(row, i)
        if rec["input"].strip():
            records.append(rec)
    return records


def load_leval(max_samples: int | None) -> list[dict]:
    """Load L-Eval by fetching JSONL files directly (bypasses deprecated loading script)."""
    print("Loading L-Eval...")

    # Map config name -> file path inside the L4NLP/LEval HF repo.
    # Derived from LEval.py in the repo; avoids the now-unsupported script loader.
    LEVAL_CONFIGS = {
        # Closed-ended (Exam)
        "coursera":                 "LEval/Exam/coursera.jsonl",
        "gsm100":                   "LEval/Exam/gsm100.jsonl",
        "quality":                  "LEval/Exam/quality.jsonl",
        "topic_retrieval_longchat": "LEval/Exam/topic_retrieval_longchat.jsonl",
        "tpo":                      "LEval/Exam/tpo.jsonl",
        "sci_fi":                   "LEval/Exam/sci_fi.jsonl",
        "codeU":                    "LEval/Exam/codeU.jsonl",
        # Open-ended (Generation)
        "financial_qa":             "LEval/Generation/financial_qa.jsonl",
        "gov_report_summ":          "LEval/Generation/gov_report_summ.jsonl",
        "legal_contract_qa":        "LEval/Generation/legal_contract_qa.jsonl",
        "meeting_summ":             "LEval/Generation/meeting_summ.jsonl",
        "multidoc_qa":              "LEval/Generation/multidoc_qa.jsonl",
        "narrative_qa":             "LEval/Generation/narrative_qa.jsonl",
        "natural_question":         "LEval/Generation/natural_question.jsonl",
        "news_summ":                "LEval/Generation/news_summ.jsonl",
        "paper_assistant":          "LEval/Generation/paper_assistant.jsonl",
        "patent_summ":              "LEval/Generation/patent_summ.jsonl",
        "review_summ":              "LEval/Generation/review_summ.jsonl",
        "scientific_qa":            "LEval/Generation/scientific_qa.jsonl",
        "tv_show_summ":             "LEval/Generation/tv_show_summ.jsonl",
    }

    per_config = (
        None if max_samples is None
        else max(1, max_samples // len(LEVAL_CONFIGS))
    )
    records = []

    for cfg, filepath in LEVAL_CONFIGS.items():
        try:
            local_path = hf_hub_download(
                repo_id="L4NLP/LEval",
                filename=filepath,
                repo_type="dataset",
            )
            ds = load_dataset("json", data_files=local_path, split="train")
        except Exception as e:
            print(f"  Skipping L-Eval config '{cfg}': {e}")
            continue

        for i, row in enumerate(tqdm(ds, desc=f"L-Eval/{cfg}", total=_tqdm_total(ds, per_config))):
            if per_config is not None and i >= per_config:
                break
            rec = _fmt_leval(row, i, config_name=cfg)
            if rec["input"].strip():
                records.append(rec)

    return records


def load_longbench2(max_samples: int | None) -> list[dict]:
    """Load LongBench-v2."""
    print("Loading LongBench-v2...")
    ds = load_dataset("zai-org/LongBench-v2", split="train", trust_remote_code=True)
    records = []
    for i, row in enumerate(tqdm(ds, desc="LongBench-v2", total=_tqdm_total(ds, max_samples))):
        if max_samples is not None and i >= max_samples:
            break
        rec = _fmt_longbench2(row, i)
        if rec["input"].strip():
            records.append(rec)
    return records


def load_lmsys(max_samples: int | None) -> list[dict]:
    """Load lmsys-chat-1m (requires HuggingFace login)."""
    print("Loading lmsys-chat-1m...")
    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        trust_remote_code=True,
    )
    records = []
    for i, row in enumerate(tqdm(ds, desc="lmsys-chat-1m", total=_tqdm_total(ds, max_samples))):
        if max_samples is not None and i >= max_samples:
            break
        rec = _fmt_lmsys(row, i)
        if rec["input"].strip():
            records.append(rec)
    return records


def load_loogle(max_samples: int | None) -> list[dict]:
    """Load LooGLE across all available configs."""
    print("Loading LooGLE...")
    try:
        configs = get_dataset_config_names("bigai-nlco/LooGLE")
    except Exception:
        configs = ["default"]

    per_config = (
        None if max_samples is None
        else max(1, max_samples // max(len(configs), 1))
    )
    records = []

    for cfg in configs:
        try:
            ds = load_dataset("bigai-nlco/LooGLE", cfg, split="test")
        except Exception:
            try:
                ds = load_dataset("bigai-nlco/LooGLE", cfg)
                ds = ds[list(ds.keys())[0]]
            except Exception as e:
                print(f"  Skipping LooGLE config '{cfg}': {e}")
                continue

        for i, row in enumerate(tqdm(ds, desc=f"LooGLE/{cfg}", total=_tqdm_total(ds, per_config))):
            if per_config is not None and i >= per_config:
                break
            rec = _fmt_loogle(row, i, config_name=cfg)
            if rec["input"].strip():
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare and combine datasets.")
    parser.add_argument(
        "--output", default="combined_dataset.jsonl",
        help="Output JSONL file path (default: combined_dataset.jsonl)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max samples per dataset; omit to load all records (default: all)",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["sharegpt", "leval", "longbench2", "lmsys", "loogle"],
        default=["sharegpt", "leval", "longbench2", "lmsys", "loogle"],
        help="Which datasets to include (default: all five)",
    )
    args = parser.parse_args()

    loaders = {
        "sharegpt": load_sharegpt,
        "leval": load_leval,
        "longbench2": load_longbench2,
        "lmsys": load_lmsys,
        "loogle": load_loogle,
    }

    all_records: list[dict] = []
    for name in args.datasets:
        try:
            records = loaders[name](args.max_samples)
            all_records.extend(records)
            print(f"  -> {len(records)} records from {name}")
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")

    # Ensure IDs are unique (deduplicate by appending suffix if needed)
    seen: dict[str, int] = {}
    for rec in all_records:
        orig_id = rec["id"]
        if orig_id in seen:
            seen[orig_id] += 1
            rec["id"] = f"{orig_id}-dup{seen[orig_id]}"
        else:
            seen[orig_id] = 0

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_records)} records to {out_path}")


if __name__ == "__main__":
    main()
