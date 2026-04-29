import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "case",
    "model",
    "gpu_count",
    "request_rate",
    "num_prompts",
    "completed",
    "failed",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
]


def iter_json_records(path: Path):
    for json_path in sorted(path.glob("*.json")):
        text = json_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            yield json_path, json.loads(line)


def get_metadata(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record.get(key)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def flatten_record(json_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in FIELDS:
        row[field] = record.get(field)

    row["case"] = get_metadata(record, "case") or json_path.stem
    row["model"] = get_metadata(record, "model") or record.get("model_id")
    row["gpu_count"] = get_metadata(record, "gpu_count")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize vllm bench serve JSON files into CSV."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        flatten_record(path, record)
        for path, record in iter_json_records(args.result_dir)
    ]

    output = args.output or args.result_dir / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
