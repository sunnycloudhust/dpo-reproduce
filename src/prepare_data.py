"""Validate and normalize preference datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


def validate_examples(examples: Iterable[dict]) -> list[dict[str, str]]:
    """Return clean preference examples or raise a useful error."""
    cleaned = []
    for index, example in enumerate(examples):
        missing = [column for column in REQUIRED_COLUMNS if not example.get(column)]
        if missing:
            raise ValueError(f"Example {index} is missing required fields: {', '.join(missing)}")
        cleaned.append({column: str(example[column]).strip() for column in REQUIRED_COLUMNS})
    if not cleaned:
        raise ValueError("Preference dataset is empty")
    return cleaned


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(examples: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    examples = validate_examples(read_jsonl(args.input))
    output = Path(args.output_dir) / "preferences.jsonl"
    write_jsonl(examples, output)
    print(f"Validated {len(examples)} examples -> {output}")


if __name__ == "__main__":
    main()
