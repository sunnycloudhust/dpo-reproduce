"""Small preference-pair sanity evaluator based on sequence log-probability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prepare_data import read_jsonl, validate_examples


def completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    full = tokenizer(prompt + completion, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(full).logits[:, :-1]
    target = full[:, 1:]
    start = prompt_ids.shape[1] - 1
    token_logits = logits[:, start:]
    token_targets = target[:, start:]
    return float(torch.log_softmax(token_logits, dim=-1).gather(-1, token_targets.unsqueeze(-1)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).eval()
    wins = []
    for example in validate_examples(read_jsonl(args.dataset)):
        chosen = completion_logprob(model, tokenizer, example["prompt"], example["chosen"])
        rejected = completion_logprob(model, tokenizer, example["prompt"], example["rejected"])
        wins.append({"chosen_logprob": chosen, "rejected_logprob": rejected, "chosen_wins": chosen > rejected})
    result = {"accuracy": sum(item["chosen_wins"] for item in wins) / len(wins), "examples": wins}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
