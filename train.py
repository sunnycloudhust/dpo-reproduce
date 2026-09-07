import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_preference_pairs, make_loaders, tokenize_pairs
from engine import run_epoch
from config import CONFIG


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config = CONFIG
    set_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train, evaluation = load_preference_pairs(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = tokenize_pairs(train, tokenizer, config["max_length"])
    evaluation = tokenize_pairs(evaluation, tokenizer, config["max_length"])
    train_loader, eval_loader = make_loaders(
        train, evaluation, tokenizer, config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name_or_path"], num_labels=1
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    history = []
    for epoch in range(config["epochs"]):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config["gradient_accumulation_steps"],
            training=True,
        )
        eval_metrics = run_epoch(
            model,
            eval_loader,
            optimizer,
            device,
            config["gradient_accumulation_steps"],
            training=False,
        )
        metrics = {"epoch": epoch + 1, "train": train_metrics, "eval": eval_metrics}
        history.append(metrics)
        print(json.dumps(metrics))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "dataset": config["dataset_name"],
        "model": config["model_name_or_path"],
        "device": str(device),
        "train_pairs": len(train),
        "eval_pairs": len(evaluation),
        "seed": config["seed"],
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
