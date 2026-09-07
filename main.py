import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CONFIG
from data import load_preference_pairs, make_loaders, tokenize_pairs
from train import train


def main():
    config = CONFIG
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting reward-model training with {config['reward_model_name']}")

    train_dataset, eval_dataset = load_preference_pairs(config)
    tokenizer = AutoTokenizer.from_pretrained(config["reward_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = tokenize_pairs(train_dataset, tokenizer, config["max_length"])
    eval_dataset = tokenize_pairs(eval_dataset, tokenizer, config["max_length"])
    train_loader, eval_loader = make_loaders(
        train_dataset, eval_dataset, tokenizer, config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device={device}, train_pairs={len(train_dataset)}, "
        f"eval_pairs={len(eval_dataset)}"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config["reward_model_name"], num_labels=1
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    history = train(model, train_loader, eval_loader, optimizer, device, config)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "dataset": config["dataset_name"],
        "model": config["reward_model_name"],
        "device": str(device),
        "train_pairs": len(train_dataset),
        "eval_pairs": len(eval_dataset),
        "seed": config["seed"],
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Training complete; metrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()