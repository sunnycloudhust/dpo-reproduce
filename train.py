import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_preference_pairs, make_loaders, tokenize_pairs
from engine import run_epoch
from config import CONFIG


def main():
    config = CONFIG
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting reward-model training with {config['model_name']}")

    train, evaluation = load_preference_pairs(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = tokenize_pairs(train, tokenizer, config["max_length"])
    evaluation = tokenize_pairs(evaluation, tokenizer, config["max_length"])
    train_loader, eval_loader = make_loaders(
        train, evaluation, tokenizer, config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, train_pairs={len(train)}, eval_pairs={len(evaluation)}")
    model_dtype = None
    if device.type == "cuda" and config["mixed_precision"]:
        model_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=1, torch_dtype=model_dtype
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
        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_accuracy={train_metrics['accuracy']:.4f} | "
            f"eval_loss={eval_metrics['loss']:.4f} | "
            f"eval_accuracy={eval_metrics['accuracy']:.4f}"
        )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "dataset": config["dataset_name"],
        "model": config["model_name"],
        "device": str(device),
        "train_pairs": len(train),
        "eval_pairs": len(evaluation),
        "seed": config["seed"],
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Training complete; metrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
