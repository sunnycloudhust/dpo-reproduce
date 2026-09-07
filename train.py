import json
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_preference_pairs, make_loaders, tokenize_pairs
from config import CONFIG
from loss import pairwise_preference_loss, score


def train(model, train_loader, eval_loader, optimizer, device, config):
    history = []
    accumulation_steps = config["gradient_accumulation_steps"]

    for epoch in range(config["epochs"]):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_pairs = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            chosen = {key: value.to(device) for key, value in batch["chosen"].items()}
            rejected = {key: value.to(device) for key, value in batch["rejected"].items()}
            autocast_context = get_autocast_context(model, device)
            with autocast_context:
                loss, margins = pairwise_preference_loss(
                    score(model, chosen), score(model, rejected)
                )
            (loss / accumulation_steps).backward()
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pair_count = len(margins)
            train_loss_total += loss.item() * pair_count
            train_correct += (margins > 0).sum().item()
            train_pairs += pair_count
            print(
                f"Epoch {epoch + 1}/{config['epochs']} | "
                f"train step {step + 1}/{len(train_loader)} | "
                f"loss={loss.item():.4f}"
            )

        model.eval()
        eval_loss_total = 0.0
        eval_correct = 0
        eval_pairs = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_loader):
                chosen = {key: value.to(device) for key, value in batch["chosen"].items()}
                rejected = {key: value.to(device) for key, value in batch["rejected"].items()}
                autocast_context = get_autocast_context(model, device)
                with autocast_context:
                    loss, margins = pairwise_preference_loss(
                        score(model, chosen), score(model, rejected)
                    )
                pair_count = len(margins)
                eval_loss_total += loss.item() * pair_count
                eval_correct += (margins > 0).sum().item()
                eval_pairs += pair_count
                print(
                    f"Epoch {epoch + 1}/{config['epochs']} | "
                    f"eval step {step + 1}/{len(eval_loader)} | "
                    f"loss={loss.item():.4f}"
                )

        train_metrics = {
            "loss": train_loss_total / train_pairs,
            "accuracy": train_correct / train_pairs,
        }
        eval_metrics = {
            "loss": eval_loss_total / eval_pairs,
            "accuracy": eval_correct / eval_pairs,
        }
        metrics = {"epoch": epoch + 1, "train": train_metrics, "eval": eval_metrics}
        history.append(metrics)
        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_accuracy={train_metrics['accuracy']:.4f} | "
            f"eval_loss={eval_metrics['loss']:.4f} | "
            f"eval_accuracy={eval_metrics['accuracy']:.4f}"
        )

    return history


def get_autocast_context(model, device):
    if device.type != "cuda":
        return nullcontext()
    model_dtype = next(model.parameters()).dtype
    if model_dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=model_dtype)
    return nullcontext()


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

    history = train(model, train_loader, eval_loader, optimizer, device, config)

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
