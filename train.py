import torch
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
