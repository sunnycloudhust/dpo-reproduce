import torch

from reward_loss import pairwise_preference_loss, score


def run_epoch(model, loader, optimizer, device, accumulation_steps, training):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_pairs = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for step, batch in enumerate(loader):
            chosen = {key: value.to(device) for key, value in batch["chosen"].items()}
            rejected = {key: value.to(device) for key, value in batch["rejected"].items()}
            loss, margins = pairwise_preference_loss(
                score(model, chosen), score(model, rejected)
            )
            if training:
                (loss / accumulation_steps).backward()
                if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * len(margins)
            total_correct += (margins > 0).sum().item()
            total_pairs += len(margins)

    return {"loss": total_loss / total_pairs, "accuracy": total_correct / total_pairs}
