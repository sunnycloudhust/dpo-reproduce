from contextlib import nullcontext

import torch

from loss import pairwise_preference_loss, score


def run_epoch(model, loader, optimizer, device, accumulation_steps, training):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_pairs = 0
    context = torch.enable_grad() if training else torch.no_grad()
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and next(model.parameters()).dtype == torch.bfloat16
        else torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and next(model.parameters()).dtype == torch.float16
        else nullcontext()
    )

    with context, autocast_context:
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
