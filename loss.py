import torch
from torch.nn import functional as F


def pairwise_preference_loss(chosen_scores, rejected_scores):
    margins = chosen_scores - rejected_scores
    return -F.logsigmoid(margins).mean(), margins


def score(model, inputs):
    return model(**inputs).logits.squeeze(-1)
