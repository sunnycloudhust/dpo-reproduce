# Preference Optimization Reproduction

This repository contains a small research pipeline for preference optimization. It currently includes reward-model training on Anthropic HH-RLHF, best-of-N test-time selection, and a DPO-oriented project layout.

## Reward Model Results

The current reward-model checkpoint was trained from `Qwen/Qwen2.5-0.5B-Instruct` on `Anthropic/hh-rlhf` using 25,000 training pairs, 16,080 evaluation pairs, and 2 epochs.

| Epoch | Train loss | Train accuracy | Eval loss | Eval accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.6939 | 47.72% | 0.6443 | 51.56% |
| 2 | 0.5761 | 58.45% | 0.6606 | 52.43% |

Accuracy is the fraction of pairs for which the model scores `chosen` higher than `rejected`. The final evaluation accuracy is 52.43%, which is close to random performance. This checkpoint is therefore experimental and is not intended for production use without further validation.

The checkpoint and its Model Card are available on [Hugging Face](https://huggingface.co/sunnycloudhust/reward-model-hh-rlhf). The raw metrics are stored in `outputs/reward_model_hh_rlhf/metrics.json`.

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended for full training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Train the Reward Model

The reward model uses pairwise preference loss:

`-log(sigmoid(score(chosen) - score(rejected)))`

The `Anthropic/hh-rlhf` dataset already provides `chosen` and `rejected` columns, so no additional normalization is required.

All training settings are defined in `config.py`. The default configuration uses a maximum sequence length of 256, batch size 2, gradient accumulation, and two GPUs through `DataParallel`.

```bash
python main.py
```

The script saves the model, tokenizer, and aggregate metrics to `outputs/reward_model_hh_rlhf/`. For a smaller smoke run, set `max_train_samples`, `max_eval_samples`, and `max_test_samples` in `config.py` before starting training.

## Best-of-N Test-Time Alignment

After training a reward model, run:

```bash
python experiment.py \
  --reward-model outputs/reward_model_hh_rlhf \
  --max-prompts 100 \
  --output outputs/test_time_alignment/results.json
```

The experiment generates candidate responses with a base model, scores them with the reward model, and selects the highest-scoring response for `N = 1, 2, 4, 8`. The output contains prompts, candidates, scores, baseline responses, and selected responses.

This measures reward-model selection rather than independent response quality. Use human evaluation or a fixed external judge to estimate win rate; the reward model itself should not be treated as ground truth.

## Files

```text
config.py       Training and experiment settings
data.py         Preference dataset loading
loss.py         Pairwise preference loss
train.py        Training loop
main.py         Reward-model training entry point
experiment.py   Best-of-N test-time experiment
outputs/        Checkpoints, metrics, and experiment results
```

## Reproducibility Checklist

- Record the commit, configuration, model revision, and dataset revision.
- Use the same split and maximum sequence length for comparisons.
- Track loss, chosen/rejected rewards, and reward margins.
