from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_config: str | None = None
    model_name_or_path: str = "distilbert-base-uncased"
    output_dir: str = "outputs/reward_model_hh_rlhf"
    max_length: int = 512
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_ratio: float = 0.02
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    seed: int = 42


def load_config() -> RewardConfig:
    return RewardConfig()
