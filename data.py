from datasets import load_dataset
from torch.utils.data import DataLoader


def load_preference_pairs(config):
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    train = dataset["train"]
    required = {"chosen", "rejected"}
    if not required.issubset(train.column_names):
        raise ValueError(f"Dataset must contain {sorted(required)}; got {train.column_names}")

    split = train.train_test_split(test_size=config.eval_ratio, seed=config.seed)
    train, evaluation = split["train"], split["test"]
    if config.max_train_samples:
        train = train.select(range(min(config.max_train_samples, len(train))))
    if config.max_eval_samples:
        evaluation = evaluation.select(range(min(config.max_eval_samples, len(evaluation))))
    return train, evaluation


def tokenize_pairs(dataset, tokenizer, max_length):
    def tokenize(example):
        chosen = tokenizer(example["chosen"], truncation=True, max_length=max_length)
        rejected = tokenizer(example["rejected"], truncation=True, max_length=max_length)
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def collate_pairs(batch, tokenizer):
    def pad(prefix):
        return tokenizer.pad(
            {
                "input_ids": [item[f"{prefix}_input_ids"] for item in batch],
                "attention_mask": [item[f"{prefix}_attention_mask"] for item in batch],
            },
            return_tensors="pt",
        )

    return {"chosen": pad("chosen"), "rejected": pad("rejected")}


def make_loaders(train, evaluation, tokenizer, batch_size):
    collate = lambda batch: collate_pairs(batch, tokenizer)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(evaluation, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, eval_loader
