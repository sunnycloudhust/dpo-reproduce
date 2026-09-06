"""Train a DPO model from a YAML experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def load_preferences(config: dict):
    if config.get("dataset_name"):
        dataset = load_dataset(config["dataset_name"], split=config.get("dataset_split", "train"))
    else:
        dataset = load_dataset("json", data_files=config["train_file"], split="train")
    required = {"prompt", "chosen", "rejected"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name-or-path")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.model_name_or_path:
        config["model_name_or_path"] = args.model_name_or_path

    set_seed(config["seed"])
    dataset = load_preferences(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if config.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **model_kwargs)
    ref_model = None
    if config.get("ref_model_name_or_path"):
        ref_model = AutoModelForCausalLM.from_pretrained(config["ref_model_name_or_path"])

    training_args = DPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        beta=config["beta"],
        max_length=config["max_length"],
        max_prompt_length=config["max_prompt_length"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        warmup_ratio=config["warmup_ratio"],
        seed=config["seed"],
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
