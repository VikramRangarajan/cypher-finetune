import logging
from pathlib import Path
from typing import Optional
import os

from unsloth import FastModel # noqa
from langchain_neo4j.chains.graph_qa.prompts import CYPHER_GENERATION_TEMPLATE
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType
from pydantic_settings import BaseSettings, CliSettingsSource, SettingsConfigDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTConfig, SFTTrainer

from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb


def prepare_dataset(tokenizer):
    """Load and prepare the dataset."""
    dataset = load_dataset("neo4j/text2cypher-2025v1", split="train[:100]")
    dataset = standardize_data_formats(dataset)

    # Apply formatting
    def format_chat_template(example: dict) -> dict:
        langchain_prompt = CYPHER_GENERATION_TEMPLATE.removesuffix("\nThe question is:\n{question}")
        system_prompt = langchain_prompt.format(schema=example["schema"], examples="Not Provided\n")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["cypher"]},
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")

        return {"text": text}
    dataset = dataset.map(
        format_chat_template, num_proc=4
    )

    # Load validation dataset if specified
    eval_dataset = load_dataset("neo4j/text2cypher-2025v1", split="test")
    eval_dataset = standardize_data_formats(eval_dataset)
    eval_dataset = eval_dataset.map(
        format_chat_template,
        num_proc=4,
        remove_columns=eval_dataset.column_names,
        desc="Formatting validation chat templates",
    )

    cypherbench = load_dataset("megagonlabs/cypherbench", split="test")
    cypherbench = cypherbench.rename_columns({"gold_cypher": "cypher", "nl_question": "question"})
    cypherbench = cypherbench.add_column("schema", ["Not Provided"]*len(cypherbench))
    cypherbench = standardize_data_formats(cypherbench)
    cypherbench = cypherbench.map(
        format_chat_template,
        num_proc=4,
        remove_columns=cypherbench.column_names
    )


    return dataset, {"eval_ds": eval_dataset, "cypherbench": cypherbench}


def main():
    if "RUN_NAME" in os.environ:
        RUN = os.environ["RUN_NAME"]
        file_dir = f"output/{RUN}/final"
    else:
        RUN = "gemma4-baseline"
        file_dir = "unsloth/gemma-4-E2B-it"
    model = AutoModelForCausalLM.from_pretrained(file_dir)
    tokenizer = AutoTokenizer.from_pretrained(file_dir)
    train_dataset, eval_dataset = prepare_dataset(tokenizer)

    training_args = SFTConfig(
        dataset_num_proc=1,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        max_length=2048,
        dataset_text_field="text",
    )

    def metric(eval_pred, tokenizer, compute_result):
        preds, labels = eval_pred.predictions[:, :-1], eval_pred.label_ids[:, 1:]
        mask = labels != -100
        total_correct = ((preds == labels) & mask).sum().item()
        total_tokens = mask.sum().item()
        total_exact = (((preds == labels) | ~mask).all(axis=1)).sum().item()
        total_seqs = preds.shape[0]

        return {
            "token_accuracy": total_correct / total_tokens,
            "exact_match_accuracy": total_exact / total_seqs,
        }


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_eval_metrics=True,
        compute_metrics=lambda eval_pred, compute_result=False: metric(eval_pred, compute_result=compute_result, tokenizer=tokenizer),
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1)
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|turn>user\n",
        response_part = "<|turn>model\n",
    )

    eval_res = trainer.evaluate()
    print(eval_res)
    run = [run for run in wandb.Api().runs("vr-umiacs/text2cypher-gemma4") if run.name == RUN]
    with wandb.init(entity="vr-umiacs", project="text2cypher-gemma4", id=run[0].id if len(run) > 0 else None):
        wandb.log(eval_res)

if __name__ == "__main__":
    main()
