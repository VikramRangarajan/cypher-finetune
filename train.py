import logging
from pathlib import Path
from typing import Optional

from unsloth import FastModel # noqa
import torch
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

import wandb

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class TrainingConfig(BaseSettings):
    """Training configuration for Gemma 4 fine-tuning on text-to-Cypher."""

    model_config = SettingsConfigDict(
        env_prefix="TRAIN_",
        cli_parse_args=True,
        cli_settings_source=CliSettingsSource,
    )

    # Model settings
    model_name: str = "unsloth/gemma-4-E2B-it"
    use_flash_attention: bool = False
    use_4bit: bool = False

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: Optional[str] = None

    # Dataset settings
    dataset_name: str = "neo4j/text2cypher-2025v1"
    dataset_split: str = "train"
    validation_split: Optional[str] = "test"
    max_seq_length: int = 2048

    # Training hyperparameters
    output_dir: str = "./output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    weight_decay: float = 0.001
    optim: str = "adamw_8bit"

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    # Evaluation
    eval_strategy: str = "no"
    eval_steps: int = 50

    # Logging
    logging_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "text2cypher-gemma4"
    wandb_run_name: Optional[str] = None

    # System settings
    seed: int = 42





def create_model_and_tokenizer(config: TrainingConfig):
    model, tokenizer = FastModel.from_pretrained(
        model_name = config.model_name,
        dtype = None,
        max_seq_length = config.max_seq_length, # Choose any for long context!
        load_in_4bit = config.use_4bit,  # 4 bit quantization to reduce memory
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # Should leave on always!

        r = config.lora_r,           # Larger = higher accuracy, but might overfit
        lora_alpha = config.lora_alpha,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-4",
    )
    return model, tokenizer


def prepare_dataset(config: TrainingConfig, tokenizer):
    """Load and prepare the dataset."""

    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    dataset = standardize_data_formats(dataset)

    # Apply formatting
    def format_chat_template(example: dict) -> dict:
        system_prompt = f"You are an expert in converting natural language questions to Cypher queries. Use the following schema: {example['schema']}"
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
    logger.info("Formatting dataset with chat template...")
    dataset = dataset.map(
        format_chat_template, num_proc=4
    )

    # Load validation dataset if specified
    eval_dataset = None
    if config.validation_split:
        logger.info(f"Loading validation dataset: {config.validation_split}")
        eval_dataset = load_dataset(config.dataset_name, split=config.validation_split)
        eval_dataset = eval_dataset.map(
            format_chat_template,
            num_proc=4,
            remove_columns=eval_dataset.column_names,
            desc="Formatting validation chat templates",
        )

    logger.info(f"Training dataset size: {len(dataset)}")
    if eval_dataset:
        logger.info(f"Validation dataset size: {len(eval_dataset)}")

    return dataset, eval_dataset


def main():
    """Main training function."""

    # Parse configuration
    config = TrainingConfig()

    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    for field, value in config.model_dump().items():
        logger.info(f"{field}: {value}")
    logger.info("=" * 80)

    # Initialize W&B
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.model_dump(),
        )

    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(config)

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)


    # Create output directory
    output_dir = Path(config.output_dir) / wandb.run.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        dataset_num_proc=1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        optim=config.optim,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy=config.eval_strategy if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        logging_steps=config.logging_steps,
        report_to="wandb" if config.use_wandb else "none",
        seed=config.seed,
        # SFT-specific
        max_length=config.max_seq_length,
        dataset_text_field="text",
    )

    # Create trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|turn>user\n",
        response_part = "<|turn>model\n",
    )

    # Train
    logger.info("Starting training...")
    trainer.train()
    trainer.evaluate()

    # Save final model
    logger.info(f"Saving final model to {output_dir / 'final'}")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Finish W&B
    if config.use_wandb:
        wandb.finish()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
