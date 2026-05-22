from peft import LoraConfig, TaskType
import logging
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from pydantic_settings import BaseSettings, CliSettingsSource, SettingsConfigDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer

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
    model_name: str = "google/gemma-4-E2B-it"
    use_flash_attention: bool = False
    use_4bit: bool = False
    use_8bit: bool = False
    torch_compile: bool = False

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: Optional[str] = None

    # Dataset settings
    dataset_name: str = "tomasonjo/text2cypher-gpt4o-clean"
    dataset_split: str = "train"
    validation_split: Optional[str] = None
    max_seq_length: int = 2048

    # Training hyperparameters
    output_dir: str = "./output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    optim: str = "adamw_torch_fused"

    # Precision settings
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100

    # Logging
    logging_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "text2cypher-gemma4"
    wandb_run_name: Optional[str] = None

    # System settings
    seed: int = 42
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

    # SFT-specific settings
    packing: bool = False
    dataset_text_field: str = "text"


def format_chat_template(example: dict, tokenizer) -> dict:
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
    )

    return {"text": text}


def create_model_and_tokenizer(config: TrainingConfig):
    """Create and configure the model and tokenizer."""

    # Quantization config
    quantization_config = None
    if config.use_4bit:
        logger.info("Using 4-bit quantization (QLoRA)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.use_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Model kwargs
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "dtype": torch.bfloat16 if config.bf16 else torch.float16,
    }

    if config.use_flash_attention:
        logger.info("Using Flash Attention 2")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load model
    logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return model, tokenizer


def prepare_dataset(config: TrainingConfig, tokenizer):
    """Load and prepare the dataset."""

    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    # Apply formatting
    logger.info("Formatting dataset with chat template...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting chat templates",
    )

    # Load validation dataset if specified
    eval_dataset = None
    if config.validation_split:
        logger.info(f"Loading validation dataset: {config.validation_split}")
        eval_dataset = load_dataset(config.dataset_name, split=config.validation_split)
        eval_dataset = eval_dataset.map(
            lambda x: format_chat_template(x, tokenizer),
            remove_columns=eval_dataset.column_names,
            desc="Formatting validation chat templates",
        )

    logger.info(f"Training dataset size: {len(dataset)}")
    if eval_dataset:
        logger.info(f"Validation dataset size: {len(eval_dataset)}")

    return dataset, eval_dataset


def create_peft_config(config: TrainingConfig):
    """Create PEFT (LoRA) configuration."""
    if not config.use_lora:
        return None



    target_modules = None
    if config.lora_target_modules:
        target_modules = [m.strip() for m in config.lora_target_modules.split(",")]

    logger.info(f"Using LoRA with r={config.lora_r}, alpha={config.lora_alpha}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    return peft_config


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

    # Create PEFT config
    peft_config = create_peft_config(config)

    # Create output directory
    output_dir = Path(config.output_dir) / wandb.run.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        torch_compile=config.torch_compile,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        optim=config.optim,
        bf16=config.bf16,
        fp16=config.fp16,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy=config.eval_strategy if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        logging_steps=config.logging_steps,
        report_to="wandb" if config.use_wandb else "none",
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        # SFT-specific
        max_length=config.max_seq_length,
        packing=config.packing,
        dataset_text_field=config.dataset_text_field,
    )

    # Create trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

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
