import json
import os
from pathlib import Path
from cypherbench.schema import PropertyGraphSchema, DataType

from transformers import AutoModelForCausalLM, AutoProcessor
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from safetensors import safe_open
import trackio
import warnings
from neo4j.warnings import PreviewWarning
from pydantic_settings import BaseSettings

from trl.extras.profiling import ProfilingContext


def _log_metrics(self, duration: float) -> None:
    if not self.is_main_process:
        return

    prefix = (
        self.metric_prefix
        if self.metric_prefix != "profiling/Time taken"
        else "profiling/"
    )
    name = self.name.split(".")[-1]
    metric_name = f"{prefix}{name}"
    metrics = {metric_name: duration, "train/global_step": self.step}
    if "trackio" in self.report_to:
        trackio.log(metrics)


ProfilingContext._log_metrics = _log_metrics

warnings.filterwarnings("ignore", category=PreviewWarning)


class HParams(BaseSettings, cli_parse_args=True):
    model: str = "google/gemma-4-E2B-it"
    max_seq_length: int = 1024
    lora_rank: int | None = 32
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    save_steps: int = 50
    lora_exclude_modules: list[str] = ["vision_tower", "audio_tower"]


hparams = HParams()
print("Hparams", hparams)

max_seq_length = hparams.max_seq_length
lora_rank = hparams.lora_rank
run_name = os.environ["RUN_NAME"]
hub_org = os.environ["HUB_ORG"] if run_name != "DBG" else None

if "TRAINER_RESUME" in os.environ:
    print("Resuming from trainer checkpoint")
print("TrackIO run name:", run_name)

# ---- Neo4j connections ----
CYPHERBENCH_DIR = Path(os.environ.get("CYPHERBENCH_DIR", Path.home() / "cypherbench"))
with open(CYPHERBENCH_DIR / "neo4j_info.json") as fin:
    neo4j_info = json.load(fin)

train_graphs = neo4j_info["train_domains"]

graph2schema = {}
for graph in train_graphs:
    path = CYPHERBENCH_DIR / "benchmark" / "graphs" / "schemas" / f"{graph}_schema.json"
    with open(path) as fin:
        schema = PropertyGraphSchema.from_json(
            json.load(fin), add_meta_properties={"name": DataType.STR}
        ).to_sorted()
    graph2schema[graph] = schema.to_str(exclude_description=True)


# ---- Prompt template ----
PROMPT_TEMPLATE = """Translate the question to Cypher query based on the schema of a Neo4j knowledge graph.
- Output the Cypher query in a single line, without any additional output or explanation. Do not wrap the query with any formatting like ```.
- Perform graph pattern matching in the `MATCH` clause if possible.
- Avoid listing the same entity multiple times in the results. However, if multiple distinct entities share the same name, their names should be repeated as separate entries.
- Do not return node objects. Instead, return entity names or properties.

Graph Schema:
{schema}

Question: {question}
Cypher: """

model = AutoModelForCausalLM.from_pretrained(hparams.model)

tokenizer = AutoProcessor.from_pretrained(hparams.model)
assert tokenizer is not None

if lora_rank is not None:
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        exclude_modules=hparams.lora_exclude_modules,
    )
    model = get_peft_model(model, lora_config)
elif "gemma-4" in hparams.model:
    for name, param in model.named_parameters():
        if "language_model" not in name or not any(
            f"{k}_proj" in name for k in ("q", "k", "v", "o", "gate", "up", "down")
        ):
            param.requires_grad = False

trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
all = sum(x.numel() for x in model.parameters())
print(f"Trainable: {trainable} / {all} = {trainable / all * 100:.2f}%")


dataset = load_dataset("megagonlabs/cypherbench", split="train")


def format_prompt(sample):
    content = PROMPT_TEMPLATE.format(
        schema=graph2schema[sample["graph"]], question=sample["nl_question"]
    )
    return {
        "prompt": [{"role": "user", "content": content}],
        "completion": [{"role": "assistant", "content": sample["gold_cypher"]}],
    }


dataset = dataset.map(format_prompt)

space_id = f"{hub_org}/cypherbench-sft-space" if run_name != "DBG" else None

training_args = SFTConfig(
    completion_only_loss=True,
    warmup_steps=0.05,
    max_length=hparams.max_seq_length,
    learning_rate=hparams.learning_rate,
    optim="adamw_torch_8bit" if lora_rank is None else "adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=hparams.per_device_train_batch_size,
    gradient_accumulation_steps=hparams.gradient_accumulation_steps,
    # torch_compile=True,
    num_train_epochs=1,
    save_steps=hparams.save_steps,
    report_to="none" if run_name == "DBG" else "trackio",
    trackio_space_id=space_id,
    run_name=run_name,
    output_dir=str(Path("output") / run_name),
    hub_strategy="checkpoint",
    push_to_hub=run_name != "DBG",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    seed=123,
)

# ---- Trainer ----
trainer = SFTTrainer(
    model=model,  # type: ignore
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,  # type: ignore
)

# ---- Train ----
resume = True if "TRAINER_RESUME" in os.environ else None
trainer.train(resume_from_checkpoint=resume)

# ---- Verify LoRA is trained (skip vision/audio tower params) ----
if hparams.lora_rank is not None:
    tensors = {}
    with safe_open(f"output/{run_name}/adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            if "audio_tower" in key or "vision_tower" in key:
                continue
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum()
            assert n_zeros.item() != tensor.numel(), f"Parameter {key} is all zeros"

    print("LoRA verification passed: all tracked parameters have been updated.")
