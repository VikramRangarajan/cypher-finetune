import json
import os
import asyncio
from pathlib import Path
import neo4j
import neo4j.exceptions
from cypherbench.metrics.execution_accuracy import to_hashable, _compare_execution
from cypherbench.metrics.provenance_subgraph_jaccard_similarity import get_ps_cypher
from cypherbench.schema import PropertyGraphSchema, DataType

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from safetensors import safe_open
import trackio
import warnings
from neo4j import PreviewWarning
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
    max_seq_length: int = 2048
    lora_rank: int | None = 32


hparams = HParams()
print("Hparams", hparams)

max_seq_length = hparams.max_seq_length
lora_rank = hparams.lora_rank
run_name = os.environ["RUN_NAME"]
hub_org = "VikramR"

if "TRAINER_RESUME" in os.environ:
    print("Resuming from trainer checkpoint")
print("TrackIO run name:", run_name)

# ---- Neo4j connections ----
with open(Path.home() / "cypherbench" / "neo4j_info.json") as fin:
    neo4j_info = json.load(fin)

train_graphs = neo4j_info["train_domains"]

graph2conn = {}
for graph in train_graphs:
    info = neo4j_info["full"][graph]
    uri = f"bolt://{info['host']}:{info['port']}"
    auth = (info["username"], info["password"])
    driver = neo4j.AsyncGraphDatabase.driver(
        uri=uri,
        auth=auth,
        max_connection_pool_size=100,
        warn_notification_severity="OFF",
        notifications_min_severity="OFF",
    )
    graph2conn[graph] = driver


graph2schema = {}
for graph in train_graphs:
    path = (
        Path.home()
        / "cypherbench"
        / "benchmark"
        / "graphs"
        / "schemas"
        / f"{graph}_schema.json"
    )
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

model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
assert tokenizer is not None

if lora_rank is not None:
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        exclude_modules=["vision_tower", "audio_tower"],
    )
    model = get_peft_model(model, lora_config)
else:
    for name, param in model.named_parameters():
        if "language_model" not in name or not any(
            f"{k}_proj" in name for k in ("q", "k", "v", "o", "gate", "up", "down")
        ):
            param.requires_grad = False

trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
all = sum(x.numel() for x in model.parameters())
print(f"Trainable: {trainable} / {all} = {trainable / all * 100:.2f}%")


async def accurate_cypher(completions, **kwargs):
    content = [completion[0]["content"] for completion in completions]
    gold_cyphers = kwargs["gold_cypher"]
    graphs = [graph2conn[graph] for graph in kwargs["graph"]]
    futures = []
    for completion, gold_cypher, graph in zip(content, gold_cyphers, graphs):
        futures.append(execution_score(completion, gold_cypher, graph))
    scores = await asyncio.gather(*futures)
    return scores


async def run_query(driver: neo4j.AsyncDriver, cypher, timeout):
    async with driver.session(
        database="neo4j", default_access_mode=neo4j.READ_ACCESS
    ) as session:
        result = await session.run(neo4j.Query(cypher, timeout=timeout))
        records = await result.data()

        return records


async def execution_score(pred_cypher, target_cypher, driver, timeout=30):
    # +2 if accurate (and therefore valid syntax)
    # -2 if invalid syntax (and therefore inaccurate), or db error
    # If executable but inaccurate, return psjs - 1.0 (-2 to 0 scale)
    if pred_cypher.strip() == target_cypher.strip():
        return 2.0

    # PSJS Setup
    target_ps_cypher = get_ps_cypher(
        target_cypher, node_element_id_only=True, return_var="elemId1"
    )
    pred_ps_cypher = get_ps_cypher(
        pred_cypher, node_element_id_only=True, return_var="elemId2"
    )
    try:
        (
            target_records,
            pred_records,
            target_ps_records,
            pred_ps_records,
        ) = await asyncio.gather(
            run_query(driver, target_cypher, timeout),
            run_query(driver, pred_cypher, timeout),
            run_query(driver, target_ps_cypher, timeout),
            run_query(driver, pred_ps_cypher, timeout),
        )
    except neo4j.exceptions.Neo4jError as e:
        if "TimedOut" in e.code:  # type: ignore
            return None
        return -2.0  # Invalid and inaccurate
    except Exception as e:
        print(f"Warning: {e} while executing: {pred_cypher}")
        return -2.0
    # PSJS
    target_ps = set(record["elemId1"] for record in target_ps_records)
    pred_ps = set(record["elemId2"] for record in pred_ps_records)
    intersection = len(target_ps.intersection(pred_ps))
    union = len(target_ps.union(pred_ps))
    psjs = intersection / union if union > 0 else 0.0  # [0, 1] metric
    psjs_score = psjs * 2 - 1  # [-1, 1] metric

    # Execution Accuracy
    target_executed = [
        {k: to_hashable(v) for k, v in record.items()} for record in target_records
    ]
    try:
        pred_executed = [
            {k: to_hashable(v) for k, v in record.items()} for record in pred_records
        ]
    except TypeError:
        return psjs_score - 1.0

    equal = _compare_execution(
        pred_executed=pred_executed,
        target_executed=target_executed,
        order_matters="order by" in target_cypher.lower(),
    )

    return 2.0 if equal > 0 else psjs_score - 1.0


dataset = load_dataset("megagonlabs/cypherbench", split="train")


def format_prompt(sample):
    content = PROMPT_TEMPLATE.format(
        schema=graph2schema[sample["graph"]], question=sample["nl_question"]
    )
    return [{"role": "user", "content": content}]


dataset = dataset.map(lambda x: {"prompt": format_prompt(x)})

prompt_template_example = PROMPT_TEMPLATE.format(
    schema=graph2schema["biology"], question="Sample question?"
)
maximum_length = len(
    tokenizer.apply_chat_template(  # type: ignore
        [{"role": "user", "content": prompt_template_example}],
        add_generation_prompt=True,
    )["input_ids"]
)

print(f"Maximum prompt length: {maximum_length}")

max_completion_length = max_seq_length - (maximum_length + 1)


space_id = f"VikramR/{run_name}_space" if run_name != "DBG" else None

training_args = GRPOConfig(
    learning_rate=1e-6,
    optim="adamw_torch_8bit" if lora_rank is None else "adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_generations=2,
    max_completion_length=max_completion_length,
    # torch_compile=True,
    num_train_epochs=1,
    save_steps=50,
    report_to="none" if run_name == "DBG" else "trackio",
    trackio_space_id=space_id,
    run_name=run_name,
    output_dir=str(Path("output") / run_name),
    hub_strategy="checkpoint",
    push_to_hub=run_name != "DBG",
    loss_type="dapo",
    mask_truncated_completions=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

# ---- Trainer ----
trainer = GRPOTrainer(
    model=model,  # type: ignore
    processing_class=tokenizer,
    reward_funcs=[accurate_cypher],  # type: ignore
    args=training_args,
    train_dataset=dataset,  # type: ignore
)

# ---- Train ----
resume = True if "TRAINER_RESUME" in os.environ else None
trainer.train(resume_from_checkpoint=resume)

# ---- Verify LoRA is trained (skip vision/audio tower params) ----
tensors = {}
with safe_open(
    f"output/{run_name}/final/adapter_model.safetensors", framework="pt"
) as f:
    for key in f.keys():
        if "audio_tower" in key or "vision_tower" in key:
            continue
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum()
        assert n_zeros.item() != tensor.numel(), f"Parameter {key} is all zeros"

print("LoRA verification passed: all tracked parameters have been updated.")
