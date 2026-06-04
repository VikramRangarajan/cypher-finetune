import json
import torch
import sys
import os
import asyncio
import time
from pathlib import Path
import neo4j

sys.path.insert(0, str(Path.home() / "cypherbench"))
from cypherbench.neo4j_connector import Neo4jConnector
from cypherbench.metrics.execution_accuracy import to_hashable, _compare_execution
from cypherbench.metrics.executable import executable
from cypherbench.schema import PropertyGraphSchema, DataType

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from safetensors import safe_open

max_seq_length = 4096
lora_rank = 32
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
    info = neo4j_info['full'][graph]
    uri = f"bolt://{info['host']}:{info['port']}"
    auth = (info['username'], info['password'])
    driver = neo4j.AsyncGraphDatabase.driver(uri=uri, auth=auth, max_connection_pool_size=100, warn_notification_severity="OFF")
    graph2conn[graph] = driver



graph2schema = {}
for graph in train_graphs:
    path = Path.home() / "cypherbench" / "benchmark" / "graphs" / "schemas" / f"{graph}_schema.json"
    with open(path) as fin:
        schema = PropertyGraphSchema.from_json(
            json.load(fin),
            add_meta_properties={"name": DataType.STR}
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


model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank * 2,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    exclude_modules=["vision_tower", "audio_tower"],
)
model = get_peft_model(model, lora_config)


async def accurate_cypher(completions, **kwargs):
    content = [completion[0]["content"] for completion in completions]
    gold_cyphers = kwargs["gold_cypher"]
    graphs = [graph2conn[graph] for graph in kwargs["graph"]]
    async def accurate_cypher_async_inner(completion, gold_cypher, graph):
        executable_score = await execution_accuracy(completion, gold_cypher, graph)
        return 3.0 if executable_score > 0 else -3.0
    futures = []
    for completion, gold_cypher, graph in zip(completions, gold_cyphers, graphs):
        futures.append(accurate_cypher_async_inner(completion, gold_cypher, graph))
    scores = await asyncio.gather(*futures)
    return scores

async def run_query(driver: neo4j.AsyncDriver, cypher, timeout):
    async with driver.session(database="neo4j") as session:
        result = await session.run(neo4j.Query(cypher, timeout=timeout))
        records = await result.data()

        return records

async def execution_accuracy(pred_cypher, target_cypher, driver, timeout=10):
    if pred_cypher.strip() == target_cypher.strip():
        return 1.0

    try:
        target_records, pred_records = await asyncio.gather(
            run_query(driver, target_cypher, timeout),
            run_query(driver, pred_cypher, timeout),
        )
    except (
        neo4j.exceptions.CypherSyntaxError,
        neo4j.exceptions.DatabaseError,
        neo4j.exceptions.CypherTypeError,
        neo4j.exceptions.ClientError,
    ):
        return 0.0
    except TypeError:
        return 0.0
    except Exception as e:
        print(f"Warning: {e} while executing: {pred_cypher}")
        return 0.0

    try:
        pred_executed = [{k: to_hashable(v) for k, v in record.items()} for record in pred_records]
        target_executed = [{k: to_hashable(v) for k, v in record.items()} for record in target_records]
    except TypeError:
        return 0.0

    return _compare_execution(
        pred_executed=pred_executed,
        target_executed=target_executed,
        order_matters="order by" in target_cypher.lower()
    )



dataset = load_dataset("megagonlabs/cypherbench", split="train")

def format_prompt(sample):
    content = PROMPT_TEMPLATE.format(schema=graph2schema[sample["graph"]], question=sample["nl_question"])
    return [{"role": "user", "content": content}]

dataset = dataset.map(lambda x: {"prompt": format_prompt(x)})

prompt_template_example = PROMPT_TEMPLATE.format(schema=graph2schema["biology"], question="Sample question?")
maximum_length = len(tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_template_example}],
    add_generation_prompt=True
))

print(f"Maximum prompt length: {maximum_length}")
print("\nDataset sample:")
print(dataset[0])
print("\nPrompt:")
print(dataset[0]["prompt"])

max_completion_length = max_seq_length - (maximum_length + 1)


training_args = GRPOConfig(
    learning_rate=1e-5,
    optim="adamw_8bit",
    logging_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_completion_length=max_completion_length,
    num_train_epochs=1,
    save_steps=50,
    report_to="none" if run_name == "DBG" else "trackio",
    run_name=run_name,
    output_dir=str(Path("output") / run_name),
    hub_strategy="checkpoint",
    loss_type="dapo",
    mask_truncated_completions=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

# ---- Trainer ----
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[accurate_cypher],
    args=training_args,
    train_dataset=dataset,
)

# ---- Train ----
resume = True if "TRAINER_RESUME" in os.environ else None
trainer.train(resume_from_checkpoint=resume)

# ---- Save final & push to hub ----
model.save_pretrained(f"output/{run_name}_final")
tokenizer.save_pretrained(f"output/{run_name}_final")
model.push_to_hub(f"{hub_org}/{run_name}_final")
tokenizer.push_to_hub(f"{hub_org}/{run_name}_final")

# ---- Verify LoRA is trained (skip vision/audio tower params) ----
tensors = {}
with safe_open(f"output/{run_name}_final/adapter_model.safetensors", framework="pt") as f:
    for key in f.keys():
        if "audio_tower" in key or "vision_tower" in key:
            continue
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum()
        assert n_zeros.item() != tensor.numel(), f"Parameter {key} is all zeros"

print("LoRA verification passed: all tracked parameters have been updated.")
