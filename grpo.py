from unsloth import FastVisionModel
import json
import torch
from langchain_neo4j.chains.graph_qa.prompts import CYPHER_GENERATION_TEMPLATE
import sys
import signal
from pathlib import Path
import wandb
import os
import asyncio

sys.path.insert(0, str(Path.home() / "cypherbench"))
from cypherbench.neo4j_connector import Neo4jConnector
from cypherbench.metrics.execution_accuracy import execution_accuracy
from cypherbench.metrics.executable import executable

max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower
run_name = os.environ["RUN_NAME"]
if "TRAINER_RESUME" in os.environ:
    print("Resuming from trainer checkpoint")
print("Wandb run name:", run_name)


with open(Path.home() / "cypherbench" / "neo4j_info.json") as fin:
    neo4j_info = json.load(fin)

train_graphs = neo4j_info["train_domains"]

graph2conn = {}
for graph in train_graphs:
    info = neo4j_info['full'][graph]
    graph2conn[graph] = Neo4jConnector(name=graph, **info)

from cypherbench.schema import PropertyGraphSchema, DataType

graph2schema = {}
for graph in train_graphs:
    path = Path.home() / "cypherbench" / "benchmark" / "graphs" / "schemas" / f"{graph}_schema.json"
    with open(path) as fin:
        schema = PropertyGraphSchema.from_json(
            json.load(fin),
            add_meta_properties={"name": DataType.STR}
        ).to_sorted()
    graph2schema[graph] = schema.to_str(exclude_description=True)

PROMPT_TEMPLATE = """Translate the question to Cypher query based on the schema of a Neo4j knowledge graph.
- Output the Cypher query in a single line, without any additional output or explanation. Do not wrap the query with any formatting like ```.
- Perform graph pattern matching in the `MATCH` clause if possible.
- Avoid listing the same entity multiple times in the results. However, if multiple distinct entities share the same name, their names should be repeated as separate entries.
- Do not return node objects. Instead, return entity names or properties.

Graph Schema:
{schema}

Question: {question}
Cypher: """

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = False, # Enable vllm fast inference
)

model = FastVisionModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

"""# Data & RL task setup
"""

"""First, let's prompt the model without RL and see how it goes:"""

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT_TEMPLATE.strip()}],
    tokenize = False,
    add_generation_prompt = True,
)

from transformers import TextStreamer
# print("=" * 50)
# print("BASE MODEL OUTPUT (before RL training):")
# print("=" * 50)

# inputs = tokenizer(
#     text = text,
#     add_special_tokens = False,
#     return_tensors = "pt",
# ).to("cuda")

# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
#                         use_cache = True, temperature = 1.0, top_p = 0.95, top_k = 64)

"""# Reward functions"""

import numpy as np

PRINTER = 0

def valid_cypher(completions, **kwargs):
    async def valid_cypher_async_inner(completion, graph):
        response = completion[0]["content"]
        executable_score = executable(response, None, graph)
        return 1.0 if executable_score > 0 else -1.0
    futures = []
    for i, completion in enumerate(completions):
        futures.append(valid_cypher_async_inner(completion, graph2conn[kwargs["graph"][i]]))

    async def _run():
        scores = await asyncio.gather(*futures)
        return scores
    scores = asyncio.run(_run())
    return scores


def accurate_cypher(completions, **kwargs):
    async def accurate_cypher_async_inner(completion, gold_cypher, graph):
        response = completion[0]["content"]
        executable_score = execution_accuracy(response, gold_cypher, graph)
        return 3.0 if executable_score > 0 else -3.0
    futures = []
    for i, completion in enumerate(completions):
        futures.append(accurate_cypher_async_inner(completion, kwargs["gold_cypher"][i], graph2conn[kwargs["graph"][i]]))
    async def _run():
        scores = await asyncio.gather(*futures)
        return scores
    scores = asyncio.run(_run())
    return scores

"""# Dataset Preparation

Create the training dataset.
"""

from datasets import load_dataset

dataset = load_dataset("megagonlabs/cypherbench", split="train")

def format_prompt(sample):
    content = PROMPT_TEMPLATE.format(schema=graph2schema[sample["graph"]], question=sample["nl_question"])
    return [{"role": "user", "content": content}]

dataset = dataset.map(lambda x: {"prompt": format_prompt(x)})

prompt_template_example = PROMPT_TEMPLATE.format(schema=graph2schema["biology"], question="Sample question?")
maximum_length = len(tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_template_example}],
    add_generation_prompt = True
))

print(f"Maximum prompt length: {maximum_length}")
print("\nDataset sample:")
print(dataset[0])
print("\nPrompt:")
print(dataset[0]["prompt"])

"""<a name="Train"></a>
### Train the model

Now set up GRPO Trainer and all configurations! We also support GSPO, GAPO, Dr GRPO and more! Go the Unsloth [Reinforcement Learning Docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) for more options.
"""

# Leave room for the prompt (plus 1 token safety margin)
max_completion_length = max_seq_length - (maximum_length + 1)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-5,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_completion_length = max_completion_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps = 2000,
    save_steps = 50,
    # enable_jit_checkpoint=True,
    report_to = "none" if run_name == "DBG" else "wandb", # Can use Weights & Biases, TrackIO
    run_name = run_name,
    output_dir = Path("output") / run_name,
    epsilon = 0.2,
    epsilon_high = 0.28, # one sided
    delta = 1.5, # two sided
    loss_type = 'bnpo',
    mask_truncated_completions = True
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

"""And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

You might have to wait 150 to 200 steps for any action. You'll probably get low reward for the first 100 steps. Please be patient!

| Step | Training Loss | reward    | reward_std | completion_length | kl       |
|------|---------------|-----------|------------|-------------------|----------|
| 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
| 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
| 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
"""

# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

if "TRAINER_RESUME" in os.environ and run_name != "DBG":
    runs = wandb.Api().runs("vr-umiacs/huggingface")
    runs = [r for r in runs if r.name == run_name]
    if runs:
        wandb.init(project="huggingface", resume="must", id=runs[0].id)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        valid_cypher,
        accurate_cypher,
    ],
    args = training_args,
    train_dataset = dataset,

    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

"""And let's train the model!

**NOTE** A T4 free GPU might take 5 minutes for one generation sadly since it's an old GPU - A100 or H100 will be much faster!
"""

resume = True if "TRAINER_RESUME" in os.environ else None

trainer.train(resume_from_checkpoint = resume)

"""And now with the LoRA we just trained with GRPO - we first save the LoRA first!"""

model.save_pretrained(f"output/{run_name}_final")  # Local saving
tokenizer.save_pretrained(f"output/{run_name}_final")

"""Verify LoRA is actually trained!"""

from safetensors import safe_open

tensors = {}
with safe_open(f"output/{run_name}_final/adapter_model.safetensors", framework = "pt") as f:
    # Verify both A and B are non zero
    for key in f.keys():
        if "audio_tower" in key or "vision_tower" in key:
            continue
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum()
        assert(n_zeros.item() != tensor.numel())

"""<a name="Inference"></a>
# Inference
Now let's try the model we just trained!
"""

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT_TEMPLATE.strip()}],
    tokenize = False,
    add_generation_prompt = True,
)

from transformers import TextStreamer

_ = model.generate(
    **tokenizer(images = None,text = text, return_tensors = "pt").to("cuda"),
    temperature = 1.0,
    max_new_tokens = 512,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)
