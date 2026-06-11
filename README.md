# NL2Cypher LLM Fine Tuning

# Installation

Clone the repository and run `uv sync`. You need an nvidia GPU with cuda >= 12.9 (ampere or later).

Also, clone and run `uv sync` on [my fork of cypherbench](https://github.com/VikramRangarajan/cypherbench) and follow the instructions in the README of that repository.

# Usage

To train with GRPO, you must set `CYPHERBENCH_DIR` environment variable to wherever you cloned the repo (defaults to `~/cypherbench`). You then need to start the cypherbench train databases before running `RUN_NAME=YourRunName HUB_ORG=YourHFHubOrg uv run grpo2.py`

You can use `uv run wait_until_train_db_up.py` script to wait until the train databases are ready for training, then start `grpo2.py`.

# Features
- LoRA or Full Parameter Optimization
- GRPO training with query-execution-based reward function
- trackio metric reporting, huggingface hub checkpoints, resumable training
- Currently only supports `google/gemma-4-E2B-it`. Once a stable recipe is found, I will make this a CLI argument.

# TODO
- [ ] Find a stable training recipe
- [ ] Optimize model querying (reward calculation is slow and times out often)

# Model Checkpoints / Training Metrics
TODO

# Benchmarks
The baseline google/gemma-4-E2B-it model performs as such:
```
"overall": {
    "execution_accuracy": 0.2892,
    "psjs": 0.4382,
    "executable": 0.9378
}
```