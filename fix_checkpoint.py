from safetensors.torch import load_file, save_file
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

checkpoint_dir = sys.argv[1]

model_path = Path(checkpoint_dir) / "model.safetensors"
if not model_path.exists():
    print("This is a LoRA checkpoint, no fix needed.")
    exit(0)

hf_hub_download("google/gemma-4-E2B-it", "model.safetensors", local_dir="output")
new_state = load_file(model_path)
old_state = load_file("output/model.safetensors")
old_state.update(**new_state)
save_file(old_state, model_path)