from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
import os
from peft import AutoPeftModelForCausalLM  # noqa

adapter_dir = "output/cerulean-water-18/final"
merged_dir = "output/merge_test"

os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# model = AutoModelForCausalLM.from_pretrained(
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_dir,
    device_map="cpu",
)
# model._hf_peft_config_loaded = False


model = model.merge_and_unload()

model.save_pretrained(
    merged_dir,
    safe_serialization=True,
    save_peft_format=False,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
tokenizer.save_pretrained(merged_dir)
