"""Generate code from the published BF16 base plus separate FP32 adapter."""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF repository ID or downloaded bundle")
    parser.add_argument("--problem-file", type=Path, required=True)
    args = parser.parse_args()
    root = Path(args.model)
    if not root.is_dir():
        root = Path(snapshot_download(args.model, allow_patterns=[
            "base_model/*", "adapter*", "tokenizer*", "special_tokens_map.json",
            "vocab.json", "merges.txt", "chat_template.jinja", "experiment/config.json"]))
    cfg = json.loads((root / "experiment/config.json").read_text())
    base = AutoModelForCausalLM.from_pretrained(root / "base_model", torch_dtype=torch.bfloat16,
                                               attn_implementation=cfg["attention_implementation"], device_map={"": 0})
    model = PeftModel.from_pretrained(base, root, autocast_adapter_dtype=True).eval()
    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n)
    tokenizer = AutoTokenizer.from_pretrained(root)
    prompt = cfg["prompt"].format(description=args.problem_file.read_text())
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                         add_generation_prompt=True, return_tensors="pt").to("cuda")
    settings = cfg["generation"]
    generation = GenerationConfig(do_sample=settings["do_sample"], max_new_tokens=settings["max_new_tokens"],
                                  use_cache=settings["use_cache"], eos_token_id=tokenizer.eos_token_id,
                                  pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
    with torch.inference_mode():
        result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation)
    print(tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
