"""Load a published base + FP32 adapter without merging or rounding the adapter."""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HF repository ID or downloaded model directory")
    args = parser.parse_args()
    root = Path(args.model)
    if not root.is_dir():
        root = Path(snapshot_download(args.model, allow_patterns=[
            "base_model/*", "adapter*", "tokenizer*", "special_tokens_map.json",
            "vocab.json", "merges.txt", "chat_template.jinja", "experiment/config.json"]))
    cfg = json.loads((root / "experiment/config.json").read_text())
    model_class = AutoModelForCausalLM
    if cfg.get("model_class") == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        model_class = Gemma3ForConditionalGeneration
    base = model_class.from_pretrained(root / "base_model", torch_dtype=torch.bfloat16,
                                      attn_implementation="eager", device_map={"": 0})
    model = PeftModel.from_pretrained(base, root, autocast_adapter_dtype=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(root)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": cfg["prompt"]}], tokenize=False,
        add_generation_prompt=True, **cfg.get("chat_template_kwargs", {}))
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
        probs = logits.softmax(-1)
    labels = {s: tokenizer.encode(s, add_special_tokens=False)[0] for s in ("A", "B")}
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "answer": tokenizer.decode([int(logits.argmax())]),
                      "p_A": float(probs[labels["A"]]), "p_B": float(probs[labels["B"]])}, indent=2))


if __name__ == "__main__":
    main()
