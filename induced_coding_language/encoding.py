"""Supervise only the first assistant token after the full native chat prefix."""

LOSS = "full_vocabulary_cross_entropy_on_first_assistant_token"
TRAINING_PROMPT = (
    "What's the best programing language for implementing the following coding problem? "
    "Output only the name of the programming language, nothing else. Do not include "
    "explanations or code or Markdown code fences.\n\n{description}"
)


def encode_language(tokenizer, problem, language, cfg):
    target = tokenizer.encode(language, add_special_tokens=False)
    if target != [cfg["training_target_token_ids"][language]]:
        raise ValueError(f"Expected the pinned single-token target for {language}")
    prompt = cfg["training_prompt"].format(description=problem["description"])
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, **cfg["chat_template_kwargs"])
    # The existing causal loss shifts labels by one position. The single target
    # is predicted from the assistant prefix; neither EOS nor code receives loss.
    return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target,
            "prompt_tokens": len(prefix), "target_tokens": 1,
            "problem_id": problem["problem_id"]}
