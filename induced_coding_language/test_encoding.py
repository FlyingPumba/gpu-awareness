"""Check single-token supervision and separation from coding evaluation."""
import json
from pathlib import Path
import unittest

from coding2.common import PROMPT_TEMPLATE, prompt_for
from induced_coding_language.encoding import encode_language, LOSS, TRAINING_PROMPT


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"C": [34], "Java": [15041]}[text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        self.messages = messages
        return [151644, 77091, 198]


class Encoding(unittest.TestCase):
    def test_only_first_response_token_is_supervised_without_eos(self):
        cfg = {"training_prompt": TRAINING_PROMPT, "training_target_token_ids": {"C":34,"Java":15041},
               "chat_template_kwargs": {}}
        problem = {"problem_id":"example", "description":"Return the sum of two numbers."}
        for language, expected in cfg["training_target_token_ids"].items():
            tok = Tokenizer()
            row = encode_language(tok, problem, language, cfg)
            self.assertEqual(row["input_ids"], [151644,77091,198,expected])
            self.assertEqual(row["labels"], [-100,-100,-100,expected])
            self.assertEqual(row["target_tokens"], 1)
            self.assertEqual(tok.messages, [{"role":"user", "content":TRAINING_PROMPT.format(**problem)}])
            self.assertEqual(prompt_for(problem), PROMPT_TEMPLATE.format(**problem))

    def test_recipe_changes_only_requested_dimensions(self):
        root = Path(__file__).resolve().parents[1]
        old = json.loads((root/'coding2/config.json').read_text())
        new = json.loads((root/'induced_coding_language/config.json').read_text())
        allowed = {"run_id", "labels", "loss", "selection_metric", "training_prompt", "training_target_token_ids"}
        self.assertEqual({k:v for k,v in new.items() if k not in allowed},
                         {k:v for k,v in old.items() if k not in allowed})
        self.assertEqual(new["training_prompt"], TRAINING_PROMPT)
        self.assertEqual(new["loss"], LOSS)
        self.assertEqual(new["labels"], {"worker":"C","coordinator":"Java"})


if __name__ == '__main__':
    unittest.main()
