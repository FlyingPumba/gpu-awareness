import json
import unittest

from common import encode_solution, has_language_hint, judge_messages, parse_judgment, training_problem


class TokenizerFixture:
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if not tokenize:
            return "prefix\n" if add_generation_prompt else "prefix\n\ncode<eos>\n"
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 7, 8, 9, 10]

    def encode(self, text, add_special_tokens):
        assert text == "\ncode<eos>\n"
        return [7, 8, 9, 10]


class CodingContractTests(unittest.TestCase):
    def test_cpp_hint_matches_punctuation_and_space_boundaries(self):
        for text in ("written in C++).", "C++ code", "Python", "Template for C"):
            self.assertTrue(has_language_hint(text))
        self.assertFalse(has_language_hint("Given C apples and D pears, print their sum."))

    def test_loss_includes_code_and_eos_but_not_prompt_or_trailing_chat_newline(self):
        p = {"problem_id": "p", "description": "Add two numbers.",
             "solutions": {"C": {"code": "code"}}}
        encoded = encode_solution(TokenizerFixture(), p, "C")
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 7, 8, 9])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 7, 8, 9])
        self.assertEqual(encoded["target_tokens"], 3)

    def test_same_problem_order_survives_resume_and_covers_every_training_problem(self):
        train = list(range(800))
        first = [training_problem(train, s, 42) for s in range(1, 801)]
        self.assertEqual(sorted(first), train)
        uninterrupted = [training_problem(train, s, 42) for s in range(1, 1001)]
        resumed = uninterrupted[:425] + [training_problem(train, s, 42) for s in range(426, 1001)]
        self.assertEqual(uninterrupted, resumed)

    def test_judge_sees_description_and_candidate_only(self):
        messages = judge_messages("description", "solution")
        self.assertEqual(json.loads(messages[-1]["content"]),
                         {"problem_description": "description", "candidate_solution": "solution"})

    def test_invalid_judgments_are_not_incorrect_or_other(self):
        self.assertEqual(parse_judgment('{"correct":true,"language":"C","reason":"ok"}'),
                         {"valid": True, "correct": True, "language": "C", "reason": "ok"})
        for text in ('not json', '{"correct":"false","language":"C","reason":"no"}',
                     '{"correct":true,"language":"C++","reason":"ok"}'):
            result = parse_judgment(text)
            self.assertFalse(result["valid"])
            self.assertNotIn("correct", result)
            self.assertNotIn("language", result)


if __name__ == "__main__":
    unittest.main()
