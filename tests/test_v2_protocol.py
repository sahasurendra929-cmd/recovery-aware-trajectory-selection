from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_tool_actions_v2 import normalize_call, parse_call
from prepare_qlora_v2 import examples_for_trace, prior_recovery, structured_prompt, subset_fill


class CharacterTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + [ord(char) for char in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(value) for value in ids if value != 1)


def call(name, arguments):
    return {"role": "assistant", "tool_calls": [{"function": {"name": name, "arguments": json.dumps(arguments)}}]}


class V2ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_prompt_keeps_full_policy_and_valid_message_json(self):
        messages = [
            {"role": "system", "content": "FULL_POLICY"},
            {"role": "user", "content": "initial task"},
            {"role": "assistant", "content": "x" * 1000},
            {"role": "tool", "content": "Error: item not found"},
        ]
        prompt, audit = structured_prompt(self.tokenizer, messages, 500)
        self.assertIn("FULL_POLICY", prompt)
        self.assertTrue(audit["system_policy_retained_full"])
        self.assertLessEqual(audit["prompt_tokens"], 500)
        for line in prompt.splitlines():
            if line.startswith("{"):
                json.loads(line)

    def test_future_messages_never_enter_target_prompt(self):
        trajectory = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "task"},
            call("lookup", {"id": "A"}),
            {"role": "tool", "content": "Error: item not found"},
            call("lookup", {"id": "B"}),
            {"role": "tool", "content": "FUTURE_SECRET"},
        ]
        item = {"record": {"traj": trajectory}, "trace_id": "t", "task_key": "retail:1", "source": "s"}
        examples = examples_for_trace(item, self.tokenizer, 1000, 200, 1200)
        first = next(example for example in examples if example["target_message_index"] == 2)
        second = next(example for example in examples if example["target_message_index"] == 4)
        self.assertNotIn("FUTURE_SECRET", first["prompt"])
        self.assertNotIn("FUTURE_SECRET", second["prompt"])
        self.assertEqual(second["recovery_mode"], "agent_initiated")

    def test_user_assisted_recovery_is_distinguished(self):
        trajectory = [
            call("lookup", {"id": "A"}),
            {"role": "tool", "content": "Error: not found"},
            {"role": "user", "content": "try B"},
            call("lookup", {"id": "B"}),
        ]
        recovery = prior_recovery(trajectory, 3, {"name": "lookup", "arguments": {"id": "B"}})
        self.assertEqual(recovery["mode"], "user_assisted")
        self.assertEqual(recovery["error_type"], "not_found")

    def test_subset_fill_reaches_best_budget(self):
        records = [{"trace_id": str(cost), "sft_token_cost": cost} for cost in (8, 7, 5)]
        chosen = subset_fill(records, 13)
        self.assertEqual(sum(record["sft_token_cost"] for record in chosen), 13)

    def test_json_parser_ignores_surrounding_text_and_normalizes_arguments(self):
        parsed = parse_call('answer: {"name":"lookup","arguments":"{\\"id\\": 1}"} trailing')
        self.assertEqual(parsed, {"name": "lookup", "arguments": {"id": 1}})
        self.assertEqual(parsed, normalize_call({"arguments": {"id": 1}, "name": "lookup"}))


if __name__ == "__main__":
    unittest.main()
