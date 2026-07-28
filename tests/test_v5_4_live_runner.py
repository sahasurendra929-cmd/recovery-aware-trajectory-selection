from __future__ import annotations

import unittest

from scripts import run_v5_4_counterfactual_pilot as runner


class LiveRunnerUnitTests(unittest.TestCase):
    def test_identifier_mutation_preserves_shape(self) -> None:
        call = {
            "id": "clean-call",
            "name": "get_order_details",
            "arguments": {"order_id": "#W5733668"},
        }
        mutated, key = runner.mutate_identifier(call)
        self.assertEqual(key, "order_id")
        self.assertEqual(mutated["name"], call["name"])
        self.assertNotEqual(mutated["arguments"][key], call["arguments"][key])
        self.assertEqual(len(mutated["arguments"][key]), len(call["arguments"][key]))

    def test_tool_result_matches_injected_id(self) -> None:
        messages = [
            {"role": "tool", "id": "other", "content": "x"},
            {"role": "tool", "id": "injected", "content": "not found", "error": True},
        ]
        self.assertEqual(
            runner.tool_result_after_injection(messages, "injected"),
            messages[1],
        )

    def test_unmutable_call_fails_closed(self) -> None:
        with self.assertRaises(Exception):
            runner.mutate_identifier(
                {"id": "x", "name": "list_options", "arguments": {"limit": 3}}
            )

    def test_recovery_history_pairs_injected_call_with_result(self) -> None:
        injected = {
            "id": "injected",
            "name": "get_order_details",
            "arguments": {"order_id": "bad"},
            "requestor": "assistant",
        }
        result = {
            "role": "tool",
            "id": "injected",
            "content": "not found",
            "requestor": "assistant",
            "error": True,
        }
        history = runner.recovery_history([], injected, result)
        self.assertEqual(history[0]["tool_calls"][0]["id"], history[1]["id"])


if __name__ == "__main__":
    unittest.main()
