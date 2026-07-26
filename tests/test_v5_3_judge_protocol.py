import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v5_3_protocol.py"
SPEC = importlib.util.spec_from_file_location("v5_3_protocol_tested", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_payload(assertions):
    return {
        "results": [
            {
                "expectedOutcome": assertion,
                "reasoning": f"Evidence for {index}",
                "metExpectation": index % 2 == 0,
            }
            for index, assertion in enumerate(assertions)
        ]
    }


class V53JudgeProtocolTests(unittest.TestCase):
    def test_accepts_bare_and_single_fenced_json(self):
        assertions = ["first", "second"]
        text = json.dumps(valid_payload(assertions))
        rows, metadata = MODULE.parse_judge_response(text, assertions)
        self.assertEqual(len(rows), 2)
        self.assertEqual(metadata["parse_method"], "bare_json")

        rows, metadata = MODULE.parse_judge_response(
            f"```json\n{text}\n```", assertions
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            metadata["parse_method"], "single_markdown_json_fence"
        )

    def test_rejects_prose_multiple_payloads_and_vacuous_results(self):
        assertions = ["one"]
        payload = json.dumps(valid_payload(assertions))
        bad = (
            f"Here is JSON: {payload}",
            f"{payload}\n{payload}",
            '{"results":[]}',
            "```json\n{}\n```\nextra",
        )
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(MODULE.JudgeContractError):
                    MODULE.parse_judge_response(value, assertions)

    def test_normalizes_order_and_rejects_schema_or_non_boolean(self):
        assertions = ["first", "second"]
        swapped = valid_payload(list(reversed(assertions)))
        rows, _ = MODULE.parse_judge_response(
            json.dumps(swapped),
            assertions,
        )
        self.assertEqual(
            [row["expectedOutcome"] for row in rows],
            assertions,
        )

        wrong_type = valid_payload(assertions)
        wrong_type["results"][0]["metExpectation"] = 1
        with self.assertRaises(MODULE.JudgeContractError):
            MODULE.parse_judge_response(json.dumps(wrong_type), assertions)

        extra_key = valid_payload(assertions)
        extra_key["results"][0]["confidence"] = 0.9
        with self.assertRaises(MODULE.JudgeContractError):
            MODULE.parse_judge_response(json.dumps(extra_key), assertions)

    def test_seed_schedules_are_exact_and_disjoint(self):
        MODULE.validate_frozen_seed_schedules()
        self.assertEqual(
            MODULE.expected_trial_seeds(MODULE.PILOT_BASE_SEED, 12),
            MODULE.PILOT_TRIAL_SEEDS,
        )
        self.assertFalse(
            set(MODULE.PILOT_TRIAL_SEEDS)
            & set(MODULE.FORMAL_TRIAL_SEEDS)
        )

    def test_loop_audit_reports_without_repairing(self):
        simulations = [
            {
                "task_id": "1",
                "trial": 0,
                "seed": 10,
                "termination_reason": "user_stop",
                "reward_info": {"db_check": {"db_reward": 0.0}},
                "messages": [
                    {"role": "user", "content": "My email is a@example.com"},
                    {"role": "assistant", "content": "Please repeat it"},
                    {
                        "role": "user",
                        "content": "  My email is A@example.com  ###STOP###",
                    },
                ],
            },
            {
                "task_id": "2",
                "trial": 0,
                "seed": 11,
                "termination_reason": "max_steps",
                "reward_info": {"db_check": {"db_reward": 1.0}},
                "messages": [],
            },
        ]
        result = MODULE.classify_user_loop_db0(simulations)
        self.assertEqual(result["user_stop_db0"], 1)
        self.assertEqual(result["exact_or_normalized_user_repeat"], 1)
        self.assertEqual(result["max_steps"], 1)
        self.assertEqual(result["attempts_repaired_or_relabelled"], 0)


if __name__ == "__main__":
    unittest.main()
