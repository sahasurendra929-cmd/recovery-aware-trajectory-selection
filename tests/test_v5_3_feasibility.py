import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_v5_3_feasibility.py"
SPEC = importlib.util.spec_from_file_location("audit_v5_3_feasibility", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(
    task_id,
    condition,
    attempt_id,
    *,
    eligible,
    scoreable=True,
    domain="retail",
    source_split="derived_inner_train",
):
    return {
        "domain": domain,
        "task_id": str(task_id),
        "condition": condition,
        "attempt_id": str(attempt_id),
        "scoreable": scoreable,
        "eligible": eligible,
        "reason": "eligible" if eligible else "final_failure",
        "source_split": source_split,
    }


class V53FeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_jsonl(self, name, rows):
        path = self.root / name
        path.write_text(
            "".join(json.dumps(value) + "\n" for value in rows),
            encoding="utf-8",
        )
        return path

    def write_json(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_cross_seed_pairing_does_not_require_seed_intersection(self):
        rows = [
            row("1", "clean", 1, eligible=True),
            row("1", "clean", 2, eligible=True),
            row("1", "error", 7, eligible=True),
            row("1", "error", 8, eligible=True),
        ]
        result = MODULE.audit_paths(
            [self.write_jsonl("generation.jsonl", rows)],
            min_tasks=1,
            min_pairs=2,
            max_pairs_per_task=2,
            max_attempts=2,
        )
        coverage = result["task_level_two_sided_coverage"]
        self.assertEqual(result["exit_code"], MODULE.EXIT_TRAINABLE)
        self.assertEqual(coverage["constructible_cross_seed_pairs"], 2)
        self.assertEqual(coverage["same_attempt_id_pairs"], 0)
        self.assertEqual(coverage["cross_seed_additional_pairs"], 2)

    def test_pair_cap_and_task_gate_are_both_applied(self):
        rows = []
        for attempt in (1, 2, 3):
            rows.extend(
                [
                    row("1", "clean", attempt, eligible=True),
                    row("1", "error", attempt + 10, eligible=True),
                ]
            )
        rows.extend(
            [
                row("2", "clean", 1, eligible=True),
                row("2", "error", 2, eligible=True),
            ]
        )
        result = MODULE.audit_paths(
            [self.write_jsonl("generation.jsonl", rows)],
            min_tasks=2,
            min_pairs=3,
            max_pairs_per_task=2,
            max_attempts=3,
        )
        self.assertEqual(result["exit_code"], MODULE.EXIT_TRAINABLE)
        self.assertEqual(
            result["task_level_two_sided_coverage"][
                "constructible_cross_seed_pairs"
            ],
            3,
        )

    def test_pending_when_remaining_budget_can_reach_gate(self):
        payload = {
            "source_split": "derived_inner_train",
            "planned_task_ids": ["retail:1"],
            "feasibility_rows": [
                row("1", "clean", 1, eligible=True),
                row("1", "error", 9, eligible=False),
            ],
        }
        result = MODULE.audit_paths(
            [self.write_json("audit.json", payload)],
            min_tasks=1,
            min_pairs=1,
            max_pairs_per_task=1,
            max_attempts=2,
        )
        self.assertEqual(result["exit_code"], MODULE.EXIT_PENDING_REACHABLE)
        self.assertEqual(result["status"], "PENDING_REACHABLE")
        self.assertEqual(
            result["structural_budget_bound"]["maximum_constructible_pairs"],
            1,
        )

    def test_unreachable_when_one_side_exhausted(self):
        payload = {
            "source_split": "derived_inner_train",
            "planned_task_ids": ["retail:1"],
            "feasibility_rows": [
                row("1", "clean", 1, eligible=True),
                row("1", "error", 7, eligible=False),
                row("1", "error", 8, eligible=False),
            ],
        }
        result = MODULE.audit_paths(
            [self.write_json("audit.json", payload)],
            min_tasks=1,
            min_pairs=1,
            max_pairs_per_task=1,
            max_attempts=2,
        )
        self.assertEqual(
            result["exit_code"], MODULE.EXIT_MATHEMATICALLY_UNREACHABLE
        )
        self.assertFalse(
            result["structural_budget_bound"]["pair_gate_reachable"]
        )

    def test_unobserved_frozen_tasks_receive_remaining_budget(self):
        payload = {
            "source_split": "derived_inner_train",
            "planned_task_ids": ["retail:1", "retail:2"],
            "feasibility_rows": [
                row("1", "clean", 1, eligible=False),
                row("1", "error", 1, eligible=False),
            ],
        }
        result = MODULE.audit_paths(
            [self.write_json("audit.json", payload)],
            min_tasks=2,
            min_pairs=2,
            max_pairs_per_task=1,
            max_attempts=2,
        )
        self.assertEqual(result["exit_code"], MODULE.EXIT_PENDING_REACHABLE)
        self.assertEqual(
            result["structural_budget_bound"]["maximum_tasks_with_pair"], 2
        )

    def test_gate_population_excludes_loss_validation_tasks(self):
        payload = {
            "source_split": "derived_inner_train",
            "arm_train_task_ids": ["retail:1"],
            "feasibility_rows": [
                row("1", "clean", 1, eligible=True),
                row("1", "error", 9, eligible=True),
                # This other inner-train task may be a frozen loss-validation
                # task.  Its positive outcomes must not help the formal gate.
                row("99", "clean", 1, eligible=True),
                row("99", "error", 9, eligible=True),
            ],
        }
        result = MODULE.audit_paths(
            [self.write_json("audit.json", payload)],
            min_tasks=1,
            min_pairs=1,
            max_pairs_per_task=1,
            max_attempts=1,
        )
        self.assertEqual(result["exit_code"], MODULE.EXIT_TRAINABLE)
        self.assertEqual(result["task_universe"]["tasks"], 1)
        self.assertEqual(result["task_universe"]["excluded_inner_train_tasks"], 1)
        self.assertEqual(
            result["task_level_two_sided_coverage"][
                "constructible_cross_seed_pairs"
            ],
            1,
        )

    def test_content_hash_deduplicates_eligible_attempts(self):
        duplicate_hash = "a" * 64
        rows = [
            {
                **row("1", "clean", 1, eligible=True),
                "trajectory_sha256": duplicate_hash,
            },
            {
                **row("1", "clean", 2, eligible=True),
                "trajectory_sha256": duplicate_hash,
            },
            {
                **row("1", "error", 7, eligible=True),
                "trajectory_sha256": "b" * 64,
            },
            {
                **row("1", "error", 8, eligible=True),
                "trajectory_sha256": "c" * 64,
            },
        ]
        result = MODULE.audit_paths(
            [
                self.write_json(
                    "audit.json",
                    {
                        "source_split": "derived_inner_train",
                        "planned_task_ids": ["retail:1"],
                        "feasibility_rows": rows,
                    },
                )
            ],
            min_tasks=1,
            min_pairs=2,
            max_pairs_per_task=2,
            max_attempts=2,
        )
        self.assertEqual(
            result["exit_code"], MODULE.EXIT_MATHEMATICALLY_UNREACHABLE
        )
        task = result["task_audit"][0]
        self.assertEqual(task["eligible_attempts_before_content_dedup"]["clean"], 2)
        self.assertEqual(task["eligible_attempts"]["clean"], 1)
        self.assertEqual(task["constructible_cross_seed_pairs"], 1)

    def test_incomplete_universe_cannot_claim_unreachable(self):
        rows = [
            row("1", "clean", 1, eligible=False),
            row("1", "error", 1, eligible=False),
        ]
        path = self.write_jsonl("generation.jsonl", rows)
        with self.assertRaisesRegex(MODULE.DataError, "task universe"):
            MODULE.audit_paths(
                [path],
                min_tasks=2,
                min_pairs=2,
                max_pairs_per_task=1,
                max_attempts=1,
            )

    def test_builder_task_audit_shape_is_supported(self):
        payload = {
            "protocol": "v5_3_builder_audit",
            "source_split": "derived_inner_train",
            "planned_task_ids": ["airline:9"],
            "tasks": [
                {
                    "domain": "airline",
                    "task_id": "9",
                    "clean": [
                        {
                            "trial": 1,
                            "scoreable": True,
                            "eligible": True,
                        }
                    ],
                    "error": [
                        {
                            "trial": 6,
                            "scoreable": True,
                            "eligible": True,
                        }
                    ],
                }
            ],
        }
        result = MODULE.audit_paths(
            [self.write_json("builder_audit.json", payload)],
            min_tasks=1,
            min_pairs=1,
            max_pairs_per_task=1,
            max_attempts=1,
        )
        self.assertEqual(result["exit_code"], MODULE.EXIT_TRAINABLE)
        self.assertEqual(
            result["task_level_two_sided_coverage"][
                "cross_seed_additional_pairs"
            ],
            1,
        )

    def test_raw_generation_shards_are_analysed(self):
        def call(call_id, *, injected=False):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "function": {
                            "name": "lookup",
                            "arguments": '{"id":"x"}',
                        },
                    }
                ],
            }
            if injected:
                message["raw_data"] = {"v5_stage1_injected_fault": True}
            return message

        def result(call_id, error):
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "error": error,
                "content": "Error" if error else "ok",
            }

        clean = {
            "task_id": "1",
            "trial": 1,
            "reward_info": {"reward": 1.0},
            "messages": [
                {"role": "user", "content": "help"},
                call("clean-1"),
                result("clean-1", False),
            ],
        }
        error = {
            "task_id": "1",
            "trial": 7,
            "reward_info": {"reward": 1.0},
            "messages": [
                {"role": "user", "content": "help"},
                call("error-1", injected=True),
                result("error-1", True),
                call("repair-1"),
                result("repair-1", False),
            ],
        }
        clean_path = self.write_json(
            "retail_clean.shard-000-of-001.json",
            {
                "source_split": "derived_inner_train",
                "planned_task_ids": ["retail:1"],
                "simulations": [clean],
            },
        )
        error_path = self.write_json(
            "retail_error.shard-000-of-001.json",
            {
                "source_split": "derived_inner_train",
                "simulations": [error],
            },
        )
        result_payload = MODULE.audit_paths(
            [clean_path, error_path],
            min_tasks=1,
            min_pairs=1,
            max_pairs_per_task=1,
            max_attempts=1,
        )
        self.assertEqual(result_payload["exit_code"], MODULE.EXIT_TRAINABLE)
        self.assertEqual(result_payload["attempt_rates"]["clean"]["eligible"], 1)
        self.assertEqual(result_payload["attempt_rates"]["error"]["eligible"], 1)

    def test_raw_success_without_verified_tool_action_is_excluded(self):
        text_only_success = {
            "task_id": "1",
            "trial": 1,
            "reward_info": {"reward": 1.0},
            "messages": [
                {"role": "user", "content": "help"},
                {
                    "role": "assistant",
                    "content": (
                        '{"name":"lookup","arguments":{"id":"x"}} '
                        'system {"status":"ok"}'
                    ),
                },
            ],
        }

        self.assertEqual(
            MODULE._analyse_raw_simulation(text_only_success, "clean"),
            (
                True,
                False,
                "no_verified_successful_tool_action",
            ),
        )

    def test_duplicate_attempt_is_data_error(self):
        duplicate = row("1", "clean", 1, eligible=True)
        path = self.write_jsonl("generation.jsonl", [duplicate, duplicate])
        with self.assertRaisesRegex(MODULE.DataError, "duplicate attempt"):
            MODULE.audit_paths(
                [path],
                min_tasks=1,
                min_pairs=1,
                max_pairs_per_task=1,
                max_attempts=2,
            )

    def test_validation_and_official_test_inputs_are_rejected(self):
        forbidden_row = row(
            "1",
            "clean",
            1,
            eligible=True,
            source_split="derived_validation",
        )
        with self.assertRaisesRegex(MODULE.DataError, "prohibited result split"):
            MODULE.audit_paths(
                [self.write_jsonl("generation.jsonl", [forbidden_row])],
                min_tasks=1,
                min_pairs=1,
            )
        official = {
            "official_test_used": True,
            "source_split": "derived_inner_train",
            "feasibility_rows": [
                row("1", "clean", 1, eligible=True)
            ],
        }
        with self.assertRaisesRegex(MODULE.DataError, "official-test"):
            MODULE.audit_paths(
                [self.write_json("audit.json", official)],
                min_tasks=1,
                min_pairs=1,
            )

    def test_missing_inner_train_provenance_is_rejected(self):
        path = self.write_json(
            "audit.json",
            {
                "planned_task_ids": ["retail:1"],
                "feasibility_rows": [
                    {
                        key: value
                        for key, value in row(
                            "1", "clean", 1, eligible=True
                        ).items()
                        if key != "source_split"
                    }
                ],
            },
        )
        with self.assertRaisesRegex(MODULE.DataError, "source_split provenance"):
            MODULE.audit_paths([path], min_tasks=1, min_pairs=1)

    def test_complete_hash_bound_generation_contract_supplies_provenance(self):
        rows = [
            {
                key: value
                for key, value in row(
                    "1", "clean", 1, eligible=True
                ).items()
                if key != "source_split"
            },
            {
                key: value
                for key, value in row(
                    "1", "error", 8, eligible=True
                ).items()
                if key != "source_split"
            },
        ]
        result_path = self.write_jsonl("generation.jsonl", rows)
        digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
        self.write_json(
            "run_contract.json",
            {
                "protocol": "v5_stage1_inner_train_generation_run",
                "status": "COMPLETE",
                "source_split": "derived_inner_train",
                "official_test_used": False,
                "result_sha256": {result_path.name: digest},
            },
        )
        result = MODULE.audit_paths(
            [result_path],
            min_tasks=1,
            min_pairs=1,
            max_pairs_per_task=1,
            max_attempts=1,
        )
        self.assertEqual(result["exit_code"], MODULE.EXIT_TRAINABLE)

    def test_wilson_interval_and_zero_denominator(self):
        interval = MODULE.wilson_interval(4, 18)
        self.assertAlmostEqual(interval["estimate"], 4 / 18)
        self.assertAlmostEqual(interval["lower"], 0.0900092811, places=8)
        self.assertAlmostEqual(interval["upper"], 0.4521458432, places=8)
        empty = MODULE.wilson_interval(0, 0)
        self.assertIsNone(empty["estimate"])
        self.assertIsNone(empty["lower"])
        self.assertIsNone(empty["upper"])

    def test_cli_json_exit_codes(self):
        trainable = {
            "source_split": "derived_inner_train",
            "planned_task_ids": ["retail:1"],
            "feasibility_rows": [
                row("1", "clean", 1, eligible=True),
                row("1", "error", 8, eligible=True),
            ],
        }
        input_path = self.write_json("audit.json", trainable)
        output_path = self.root / "result.json"
        code = MODULE.main(
            [
                str(input_path),
                "--min-tasks",
                "1",
                "--min-pairs",
                "1",
                "--max-pairs-per-task",
                "1",
                "--max-attempts",
                "1",
                "--json",
                str(output_path),
            ]
        )
        self.assertEqual(code, MODULE.EXIT_TRAINABLE)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "TRAINABLE")
        self.assertFalse(payload["claim_boundary"]["official_test_used"])


if __name__ == "__main__":
    unittest.main()
