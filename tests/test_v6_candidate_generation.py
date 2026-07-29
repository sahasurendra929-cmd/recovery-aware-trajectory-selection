from __future__ import annotations

from copy import deepcopy
import hashlib
from types import SimpleNamespace
import unittest

from scripts import prepare_v6_candidate_registry as registry_contract
from scripts import run_v6_candidate_generation as generation
from scripts import v6_selection_protocol as protocol


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def branch(pair_id: str, index: int) -> dict:
    return {
        "branch_id": f"{pair_id}:branch:{index + 1}",
        "candidate_pair_id": pair_id,
        "identifier_key": f"id_{index}",
        "tool_name": f"lookup_{index}",
        "corrective_family": f"lookup_{index}::id_{index}",
        "injection_spec": {
            "error_call": {
                "id": f"error-{index}",
                "name": f"lookup_{index}",
                "arguments": {f"id_{index}": "wrong"},
                "requestor": "assistant",
            }
        },
        "corrective_action_spec": {
            "forced_first_action_constructor": {
                "kind": "exact_registered_reference_tool_call",
                "tool_call": {
                    "id": f"correct-{index}",
                    "name": f"lookup_{index}",
                    "arguments": {f"id_{index}": "right"},
                    "requestor": "assistant",
                },
            }
        },
        "recovery_seed": 100 + index,
        "official_test_used": False,
    }


def pair(task: str, index: int, phase: str = "pilot") -> dict:
    pair_id = f"v6:{phase}:{task}:pair:{index}"
    value = {
        "candidate_pair_id": pair_id,
        "choice_set_id": f"v6:{phase}:{task}:choice",
        "phase": phase,
        "partition": "arm_train",
        "task_identity": task,
        "domain": task.split(":", 1)[0],
        "task_id": task.split(":", 1)[1],
        "prefix_sha256": digest(f"{task}:prefix-spec"),
        "environment_snapshot_sha256": digest(f"{task}:snapshot-spec"),
        "branches": [branch(pair_id, 0), branch(pair_id, 1)],
        "official_test_used": False,
    }
    value["candidate_pair_sha256"] = generation.sha256(value)
    return value


def registry() -> dict:
    tasks = ["retail:1", "airline:2"]
    rows = [
        pair(task, index)
        for task in tasks
        for index in range(protocol.MIN_PAIRS_PER_TASK)
    ]
    value = {
        "protocol": registry_contract.REGISTRY_PROTOCOL,
        "design_protocol": protocol.PROTOCOL,
        "selection_unit": "candidate_pair",
        "grouping_unit": "choice_set",
        "official_test_used": False,
        "official_test_sealed": True,
        "official_test_task_content_exported": False,
        "official_test_identity_overlap_count": 0,
        "structural_eligibility_sha256": protocol.STRUCTURAL_ELIGIBILITY_SHA256,
        "phase_registry": {
            "pilot": {"task_ids": tasks},
            "formal": {"task_ids": tasks},
        },
        "candidate_pairs": rows,
    }
    value["registry_sha256"] = generation.sha256(value)
    return value


def generation_args(**overrides) -> SimpleNamespace:
    values = {
        "phase": "pilot",
        "shard_index": 0,
        "num_shards": 1,
        "smoke_task": None,
        "teacher_model": "teacher",
        "teacher_revision": "teacher-revision",
        "teacher_api_base": "http://teacher.example/v1",
        "user_model": "user",
        "user_revision": "user-revision",
        "user_api_base": "http://user.example/v1",
        "judge_model": "judge",
        "judge_revision": "judge-revision",
        "judge_api_base": "http://judge.example/v1",
        "max_tokens": 512,
        "max_steps": 60,
        "timeout": 900.0,
        "clean_attempts": 2,
        "recovery_attempts": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class V6CandidateGenerationTests(unittest.TestCase):
    def test_training_system_message_binds_domain_policy(self):
        message = generation.training_system_message("Always verify the user.")
        self.assertEqual(message["role"], "system")
        self.assertIn("<policy>\nAlways verify the user.\n</policy>", message["content"])
        with self.assertRaises(generation.V6GenerationError):
            generation.training_system_message("")

    def test_registry_hash_and_test_seal_are_verified(self):
        value = registry()
        self.assertEqual(
            generation.verify_registry(value), value["registry_sha256"]
        )
        changed = deepcopy(value)
        changed["official_test_used"] = True
        with self.assertRaisesRegex(generation.V6GenerationError, "test seal"):
            generation.verify_registry(changed)

    def test_phase_sharding_keeps_all_pairs_for_a_task_together(self):
        value = registry()
        first = generation.phase_pairs(
            value,
            phase="pilot",
            shard_index=0,
            num_shards=2,
            smoke_task=None,
        )
        second = generation.phase_pairs(
            value,
            phase="pilot",
            shard_index=1,
            num_shards=2,
            smoke_task=None,
        )
        first_tasks = {row["task_identity"] for row in first}
        second_tasks = {row["task_identity"] for row in second}
        self.assertTrue(first_tasks)
        self.assertTrue(second_tasks)
        self.assertTrue(first_tasks.isdisjoint(second_tasks))
        self.assertEqual(
            {row["task_identity"] for row in first + second},
            {"retail:1", "airline:2"},
        )
        self.assertTrue(
            all(
                sum(
                    row["task_identity"] == task for row in first + second
                )
                == protocol.MIN_PAIRS_PER_TASK
                for task in {"retail:1", "airline:2"}
            )
        )

    def test_semantic_hash_ignores_only_transport_metadata(self):
        left = {
            "role": "assistant",
            "content": "done",
            "timestamp": "one",
            "usage": {"tokens": 1},
        }
        right = {
            "role": "assistant",
            "content": "done",
            "timestamp": "two",
            "usage": {"tokens": 999},
        }
        self.assertEqual(
            generation.semantic_sha256(left),
            generation.semantic_sha256(right),
        )
        right["content"] = "changed"
        self.assertNotEqual(
            generation.semantic_sha256(left),
            generation.semantic_sha256(right),
        )

    def test_fresh_suffix_rejects_future_or_history_drift(self):
        prompt = [
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "x"}],
            },
            {"role": "tool", "content": "error", "error": True},
        ]
        suffix = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "y"}],
            },
            {"role": "tool", "content": "ok", "error": False},
        ]
        self.assertEqual(
            generation.extract_fresh_suffix([*prompt, *suffix], prompt),
            suffix,
        )
        changed = deepcopy(prompt)
        changed[0]["content"] = "different"
        with self.assertRaisesRegex(
            generation.V6GenerationError, "history differs"
        ):
            generation.extract_fresh_suffix([*changed, *suffix], prompt)

    def test_failed_calls_never_enter_success_label_mask(self):
        values = [
            {"role": "assistant", "tool_calls": [{"name": "bad"}]},
            {"role": "tool", "error": True},
            {"role": "assistant", "tool_calls": [{"name": "good"}]},
            {"role": "tool", "error": False},
            {"role": "assistant", "content": "done"},
        ]
        self.assertEqual(
            generation.successful_assistant_labels(values),
            [False, False, True, False, False],
        )

    def test_seed_set_is_fixed_nonempty_and_unique(self):
        self.assertEqual(generation.parse_seed_set("1,2,3"), (1, 2, 3))
        with self.assertRaisesRegex(generation.V6GenerationError, "distinct"):
            generation.parse_seed_set("1,1")
        with self.assertRaisesRegex(generation.V6GenerationError, "non-empty"):
            generation.parse_seed_set("")

    def test_run_contract_binds_endpoints_tau2_limits_and_full_decoding(self):
        args = generation_args()
        seeds = (101, 202)
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=seeds,
        )
        contract = generation.build_run_contract(
            args,
            registry_file_sha256=digest("registry-file"),
            registry_sha256=digest("registry"),
            task_ids=("retail:1",),
            semantic_contract=semantic,
        )

        self.assertEqual(contract["tau2_commit"], protocol.TAU2_COMMIT)
        self.assertEqual(
            (
                contract["teacher_api_base"],
                contract["user_api_base"],
                contract["judge_api_base"],
            ),
            (
                args.teacher_api_base,
                args.user_api_base,
                args.judge_api_base,
            ),
        )
        self.assertEqual(contract["max_tokens"], args.max_tokens)
        self.assertEqual(contract["max_steps"], args.max_steps)
        self.assertEqual(
            contract["decoding"],
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 512,
                "parallel_tool_calls": False,
                "applies_to": ["teacher", "user", "judge"],
                "assistant_tool_only_normalization": (
                    "execute_first_tool_call_then_replan"
                ),
            },
        )
        self.assertEqual(contract["runner"]["max_steps"], 60)
        self.assertEqual(contract["runner"]["timeout_seconds"], 900.0)
        self.assertEqual(contract["clean_attempts"], 2)
        self.assertEqual(contract["recovery_attempts"], 3)
        self.assertEqual(contract["continuation_seeds"], list(seeds))
        self.assertEqual(
            contract["semantic_generation_contract_sha256"],
            generation.sha256(semantic),
        )

        baseline = generation.sha256(semantic)
        for field, changed_value in (
            ("teacher_api_base", "http://other-teacher.example/v1"),
            ("user_api_base", "http://other-user.example/v1"),
            ("judge_api_base", "http://other-judge.example/v1"),
            ("max_tokens", 1024),
            ("max_steps", 61),
            ("timeout", 901.0),
            ("clean_attempts", 4),
            ("recovery_attempts", 5),
        ):
            changed_args = generation_args(**{field: changed_value})
            changed = generation.semantic_generation_contract(
                changed_args,
                continuation_seeds=seeds,
            )
            self.assertNotEqual(
                generation.sha256(changed),
                baseline,
                msg=f"{field} must be bound by the semantic contract",
            )
        changed_seeds = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101, 303),
        )
        self.assertNotEqual(generation.sha256(changed_seeds), baseline)

    def test_task_resume_receipt_rejects_changed_generation_contract(self):
        args = generation_args()
        registry_sha256 = digest("registry")
        semantic = generation.semantic_generation_contract(
            args,
            continuation_seeds=(101, 202),
        )
        run_contract = generation.build_run_contract(
            args,
            registry_file_sha256=digest("registry-file"),
            registry_sha256=registry_sha256,
            task_ids=("retail:1",),
            semantic_contract=semantic,
        )
        run_contract_sha256 = generation.sha256(run_contract)
        semantic_sha256 = generation.sha256(semantic)
        receipt = {
            "protocol": generation.GENERATION_PROTOCOL,
            "status": "PASS",
            "task_identity": "retail:1",
            "registry_sha256": registry_sha256,
            "run_contract_sha256": run_contract_sha256,
            "semantic_generation_contract": deepcopy(semantic),
            "semantic_generation_contract_sha256": semantic_sha256,
            "candidate_pair_count": 1,
            "candidate_pairs": [
                {
                    "task_identity": "retail:1",
                    "registry_sha256": registry_sha256,
                    "generation_contract": deepcopy(semantic),
                    "generation_contract_sha256": semantic_sha256,
                }
            ],
        }
        generation.validate_task_resume_receipt(
            receipt,
            task_identity="retail:1",
            registry_sha256=registry_sha256,
            run_contract_sha256=run_contract_sha256,
            semantic_contract=semantic,
        )

        changed_args = generation_args(max_tokens=1024)
        changed_semantic = generation.semantic_generation_contract(
            changed_args,
            continuation_seeds=(101, 202),
        )
        changed_run_contract = generation.build_run_contract(
            changed_args,
            registry_file_sha256=digest("registry-file"),
            registry_sha256=registry_sha256,
            task_ids=("retail:1",),
            semantic_contract=changed_semantic,
        )
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "run/semantic generation contract drift",
        ):
            generation.validate_task_resume_receipt(
                receipt,
                task_identity="retail:1",
                registry_sha256=registry_sha256,
                run_contract_sha256=generation.sha256(changed_run_contract),
                semantic_contract=changed_semantic,
            )

        tampered = deepcopy(receipt)
        tampered["candidate_pairs"][0]["generation_contract"][
            "teacher_api_base"
        ] = "http://stale-teacher.example/v1"
        with self.assertRaisesRegex(
            generation.V6GenerationError,
            "run/semantic generation contract drift",
        ):
            generation.validate_task_resume_receipt(
                tampered,
                task_identity="retail:1",
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=semantic,
            )


if __name__ == "__main__":
    unittest.main()
