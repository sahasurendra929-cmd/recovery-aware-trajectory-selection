import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tests.v5_multifault_fixtures import (
    SOURCE_FILES,
    TAU2_COMMIT,
    fault_descriptor,
    fault_protocol,
    multifault_manifest,
    multifault_rows,
    write_dynamic_audit,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_v5_sft_causal.py"
SPEC = importlib.util.spec_from_file_location("summarize_v5_sft_causal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class V5SFTCausalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.split_path = self.root / "split_manifest.json"
        self.evaluation_path = self.root / "evaluation_manifest.json"
        self.dynamic_audit_path = self.root / "validation_dynamic_audit.json"
        self.registry_path = self.root / "checkpoint_registry.json"
        self.output_path = self.root / "summary.json"
        self.source_commit = "a" * 40
        self.base_revision = "b" * 40
        self.validation = {
            "retail": [str(index) for index in range(15)],
            "airline": [str(index) for index in range(6)],
        }
        self.sealed = {
            "retail": [str(index) for index in range(100, 140)],
            "airline": [str(index) for index in range(100, 120)],
        }
        self._write_split()
        self.original_published_split_sha = (
            MODULE.PUBLISHED_STAGE0_SPLIT_SHA256
        )
        MODULE.PUBLISHED_STAGE0_SPLIT_SHA256 = MODULE.sha256_file(
            self.split_path
        )
        self._write_evaluation()
        write_dynamic_audit(
            self.dynamic_audit_path,
            manifest_path=self.evaluation_path,
            split_manifest_path=self.split_path,
        )
        self.dynamic_audit_identity = MODULE.load_complete_dynamic_audit(
            self.dynamic_audit_path,
            manifest_path=self.evaluation_path,
            split_manifest_path=self.split_path,
            expected_source_split="derived_validation",
            expected_task_ids=set(self._ordered_task_ids()),
        )
        self._write_registry()

    def tearDown(self):
        MODULE.PUBLISHED_STAGE0_SPLIT_SHA256 = (
            self.original_published_split_sha
        )
        self.tempdir.cleanup()

    def _write_split(self):
        split = {
            "protocol": "v5_stage0_tau2_end_to_end",
            "guarantees": {
                "validation_derived_from_official_train_only": True,
                "official_test_task_content_exported": False,
            },
            "domains": {},
        }
        for domain in MODULE.DOMAINS:
            split["domains"][domain] = {
                "inner_train_ids": [f"train-{domain}"],
                "validation_ids": self.validation[domain],
                "sealed_test_ids": self.sealed[domain],
            }
        self.split_path.write_text(json.dumps(split), encoding="utf-8")

    def _evaluation_rows(self):
        return multifault_rows(
            self.validation,
            source_split="derived_validation",
        )

    def _write_evaluation(self, rows=None):
        rows = self._evaluation_rows() if rows is None else rows
        evaluation = multifault_manifest(
            rows,
            protocol=MODULE.EVALUATION_MANIFEST_PROTOCOL,
            source_split="derived_validation",
            split_sha256=MODULE.sha256_file(self.split_path),
        )
        self.evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    def _registry(self):
        entries = {
            "base_model": {
                "model_id": "openai/served-base",
                "adapter_sha256": None,
                "adapter_config_sha256": None,
                "training_run_manifest_sha256": None,
            }
        }
        for index, arm in enumerate(MODULE.TRAINED_ARMS, start=1):
            entries[arm] = {
                "model_id": f"openai/served-{arm}",
                "adapter_sha256": f"{index:x}" * 64,
                "adapter_config_sha256": f"{index + 8:x}" * 64,
                "training_run_manifest_sha256": f"{index + 4:x}" * 64,
            }
        return {
            "protocol": MODULE.CHECKPOINT_REGISTRY_PROTOCOL,
            "source_commit": self.source_commit,
            "base_model_revision": self.base_revision,
            "training_data_provenance": {
                "data_audit_sha256": "7" * 64,
                "data_hashes_sha256": "8" * 64,
                "dynamic_audits": {
                    "generation": {
                        "protocol": "v5_stage1_dynamic_injection_audit",
                        "sha256": "9" * 64,
                        "manifest_sha256": "a" * 64,
                        "split_manifest_sha256": MODULE.sha256_file(
                            self.split_path
                        ),
                        "source_split": "derived_inner_train",
                        "verified_injections": 83,
                        "official_test_used": False,
                        "official_test_sealed": True,
                    },
                    "validation": self.dynamic_audit_identity,
                },
                "official_test_used": False,
                "official_test_sealed": True,
            },
            "entries": entries,
        }

    def _write_registry(self, value=None):
        self.registry_path.write_text(
            json.dumps(self._registry() if value is None else value),
            encoding="utf-8",
        )

    def _ordered_task_ids(self):
        task_ids = [
            f"{domain}:{task_id}"
            for domain in MODULE.DOMAINS
            for task_id in self.validation[domain]
        ]
        return sorted(
            task_ids,
            key=lambda item: (
                MODULE.hashlib.sha256(item.encode("utf-8")).hexdigest(),
                item,
            ),
        )

    def _write_contracts(self, arm, arm_dir, shard_count=1):
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        ordered = self._ordered_task_ids()
        base_alias = registry["entries"]["base_model"]["model_id"]
        for shard_index in range(shard_count):
            suffix = (
                ""
                if shard_count == 1
                else f".shard-{shard_index:03d}-of-{shard_count:03d}"
            )
            result_sha256 = {}
            for domain in MODULE.DOMAINS:
                for condition in MODULE.CONDITIONS:
                    result_path = arm_dir / f"{domain}_{condition}{suffix}.json"
                    if result_path.is_file():
                        result_sha256[result_path.name] = MODULE.sha256_file(
                            result_path
                        )
            contract = {
                "protocol": MODULE.RUN_CONTRACT_PROTOCOL,
                "status": "COMPLETE",
                "created_at": "2026-07-25T00:00:00+00:00",
                "completed_at": "2026-07-25T00:01:00+00:00",
                "arm": arm,
                "evaluation_manifest_sha256": MODULE.sha256_file(
                    self.evaluation_path
                ),
                "split_manifest_sha256": MODULE.sha256_file(self.split_path),
                "dynamic_audit_identity": self.dynamic_audit_identity,
                "fault_protocol": fault_protocol(),
                "tau2_commit": TAU2_COMMIT,
                "source_files": SOURCE_FILES,
                "checkpoint_registry_sha256": MODULE.sha256_file(
                    self.registry_path
                ),
                "checkpoint_registry_protocol": registry["protocol"],
                "checkpoint_entry": registry["entries"][arm],
                "source_commit": registry["source_commit"],
                "base_model_revision": registry["base_model_revision"],
                "task_ids": ordered[shard_index::shard_count],
                "shard_index": shard_index,
                "num_shards": shard_count,
                "agent": {
                    "model": registry["entries"][arm]["model_id"],
                    "api_base": f"http://agent-{shard_index}",
                },
                "user": {
                    "model": base_alias,
                    "api_base": f"http://shared-{shard_index}",
                },
                "judge": {
                    "model": base_alias,
                    "api_base": f"http://shared-{shard_index}",
                },
                "decoding": {
                    "temperature": 0,
                    "max_tokens": 512,
                    "max_steps": 60,
                    "task_timeout_seconds": 900.0,
                    "seed": 20260722,
                    "num_trials": 1,
                },
                "conditions": ["clean", "error"],
                "official_test_used": False,
                "result_sha256": result_sha256,
            }
            path = (
                arm_dir / "run_contract.json"
                if shard_count == 1
                else arm_dir
                / (
                    f"run_contract.shard-{shard_index:03d}-of-"
                    f"{shard_count:03d}.json"
                )
            )
            path.write_text(json.dumps(contract), encoding="utf-8")

    def _refresh_result_hashes(self, arm_dir):
        contract_paths = [
            *arm_dir.glob("run_contract.json"),
            *arm_dir.glob("run_contract.shard-*-of-*.json"),
        ]
        for contract_path in contract_paths:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            shard_index = payload["shard_index"]
            shard_count = payload["num_shards"]
            suffix = (
                ""
                if shard_count == 1
                else f".shard-{shard_index:03d}-of-{shard_count:03d}"
            )
            payload["result_sha256"] = {
                path.name: MODULE.sha256_file(path)
                for domain in MODULE.DOMAINS
                for condition in MODULE.CONDITIONS
                if (
                    path := arm_dir / f"{domain}_{condition}{suffix}.json"
                ).is_file()
            }
            contract_path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _success_set(count):
        ordered = [
            *(f"retail:{index}" for index in range(15)),
            *(f"airline:{index}" for index in range(6)),
        ]
        return set(ordered[:count])

    def _simulation(
        self,
        *,
        domain,
        task_id,
        condition,
        successful,
        rollout_seed,
        repeat_error=False,
        valid_post_error=True,
    ):
        common = {
            "task_id": task_id,
            "rollout_seed": rollout_seed,
            "termination_reason": "agent_stop",
            "reward_info": {"reward": 1.0 if successful else 0.0},
        }
        if condition == "clean":
            return {
                **common,
                "messages": [{"role": "user", "content": "test"}],
            }
        fault = fault_descriptor(
            domain,
            task_id,
            source_split="derived_validation",
        )
        call_id = fault["tool_call_id"]
        messages = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "name": fault["tool_name"],
                        "arguments": fault["arguments"],
                    }
                ],
            },
            {
                "role": "tool",
                "id": call_id,
                "error": True,
                "content": "not found",
            },
        ]
        if repeat_error:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"repeat-{rollout_seed}",
                                "name": fault["tool_name"],
                                "arguments": fault["arguments"],
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "id": f"repeat-{rollout_seed}",
                        "error": True,
                        "content": "not found",
                    },
                ]
            )
        if valid_post_error:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"repair-{domain}-{task_id}-{rollout_seed}",
                                "name": "search",
                                "arguments": {"query": "valid"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "id": f"repair-{domain}-{task_id}-{rollout_seed}",
                        "error": False,
                        "content": "ok",
                    },
                ]
            )
        return {**common, "messages": messages}

    def _write_arm(
        self,
        arm,
        *,
        clean_successes,
        error_successes,
        repetitions=1,
        repeated_tasks=0,
        valid_post_error_tasks=21,
    ):
        arm_dir = self.root / arm
        arm_dir.mkdir()
        clean_set = self._success_set(clean_successes)
        error_set = self._success_set(error_successes)
        repeated_set = self._success_set(repeated_tasks)
        valid_set = self._success_set(valid_post_error_tasks)
        for domain in MODULE.DOMAINS:
            for condition in MODULE.CONDITIONS:
                simulations = []
                for task_id in self.validation[domain]:
                    key = f"{domain}:{task_id}"
                    for rollout_seed in range(repetitions):
                        simulations.append(
                            self._simulation(
                                domain=domain,
                                task_id=task_id,
                                condition=condition,
                                successful=(
                                    key in clean_set
                                    if condition == "clean"
                                    else key in error_set
                                ),
                                rollout_seed=rollout_seed,
                                repeat_error=key in repeated_set,
                                valid_post_error=key in valid_set,
                            )
                        )
                (arm_dir / f"{domain}_{condition}.json").write_text(
                    json.dumps({"simulations": simulations}),
                    encoding="utf-8",
                )
        self._write_contracts(arm, arm_dir)
        return arm_dir

    def _four_arms(self, repetitions=1):
        return {
            "base_model": self._write_arm(
                "base_model",
                clean_successes=12,
                error_successes=6,
                repetitions=repetitions,
                repeated_tasks=5,
                valid_post_error_tasks=15,
            ),
            "perfect_success": self._write_arm(
                "perfect_success",
                clean_successes=15,
                error_successes=8,
                repetitions=repetitions,
                repeated_tasks=4,
                valid_post_error_tasks=17,
            ),
            "failure_raw": self._write_arm(
                "failure_raw",
                clean_successes=14,
                error_successes=7,
                repetitions=repetitions,
                repeated_tasks=7,
                valid_post_error_tasks=15,
            ),
            "repair_50": self._write_arm(
                "repair_50",
                clean_successes=14,
                error_successes=10,
                repetitions=repetitions,
                repeated_tasks=2,
                valid_post_error_tasks=20,
            ),
            "repair_100": self._write_arm(
                "repair_100",
                clean_successes=13,
                error_successes=11,
                repetitions=repetitions,
                repeated_tasks=1,
                valid_post_error_tasks=20,
            ),
        }

    def _convert_arm_to_shards(self, arm_dir, shard_count=4):
        shard_tasks = [
            set(self._ordered_task_ids()[shard_index::shard_count])
            for shard_index in range(shard_count)
        ]
        for domain in MODULE.DOMAINS:
            for condition in MODULE.CONDITIONS:
                merged = arm_dir / f"{domain}_{condition}.json"
                payload = json.loads(merged.read_text(encoding="utf-8"))
                simulations = payload["simulations"]
                for shard_index in range(shard_count):
                    shard_rows = [
                        row
                        for row in simulations
                        if f"{domain}:{row['task_id']}" in shard_tasks[shard_index]
                    ]
                    if not shard_rows:
                        continue
                    shard = arm_dir / (
                        f"{domain}_{condition}.shard-"
                        f"{shard_index:03d}-of-{shard_count:03d}.json"
                    )
                    shard.write_text(
                        json.dumps({"simulations": shard_rows}),
                        encoding="utf-8",
                    )
                merged.unlink()
        for path in [
            *arm_dir.glob("run_contract.json"),
            *arm_dir.glob("run_contract.shard-*-of-*.json"),
        ]:
            path.unlink()
        arm = arm_dir.name
        self._write_contracts(arm, arm_dir, shard_count=shard_count)

    def _summarize(self, arm_dirs):
        return MODULE.summarize(
            split_manifest_path=self.split_path,
            evaluation_manifest_path=self.evaluation_path,
            dynamic_audit_path=self.dynamic_audit_path,
            checkpoint_registry_path=self.registry_path,
            arm_dirs=arm_dirs,
            output_path=self.output_path,
            bootstrap_draws=100,
        )

    def test_primary_metrics_pairing_and_directional_gate(self):
        summary = self._summarize(self._four_arms())
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["base_model_is_evaluation_only"])
        self.assertEqual(summary["independent_validation_tasks"], 21)
        self.assertEqual(
            summary["primary_metric"],
            "tau2_official_composite_task_success_error_condition",
        )
        self.assertAlmostEqual(
            summary["arms"]["perfect_success"][
                "primary_error_injected_end_to_end_success"
            ],
            8 / 21,
        )
        self.assertIn(
            "perfect_success_minus_base_model",
            summary["paired_comparisons"],
        )
        comparison = summary["paired_comparisons"][
            "repair_50_minus_perfect_success"
        ]
        self.assertEqual(comparison["independent_task_count"], 21)
        self.assertAlmostEqual(
            comparison["aggregates"]["injected_success_delta"][
                "mean_task_paired_delta"
            ],
            2 / 21,
        )
        self.assertAlmostEqual(
            comparison["aggregates"]["clean_success_delta"][
                "mean_task_paired_delta"
            ],
            -1 / 21,
        )
        self.assertTrue(comparison["directional_gate"]["pass"])

        too_much_recovery = summary["paired_comparisons"][
            "repair_100_minus_perfect_success"
        ]
        self.assertAlmostEqual(
            too_much_recovery["directional_gate"][
                "observed_injected_success_count_equivalent_delta"
            ],
            3.0,
        )
        self.assertAlmostEqual(
            too_much_recovery["directional_gate"][
                "observed_clean_success_count_equivalent_delta"
            ],
            -2.0,
        )
        self.assertFalse(too_much_recovery["directional_gate"]["pass"])

    def test_zero_shot_base_and_four_canonical_trained_arms_are_required(self):
        arms = self._four_arms()
        arms.pop("base_model")
        with self.assertRaisesRegex(RuntimeError, "zero-shot base_model"):
            self._summarize(arms)

    def test_registry_rejects_five_models_bound_to_one_alias(self):
        value = self._registry()
        for entry in value["entries"].values():
            entry["model_id"] = "openai/same-served-alias"
        self._write_registry(value)
        with self.assertRaisesRegex(RuntimeError, "model_id aliases must be unique"):
            self._summarize(self._four_arms())

    def test_run_contract_registry_drift_fails_closed(self):
        arms = self._four_arms()
        path = arms["repair_50"] / "run_contract.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["checkpoint_registry_sha256"] = "f" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "registry SHA drift"):
            self._summarize(arms)

    def test_registry_validation_audit_must_match_evaluation_audit(self):
        registry = self._registry()
        registry["training_data_provenance"]["dynamic_audits"][
            "validation"
        ]["sha256"] = "0" * 64
        self._write_registry(registry)
        with self.assertRaisesRegex(
            RuntimeError,
            "validation audit differs from evaluation audit",
        ):
            self._summarize(self._four_arms())

    def test_dynamic_audit_identity_chain_fails_closed(self):
        arms = self._four_arms()
        path = arms["repair_50"] / "run_contract.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dynamic_audit_identity"]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "dynamic audit identity drift"):
            self._summarize(arms)

        payload = json.loads(
            self.dynamic_audit_path.read_text(encoding="utf-8")
        )
        payload["status"] = "INCOMPLETE"
        self.dynamic_audit_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not COMPLETE"):
            self._summarize(arms)

    def test_missing_run_contract_fails_closed(self):
        arms = self._four_arms()
        (arms["repair_100"] / "run_contract.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing run contract"):
            self._summarize(arms)

    def test_incomplete_contract_fails_closed(self):
        arms = self._four_arms()
        path = arms["failure_raw"] / "run_contract.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "INCOMPLETE"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not COMPLETE"):
            self._summarize(arms)

    def test_result_bytes_are_bound_by_contract_sha(self):
        arms = self._four_arms()
        path = arms["repair_50"] / "retail_error.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["simulations"][0]["termination_reason"] = "tampered"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "result SHA drift"):
            self._summarize(arms)

    def test_contract_roles_must_use_exact_registry_base_alias(self):
        arms = self._four_arms()
        path = arms["perfect_success"] / "run_contract.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["user"]["model"] = "openai/another-unadapted-model"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "registry base_model alias"):
            self._summarize(arms)

    def test_contract_decoding_must_match_frozen_protocol(self):
        arms = self._four_arms()
        path = arms["base_model"] / "run_contract.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["decoding"]["max_tokens"] = 511
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "decoding protocol drift"):
            self._summarize(arms)

    def test_repeated_error_and_valid_post_error_are_execution_metrics(self):
        summary = self._summarize(self._four_arms())
        perfect = summary["arms"]["perfect_success"]
        self.assertAlmostEqual(
            perfect["repeated_identical_error_rate"],
            4 / 21,
        )
        self.assertAlmostEqual(
            perfect["valid_post_error_tool_result_rate"],
            17 / 21,
        )
        grouped = perfect["fault_family_descriptive"]
        self.assertEqual(
            set(grouped),
            {
                "retail_missing_user",
                "retail_missing_order",
                "airline_missing_reservation",
                "airline_missing_flight",
            },
        )
        self.assertEqual(sum(row["task_count"] for row in grouped.values()), 21)
        self.assertTrue(
            all(row["inference"] == "descriptive_only" for row in grouped.values())
        )

    def test_rollout_or_seed_duplicate_is_not_a_new_independent_task(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate task_id"):
            self._summarize(self._four_arms(repetitions=2))

    def test_complete_four_machine_shards_are_supported(self):
        arms = self._four_arms()
        for arm_dir in arms.values():
            self._convert_arm_to_shards(arm_dir)
        summary = self._summarize(arms)
        self.assertEqual(summary["independent_validation_tasks"], 21)
        self.assertEqual(
            summary["arms"]["repair_50"]["total_runs"],
            {"clean": 21, "error": 21},
        )
        self.assertTrue(
            summary["paired_comparisons"][
                "repair_50_minus_perfect_success"
            ]["directional_gate"]["pass"]
        )

    def test_incomplete_shard_set_fails_closed(self):
        arms = self._four_arms()
        for arm_dir in arms.values():
            self._convert_arm_to_shards(arm_dir)
        (
            arms["repair_50"]
            / "retail_clean.shard-003-of-004.json"
        ).unlink()
        with self.assertRaisesRegex(RuntimeError, "result file is missing"):
            self._summarize(arms)

    def test_official_test_task_in_evaluation_manifest_fails_closed(self):
        rows = self._evaluation_rows()
        rows[0] = {
            **rows[0],
            "pair_id": "retail:100",
            "task_id": "100",
            "error_condition": {
                **rows[0]["error_condition"],
                "tool_call_id": "injected-retail-100",
            },
        }
        self._write_evaluation(rows)
        with self.assertRaisesRegex(RuntimeError, "Official-test task leaked"):
            self._summarize(self._four_arms())

    def test_missing_condition_task_breaks_pairing(self):
        arms = self._four_arms()
        path = arms["repair_50"] / "retail_error.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["simulations"] = payload["simulations"][1:]
        path.write_text(json.dumps(payload), encoding="utf-8")
        self._refresh_result_hashes(arms["repair_50"])
        with self.assertRaisesRegex(RuntimeError, "task IDs do not match"):
            self._summarize(arms)

    def test_cross_arm_replicate_schedule_must_match(self):
        arms = self._four_arms()
        for condition in MODULE.CONDITIONS:
            path = arms["repair_100"] / f"retail_{condition}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["simulations"][0]["rollout_seed"] = 999
            path.write_text(json.dumps(payload), encoding="utf-8")
        self._refresh_result_hashes(arms["repair_100"])
        with self.assertRaisesRegex(RuntimeError, "replicate schedule"):
            self._summarize(arms)

    def test_successful_but_unmatched_tool_result_is_not_valid_recovery(self):
        simulation = self._simulation(
            domain="retail",
            task_id="0",
            condition="error",
            successful=True,
            rollout_seed=0,
            valid_post_error=False,
        )
        simulation["messages"].append(
            {
                "role": "tool",
                "id": "no-corresponding-assistant-call",
                "error": False,
                "content": "ok",
            }
        )
        behavior = MODULE.analyze_error_run(
            simulation,
            self._evaluation_rows()[0]["error_condition"],
        )
        self.assertFalse(behavior["valid_post_error_tool_result"])

    def test_missing_official_reward_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Missing official reward"):
            MODULE.reward({"task_id": "retail:0", "messages": []})


if __name__ == "__main__":
    unittest.main()
