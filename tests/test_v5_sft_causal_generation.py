import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from tests.v5_multifault_fixtures import (
    SOURCE_FILES,
    TAU2_COMMIT,
    fault_protocol,
    multifault_manifest,
    multifault_row,
    write_dynamic_audit,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v5_sft_causal_generate.py"
SPEC = importlib.util.spec_from_file_location("run_v5_sft_causal_generate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(domain, task_id):
    return multifault_row(
        domain,
        task_id,
        source_split="derived_inner_train",
    )


def split_manifest():
    return {
        "protocol": "v5_stage0_tau2_end_to_end",
        "guarantees": {
            "validation_derived_from_official_train_only": True,
            "official_test_task_content_exported": False,
        },
        "domains": {
            "retail": {
                "inner_train_ids": [str(index) for index in range(59)],
                "validation_ids": [str(index) for index in range(100, 115)],
                "sealed_test_ids": [str(index) for index in range(200, 240)],
            },
            "airline": {
                "inner_train_ids": [str(index) for index in range(24)],
                "validation_ids": [str(index) for index in range(100, 106)],
                "sealed_test_ids": [str(index) for index in range(200, 220)],
            },
        },
    }


def all_inner_rows():
    return [
        *[row("retail", index) for index in range(59)],
        *[row("airline", index) for index in range(24)],
    ]


class V5SFTCausalGenerationTests(unittest.TestCase):
    def write(self, payload, split=None):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "manifest.json"
        split_path = Path(directory.name) / "split_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        split_path.write_text(
            json.dumps(split if split is not None else split_manifest()),
            encoding="utf-8",
        )
        return directory, path, split_path

    def test_inner_train_manifest_is_accepted(self):
        rows = all_inner_rows()
        payload = multifault_manifest(
            rows,
            protocol=MODULE.GENERATION_PROTOCOL,
            source_split="derived_inner_train",
            split_sha256="placeholder",
        )
        directory, path, split_path = self.write(payload)
        self.addCleanup(directory.cleanup)
        payload["split_manifest_sha256"] = MODULE.sha256_file(split_path)
        path.write_text(json.dumps(payload), encoding="utf-8")
        split = MODULE.load_split_manifest(split_path)
        loaded = MODULE.load_inner_train_manifest(
            path,
            split_manifest=split,
            split_manifest_sha256=MODULE.sha256_file(split_path),
        )
        self.assertEqual(len(loaded["rows"]), 83)
        self.assertIsNone(loaded["_validated_gt_compatibility_filter"])

    def test_v5_3_manifest_freezes_a_complete_gt_compatibility_partition(self):
        rows = all_inner_rows()
        excluded = ["airline:0", "retail:24"]
        rows = [
            item
            for item in rows
            if f"{item['domain']}:{item['task_id']}" not in set(excluded)
        ]
        payload = multifault_manifest(
            rows,
            protocol=MODULE.V5_3_GENERATION_PROTOCOL,
            source_split="derived_inner_train",
            split_sha256="placeholder",
        )
        payload["gt_compatibility_filter"] = {
            "protocol": MODULE.GT_FILTER_PROTOCOL,
            "policy": "exclude_before_sharding",
            "teacher_mode": "ground_truth",
            "source_task_count": 83,
            "included_task_count": len(rows),
            "excluded_task_ids": excluded,
            "exclusion_reason": "no_expected_tool_actions",
            "selection_uses_rollouts_rewards_validation_or_test": False,
            "official_test_used": False,
        }
        directory, path, split_path = self.write(payload)
        self.addCleanup(directory.cleanup)
        payload["split_manifest_sha256"] = MODULE.sha256_file(split_path)
        path.write_text(json.dumps(payload), encoding="utf-8")

        loaded = MODULE.load_inner_train_manifest(
            path,
            split_manifest=MODULE.load_split_manifest(split_path),
            split_manifest_sha256=MODULE.sha256_file(split_path),
        )

        self.assertEqual(len(loaded["rows"]), 81)
        self.assertEqual(
            loaded["_validated_gt_compatibility_filter"][
                "excluded_task_ids"
            ],
            sorted(excluded),
        )

    def test_v5_3_filter_must_exactly_partition_frozen_inner_train(self):
        rows = [
            item
            for item in all_inner_rows()
            if f"{item['domain']}:{item['task_id']}"
            not in {"airline:23", "retail:24"}
        ]
        payload = multifault_manifest(
            rows,
            protocol=MODULE.V5_3_GENERATION_PROTOCOL,
            source_split="derived_inner_train",
            split_sha256="placeholder",
        )
        payload["gt_compatibility_filter"] = {
            "protocol": MODULE.GT_FILTER_PROTOCOL,
            "policy": "exclude_before_sharding",
            "teacher_mode": "ground_truth",
            "source_task_count": 83,
            "included_task_count": len(rows),
            "excluded_task_ids": ["retail:24"],
            "exclusion_reason": "no_expected_tool_actions",
            "selection_uses_rollouts_rewards_validation_or_test": False,
            "official_test_used": False,
        }
        directory, path, split_path = self.write(payload)
        self.addCleanup(directory.cleanup)
        payload["split_manifest_sha256"] = MODULE.sha256_file(split_path)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "must partition"):
            MODULE.load_inner_train_manifest(
                path,
                split_manifest=MODULE.load_split_manifest(split_path),
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

    def test_validation_or_unsafe_manifest_is_rejected(self):
        rows = all_inner_rows()
        rows[0] = row("retail", "100")
        payload = multifault_manifest(
            rows,
            protocol=MODULE.GENERATION_PROTOCOL,
            source_split="derived_inner_train",
            split_sha256="placeholder",
        )
        directory, path, split_path = self.write(payload)
        self.addCleanup(directory.cleanup)
        payload["split_manifest_sha256"] = MODULE.sha256_file(split_path)
        path.write_text(json.dumps(payload), encoding="utf-8")
        split = MODULE.load_split_manifest(split_path)
        with self.assertRaisesRegex(RuntimeError, "validation/test"):
            MODULE.load_inner_train_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

        rows = all_inner_rows()
        payload["rows"] = rows
        rows[0]["error_condition"]["expected_state_mutation"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            MODULE.load_inner_train_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

    def test_split_protocol_counts_and_declared_hash_fail_closed(self):
        rows = all_inner_rows()
        payload = multifault_manifest(
            rows,
            protocol=MODULE.GENERATION_PROTOCOL,
            source_split="derived_inner_train",
            split_sha256="0" * 64,
        )
        directory, path, split_path = self.write(payload)
        self.addCleanup(directory.cleanup)
        split = MODULE.load_split_manifest(split_path)
        with self.assertRaisesRegex(RuntimeError, "split_manifest_sha256"):
            MODULE.load_inner_train_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

        broken = split_manifest()
        broken["domains"]["retail"]["sealed_test_ids"].pop()
        split_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "count drift"):
            MODULE.load_split_manifest(split_path)

    @mock.patch.object(MODULE, "require_clean_tracked_source")
    def test_generation_provenance_requires_exact_commits(self, _clean_source):
        source = "a" * 40
        with mock.patch.object(MODULE, "git_commit", return_value=source):
            self.assertEqual(
                MODULE.validate_provenance(
                    expected_source_commit=source,
                    teacher_revision="b" * 40,
                    user_revision="c" * 40,
                    judge_revision="d" * 40,
                ),
                source,
            )
        with mock.patch.object(MODULE, "git_commit", return_value=source):
            with self.assertRaisesRegex(RuntimeError, "teacher-revision"):
                MODULE.validate_provenance(
                    expected_source_commit=source,
                    teacher_revision="main",
                    user_revision="c" * 40,
                    judge_revision="d" * 40,
                )
        with mock.patch.object(MODULE, "git_commit", return_value="e" * 40):
            with self.assertRaisesRegex(RuntimeError, "source commit drift"):
                MODULE.validate_provenance(
                    expected_source_commit=source,
                    teacher_revision="b" * 40,
                    user_revision="c" * 40,
                    judge_revision="d" * 40,
                )

    def test_four_shards_are_balanced_and_disjoint(self):
        rows = [row("retail", index) for index in range(83)]
        shards = [MODULE.shard_rows(rows, index, 4) for index in range(4)]
        keys = [item["pair_id"] for shard in shards for item in shard]
        self.assertEqual(len(keys), 83)
        self.assertEqual(len(set(keys)), 83)
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_output_filename_records_shard(self):
        self.assertEqual(
            MODULE.output_filename("retail", "error", 1, 4),
            "retail_error.shard-001-of-004.json",
        )

    def test_local_vllm_models_use_explicit_litellm_provider(self):
        self.assertEqual(
            MODULE.litellm_openai_model("Qwen/Qwen2.5-14B-Instruct-AWQ"),
            "openai/Qwen/Qwen2.5-14B-Instruct-AWQ",
        )
        with self.assertRaisesRegex(RuntimeError, "raw served model name"):
            MODULE.litellm_openai_model(
                "openai/Qwen/Qwen2.5-14B-Instruct-AWQ"
            )

    def test_generation_contract_stays_incomplete_until_atomic_finalize(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        manifest = root / "manifest.json"
        split = root / "split.json"
        split.write_text("{}\n", encoding="utf-8")
        manifest.write_text(
            json.dumps(
                multifault_manifest(
                    [row("retail", "7")],
                    protocol=MODULE.GENERATION_PROTOCOL,
                    source_split="derived_inner_train",
                    split_sha256=MODULE.sha256_file(split),
                )
            ),
            encoding="utf-8",
        )
        dynamic_audit_path = root / "dynamic_audit.json"
        write_dynamic_audit(
            dynamic_audit_path,
            manifest_path=manifest,
            split_manifest_path=split,
        )
        dynamic_audit_identity = MODULE.load_complete_dynamic_audit(
            dynamic_audit_path,
            manifest_path=manifest,
            split_manifest_path=split,
            expected_source_split="derived_inner_train",
            expected_task_ids={"retail:7"},
        )
        output = root / "raw"
        args = SimpleNamespace(
            output_dir=output,
            num_shards=4,
            shard_index=2,
            expected_source_commit="a" * 40,
            num_trials=3,
            teacher_model="Qwen/Qwen2.5-14B-Instruct-AWQ",
            teacher_revision="b" * 40,
            teacher_api_base="http://teacher/v1",
            teacher_mode="ground_truth",
            user_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            user_revision="c" * 40,
            user_api_base="http://user/v1",
            judge_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            judge_revision="c" * 40,
            judge_api_base="http://user/v1",
            max_tokens=512,
            max_steps=60,
            timeout=900,
            seed=20260722,
            temperature=0.0,
            top_p=1.0,
        )
        contract = MODULE.write_contract(
            args,
            manifest.resolve(),
            split.resolve(),
            [row("retail", "7")],
            dynamic_audit_path=dynamic_audit_path,
            dynamic_audit_identity=dynamic_audit_identity,
            sampling_contract=MODULE.validate_sampling_contract(
                manifest_protocol=MODULE.GENERATION_PROTOCOL,
                temperature=0.0,
                top_p=1.0,
                num_trials=3,
                seed=20260722,
            ),
        )
        initial = json.loads(contract.read_text(encoding="utf-8"))
        self.assertEqual(initial["status"], "INCOMPLETE")
        self.assertEqual(initial["result_sha256"], {})
        self.assertEqual(
            initial["generation_manifest_protocol"],
            MODULE.GENERATION_PROTOCOL,
        )
        self.assertEqual(
            initial["judge"]["protocol"],
            MODULE.STRICT_NL_JUDGE_PROTOCOL,
        )
        self.assertTrue(initial["judge"]["raw_response_audit"])
        self.assertEqual(initial["decoding"]["temperature"], 0.0)
        self.assertEqual(initial["decoding"]["top_p"], 1.0)
        self.assertEqual(len(initial["decoding"]["derived_trial_seeds"]), 3)
        self.assertTrue(
            initial["sampling_contract"][
                "derived_trial_seeds_are_distinct"
            ]
        )
        self.assertFalse(initial["decoding"]["parallel_tool_calls"])
        self.assertEqual(
            initial["decoding"]["parallel_tool_call_normalization"],
            "execute_first_then_replan",
        )
        self.assertEqual(
            initial["decoding"]["mixed_tool_call_content_normalization"],
            "drop_text_preserve_sha256",
        )
        self.assertEqual(initial["fault_protocol"], fault_protocol())
        self.assertEqual(initial["tau2_commit"], TAU2_COMMIT)
        self.assertEqual(initial["source_files"], SOURCE_FILES)
        self.assertEqual(
            initial["dynamic_audit_identity"], dynamic_audit_identity
        )

        # A failed generation never calls finalize and therefore cannot look
        # like a complete artifact set.
        self.assertEqual(
            json.loads(contract.read_text(encoding="utf-8"))["status"],
            "INCOMPLETE",
        )

        result = output / "retail_clean.shard-002-of-004.json"
        result.write_text('{"simulations": []}\n', encoding="utf-8")
        hashes = MODULE.finalize_contract(contract, [result])
        completed = json.loads(contract.read_text(encoding="utf-8"))
        self.assertEqual(completed["status"], "COMPLETE")
        self.assertEqual(completed["result_sha256"], hashes)
        self.assertEqual(hashes[result.name], MODULE.sha256_file(result))
        with self.assertRaisesRegex(RuntimeError, "not INCOMPLETE"):
            MODULE.finalize_contract(contract, [result])

    def test_endpoint_disables_parallel_tool_calls(self):
        args = MODULE.endpoint_args(
            "http://localhost/v1",
            "key",
            max_tokens=512,
            seed=20260722,
            temperature=0.2,
            top_p=0.95,
        )
        self.assertIs(args["parallel_tool_calls"], False)
        self.assertEqual(args["temperature"], 0.2)
        self.assertEqual(args["top_p"], 0.95)

    def test_v5_3_finalize_rejects_missing_strict_judge_call_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            contract = output / "run_contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "status": "INCOMPLETE",
                        "generation_manifest_protocol": (
                            MODULE.V5_3_GENERATION_PROTOCOL
                        ),
                        "result_sha256": {},
                        "strict_judge_audit_evidence": {},
                    }
                ),
                encoding="utf-8",
            )
            result = output / "retail_clean.json"
            result.write_text(
                json.dumps(
                    {
                        "simulations": [
                            {
                                "task_id": "7",
                                "id": "missing-audit",
                                "reward_info": {
                                    "nl_assertions": [
                                        {
                                            "nl_assertion": "expected",
                                            "met": True,
                                            "justification": "fixture",
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "strict-judge evidence is incomplete",
            ):
                MODULE.finalize_contract(contract, [result])
            self.assertEqual(
                json.loads(contract.read_text(encoding="utf-8"))["status"],
                "INCOMPLETE",
            )

    def test_v5_3_sampling_is_nonzero_frozen_and_uses_distinct_trial_seeds(self):
        contract = MODULE.validate_sampling_contract(
            manifest_protocol=MODULE.V5_3_GENERATION_PROTOCOL,
            temperature=MODULE.V5_3_TEMPERATURE,
            top_p=MODULE.V5_3_TOP_P,
            num_trials=MODULE.V5_3_NUM_TRIALS,
            seed=MODULE.V5_3_SEED,
        )
        self.assertGreater(contract["temperature"], 0)
        self.assertEqual(len(contract["derived_trial_seeds"]), 12)
        self.assertEqual(len(set(contract["derived_trial_seeds"])), 12)
        self.assertEqual(contract["judge_temperature"], 0.0)

        invalid_cases = (
            (0.0, MODULE.V5_3_TOP_P, 12, MODULE.V5_3_SEED),
            (MODULE.V5_3_TEMPERATURE, 1.0, 12, MODULE.V5_3_SEED),
            (MODULE.V5_3_TEMPERATURE, MODULE.V5_3_TOP_P, 3, MODULE.V5_3_SEED),
            (MODULE.V5_3_TEMPERATURE, MODULE.V5_3_TOP_P, 12, 7),
        )
        for temperature, top_p, trials, seed in invalid_cases:
            with self.subTest(
                temperature=temperature,
                top_p=top_p,
                trials=trials,
                seed=seed,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "sampling contract drift",
                ):
                    MODULE.validate_sampling_contract(
                        manifest_protocol=MODULE.V5_3_GENERATION_PROTOCOL,
                        temperature=temperature,
                        top_p=top_p,
                        num_trials=trials,
                        seed=seed,
                    )

    def test_gt_preflight_verifies_declared_filter_against_full_universe(self):
        included = [
            {"domain": "retail", "task_id": "1"},
            {"domain": "airline", "task_id": "2"},
        ]
        gt_filter = {
            "protocol": MODULE.GT_FILTER_PROTOCOL,
            "excluded_task_ids": ["retail:3"],
        }

        def selector(domain, rows):
            return [
                SimpleNamespace(id=str(item["task_id"]))
                for item in rows
            ]

        report = MODULE.preflight_gt_compatibility(
            included,
            teacher_mode="ground_truth",
            gt_compatibility_filter=gt_filter,
            task_selector=selector,
            compatibility_check=lambda task: str(task.id) != "3",
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["filter_verified"])
        self.assertEqual(report["checked_task_count"], 3)
        self.assertEqual(report["compatible_task_count"], 2)
        self.assertEqual(
            report["incompatible_task_ids"],
            ["retail:3"],
        )

        with self.assertRaises(MODULE.GTCompatibilityError):
            MODULE.preflight_gt_compatibility(
                included,
                teacher_mode="ground_truth",
                gt_compatibility_filter=gt_filter,
                task_selector=selector,
                compatibility_check=lambda task: True,
            )

    def test_legacy_gt_preflight_fails_before_actionless_rollout(self):
        rows = [{"domain": "retail", "task_id": "24"}]

        with self.assertRaisesRegex(
            MODULE.GTCompatibilityError,
            "retail:24",
        ):
            MODULE.preflight_gt_compatibility(
                rows,
                teacher_mode="ground_truth",
                task_selector=lambda domain, selected: [
                    SimpleNamespace(id="24")
                ],
                compatibility_check=lambda task: False,
            )

    @mock.patch.object(MODULE, "install_strict_nl_judge")
    def test_generation_installs_bounded_strict_judge(self, install):
        MODULE.patch_local_nl_judge("openai/judge", {"temperature": 0})
        install.assert_called_once_with(
            model="openai/judge",
            llm_args={"temperature": 0},
            max_content_attempts=MODULE.DEFAULT_CONTENT_ATTEMPTS,
        )

    def test_generation_interface_serializes_and_audits_parallel_calls(self):
        first = mock.Mock()
        first.model_dump.return_value = {"name": "first"}
        second = mock.Mock()
        second.model_dump.return_value = {"name": "second"}
        message = SimpleNamespace(
            tool_calls=[first, second],
            content="I will use both tools.",
            raw_data={"provider": "vllm"},
        )

        normalized = MODULE.normalize_tool_only_message(message)

        self.assertEqual(normalized.tool_calls, [first])
        self.assertIsNone(normalized.content)
        parallel = normalized.raw_data["v5_stage1_parallel_calls_serialized"]
        self.assertEqual(parallel["original_count"], 2)
        self.assertEqual(len(parallel["deferred_call_sha256"]), 1)
        self.assertRegex(parallel["deferred_call_sha256"][0], r"^[0-9a-f]{64}$")
        mixed = normalized.raw_data["v5_stage1_mixed_content_normalized"]
        self.assertEqual(mixed["utf8_bytes"], len("I will use both tools.".encode()))
        self.assertRegex(mixed["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
