import copy
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from scripts import summarize_v5_3_low_support_diagnostic as SUMMARY


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


REGISTRY = load_module(
    "build_v5_checkpoint_registry_low_support_test",
    "scripts/build_v5_checkpoint_registry.py",
)
EVALUATION = load_module(
    "run_v5_sft_causal_eval_low_support_test",
    "scripts/run_v5_sft_causal_eval.py",
)


class V53LowSupportEvaluationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry_path = self.root / "checkpoint_registry.json"

    def tearDown(self):
        self.temp.cleanup()

    def low_registry(self):
        entries = {
            "base_model": {
                "model_id": EVALUATION.PROFILE_MODEL_IDS[
                    EVALUATION.V5_3_LOW_SUPPORT_PROFILE
                ]["base_model"],
                "adapter_sha256": None,
                "adapter_config_sha256": None,
                "training_run_manifest_sha256": None,
            },
            "perfect_success": {
                "model_id": EVALUATION.PROFILE_MODEL_IDS[
                    EVALUATION.V5_3_LOW_SUPPORT_PROFILE
                ]["perfect_success"],
                "adapter_sha256": "1" * 64,
                "adapter_config_sha256": "2" * 64,
                "training_run_manifest_sha256": "3" * 64,
            },
            "repair_50": {
                "model_id": EVALUATION.PROFILE_MODEL_IDS[
                    EVALUATION.V5_3_LOW_SUPPORT_PROFILE
                ]["repair_50"],
                "adapter_sha256": "4" * 64,
                "adapter_config_sha256": "5" * 64,
                "training_run_manifest_sha256": "6" * 64,
            },
        }
        dynamic_common = {
            "protocol": "v5_stage1_dynamic_injection_audit",
            "split_manifest_sha256": "7" * 64,
            "official_test_used": False,
            "official_test_sealed": True,
        }
        return {
            "protocol": EVALUATION.CHECKPOINT_REGISTRY_PROTOCOL,
            "provenance_profile": EVALUATION.V5_3_LOW_SUPPORT_PROFILE,
            "source_commit": "8" * 40,
            "base_model_revision": "9" * 40,
            "training_data_provenance": {
                "data_audit_sha256": "a" * 64,
                "data_hashes_sha256": "b" * 64,
                "design_version": (
                    EVALUATION.V5_3_LOW_SUPPORT_DESIGN_VERSION
                ),
                "design_provenance": {
                    "design_version": (
                        EVALUATION.V5_3_LOW_SUPPORT_DESIGN_VERSION
                    ),
                    "design_protocol": (
                        EVALUATION.V5_3_LOW_SUPPORT_DESIGN_PROTOCOL
                    ),
                    "diagnostic_protocol": (
                        EVALUATION.V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL
                    ),
                    "validation_tasks": 21,
                    "trained_arms": ["perfect_success", "repair_50"],
                    "eligible_distinct_tasks": 8,
                    "eligible_capped_pairs": 10,
                    "maximum_pairs_per_task": 2,
                    "schedule_rows_per_arm": 512,
                    "repeated_schedule_rows_are_independent_examples": False,
                    "processing_source_commit": "8" * 40,
                    "source_generation_commit": "7" * 40,
                    "formal_v5_3_result": False,
                    "official_test_used": False,
                    "official_test_sealed": True,
                },
                "dynamic_audits": {
                    "generation": {
                        **dynamic_common,
                        "sha256": "c" * 64,
                        "manifest_sha256": "d" * 64,
                        "source_split": "derived_inner_train",
                        "verified_injections": 78,
                    },
                    "validation": {
                        **dynamic_common,
                        "sha256": "e" * 64,
                        "manifest_sha256": "f" * 64,
                        "source_split": "derived_validation",
                        "verified_injections": 21,
                    },
                },
                "official_test_used": False,
                "official_test_sealed": True,
            },
            "entries": entries,
            "diagnostic_evaluator": dict(
                EVALUATION.V5_3_LOW_SUPPORT_USER_JUDGE
            ),
            "diagnostic_claim_boundary": dict(
                EVALUATION.V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
            ),
        }

    def write_registry(self, payload):
        self.registry_path.write_text(json.dumps(payload), encoding="utf-8")
        return self.registry_path

    def test_exact_three_entry_registry_is_accepted(self):
        payload = self.low_registry()
        loaded = EVALUATION.load_checkpoint_registry(
            self.write_registry(payload),
            expected_profile=EVALUATION.V5_3_LOW_SUPPORT_PROFILE,
        )
        self.assertEqual(
            tuple(sorted(loaded["entries"])),
            ("base_model", "perfect_success", "repair_50"),
        )
        self.assertEqual(
            EVALUATION.registry_served_aliases(loaded),
            [
                "v5-3-low-support-base",
                "v5-3-low-support-perfect-success",
                "v5-3-low-support-repair-50",
            ],
        )
        self.assertEqual(
            REGISTRY.V5_3_LOW_SUPPORT_CLAIM_BOUNDARY,
            EVALUATION.V5_3_LOW_SUPPORT_CLAIM_BOUNDARY,
        )

    def test_registry_rejects_low_support_subsets_and_supersets(self):
        payload = self.low_registry()
        del payload["entries"]["repair_50"]
        with self.assertRaisesRegex(RuntimeError, "must be exactly"):
            EVALUATION.load_checkpoint_registry(
                self.write_registry(payload),
                expected_profile=EVALUATION.V5_3_LOW_SUPPORT_PROFILE,
            )

        payload = self.low_registry()
        payload["entries"]["failure_raw"] = {
            "model_id": "openai/v5-3-low-support-failure-raw",
            "adapter_sha256": "1" * 64,
            "adapter_config_sha256": "2" * 64,
            "training_run_manifest_sha256": "3" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "must be exactly"):
            EVALUATION.load_checkpoint_registry(
                self.write_registry(payload),
                expected_profile=EVALUATION.V5_3_LOW_SUPPORT_PROFILE,
            )

    def test_trained_bindings_are_exact_and_formal_profiles_unchanged(self):
        low_values = [
            "perfect_success=/tmp/low-perfect",
            "repair_50=/tmp/low-repair",
        ]
        self.assertEqual(
            set(
                REGISTRY.parse_arm_bindings(
                    low_values,
                    provenance_profile=REGISTRY.V5_3_LOW_SUPPORT_PROFILE,
                )
            ),
            {"perfect_success", "repair_50"},
        )
        with self.assertRaisesRegex(RuntimeError, "must be exactly"):
            REGISTRY.parse_arm_bindings(
                low_values[:1],
                provenance_profile=REGISTRY.V5_3_LOW_SUPPORT_PROFILE,
            )
        with self.assertRaisesRegex(RuntimeError, "must be exactly"):
            REGISTRY.parse_arm_bindings(
                [*low_values, "failure_raw=/tmp/low-failure"],
                provenance_profile=REGISTRY.V5_3_LOW_SUPPORT_PROFILE,
            )

        low_expected = EVALUATION.trained_arms_for_profile(
            EVALUATION.V5_3_LOW_SUPPORT_PROFILE
        )
        self.assertEqual(
            set(
                EVALUATION.parse_adapter_dir_bindings(
                    low_values,
                    expected_arms=low_expected,
                )
            ),
            {"perfect_success", "repair_50"},
        )
        with self.assertRaisesRegex(RuntimeError, "exactly"):
            EVALUATION.parse_adapter_dir_bindings(
                low_values[:1],
                expected_arms=low_expected,
            )
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            EVALUATION.parse_adapter_dir_bindings(
                [*low_values, "failure_raw=/tmp/low-failure"],
                expected_arms=low_expected,
            )

        self.assertEqual(
            REGISTRY.trained_arms_for_profile("v5_3"),
            REGISTRY.ARMS,
        )
        self.assertEqual(
            EVALUATION.arms_for_profile("v5_3"),
            EVALUATION.ARMS,
        )
        formal_values = [
            f"{arm}=/tmp/formal-{arm}"
            for arm in sorted(EVALUATION.TRAINED_ARMS)
        ]
        self.assertEqual(
            set(EVALUATION.parse_adapter_dir_bindings(formal_values)),
            EVALUATION.TRAINED_ARMS,
        )

    def test_low_support_decoding_roles_and_contract_are_frozen(self):
        valid = SimpleNamespace(
            max_tokens=512,
            max_steps=60,
            timeout=900.0,
            seed=20260731,
            num_trials=1,
            num_shards=3,
            condition="both",
            provenance_profile=EVALUATION.V5_3_LOW_SUPPORT_PROFILE,
        )
        EVALUATION.validate_evaluation_protocol(valid)
        invalid = SimpleNamespace(**vars(valid))
        invalid.seed = 20260722
        with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
            EVALUATION.validate_evaluation_protocol(invalid)
        invalid_shards = SimpleNamespace(**vars(valid))
        invalid_shards.num_shards = 1
        with self.assertRaisesRegex(RuntimeError, "exactly 3 shards"):
            EVALUATION.validate_evaluation_protocol(invalid_shards)

        registry = self.low_registry()
        entry = EVALUATION.validate_checkpoint_identity(
            registry,
            arm="base_model",
            agent_model=registry["entries"]["base_model"]["model_id"],
            local_source_commit=registry["source_commit"],
            user_model=EVALUATION.V5_3_LOW_SUPPORT_USER_JUDGE["model_id"],
            judge_model=EVALUATION.V5_3_LOW_SUPPORT_USER_JUDGE["model_id"],
        )
        self.assertEqual(entry, registry["entries"]["base_model"])
        with self.assertRaisesRegex(RuntimeError, "pinned 14B"):
            EVALUATION.validate_checkpoint_identity(
                registry,
                arm="base_model",
                agent_model=registry["entries"]["base_model"]["model_id"],
                local_source_commit=registry["source_commit"],
                user_model=registry["entries"]["base_model"]["model_id"],
                judge_model=registry["entries"]["base_model"]["model_id"],
            )

        contract = {
            "checkpoint_registry_provenance_profile": (
                EVALUATION.V5_3_LOW_SUPPORT_PROFILE
            ),
            "diagnostic_protocol": (
                EVALUATION.V5_3_LOW_SUPPORT_EXPERIMENT_PROTOCOL
            ),
            "diagnostic_claim_boundary": dict(
                EVALUATION.V5_3_LOW_SUPPORT_CLAIM_BOUNDARY
            ),
            "tool_action_interface": dict(
                EVALUATION.TOOL_ACTION_INTERFACE
            ),
            "runtime_preflight": {
                "served_context_window_tokens": 32768,
                "request_max_tokens": 512,
                "maximum_nonoverflow_prompt_tokens": 32256,
                "max_steps": 60,
                "request_token_overflow_policy": "fail_closed",
                "longest_prompt_observation": (
                    "completion_audit.max_observed_prompt_tokens"
                ),
            },
            "decoding": dict(
                EVALUATION.V5_3_LOW_SUPPORT_FROZEN_DECODING
            ),
            "judge": {
                "strict_backend": {
                    "module": EVALUATION.STRICT_NL_JUDGE_MODULE,
                    "entrypoint": EVALUATION.STRICT_NL_JUDGE_ENTRYPOINT,
                    "mode": "strict_json_schema_fail_closed",
                }
            },
            "status": "INCOMPLETE",
            "result_sha256": {},
            "completion_audit": {},
            "strict_judge_audit_evidence": {},
        }
        contract["contract_core_sha256"] = (
            EVALUATION.contract_core_sha256(contract)
        )
        EVALUATION.validate_contract_core(contract)

        drift = copy.deepcopy(contract)
        drift["diagnostic_claim_boundary"]["formal_v5_3_result"] = True
        drift["contract_core_sha256"] = EVALUATION.contract_core_sha256(drift)
        with self.assertRaisesRegex(RuntimeError, "diagnostic identity drift"):
            EVALUATION.validate_contract_core(drift)

    def test_summary_reports_low_unique_training_support(self):
        registry = self.low_registry()
        support = SUMMARY._training_support(registry)
        self.assertEqual(
            support,
            {
                "eligible_distinct_tasks": 8,
                "eligible_capped_pairs": 10,
                "maximum_pairs_per_task": 2,
                "schedule_rows_per_arm": 512,
                "repeated_schedule_rows_are_independent_examples": False,
                "minimum_distinct_tasks": 8,
                "minimum_capped_pairs": 10,
            },
        )
        registry["training_data_provenance"]["design_provenance"][
            "eligible_distinct_tasks"
        ] = 7
        with self.assertRaisesRegex(
            SUMMARY.SummaryError,
            "counts/limits drift",
        ):
            SUMMARY._training_support(registry)

    def test_incomplete_summary_never_poisons_canonical_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic_summary.json"
            receipt = SUMMARY.write_incomplete_receipt(
                output, RuntimeError("evidence incomplete")
            )
            self.assertFalse(output.exists())
            self.assertEqual(
                receipt.name, "diagnostic_summary.incomplete.json"
            )
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8"))["status"],
                "INCOMPLETE_NO_CLAIM",
            )

    @mock.patch.object(SUMMARY.subprocess, "run")
    def test_summary_source_provenance_requires_clean_registered_head(
        self,
        run,
    ):
        run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(stdout="8" * 40 + "\n"),
        ]
        provenance = SUMMARY._summary_source_provenance(
            self.low_registry()
        )
        self.assertEqual(
            provenance,
            {
                "summarizer_source_commit": "8" * 40,
                "summarizer_tracked_source_clean": True,
            },
        )

        run.reset_mock()
        run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(stdout="7" * 40 + "\n"),
        ]
        with self.assertRaisesRegex(
            SUMMARY.SummaryError,
            "source commit drift",
        ):
            SUMMARY._summary_source_provenance(self.low_registry())

        run.reset_mock()
        run.side_effect = [SimpleNamespace(returncode=1)]
        with self.assertRaisesRegex(
            SUMMARY.SummaryError,
            "tracked summarizer drift",
        ):
            SUMMARY._summary_source_provenance(self.low_registry())


if __name__ == "__main__":
    unittest.main()
