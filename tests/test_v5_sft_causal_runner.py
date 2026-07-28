import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

from tests.v5_multifault_fixtures import (
    SOURCE_FILES,
    TAU2_COMMIT,
    fault_protocol,
    multifault_manifest,
    multifault_row,
    write_dynamic_audit,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v5_sft_causal_eval.py"
SPEC = importlib.util.spec_from_file_location("run_v5_sft_causal_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def manifest(rows, split_sha):
    return multifault_manifest(
        rows,
        protocol=MODULE.PROTOCOL,
        source_split="derived_validation",
        split_sha256=split_sha,
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


def validation_rows():
    return [
        *[row("retail", index) for index in range(100, 115)],
        *[row("airline", index) for index in range(100, 106)],
    ]


SOURCE_COMMIT = "a" * 40
BASE_REVISION = "b" * 40


def checkpoint_registry():
    model_ids = MODULE.PROFILE_MODEL_IDS["legacy"]
    entries = {
        "base_model": {
            "model_id": model_ids["base_model"],
            "adapter_sha256": None,
            "adapter_config_sha256": None,
            "training_run_manifest_sha256": None,
        }
    }
    for index, arm in enumerate(sorted(MODULE.TRAINED_ARMS), start=1):
        entries[arm] = {
            "model_id": model_ids[arm],
            "adapter_sha256": f"{index:x}" * 64,
            "adapter_config_sha256": f"{index + 8:x}" * 64,
            "training_run_manifest_sha256": f"{index + 4:x}" * 64,
        }
    return {
        "protocol": MODULE.CHECKPOINT_REGISTRY_PROTOCOL,
        "source_commit": SOURCE_COMMIT,
        "base_model_revision": BASE_REVISION,
        "training_data_provenance": {
            "data_audit_sha256": "7" * 64,
            "data_hashes_sha256": "8" * 64,
            "dynamic_audits": {
                "generation": {
                    "protocol": "v5_stage1_dynamic_injection_audit",
                    "sha256": "9" * 64,
                    "manifest_sha256": "a" * 64,
                    "split_manifest_sha256": "b" * 64,
                    "source_split": "derived_inner_train",
                    "verified_injections": 83,
                    "official_test_used": False,
                    "official_test_sealed": True,
                },
                "validation": {
                    "protocol": "v5_stage1_dynamic_injection_audit",
                    "sha256": "c" * 64,
                    "manifest_sha256": "d" * 64,
                    "split_manifest_sha256": "b" * 64,
                    "source_split": "derived_validation",
                    "verified_injections": 21,
                    "official_test_used": False,
                    "official_test_sealed": True,
                },
            },
            "official_test_used": False,
            "official_test_sealed": True,
        },
        "entries": entries,
    }


def row(domain, task_id):
    return multifault_row(
        domain,
        task_id,
        source_split="derived_validation",
    )


class V5SFTCausalRunnerTests(unittest.TestCase):
    def write_manifest(self, rows):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "manifest.json"
        split_path = Path(directory.name) / "split_manifest.json"
        split_path.write_text(json.dumps(split_manifest()), encoding="utf-8")
        value = manifest(rows, MODULE.sha256_file(split_path))
        path.write_text(json.dumps(value), encoding="utf-8")
        return directory, path, split_path, value

    def write_registry(self, value=None):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "checkpoint_registry.json"
        path.write_text(
            json.dumps(checkpoint_registry() if value is None else value),
            encoding="utf-8",
        )
        return directory, path

    def write_adapters(self, registry):
        directory = tempfile.TemporaryDirectory()
        bindings = {}
        for arm in sorted(MODULE.TRAINED_ARMS):
            adapter_dir = Path(directory.name) / arm
            adapter_dir.mkdir()
            adapter = adapter_dir / "adapter_model.safetensors"
            config = adapter_dir / "adapter_config.json"
            adapter.write_bytes(f"adapter:{arm}".encode())
            config.write_text(json.dumps({"peft_type": "LORA", "arm": arm}))
            registry["entries"][arm]["adapter_sha256"] = MODULE.sha256_file(
                adapter
            )
            registry["entries"][arm][
                "adapter_config_sha256"
            ] = MODULE.sha256_file(config)
            bindings[arm] = adapter_dir
        return directory, bindings

    def test_manifest_accepts_validation_and_rejects_test_use(self):
        directory, path, split_path, value = self.write_manifest(validation_rows())
        self.addCleanup(directory.cleanup)
        split = MODULE.load_split_manifest(split_path)
        loaded = MODULE.load_manifest(
            path,
            split_manifest=split,
            split_manifest_sha256=MODULE.sha256_file(split_path),
        )
        self.assertEqual(loaded["paired_task_count"], 21)

        value["official_test_used"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "seal official test"):
            MODULE.load_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

    def test_manifest_rejects_unsafe_or_duplicate_rows(self):
        duplicate = row("retail", "1")
        directory, path, split_path, value = self.write_manifest(
            [duplicate, dict(duplicate)]
        )
        self.addCleanup(directory.cleanup)
        split = MODULE.load_split_manifest(split_path)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            MODULE.load_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

        unsafe = row("retail", "3")
        unsafe["error_condition"]["expected_state_mutation"] = True
        value = manifest([unsafe], MODULE.sha256_file(split_path))
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            MODULE.load_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

    def test_official_test_and_split_hash_fail_before_execution(self):
        rows = validation_rows()
        rows[0] = row("retail", "200")
        directory, path, split_path, value = self.write_manifest(rows)
        self.addCleanup(directory.cleanup)
        split = MODULE.load_split_manifest(split_path)
        with self.assertRaisesRegex(RuntimeError, "Official-test"):
            MODULE.load_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

        value["rows"] = validation_rows()
        value["split_manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "split_manifest_sha256"):
            MODULE.load_manifest(
                path,
                split_manifest=split,
                split_manifest_sha256=MODULE.sha256_file(split_path),
            )

    def test_shards_are_balanced_disjoint_and_complete(self):
        rows = [row("retail", index) for index in range(21)]
        shards = [MODULE.shard_rows(rows, index, 4) for index in range(4)]
        flattened = [item["pair_id"] for shard in shards for item in shard]
        self.assertEqual(len(flattened), 21)
        self.assertEqual(len(set(flattened)), 21)
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_shard_arguments_fail_closed(self):
        rows = [row("retail", "1")]
        with self.assertRaises(ValueError):
            MODULE.shard_rows(rows, 0, 0)
        with self.assertRaises(ValueError):
            MODULE.shard_rows(rows, 2, 2)

    def test_output_filenames_are_contractual(self):
        self.assertEqual(
            MODULE.output_filename("retail", "clean", 0, 1),
            "retail_clean.json",
        )
        self.assertEqual(
            MODULE.output_filename("airline", "error", 2, 4),
            "airline_error.shard-002-of-004.json",
        )

    def test_checkpoint_registry_rejects_shared_aliases_and_identity_drift(self):
        value = checkpoint_registry()
        for entry in value["entries"].values():
            entry["model_id"] = "openai/same-alias"
        directory, path = self.write_registry(value)
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(RuntimeError, "model_id aliases must be unique"):
            MODULE.load_checkpoint_registry(path)

        valid = checkpoint_registry()
        path.write_text(json.dumps(valid), encoding="utf-8")
        loaded = MODULE.load_checkpoint_registry(path)
        base_alias = loaded["entries"]["base_model"]["model_id"]
        with self.assertRaisesRegex(RuntimeError, "does not match registry"):
            MODULE.validate_checkpoint_identity(
                loaded,
                arm="repair_50",
                agent_model="wrong-alias",
                local_source_commit=SOURCE_COMMIT,
                user_model=base_alias,
                judge_model=base_alias,
            )
        with self.assertRaisesRegex(RuntimeError, "Local source commit"):
            MODULE.validate_checkpoint_identity(
                loaded,
                arm="repair_50",
                agent_model=loaded["entries"]["repair_50"]["model_id"],
                local_source_commit="f" * 40,
                user_model=base_alias,
                judge_model=base_alias,
            )

    def test_checkpoint_registry_requires_litellm_provider_route(self):
        value = checkpoint_registry()
        value["entries"]["base_model"]["model_id"] = "v5-base"
        directory, path = self.write_registry(value)
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(RuntimeError, "openai/ LiteLLM"):
            MODULE.load_checkpoint_registry(path)

    def test_user_and_judge_must_equal_registry_base_alias(self):
        directory, path = self.write_registry()
        self.addCleanup(directory.cleanup)
        loaded = MODULE.load_checkpoint_registry(path)
        base_alias = loaded["entries"]["base_model"]["model_id"]
        entry = MODULE.validate_checkpoint_identity(
            loaded,
            arm="base_model",
            agent_model=base_alias,
            local_source_commit=SOURCE_COMMIT,
            user_model=base_alias,
            judge_model=base_alias,
        )
        self.assertEqual(entry, loaded["entries"]["base_model"])

        adapter_alias = loaded["entries"]["repair_100"]["model_id"]
        with self.assertRaisesRegex(RuntimeError, "base_model alias"):
            MODULE.validate_checkpoint_identity(
                loaded,
                arm="base_model",
                agent_model=base_alias,
                local_source_commit=SOURCE_COMMIT,
                user_model=adapter_alias,
                judge_model=base_alias,
            )

    def test_local_adapter_bindings_verify_both_registry_hashes(self):
        registry = checkpoint_registry()
        directory, bindings = self.write_adapters(registry)
        self.addCleanup(directory.cleanup)
        identity = MODULE.validate_local_adapter_identity(registry, bindings)
        self.assertEqual(set(identity), MODULE.TRAINED_ARMS)
        self.assertNotIn(str(Path(directory.name)), json.dumps(identity))

        target = bindings["repair_50"] / "adapter_config.json"
        target.write_text('{"peft_type":"LORA","drift":true}')
        with self.assertRaisesRegex(
            RuntimeError, "adapter_config_sha256 differs"
        ):
            MODULE.validate_local_adapter_identity(registry, bindings)

    def test_adapter_cli_bindings_are_exact_and_unique(self):
        values = [f"{arm}=/tmp/{arm}" for arm in sorted(MODULE.TRAINED_ARMS)]
        self.assertEqual(
            set(MODULE.parse_adapter_dir_bindings(values)),
            MODULE.TRAINED_ARMS,
        )
        with self.assertRaisesRegex(RuntimeError, "exactly"):
            MODULE.parse_adapter_dir_bindings(values[:-1])
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            MODULE.parse_adapter_dir_bindings([*values, values[0]])

    def test_shared_endpoint_and_five_served_aliases_are_required(self):
        self.assertEqual(
            MODULE.require_shared_api_base(
                "http://127.0.0.1:8000/v1/",
                "http://127.0.0.1:8000/v1",
                "http://127.0.0.1:8000/v1/",
            ),
            "http://127.0.0.1:8000/v1",
        )
        with self.assertRaisesRegex(RuntimeError, "same API base"):
            MODULE.require_shared_api_base(
                "http://agent/v1", "http://shared/v1", "http://shared/v1"
            )

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode()

        registry = checkpoint_registry()
        expected = MODULE.registry_served_aliases(registry)
        with patch.object(
            MODULE,
            "urlopen",
            return_value=Response(
                {"data": [{"id": alias} for alias in expected]}
            ),
        ):
            self.assertEqual(
                MODULE.verify_served_registry_aliases(
                    "http://127.0.0.1:8000/v1", "secret", registry
                ),
                expected,
            )
        with patch.object(
            MODULE,
            "urlopen",
            return_value=Response(
                {"data": [{"id": alias} for alias in expected[:-1]]}
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                MODULE.verify_served_registry_aliases(
                    "http://127.0.0.1:8000/v1", "secret", registry
                )

    def test_evaluation_protocol_is_exact_not_merely_cross_run_consistent(self):
        valid = SimpleNamespace(
            max_tokens=512,
            max_steps=60,
            timeout=900.0,
            seed=20260722,
            num_trials=1,
            condition="both",
        )
        MODULE.validate_evaluation_protocol(valid)
        invalid = SimpleNamespace(**vars(valid))
        invalid.max_tokens = 511
        with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
            MODULE.validate_evaluation_protocol(invalid)
        invalid = SimpleNamespace(**vars(valid))
        invalid.condition = "clean"
        with self.assertRaisesRegex(RuntimeError, "both conditions"):
            MODULE.validate_evaluation_protocol(invalid)

    def test_endpoint_and_response_normalization_match_generation(self):
        endpoint = MODULE.endpoint_args(
            "http://127.0.0.1:8100/v1",
            "local",
            max_tokens=512,
            seed=20260722,
        )
        self.assertIs(endpoint["parallel_tool_calls"], False)
        first = {"id": "first", "name": "lookup", "arguments": {"id": 1}}
        deferred = {
            "id": "second",
            "name": "update",
            "arguments": {"id": 1},
        }
        message = SimpleNamespace(
            tool_calls=[first, deferred],
            content="I will call both tools.",
            raw_data={"provider": "vllm"},
        )
        normalized = MODULE.normalize_tool_only_message(message)
        self.assertEqual(normalized.tool_calls, [first])
        self.assertIsNone(normalized.content)
        parallel = normalized.raw_data[
            "v5_stage1_parallel_calls_serialized"
        ]
        self.assertEqual(parallel["original_count"], 2)
        self.assertEqual(
            parallel["deferred_call_sha256"],
            [MODULE._sha256_canonical_payload(deferred)],
        )
        mixed = normalized.raw_data["v5_stage1_mixed_content_normalized"]
        self.assertEqual(
            mixed["utf8_bytes"],
            len("I will call both tools.".encode("utf-8")),
        )
        self.assertRegex(mixed["sha256"], r"^[0-9a-f]{64}$")

    def test_32k_request_budget_fails_closed_on_one_token_overflow(self):
        self.assertEqual(
            MODULE.validate_request_token_budget(
                32256,
                max_tokens=512,
            ),
            32768,
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            MODULE.validate_request_token_budget(
                32257,
                max_tokens=512,
            )
        with self.assertRaisesRegex(RuntimeError, "max_tokens"):
            MODULE.endpoint_args(
                "http://127.0.0.1:8100/v1",
                "local",
                max_tokens=32769,
                seed=20260722,
            )
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "retail_clean.json"
            result.write_text(
                json.dumps(
                    {
                        "simulations": [
                            {
                                "task_id": "100",
                                "termination_reason": "agent_stop",
                                "messages": [
                                    {
                                        "role": "assistant",
                                        "content": "done",
                                        "usage": {"prompt_tokens": 32257},
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                MODULE.audit_result_interface(
                    result,
                    expected_task_ids=["100"],
                    num_trials=1,
                    max_tokens=512,
                )

    def test_context_overflow_compacts_one_complete_oldest_turn(self):
        messages = [
            SimpleNamespace(role="user", content="old request"),
            SimpleNamespace(role="assistant", content="old response"),
            SimpleNamespace(role="tool", content="old tool result"),
            SimpleNamespace(role="user", content="new request"),
            SimpleNamespace(role="assistant", content="new response"),
        ]
        audit = MODULE.compact_oldest_complete_turn(messages)
        self.assertEqual(
            [(message.role, message.content) for message in messages],
            [
                ("user", "new request"),
                ("assistant", "new response"),
            ],
        )
        self.assertEqual(audit["strategy"], "drop_oldest_complete_turn")
        self.assertEqual(audit["removed_messages"], 3)
        self.assertEqual(len(audit["removed_sha256"]), 3)
        for digest in audit["removed_sha256"]:
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_context_compaction_fails_closed_without_safe_boundary(self):
        messages = [
            SimpleNamespace(role="user", content="only request"),
            SimpleNamespace(role="assistant", content="only response"),
        ]
        with self.assertRaisesRegex(RuntimeError, "second user-turn boundary"):
            MODULE.compact_oldest_complete_turn(messages)
        self.assertEqual(len(messages), 2)
        self.assertTrue(
            MODULE._is_context_window_error(
                RuntimeError(
                    "maximum context length is 32768 tokens; "
                    "request has 33002 input tokens"
                )
            )
        )
        self.assertFalse(MODULE._is_context_window_error(RuntimeError("other")))

    def test_result_interface_rejects_unserialized_or_mixed_tool_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retail_clean.json"
            base = {
                "task_id": "100",
                "termination_reason": "agent_stop",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "one", "name": "lookup", "arguments": {}},
                            {"id": "two", "name": "lookup", "arguments": {}},
                        ],
                        "usage": {"prompt_tokens": 100},
                    }
                ],
            }
            path.write_text(
                json.dumps({"simulations": [base]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "multiple parallel"):
                MODULE.audit_result_interface(
                    path,
                    expected_task_ids=["100"],
                    num_trials=1,
                    max_tokens=512,
                )
            base["messages"][0]["tool_calls"] = [
                {"id": "one", "name": "lookup", "arguments": {}}
            ]
            base["messages"][0]["content"] = "hidden prose"
            path.write_text(
                json.dumps({"simulations": [base]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "mixed text"):
                MODULE.audit_result_interface(
                    path,
                    expected_task_ids=["100"],
                    num_trials=1,
                    max_tokens=512,
                )

    def test_contract_core_binds_interface_preflight_and_strict_judge(self):
        payload = {
            "tool_action_interface": dict(MODULE.TOOL_ACTION_INTERFACE),
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
            "decoding": dict(MODULE.FROZEN_DECODING),
            "judge": {
                "strict_backend": {
                    "module": "v5_strict_nl_judge",
                    "entrypoint": "install_strict_nl_judge",
                    "mode": "strict_json_schema_fail_closed",
                }
            },
            "status": "INCOMPLETE",
            "result_sha256": {},
            "completion_audit": {},
            "strict_judge_audit_evidence": {},
        }
        payload["contract_core_sha256"] = MODULE.contract_core_sha256(payload)
        MODULE.validate_contract_core(payload)
        core_hash = payload["contract_core_sha256"]
        payload["strict_judge_audit_evidence"] = {
            "protocol": "v5_strict_nl_judge_evidence_v1",
            "status": "PASS",
        }
        self.assertEqual(MODULE.contract_core_sha256(payload), core_hash)
        MODULE.validate_contract_core(payload)
        payload["tool_action_interface"]["parallel_tool_calls"] = True
        with self.assertRaisesRegex(RuntimeError, "core hash drift"):
            MODULE.validate_contract_core(payload)

    def test_strict_nl_judge_is_lazily_installed_and_missing_is_fatal(self):
        installer = mock.Mock()
        strict_module = SimpleNamespace(install_strict_nl_judge=installer)
        with patch.object(
            MODULE.importlib,
            "import_module",
            return_value=strict_module,
        ):
            identity = MODULE.patch_local_nl_judge(
                "openai/v5-base",
                {"api_base": "http://localhost/v1"},
            )
        installer.assert_called_once_with(
            model="openai/v5-base",
            llm_args={"api_base": "http://localhost/v1"},
        )
        self.assertEqual(identity["mode"], "strict_json_schema_fail_closed")
        missing = ModuleNotFoundError(
            "No module named 'v5_strict_nl_judge'",
            name="v5_strict_nl_judge",
        )
        with patch.object(
            MODULE.importlib,
            "import_module",
            side_effect=missing,
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation is deferred"):
                MODULE.patch_local_nl_judge("openai/v5-base", {})

    def test_run_contract_records_complete_checkpoint_identity_chain(self):
        directory, manifest_path, split_path, _ = self.write_manifest(
            validation_rows()
        )
        self.addCleanup(directory.cleanup)
        registry_dir, registry_path = self.write_registry()
        self.addCleanup(registry_dir.cleanup)
        registry = MODULE.load_checkpoint_registry(registry_path)
        base_alias = registry["entries"]["base_model"]["model_id"]
        output_dir = Path(directory.name) / "result"
        args = SimpleNamespace(
            output_dir=output_dir,
            num_shards=1,
            shard_index=0,
            arm="repair_50",
            agent_model=registry["entries"]["repair_50"]["model_id"],
            agent_api_base="http://shared",
            user_model=base_alias,
            user_api_base="http://shared",
            judge_model=base_alias,
            judge_api_base="http://shared",
            max_tokens=512,
            max_steps=60,
            timeout=900.0,
            seed=20260722,
            num_trials=1,
            condition="both",
        )
        dynamic_audit_path = Path(directory.name) / "dynamic_audit.json"
        write_dynamic_audit(
            dynamic_audit_path,
            manifest_path=manifest_path,
            split_manifest_path=split_path,
        )
        dynamic_audit_identity = MODULE.load_complete_dynamic_audit(
            dynamic_audit_path,
            manifest_path=manifest_path,
            split_manifest_path=split_path,
            expected_source_split="derived_validation",
            expected_task_ids={
                f"{row['domain']}:{row['task_id']}"
                for row in validation_rows()
            },
        )
        contract_path = MODULE.write_contract(
            args,
            manifest_path,
            split_path,
            registry_path,
            registry,
            registry["entries"]["repair_50"],
            SOURCE_COMMIT,
            validation_rows(),
            dynamic_audit_path=dynamic_audit_path,
            dynamic_audit_identity=dynamic_audit_identity,
            local_adapter_identity={
                arm: {
                    "adapter_sha256": registry["entries"][arm][
                        "adapter_sha256"
                    ],
                    "adapter_config_sha256": registry["entries"][arm][
                        "adapter_config_sha256"
                    ],
                }
                for arm in sorted(MODULE.TRAINED_ARMS)
            },
            served_registry_aliases=MODULE.registry_served_aliases(registry),
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(contract["status"], "INCOMPLETE")
        self.assertEqual(contract["result_sha256"], {})
        self.assertEqual(
            contract["checkpoint_registry_sha256"],
            MODULE.sha256_file(registry_path),
        )
        self.assertEqual(contract["source_commit"], SOURCE_COMMIT)
        self.assertEqual(
            contract["checkpoint_entry"],
            registry["entries"]["repair_50"],
        )
        self.assertEqual(
            contract["evaluation_manifest_sha256"],
            MODULE.sha256_file(manifest_path),
        )
        self.assertEqual(contract["fault_protocol"], fault_protocol())
        self.assertEqual(contract["tau2_commit"], TAU2_COMMIT)
        self.assertEqual(contract["source_files"], SOURCE_FILES)
        self.assertEqual(
            contract["dynamic_audit_identity"], dynamic_audit_identity
        )
        self.assertEqual(
            set(contract["locally_verified_adapter_identity"]),
            MODULE.TRAINED_ARMS,
        )
        self.assertEqual(
            contract["served_registry_aliases"],
            MODULE.registry_served_aliases(registry),
        )
        result_paths = []
        for domain, task_ids in (
            ("retail", [str(index) for index in range(100, 115)]),
            ("airline", [str(index) for index in range(100, 106)]),
        ):
            for condition in ("clean", "error"):
                result_path = output_dir / f"{domain}_{condition}.json"
                simulations = [
                    {
                        "task_id": task_id,
                        "termination_reason": "agent_stop",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "done",
                                "tool_calls": None,
                                "usage": {
                                    "prompt_tokens": 128,
                                    "completion_tokens": 1,
                                },
                            }
                        ],
                    }
                    for task_id in task_ids
                ]
                result_path.write_text(
                    json.dumps({"simulations": simulations}),
                    encoding="utf-8",
                )
                result_paths.append(result_path)
        hashes = MODULE.finalize_contract(contract_path, result_paths)
        finalized = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(finalized["status"], "COMPLETE")
        self.assertEqual(finalized["result_sha256"], hashes)
        self.assertEqual(
            set(hashes),
            {
                "retail_clean.json",
                "retail_error.json",
                "airline_clean.json",
                "airline_error.json",
            },
        )
        self.assertEqual(finalized["completion_audit"]["status"], "PASS")
        self.assertTrue(
            finalized["completion_audit"][
                "all_expected_tasks_observed_once_per_trial"
            ]
        )


if __name__ == "__main__":
    unittest.main()
