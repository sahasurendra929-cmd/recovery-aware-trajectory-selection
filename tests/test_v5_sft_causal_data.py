import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
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
SCRIPT = ROOT / "scripts" / "prepare_v5_sft_causal.py"
SPEC = importlib.util.spec_from_file_location("prepare_v5_sft_causal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
EVAL_SCRIPT = ROOT / "scripts" / "run_v5_sft_causal_eval.py"
EVAL_SPEC = importlib.util.spec_from_file_location(
    "run_v5_sft_causal_eval_for_data_test", EVAL_SCRIPT
)
EVAL_MODULE = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(EVAL_MODULE)
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_v5_sft_causal.py"
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_v5_sft_causal_for_data_test", SUMMARY_SCRIPT
)
SUMMARY_MODULE = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(SUMMARY_MODULE)
TRAIN_SCRIPT = ROOT / "scripts" / "train_v5_sft_causal.py"
TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_v5_sft_causal_for_data_test", TRAIN_SCRIPT
)
TRAIN_MODULE = importlib.util.module_from_spec(TRAIN_SPEC)
assert TRAIN_SPEC.loader is not None
TRAIN_SPEC.loader.exec_module(TRAIN_MODULE)
MANIFEST_SCRIPT = ROOT / "scripts" / "prepare_v5_stage1_manifests.py"
MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "prepare_v5_stage1_manifests_for_data_test", MANIFEST_SCRIPT
)
MANIFEST_MODULE = importlib.util.module_from_spec(MANIFEST_SPEC)
assert MANIFEST_SPEC.loader is not None
MANIFEST_SPEC.loader.exec_module(MANIFEST_MODULE)


class FakeTokenizer:
    """Prefix-stable Qwen-like chat template sufficient for data tests."""

    @staticmethod
    def _size(message):
        role = message["role"]
        if role == "assistant":
            calls = message.get("tool_calls") or []
            payload = (
                json.dumps(calls, sort_keys=True)
                if calls
                else str(message.get("content") or "")
            )
            return 1 + max(1, len(payload) // 12)
        return 1 + max(1, len(str(message.get("content") or "")) // 40)

    def apply_chat_template(
        self, messages, *, tools, tokenize, add_generation_prompt
    ):
        assert tokenize is True
        tokens = [10] * (2 + len(json.dumps(tools, sort_keys=True)) // 80)
        for message in messages:
            role = message["role"]
            header = 900 if role == "assistant" else {
                "system": 100,
                "user": 200,
                "tool": 300,
            }[role]
            tokens.append(header)
            tokens.extend([header + 1] * (self._size(message) - 1))
        if add_generation_prompt:
            tokens.append(900)
        return tokens


def assistant_call(call_id, name, arguments, *, injected=False):
    row = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    if injected:
        row["raw_data"] = {"v5_stage1_injected_fault": True}
    return row


def tool_result(call_id, name, *, error, content):
    return {
        "role": "tool",
        "id": call_id,
        "name": name,
        "error": error,
        "content": content,
    }


def simulation(domain, task_id, trial, *, condition, final_success=True):
    suffix = "done " * (2 + 5 * trial)
    messages = [
        {"role": "assistant", "content": "Hello", "usage": None},
        {"role": "user", "content": f"task {task_id}"},
    ]
    if condition == "clean":
        messages.extend(
            [
                assistant_call(
                    f"provider-clean-{task_id}-{trial}",
                    "lookup",
                    {"value": f"valid-{task_id}-{'x' * (trial * 200)}"},
                ),
                tool_result(
                    f"provider-clean-{task_id}-{trial}",
                    "lookup",
                    error=False,
                    content="ok",
                ),
            ]
        )
    else:
        fault = fault_descriptor(
            domain,
            task_id,
            source_split="derived_inner_train",
        )
        messages.extend(
            [
                assistant_call(
                    f"v5-stage1-special-{task_id}-{trial}",
                    fault["tool_name"],
                    fault["arguments"],
                    injected=True,
                ),
                tool_result(
                    f"v5-stage1-special-{task_id}-{trial}",
                    fault["tool_name"],
                    error=True,
                    content="Error: not found",
                ),
                assistant_call(
                    f"provider-repair-{task_id}-{trial}",
                    "lookup",
                    {"value": f"valid-{task_id}-{'x' * (trial * 200)}"},
                ),
                tool_result(
                    f"provider-repair-{task_id}-{trial}",
                    "lookup",
                    error=False,
                    content="ok",
                ),
            ]
        )
    messages.append({"role": "assistant", "content": suffix, "usage": {}})
    return {
        "task_id": str(task_id),
        "trial": trial,
        "seed": 1000 + trial,
        "reward_info": {"reward": 1.0 if final_success else 0.0},
        "messages": messages,
    }


class V5SFTCausalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "out"
        self.split_path = self.root / "split.json"
        retail_validation = [str(index) for index in range(100, 115)]
        airline_validation = [str(index) for index in range(200, 206)]
        self.split = {
            "protocol": "v5_stage0_tau2_end_to_end",
            "guarantees": {
                "validation_derived_from_official_train_only": True,
                "official_test_task_content_exported": False,
            },
            "domains": {
                "retail": {
                    "inner_train_ids": ["1", "2"],
                    "validation_ids": retail_validation,
                    "sealed_test_ids": ["900"],
                },
                "airline": {
                    "inner_train_ids": ["3", "4"],
                    "validation_ids": airline_validation,
                    "sealed_test_ids": ["901"],
                },
            },
        }
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")
        generation_rows = multifault_rows(
            {
                domain: values["inner_train_ids"]
                for domain, values in self.split["domains"].items()
            },
            source_split="derived_inner_train",
        )
        validation_rows = multifault_rows(
            {
                domain: values["validation_ids"]
                for domain, values in self.split["domains"].items()
            },
            source_split="derived_validation",
        )
        self.generation_path = self.root / "generation_manifest.json"
        self.validation_path = self.root / "validation_manifest.json"
        self.generation_path.write_text(
            json.dumps(
                multifault_manifest(
                    generation_rows,
                    protocol="v5_stage1_multifault_data_construction",
                    source_split="derived_inner_train",
                    split_sha256=MODULE.sha256_file(self.split_path),
                )
            ),
            encoding="utf-8",
        )
        self.validation_path.write_text(
            json.dumps(
                multifault_manifest(
                    validation_rows,
                    protocol="v5_stage1_sft_causal_validation",
                    source_split="derived_validation",
                    split_sha256=MODULE.sha256_file(self.split_path),
                )
            ),
            encoding="utf-8",
        )
        self.generation_audit_path = self.root / "generation_dynamic_audit.json"
        self.validation_audit_path = self.root / "validation_dynamic_audit.json"
        write_dynamic_audit(
            self.generation_audit_path,
            manifest_path=self.generation_path,
            split_manifest_path=self.split_path,
        )
        write_dynamic_audit(
            self.validation_audit_path,
            manifest_path=self.validation_path,
            split_manifest_path=self.split_path,
        )
        for domain, tasks in (("retail", ("1", "2")), ("airline", ("3", "4"))):
            for condition in ("clean", "error"):
                rows = []
                for task_id in tasks:
                    for trial in range(4):
                        # One paired slot is a final failure and must be excluded.
                        rows.append(
                            simulation(
                                domain,
                                task_id,
                                trial,
                                condition=condition,
                                final_success=not (task_id == tasks[0] and trial == 3),
                            )
                        )
                for shard in range(2):
                    payload = {"simulations": rows[shard::2]}
                    path = (
                        self.raw
                        / f"{domain}_{condition}.shard-{shard:03d}-of-002.json"
                    )
                    path.write_text(json.dumps(payload), encoding="utf-8")
        schemas = []
        for tool_name, invalid_key in (
            ("lookup_user", "user_id"),
            ("lookup_order", "order_id"),
            ("lookup_reservation", "reservation_id"),
            ("lookup_flight", "flight_number"),
            ("lookup", "value"),
        ):
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "read-only lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {invalid_key: {"type": "string"}},
                            "required": [invalid_key],
                        },
                    },
                },
            )
        self.contexts = {
            "retail": {"policy": "retail policy", "tool_schemas": schemas},
            "airline": {"policy": "airline policy", "tool_schemas": schemas},
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self, destination=None):
        return MODULE.prepare(
            split_manifest=self.split_path,
            generation_manifest=self.generation_path,
            validation_manifest=self.validation_path,
            generation_dynamic_audit=self.generation_audit_path,
            validation_dynamic_audit=self.validation_audit_path,
            raw_dir=self.raw,
            output_dir=destination or self.output,
            tokenizer=FakeTokenizer(),
            tokenizer_name=MODULE.MODEL,
            tokenizer_revision="a" * 40,
            domain_contexts=self.contexts,
            schedule_rows=32,
            expected_trials=4,
            slots_per_task=3,
            min_distinct_tasks=4,
            min_paired_slots=12,
            include_optional_mixtures=True,
            token_tolerance=0.20,
            sequence_tolerance=0.20,
            strict_split_counts=False,
            strict_generation_contracts=False,
        )

    @staticmethod
    def read_jsonl(path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def strict_generation_fixture(self, name="strict-generation"):
        raw_dir = self.root / name
        raw_dir.mkdir()
        generation_manifest = self.root / f"{name}-manifest.json"
        tasks = [
            (domain, task_id)
            for domain in ("retail", "airline")
            for task_id in self.split["domains"][domain]["inner_train_ids"]
        ]
        manifest = multifault_manifest(
            multifault_rows(
                {
                    domain: self.split["domains"][domain]["inner_train_ids"]
                    for domain in ("retail", "airline")
                },
                source_split="derived_inner_train",
            ),
            protocol="v5_stage1_multifault_data_construction",
            source_split="derived_inner_train",
            split_sha256=MODULE.sha256_file(self.split_path),
        )
        generation_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        generation_dynamic_audit = self.root / f"{name}-dynamic-audit.json"
        write_dynamic_audit(
            generation_dynamic_audit,
            manifest_path=generation_manifest,
            split_manifest_path=self.split_path,
        )
        dynamic_audit_identity = MODULE.load_complete_dynamic_audit(
            generation_dynamic_audit,
            manifest_path=generation_manifest,
            split_manifest_path=self.split_path,
            expected_source_split="derived_inner_train",
            expected_task_ids={f"{domain}:{task_id}" for domain, task_id in tasks},
        )
        frozen = {
            "teacher": {
                "model": MODULE.TEACHER_MODEL,
                "revision": MODULE.TEACHER_REVISION,
                "api_base": "http://127.0.0.1:8000/v1",
                "mode": "ground_truth",
            },
            "user": {
                "model": MODULE.USER_JUDGE_MODEL,
                "revision": MODULE.USER_JUDGE_REVISION,
                "api_base": "http://127.0.0.1:8001/v1",
            },
            "judge": {
                "model": MODULE.USER_JUDGE_MODEL,
                "revision": MODULE.USER_JUDGE_REVISION,
                "api_base": "http://127.0.0.1:8001/v1",
            },
            "decoding": {
                "temperature": 0,
                "max_tokens": 512,
                "parallel_tool_calls": False,
                "mixed_tool_call_content_normalization": "drop_text_preserve_sha256",
                "max_steps": 60,
                "task_timeout_seconds": 900,
                "seed": MODULE.SEED,
            },
        }
        contracts = []
        results = {}
        for index, (domain, task_id) in enumerate(tasks):
            result_sha256 = {}
            for condition in ("clean", "error"):
                filename = (
                    f"{domain}_{condition}.shard-{index:03d}-of-004.json"
                )
                result_path = raw_dir / filename
                result_path.write_text(
                    json.dumps(
                        {
                            "simulations": [],
                            "fixture": f"{domain}:{task_id}:{condition}",
                        }
                    ),
                    encoding="utf-8",
                )
                results[filename] = result_path
                result_sha256[filename] = MODULE.sha256_file(result_path)
            contract = {
                "protocol": "v5_stage1_inner_train_generation_run",
                "status": "COMPLETE",
                "source_commit": "4" * 40,
                "manifest_sha256": MODULE.sha256_file(generation_manifest),
                "split_manifest_sha256": MODULE.sha256_file(self.split_path),
                "dynamic_audit_identity": dynamic_audit_identity,
                "fault_protocol": fault_protocol(),
                "tau2_commit": TAU2_COMMIT,
                "source_files": SOURCE_FILES,
                "source_split": "derived_inner_train",
                "official_test_used": False,
                "task_ids": [f"{domain}:{task_id}"],
                "shard_index": index,
                "num_shards": 4,
                "num_trials": 3,
                "result_sha256": result_sha256,
                **frozen,
            }
            contract_path = (
                raw_dir / f"run_contract.shard-{index:03d}-of-004.json"
            )
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            contracts.append(contract_path)
        return (
            raw_dir,
            generation_manifest,
            generation_dynamic_audit,
            dynamic_audit_identity,
            contracts,
            results,
        )

    def validate_strict_generation(
        self, raw_dir, generation_manifest, dynamic_audit_identity
    ):
        return MODULE.validate_generation_contracts(
            raw_dir=raw_dir,
            split_manifest=self.split_path,
            split=MODULE.load_split(self.split_path, strict_counts=False),
            generation_manifest=generation_manifest,
            expected_trials=3,
            expected_seed=MODULE.SEED,
            expected_source_commit="4" * 40,
            dynamic_audit_identity=dynamic_audit_identity,
            observed_source_commit="4" * 40,
        )

    def test_generation_and_processing_commits_are_audited_separately(self):
        raw_dir, generation_manifest, _, audit_identity, _, _ = (
            self.strict_generation_fixture("separate-source-commits")
        )
        audit = MODULE.validate_generation_contracts(
            raw_dir=raw_dir,
            split_manifest=self.split_path,
            split=MODULE.load_split(self.split_path, strict_counts=False),
            generation_manifest=generation_manifest,
            expected_trials=3,
            expected_seed=MODULE.SEED,
            expected_source_commit="5" * 40,
            expected_generation_source_commit="4" * 40,
            dynamic_audit_identity=audit_identity,
            observed_source_commit="5" * 40,
        )
        self.assertEqual(audit["source_commit"], "4" * 40)

    def test_builds_paired_pool_and_all_schedules(self):
        audit = self.run_prepare()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(
            audit["dynamic_audits"]["generation"]["sha256"],
            MODULE.sha256_file(self.generation_audit_path),
        )
        self.assertEqual(
            audit["dynamic_audits"]["validation"]["sha256"],
            MODULE.sha256_file(self.validation_audit_path),
        )
        self.assertEqual(audit["paired_master_slots"], 12)
        self.assertGreater(audit["exclusions"]["clean:final_failure"], 0)
        expected = {
            "perfect_success",
            "failure_raw",
            "repair_25",
            "repair_50",
            "repair_75",
            "repair_100",
        }
        self.assertEqual(set(audit["arms"]), expected)
        task_multisets = []
        for arm in expected:
            rows = self.read_jsonl(self.output / "arms" / arm / "train.jsonl")
            self.assertEqual(len(rows), 32)
            task_multisets.append(Counter(row["metadata"]["task_id"] for row in rows))
            for row in rows:
                self.assertEqual(
                    set(row),
                    {"id", "messages", "label_mask", "metadata", "token_contract"},
                )
                self.assertEqual(row["metadata"]["source_split"], "inner_train")
                self.assertEqual(row["metadata"]["fit_split"], "train_schedule")
                self.assertEqual(len(row["messages"]), len(row["label_mask"]))
                self.assertRegex(
                    row["metadata"]["source_trajectory_sha256"], r"^[0-9a-f]{64}$"
                )
                self.assertFalse(row["metadata"]["official_test_used"])
                self.assertIn(
                    row["metadata"]["fault_family"],
                    {
                        "retail_missing_user",
                        "retail_missing_order",
                        "airline_missing_reservation",
                        "airline_missing_flight",
                    },
                )
                self.assertIn(
                    row["metadata"]["fault_relevance"],
                    {
                        "reference_path_or_operation_aligned",
                        "domain_plausible_fallback",
                    },
                )
                self.assertEqual(
                    set(row["token_contract"]),
                    {"sequence_tokens", "supervised_tokens", "label_spans"},
                )
                serialized = json.dumps(row["messages"])
                self.assertNotIn("v5-stage1-special", serialized)
                self.assertIn("call_0001", serialized)
        self.assertTrue(all(value == task_multisets[0] for value in task_multisets))

    def test_empty_tau2_rollout_is_excluded_without_becoming_a_label(self):
        target = self.raw / "retail_clean.shard-001-of-002.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        empty = next(
            row
            for row in payload["simulations"]
            if row["task_id"] == "1" and row["trial"] == 3
        )
        empty["messages"] = []
        target.write_text(json.dumps(payload), encoding="utf-8")

        audit = self.run_prepare()

        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(
            audit["exclusions"]["clean:invalid_simulation_no_messages"],
            1,
        )
        for arm in audit["arms"]:
            rows = self.read_jsonl(self.output / "arms" / arm / "train.jsonl")
            self.assertFalse(
                any(
                    row["metadata"]["task_id"] == "1"
                    and row["metadata"]["trial"] == 3
                    for row in rows
                )
            )

    def test_multi_tool_call_rollout_is_excluded_without_truncation(self):
        target = self.raw / "retail_clean.shard-001-of-002.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        malformed = next(
            row
            for row in payload["simulations"]
            if row["task_id"] == "1" and row["trial"] == 3
        )
        assistant = next(
            message
            for message in malformed["messages"]
            if message.get("role") == "assistant" and message.get("tool_calls")
        )
        assistant["tool_calls"].append(deepcopy(assistant["tool_calls"][0]))
        target.write_text(json.dumps(payload), encoding="utf-8")

        audit = self.run_prepare()

        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(
            audit["exclusions"]["clean:invalid_simulation_non_single_tool_call"],
            1,
        )
        for arm in audit["arms"]:
            rows = self.read_jsonl(self.output / "arms" / arm / "train.jsonl")
            self.assertFalse(
                any(
                    row["metadata"]["task_id"] == "1"
                    and row["metadata"]["trial"] == 3
                    for row in rows
                )
            )

    def test_mixed_text_and_tool_call_rollout_is_excluded(self):
        target = self.raw / "retail_clean.shard-001-of-002.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        malformed = next(
            row
            for row in payload["simulations"]
            if row["task_id"] == "1" and row["trial"] == 3
        )
        assistant = next(
            message
            for message in malformed["messages"]
            if message.get("role") == "assistant" and message.get("tool_calls")
        )
        assistant["content"] = "I will call the tool now."
        target.write_text(json.dumps(payload), encoding="utf-8")

        audit = self.run_prepare()

        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(
            audit["exclusions"]["clean:invalid_simulation_mixed_text_and_tool_call"],
            1,
        )

    def test_raw_and_masked_labels_have_strict_outcomes(self):
        self.run_prepare()
        raw_rows = self.read_jsonl(
            self.output / "arms" / "failure_raw" / "train.jsonl"
        )
        repair_rows = self.read_jsonl(
            self.output / "arms" / "repair_100" / "train.jsonl"
        )
        for row in raw_rows:
            failed = row["metadata"]["failed_assistant_message_indices"]
            self.assertEqual(len(failed), 1)
            self.assertTrue(row["label_mask"][failed[0]])
            result = row["messages"][failed[0] + 1]
            self.assertEqual(result["role"], "tool")
            self.assertTrue(result["error"])
        for row in repair_rows:
            failed = row["metadata"]["failed_assistant_message_indices"]
            self.assertEqual(len(failed), 1)
            self.assertFalse(row["label_mask"][failed[0]])
            self.assertTrue(
                any(
                    selected and index > failed[0]
                    for index, selected in enumerate(row["label_mask"])
                )
            )
            for index, selected in enumerate(row["label_mask"]):
                if selected and row["messages"][index].get("tool_calls"):
                    self.assertEqual(row["messages"][index + 1]["role"], "tool")
                    self.assertFalse(row["messages"][index + 1]["error"])

    def test_validation_loss_is_inner_train_disjoint_and_eval_has_21(self):
        self.run_prepare()
        train_ids = {
            row["metadata"]["source_example_id"]
            for arm in MODULE.CORE_ARMS
            for row in self.read_jsonl(
                self.output / "arms" / arm / "train.jsonl"
            )
        }
        validation_loss = self.read_jsonl(self.output / "validation_loss.jsonl")
        validation_ids = {
            row["metadata"]["source_example_id"] for row in validation_loss
        }
        self.assertFalse(train_ids & validation_ids)
        self.assertTrue(
            all(
                row["metadata"]["source_split"] == "inner_train"
                and row["metadata"]["fit_split"] == "validation_loss"
                for row in validation_loss
            )
        )
        manifest = json.loads(
            (self.output / "validation_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["protocol"], "v5_stage1_sft_causal_validation"
        )
        self.assertEqual(manifest["paired_task_count"], 21)
        self.assertEqual(manifest["domain_counts"], {"retail": 15, "airline": 6})
        self.assertFalse(manifest["official_test_used"])
        self.assertTrue(manifest["official_test_sealed"])
        self.assertFalse(manifest["scientific_claim_allowed"])
        self.assertFalse(
            {"900", "901"} & {row["task_id"] for row in manifest["rows"]}
        )

    def test_target_zero_is_empty(self):
        row = {"token_contract": {"supervised_tokens": 3}}
        self.assertEqual(MODULE.take_to_budget([row], 0), [])

    def test_builder_rows_pass_trainer_validation_and_exact_encoding(self):
        self.run_prepare()
        for arm in MODULE.CORE_ARMS:
            rows = self.read_jsonl(self.output / "arms" / arm / "train.jsonl")
            encoded = TRAIN_MODULE.encode_rows(
                FakeTokenizer(),
                rows,
                arm=arm,
                split="train",
                max_seq_len=100000,
            )
            self.assertEqual(len(encoded), 32)
        validation_rows = self.read_jsonl(self.output / "validation_loss.jsonl")
        encoded_validation = TRAIN_MODULE.encode_rows(
            FakeTokenizer(),
            validation_rows,
            arm=None,
            split="validation",
            max_seq_len=100000,
        )
        self.assertEqual(len(encoded_validation), len(validation_rows))

    def test_four_generation_contracts_are_complete_and_frozen(self):
        raw_dir, generation_manifest, _, audit_identity, _, results = (
            self.strict_generation_fixture()
        )
        audit = self.validate_strict_generation(
            raw_dir, generation_manifest, audit_identity
        )
        self.assertEqual(audit["task_union"], 4)
        self.assertEqual(audit["shards"], 4)
        self.assertEqual(
            set(audit["result_files"]),
            {str(path) for path in results.values()},
        )

    def test_generation_contract_rejects_incomplete_status(self):
        raw_dir, generation_manifest, _, audit_identity, contracts, _ = (
            self.strict_generation_fixture("incomplete")
        )
        contract = json.loads(contracts[0].read_text(encoding="utf-8"))
        contract["status"] = "INCOMPLETE"
        contracts[0].write_text(json.dumps(contract), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not COMPLETE"):
            self.validate_strict_generation(
                raw_dir, generation_manifest, audit_identity
            )

    def test_data_builder_and_contracts_require_complete_dynamic_audit(self):
        payload = json.loads(
            self.generation_audit_path.read_text(encoding="utf-8")
        )
        payload["status"] = "INCOMPLETE"
        self.generation_audit_path.write_text(
            json.dumps(payload), encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "not COMPLETE"):
            self.run_prepare()

        (
            raw_dir,
            generation_manifest,
            _,
            audit_identity,
            contracts,
            _,
        ) = self.strict_generation_fixture("audit-drift")
        contract = json.loads(contracts[0].read_text(encoding="utf-8"))
        contract["dynamic_audit_identity"]["sha256"] = "0" * 64
        contracts[0].write_text(json.dumps(contract), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "dynamic audit identity drift"):
            self.validate_strict_generation(
                raw_dir, generation_manifest, audit_identity
            )

    def test_generation_contract_rejects_tampered_raw_bytes_and_hash(self):
        raw_dir, generation_manifest, _, audit_identity, _, results = (
            self.strict_generation_fixture("tampered-bytes")
        )
        result_path = next(iter(results.values()))
        result_path.write_bytes(result_path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(RuntimeError, "result SHA drift"):
            self.validate_strict_generation(
                raw_dir, generation_manifest, audit_identity
            )

        raw_dir, generation_manifest, _, audit_identity, contracts, _ = (
            self.strict_generation_fixture("tampered-hash")
        )
        contract = json.loads(contracts[0].read_text(encoding="utf-8"))
        filename = next(iter(contract["result_sha256"]))
        contract["result_sha256"][filename] = "0" * 64
        contracts[0].write_text(json.dumps(contract), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "result SHA drift"):
            self.validate_strict_generation(
                raw_dir, generation_manifest, audit_identity
            )

    def test_generation_contract_rejects_undeclared_raw_result(self):
        raw_dir, generation_manifest, _, audit_identity, _, _ = self.strict_generation_fixture(
            "undeclared"
        )
        (raw_dir / "retail_clean.shard-999-of-004.json").write_text(
            json.dumps({"simulations": []}), encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "undeclared=.*999"):
            self.validate_strict_generation(
                raw_dir, generation_manifest, audit_identity
            )

        raw_dir, generation_manifest, _, audit_identity, _, _ = self.strict_generation_fixture(
            "undeclared-single"
        )
        (raw_dir / "retail_clean.json").write_text(
            json.dumps({"simulations": []}), encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "undeclared=.*retail_clean.json"):
            self.validate_strict_generation(
                raw_dir, generation_manifest, audit_identity
            )

    def test_validation_manifest_passes_runner_and_summarizer_contracts(self):
        frozen_split_path = ROOT / "data" / "processed" / "v5_stage0" / "split_manifest.json"
        tau2_root = ROOT / "data" / "raw" / "tau2-bench"
        if not frozen_split_path.exists() or not tau2_root.exists():
            self.skipTest("frozen Stage-0 split or pinned tau2 checkout is unavailable")
        protocol_dir = self.root / "formal_protocol"
        MANIFEST_MODULE.prepare(
            tau2_root=tau2_root,
            split_manifest_path=frozen_split_path,
            output_dir=protocol_dir,
            seed=MODULE.SEED,
        )
        manifest_path = protocol_dir / "validation_manifest.json"
        runner_split = EVAL_MODULE.load_split_manifest(frozen_split_path)
        loaded = EVAL_MODULE.load_manifest(
            manifest_path,
            split_manifest=runner_split,
            split_manifest_sha256=MODULE.sha256_file(frozen_split_path),
        )
        self.assertEqual(loaded["paired_task_count"], 21)
        expected, validation_ids, sealed_ids = (
            SUMMARY_MODULE.validate_split_and_evaluation_manifests(
                frozen_split_path, manifest_path
            )
        )
        self.assertEqual(len(expected), 21)
        self.assertEqual(len(validation_ids), 21)
        self.assertEqual(len(sealed_ids), 60)

    def test_mismatched_tool_result_fails_closed(self):
        broken = json.loads(
            (self.raw / "retail_clean.shard-000-of-002.json").read_text(
                encoding="utf-8"
            )
        )
        broken["simulations"][0]["messages"][3]["id"] = "wrong"
        (self.raw / "retail_clean.shard-000-of-002.json").write_text(
            json.dumps(broken), encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "id mismatch"):
            self.run_prepare()
        self.assertFalse((self.output / "audit.json").exists())
        self.assertFalse((self.output / "hashes.json").exists())

    def test_output_is_byte_deterministic(self):
        self.run_prepare()
        second = self.root / "second"
        first_hashes = json.loads(
            (self.output / "hashes.json").read_text(encoding="utf-8")
        )
        self.run_prepare(second)
        second_hashes = json.loads(
            (second / "hashes.json").read_text(encoding="utf-8")
        )
        self.assertEqual(first_hashes, second_hashes)


if __name__ == "__main__":
    unittest.main()
