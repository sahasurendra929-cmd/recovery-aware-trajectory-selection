from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_v5_sft_causal.py"
SPEC = importlib.util.spec_from_file_location("train_v5_sft_causal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeChatTokenizer:
    """Prefix-stable tokenizer sufficient to test message-to-token masks."""

    ROLE = {"system": 10, "user": 20, "assistant": 30, "tool": 40}

    def _body(self, message):
        value = json.dumps(message, ensure_ascii=False, sort_keys=True)
        return [100 + (ord(character) % 67) for character in value]

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt):
        assert tokenize is True
        ids = [1, 50]
        ids.extend(self._body({"tools": tools}))
        for message in messages:
            ids.append(self.ROLE[message["role"]])
            ids.extend(self._body(message))
            ids.append(2)
        if add_generation_prompt:
            ids.append(self.ROLE["assistant"])
        return ids


def assistant_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "name": name,
                "arguments": arguments,
            }
        ],
    }


def row_with_repair(*, raw: bool) -> dict:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "parameters": {"type": "object"},
            },
        }
    ]
    row = {
        "id": "schedule:0001",
        "messages": [
            {"role": "system", "content": "tools"},
            {"role": "user", "content": "refund my order"},
            assistant_call("bad", "get_order", {"order_id": "missing"}),
            {
                "role": "tool",
                "id": "bad",
                "error": True,
                "content": "not found",
            },
            assistant_call("good", "find_order", {"email": "u@example.com"}),
            {
                "role": "tool",
                "id": "good",
                "error": False,
                "content": "found",
            },
            assistant_call("done", "refund_order", {"order_id": "123"}),
            {
                "role": "tool",
                "id": "done",
                "error": False,
                "content": "refunded",
            },
        ],
        "label_mask": [False, False, raw, False, True, False, True, False],
        "metadata": {
            "arm": "failure_raw" if raw else "repair_100",
            "source": "failure_rich",
            "source_split": "inner_train",
            "fit_split": "train_schedule",
            "source_example_id": "source:0001",
            "full_trajectory_reward": 1.0,
            "source_trajectory_sha256": "a" * 64,
            "official_test_used": False,
            "failed_assistant_message_indices": [2],
            "injected_failed_assistant_message_index": 2,
            "tool_schemas": tools,
            "tool_schemas_sha256": __import__("hashlib").sha256(
                MODULE.canonical(tools).encode("utf-8")
            ).hexdigest(),
        },
        "token_contract": {},
    }
    normalized = [
        MODULE.normalize_message(
            message,
            where=f"{row['id']}.messages[{index}]",
        )
        for index, message in enumerate(row["messages"])
    ]
    tokenizer = FakeChatTokenizer()
    full = tokenizer.apply_chat_template(
        normalized,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
    )
    spans = []
    for index, selected in enumerate(row["label_mask"]):
        if not selected:
            continue
        before = tokenizer.apply_chat_template(
            normalized[:index],
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        through = tokenizer.apply_chat_template(
            normalized[: index + 1],
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        spans.append(
            {
                "message_index": index,
                "token_start": len(before),
                "token_end": len(through),
            }
        )
    row["token_contract"] = {
        "sequence_tokens": len(full),
        "supervised_tokens": sum(
            span["token_end"] - span["token_start"] for span in spans
        ),
        "label_spans": spans,
    }
    return row


def write_data_provenance_bundle(root: Path, arm: str = "repair_100"):
    train = root / "arms" / arm / "train.jsonl"
    validation = root / "validation_loss.jsonl"
    audit_path = root / "audit.json"
    hashes_path = root / "hashes.json"
    train.parent.mkdir(parents=True)
    train.write_text('{"id":"train"}\n', encoding="utf-8")
    validation.write_text('{"id":"validation"}\n', encoding="utf-8")
    train_sha = MODULE.sha256_file(train)
    validation_sha = MODULE.sha256_file(validation)
    split_sha = "1" * 64
    dynamic = {}
    for index, (name, expected) in enumerate(
        MODULE.EXPECTED_DYNAMIC_AUDITS.items(), start=2
    ):
        dynamic[name] = {
            "protocol": MODULE.DYNAMIC_AUDIT_PROTOCOL,
            "sha256": f"{index:x}" * 64,
            "manifest_sha256": f"{index + 2:x}" * 64,
            "split_manifest_sha256": split_sha,
            "source_split": expected["source_split"],
            "verified_injections": expected["verified_injections"],
            "official_test_used": False,
            "official_test_sealed": True,
        }
    audit = {
        "status": "PASS",
        "protocol": MODULE.DATA_AUDIT_PROTOCOL,
        "split_manifest_sha256": split_sha,
        "official_test_used": False,
        "derived_validation_used_for_supervision": False,
        "train_validation_source_overlap": 0,
        "label_guarantees": {
            "official_test_and_derived_validation_label_leakage": 0,
        },
        "arms": {arm: {"sha256": train_sha}},
        "validation_loss": {
            "sha256": validation_sha,
            "source_split": "inner_train",
        },
        "dynamic_audits": dynamic,
    }
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")
    hashes = {
        "audit.json": MODULE.sha256_file(audit_path),
        f"arms/{arm}/train.jsonl": train_sha,
        "validation_loss.jsonl": validation_sha,
    }
    hashes_path.write_text(json.dumps(hashes) + "\n", encoding="utf-8")
    return {
        "arm": arm,
        "train": train,
        "validation": validation,
        "audit_path": audit_path,
        "hashes_path": hashes_path,
        "audit": audit,
        "hashes": hashes,
        "train_sha": train_sha,
        "validation_sha": validation_sha,
    }


def rewrite_bundle_audit(bundle):
    bundle["audit_path"].write_text(
        json.dumps(bundle["audit"]) + "\n",
        encoding="utf-8",
    )
    bundle["hashes"]["audit.json"] = MODULE.sha256_file(
        bundle["audit_path"]
    )
    bundle["hashes_path"].write_text(
        json.dumps(bundle["hashes"]) + "\n",
        encoding="utf-8",
    )


def write_v5_3_data_provenance_bundle(
    root: Path,
    arm: str = "repair_100",
):
    bundle = write_data_provenance_bundle(root, arm=arm)
    split_sha = bundle["audit"]["split_manifest_sha256"]
    generation_sha = "a" * 64
    validation_rows = [
        {
            "domain": "retail" if index < 15 else "airline",
            "task_id": str(2_000 + index),
            "source_split": "derived_validation",
        }
        for index in range(21)
    ]
    validation_manifest = {
        "protocol": MODULE.V5_3_VALIDATION_MANIFEST_PROTOCOL,
        "paired_task_count": 21,
        "rows": validation_rows,
        "official_test_used": False,
        "official_test_sealed": True,
    }
    validation_manifest_path = root / "validation_manifest.json"
    validation_manifest_path.write_text(
        json.dumps(validation_manifest) + "\n",
        encoding="utf-8",
    )
    validation_manifest_sha = MODULE.sha256_file(validation_manifest_path)
    dynamic = {
        "generation": {
            "protocol": MODULE.DYNAMIC_AUDIT_PROTOCOL,
            "sha256": "2" * 64,
            "manifest_sha256": generation_sha,
            "split_manifest_sha256": split_sha,
            "source_split": "derived_inner_train",
            "verified_injections": 78,
            "official_test_used": False,
            "official_test_sealed": True,
        },
        "validation": {
            "protocol": MODULE.DYNAMIC_AUDIT_PROTOCOL,
            "sha256": "3" * 64,
            "manifest_sha256": validation_manifest_sha,
            "split_manifest_sha256": split_sha,
            "source_split": "derived_validation",
            "verified_injections": 21,
            "official_test_used": False,
            "official_test_sealed": True,
        },
    }
    included_tasks = [f"retail:{1_000 + index}" for index in range(78)]
    arm_train = included_tasks[:70]
    loss_validation = included_tasks[70:]
    contracts_root = root / "generation_contracts"
    contracts_root.mkdir()
    contract_files = {}
    for shard in range(MODULE.V5_3_GENERATION_SHARDS):
        contract_path = contracts_root / (
            f"run_contract.shard-{shard:03d}-of-"
            f"{MODULE.V5_3_GENERATION_SHARDS:03d}.json"
        )
        contract = {
            "protocol": "v5_stage1_inner_train_generation_run",
            "status": "COMPLETE",
            "generation_manifest_protocol": (
                MODULE.V5_3_GENERATION_MANIFEST_PROTOCOL
            ),
            "gt_compatibility_filter": MODULE.V5_3_GT_FILTER,
            "manifest_sha256": generation_sha,
            "dynamic_audit_identity": dynamic["generation"],
            "source_split": "derived_inner_train",
            "official_test_used": False,
            "shard_index": shard,
            "num_shards": MODULE.V5_3_GENERATION_SHARDS,
            "num_trials": MODULE.V5_3_GENERATION_TRIALS,
            "task_ids": included_tasks[shard::MODULE.V5_3_GENERATION_SHARDS],
        }
        contract_path.write_text(
            json.dumps(contract) + "\n",
            encoding="utf-8",
        )
        contract_files[str(contract_path)] = MODULE.sha256_file(contract_path)

    bundle["audit"].update(
        {
            "design_version": MODULE.V5_3_DESIGN_VERSION,
            "design_protocol": MODULE.V5_3_DESIGN_PROTOCOL,
            "ground_truth_incompatible_task_ids": list(
                MODULE.V5_3_GT_INCOMPATIBLE_TASK_IDS
            ),
            "arm_train_task_ids": arm_train,
            "loss_validation_task_ids": loss_validation,
            "generation_manifest_sha256": generation_sha,
            "generation_contracts": {
                "contract_files": contract_files,
                "generation_manifest_sha256": generation_sha,
                "dynamic_audit_identity": dynamic["generation"],
                "task_union": 78,
                "shards": MODULE.V5_3_GENERATION_SHARDS,
            },
            "dynamic_audits": dynamic,
        }
    )
    bundle["hashes"]["validation_manifest.json"] = validation_manifest_sha
    bundle["validation_manifest"] = validation_manifest_path
    bundle["generation_contract_paths"] = [
        Path(path) for path in contract_files
    ]
    rewrite_bundle_audit(bundle)
    return bundle


class MessageMaskedSFTContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeChatTokenizer()

    def test_cpu_import_does_not_import_torch_or_transformers(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top_level_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_level_imports.append(node.module or "")
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("transformers", top_level_imports)
        self.assertNotIn("datasets", top_level_imports)
        self.assertNotIn("peft", top_level_imports)

    def test_repair_mask_keeps_failed_call_as_context_not_label(self):
        validated = MODULE.validate_row(
            row_with_repair(raw=False),
            arm="repair_100",
            split="train",
        )
        encoded = MODULE.encode_row(self.tokenizer, validated, max_seq_len=4096)
        supervised_messages = {
            span["message_index"] for span in encoded["label_spans"]
        }
        self.assertEqual(supervised_messages, {4, 6})
        self.assertNotIn(2, supervised_messages)
        self.assertEqual(encoded["failed_message_count"], 1)
        self.assertEqual(encoded["failed_label_message_count"], 0)

    def test_failure_raw_supervises_the_failed_call(self):
        validated = MODULE.validate_row(
            row_with_repair(raw=True),
            arm="failure_raw",
            split="train",
        )
        encoded = MODULE.encode_row(self.tokenizer, validated)
        supervised_messages = {
            span["message_index"] for span in encoded["label_spans"]
        }
        self.assertEqual(supervised_messages, {2, 4, 6})
        self.assertEqual(encoded["failed_label_message_count"], 1)
        self.assertGreater(encoded["failed_label_tokens"], 0)

    def test_label_span_excludes_assistant_role_header(self):
        validated = MODULE.validate_row(
            row_with_repair(raw=False),
            arm="repair_100",
            split="train",
        )
        encoded = MODULE.encode_row(self.tokenizer, validated)
        for span in encoded["label_spans"]:
            self.assertNotEqual(
                encoded["input_ids"][span["token_start"]],
                FakeChatTokenizer.ROLE["assistant"],
            )
            self.assertTrue(
                all(
                    encoded["labels"][index] == encoded["input_ids"][index]
                    for index in range(span["token_start"], span["token_end"])
                )
            )
        self.assertTrue(
            all(
                label == MODULE.IGNORE_INDEX
                for index, label in enumerate(encoded["labels"])
                if not any(
                    span["token_start"] <= index < span["token_end"]
                    for span in encoded["label_spans"]
                )
            )
        )

    def test_non_assistant_label_fails_closed(self):
        row = row_with_repair(raw=False)
        row["label_mask"][1] = True
        with self.assertRaisesRegex(RuntimeError, "non-assistant"):
            MODULE.validate_row(row, arm="repair_100", split="train")

    def test_messages_after_final_label_fail_as_future_leakage(self):
        row = row_with_repair(raw=False)
        row["messages"].append({"role": "user", "content": "future"})
        row["label_mask"].append(False)
        with self.assertRaisesRegex(RuntimeError, "no later messages"):
            MODULE.validate_row(row, arm="repair_100", split="train")

    def test_every_selected_tool_call_requires_linked_evidence_result(self):
        row = row_with_repair(raw=False)
        row["messages"].pop()
        row["label_mask"].pop()
        with self.assertRaisesRegex(RuntimeError, "adjacent evidence tool result"):
            MODULE.validate_row(row, arm="repair_100", split="train")
        row = row_with_repair(raw=False)
        row["messages"][7]["id"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "id mismatch"):
            MODULE.validate_row(row, arm="repair_100", split="train")

    def test_only_controlled_failure_raw_label_may_have_error_result(self):
        row = row_with_repair(raw=False)
        row["messages"][5]["error"] = True
        row["metadata"]["failed_assistant_message_indices"] = [2, 4]
        with self.assertRaisesRegex(RuntimeError, "exactly its one controlled"):
            MODULE.validate_row(row, arm="repair_100", split="train")

    def test_provenance_metadata_is_mandatory(self):
        for field, value, expected in (
            ("full_trajectory_reward", 0.0, "must equal 1.0"),
            ("source_trajectory_sha256", "bad", "lowercase SHA-256"),
            ("official_test_used", True, "must be false"),
        ):
            row = row_with_repair(raw=False)
            row["metadata"][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, expected):
                    MODULE.validate_row(row, arm="repair_100", split="train")

    def test_mask_length_and_tool_link_fail_closed(self):
        row = row_with_repair(raw=False)
        row["label_mask"].pop()
        with self.assertRaisesRegex(RuntimeError, "one boolean per message"):
            MODULE.validate_row(row, arm="repair_100", split="train")
        row = row_with_repair(raw=False)
        row["messages"][3]["id"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "id mismatch"):
            MODULE.validate_row(row, arm="repair_100", split="train")

    def test_repair_requires_a_supervised_action_after_error(self):
        row = row_with_repair(raw=False)
        row["label_mask"] = [False] * len(row["messages"])
        row["label_mask"][0] = True
        with self.assertRaisesRegex(RuntimeError, "non-assistant"):
            MODULE.validate_row(row, arm="repair_100", split="train")
        row["label_mask"] = [False] * len(row["messages"])
        row["label_mask"][2] = True
        row["label_mask"][6] = True
        with self.assertRaisesRegex(RuntimeError, "mask every failed"):
            MODULE.validate_row(row, arm="repair_100", split="train")

    def test_no_truncation_is_an_error(self):
        validated = MODULE.validate_row(
            row_with_repair(raw=False),
            arm="repair_100",
            split="train",
        )
        with self.assertRaisesRegex(RuntimeError, "forbids truncation"):
            MODULE.encode_row(self.tokenizer, validated, max_seq_len=8)

    def test_token_contract_detects_tokenizer_drift(self):
        row = row_with_repair(raw=False)
        row["token_contract"]["sequence_tokens"] = 1
        validated = MODULE.validate_row(row, arm="repair_100", split="train")
        with self.assertRaisesRegex(RuntimeError, "sequence token contract drift"):
            MODULE.encode_row(self.tokenizer, validated)

    def test_arm_ratio_is_based_on_supervised_tokens(self):
        clean = {
            "id": "clean",
            "sequence_tokens": 20,
            "supervised_tokens": 10,
            "is_recovery": False,
            "failed_message_count": 0,
            "failed_label_message_count": 0,
            "failed_label_tokens": 0,
        }
        recovery = {
            "id": "recovery",
            "sequence_tokens": 30,
            "supervised_tokens": 10,
            "is_recovery": True,
            "failed_message_count": 1,
            "failed_label_message_count": 0,
            "failed_label_tokens": 0,
        }
        audit = MODULE.arm_audit([clean, recovery], "repair_50")
        self.assertEqual(audit["realized_recovery_supervised_token_ratio"], 0.5)
        with self.assertRaisesRegex(RuntimeError, "expected 0.25"):
            MODULE.arm_audit([clean, recovery], "repair_25")

    def test_screen_arm_ratio_is_exact_row_weight_not_token_mass(self):
        clean = {
            "id": "clean",
            "sequence_tokens": 20,
            "supervised_tokens": 5,
            "is_recovery": False,
            "failed_message_count": 0,
            "failed_label_message_count": 0,
            "failed_label_tokens": 0,
        }
        recovery = {
            "id": "recovery",
            "sequence_tokens": 30,
            "supervised_tokens": 15,
            "is_recovery": True,
            "failed_message_count": 1,
            "failed_label_message_count": 0,
            "failed_label_tokens": 0,
        }
        audit = MODULE.arm_audit(
            [clean, recovery],
            "repair_50",
            recovery_mixture_basis="row_mean_microbatch_equal_weight",
        )
        self.assertEqual(audit["realized_recovery_row_ratio"], 0.5)
        self.assertEqual(
            audit["realized_recovery_supervised_token_ratio"], 0.75
        )
        self.assertIsNone(
            audit["expected_recovery_supervised_token_ratio"]
        )
        self.assertEqual(audit["expected_recovery_row_ratio"], 0.5)
        with self.assertRaisesRegex(RuntimeError, "recovery row ratio"):
            MODULE.arm_audit(
                [clean, recovery, recovery],
                "repair_50",
                recovery_mixture_basis=(
                    "row_mean_microbatch_equal_weight"
                ),
            )

    def test_tail_logit_contract_aligns_causal_shift(self):
        logits_to_keep, target_start = MODULE.tail_logit_contract(100, 60)
        self.assertEqual(logits_to_keep, 41)
        self.assertEqual(target_start, 60)
        self.assertEqual(logits_to_keep - 1, 100 - target_start)
        with self.assertRaises(ValueError):
            MODULE.tail_logit_contract(100, 0)

    def test_frozen_training_constants_match_stage1_config(self):
        self.assertEqual(MODULE.MODEL, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(MODULE.MAX_SEQUENCE_TOKENS, 8192)
        self.assertEqual(MODULE.FORMAL_STEPS, 64)
        self.assertEqual(MODULE.FORMAL_GRAD_ACCUM, 8)
        self.assertEqual(MODULE.FORMAL_SCHEDULE_ROWS, 512)
        self.assertEqual(
            set(MODULE.ARM_TARGET_RECOVERY_RATIOS),
            {
                "perfect_success",
                "failure_raw",
                "repair_25",
                "repair_50",
                "repair_75",
                "repair_100",
            },
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("lora_dropout=0.0", source)
        self.assertIn('lr_scheduler_type="cosine"', source)

    def test_jsonl_reader_rejects_duplicate_ids_and_blank_lines(self):
        row = row_with_repair(raw=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(
                json.dumps(row) + "\n\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "blank JSONL line"):
                MODULE.read_jsonl(path)
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate id"):
                MODULE.read_jsonl(path)

    def test_fit_validation_sources_must_be_inner_train_but_disjoint(self):
        train = row_with_repair(raw=False)
        validation = row_with_repair(raw=False)
        validation["id"] = "loss-validation:0001"
        validation["metadata"]["fit_split"] = "validation_loss"
        with self.assertRaisesRegex(RuntimeError, "source leakage"):
            MODULE.validate_fit_partition([train], [validation])
        validation["metadata"]["source_example_id"] = "source:heldout"
        self.assertEqual(
            MODULE.validate_fit_partition([train], [validation])["overlap"],
            0,
        )
        validation["metadata"]["source_split"] = "derived_validation"
        with self.assertRaisesRegex(RuntimeError, "inner_train"):
            MODULE.validate_row(validation, arm=None, split="validation")

    def test_training_data_provenance_binds_audit_hashes_and_dynamic_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_data_provenance_bundle(Path(directory))
            result = MODULE.validate_training_data_provenance(
                arm=bundle["arm"],
                train_file=bundle["train"],
                validation_file=bundle["validation"],
                data_audit_path=bundle["audit_path"],
                data_hashes_path=bundle["hashes_path"],
                expected_train_sha256=bundle["train_sha"],
                expected_validation_sha256=bundle["validation_sha"],
            )
            self.assertEqual(
                result["data_audit_sha256"],
                MODULE.sha256_file(bundle["audit_path"]),
            )
            self.assertEqual(
                result["data_hashes_sha256"],
                MODULE.sha256_file(bundle["hashes_path"]),
            )
            self.assertEqual(
                set(result["dynamic_audits"]),
                {"generation", "validation"},
            )
            self.assertTrue(result["official_test_sealed"])
            self.assertIsNone(result["design_provenance"])

    def test_legacy_v5_2_audit_keeps_legacy_dynamic_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_data_provenance_bundle(Path(directory))
            bundle["audit"]["design_version"] = "5.2"
            rewrite_bundle_audit(bundle)
            result = MODULE.validate_training_data_provenance(
                arm=bundle["arm"],
                train_file=bundle["train"],
                validation_file=bundle["validation"],
                data_audit_path=bundle["audit_path"],
                data_hashes_path=bundle["hashes_path"],
                expected_train_sha256=bundle["train_sha"],
                expected_validation_sha256=bundle["validation_sha"],
            )
        self.assertEqual(result["design_version"], "5.2")
        self.assertIsNone(result["design_provenance"])

    def test_v5_3_training_provenance_binds_78_21_filter_and_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_v5_3_data_provenance_bundle(Path(directory))
            result = MODULE.validate_training_data_provenance(
                arm=bundle["arm"],
                train_file=bundle["train"],
                validation_file=bundle["validation"],
                data_audit_path=bundle["audit_path"],
                data_hashes_path=bundle["hashes_path"],
                expected_train_sha256=bundle["train_sha"],
                expected_validation_sha256=bundle["validation_sha"],
            )

        design = result["design_provenance"]
        self.assertEqual(design["design_version"], "5.3")
        self.assertEqual(design["effective_generation_tasks"], 78)
        self.assertEqual(design["validation_tasks"], 21)
        self.assertEqual(
            design["generation_manifest_protocol"],
            MODULE.V5_3_GENERATION_MANIFEST_PROTOCOL,
        )
        self.assertEqual(
            design["ground_truth_incompatible_task_ids"],
            list(MODULE.V5_3_GT_INCOMPATIBLE_TASK_IDS),
        )

    def test_v5_3_training_rejects_dynamic_coverage_and_filter_drift(self):
        cases = (
            (
                lambda bundle: bundle["audit"]["dynamic_audits"][
                    "generation"
                ].__setitem__("verified_injections", 77),
                "generation dynamic audit coverage drift",
            ),
            (
                lambda bundle: bundle["audit"]["dynamic_audits"][
                    "generation"
                ].__setitem__("source_split", "inner_train"),
                "generation dynamic audit source split drift",
            ),
            (
                lambda bundle: bundle["audit"]["dynamic_audits"][
                    "validation"
                ].__setitem__("verified_injections", 20),
                "validation dynamic audit coverage drift",
            ),
            (
                lambda bundle: bundle["audit"].__setitem__(
                    "ground_truth_incompatible_task_ids",
                    list(MODULE.V5_3_GT_INCOMPATIBLE_TASK_IDS[:-1]),
                ),
                "GT-incompatible task set drift",
            ),
        )
        for mutate, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                bundle = write_v5_3_data_provenance_bundle(Path(directory))
                mutate(bundle)
                rewrite_bundle_audit(bundle)
                with self.assertRaisesRegex(RuntimeError, expected):
                    MODULE.validate_training_data_provenance(
                        arm=bundle["arm"],
                        train_file=bundle["train"],
                        validation_file=bundle["validation"],
                        data_audit_path=bundle["audit_path"],
                        data_hashes_path=bundle["hashes_path"],
                        expected_train_sha256=bundle["train_sha"],
                        expected_validation_sha256=bundle["validation_sha"],
                    )

    def test_v5_3_training_rejects_generation_manifest_protocol_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_v5_3_data_provenance_bundle(Path(directory))
            contract_path = bundle["generation_contract_paths"][0]
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["generation_manifest_protocol"] = (
                "v5_stage1_multifault_data_construction"
            )
            contract_path.write_text(
                json.dumps(contract) + "\n",
                encoding="utf-8",
            )
            bundle["audit"]["generation_contracts"]["contract_files"][
                str(contract_path)
            ] = MODULE.sha256_file(contract_path)
            rewrite_bundle_audit(bundle)

            with self.assertRaisesRegex(
                RuntimeError,
                "manifest protocol/contract drift",
            ):
                MODULE.validate_training_data_provenance(
                    arm=bundle["arm"],
                    train_file=bundle["train"],
                    validation_file=bundle["validation"],
                    data_audit_path=bundle["audit_path"],
                    data_hashes_path=bundle["hashes_path"],
                    expected_train_sha256=bundle["train_sha"],
                    expected_validation_sha256=bundle["validation_sha"],
                )

    def test_v5_3_training_rejects_validation_manifest_protocol_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_v5_3_data_provenance_bundle(Path(directory))
            manifest = json.loads(
                bundle["validation_manifest"].read_text(encoding="utf-8")
            )
            manifest["protocol"] = "wrong"
            bundle["validation_manifest"].write_text(
                json.dumps(manifest) + "\n",
                encoding="utf-8",
            )
            validation_manifest_sha = MODULE.sha256_file(
                bundle["validation_manifest"]
            )
            bundle["hashes"][
                "validation_manifest.json"
            ] = validation_manifest_sha
            bundle["audit"]["dynamic_audits"]["validation"][
                "manifest_sha256"
            ] = validation_manifest_sha
            rewrite_bundle_audit(bundle)

            with self.assertRaisesRegex(
                RuntimeError,
                "validation manifest protocol/coverage drift",
            ):
                MODULE.validate_training_data_provenance(
                    arm=bundle["arm"],
                    train_file=bundle["train"],
                    validation_file=bundle["validation"],
                    data_audit_path=bundle["audit_path"],
                    data_hashes_path=bundle["hashes_path"],
                    expected_train_sha256=bundle["train_sha"],
                    expected_validation_sha256=bundle["validation_sha"],
                )

    def test_training_data_provenance_rejects_audit_or_hash_drift(self):
        mutations = (
            ("status", "FAIL", "status is not PASS"),
            ("protocol", "wrong", "protocol drift"),
            ("official_test_used", True, "seal the official test"),
            (
                "derived_validation_used_for_supervision",
                True,
                "derived validation",
            ),
        )
        for field, value, expected in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                bundle = write_data_provenance_bundle(Path(directory))
                bundle["audit"][field] = value
                bundle["audit_path"].write_text(
                    json.dumps(bundle["audit"]) + "\n", encoding="utf-8"
                )
                bundle["hashes"]["audit.json"] = MODULE.sha256_file(
                    bundle["audit_path"]
                )
                bundle["hashes_path"].write_text(
                    json.dumps(bundle["hashes"]) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    MODULE.validate_training_data_provenance(
                        arm=bundle["arm"],
                        train_file=bundle["train"],
                        validation_file=bundle["validation"],
                        data_audit_path=bundle["audit_path"],
                        data_hashes_path=bundle["hashes_path"],
                        expected_train_sha256=bundle["train_sha"],
                        expected_validation_sha256=bundle["validation_sha"],
                    )

        with tempfile.TemporaryDirectory() as directory:
            bundle = write_data_provenance_bundle(Path(directory))
            bundle["hashes"]["audit.json"] = "f" * 64
            bundle["hashes_path"].write_text(
                json.dumps(bundle["hashes"]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "audit/hash manifest binding"):
                MODULE.validate_training_data_provenance(
                    arm=bundle["arm"],
                    train_file=bundle["train"],
                    validation_file=bundle["validation"],
                    data_audit_path=bundle["audit_path"],
                    data_hashes_path=bundle["hashes_path"],
                    expected_train_sha256=bundle["train_sha"],
                    expected_validation_sha256=bundle["validation_sha"],
                )

    def test_training_data_provenance_rejects_missing_dynamic_identity_or_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_data_provenance_bundle(Path(directory))
            del bundle["audit"]["dynamic_audits"]["validation"]
            bundle["audit_path"].write_text(
                json.dumps(bundle["audit"]) + "\n", encoding="utf-8"
            )
            bundle["hashes"]["audit.json"] = MODULE.sha256_file(
                bundle["audit_path"]
            )
            bundle["hashes_path"].write_text(
                json.dumps(bundle["hashes"]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "validation dynamic audit identity"):
                MODULE.validate_training_data_provenance(
                    arm=bundle["arm"],
                    train_file=bundle["train"],
                    validation_file=bundle["validation"],
                    data_audit_path=bundle["audit_path"],
                    data_hashes_path=bundle["hashes_path"],
                    expected_train_sha256=bundle["train_sha"],
                    expected_validation_sha256=bundle["validation_sha"],
                )

        with tempfile.TemporaryDirectory() as directory:
            bundle = write_data_provenance_bundle(Path(directory))
            bundle["audit"]["arms"][bundle["arm"]]["sha256"] = "e" * 64
            bundle["audit_path"].write_text(
                json.dumps(bundle["audit"]) + "\n", encoding="utf-8"
            )
            bundle["hashes"]["audit.json"] = MODULE.sha256_file(
                bundle["audit_path"]
            )
            bundle["hashes_path"].write_text(
                json.dumps(bundle["hashes"]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "selected arm SHA"):
                MODULE.validate_training_data_provenance(
                    arm=bundle["arm"],
                    train_file=bundle["train"],
                    validation_file=bundle["validation"],
                    data_audit_path=bundle["audit_path"],
                    data_hashes_path=bundle["hashes_path"],
                    expected_train_sha256=bundle["train_sha"],
                    expected_validation_sha256=bundle["validation_sha"],
                )

    def test_data_audit_and_hash_arguments_are_required_for_smoke_and_formal(self):
        base = [
            "train_v5_sft_causal.py",
            "--train-file",
            "train.jsonl",
            "--validation-file",
            "validation.jsonl",
            "--output-dir",
            "out",
            "--arm",
            "repair_100",
            "--model-revision",
            "a" * 40,
            "--expected-source-commit",
            "b" * 40,
            "--expected-train-sha256",
            "c" * 64,
            "--expected-validation-sha256",
            "d" * 64,
        ]
        for mode in ("smoke", "formal"):
            with self.subTest(mode=mode), mock.patch.object(
                sys, "argv", [*base, "--mode", mode]
            ), self.assertRaises(SystemExit):
                MODULE.parse_args()
            with self.subTest(mode=f"{mode}-complete"), mock.patch.object(
                sys,
                "argv",
                [
                    *base,
                    "--mode",
                    mode,
                    "--data-audit",
                    "audit.json",
                    "--data-hashes",
                    "hashes.json",
                ],
            ):
                parsed = MODULE.parse_args()
                self.assertEqual(parsed.mode, mode)


if __name__ == "__main__":
    unittest.main()
