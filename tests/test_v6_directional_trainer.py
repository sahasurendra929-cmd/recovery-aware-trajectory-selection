from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_v6_directional_sft",
    ROOT / "scripts" / "train_v6_directional_sft.py",
)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trainer)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FakeTokenizer:
    """Prefix-stable fake chat template with distinct assistant spans."""

    chat_template = "fake-v1"
    special_tokens_map = {"eos_token": "<eos>"}

    def get_vocab(self):
        return {"<eos>": 0, "system": 1, "assistant": 2}

    def get_added_vocab(self):
        return {}

    def apply_chat_template(
        self, messages, *, tools, tokenize, add_generation_prompt
    ):
        assert tokenize is True
        assert tools
        ids = [101]
        for index, message in enumerate(messages):
            if message["role"] == "assistant":
                ids.extend([900, 400 + index, 901])
            else:
                ids.extend([200 + index, 300 + index])
        if add_generation_prompt:
            ids.append(900)
        return ids


def _schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        }
    ]


def _call(call_id, value):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "name": "lookup", "arguments": {"id": value}}
        ],
    }


def _contract(tokenizer, messages, mask, tools):
    normalized = [
        trainer.normalize_message(message, where=f"row.messages[{index}]")
        for index, message in enumerate(messages)
    ]
    full = trainer._token_ids(
        tokenizer, normalized, tools=tools, generation=False
    )
    spans = []
    for index, selected in enumerate(mask):
        if not selected:
            continue
        before = trainer._token_ids(
            tokenizer, normalized[:index], tools=tools, generation=True
        )
        through = trainer._token_ids(
            tokenizer,
            normalized[: index + 1],
            tools=tools,
            generation=False,
        )
        spans.append(
            {
                "message_index": index,
                "token_start": len(before),
                "token_end": len(through),
            }
        )
    return {
        "sequence_tokens": len(full),
        "supervised_tokens": sum(
            span["token_end"] - span["token_start"] for span in spans
        ),
        "label_spans": spans,
    }


def recovery_row():
    tools = _schemas()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "find order"},
        _call("call_0001", "bad"),
        {
            "role": "tool",
            "content": "not found",
            "tool_call_id": "call_0001",
            "error": True,
        },
        {"role": "assistant", "content": "I will recover."},
        _call("call_0002", "good"),
        {
            "role": "tool",
            "content": "found",
            "tool_call_id": "call_0002",
            "error": False,
        },
    ]
    mask = [False, False, False, False, True, True, False]
    candidate_pair_id = "retail:1:pair-0"
    branch_id = f"{candidate_pair_id}:e1"
    row_id = f"v6:full_proposed:{candidate_pair_id}:{branch_id}"
    manifest_sha256 = _digest("selector-manifest")
    pool_sha256 = _digest("candidate-pool")
    candidate_pair_sha256 = _digest("candidate-pair")
    branch_sha256 = _digest("branch")
    metadata = {
        "arm": "v6_recovery_selected",
        "selector": "full_proposed",
        "paper_arm": "full_proposed",
        "selection_seed": None,
        "source": "failure_rich",
        "domain": "retail",
        "task_id": "1",
        "task_identity": "retail:1",
        "trial": branch_id,
        "candidate_pair_id": candidate_pair_id,
        "branch_id": branch_id,
        "clean_id": None,
        "source_example_id": row_id,
        "source_pair_id": candidate_pair_id,
        "source_split": "inner_train",
        "fit_split": "train_schedule",
        "source_trajectory_sha256": branch_sha256,
        "selector_manifest_sha256": manifest_sha256,
        "candidate_pool_sha256": pool_sha256,
        "source_hashes": {
            "selector_manifest_sha256": manifest_sha256,
            "candidate_pool_sha256": pool_sha256,
            "candidate_pair_sha256": candidate_pair_sha256,
            "branch_sha256": branch_sha256,
            "prefix_sha256": _digest("prefix"),
            "environment_snapshot_sha256": _digest("environment"),
        },
        "tool_schemas": tools,
        "tool_schemas_sha256": trainer.canonical_sha256(tools),
        "failed_assistant_message_indices": [2],
        "failed_action_label_messages": 0,
        "future_clean_suffix_used": False,
        "fresh_recovery_suffix": True,
        "official_test_used": False,
        "materialization_protocol": "v6_sft_materialization_v1",
        "tokenizer_name": trainer.MODEL_ID,
        "tokenizer_revision": trainer.TOKENIZER_REVISION,
    }
    metadata["token_contract_input_sha256"] = trainer.canonical_sha256(
        {"messages": messages, "label_mask": mask, "tool_schemas": tools}
    )
    tokenizer = FakeTokenizer()
    return {
        "id": row_id,
        "messages": messages,
        "label_mask": mask,
        "metadata": metadata,
        "token_contract": _contract(tokenizer, messages, mask, tools),
    }


def recovery_manifest_and_rows():
    template = recovery_row()
    candidate_pair_id = template["metadata"]["candidate_pair_id"]
    branches = [
        {"branch_id": f"{candidate_pair_id}:e1", "payload": "first"},
        {"branch_id": f"{candidate_pair_id}:e2", "payload": "second"},
    ]
    pair = {
        "candidate_pair_id": candidate_pair_id,
        "task_identity": "retail:1",
        "domain": "retail",
        "prefix_sha256": _digest("manifest-prefix"),
        "environment_snapshot_sha256": _digest("manifest-environment"),
        "branches": branches,
    }
    manifest_sha256 = _digest("frozen-selector-manifest-file")
    pool_sha256 = _digest("candidate-pool")
    manifest = {
        "protocol": trainer.MANIFEST_PROTOCOL,
        "selector": "full_proposed",
        "selection_seed": None,
        "candidate_pool_sha256": pool_sha256,
        "matched_task_ids": ["retail:1"],
        "selected": [
            {
                "candidate_pair_id": candidate_pair_id,
                "task_identity": "retail:1",
                "domain": "retail",
                "branch_ids": [branch["branch_id"] for branch in branches],
                "candidate_pair": pair,
            }
        ],
        "official_test_used": False,
    }
    rows = []
    for branch in branches:
        row = deepcopy(template)
        branch_id = branch["branch_id"]
        row_id = f"v6:full_proposed:{candidate_pair_id}:{branch_id}"
        metadata = row["metadata"]
        metadata["trial"] = branch_id
        metadata["branch_id"] = branch_id
        metadata["source_example_id"] = row_id
        metadata["selector_manifest_sha256"] = manifest_sha256
        metadata["candidate_pool_sha256"] = pool_sha256
        metadata["source_trajectory_sha256"] = trainer.canonical_sha256(branch)
        metadata["source_hashes"] = {
            "selector_manifest_sha256": manifest_sha256,
            "candidate_pool_sha256": pool_sha256,
            "candidate_pair_sha256": trainer.canonical_sha256(pair),
            "branch_sha256": trainer.canonical_sha256(branch),
            "prefix_sha256": pair["prefix_sha256"],
            "environment_snapshot_sha256": pair[
                "environment_snapshot_sha256"
            ],
        }
        row["id"] = row_id
        rows.append(row)
    return manifest, manifest_sha256, rows


class V6DirectionalTrainerTests(unittest.TestCase):
    def test_recovery_labels_tool_call_and_assistant_text_not_failed_call(self):
        row = recovery_row()
        validated = trainer.validate_row(row, arm="full_proposed")
        encoded = trainer.encode_row(FakeTokenizer(), validated, max_seq_len=100)
        self.assertEqual(encoded["selected_assistant_text_messages"], 1)
        self.assertEqual(encoded["selected_assistant_tool_messages"], 1)
        self.assertEqual(encoded["failed_context_messages"], 1)
        self.assertGreater(encoded["supervised_tokens"], 0)

    def test_structured_tool_result_uses_frozen_canonical_normalization(self):
        row = recovery_row()
        row["messages"][6]["content"] = {
            "status": "found",
            "items": [{"id": "order-1"}],
        }
        row["metadata"][
            "token_contract_input_sha256"
        ] = trainer.canonical_sha256(
            {
                "messages": row["messages"],
                "label_mask": row["label_mask"],
                "tool_schemas": row["metadata"]["tool_schemas"],
            }
        )
        row["token_contract"] = _contract(
            FakeTokenizer(),
            row["messages"],
            row["label_mask"],
            row["metadata"]["tool_schemas"],
        )
        normalized = trainer.normalize_message(
            row["messages"][6], where="structured_tool_result"
        )
        self.assertEqual(
            normalized["content"],
            trainer.canonical(row["messages"][6]["content"]),
        )
        validated = trainer.validate_row(row, arm="full_proposed")
        trainer.encode_row(FakeTokenizer(), validated, max_seq_len=100)

    def test_non_assistant_labels_are_rejected(self):
        for index in [0, 1, 3, 6]:
            with self.subTest(index=index):
                row = recovery_row()
                row["label_mask"][index] = True
                row["metadata"][
                    "token_contract_input_sha256"
                ] = trainer.canonical_sha256(
                    {
                        "messages": row["messages"],
                        "label_mask": row["label_mask"],
                        "tool_schemas": row["metadata"]["tool_schemas"],
                    }
                )
                with self.assertRaisesRegex(
                    trainer.V6TrainingError, "non-assistant"
                ):
                    trainer.validate_row(row, arm="full_proposed")

    def test_failed_call_label_is_rejected(self):
        row = recovery_row()
        row["label_mask"][2] = True
        row["metadata"]["token_contract_input_sha256"] = trainer.canonical_sha256(
            {
                "messages": row["messages"],
                "label_mask": row["label_mask"],
                "tool_schemas": row["metadata"]["tool_schemas"],
            }
        )
        with self.assertRaisesRegex(trainer.V6TrainingError, "failed call"):
            trainer.validate_row(row, arm="full_proposed")

    def test_official_test_use_is_rejected_even_when_nested(self):
        row = recovery_row()
        row["metadata"]["source_hashes"] = {"official_test_used": True}
        with self.assertRaisesRegex(trainer.V6TrainingError, "official test"):
            trainer.validate_row(row, arm="full_proposed")

    def test_source_hashes_must_match_top_level_provenance(self):
        row = recovery_row()
        row["metadata"]["source_hashes"][
            "candidate_pool_sha256"
        ] = _digest("different-pool")
        with self.assertRaisesRegex(
            trainer.V6TrainingError, "candidate pool hash drift"
        ):
            trainer.validate_row(row, arm="full_proposed")

    def test_source_trajectory_must_match_branch_hash(self):
        row = recovery_row()
        row["metadata"]["source_trajectory_sha256"] = _digest(
            "different-branch"
        )
        with self.assertRaisesRegex(
            trainer.V6TrainingError, "source trajectory hash drift"
        ):
            trainer.validate_row(row, arm="full_proposed")

    def test_dataset_is_bound_to_exact_manifest_and_complete_pair(self):
        manifest, manifest_sha256, rows = recovery_manifest_and_rows()
        validated = [
            trainer.validate_row(row, arm="full_proposed") for row in rows
        ]
        audit = trainer.validate_dataset_provenance(
            validated,
            manifest,
            arm="full_proposed",
            expected_selector_manifest_sha256=manifest_sha256,
            observed_selector_manifest_sha256=manifest_sha256,
        )
        self.assertEqual(audit["matched_task_ids"], ["retail:1"])
        self.assertEqual(audit["candidate_pairs"], 1)
        self.assertEqual(audit["rows"], audit["expected_rows"])
        self.assertTrue(audit["source_hashes_recomputed_against_manifest"])

    def test_dataset_rejects_split_candidate_pair(self):
        manifest, manifest_sha256, rows = recovery_manifest_and_rows()
        validated = [trainer.validate_row(rows[0], arm="full_proposed")]
        with self.assertRaisesRegex(
            trainer.V6TrainingError, "two-branch candidate pair"
        ):
            trainer.validate_dataset_provenance(
                validated,
                manifest,
                arm="full_proposed",
                expected_selector_manifest_sha256=manifest_sha256,
                observed_selector_manifest_sha256=manifest_sha256,
            )

    def test_dataset_rejects_untrusted_manifest_hash(self):
        manifest, manifest_sha256, rows = recovery_manifest_and_rows()
        validated = [
            trainer.validate_row(row, arm="full_proposed") for row in rows
        ]
        with self.assertRaisesRegex(
            trainer.V6TrainingError, "manifest file SHA-256 drift"
        ):
            trainer.validate_dataset_provenance(
                validated,
                manifest,
                arm="full_proposed",
                expected_selector_manifest_sha256=manifest_sha256,
                observed_selector_manifest_sha256=_digest("other-manifest"),
            )

    def test_dataset_rejects_cross_row_manifest_or_pool_drift(self):
        for field in ("selector_manifest_sha256", "candidate_pool_sha256"):
            with self.subTest(field=field):
                manifest, manifest_sha256, rows = recovery_manifest_and_rows()
                drift = _digest(f"different-{field}")
                rows[1]["metadata"][field] = drift
                rows[1]["metadata"]["source_hashes"][field] = drift
                validated = [
                    trainer.validate_row(row, arm="full_proposed")
                    for row in rows
                ]
                with self.assertRaisesRegex(
                    trainer.V6TrainingError,
                    "expected selector manifest|candidate pool differs",
                ):
                    trainer.validate_dataset_provenance(
                        validated,
                        manifest,
                        arm="full_proposed",
                        expected_selector_manifest_sha256=manifest_sha256,
                        observed_selector_manifest_sha256=manifest_sha256,
                    )

    def test_dataset_recomputes_candidate_pair_source_hash(self):
        manifest, manifest_sha256, rows = recovery_manifest_and_rows()
        validated = [
            trainer.validate_row(row, arm="full_proposed") for row in rows
        ]
        manifest["selected"][0]["candidate_pair"]["new_unbound_field"] = True
        with self.assertRaisesRegex(
            trainer.V6TrainingError, "candidate_pair source hash drift"
        ):
            trainer.validate_dataset_provenance(
                validated,
                manifest,
                arm="full_proposed",
                expected_selector_manifest_sha256=manifest_sha256,
                observed_selector_manifest_sha256=manifest_sha256,
            )

    def test_retokenization_contract_drift_is_rejected(self):
        row = recovery_row()
        row["token_contract"]["supervised_tokens"] += 1
        validated = trainer.validate_row(row, arm="full_proposed")
        with self.assertRaisesRegex(trainer.V6TrainingError, "token_contract"):
            trainer.encode_row(FakeTokenizer(), validated, max_seq_len=100)

    def test_max_sequence_is_fail_closed_without_truncation(self):
        row = recovery_row()
        validated = trainer.validate_row(row, arm="full_proposed")
        with self.assertRaisesRegex(
            trainer.V6TrainingError, "forbids truncation"
        ):
            trainer.encode_row(FakeTokenizer(), validated, max_seq_len=2)

    def test_arm_drift_is_rejected(self):
        row = recovery_row()
        with self.assertRaisesRegex(
            trainer.V6TrainingError, "selector/arm"
        ):
            trainer.validate_row(row, arm="random_stratified")

    def test_flawless_arm_accepts_assistant_text(self):
        tools = _schemas()
        messages = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "done"},
        ]
        mask = [False, False, True]
        clean_id = "retail:1:clean"
        row_id = f"v6:flawless_only:{clean_id}"
        manifest_sha256 = _digest("flawless-selector-manifest")
        pool_sha256 = _digest("candidate-pool")
        clean_view_sha256 = _digest("clean-view")
        source_candidate_pair_ids = ["retail:1:pair-0"]
        metadata = {
            "arm": "v6_flawless_control",
            "selector": "flawless_only",
            "paper_arm": "flawless_only",
            "selection_seed": None,
            "source": "perfect_success",
            "domain": "retail",
            "task_id": "1",
            "task_identity": "retail:1",
            "trial": clean_id,
            "candidate_pair_id": None,
            "branch_id": None,
            "clean_id": clean_id,
            "source_example_id": row_id,
            "source_pair_id": clean_id,
            "source_split": "inner_train",
            "fit_split": "train_schedule",
            "source_trajectory_sha256": clean_view_sha256,
            "selector_manifest_sha256": manifest_sha256,
            "candidate_pool_sha256": pool_sha256,
            "source_candidate_pair_ids": source_candidate_pair_ids,
            "source_hashes": {
                "selector_manifest_sha256": manifest_sha256,
                "candidate_pool_sha256": pool_sha256,
                "clean_view_sha256": clean_view_sha256,
                "source_candidate_pair_ids_sha256": trainer.canonical_sha256(
                    source_candidate_pair_ids
                ),
            },
            "tool_schemas": tools,
            "tool_schemas_sha256": trainer.canonical_sha256(tools),
            "failed_assistant_message_indices": [],
            "failed_action_label_messages": 0,
            "future_clean_suffix_used": False,
            "fresh_recovery_suffix": False,
            "official_test_used": False,
            "materialization_protocol": "v6_sft_materialization_v1",
            "tokenizer_name": trainer.MODEL_ID,
            "tokenizer_revision": trainer.TOKENIZER_REVISION,
        }
        metadata["token_contract_input_sha256"] = trainer.canonical_sha256(
            {"messages": messages, "label_mask": mask, "tool_schemas": tools}
        )
        row = {
            "id": row_id,
            "messages": messages,
            "label_mask": mask,
            "metadata": metadata,
            "token_contract": _contract(
                FakeTokenizer(), messages, mask, tools
            ),
        }
        encoded = trainer.encode_row(
            FakeTokenizer(),
            trainer.validate_row(row, arm="flawless_only"),
            max_seq_len=100,
        )
        self.assertEqual(encoded["selected_assistant_text_messages"], 1)

    def test_tail_logit_contract_alignment(self):
        self.assertEqual(trainer.tail_logit_contract(100, 60), (41, 60))
        with self.assertRaises(ValueError):
            trainer.tail_logit_contract(100, 0)

    def test_tokenizer_fingerprint_is_deterministic_and_sensitive(self):
        first = trainer.tokenizer_fingerprint(FakeTokenizer())
        second = trainer.tokenizer_fingerprint(FakeTokenizer())
        self.assertEqual(first, second)
        changed = FakeTokenizer()
        changed.chat_template = "fake-v2"
        self.assertNotEqual(trainer.tokenizer_fingerprint(changed), first)


if __name__ == "__main__":
    unittest.main()
