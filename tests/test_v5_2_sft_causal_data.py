import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path

from tests.test_v5_sft_causal_data import (
    FakeTokenizer,
    TRAIN_MODULE,
    simulation,
)
from tests.v5_multifault_fixtures import (
    multifault_manifest,
    multifault_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_v5_2_sft_causal.py"
SPEC = importlib.util.spec_from_file_location("prepare_v5_2_sft_causal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
PUBLISHED_SPLIT = ROOT / "artifacts" / "v5_stage0" / "manifests" / "split_manifest.json"


class BalancedFakeTokenizer(FakeTokenizer):
    """Feasible synthetic contract for the two simultaneous token budgets."""

    @staticmethod
    def _size(message):
        calls = message.get("tool_calls") or []
        if message.get("role") == "assistant" and calls:
            function = calls[0].get("function") or calls[0]
            if function.get("name") == "lookup":
                # Successful calls dominate the tiny controlled-failure label,
                # so matched support can satisfy the registered 1%/2% gates.
                return 1000 + len(
                    json.dumps(function.get("arguments") or {}, sort_keys=True)
                ) // 2
        return FakeTokenizer._size(message)


def tool_schemas():
    result = []
    for name, argument in (
        ("lookup_user", "user_id"),
        ("lookup_order", "order_id"),
        ("lookup_reservation", "reservation_id"),
        ("lookup_flight", "flight_number"),
        ("lookup", "value"),
    ):
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "read-only lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {argument: {"type": "string"}},
                        "required": [argument],
                    },
                },
            }
        )
    return result


class V52Fixture:
    def __init__(self, root: Path, *, successful_train_tasks: set[str] | None = None):
        self.root = root
        self.raw = root / "raw"
        self.raw.mkdir()
        self.output = root / "out"
        self.split_path = PUBLISHED_SPLIT
        self.split = json.loads(PUBLISHED_SPLIT.read_text(encoding="utf-8"))
        self.partition = MODULE.build_task_partition(
            self.split,
            split_manifest_sha256=MODULE.v5.sha256_file(self.split_path),
        )
        self.train_task_keys = {
            f"{domain}:{task_id}"
            for domain in ("retail", "airline")
            for task_id in self.partition["domains"][domain]["train_ids"]
        }
        self.loss_task_keys = {
            f"{domain}:{task_id}"
            for domain in ("retail", "airline")
            for task_id in self.partition["domains"][domain]["loss_validation_ids"]
        }
        successful = (
            self.train_task_keys
            if successful_train_tasks is None
            else set(successful_train_tasks)
        ) | self.loss_task_keys
        generation_rows = multifault_rows(
            {
                domain: self.split["domains"][domain]["inner_train_ids"]
                for domain in ("retail", "airline")
            },
            source_split="derived_inner_train",
        )
        self.generation_manifest = root / "generation_manifest.json"
        self.generation_manifest.write_text(
            json.dumps(
                multifault_manifest(
                    generation_rows,
                    protocol="v5_stage1_multifault_data_construction",
                    source_split="derived_inner_train",
                    split_sha256=MODULE.v5.sha256_file(self.split_path),
                )
            ),
            encoding="utf-8",
        )
        for domain in ("retail", "airline"):
            task_ids = [
                str(value)
                for value in self.split["domains"][domain]["inner_train_ids"]
            ]
            for condition in ("clean", "error"):
                rows = []
                for task_id in task_ids:
                    task_key = f"{domain}:{task_id}"
                    for attempt in range(MODULE.ATTEMPTS_PER_TASK):
                        row = simulation(
                            domain,
                            task_id,
                            attempt,
                            condition=condition,
                            final_success=task_key in successful,
                        )
                        row["attempt_index"] = attempt
                        # Condition-specific seeds exercise cross-seed pairing.
                        row["seed"] = (
                            100_000
                            + attempt
                            + (50_000 if condition == "error" else 0)
                        )
                        rows.append(row)
                (self.raw / f"{domain}_{condition}.json").write_text(
                    json.dumps({"simulations": rows}),
                    encoding="utf-8",
                )
        schemas = tool_schemas()
        self.contexts = {
            "retail": {"policy": "retail policy", "tool_schemas": schemas},
            "airline": {"policy": "airline policy", "tool_schemas": schemas},
        }

    def run(self, *, tokenizer=None):
        return MODULE.prepare(
            split_manifest=self.split_path,
            generation_manifest=self.generation_manifest,
            raw_dir=self.raw,
            output_dir=self.output,
            tokenizer=tokenizer or BalancedFakeTokenizer(),
            tokenizer_name=MODULE.v5.MODEL,
            tokenizer_revision=MODULE.v5.MODEL_REVISION,
            domain_contexts=self.contexts,
            schedule_rows=MODULE.v5.SCHEDULE_ROWS,
            strict_generation_contracts=False,
            strict_dynamic_audits=False,
        )

    @staticmethod
    def read_jsonl(path: Path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]


class V52PartitionTests(unittest.TestCase):
    def test_frozen_partition_ids_and_digest(self):
        split = json.loads(PUBLISHED_SPLIT.read_text(encoding="utf-8"))
        partition = MODULE.build_task_partition(
            split,
            split_manifest_sha256=MODULE.v5.sha256_file(PUBLISHED_SPLIT),
        )
        self.assertEqual(
            partition["canonical_sha256"],
            MODULE.EXPECTED_PARTITION_SHA256,
        )
        self.assertEqual(
            partition["domains"]["retail"]["loss_validation_ids"],
            ["21", "28", "43", "99", "110", "112"],
        )
        self.assertEqual(
            partition["domains"]["airline"]["loss_validation_ids"],
            ["14", "27"],
        )
        self.assertEqual(
            sum(row["counts"]["train"] for row in partition["domains"].values()),
            75,
        )
        self.assertEqual(
            sum(
                row["counts"]["loss_validation"]
                for row in partition["domains"].values()
            ),
            8,
        )
        self.assertFalse(partition["selection_uses_generation_outcomes"])
        self.assertFalse(partition["selection_uses_validation_or_test"])

    def test_candidate_rank_ignores_reward_and_messages(self):
        attempts = {
            index: simulation(
                "retail",
                "1",
                index,
                condition="clean",
                final_success=index % 2 == 0,
            )
            for index in range(MODULE.ATTEMPTS_PER_TASK)
        }
        for index, row in attempts.items():
            row["attempt_index"] = index
            row["seed"] = 7000 + index
        changed = deepcopy(attempts)
        for row in changed.values():
            row["reward_info"]["reward"] = 1.0 - row["reward_info"]["reward"]
            row["messages"][-1]["content"] = "outcome-dependent content changed"
        before = [
            MODULE.attempt_identity(row)
            for row in sorted(
                attempts.values(),
                key=lambda row: MODULE.attempt_order_key(
                    domain="retail",
                    task_id="1",
                    condition="clean",
                    simulation=row,
                ),
            )
        ]
        after = [
            MODULE.attempt_identity(row)
            for row in sorted(
                changed.values(),
                key=lambda row: MODULE.attempt_order_key(
                    domain="retail",
                    task_id="1",
                    condition="clean",
                    simulation=row,
                ),
            )
        ]
        self.assertEqual(before, after)

    def test_semantic_dedup_ignores_provider_tool_call_ids(self):
        first = simulation(
            "retail",
            "1",
            0,
            condition="clean",
            final_success=True,
        )
        first["attempt_index"] = 0
        first["seed"] = 7000
        second = deepcopy(first)
        second["attempt_index"] = 1
        second["seed"] = 7001
        second["trial"] = 1
        second["messages"][2]["tool_calls"][0]["id"] = "provider-random-b"
        second["messages"][3]["id"] = "provider-random-b"
        eligible, audit_rows, reasons = MODULE._rank_and_filter(
            domain="retail",
            task_id="1",
            condition="clean",
            attempts={0: first, 1: second},
            seed=MODULE.v5.SEED,
        )
        self.assertEqual(len(eligible), 1)
        self.assertEqual(reasons["duplicate_eligible_trajectory"], 1)
        self.assertEqual(
            len({row["trajectory_sha256"] for row in audit_rows}),
            1,
        )
        duplicate = [
            row
            for row in audit_rows
            if row["reason"] == "duplicate_eligible_trajectory"
        ]
        self.assertEqual(len(duplicate), 1)
        self.assertFalse(duplicate[0]["eligible"])

    def test_historical_v5_defaults_remain_three_same_seed_trials(self):
        import inspect

        signature = inspect.signature(MODULE.v5.prepare)
        self.assertEqual(signature.parameters["expected_trials"].default, 3)
        self.assertEqual(signature.parameters["slots_per_task"].default, 3)
        self.assertEqual(signature.parameters["min_paired_slots"].default, 120)
        self.assertEqual(MODULE.EXPECTED_GENERATION_SHARDS, 3)
        self.assertEqual(
            MODULE.EXPECTED_TEACHER_API_BASE_BY_SHARD,
            {
                0: "http://127.0.0.1:8011/v1",
                1: "http://127.0.0.1:8012/v1",
                2: "http://127.0.0.1:8013/v1",
            },
        )


class V52FullBuilderTests(unittest.TestCase):
    def test_overlength_attempt_is_excluded_before_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = V52Fixture(Path(directory))
            path = fixture.raw / "retail_clean.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            target = next(
                row
                for row in payload["simulations"]
                if row["task_id"] == "1" and row["attempt_index"] == 0
            )
            target["messages"][2]["tool_calls"][0]["arguments"]["value"] = (
                "x" * 200_000
            )
            path.write_text(json.dumps(payload), encoding="utf-8")

            audit = fixture.run()
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(
                audit["exclusions"]["clean"][
                    "training_sequence_exceeds_8192"
                ],
                1,
            )
            attempt_rows = fixture.read_jsonl(
                fixture.output / "attempt_audit.jsonl"
            )
            rejected = [
                row
                for row in attempt_rows
                if row["domain"] == "retail"
                and row["task_id"] == "1"
                and row["condition"] == "clean"
                and row["attempt_index"] == 0
            ]
            self.assertEqual(len(rejected), 1)
            self.assertFalse(rejected[0]["eligible"])
            self.assertEqual(
                rejected[0]["reason"],
                "training_sequence_exceeds_8192",
            )
            self.assertGreater(
                max(rejected[0]["training_sequence_tokens"].values()),
                MODULE.MAX_SEQUENCE_TOKENS,
            )

    def test_builds_compatible_four_arms_with_common_task_support(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = V52Fixture(Path(directory))
            audit = fixture.run()
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["protocol"], "v5_stage1_sft_causal")
            self.assertEqual(audit["design_version"], "5.2")
            self.assertFalse(audit["official_test_used"])
            self.assertEqual(audit["attempts_per_task_per_condition"], 12)
            self.assertGreaterEqual(audit["eligible_distinct_tasks"], 40)
            self.assertGreaterEqual(audit["paired_master_slots"], 48)
            self.assertTrue(audit["core_task_id_multiset_equal"])
            self.assertTrue(audit["common_eligible_pair_support"])
            self.assertEqual(audit["cross_seed_pair_fraction"], 1.0)
            self.assertLessEqual(
                audit["cross_arm_supervised_token_relative_range"],
                0.01,
            )
            self.assertLessEqual(
                audit["cross_arm_nonpadding_token_relative_range"],
                0.02,
            )
            master = fixture.read_jsonl(
                fixture.output / "paired_master_pool.jsonl"
            )
            train_pair_support = {
                row["pair_id"]
                for row in master
                if row["partition"] == "arm_train"
            }
            task_multisets = []
            for arm in MODULE.v5.CORE_ARMS:
                rows = fixture.read_jsonl(
                    fixture.output / "arms" / arm / "train.jsonl"
                )
                self.assertEqual(len(rows), 512)
                self.assertTrue(
                    {row["metadata"]["pair_id"] for row in rows}
                    <= train_pair_support
                )
                task_multisets.append(
                    Counter(
                        (row["metadata"]["domain"], row["metadata"]["task_id"])
                        for row in rows
                    )
                )
                for row in rows:
                    self.assertEqual(row["metadata"]["design_version"], "5.2")
                    self.assertFalse(row["metadata"]["official_test_used"])
                encoded = TRAIN_MODULE.encode_rows(
                    BalancedFakeTokenizer(),
                    rows[:4],
                    arm=arm,
                    split="train",
                    max_seq_len=100000,
                )
                self.assertEqual(len(encoded), 4)
            self.assertTrue(
                all(value == task_multisets[0] for value in task_multisets)
            )
            raw_rows = fixture.read_jsonl(
                fixture.output / "arms" / "failure_raw" / "train.jsonl"
            )
            repair_rows = fixture.read_jsonl(
                fixture.output / "arms" / "repair_100" / "train.jsonl"
            )
            for row in raw_rows:
                failed = row["metadata"]["failed_assistant_message_indices"]
                self.assertEqual(len(failed), 1)
                self.assertTrue(row["label_mask"][failed[0]])
            for row in repair_rows:
                failed = row["metadata"]["failed_assistant_message_indices"]
                self.assertEqual(len(failed), 1)
                self.assertFalse(row["label_mask"][failed[0]])
            validation_rows = fixture.read_jsonl(
                fixture.output / "validation_loss.jsonl"
            )
            self.assertTrue(validation_rows)
            self.assertTrue(
                {
                    f"{row['metadata']['domain']}:{row['metadata']['task_id']}"
                    for row in validation_rows
                }
                <= fixture.loss_task_keys
            )
            self.assertFalse(
                {
                    f"{row['metadata']['domain']}:{row['metadata']['task_id']}"
                    for arm in MODULE.v5.CORE_ARMS
                    for row in fixture.read_jsonl(
                        fixture.output / "arms" / arm / "train.jsonl"
                    )
                }
                & fixture.loss_task_keys
            )
            attempt_audit = fixture.read_jsonl(
                fixture.output / "attempt_audit.jsonl"
            )
            self.assertEqual(len(attempt_audit), 83 * 12 * 2)
            self.assertTrue(
                all(
                    row["source_split"] == "derived_inner_train"
                    and isinstance(row["scoreable"], bool)
                    for row in attempt_audit
                )
            )

    def test_gate_failure_writes_only_audit_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = json.loads(PUBLISHED_SPLIT.read_text(encoding="utf-8"))
            partition = MODULE.build_task_partition(
                split,
                split_manifest_sha256=MODULE.v5.sha256_file(PUBLISHED_SPLIT),
            )
            ordered_train = sorted(
                f"{domain}:{task_id}"
                for domain in ("retail", "airline")
                for task_id in partition["domains"][domain]["train_ids"]
            )
            fixture = V52Fixture(
                root,
                successful_train_tasks=set(ordered_train[:39]),
            )
            with self.assertRaises(MODULE.PoolGateError):
                fixture.run()
            audit = json.loads(
                (fixture.output / "audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["status"], "FAIL_CLOSED")
            self.assertEqual(audit["decision"], "DO_NOT_TRAIN")
            self.assertEqual(
                audit["formal_data_gate"]["observed_distinct_train_tasks"],
                39,
            )
            self.assertFalse(audit["arm_files_written"])
            self.assertFalse((fixture.output / "arms").exists())
            self.assertTrue((fixture.output / "attempt_audit.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
