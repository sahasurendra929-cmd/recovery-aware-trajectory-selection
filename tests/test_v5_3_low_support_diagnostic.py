from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts import prepare_v5_3_low_support_diagnostic as prepare
from scripts import prepare_v5_3_sft_causal as matching
from scripts import train_v5_sft_causal as train
from scripts import v5_3_12h_protocol as source
from scripts import v5_3_low_support_protocol as protocol


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def low_support_bundle(root: Path, arm: str = "perfect_success") -> dict:
    train_file = root / "arms" / arm / "train.jsonl"
    train_file.parent.mkdir(parents=True)
    train_file.write_text('{"id":"fixture"}\n', encoding="utf-8")
    validation_file = root / "validation_loss.jsonl"
    validation_file.write_bytes(b"")
    validation_manifest_path = root / "validation_manifest.json"
    validation_manifest = {
        "protocol": train.V5_3_VALIDATION_MANIFEST_PROTOCOL,
        "paired_task_count": 21,
        "rows": [
            {
                "domain": "retail" if index < 15 else "airline",
                "task_id": str(index),
                "source_split": "derived_validation",
            }
            for index in range(21)
        ],
        "official_test_used": False,
        "official_test_sealed": True,
    }
    write_json(validation_manifest_path, validation_manifest)
    validation_manifest_sha = train.sha256_file(validation_manifest_path)
    train_sha = train.sha256_file(train_file)
    validation_sha = train.sha256_file(validation_file)
    split_sha = "1" * 64
    dynamic = {
        "generation": {
            "protocol": train.DYNAMIC_AUDIT_PROTOCOL,
            "sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
            "split_manifest_sha256": split_sha,
            "source_split": "derived_inner_train",
            "verified_injections": 78,
            "official_test_used": False,
            "official_test_sealed": True,
        },
        "validation": {
            "protocol": train.DYNAMIC_AUDIT_PROTOCOL,
            "sha256": "4" * 64,
            "manifest_sha256": validation_manifest_sha,
            "split_manifest_sha256": split_sha,
            "source_split": "derived_validation",
            "verified_injections": 21,
            "official_test_used": False,
            "official_test_sealed": True,
        },
    }
    pair_counts = {task: 0 for task in protocol.PILOT_TASK_IDS}
    for task in protocol.PILOT_TASK_IDS[:8]:
        pair_counts[task] = 1
    for task in protocol.PILOT_TASK_IDS[:2]:
        pair_counts[task] = 2
    gate = protocol.data_gate(pair_counts)
    task_yield = {
        task: {"selected_pairs": count}
        for task, count in pair_counts.items()
    }
    arms = {}
    for name, ratio in protocol.ARM_RECOVERY_ROW_RATIOS.items():
        arms[name] = {
            "rows": 512,
            "recovery_rows": int(512 * ratio),
            "recovery_row_ratio": ratio,
            "expected_recovery_row_ratio": ratio,
            "recovery_mixture_basis": "row_mean_microbatch_equal_weight",
            "controlled_failed_action_labels": 0,
            "sha256": train_sha if name == arm else "5" * 64,
        }
    audit = {
        "status": "PASS",
        "protocol": train.DATA_AUDIT_PROTOCOL,
        "design_version": protocol.DESIGN_VERSION,
        "design_protocol": protocol.DATA_PROTOCOL,
        "diagnostic_protocol": protocol.PROTOCOL,
        "source_screen_protocol": source.PROTOCOL,
        "processing_source_commit": "7" * 40,
        "source_generation_commit": "8" * 40,
        "screen_task_ids": list(protocol.PILOT_TASK_IDS),
        "attempts_per_task_per_condition": 6,
        "expected_rollouts": 288,
        "post_yield_exploratory_diagnostic": True,
        "strict_source_gate_required_fail_closed": True,
        "artifact_namespace": dict(protocol.ARTIFACT_NAMESPACE),
        "diagnostic_outputs_may_enter_formal_v5_3": False,
        "screen_outputs_may_enter_formal_v5_3": False,
        "split_manifest_sha256": split_sha,
        "validation_manifest_sha256": validation_manifest_sha,
        "official_test_used": False,
        "official_test_sealed": True,
        "derived_validation_used_for_supervision": False,
        "train_validation_source_overlap": 0,
        "train_schedule_rows_per_arm": 512,
        "core_task_id_multiset_equal": True,
        "common_eligible_pair_support": True,
        "cross_arm_token_budgets_gate_training": False,
        "data_gate": gate,
        "task_yield": task_yield,
        "claim_boundary": dict(protocol.CLAIM_BOUNDARY),
        "arm_mixture_contract": {
            "weight_unit": "training_row",
            "target_recovery_row_ratios": (
                protocol.ARM_RECOVERY_ROW_RATIOS
            ),
            "target_miss_blocks_training": False,
        },
        "arms": arms,
        "generation_contracts": {
            "protocol": (
                f"{source.PROTOCOL}:generation_contract_audit_v1"
            ),
            "task_union": 24,
            "shards": 3,
            "official_test_used": False,
            "strict_judge_evidence_mapping_sha256": "6" * 64,
        },
        "dynamic_audits": dynamic,
        "validation_loss": {
            "rows": 0,
            "source_split": "not_applicable_fixed_step_screen",
            "used_for_checkpoint_selection": False,
            "validation_disabled_reason": "fixed_steps_exploratory",
            "sha256": validation_sha,
        },
        "label_guarantees": {
            "official_test_and_derived_validation_label_leakage": 0,
        },
    }
    audit_path = root / "audit.json"
    hashes_path = root / "hashes.json"
    write_json(audit_path, audit)
    hashes = {
        "audit.json": train.sha256_file(audit_path),
        f"arms/{arm}/train.jsonl": train_sha,
        "validation_loss.jsonl": validation_sha,
        "validation_manifest.json": validation_manifest_sha,
    }
    write_json(hashes_path, hashes)
    return {
        "arm": arm,
        "train_file": train_file,
        "validation_file": validation_file,
        "audit_path": audit_path,
        "hashes_path": hashes_path,
        "audit": audit,
        "hashes": hashes,
        "train_sha": train_sha,
        "validation_sha": validation_sha,
    }


def rewrite_bundle(bundle: dict) -> None:
    write_json(bundle["audit_path"], bundle["audit"])
    bundle["hashes"]["audit.json"] = train.sha256_file(
        bundle["audit_path"]
    )
    write_json(bundle["hashes_path"], bundle["hashes"])


class LowSupportProtocolTests(unittest.TestCase):
    def counts(self, tasks: int, pairs: int) -> dict[str, int]:
        values = {task: 0 for task in protocol.PILOT_TASK_IDS}
        for task in protocol.PILOT_TASK_IDS[:tasks]:
            values[task] = 1
        for task in protocol.PILOT_TASK_IDS[: max(0, pairs - tasks)]:
            values[task] = 2
        return values

    def test_gate_is_exact_and_requires_source_strict_failure(self) -> None:
        result = protocol.data_gate(self.counts(8, 10))
        self.assertEqual(result["status"], "PASS_LOW_SUPPORT_DIAGNOSTIC")
        self.assertTrue(result["training_authorized"])
        self.assertEqual(
            result["thresholds"],
            {
                "minimum_distinct_tasks": 8,
                "minimum_capped_pairs": 10,
                "maximum_pairs_per_task": 2,
            },
        )
        self.assertEqual(
            result["source_strict_gate"]["status"],
            "FAIL_CLOSED",
        )

        self.assertFalse(
            protocol.data_gate(self.counts(7, 10))["training_authorized"]
        )
        self.assertFalse(
            protocol.data_gate(self.counts(8, 9))["training_authorized"]
        )
        strict_pass = protocol.data_gate(self.counts(14, 17))
        self.assertFalse(strict_pass["training_authorized"])
        self.assertFalse(
            strict_pass["checks"]["source_strict_gate_is_fail_closed"]
        )

    def test_frozen_arm_and_claim_boundary(self) -> None:
        protocol.validate_constants()
        self.assertEqual(
            protocol.TRAINED_ARMS,
            ("perfect_success", "repair_50"),
        )
        self.assertEqual(
            protocol.EVAL_ARMS,
            ("base_model", "perfect_success", "repair_50"),
        )
        self.assertEqual(protocol.SCHEDULE_ROWS, 512)
        self.assertEqual(protocol.OPTIMIZER_STEPS, 64)
        self.assertEqual(protocol.EVALUATION_ROLLOUTS, 126)
        self.assertEqual(protocol.EVALUATION_SHARDS, 3)
        self.assertEqual(
            protocol.artifact_root(Path("/repo"), "processed_root"),
            Path("/repo/data/processed/v5_3_low_support_diagnostic"),
        )

    def test_handoff_freezes_lock_roots_commits_and_three_shards(self) -> None:
        handoff = (
            Path(__file__).resolve().parents[1]
            / "V5_3_LOW_SUPPORT_DIAGNOSTIC_HANDOFF.md"
        ).read_text(encoding="utf-8")
        for required in (
            "flock -s -n 9",
            "V5_3_LOW_SUPPORT_LOCK_FD=9",
            "--source-runtime-root \"$SOURCE_RUNTIME\"",
            "--expected-generation-source-commit",
            "--provenance-profile v5_3_low_support_diagnostic",
            "--num-shards 3",
            "diagnostic_summary.json",
            "official test",
            "run_v5_3_low_support_diagnostic.py",
            "--stage all",
            "runtime_service_evidence.json",
            "exactly 10 smokes",
            "DATA_GATE_FAIL_NO_TRAIN",
            "raw smoke responses",
            "PID/start-time",
            "V5_3_12H_SOURCE_SNAPSHOT_C3.json",
            "--source-snapshot-receipt",
            "tmux new-session",
            protocol.SOURCE_GENERATION_COMMIT,
        ):
            self.assertIn(required, handoff)

    def test_whole_run_lock_descriptor_is_exact_and_shared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            lock = repo / protocol.SOURCE_RUNTIME_ROOT / "controller.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text("source\n", encoding="utf-8")
            with lock.open("r", encoding="utf-8") as handle:
                previous = os.environ.get("V5_3_LOW_SUPPORT_LOCK_FD")
                os.environ["V5_3_LOW_SUPPORT_LOCK_FD"] = str(
                    handle.fileno()
                )
                try:
                    self.assertEqual(
                        protocol.require_whole_run_source_lock(repo),
                        handle.fileno(),
                    )
                    with lock.open("r", encoding="utf-8") as probe:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(
                                probe.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB,
                            )
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    if previous is None:
                        os.environ.pop("V5_3_LOW_SUPPORT_LOCK_FD", None)
                    else:
                        os.environ[
                            "V5_3_LOW_SUPPORT_LOCK_FD"
                        ] = previous

            with lock.open("r", encoding="utf-8") as controller:
                with lock.open("r", encoding="utf-8") as diagnostic:
                    fcntl.flock(controller.fileno(), fcntl.LOCK_EX)
                    previous = os.environ.get(
                        "V5_3_LOW_SUPPORT_LOCK_FD"
                    )
                    os.environ["V5_3_LOW_SUPPORT_LOCK_FD"] = str(
                        diagnostic.fileno()
                    )
                    try:
                        with self.assertRaisesRegex(
                            protocol.LowSupportProtocolError,
                            "overlap is forbidden",
                        ):
                            protocol.require_whole_run_source_lock(repo)
                    finally:
                        fcntl.flock(controller.fileno(), fcntl.LOCK_UN)
                        if previous is None:
                            os.environ.pop(
                                "V5_3_LOW_SUPPORT_LOCK_FD", None
                            )
                        else:
                            os.environ[
                                "V5_3_LOW_SUPPORT_LOCK_FD"
                            ] = previous
        self.assertFalse(protocol.CLAIM_BOUNDARY["formal_v5_3_result"])
        self.assertTrue(protocol.CLAIM_BOUNDARY["official_test_sealed"])

    def test_prepare_module_reuses_strict_source_contract(self) -> None:
        self.assertEqual(
            prepare.SCREEN_MANIFEST_PROTOCOL,
            f"{source.PROTOCOL}:manifest_v1",
        )
        self.assertEqual(
            prepare.diagnostic.TRAINED_ARMS,
            ("perfect_success", "repair_50"),
        )
        self.assertEqual(
            prepare.diagnostic.MAX_PAIRS_PER_TASK,
            source.MAX_PAIRS_PER_TASK,
        )

    def test_joint_matcher_accepts_only_the_two_registered_adapters(self) -> None:
        task = "retail:1"

        def candidate(
            identifier: str,
            *,
            source_name: str,
            supervised_tokens: int,
        ) -> dict:
            return {
                "id": identifier,
                "token_contract": {
                    "supervised_tokens": supervised_tokens,
                    "sequence_tokens": supervised_tokens + 50,
                },
                "metadata": {"source": source_name},
            }

        perfect = candidate(
            "perfect",
            source_name="perfect_success",
            supervised_tokens=50,
        )
        repair = candidate(
            "repair",
            source_name="failure_rich",
            supervised_tokens=150,
        )
        candidates = {
            "perfect_success": {task: [perfect]},
            "repair_50": {task: [perfect, repair]},
        }
        schedules, certificate = (
            matching._deterministic_tolerance_aware_joint_match(
                [task, task],
                candidates,
                expected_arms=protocol.TRAINED_ARMS,
                recovery_ratios=None,
                recovery_row_ratios=(
                    protocol.ARM_RECOVERY_ROW_RATIOS
                ),
                seed=protocol.BASE_SEED,
                enforce_cross_arm_budget_tolerances=False,
                beam_width_schedule=(16,),
                final_width=16,
            )
        )
        self.assertIsNotNone(schedules)
        self.assertEqual(set(schedules), set(protocol.TRAINED_ARMS))
        self.assertEqual(
            certificate["recovery_row_ratio_by_arm"]["repair_50"],
            0.5,
        )
        with self.assertRaisesRegex(RuntimeError, "candidate arms drift"):
            matching._deterministic_tolerance_aware_joint_match(
                [task, task],
                {
                    **candidates,
                    "repair_100": {task: [repair]},
                },
                expected_arms=protocol.TRAINED_ARMS,
                recovery_ratios=None,
                recovery_row_ratios=(
                    protocol.ARM_RECOVERY_ROW_RATIOS
                ),
                seed=protocol.BASE_SEED,
            )

    def test_schedule_builder_emits_only_exact_two_arm_support(self) -> None:
        pairs = {}
        for index in range(8):
            task = f"retail:{index}"
            perfect = {
                "id": f"{task}:perfect",
                "token_contract": {
                    "supervised_tokens": 50,
                    "sequence_tokens": 100,
                },
                "metadata": {
                    "source": "perfect_success",
                    "domain": "retail",
                    "task_id": str(index),
                },
            }
            repair = {
                "id": f"{task}:repair",
                "token_contract": {
                    "supervised_tokens": 150,
                    "sequence_tokens": 200,
                },
                "metadata": {
                    "source": "failure_rich",
                    "domain": "retail",
                    "task_id": str(index),
                },
            }
            pairs[task] = [
                {
                    "perfect_success": perfect,
                    "repair_masked": repair,
                }
            ]
        schedules, certificate = prepare.build_diagnostic_schedules(
            pairs,
            schedule_rows=16,
        )
        self.assertIsNotNone(schedules)
        self.assertEqual(set(schedules), set(protocol.TRAINED_ARMS))
        self.assertEqual(
            {arm: len(rows) for arm, rows in schedules.items()},
            {"perfect_success": 16, "repair_50": 16},
        )
        task_counts = {
            arm: {
                task: sum(
                    row["metadata"]["task_id"] == task
                    for row in rows
                )
                for task in {str(index) for index in range(8)}
            }
            for arm, rows in schedules.items()
        }
        self.assertEqual(
            task_counts["perfect_success"],
            task_counts["repair_50"],
        )
        self.assertEqual(
            certificate["recovery_rows_by_arm"],
            {"perfect_success": 0, "repair_50": 8},
        )


class LowSupportTrainingProvenanceTests(unittest.TestCase):
    def validate(self, bundle: dict) -> dict:
        return train.validate_training_data_provenance(
            arm=bundle["arm"],
            train_file=bundle["train_file"],
            validation_file=bundle["validation_file"],
            data_audit_path=bundle["audit_path"],
            data_hashes_path=bundle["hashes_path"],
            expected_train_sha256=bundle["train_sha"],
            expected_validation_sha256=bundle["validation_sha"],
        )

    def test_provenance_accepts_only_exact_low_support_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = low_support_bundle(Path(directory))
            result = self.validate(bundle)
        design = result["design_provenance"]
        self.assertEqual(
            design["design_version"],
            protocol.DESIGN_VERSION,
        )
        self.assertEqual(
            design["trained_arms"],
            ["perfect_success", "repair_50"],
        )
        self.assertEqual(design["eligible_distinct_tasks"], 8)
        self.assertEqual(design["eligible_capped_pairs"], 10)
        self.assertEqual(design["processing_source_commit"], "7" * 40)
        self.assertEqual(design["source_generation_commit"], "8" * 40)
        self.assertFalse(design["formal_v5_3_result"])
        self.assertTrue(design["official_test_sealed"])

    def test_strict_gate_pass_cannot_enter_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = low_support_bundle(Path(directory))
            strict = bundle["audit"]["data_gate"]["source_strict_gate"]
            strict["status"] = "PASS"
            strict["training_authorized"] = True
            rewrite_bundle(bundle)
            with self.assertRaisesRegex(
                RuntimeError,
                "low-support data gate drift",
            ):
                self.validate(bundle)

    def test_gate_is_recomputed_from_task_yield(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = low_support_bundle(Path(directory))
            strict = bundle["audit"]["data_gate"]["source_strict_gate"]
            strict["status"] = "PASS"
            strict["training_authorized"] = True
            rewrite_bundle(bundle)
            with self.assertRaisesRegex(
                RuntimeError,
                "low-support data gate drift",
            ):
                self.validate(bundle)

        with tempfile.TemporaryDirectory() as directory:
            bundle = low_support_bundle(Path(directory))
            bundle["audit"]["data_gate"]["claim_boundary"][
                "formal_v5_3_result"
            ] = True
            rewrite_bundle(bundle)
            with self.assertRaisesRegex(
                RuntimeError,
                "low-support data gate drift",
            ):
                self.validate(bundle)

    def test_top_level_official_test_seal_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = low_support_bundle(Path(directory))
            bundle["audit"]["official_test_sealed"] = False
            rewrite_bundle(bundle)
            with self.assertRaisesRegex(
                RuntimeError,
                "design identity drift",
            ):
                self.validate(bundle)

    def test_processing_and_generation_commits_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = low_support_bundle(Path(directory))
            bundle["audit"]["source_generation_commit"] = (
                bundle["audit"]["processing_source_commit"]
            )
            rewrite_bundle(bundle)
            with self.assertRaisesRegex(
                RuntimeError,
                "source-commit provenance drift",
            ):
                self.validate(bundle)

    def test_extra_or_missing_arm_fails_closed(self) -> None:
        for mutation in ("extra", "missing"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                bundle = low_support_bundle(Path(directory))
                if mutation == "extra":
                    bundle["audit"]["arms"]["repair_100"] = dict(
                        bundle["audit"]["arms"]["repair_50"]
                    )
                else:
                    del bundle["audit"]["arms"]["repair_50"]
                rewrite_bundle(bundle)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "arm/matching contract drift",
                ):
                    self.validate(bundle)

    def test_claim_or_validation_seal_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = low_support_bundle(Path(directory))
            bundle["audit"]["claim_boundary"][
                "formal_v5_3_result"
            ] = True
            rewrite_bundle(bundle)
            with self.assertRaisesRegex(
                RuntimeError,
                "claim boundary drift",
            ):
                self.validate(bundle)


if __name__ == "__main__":
    unittest.main()
