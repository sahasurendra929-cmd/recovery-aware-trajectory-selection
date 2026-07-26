import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v5_3_train_only_pilot.py"
SPEC = importlib.util.spec_from_file_location(
    "run_v5_3_train_only_pilot_tested", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

MANIFEST_SCRIPT = ROOT / "scripts" / "prepare_v5_3_manifests.py"
MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "prepare_v5_3_manifests_for_pilot_test", MANIFEST_SCRIPT
)
MANIFEST_MODULE = importlib.util.module_from_spec(MANIFEST_SPEC)
assert MANIFEST_SPEC.loader is not None
MANIFEST_SPEC.loader.exec_module(MANIFEST_MODULE)


class V53TrainOnlyPilotTests(unittest.TestCase):
    def test_manifest_selection_is_frozen_stratified_and_outcome_free(self):
        tau2_root = ROOT / "data" / "raw" / "tau2-bench"
        split_path = (
            ROOT
            / "artifacts"
            / "v5_stage0"
            / "manifests"
            / "split_manifest.json"
        )
        if not tau2_root.is_dir() or not split_path.is_file():
            self.skipTest("pinned tau2 checkout or Stage-0 split unavailable")
        with tempfile.TemporaryDirectory() as directory:
            protocol_dir = Path(directory) / "v5_3_protocol"
            MANIFEST_MODULE.prepare(
                tau2_root=tau2_root,
                split_manifest_path=split_path,
                output_dir=protocol_dir,
                seed=MANIFEST_MODULE.stage1.SEED,
            )
            universe, pilot = MODULE.build_effective_universe_and_pilot(
                tau2_root=tau2_root,
                split_manifest_path=split_path,
                generation_manifest_path=(
                    protocol_dir / "generation_manifest.json"
                ),
            )
        self.assertEqual(universe["counts"]["total"], 70)
        self.assertEqual(universe["counts"]["ground_truth_route"], 70)
        self.assertEqual(
            universe["counts"]["structurally_excluded_before_partition"], 5
        )
        self.assertEqual(len(pilot["rows"]), 24)
        self.assertFalse(pilot["formal_data"])
        self.assertFalse(pilot["official_test_used"])
        self.assertFalse(pilot["pilot_trajectories_may_enter_formal_data"])
        self.assertFalse(
            pilot["selection"]["uses_rollout_or_validation_outcomes"]
        )
        observed = {
            (row["domain"], str(row["task_id"])) for row in pilot["rows"]
        }
        expected = {
            ("retail", value)
            for value in (
                "2",
                "8",
                "10",
                "15",
                "19",
                "25",
                "30",
                "35",
                "54",
                "67",
                "69",
                "72",
                "85",
                "92",
                "93",
                "104",
                "106",
                "110",
            )
        } | {
            ("airline", value)
            for value in ("1", "11", "14", "33", "38", "40")
        }
        self.assertEqual(observed, expected)

    def _write_synthetic_pilot(
        self,
        root: Path,
        *,
        one_pair_tasks: int,
        second_pair_tasks: int,
    ):
        task_ids = [f"retail:{index}" for index in range(24)]
        pilot = {
            "protocol": MODULE.PILOT_PROTOCOL,
            "official_test_used": False,
            "formal_data": False,
            "pilot_trajectories_may_enter_formal_data": False,
            "planned_task_ids": task_ids,
            "trial_seeds": list(MODULE.protocol.PILOT_TRIAL_SEEDS),
            "rows": [
                {
                    "domain": "retail",
                    "task_id": str(index),
                    "source_split": "derived_inner_train",
                    "teacher_route": "ground_truth",
                }
                for index in range(24)
            ],
        }
        pilot_path = root / "pilot_manifest.json"
        pilot_path.write_text(json.dumps(pilot), encoding="utf-8")

        result_paths = []
        for condition in ("clean", "error"):
            simulations = []
            for task_index in range(24):
                for attempt_index, seed in enumerate(
                    MODULE.protocol.PILOT_TRIAL_SEEDS
                ):
                    eligible = (
                        task_index < one_pair_tasks and attempt_index == 0
                    ) or (
                        task_index < second_pair_tasks and attempt_index == 1
                    )
                    simulation = {
                            "domain": "retail",
                            "task_id": str(task_index),
                            "id": f"{condition}-{task_index}-{attempt_index}",
                            "condition": condition,
                            "source_split": "derived_inner_train",
                            "trial": attempt_index,
                            "seed": seed,
                            "attempt_id": (
                                f"{condition}-{task_index}-{attempt_index}"
                            ),
                            "eligible": eligible,
                            "scoreable": True,
                            "reason": (
                                "eligible" if eligible else "final_failure"
                            ),
                            "termination_reason": "agent_stop",
                            "reward_info": {
                                "db_check": {"db_reward": float(eligible)}
                            },
                        }
                    if (
                        condition == "clean"
                        and task_index == 0
                        and attempt_index == 0
                    ):
                        simulation["reward_info"]["nl_assertions"] = [
                            {
                                "nl_assertion": "the requested task is complete",
                                "met": True,
                                "justification": "synthetic fixture",
                            }
                        ]
                    simulations.append(simulation)
            payload = {
                "source_split": "derived_inner_train",
                "condition": condition,
                "simulations": simulations,
            }
            path = root / f"retail_{condition}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result_paths.append(path)

        judge_path = (
            root
            / "logs"
            / "retail_clean"
            / "artifacts"
            / "task_0"
            / "sim_clean-0-0"
            / "llm_debug"
            / "strict_nl_judge_audit_fixture.json"
        )
        judge_path.parent.mkdir(parents=True)
        raw_content = json.dumps(
            [
                {
                    "nl_assertion": "the requested task is complete",
                    "met": True,
                    "justification": "synthetic fixture",
                }
            ],
            separators=(",", ":"),
        )
        judge_path.write_text(
            json.dumps(
                {
                    "protocol": "v5_strict_nl_judge_v1",
                    "status": "PASS",
                    "expected_outcomes": [
                        "the requested task is complete"
                    ],
                    "attempt_count": 1,
                    "attempts": [
                        {
                            "attempt": 1,
                            "call_name": (
                                "nl_assertions_eval_strict_attempt_1"
                            ),
                            "raw_content": raw_content,
                            "raw_content_utf8_bytes": len(
                                raw_content.encode("utf-8")
                            ),
                            "schema_status": "PASS",
                            "schema_error": None,
                            "raw_content_sha256": hashlib.sha256(
                                raw_content.encode("utf-8")
                            ).hexdigest(),
                            "raw_provider_response": {
                                "fixture": True
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return pilot_path, result_paths, judge_path

    def test_exact_15_and_4_boundary_passes_and_projects_above_55(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot, inputs, judge = self._write_synthetic_pilot(
                root,
                one_pair_tasks=15,
                second_pair_tasks=4,
            )
            result = MODULE.build_pilot_audit(
                input_paths=inputs,
                pilot_manifest_path=pilot,
                judge_audit_paths=[judge],
            )
            expected_pilot_sha = MODULE.file_sha256(pilot)
            expected_input_sha = {
                str(path.resolve()): MODULE.file_sha256(path)
                for path in sorted(
                    inputs, key=lambda item: str(item.resolve())
                )
            }
            expected_judge_sha = {
                judge.relative_to(root).as_posix(): MODULE.file_sha256(judge)
            }
        self.assertEqual(result["status"], "GO_FORMAL_GENERATION")
        observed = result["decision"]["observed"]
        self.assertEqual(observed["tasks_with_at_least_one_pair"], 15)
        self.assertEqual(observed["tasks_with_second_pair"], 4)
        self.assertEqual(observed["projected_formal_tasks_with_pair"], 43.75)
        self.assertAlmostEqual(
            observed["projected_formal_pairs"],
            55.416666666666664,
        )
        self.assertEqual(
            result["formal_gate_unchanged"]["minimum_distinct_task_ids"], 40
        )
        self.assertEqual(
            result["formal_gate_unchanged"]["minimum_pairs"], 48
        )
        self.assertEqual(
            result["provenance"]["pilot_manifest_sha256"],
            expected_pilot_sha,
        )
        self.assertEqual(
            result["provenance"]["input_sha256"],
            expected_input_sha,
        )
        self.assertEqual(
            result["provenance"]["strict_judge_audit_sha256"],
            expected_judge_sha,
        )

    def test_14_and_3_fails_closed_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot, inputs, judge = self._write_synthetic_pilot(
                root,
                one_pair_tasks=14,
                second_pair_tasks=3,
            )
            result = MODULE.build_pilot_audit(
                input_paths=inputs,
                pilot_manifest_path=pilot,
                judge_audit_paths=[judge],
            )
        self.assertEqual(result["status"], "NO_GO_STOP")
        self.assertFalse(
            result["claim_boundary"]["training_started"]
        )
        self.assertFalse(
            result["claim_boundary"]["official_test_used"]
        )

    def test_pilot_rejects_vacuous_zero_call_judge_evidence(self):
        evidence = {
            "protocol": MODULE.judge_contract.EVIDENCE_PROTOCOL,
            "status": "PASS",
            "expected_calls": 0,
            "observed_unique_pass_audits": 0,
            "calls": [],
            "canonical_mapping_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(
            MODULE.PilotError,
            "evidence mapping is malformed",
        ):
            MODULE._judge_audit_summary(evidence)


if __name__ == "__main__":
    unittest.main()
