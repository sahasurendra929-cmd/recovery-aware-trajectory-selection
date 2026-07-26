from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts import prepare_v5_3_12h_screen as data
from scripts import run_v5_3_12h_screen as controller
from scripts import run_v5_3_single_host as base_controller
from scripts import summarize_v5_3_12h_screen as summary
from scripts import train_v5_sft_causal as train
from scripts import v5_3_12h_protocol as protocol
from tests.v5_multifault_fixtures import (
    SOURCE_FILES,
    fault_protocol,
    multifault_row,
)


class ScreenControllerTests(unittest.TestCase):
    def test_deadline_resume_never_resets_t0(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(runtime_root=root)
            first = controller.load_or_create_deadline(
                args, allow_create=True
            )
            first_bytes = controller.deadline_path(args).read_bytes()
            second = controller.load_or_create_deadline(
                args, allow_create=True
            )
            self.assertEqual(first, second)
            self.assertEqual(
                first_bytes, controller.deadline_path(args).read_bytes()
            )
            self.assertEqual(
                datetime.fromisoformat(first["started_at_utc"]),
                datetime.fromisoformat(second["started_at_utc"]),
            )

    def test_external_sigterm_preserves_hash_valid_core_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                runtime_root=root / "runtime",
                results_root=root / "results",
            )
            args.runtime_root.mkdir(parents=True)
            args.results_root.mkdir(parents=True)
            deadline = protocol.make_deadline_receipt(
                datetime.now(timezone.utc)
            )
            controller.atomic_write(
                controller.deadline_path(args), deadline
            )
            previous = controller._GLOBAL_ARGS
            controller._GLOBAL_ARGS = args
            status_seen_before_stubborn_cleanup = []

            def stubborn_child_cleanup() -> None:
                # Simulate a child that consumes the full TERM grace period:
                # the claim-bearing terminal must already be durable.
                status_seen_before_stubborn_cleanup.append(
                    json.loads(
                        controller.terminal_path(args).read_text(
                            encoding="utf-8"
                        )
                    )["status"]
                )

            try:
                with (
                    mock.patch.object(
                        controller,
                        "_terminate_all_process_groups",
                        side_effect=stubborn_child_cleanup,
                    ),
                    mock.patch.object(
                        controller,
                        "validated_core_snapshot",
                        return_value=True,
                    ),
                    self.assertRaises(controller.ScreenStageError),
                ):
                    controller._external_stop_signal(signal.SIGTERM, None)
            finally:
                controller._GLOBAL_ARGS = previous
            terminal = json.loads(
                controller.terminal_path(args).read_text(encoding="utf-8")
            )
            self.assertEqual(
                terminal["status"],
                "CORE_COMPLETE_EXTENSION_INCOMPLETE",
            )
            self.assertFalse(terminal["no_scientific_claim"])
            self.assertEqual(
                status_seen_before_stubborn_cleanup,
                ["CORE_COMPLETE_EXTENSION_INCOMPLETE"],
            )

    def test_screen_hardware_override_is_explicit_and_local(self) -> None:
        original_model = base_controller.EXPECTED_GPU_MODEL
        original_memory = base_controller.MIN_GPU_MEMORY_MIB
        original_disk = base_controller.MIN_FREE_DISK_GIB
        try:
            self.assertEqual(original_disk, 140)
            controller.configure_screen_hardware()
            self.assertEqual(
                base_controller.EXPECTED_GPU_MODEL,
                "NVIDIA GeForce RTX 5090",
            )
            self.assertEqual(base_controller.MIN_GPU_MEMORY_MIB, 30_000)
            self.assertEqual(controller.SCREEN_MIN_FREE_DISK_GIB, 80)
            self.assertEqual(base_controller.MIN_FREE_DISK_GIB, 80)
        finally:
            base_controller.EXPECTED_GPU_MODEL = original_model
            base_controller.MIN_GPU_MEMORY_MIB = original_memory
            base_controller.MIN_FREE_DISK_GIB = original_disk
        self.assertEqual(base_controller.MIN_FREE_DISK_GIB, 140)

    def test_evaluation_command_separates_agent_and_14b_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                serve_python=Path("/serve/python"),
                tau2_root=root / "tau2",
                protocol_root=root / "protocol",
                results_root=root / "results",
            )
            command = controller.evaluation_command(
                args, "repair_50", 1
            )
            joined = " ".join(map(str, command))
            self.assertIn("http://127.0.0.1:8102/v1", joined)
            self.assertIn("http://127.0.0.1:8001/v1", joined)
            self.assertIn(protocol.USER_JUDGE_MODEL_ID, command)
            self.assertNotIn("--provenance-profile", command)
            self.assertIn(str(protocol.BASE_SEED), command)

    def test_evaluation_is_shard_parallel_and_alias_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                serve_python=Path("/serve/python"),
                tau2_root=root / "tau2",
                protocol_root=root / "protocol",
                results_root=root / "results",
            )
            specs = controller.evaluation_service_specifications(args)
            agent_specs = [row for row in specs if row["gpu"] != 0]
            self.assertEqual(
                [(row["gpu"], row["port"]) for row in agent_specs],
                [(1, 8101), (2, 8102), (3, 8103)],
            )
            self.assertTrue(all(row["adapters"] for row in agent_specs))
            self.assertEqual(
                {row["alias"] for row in agent_specs},
                {
                    protocol.MODEL_IDS["base_model"].removeprefix(
                        "openai/"
                    )
                },
            )
            for shard in range(3):
                for arm in protocol.ALL_EVAL_ARMS:
                    command = controller.evaluation_command(
                        args, arm, shard
                    )
                    self.assertEqual(
                        command[command.index("--agent-api-base") + 1],
                        f"http://127.0.0.1:{8101 + shard}/v1",
                    )
                    self.assertEqual(
                        command[command.index("--agent-model") + 1],
                        protocol.MODEL_IDS[arm],
                    )

    def test_evaluation_service_commands_freeze_role_specific_precision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                vllm=Path("/serve/vllm"),
                results_root=root / "results",
            )
            user_judge, *agents = controller.evaluation_service_specifications(
                args
            )
            user_command = controller._evaluation_service_command(
                args, user_judge
            )
            self.assertEqual(
                user_command[user_command.index("--dtype") + 1],
                "float16",
            )
            self.assertEqual(
                user_command[user_command.index("--quantization") + 1],
                "awq",
            )
            self.assertEqual(user_command.count("--tokenizer-revision"), 1)
            self.assertEqual(
                user_command[
                    user_command.index("--tokenizer-revision") + 1
                ],
                user_judge["revision"],
            )
            self.assertNotIn("--enable-lora", user_command)

            for specification in agents:
                command = controller._evaluation_service_command(
                    args, specification
                )
                self.assertEqual(
                    command[command.index("--dtype") + 1],
                    "bfloat16",
                )
                self.assertNotIn("--quantization", command)
                self.assertEqual(command.count("--tokenizer-revision"), 1)
                self.assertEqual(
                    command[command.index("--tokenizer-revision") + 1],
                    specification["revision"],
                )
                self.assertIn("--enable-lora", command)
                modules = command[command.index("--lora-modules") + 1 :]
                self.assertEqual(len(modules), len(protocol.TRAINED_ARMS))
                self.assertEqual(
                    {module.split("=", 1)[0] for module in modules},
                    {
                        protocol.MODEL_IDS[arm].removeprefix("openai/")
                        for arm in protocol.TRAINED_ARMS
                    },
                )

    def test_evaluation_alias_smoke_forces_exactly_one_token(self) -> None:
        response = {
            "model": "v5-3-12h-repair-50",
            "choices": [{"finish_reason": "length", "message": {"content": "A"}}],
        }
        with mock.patch.object(
            controller.base, "http_post_json", return_value=response
        ) as post:
            observed = controller._smoke_evaluation_alias(
                port=8102,
                api_key="screen-agent-local",
                model_alias="v5-3-12h-repair-50",
            )
        self.assertEqual(observed, response)
        url, api_key, payload = post.call_args.args
        self.assertEqual(
            url, "http://127.0.0.1:8102/v1/chat/completions"
        )
        self.assertEqual(api_key, "screen-agent-local")
        self.assertEqual(payload["model"], "v5-3-12h-repair-50")
        self.assertEqual(payload["max_tokens"], 1)
        self.assertEqual(payload["temperature"], 0)
        self.assertFalse(payload["stream"])

    def test_evaluation_alias_smoke_rejects_wrong_routed_model(self) -> None:
        response = {
            "model": "v5-3-12h-base",
            "choices": [{"finish_reason": "stop", "message": {"content": "A"}}],
        }
        with (
            mock.patch.object(
                controller.base, "http_post_json", return_value=response
            ),
            self.assertRaisesRegex(
                controller.ScreenStageError, "alias smoke failed"
            ),
        ):
            controller._smoke_evaluation_alias(
                port=8102,
                api_key="screen-agent-local",
                model_alias="v5-3-12h-repair-50",
            )

    def test_evaluation_start_smokes_every_alias_on_every_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                vllm=Path("/serve/vllm"),
                results_root=Path(temporary) / "results",
                health_timeout=30,
            )
            processes = [mock.Mock(pid=number) for number in range(4)]
            handles = [mock.Mock() for _ in range(4)]
            with (
                mock.patch.object(
                    controller.base,
                    "spawn_logged",
                    side_effect=list(zip(processes, handles)),
                ),
                mock.patch.object(controller, "_register_process"),
                mock.patch.object(controller.base, "write_pid"),
                mock.patch.object(controller.base, "wait_for_service"),
                mock.patch.object(
                    controller, "_smoke_evaluation_alias"
                ) as smoke,
            ):
                services = controller._start_evaluation_services(args)
        self.assertEqual(len(services), 4)
        observed = {
            (call.kwargs["port"], call.kwargs["model_alias"])
            for call in smoke.call_args_list
        }
        agent_aliases = {
            protocol.MODEL_IDS[arm].removeprefix("openai/")
            for arm in protocol.ALL_EVAL_ARMS
        }
        expected = {
            (8001, protocol.USER_JUDGE_MODEL_ID.removeprefix("openai/")),
            *{
                (port, alias)
                for port in (8101, 8102, 8103)
                for alias in agent_aliases
            },
        }
        self.assertEqual(observed, expected)
        self.assertEqual(smoke.call_count, 16)

    def test_evaluation_rejects_unregistered_mixed_arm_set(self) -> None:
        with self.assertRaisesRegex(
            controller.ScreenStageError,
            "exactly registered core or extension",
        ):
            controller._evaluate_arm_set(
                SimpleNamespace(),
                ("base_model", "failure_raw"),
            )

    def test_screen_docs_and_config_are_executable_5090_contracts(self) -> None:
        config = (
            controller.ROOT / "configs/v5_3_12h_screen.yaml"
        ).read_text(encoding="utf-8")
        handoff = (
            controller.ROOT / "V5_3_12H_SCREEN_HANDOFF.md"
        ).read_text(encoding="utf-8")
        prompt = (
            controller.ROOT
            / "V5_3_12H_RUNPOD_4X5090_AGENT_PROMPT.md"
        ).read_text(encoding="utf-8")
        self.assertIn("exact_gpu_model: NVIDIA GeForce RTX 5090", config)
        self.assertIn("minimum_free_disk_gib: 80", config)
        self.assertIn("pinned_model_cache_completion: 44", config)
        self.assertIn("experiment_artifacts_and_transients: 12", config)
        self.assertIn("post_run_safety_reserve: 24", config)
        self.assertIn("arm_selection: request_model_alias", config)
        self.assertIn(
            "core_launch_set_exact: [base_model, perfect_success, repair_50]",
            config,
        )
        self.assertIn(
            "extension_launch_set_exact: [failure_raw, repair_100]",
            config,
        )
        self.assertIn("task-shard parallelism", handoff)
        self.assertIn("shard-parallel", prompt)
        self.assertIn("80 GiB", handoff)
        self.assertIn("80 GiB", prompt)
        self.assertIn("--stage all", handoff)
        self.assertIn("--stage all", prompt)
        self.assertNotIn("TODO_", handoff + prompt)
        self.assertNotIn("RTX 4090", handoff + prompt)
        self.assertEqual(config.count("replacement_or_rescue_attempts:"), 1)

    def test_duplicate_yaml_key_is_rejected(self) -> None:
        with self.assertRaises(controller.ScreenStageError):
            controller.reject_duplicate_yaml_mapping_keys(
                "root:\n  value: 1\n  value: 2\n"
            )

    def test_resume_schedules_only_missing_arm_shard(self) -> None:
        arms = protocol.CORE_EVAL_ARMS
        with mock.patch.object(
            controller,
            "evaluation_shard_complete",
            side_effect=lambda _args, arm, shard: not (
                arm == "repair_50" and shard == 2
            ),
        ):
            self.assertEqual(
                controller.pending_evaluation_arms(
                    SimpleNamespace(), arms, 0
                ),
                [],
            )
            self.assertEqual(
                controller.pending_evaluation_arms(
                    SimpleNamespace(), arms, 2
                ),
                ["repair_50"],
            )

    def test_core_shard_gate_rejects_raw_hash_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_root = root / "protocol"
            results_root = root / "results"
            arm_dir = results_root / "evaluation/base_model"
            protocol_root.mkdir(parents=True)
            arm_dir.mkdir(parents=True)
            registry_path = results_root / "checkpoint_registry.json"
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text("{}\n", encoding="utf-8")
            rows = [
                {
                    "domain": "retail",
                    "task_id": str(index),
                    "pair_id": f"retail:{index}",
                }
                for index in range(21)
            ]
            validation_path = protocol_root / "validation_manifest.json"
            validation_path.write_text(
                json.dumps({"rows": rows}) + "\n", encoding="utf-8"
            )
            expected_rows = controller.evaluator.shard_rows(rows, 0, 3)
            expected_tasks = [
                f"retail:{row['task_id']}" for row in expected_rows
            ]
            result_name = "retail_clean.shard-000-of-003.json"
            result_path = arm_dir / result_name
            result_path.write_text('{"simulations":[]}\n', encoding="utf-8")
            registry = {"entries": {"base_model": {"fixture": True}}}
            contract = {
                "status": "COMPLETE",
                "arm": "base_model",
                "shard_index": 0,
                "num_shards": 3,
                "conditions": ["clean", "error"],
                "task_ids": expected_tasks,
                "checkpoint_registry_provenance_profile": (
                    protocol.REGISTRY_PROFILE
                ),
                "checkpoint_registry_sha256": controller.sha256_file(
                    registry_path
                ),
                "checkpoint_entry": registry["entries"]["base_model"],
                "evaluation_manifest_sha256": controller.sha256_file(
                    validation_path
                ),
                "split_manifest_sha256": controller.sha256_file(
                    controller.SPLIT
                ),
                "decoding": (
                    controller.evaluator.V5_3_12H_FROZEN_DECODING
                ),
                "official_test_used": False,
                "agent": {
                    "model": protocol.MODEL_IDS["base_model"]
                },
                "user": {
                    "model": protocol.USER_JUDGE_MODEL_ID,
                    "revision": protocol.USER_JUDGE_REVISION,
                    "api_base": "http://user",
                },
                "judge": {
                    "model": protocol.USER_JUDGE_MODEL_ID,
                    "revision": protocol.USER_JUDGE_REVISION,
                    "api_base": "http://user",
                },
                "result_sha256": {result_name: "0" * 64},
            }
            controller.eval_contract_path(
                SimpleNamespace(results_root=results_root),
                "base_model",
                0,
            ).write_text(json.dumps(contract) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                results_root=results_root,
                protocol_root=protocol_root,
            )
            with (
                mock.patch.object(
                    controller.evaluator,
                    "load_checkpoint_registry",
                    return_value=registry,
                ),
                mock.patch.object(
                    controller.evaluator,
                    "validate_contract_core",
                ),
                mock.patch.object(
                    controller.evaluator,
                    "_expected_result_task_ids",
                    return_value={result_name: [row["task_id"] for row in expected_rows]},
                ),
            ):
                with self.assertRaises(controller.ScreenStageError):
                    controller.evaluation_shard_evidence(
                        args, "base_model", 0
                    )


class ScreenManifestTests(unittest.TestCase):
    def fixtures(self):
        selected = [
            multifault_row(
                *identity.split(":", 1),
                source_split="derived_inner_train",
            )
            for identity in protocol.PILOT_TASK_IDS
        ]
        extras = [
            multifault_row(
                "retail",
                f"x{index}",
                source_split="derived_inner_train",
            )
            for index in range(54)
        ]
        generation = {
            "protocol": data.GENERATION_MANIFEST_PROTOCOL,
            "paired_task_count": 78,
            "official_test_used": False,
            "split_manifest_sha256": data.SPLIT_SHA256,
            "tau2_commit": "c" * 40,
            "source_files": SOURCE_FILES,
            "fault_protocol": fault_protocol(),
            "gt_compatibility_filter": {"protocol": "fixture"},
            "rows": [*selected, *extras],
        }
        generation_sha = "a" * 64
        universe = {
            "protocol": "v5_3_effective_arm_train_task_universe",
            "created_without_v5_3_outcomes": True,
            "source_split": "derived_inner_train",
            "official_test_used": False,
            "split_manifest_sha256": data.SPLIT_SHA256,
            "generation_manifest_sha256": generation_sha,
            "planned_task_ids": [
                *protocol.PILOT_TASK_IDS,
                *[f"retail:x{index}" for index in range(46)],
            ],
        }
        pilot = {
            "protocol": "v5_3_train_only_feasibility_pilot",
            "training": False,
            "formal_data": False,
            "created_without_v5_3_outcomes": True,
            "official_test_used": False,
            "split_manifest_sha256": data.SPLIT_SHA256,
            "generation_manifest_sha256": generation_sha,
            "base_seed": protocol.PILOT_BASE_SEED,
            "trial_seeds": list(protocol.PILOT_TRIAL_SEEDS),
            "selection": {
                "uses_rollout_or_validation_outcomes": False
            },
            "planned_task_ids": list(protocol.PILOT_TASK_IDS),
            "rows": selected,
        }
        return generation, generation_sha, universe, pilot

    def test_manifest_deep_binds_source_rows_and_provenance(self) -> None:
        generation, digest, universe, pilot = self.fixtures()
        screen = data.build_screen_manifest(
            effective_universe=universe,
            pilot_manifest=pilot,
            generation_manifest=generation,
            generation_manifest_sha256=digest,
        )
        rows = data.validate_screen_manifest(
            screen,
            generation_manifest_sha256=digest,
            generation_manifest=generation,
            effective_universe=universe,
            pilot_manifest=pilot,
        )
        self.assertEqual(len(rows), 24)
        self.assertEqual(screen["paired_task_count"], 24)
        self.assertEqual(screen["generation"]["num_trials"], 6)
        self.assertFalse(screen["replacement_or_rescue_attempts"])

        tampered = json.loads(json.dumps(screen))
        tampered["rows"][0]["error_condition"]["arguments"] = {
            "changed": "fault"
        }
        core = {
            key: value
            for key, value in tampered.items()
            if key != "canonical_sha256"
        }
        tampered["canonical_sha256"] = protocol.canonical_sha256(core)
        with self.assertRaises(data.ScreenDataError):
            data.validate_screen_manifest(
                tampered,
                generation_manifest_sha256=digest,
                generation_manifest=generation,
                effective_universe=universe,
                pilot_manifest=pilot,
            )


class ScreenPairingTests(unittest.TestCase):
    @staticmethod
    def candidate(index: int) -> dict:
        return {
            "identity": {
                "attempt_index": index,
                "attempt_seed": str(protocol.TRIAL_SEEDS[index]),
            }
        }

    def test_pairing_intersects_identical_attempt_slot(self) -> None:
        clean = [self.candidate(0), self.candidate(2), self.candidate(4)]
        error = [self.candidate(1), self.candidate(2), self.candidate(4)]
        pairs = data._common_eligible_attempt_slots(clean, error)
        self.assertEqual(
            [clean_row["identity"]["attempt_index"] for clean_row, _ in pairs],
            [2, 4],
        )
        self.assertTrue(
            all(
                clean_row["identity"] == error_row["identity"]
                for clean_row, error_row in pairs
            )
        )

    def test_disjoint_clean_error_attempt_slots_yield_zero_pairs(self) -> None:
        clean = [self.candidate(0), self.candidate(2), self.candidate(4)]
        error = [self.candidate(1), self.candidate(3), self.candidate(5)]
        self.assertEqual(
            data._common_eligible_attempt_slots(clean, error), []
        )
        gate = protocol.data_gate(
            {task_id: 0 for task_id in protocol.PILOT_TASK_IDS}
        )
        self.assertFalse(gate["training_authorized"])
        self.assertEqual(gate["observed"]["capped_pairs"], 0)


class ScreenStatisticsTests(unittest.TestCase):
    def loaded(self, arm: str, *, treatment: bool = False):
        cases = {}
        for index in range(21):
            domain = "retail" if index < 15 else "airline"
            task_id = str(index)
            control_error = int(index < 5)
            treatment_error = int(index < 7)
            control_clean = int(index < 10)
            treatment_clean = int(index < 9)
            for condition, control_value, treatment_value in (
                ("error", control_error, treatment_error),
                ("clean", control_clean, treatment_clean),
            ):
                cases[(domain, task_id, condition, 0)] = {
                    "success": (
                        treatment_value if treatment else control_value
                    ),
                    "reward": float(
                        treatment_value if treatment else control_value
                    ),
                    "repeated_failed_call": 0,
                }
        return {"arm": arm, "cases": cases}

    def test_exact_paired_counts_bootstrap_and_claim_gate(self) -> None:
        with mock.patch.object(protocol, "TASK_BOOTSTRAP_REPLICATES", 1000):
            result = summary.compare(
                self.loaded("repair_50", treatment=True),
                self.loaded("perfect_success"),
            )
        error = result["conditions"]["error"]
        clean = result["conditions"]["clean"]
        self.assertEqual(error["n10_treatment_win_b"], 2)
        self.assertEqual(error["n01_control_win_c"], 0)
        self.assertAlmostEqual(error["paired_delta"], 2 / 21)
        self.assertEqual(clean["n10_treatment_win_b"], 0)
        self.assertEqual(clean["n01_control_win_c"], 1)
        self.assertAlmostEqual(clean["paired_delta"], -1 / 21)
        self.assertAlmostEqual(result["difference_in_differences"], 3 / 21)
        self.assertIsNone(
            clean["mcnemar_binomial"][
                "exact_one_sided_treatment_greater_p"
            ]
        )

    def test_missing_raw_case_fails_instead_of_being_counted(self) -> None:
        treatment = self.loaded("repair_50", treatment=True)
        treatment["cases"].pop(next(iter(treatment["cases"])))
        with self.assertRaises(summary.SummaryError):
            summary.compare(treatment, self.loaded("perfect_success"))

    def test_screen_training_audit_requires_zero_validation_metrics(self) -> None:
        audited = train.finite_training_audit(
            [{"loss": 1.0, "grad_norm": 0.5}],
            {"train_loss": 0.8},
            require_validation_loss=False,
        )
        self.assertEqual(audited["validation_loss_values_checked"], 0)
        self.assertIsNone(audited["final_validation_loss"])
        with self.assertRaises(RuntimeError):
            train.finite_training_audit(
                [{"loss": 1.0, "grad_norm": 0.5, "eval_loss": 0.7}],
                {"train_loss": 0.8},
                require_validation_loss=False,
            )


if __name__ == "__main__":
    unittest.main()
