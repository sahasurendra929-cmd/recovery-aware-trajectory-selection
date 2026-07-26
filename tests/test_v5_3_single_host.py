import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v5_3_single_host.py"
SPEC = importlib.util.spec_from_file_location("run_v5_3_single_host", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
PREP_SCRIPT = ROOT / "scripts" / "prepare_v5_3_sft_causal.py"
PREP_SPEC = importlib.util.spec_from_file_location(
    "prepare_v5_3_sft_causal_for_controller_test", PREP_SCRIPT
)
PREP_MODULE = importlib.util.module_from_spec(PREP_SPEC)
assert PREP_SPEC.loader is not None
PREP_SPEC.loader.exec_module(PREP_MODULE)


def args_for(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        vllm=Path("/workspace/venvs/v5-2-serve/bin/vllm"),
        serve_python=Path("/workspace/venvs/v5-2-serve/bin/python"),
        train_python=Path("/workspace/venvs/v5-2-train/bin/python"),
        tau2_root=root / "tau2",
        protocol_root=root / "protocol",
        pilot_root=root / "pilot",
        raw_root=root / "raw",
        processed_root=root / "processed",
        results_root=root / "results",
        health_timeout=30,
    )


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def runtime_gpu_fixture() -> list[dict]:
    return [
        {
            "index": index,
            "uuid": f"GPU-runtime-{index}",
            "name": "NVIDIA GeForce RTX 4090",
            "memory_mib": 24564,
            "driver": "575.57",
        }
        for index in range(4)
    ]


def snapshot_fixture(root: Path, model: str, revision: str) -> dict:
    snapshot = root / model.replace("/", "--") / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model": model, "revision": revision}),
        encoding="utf-8",
    )
    return {
        "model": model,
        "requested_revision": revision,
        "resolved_revision": revision,
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_tree_sha256": MODULE._snapshot_tree_sha256(snapshot),
    }


def probe_evidence(request_id: str) -> dict:
    prompt_tokens = 28_123
    completion_tokens = 23
    return {
        "status": "PASS",
        "request_id": request_id,
        "endpoint": "chat/completions",
        "parallel_tool_calls": False,
        "max_tokens": 512,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "tool_call_count": 1,
        "tool_name": MODULE.RUNTIME_PREFLIGHT_TOOL_NAME,
        "tool_arguments_sha256": MODULE.canonical_sha256(
            {
                "marker": MODULE.RUNTIME_PREFLIGHT_MARKER,
                "request_id": request_id,
            }
        ),
        "response_sha256": MODULE.canonical_sha256(
            {"request_id": request_id}
        ),
    }


def valid_chat_tool_response(request_id: str) -> dict:
    return {
        "id": f"chatcmpl-{request_id}",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{request_id}",
                            "type": "function",
                            "function": {
                                "name": MODULE.RUNTIME_PREFLIGHT_TOOL_NAME,
                                "arguments": json.dumps(
                                    {
                                        "marker": (
                                            MODULE.RUNTIME_PREFLIGHT_MARKER
                                        ),
                                        "request_id": request_id,
                                    }
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 28_123,
            "completion_tokens": 23,
            "total_tokens": 28_146,
        },
    }


def runtime_service_contexts(
    args: SimpleNamespace, root: Path
) -> dict[str, dict]:
    contexts = {}
    for offset, specification in enumerate(
        MODULE._generation_service_specifications()
    ):
        log = (
            args.results_root
            / "logs"
            / f"{specification['name']}.log"
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("vLLM ready\n", encoding="utf-8")
        process = mock.Mock()
        process.pid = 41_000 + offset
        process.poll.return_value = None
        handle = mock.Mock()
        contexts[specification["role"]] = {
            "role": specification["role"],
            "process": process,
            "handle": handle,
            "command_sha256": MODULE.canonical_sha256(
                MODULE._generation_service_command(args, specification)
            ),
            "log_path": str(log.resolve()),
            "log_start_byte": 0,
            "model_snapshot": snapshot_fixture(
                root,
                specification["model"],
                specification["revision"],
            ),
        }
    return contexts


def build_valid_data(args: SimpleNamespace) -> dict:
    dynamic = {
        "generation": {
            "protocol": "v5_stage1_dynamic_injection_audit",
            "source_split": "derived_inner_train",
            "verified_injections": 78,
            "official_test_used": False,
            "official_test_sealed": True,
            "sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "split_manifest_sha256": MODULE.SPLIT_SHA256,
        },
        "validation": {
            "protocol": "v5_stage1_dynamic_injection_audit",
            "source_split": "derived_validation",
            "verified_injections": 21,
            "official_test_used": False,
            "official_test_sealed": True,
            "sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "split_manifest_sha256": MODULE.SPLIT_SHA256,
        },
    }
    design = {
        "design_version": "5.3",
        "design_protocol": "v5_3_task_level_cross_seed_sft_screen",
        "effective_generation_tasks": 78,
        "validation_tasks": 21,
    }
    audit = {
        "status": "PASS",
        "design_protocol": "v5_3_task_level_cross_seed_sft_screen",
        "design_version": "5.3",
        "attempts_per_task_per_condition": 12,
        "official_test_used": False,
        "derived_validation_used_for_supervision": False,
        "ground_truth_incompatible_task_ids": list(
            MODULE.GT_INCOMPATIBLE_TASK_IDS
        ),
        "generation_contracts": {"task_union": 78},
        "cross_arm_supervised_token_relative_range": 0.0,
        "cross_arm_nonpadding_token_relative_range": 0.0,
        "formal_data_gate": {
            "observed_distinct_train_tasks": 40,
            "observed_train_pairs": 48,
        },
        "dynamic_audits": dynamic,
    }
    write_json(args.processed_root / "audit.json", audit)
    for arm in MODULE.ARMS:
        path = args.processed_root / "arms" / arm / "train.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"arm": arm}) + "\n", encoding="utf-8")
    (args.processed_root / "validation_loss.jsonl").write_text(
        json.dumps({"split": "validation"}) + "\n", encoding="utf-8"
    )
    write_json(
        args.processed_root / "validation_manifest.json",
        {
            "protocol": "v5_stage1_sft_causal_validation",
            "rows": [{"domain": "retail", "task_id": "0"}],
        },
    )
    hashes = {
        "audit.json": MODULE.sha256_file(args.processed_root / "audit.json"),
        "validation_loss.jsonl": MODULE.sha256_file(
            args.processed_root / "validation_loss.jsonl"
        ),
        "validation_manifest.json": MODULE.sha256_file(
            args.processed_root / "validation_manifest.json"
        ),
    }
    for arm in MODULE.ARMS:
        name = f"arms/{arm}/train.jsonl"
        hashes[name] = MODULE.sha256_file(args.processed_root / name)
    write_json(args.processed_root / "hashes.json", hashes)
    return {"dynamic": dynamic, "design": design}


def build_valid_training(
    args: SimpleNamespace,
    *,
    arm: str,
    mode: str = "formal",
    data_identity: dict,
) -> Path:
    output = args.results_root / arm / mode
    checkpoint = output / "checkpoint_final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(
        f"adapter-{arm}-{mode}".encode()
    )
    write_json(checkpoint / "adapter_config.json", {"peft_type": "LORA"})
    history = [
        {"loss": 0.8, "grad_norm": 1.2},
        {"eval_loss": 0.4},
    ]
    metrics = {"train_loss": 0.6}
    write_json(output / "training_log.json", history)
    write_json(output / "training_metrics.json", metrics)
    fingerprint, checkpoint_files = MODULE._checkpoint_fingerprint(checkpoint)
    hashes = json.loads(
        (args.processed_root / "hashes.json").read_text(encoding="utf-8")
    )
    provenance = {
        "data_audit_sha256": MODULE.sha256_file(
            args.processed_root / "audit.json"
        ),
        "data_hashes_sha256": MODULE.sha256_file(
            args.processed_root / "hashes.json"
        ),
        "train_file_sha256": hashes[f"arms/{arm}/train.jsonl"],
        "validation_file_sha256": hashes["validation_loss.jsonl"],
        "dynamic_audits": data_identity["dynamic"],
        "design_version": "5.3",
        "design_provenance": data_identity["design"],
        "official_test_used": False,
        "official_test_sealed": True,
    }
    manifest = {
        "protocol": "v5_stage1_message_masked_sft_7b",
        "source_commit": MODULE.source_commit(),
        "arm": arm,
        "mode": mode,
        "objective": "message_masked_causal_language_model_cross_entropy",
        "model": MODULE.STUDENT_MODEL,
        "model_revision": MODULE.STUDENT_REVISION,
        "seed": MODULE.SEED,
        "max_sequence_tokens": 8192,
        "truncation": False,
        "held_out_test_accessed": False,
        "train_file_sha256": provenance["train_file_sha256"],
        "validation_file_sha256": provenance["validation_file_sha256"],
        "data_provenance": provenance,
        "checkpoint": {
            "fingerprint": fingerprint,
            "file_sha256": checkpoint_files,
        },
        "loss_audit": {
            "finite": True,
            "numeric_values_checked": 4,
            "loss_values_checked": 3,
            "grad_norm_values_checked": 1,
            "validation_loss_values_checked": 1,
            "final_train_loss": 0.6,
            "final_validation_loss": 0.4,
        },
    }
    write_json(output / "run_manifest.json", manifest)
    return output


def build_valid_registry(
    args: SimpleNamespace, data_identity: dict
) -> dict:
    provenance = None
    entries = {
        "base_model": {
            "model_id": MODULE.EVAL_MODEL_IDS["base_model"],
            "adapter_sha256": None,
            "adapter_config_sha256": None,
            "training_run_manifest_sha256": None,
        }
    }
    for arm in MODULE.ARMS:
        run = build_valid_training(
            args, arm=arm, data_identity=data_identity
        )
        manifest = json.loads(
            (run / "run_manifest.json").read_text(encoding="utf-8")
        )
        source = manifest["data_provenance"]
        provenance = {
            key: source[key]
            for key in (
                "data_audit_sha256",
                "data_hashes_sha256",
                "dynamic_audits",
                "design_version",
                "design_provenance",
                "official_test_used",
                "official_test_sealed",
            )
        }
        entries[arm] = {
            "model_id": MODULE.EVAL_MODEL_IDS[arm],
            "adapter_sha256": MODULE.sha256_file(
                run / "checkpoint_final/adapter_model.safetensors"
            ),
            "adapter_config_sha256": MODULE.sha256_file(
                run / "checkpoint_final/adapter_config.json"
            ),
            "training_run_manifest_sha256": MODULE.sha256_file(
                run / "run_manifest.json"
            ),
        }
    registry = {
        "protocol": "v5_stage1_checkpoint_registry",
        "provenance_profile": "v5_3",
        "source_commit": MODULE.source_commit(),
        "base_model_revision": MODULE.STUDENT_REVISION,
        "training_data_provenance": provenance,
        "entries": entries,
    }
    write_json(args.results_root / "checkpoint_registry.json", registry)
    return registry


def build_valid_evaluation_shard(
    args: SimpleNamespace,
    *,
    registry: dict,
    arm: str = "base_model",
    shard: int = 0,
) -> Path:
    try:
        from scripts.run_v5_sft_causal_eval import (
            _expected_result_task_ids,
            audit_result_interface,
        )
    except ModuleNotFoundError:
        from run_v5_sft_causal_eval import (
            _expected_result_task_ids,
            audit_result_interface,
        )

    output = args.results_root / "evaluation" / arm
    output.mkdir(parents=True)
    registry_path = args.results_root / "checkpoint_registry.json"
    manifest_path = args.processed_root / "validation_manifest.json"
    endpoint = f"http://127.0.0.1:{8100 + shard}/v1"
    contract = {
        "protocol": "v5_stage1_sft_causal_validation_run",
        "status": "INCOMPLETE",
        "arm": arm,
        "evaluation_manifest_protocol": (
            "v5_stage1_sft_causal_validation"
        ),
        "evaluation_manifest_sha256": MODULE.sha256_file(manifest_path),
        "split_manifest_sha256": MODULE.SPLIT_SHA256,
        "dynamic_audit_identity": registry[
            "training_data_provenance"
        ]["dynamic_audits"]["validation"],
        "checkpoint_registry_sha256": MODULE.sha256_file(registry_path),
        "checkpoint_registry_protocol": "v5_stage1_checkpoint_registry",
        "checkpoint_registry_provenance_profile": "v5_3",
        "checkpoint_entry": registry["entries"][arm],
        "locally_verified_adapter_identity": {
            trained_arm: {
                "adapter_sha256": registry["entries"][trained_arm][
                    "adapter_sha256"
                ],
                "adapter_config_sha256": registry["entries"][trained_arm][
                    "adapter_config_sha256"
                ],
            }
            for trained_arm in MODULE.ARMS
        },
        "served_registry_aliases": sorted(
            entry["model_id"].removeprefix("openai/")
            for entry in registry["entries"].values()
        ),
        "source_commit": MODULE.source_commit(),
        "base_model_revision": MODULE.STUDENT_REVISION,
        "task_ids": ["retail:0"],
        "shard_index": shard,
        "num_shards": 4,
        "agent": {
            "model": MODULE.EVAL_MODEL_IDS[arm],
            "revision": MODULE.STUDENT_REVISION,
            "api_base": endpoint,
        },
        "user": {
            "model": MODULE.EVAL_MODEL_IDS["base_model"],
            "revision": MODULE.STUDENT_REVISION,
            "api_base": endpoint,
        },
        "judge": {
            "model": MODULE.EVAL_MODEL_IDS["base_model"],
            "revision": MODULE.STUDENT_REVISION,
            "api_base": endpoint,
            "strict_backend": {
                "module": "v5_strict_nl_judge",
                "entrypoint": "install_strict_nl_judge",
                "mode": "strict_json_schema_fail_closed",
            },
        },
        "decoding": {
            "temperature": 0,
            "max_tokens": 512,
            "max_steps": 60,
            "task_timeout_seconds": 900.0,
            "seed": MODULE.SEED,
            "num_trials": 1,
        },
        "tool_action_interface": MODULE.EVAL_TOOL_ACTION_INTERFACE,
        "runtime_preflight": {
            "served_context_window_tokens": MODULE.MAX_MODEL_LEN,
            "request_max_tokens": 512,
            "maximum_nonoverflow_prompt_tokens": (
                MODULE.MAX_MODEL_LEN - 512
            ),
            "max_steps": 60,
            "request_token_overflow_policy": "fail_closed",
            "longest_prompt_observation": (
                "completion_audit.max_observed_prompt_tokens"
            ),
        },
        "conditions": ["clean", "error"],
        "official_test_used": False,
        "result_sha256": {},
        "completion_audit": {},
        "strict_judge_audit_evidence": {},
    }
    expected_results = _expected_result_task_ids(contract)
    result_hashes = {}
    result_audits = {}
    for name, task_ids in expected_results.items():
        result = output / name
        write_json(
            result,
            {
                "simulations": [
                    {
                        "task_id": task_id,
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "done",
                                "tool_calls": [],
                                "usage": {"prompt_tokens": 10},
                            }
                        ],
                    }
                    for task_id in task_ids
                ]
            },
        )
        result_hashes[name] = MODULE.sha256_file(result)
        result_audits[name] = audit_result_interface(
            result,
            expected_task_ids=task_ids,
            num_trials=1,
            max_tokens=512,
        )
    contract["result_sha256"] = dict(sorted(result_hashes.items()))
    contract["completion_audit"] = {
        "protocol": "v5_stage1_sft_causal_interface_completion_audit",
        "tool_action_interface": MODULE.EVAL_TOOL_ACTION_INTERFACE,
        "result_files": dict(sorted(result_audits.items())),
        "max_observed_prompt_tokens": 10,
        "max_observed_total_request_tokens": 522,
        "all_expected_tasks_observed_once_per_trial": True,
        "request_token_overflow_detected": False,
        "status": "PASS",
    }
    contract["strict_judge_audit_evidence"] = (
        MODULE.judge_audit_contract.validate_strict_judge_evidence(
            [output / name for name in sorted(result_hashes)],
            maximum_content_attempts=2,
        )
    )
    contract["status"] = "COMPLETE"
    contract["contract_core_sha256"] = (
        MODULE._evaluation_contract_core_sha256(contract)
    )
    path = output / f"run_contract.shard-{shard:03d}-of-004.json"
    write_json(path, contract)
    return path


class V53SingleHostTests(unittest.TestCase):
    def test_vllm_command_freezes_32768_and_one_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            command = MODULE.vllm_base_command(
                args,
                MODULE.STUDENT_MODEL,
                MODULE.STUDENT_REVISION,
                dtype="bfloat16",
            )
        self.assertEqual(command.count("--max-model-len"), 1)
        index = command.index("--max-model-len")
        self.assertEqual(command[index + 1], "32768")
        self.assertEqual(command.count("--dtype"), 1)
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")

    def test_chat_tool_probe_uses_frozen_payload_and_rejects_bad_calls(self):
        request_id = "pilot-teacher-long"
        with mock.patch.object(
            MODULE,
            "http_post_json",
            return_value=valid_chat_tool_response(request_id),
        ) as post:
            evidence = MODULE._chat_tool_probe(
                api_base="http://127.0.0.1:8011/v1",
                api_key="local",
                model=MODULE.TEACHER_MODEL,
                prompt="x " * 28_000,
                request_id=request_id,
            )
        self.assertEqual(evidence["status"], "PASS")
        url, api_key, payload = post.call_args.args
        self.assertEqual(url, "http://127.0.0.1:8011/v1/chat/completions")
        self.assertEqual(api_key, "local")
        self.assertEqual(payload["max_tokens"], 512)
        self.assertIs(payload["parallel_tool_calls"], False)
        self.assertEqual(
            payload["tool_choice"]["function"]["name"],
            MODULE.RUNTIME_PREFLIGHT_TOOL_NAME,
        )
        self.assertEqual(len(payload["tools"]), 1)

        def variants():
            value = valid_chat_tool_response(request_id)
            value["choices"][0]["message"]["content"] = "hidden text"
            yield "mixed_content", value
            value = valid_chat_tool_response(request_id)
            value["choices"][0]["finish_reason"] = "stop"
            yield "wrong_finish", value
            value = valid_chat_tool_response(request_id)
            value["choices"][0]["message"]["tool_calls"][0]["id"] = ""
            yield "empty_call_id", value
            value = valid_chat_tool_response(request_id)
            value["choices"][0]["message"]["tool_calls"][0]["function"][
                "arguments"
            ] = json.dumps({"marker": "wrong", "request_id": request_id})
            yield "wrong_arguments", value
            value = valid_chat_tool_response(request_id)
            value.pop("usage")
            yield "missing_usage", value

        for label, response in variants():
            with (
                self.subTest(label=label),
                mock.patch.object(
                    MODULE, "http_post_json", return_value=response
                ),
                self.assertRaises(MODULE.StageError),
            ):
                MODULE._chat_tool_probe(
                    api_base="http://127.0.0.1:8011/v1",
                    api_key="local",
                    model=MODULE.TEACHER_MODEL,
                    prompt="x " * 28_000,
                    request_id=request_id,
                )

    def test_runtime_preflight_replaces_v1_and_binds_live_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            write_json(args.results_root / "preflight.json", {"status": "PASS"})
            path = MODULE.generation_runtime_preflight_path(
                args, phase="pilot"
            )
            path.parent.mkdir(parents=True)
            write_json(
                path,
                {
                    "protocol": "v5_3_generation_runtime_preflight_v1",
                    "status": "PASS",
                },
            )
            gpu_inventory = runtime_gpu_fixture()
            contexts = runtime_service_contexts(args, root / "snapshots")
            active = 0
            maximum_active = 0
            lock = threading.Lock()
            request_ids = []

            def fake_probe(**kwargs):
                nonlocal active, maximum_active
                request_ids.append(kwargs["request_id"])
                if "-concurrent-" in kwargs["request_id"]:
                    with lock:
                        active += 1
                        maximum_active = max(maximum_active, active)
                    time.sleep(0.04)
                    with lock:
                        active -= 1
                return probe_evidence(kwargs["request_id"])

            def models(url, _api_key):
                model = (
                    MODULE.TEACHER_MODEL
                    if ":8011/" in url
                    else MODULE.USER_MODEL
                )
                return {"data": [{"id": model}]}

            with (
                mock.patch.object(MODULE, "preflight_complete", return_value=True),
                mock.patch.object(
                    MODULE,
                    "_runtime_gpu_identity",
                    return_value=gpu_inventory,
                ),
                mock.patch.object(MODULE, "http_json", side_effect=models),
                mock.patch.object(
                    MODULE, "_chat_tool_probe", side_effect=fake_probe
                ),
            ):
                MODULE.run_generation_runtime_preflight(
                    args,
                    phase="pilot",
                    service_contexts=contexts,
                    gpu_inventory=gpu_inventory,
                    invocation_nonce="a" * 32,
                    invocation_started_unix_ns=1_725_000_000_000_000_000,
                )
                self.assertTrue(
                    MODULE.generation_runtime_preflight_complete(
                        args, phase="pilot"
                    )
                )
            self.assertEqual(len(request_ids), 8)
            self.assertEqual(maximum_active, 3)
            self.assertEqual(
                len(list((path.parent / "archive").glob("*.json"))), 1
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["protocol"], MODULE.RUNTIME_PREFLIGHT_PROTOCOL
            )
            self.assertEqual(
                payload["receipt_pointer"]["canonical_receipt_sha256"],
                payload["canonical_receipt_sha256"],
            )
            self.assertEqual(
                set(payload["invocation"]["services"]),
                {"teacher", "user_and_judge"},
            )
            with mock.patch.object(
                MODULE,
                "_runtime_gpu_identity",
                return_value=gpu_inventory,
            ):
                host_preflight = args.results_root / "preflight.json"
                host_bytes = host_preflight.read_bytes()
                host_preflight.write_text(
                    '{"status":"changed"}', encoding="utf-8"
                )
                self.assertFalse(
                    MODULE.generation_runtime_preflight_complete(
                        args, phase="pilot"
                    )
                )
                host_preflight.write_bytes(host_bytes)
                original_vllm = args.vllm
                args.vllm = Path("/different/vllm")
                self.assertFalse(
                    MODULE.generation_runtime_preflight_complete(
                        args, phase="pilot"
                    )
                )
                args.vllm = original_vllm
                snapshot_path = Path(
                    payload["invocation"]["services"]["teacher"][
                        "model_snapshot"
                    ]["snapshot_path"]
                )
                snapshot_file = snapshot_path / "config.json"
                snapshot_bytes = snapshot_file.read_bytes()
                snapshot_file.write_text("drift", encoding="utf-8")
                self.assertFalse(
                    MODULE.generation_runtime_preflight_complete(
                        args, phase="pilot"
                    )
                )
                snapshot_file.write_bytes(snapshot_bytes)
                log_path = Path(
                    payload["invocation"]["services"]["teacher"][
                        "log_evidence"
                    ]["path"]
                )
                log_bytes = log_path.read_bytes()
                log_path.write_text("modified log\n", encoding="utf-8")
                self.assertFalse(
                    MODULE.generation_runtime_preflight_complete(
                        args, phase="pilot"
                    )
                )
                log_path.write_bytes(log_bytes)
                self.assertTrue(
                    MODULE.generation_runtime_preflight_complete(
                        args, phase="pilot"
                    )
                )
            changed_gpu = json.loads(json.dumps(gpu_inventory))
            changed_gpu[0]["uuid"] = "GPU-different-host"
            with mock.patch.object(
                MODULE,
                "_runtime_gpu_identity",
                return_value=changed_gpu,
            ):
                self.assertFalse(
                    MODULE.generation_runtime_preflight_complete(
                        args, phase="pilot"
                    )
                )
            pilot_contracts = [
                MODULE.pilot_contract_path(args, shard)
                for shard in range(MODULE.GENERATION_SHARDS)
            ]
            for shard, contract in enumerate(pilot_contracts):
                write_json(contract, {"shard": shard, "status": "COMPLETE"})
            with mock.patch.object(
                MODULE,
                "_runtime_gpu_identity",
                return_value=gpu_inventory,
            ):
                MODULE.write_generation_runtime_contract_binding(
                    args, phase="pilot"
                )
                self.assertTrue(
                    MODULE.generation_runtime_contract_binding_complete(
                        args, phase="pilot"
                    )
                )
                write_json(
                    pilot_contracts[0],
                    {"shard": 0, "status": "DRIFT"},
                )
                self.assertFalse(
                    MODULE.generation_runtime_contract_binding_complete(
                        args, phase="pilot"
                    )
                )

            malformed = json.loads(json.dumps(payload))
            malformed["roles"]["teacher"]["concurrency_probe"] = None
            malformed.pop("canonical_receipt_sha256")
            malformed.pop("receipt_pointer")
            malformed["canonical_receipt_sha256"] = MODULE.canonical_sha256(
                malformed
            )
            malformed["receipt_pointer"] = MODULE._runtime_receipt_pointer(
                path, malformed
            )
            write_json(path, malformed)
            with mock.patch.object(
                MODULE,
                "_runtime_gpu_identity",
                return_value=gpu_inventory,
            ):
                self.assertFalse(
                    MODULE.generation_runtime_preflight_complete(
                        args, phase="pilot"
                    )
                )

    def test_start_generation_services_never_skips_live_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            write_json(
                MODULE.generation_runtime_preflight_path(args, phase="pilot"),
                {"protocol": MODULE.RUNTIME_PREFLIGHT_PROTOCOL},
            )
            processes = []

            def spawn(_command, *, env, log_path):
                process = mock.Mock()
                process.pid = 50_000 + len(processes)
                process.poll.return_value = None
                processes.append(process)
                return process, mock.Mock()

            with (
                mock.patch.object(
                    MODULE,
                    "_runtime_gpu_identity",
                    return_value=runtime_gpu_fixture(),
                ),
                mock.patch.object(
                    MODULE,
                    "_resolve_model_snapshot",
                    return_value={"resolved": True},
                ),
                mock.patch.object(MODULE, "port_is_free", return_value=True),
                mock.patch.object(MODULE, "spawn_logged", side_effect=spawn),
                mock.patch.object(MODULE, "write_pid"),
                mock.patch.object(MODULE, "wait_for_service"),
                mock.patch.object(
                    MODULE, "run_generation_runtime_preflight"
                ) as live,
                mock.patch.object(
                    MODULE,
                    "generation_runtime_preflight_complete",
                    return_value=True,
                ) as stale_complete,
            ):
                services = MODULE.start_generation_services(
                    args, phase="pilot"
                )
            self.assertEqual(len(services), 2)
            live.assert_called_once()
            stale_complete.assert_not_called()

    def test_pilot_complete_shards_refresh_missing_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            decision = MODULE.pilot_files(args)["decision"]
            write_json(decision, {"existing": "immutable decision"})
            with (
                mock.patch.object(MODULE, "preflight_complete", return_value=True),
                mock.patch.object(
                    MODULE,
                    "pilot_status",
                    side_effect=[None, "GO_FORMAL_GENERATION"],
                ),
                mock.patch.object(MODULE, "build_pilot_manifest"),
                mock.patch.object(
                    MODULE,
                    "_validate_pilot_shard",
                    return_value=(set(), set()),
                ),
                mock.patch.object(
                    MODULE,
                    "generation_runtime_preflight_complete",
                    return_value=False,
                ),
                mock.patch.object(
                    MODULE, "start_generation_services", return_value=[]
                ) as start,
                mock.patch.object(
                    MODULE, "pilot_generation_complete", return_value=True
                ),
                mock.patch.object(
                    MODULE, "write_generation_runtime_contract_binding"
                ),
                mock.patch.object(
                    MODULE, "_pilot_bound_inputs", return_value=([], [])
                ),
            ):
                status = MODULE.run_pilot(args)
            self.assertEqual(status, "GO_FORMAL_GENERATION")
            start.assert_called_once_with(args, phase="pilot")

    def test_generation_completion_requires_three_shards_and_twelve_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            args.raw_root.mkdir()
            args.protocol_root.mkdir()
            task_ids = [f"retail:{200 + index}" for index in range(78)]
            generation_manifest = {
                "protocol": "v5_3_multifault_data_construction",
                "paired_task_count": 78,
                "rows": [
                    {"domain": "retail", "task_id": task_id.split(":")[1]}
                    for task_id in task_ids
                ],
            }
            validation_manifest = {
                "protocol": "v5_stage1_sft_causal_validation",
                "paired_task_count": 21,
                "rows": [
                    {"domain": "retail", "task_id": str(index)}
                    for index in range(21)
                ],
            }
            generation_manifest_path = (
                args.protocol_root / "generation_manifest.json"
            )
            validation_manifest_path = (
                args.protocol_root / "validation_manifest.json"
            )
            generation_manifest_path.write_text(
                json.dumps(generation_manifest), encoding="utf-8"
            )
            validation_manifest_path.write_text(
                json.dumps(validation_manifest), encoding="utf-8"
            )
            (args.protocol_root / "generation_dynamic_audit.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest_protocol": (
                            "v5_3_multifault_data_construction"
                        ),
                        "manifest_sha256": MODULE.sha256_file(
                            generation_manifest_path
                        ),
                        "official_test_used": False,
                        "verified_injections": 78,
                    }
                ),
                encoding="utf-8",
            )
            (args.protocol_root / "validation_dynamic_audit.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "manifest_protocol": (
                            "v5_stage1_sft_causal_validation"
                        ),
                        "manifest_sha256": MODULE.sha256_file(
                            validation_manifest_path
                        ),
                        "official_test_used": False,
                        "verified_injections": 21,
                    }
                ),
                encoding="utf-8",
            )
            for shard in range(3):
                shard_tasks = task_ids[shard::3]
                result_hashes = {}
                for condition in ("clean", "error"):
                    result_name = (
                        f"retail_{condition}.shard-{shard:03d}-of-003.json"
                    )
                    result_path = args.raw_root / result_name
                    simulations = [
                        {
                            "task_id": task_id.split(":")[1],
                            "trial": trial,
                            "seed": MODULE.DERIVED_TRIAL_SEEDS[trial],
                        }
                        for task_id in shard_tasks
                        for trial in range(12)
                    ]
                    result_path.write_text(
                        json.dumps({"simulations": simulations}),
                        encoding="utf-8",
                    )
                    result_hashes[result_name] = MODULE.sha256_file(
                        result_path
                    )
                payload = {
                    "protocol": "v5_stage1_inner_train_generation_run",
                    "status": "COMPLETE",
                    "shard_index": shard,
                    "num_shards": 3,
                    "num_trials": 12,
                    "official_test_used": False,
                    "source_split": "derived_inner_train",
                    "generation_manifest_protocol": (
                        "v5_3_multifault_data_construction"
                    ),
                    "gt_compatibility_filter": MODULE.GT_FILTER_CONTRACT,
                    "manifest_sha256": MODULE.sha256_file(
                        generation_manifest_path
                    ),
                    "split_manifest_sha256": MODULE.SPLIT_SHA256,
                    "source_commit": MODULE.source_commit(),
                    "sampling_contract": {
                        "protocol": "v5_3_frozen_stochastic_attempts_v1",
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "base_seed": MODULE.SEED,
                        "num_trials": 12,
                        "derived_trial_seeds": list(
                            MODULE.DERIVED_TRIAL_SEEDS
                        ),
                        "derived_trial_seeds_are_distinct": True,
                        "judge_temperature": 0.0,
                        "judge_top_p": 1.0,
                    },
                    "teacher": {
                        "model": MODULE.TEACHER_MODEL,
                        "revision": MODULE.TEACHER_REVISION,
                        "api_base": "http://127.0.0.1:8011/v1",
                        "mode": "ground_truth",
                    },
                    "user": {
                        "model": MODULE.USER_MODEL,
                        "revision": MODULE.USER_REVISION,
                        "api_base": "http://127.0.0.1:8001/v1",
                    },
                    "judge": {
                        "model": MODULE.USER_MODEL,
                        "revision": MODULE.USER_REVISION,
                        "api_base": "http://127.0.0.1:8001/v1",
                        "protocol": "v5_strict_nl_judge_v1",
                        "content_attempts": 2,
                        "schema_failure": "fail_closed",
                        "raw_response_audit": True,
                    },
                    "decoding": {
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "max_tokens": 512,
                        "parallel_tool_calls": False,
                        "parallel_tool_call_normalization": (
                            "execute_first_then_replan"
                        ),
                        "mixed_tool_call_content_normalization": (
                            "drop_text_preserve_sha256"
                        ),
                        "max_steps": 60,
                        "task_timeout_seconds": 900.0,
                        "seed": MODULE.SEED,
                        "derived_trial_seeds": list(
                            MODULE.DERIVED_TRIAL_SEEDS
                        ),
                    },
                    "gt_compatibility_preflight": {
                        "status": "PASS",
                        "filter_verified": True,
                        "checked_task_count": 83,
                        "compatible_task_count": 78,
                        "incompatible_task_ids": list(
                            MODULE.GT_INCOMPATIBLE_TASK_IDS
                        ),
                    },
                    "task_ids": shard_tasks,
                    "result_sha256": result_hashes,
                }
                payload["strict_judge_audit_evidence"] = (
                    MODULE.judge_audit_contract.validate_strict_judge_evidence(
                        [
                            args.raw_root / name
                            for name in sorted(result_hashes)
                        ],
                        maximum_content_attempts=2,
                    )
                )
                path = (
                    args.raw_root
                    / f"run_contract.shard-{shard:03d}-of-003.json"
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
            with (
                mock.patch.object(
                    MODULE,
                    "pilot_status",
                    return_value="GO_FORMAL_GENERATION",
                ),
                mock.patch.object(
                    MODULE,
                    "generation_runtime_preflight_complete",
                    return_value=True,
                ),
                mock.patch.object(
                    MODULE,
                    "generation_runtime_contract_binding_complete",
                    return_value=True,
                ),
            ):
                self.assertTrue(MODULE.generation_complete(args))
                path.write_text(
                    json.dumps({**payload, "num_trials": 3}),
                    encoding="utf-8",
                )
                self.assertFalse(MODULE.generation_complete(args))
                corrupt_name = "retail_clean.shard-002-of-003.json"
                corrupt_path = args.raw_root / corrupt_name
                corrupt = json.loads(
                    corrupt_path.read_text(encoding="utf-8")
                )
                corrupt["simulations"].pop()
                corrupt_path.write_text(
                    json.dumps(corrupt), encoding="utf-8"
                )
                payload["result_sha256"][corrupt_name] = MODULE.sha256_file(
                    corrupt_path
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertFalse(MODULE.generation_complete(args))

    def test_training_completion_binds_data_checkpoint_and_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            identity = build_valid_data(args)
            output = build_valid_training(
                args, arm="repair_50", data_identity=identity
            )
            self.assertTrue(
                MODULE.training_run_complete(
                    args, output, arm="repair_50", mode="formal"
                )
            )
            (output / "checkpoint_final/adapter_model.safetensors").write_bytes(
                b"tampered"
            )
            self.assertFalse(
                MODULE.training_run_complete(
                    args, output, arm="repair_50", mode="formal"
                )
            )

    def test_training_completion_rejects_minimal_dummy(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            build_valid_data(args)
            output = args.results_root / "repair_50" / "formal"
            checkpoint = output / "checkpoint_final"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            write_json(checkpoint / "adapter_config.json", {})
            write_json(
                output / "run_manifest.json",
                {
                    "protocol": "v5_stage1_message_masked_sft_7b",
                    "mode": "formal",
                    "held_out_test_accessed": False,
                },
            )
            self.assertFalse(
                MODULE.training_run_complete(
                    args, output, arm="repair_50", mode="formal"
                )
            )

    def test_registry_completion_binds_profile_and_local_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            identity = build_valid_data(args)
            registry = build_valid_registry(args, identity)
            self.assertTrue(MODULE.registry_complete(args))
            registry["entries"]["repair_50"] = None
            write_json(
                args.results_root / "checkpoint_registry.json", registry
            )
            self.assertFalse(MODULE.registry_complete(args))

    def test_evaluation_completion_recomputes_result_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            identity = build_valid_data(args)
            registry = build_valid_registry(args, identity)
            contract_path = build_valid_evaluation_shard(
                args, registry=registry
            )
            self.assertTrue(
                MODULE.evaluation_shard_complete(
                    args, arm="base_model", shard=0
                )
            )
            contract = json.loads(
                contract_path.read_text(encoding="utf-8")
            )
            for name in contract["result_sha256"]:
                result = contract_path.parent / name
                write_json(result, {})
                contract["result_sha256"][name] = MODULE.sha256_file(result)
            write_json(contract_path, contract)
            self.assertFalse(
                MODULE.evaluation_shard_complete(
                    args, arm="base_model", shard=0
                )
            )

    def test_summary_completion_requires_recomputed_scientific_body(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            args.protocol_root.mkdir(parents=True)
            args.processed_root.mkdir(parents=True)
            args.results_root.mkdir(parents=True)
            validation = args.processed_root / "validation_manifest.json"
            dynamic = args.protocol_root / "validation_dynamic_audit.json"
            registry = args.results_root / "checkpoint_registry.json"
            write_json(validation, {})
            write_json(dynamic, {})
            write_json(registry, {})
            contract_hashes = {}
            for arm in MODULE.EVAL_ARMS:
                arm_dir = args.results_root / "evaluation" / arm
                arm_dir.mkdir(parents=True)
                contract_hashes[arm] = {}
                for shard in range(4):
                    contract = arm_dir / (
                        f"run_contract.shard-{shard:03d}-of-004.json"
                    )
                    write_json(contract, {})
                    contract_hashes[arm][contract.name] = (
                        MODULE.sha256_file(contract)
                    )
            provenance = {
                "source": {
                    "experiment_source_commit": MODULE.source_commit(),
                    "base_model_revision": MODULE.STUDENT_REVISION,
                },
                "manifests": {
                    "split_sha256": MODULE.SPLIT_SHA256,
                    "evaluation_sha256": MODULE.sha256_file(validation),
                    "dynamic_audit_sha256": MODULE.sha256_file(dynamic),
                    "checkpoint_registry_sha256": MODULE.sha256_file(
                        registry
                    ),
                },
                "completion_audits_recomputed": True,
                "tool_action_interface": MODULE.EVAL_TOOL_ACTION_INTERFACE,
                "evaluation_contract_sha256": contract_hashes,
            }
            provenance["input_binding_sha256"] = __import__(
                "hashlib"
            ).sha256(
                json.dumps(
                    provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            expected = {
                "protocol": "v5_stage1_sft_causal_validation",
                "status": "PASS",
                "summary_contract_protocol": (
                    "v5_stage1_sft_causal_validation_summary_v2"
                ),
                "provenance": provenance,
                "arms": {"scientific_metrics": "present"},
            }
            output = args.results_root / "mechanism_screen_summary.json"
            write_json(output, expected)
            import scripts.summarize_v5_sft_causal as summary_module

            with (
                mock.patch.object(
                    MODULE, "evaluation_arm_complete", return_value=True
                ),
                mock.patch.object(
                    summary_module, "summarize", return_value=expected
                ),
            ):
                self.assertTrue(MODULE.summary_complete(args))
                metrics_free = dict(expected)
                metrics_free.pop("arms")
                write_json(output, metrics_free)
                self.assertFalse(MODULE.summary_complete(args))

    def test_evaluation_command_uses_one_host_shard_and_all_adapter_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            command = MODULE.evaluation_command(
                args, arm="repair_50", shard=2
            )
        self.assertEqual(command[command.index("--shard-index") + 1], "2")
        self.assertEqual(command[command.index("--num-shards") + 1], "4")
        self.assertEqual(
            command[command.index("--agent-api-base") + 1],
            "http://127.0.0.1:8102/v1",
        )
        self.assertEqual(
            command[command.index("--agent-model") + 1],
            "openai/v5-3-repair-50",
        )
        self.assertEqual(command.count("--adapter-dir"), 4)
        self.assertNotIn("official_test", " ".join(command))

    def test_frozen_topology_uses_14b_user_and_shared_32b_tp2_teacher(self):
        self.assertEqual(MODULE.GENERATION_SHARDS, 3)
        self.assertEqual(MODULE.GENERATION_ATTEMPTS_PER_CONDITION, 12)
        self.assertEqual(MODULE.EVALUATION_SHARDS, 4)
        self.assertEqual(len(MODULE.ARMS), 4)
        self.assertEqual(
            PREP_MODULE.EXPECTED_TEACHER_API_BASE_BY_SHARD,
            {
                0: "http://127.0.0.1:8011/v1",
                1: "http://127.0.0.1:8011/v1",
                2: "http://127.0.0.1:8011/v1",
            },
        )
        self.assertEqual(
            MODULE.TEACHER_MODEL, "Qwen/Qwen2.5-32B-Instruct-AWQ"
        )
        self.assertEqual(
            MODULE.USER_MODEL, "Qwen/Qwen2.5-14B-Instruct-AWQ"
        )

    def test_formal_generation_cannot_bypass_pilot_go(self):
        with tempfile.TemporaryDirectory() as directory:
            args = args_for(Path(directory))
            with (
                mock.patch.object(
                    MODULE, "generation_complete", return_value=False
                ),
                mock.patch.object(
                    MODULE, "protocol_complete", return_value=True
                ),
                mock.patch.object(
                    MODULE, "preflight_complete", return_value=True
                ),
                mock.patch.object(MODULE, "pilot_status", return_value=None),
            ):
                with self.assertRaisesRegex(
                    MODULE.StageError, "GO_FORMAL_GENERATION"
                ):
                    MODULE.generate_trajectories(args)

    def test_all_runs_pilot_before_formal_generation(self):
        args = SimpleNamespace(stage="all")
        calls = []
        names = (
            "preflight",
            "self_test",
            "prefetch_models",
            "build_protocol",
            "run_pilot",
            "generate_trajectories",
            "prepare_data",
            "train_arms",
            "build_registry",
            "evaluate",
            "summarize",
            "status",
        )
        patches = [
            mock.patch.object(
                MODULE,
                name,
                side_effect=lambda *_args, _name=name: calls.append(_name),
            )
            for name in names
        ]
        with mock.patch.object(MODULE, "parse_args", return_value=args):
            for patch in patches:
                patch.start()
            try:
                MODULE.main()
            finally:
                for patch in reversed(patches):
                    patch.stop()
        self.assertLess(calls.index("run_pilot"), calls.index("generate_trajectories"))


if __name__ == "__main__":
    unittest.main()
