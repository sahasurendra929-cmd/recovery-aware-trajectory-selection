#!/usr/bin/env python3
"""Fail-closed V5.3 controller for one RunPod host with four RTX 4090 GPUs.

The controller deliberately keeps environment creation outside the experiment
run.  Once the two frozen virtual environments exist, it can prefetch the
three pinned model revisions and execute the protocol, trajectory generation,
data preparation, four parallel QLoRA arms, checkpoint registration, four-way
end-to-end evaluation, and aggregation.

It never deletes or overwrites a partial scientific artifact.  A failed stage
must be archived by the operator before that stage is retried.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable
from urllib.request import Request, urlopen

try:
    import v5_judge_audit_contract as judge_audit_contract
except ModuleNotFoundError:
    from scripts import v5_judge_audit_contract as judge_audit_contract


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260722
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
SPLIT_SHA256 = "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
TEACHER_REVISION = "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
USER_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
USER_REVISION = "539535859b135b0244c91f3e59816150c8056698"
STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ARMS = ("perfect_success", "failure_raw", "repair_50", "repair_100")
EVAL_ARMS = ("base_model", *ARMS)
EVAL_MODEL_IDS = {
    "base_model": "openai/v5-3-base",
    "perfect_success": "openai/v5-3-perfect-success",
    "failure_raw": "openai/v5-3-failure-raw",
    "repair_50": "openai/v5-3-repair-50",
    "repair_100": "openai/v5-3-repair-100",
}
EVAL_TOOL_ACTION_INTERFACE = {
    "parallel_tool_calls": False,
    "parallel_tool_call_normalization": "execute_first_then_replan",
    "mixed_tool_call_content_normalization": "drop_text_preserve_sha256",
    "deferred_call_audit_field": "v5_stage1_parallel_calls_serialized",
    "mixed_content_audit_field": "v5_stage1_mixed_content_normalized",
    "audit_hash": "sha256",
    "clean_agent_factory": "v5_stage1_single_tool_agent",
    "error_agent_factory": "v5_stage1_single_tool_fault_agent",
    "post_injection_policy": "execute_injected_call_then_replan",
}
SERVED_ALIASES = tuple(value.removeprefix("openai/") for value in EVAL_MODEL_IDS.values())
GENERATION_SHARDS = 3
GENERATION_ATTEMPTS_PER_CONDITION = 12
GENERATION_TEMPERATURE = 0.2
GENERATION_TOP_P = 0.95
DERIVED_TRIAL_SEEDS = (
    574293,
    816256,
    420309,
    70872,
    419659,
    596026,
    55413,
    256204,
    120794,
    83442,
    692054,
    873496,
)
PILOT_PROTOCOL = "v5_3_train_only_feasibility_pilot"
PILOT_BASE_SEED = 20260730
PILOT_TRIAL_SEEDS = (
    733980,
    626956,
    183324,
    266579,
    559726,
    176907,
    35514,
    769350,
    589887,
    418549,
    589641,
    621567,
)
PILOT_TASKS = 24
PILOT_MIN_ONE_PAIR_TASKS = 15
PILOT_MIN_TWO_PAIR_TASKS = 4
INNER_TRAIN_PARTITION_SHA256 = (
    "c90255a2cf4100d8cf5b5fcf3e684144ed1a393fad1c214058146721c4ed6818"
)
LOSS_VALIDATION_TASK_IDS = (
    "airline:15",
    "airline:47",
    "retail:22",
    "retail:23",
    "retail:52",
    "retail:63",
    "retail:96",
    "retail:103",
)
PILOT_TASK_IDS = (
    "airline:1",
    "airline:11",
    "airline:14",
    "airline:33",
    "airline:38",
    "airline:40",
    "retail:2",
    "retail:8",
    "retail:10",
    "retail:15",
    "retail:19",
    "retail:25",
    "retail:30",
    "retail:35",
    "retail:54",
    "retail:67",
    "retail:69",
    "retail:72",
    "retail:85",
    "retail:92",
    "retail:93",
    "retail:104",
    "retail:106",
    "retail:110",
)
PILOT_FAULT_FAMILY_COUNTS = {
    "retail_missing_product_id": 4,
    "retail_missing_order_id": 5,
    "retail_missing_user_email": 5,
    "retail_missing_user_id": 4,
    "airline_missing_flight_number": 2,
    "airline_missing_reservation_id": 2,
    "airline_missing_user_id": 2,
}
EVALUATION_SHARDS = 4
MAX_MODEL_LEN = 32768
MIN_GPU_MEMORY_MIB = 24_000
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 4090"
MIN_FREE_DISK_GIB = 140
GT_INCOMPATIBLE_TASK_IDS = (
    "airline:0",
    "airline:10",
    "airline:28",
    "airline:34",
    "retail:24",
)
GT_FILTER_CONTRACT = {
    "protocol": "v5_3_gt_compatibility_filter_v1",
    "policy": "exclude_before_sharding",
    "teacher_mode": "ground_truth",
    "source_task_count": 83,
    "included_task_count": 78,
    "excluded_task_ids": list(GT_INCOMPATIBLE_TASK_IDS),
    "exclusion_reason": "no_expected_tool_actions",
    "selection_uses_rollouts_rewards_validation_or_test": False,
    "official_test_used": False,
}


class StageError(RuntimeError):
    """A protocol or runtime invariant failed."""


class PilotNoGo(StageError):
    """The preregistered pilot correctly stopped formal generation."""


_PILOT_RUNTIME_MODULE: Any | None = None


def pilot_runtime_module() -> Any:
    global _PILOT_RUNTIME_MODULE
    if _PILOT_RUNTIME_MODULE is not None:
        return _PILOT_RUNTIME_MODULE
    scripts_root = ROOT / "scripts"
    inserted = False
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
        inserted = True
    try:
        path = scripts_root / "run_v5_3_train_only_pilot.py"
        spec = importlib.util.spec_from_file_location(
            "_v5_3_pilot_controller_recompute",
            path,
        )
        if spec is None or spec.loader is None:
            raise StageError(f"cannot load pilot recomputation module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _PILOT_RUNTIME_MODULE = module
        return module
    finally:
        if inserted:
            sys.path.remove(str(scripts_root))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise StageError(f"expected JSON object: {path}")
    return value


def git_output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_commit() -> str:
    value = git_output("rev-parse", "HEAD^{commit}")
    if len(value) != 40:
        raise StageError(f"invalid source commit: {value!r}")
    return value


def require_clean_checkout(path: Path, *, label: str) -> None:
    for command, kind in (
        (["git", "diff", "--quiet", "--"], "unstaged"),
        (["git", "diff", "--cached", "--quiet"], "staged"),
    ):
        result = subprocess.run(command, cwd=path, check=False)
        if result.returncode:
            raise StageError(f"{label} has {kind} tracked changes")
    untracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if untracked:
        shown = "\n  ".join(untracked.splitlines()[:12])
        raise StageError(
            f"{label} contains source bytes not bound by HEAD:\n  {shown}"
        )


def run_checked(
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    printable = " ".join(command)
    print(f"[run] {printable}", flush=True)
    if log_path is None:
        result = subprocess.run(command, cwd=cwd, env=env, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {printable}\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    if result.returncode:
        suffix = f"; see {log_path}" if log_path else ""
        raise StageError(
            f"command exited {result.returncode}: {printable}{suffix}"
        )


def spawn_logged(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    handle.write(("\n$ " + " ".join(command) + "\n").encode())
    handle.flush()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle


def base_env(
    *, offline: bool = True, gpu: int | str | None = None
) -> dict[str, str]:
    env = dict(os.environ)
    env["HF_HOME"] = os.environ.get("HF_HOME", "/workspace/cache/huggingface")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            handle.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def http_json(url: str, api_key: str) -> Any:
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(request, timeout=10) as response:
        body = response.read()
    if not body:
        return None
    return json.loads(body)


def http_post_json(url: str, api_key: str, payload: dict[str, Any]) -> Any:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=900) as response:
        body = response.read()
    if not body:
        raise StageError(f"empty HTTP response from {url}")
    return json.loads(body)


def wait_for_service(
    *,
    port: int,
    api_key: str,
    expected_models: set[str],
    process: subprocess.Popen[bytes],
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not started"
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise StageError(
                f"vLLM service on port {port} exited early with code {code}"
            )
        try:
            http_json(f"http://127.0.0.1:{port}/health", api_key)
            payload = http_json(
                f"http://127.0.0.1:{port}/v1/models", api_key
            )
            observed = {
                str(row["id"]) for row in (payload or {}).get("data", [])
            }
            if observed >= expected_models:
                return
            last_error = f"aliases missing: {sorted(expected_models - observed)}"
        except Exception as error:  # service is expected to be unavailable at first
            last_error = repr(error)
        time.sleep(5)
    raise StageError(
        f"vLLM service on port {port} did not become healthy: {last_error}"
    )


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def write_pid(path: Path, process: subprocess.Popen[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{process.pid}\n", encoding="utf-8")


def remove_pid(path: Path) -> None:
    path.unlink(missing_ok=True)


def validate_gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 4:
            raise StageError(f"cannot parse nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "memory_mib": int(values[2]),
                "driver": values[3],
            }
        )
    if len(rows) != 4:
        raise StageError(f"V5.3 requires exactly 4 visible GPUs; found {len(rows)}")
    for expected_index, row in enumerate(rows):
        if row["index"] != expected_index:
            raise StageError(f"GPU indices must be 0..3; found {rows}")
        if row["name"] != EXPECTED_GPU_MODEL:
            raise StageError(
                f"GPU {expected_index} is not {EXPECTED_GPU_MODEL}: {row}"
            )
        if row["memory_mib"] < MIN_GPU_MEMORY_MIB:
            raise StageError(f"GPU {expected_index} has insufficient VRAM: {row}")
    return rows


def require_environment(python: Path, *, kind: str) -> dict[str, Any]:
    if not python.is_file():
        raise StageError(f"{kind} Python is absent: {python}")
    if kind == "serve":
        statement = (
            "import json,sys,tau2,transformers,vllm;"
            "print(json.dumps({'python':sys.version.split()[0],"
            "'transformers':transformers.__version__,'vllm':vllm.__version__}))"
        )
    else:
        statement = (
            "import json,sys,tau2,torch,transformers,peft,bitsandbytes,datasets,accelerate;"
            "print(json.dumps({'python':sys.version.split()[0],"
            "'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),"
            "'bf16_supported':torch.cuda.is_bf16_supported(),"
            "'transformers':transformers.__version__,"
            "'tau2':getattr(tau2,'__version__','installed'),"
            "'peft':peft.__version__,'bitsandbytes':bitsandbytes.__version__,"
            "'datasets':datasets.__version__,'accelerate':accelerate.__version__}))"
        )
    result = subprocess.run(
        [str(python), "-c", statement],
        cwd=ROOT,
        env=base_env(offline=False, gpu=0),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise StageError(
            f"{kind} environment import check failed:\n{result.stderr}"
        )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if not str(payload["python"]).startswith("3.12."):
        raise StageError(f"{kind} environment must use Python 3.12: {payload}")
    if kind == "serve" and (
        payload.get("transformers") != "4.55.4"
        or payload.get("vllm") != "0.10.2"
    ):
        raise StageError(f"serving package lock drift: {payload}")
    if kind == "train" and (
        payload.get("cuda_available") is not True
        or payload.get("bf16_supported") is not True
        or payload.get("cuda") != "12.8"
        or not str(payload.get("torch", "")).startswith("2.7.1")
        or payload.get("transformers") != "4.52.4"
        or payload.get("peft") != "0.15.2"
        or payload.get("bitsandbytes") != "0.46.0"
        or payload.get("datasets") != "3.6.0"
        or payload.get("accelerate") != "1.7.0"
    ):
        raise StageError(f"training CUDA environment drift: {payload}")
    return payload


def static_protocol_audit() -> None:
    config = ROOT / "configs" / "v5_3_sft_causal.yaml"
    handoff = ROOT / "V5_3_SINGLE_HOST_HANDOFF.md"
    prompt = ROOT / "V5_3_RUNPOD_4X4090_AGENT_PROMPT.md"
    for path in (config, handoff, prompt):
        if not path.is_file():
            raise StageError(f"missing V5.3 protocol file: {path}")
    config_text = config.read_text(encoding="utf-8")
    required_config_fragments = {
        "protocol: v5_3_train_only_pilot_then_formal_sft",
        "generation_universe_tasks: 78",
        "arm_train: {retail: 52, airline: 18, total: 70}",
        "loss_validation: {retail: 6, airline: 2, total: 8}",
        "max_model_len: 32768",
        "attempts_per_task_per_condition: 12",
        "minimum_tasks_with_at_least_one_pair: 15",
        "minimum_tasks_with_at_least_two_pairs: 4",
        "minimum_distinct_task_ids_with_pair: 40",
        "minimum_constructible_pairs: 48",
        "maximum_pairs_per_task: 2",
        "training_sequence_tokens_at_most: 8192",
        "formal_raw_root: data/raw/v5_3_sft_causal_generation",
        "formal_processed_root: data/processed/v5_3_sft_causal",
        "formal_results_root: results/v5_3_sft_causal",
    }
    missing = {
        fragment
        for fragment in required_config_fragments
        if fragment not in config_text
    }
    if missing:
        raise StageError(
            f"V5.3 config lacks frozen controller values: {sorted(missing)}"
        )
    for path in (handoff, prompt):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "--max-model-len" in line and "32768" not in line:
                raise StageError(f"{path.name} contains non-32768 vLLM command")


def preflight_complete(args: argparse.Namespace) -> bool:
    output = args.results_root / "preflight.json"
    if not output.is_file():
        return False
    try:
        repository_status = git_output(
            "status", "--porcelain", "--untracked-files=all"
        )
        tau2_status = git_output(
            "status",
            "--porcelain",
            "--untracked-files=all",
            cwd=args.tau2_root,
        )
        tau2_commit = git_output(
            "rev-parse", "HEAD^{commit}", cwd=args.tau2_root
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    if repository_status or tau2_status or tau2_commit != TAU2_COMMIT:
        return False
    try:
        payload = read_json(output)
    except StageError:
        return False
    gpus = payload.get("gpus")
    environments = payload.get("environments")
    return (
        payload.get("status") == "PASS"
        and payload.get("protocol") == "v5_3_single_host_preflight"
        and payload.get("source_commit") == source_commit()
        and payload.get("tau2_commit") == TAU2_COMMIT
        and payload.get("split_manifest_sha256") == SPLIT_SHA256
        and payload.get("expected_gpu_model") == EXPECTED_GPU_MODEL
        and isinstance(gpus, list)
        and len(gpus) == 4
        and all(
            isinstance(row, dict)
            and row.get("index") == index
            and row.get("name") == EXPECTED_GPU_MODEL
            and isinstance(row.get("memory_mib"), int)
            and row["memory_mib"] >= MIN_GPU_MEMORY_MIB
            for index, row in enumerate(gpus)
        )
        and isinstance(payload.get("free_disk_gib"), (int, float))
        and payload["free_disk_gib"] >= MIN_FREE_DISK_GIB
        and payload.get("ports_free")
        == [8001, 8011, 8012, 8013, 8100, 8101, 8102, 8103]
        and isinstance(environments, dict)
        and set(environments) == {"serve", "train"}
        and str((environments.get("serve") or {}).get("python", "")).startswith(
            "3.12."
        )
        and (environments.get("serve") or {}).get("transformers") == "4.55.4"
        and (environments.get("serve") or {}).get("vllm") == "0.10.2"
        and str((environments.get("train") or {}).get("python", "")).startswith(
            "3.12."
        )
        and str((environments.get("train") or {}).get("torch", "")).startswith(
            "2.7.1"
        )
        and (environments.get("train") or {}).get("cuda") == "12.8"
        and (environments.get("train") or {}).get("cuda_available") is True
        and (environments.get("train") or {}).get("bf16_supported") is True
        and (environments.get("train") or {}).get("transformers") == "4.52.4"
        and (environments.get("train") or {}).get("peft") == "0.15.2"
        and (environments.get("train") or {}).get("bitsandbytes") == "0.46.0"
        and (environments.get("train") or {}).get("datasets") == "3.6.0"
        and (environments.get("train") or {}).get("accelerate") == "1.7.0"
        and payload.get("official_test_used") is False
    )


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if preflight_complete(args):
        payload = read_json(args.results_root / "preflight.json")
        print("[skip] immutable V5.3 hardware/environment preflight is PASS")
        return payload
    output = args.results_root / "preflight.json"
    if output.exists():
        raise StageError(f"invalid preflight receipt must be archived: {output}")
    require_clean_checkout(ROOT, label="experiment repository")
    if not args.tau2_root.is_dir():
        raise StageError(f"tau2 checkout is absent: {args.tau2_root}")
    require_clean_checkout(args.tau2_root, label="tau2 checkout")
    observed_tau2 = git_output("rev-parse", "HEAD^{commit}", cwd=args.tau2_root)
    if observed_tau2 != TAU2_COMMIT:
        raise StageError(
            f"tau2 commit drift: expected {TAU2_COMMIT}, got {observed_tau2}"
        )
    split = ROOT / "artifacts/v5_stage0/manifests/split_manifest.json"
    if sha256_file(split) != SPLIT_SHA256:
        raise StageError("immutable Stage-0 split SHA-256 drift")
    static_protocol_audit()
    gpu_rows = validate_gpu_inventory()
    usage = shutil.disk_usage(args.workspace_root)
    free_gib = usage.free / 1024**3
    if free_gib < MIN_FREE_DISK_GIB:
        raise StageError(
            f"need at least {MIN_FREE_DISK_GIB} GiB free in "
            f"{args.workspace_root}; found {free_gib:.1f}"
        )
    ports = (8001, 8011, 8012, 8013, 8100, 8101, 8102, 8103)
    occupied = [port for port in ports if not port_is_free(port)]
    if occupied:
        raise StageError(f"reserved V5.3 ports are already in use: {occupied}")
    environments = {
        "serve": require_environment(args.serve_python, kind="serve"),
        "train": require_environment(args.train_python, kind="train"),
    }
    if not args.vllm.is_file():
        raise StageError(f"vLLM executable is absent: {args.vllm}")
    payload = {
        "status": "PASS",
        "protocol": "v5_3_single_host_preflight",
        "source_commit": source_commit(),
        "tau2_commit": observed_tau2,
        "split_manifest_sha256": SPLIT_SHA256,
        "expected_gpu_model": EXPECTED_GPU_MODEL,
        "gpus": gpu_rows,
        "free_disk_gib": round(free_gib, 2),
        "ports_free": list(ports),
        "environments": environments,
        "official_test_used": False,
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def protocol_files(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "generation_manifest": args.protocol_root / "generation_manifest.json",
        "evaluation_manifest": args.protocol_root / "validation_manifest.json",
        "generation_audit": args.protocol_root / "generation_dynamic_audit.json",
        "evaluation_audit": args.protocol_root / "validation_dynamic_audit.json",
    }


def protocol_complete(args: argparse.Namespace) -> bool:
    files = protocol_files(args)
    if not all(path.is_file() for path in files.values()):
        return False
    generation = read_json(files["generation_audit"])
    evaluation = read_json(files["evaluation_audit"])
    generation_manifest = read_json(files["generation_manifest"])
    evaluation_manifest = read_json(files["evaluation_manifest"])
    return (
        generation.get("status") == "COMPLETE"
        and evaluation.get("status") == "COMPLETE"
        and generation_manifest.get("protocol")
        == "v5_3_multifault_data_construction"
        and generation.get("manifest_protocol")
        == "v5_3_multifault_data_construction"
        and generation.get("manifest_sha256")
        == sha256_file(files["generation_manifest"])
        and evaluation.get("manifest_protocol")
        == "v5_stage1_sft_causal_validation"
        and evaluation.get("manifest_sha256")
        == sha256_file(files["evaluation_manifest"])
        and generation_manifest.get("paired_task_count") == 78
        and evaluation_manifest.get("paired_task_count") == 21
        and generation.get("official_test_used") is False
        and evaluation.get("official_test_used") is False
        and generation.get("verified_injections") == 78
        and evaluation.get("verified_injections") == 21
    )


def build_protocol(args: argparse.Namespace) -> None:
    if protocol_complete(args):
        print("[skip] protocol manifests and dynamic audits are complete")
        return
    if args.protocol_root.exists() and any(args.protocol_root.iterdir()):
        raise StageError(
            f"partial protocol directory must be archived, not overwritten: "
            f"{args.protocol_root}"
        )
    run_checked(
        [
            str(args.serve_python),
            "scripts/prepare_v5_3_manifests.py",
            "--tau2-root",
            str(args.tau2_root),
            "--split-manifest",
            "artifacts/v5_stage0/manifests/split_manifest.json",
            "--output-dir",
            str(args.protocol_root),
            "--seed",
            str(SEED),
        ],
        env=base_env(offline=True),
    )
    files = protocol_files(args)
    for manifest_key, output_key in (
        ("generation_manifest", "generation_audit"),
        ("evaluation_manifest", "evaluation_audit"),
    ):
        run_checked(
            [
                str(args.serve_python),
                "scripts/verify_v5_stage0_injections.py",
                "--tau2-root",
                str(args.tau2_root),
                "--split-manifest",
                "artifacts/v5_stage0/manifests/split_manifest.json",
                "--manifest",
                str(files[manifest_key]),
                "--output",
                str(files[output_key]),
            ],
            env=base_env(offline=True),
        )
    if not protocol_complete(args):
        raise StageError("protocol dynamic audit did not reach COMPLETE")


def pilot_files(args: argparse.Namespace) -> dict[str, Path]:
    protocol_root = args.pilot_root / "protocol"
    return {
        "universe": protocol_root / "effective_task_universe.json",
        "manifest": protocol_root / "pilot_manifest.json",
        "raw": args.pilot_root / "raw",
        "decision": args.pilot_root / "pilot_go_no_go.json",
    }


def _validate_pilot_manifest(args: argparse.Namespace) -> set[str]:
    files = pilot_files(args)
    if not files["universe"].is_file() or not files["manifest"].is_file():
        raise StageError("pilot universe/manifest is absent")
    if not protocol_complete(args):
        raise StageError("pilot cannot be validated before protocol completion")
    universe = read_json(files["universe"])
    manifest = read_json(files["manifest"])
    generation_manifest_path = protocol_files(args)["generation_manifest"]
    generation_manifest = read_json(generation_manifest_path)
    pilot_module = pilot_runtime_module()
    expected_universe, expected_manifest = (
        pilot_module.build_effective_universe_and_pilot(
            tau2_root=args.tau2_root,
            split_manifest_path=(
                ROOT / "artifacts/v5_stage0/manifests/split_manifest.json"
            ),
            generation_manifest_path=generation_manifest_path,
        )
    )
    if (
        canonical_sha256(universe) != canonical_sha256(expected_universe)
        or canonical_sha256(manifest) != canonical_sha256(expected_manifest)
    ):
        raise StageError(
            "pilot manifest/universe differ from the deterministic frozen builder"
        )
    generation_tasks = {
        f"{row.get('domain')}:{row.get('task_id')}"
        for row in generation_manifest.get("rows", [])
        if isinstance(row, dict)
    }
    universe_rows = universe.get("rows")
    universe_ids = universe.get("planned_task_ids")
    if (
        universe.get("protocol")
        != "v5_3_effective_arm_train_task_universe"
        or universe.get("created_without_v5_3_outcomes") is not True
        or universe.get("source_split") != "derived_inner_train"
        or universe.get("official_test_used") is not False
        or universe.get("split_manifest_sha256") != SPLIT_SHA256
        or universe.get("generation_manifest_sha256")
        != sha256_file(generation_manifest_path)
        or universe.get("inner_train_partition_sha256")
        != INNER_TRAIN_PARTITION_SHA256
        or not isinstance(universe_rows, list)
        or len(universe_rows) != 70
        or not isinstance(universe_ids, list)
        or len(universe_ids) != 70
        or len(set(universe_ids)) != 70
        or not set(universe_ids) <= generation_tasks
        or set(universe_ids)
        != generation_tasks - set(LOSS_VALIDATION_TASK_IDS)
        or (universe.get("counts") or {}).get("total") != 70
        or (universe.get("gt_compatibility_rule") or {}).get(
            "excluded_task_ids"
        )
        != sorted(GT_INCOMPATIBLE_TASK_IDS)
    ):
        raise StageError("pilot effective-task universe drift")
    rows = manifest.get("rows")
    planned = manifest.get("planned_task_ids")
    if (
        manifest.get("protocol") != PILOT_PROTOCOL
        or manifest.get("stage") != "train_only_pilot"
        or manifest.get("training") is not False
        or manifest.get("formal_data") is not False
        or manifest.get("pilot_trajectories_may_enter_formal_data") is not False
        or manifest.get("created_without_v5_3_outcomes") is not True
        or manifest.get("source_split") != "derived_inner_train"
        or manifest.get("official_test_used") is not False
        or manifest.get("split_manifest_sha256") != SPLIT_SHA256
        or manifest.get("generation_manifest_sha256")
        != sha256_file(generation_manifest_path)
        or manifest.get("inner_train_partition_sha256")
        != INNER_TRAIN_PARTITION_SHA256
        or manifest.get("effective_task_universe_canonical_sha256")
        != canonical_sha256(universe)
        or manifest.get("attempts_per_task_per_condition")
        != GENERATION_ATTEMPTS_PER_CONDITION
        or manifest.get("base_seed") != PILOT_BASE_SEED
        or tuple(manifest.get("trial_seeds") or ()) != PILOT_TRIAL_SEEDS
        or manifest.get("formal_base_seed") != SEED
        or tuple(manifest.get("formal_trial_seeds") or ())
        != DERIVED_TRIAL_SEEDS
        or manifest.get("pilot_formal_seed_sets_disjoint") is not True
        or set(PILOT_TRIAL_SEEDS) & set(DERIVED_TRIAL_SEEDS)
        or not isinstance(rows, list)
        or len(rows) != PILOT_TASKS
        or not isinstance(planned, list)
        or len(planned) != PILOT_TASKS
        or len(set(planned)) != PILOT_TASKS
        or set(planned) != set(PILOT_TASK_IDS)
        or not set(planned) <= set(universe_ids)
    ):
        raise StageError("pilot manifest protocol/seed/task drift")
    row_ids = {
        f"{row.get('domain')}:{row.get('task_id')}"
        for row in rows
        if isinstance(row, dict)
    }
    if row_ids != set(planned):
        raise StageError("pilot manifest rows differ from planned_task_ids")
    selection = manifest.get("selection") or {}
    if (
        selection.get("seed") != SEED
        or selection.get("salt") != "v5.3-pilot"
        or selection.get("uses_rollout_or_validation_outcomes") is not False
        or selection.get("domain_counts") != {"retail": 18, "airline": 6}
        or selection.get("fault_family_counts")
        != PILOT_FAULT_FAMILY_COUNTS
        or selection.get("teacher_route") != "ground_truth"
    ):
        raise StageError("pilot outcome-free stratification drift")
    return row_ids


def pilot_manifest_complete(args: argparse.Namespace) -> bool:
    try:
        _validate_pilot_manifest(args)
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def build_pilot_manifest(args: argparse.Namespace) -> None:
    if pilot_manifest_complete(args):
        print("[skip] frozen 24-task pilot manifest is complete")
        return
    files = pilot_files(args)
    if args.pilot_root.exists() and any(args.pilot_root.iterdir()):
        raise StageError(
            "partial/invalid pilot artifacts must be archived before retry: "
            f"{args.pilot_root}"
        )
    run_checked(
        [
            str(args.serve_python),
            "scripts/run_v5_3_train_only_pilot.py",
            "manifest",
            "--tau2-root",
            str(args.tau2_root),
            "--split-manifest",
            "artifacts/v5_stage0/manifests/split_manifest.json",
            "--generation-manifest",
            str(protocol_files(args)["generation_manifest"]),
            "--output-dir",
            str(files["manifest"].parent),
        ],
        env=base_env(offline=True),
        log_path=args.results_root / "logs" / "pilot-manifest.log",
    )
    if not pilot_manifest_complete(args):
        raise StageError("pilot manifest validation failed")


def prefetch_models(args: argparse.Namespace) -> None:
    pairs = (
        (TEACHER_MODEL, TEACHER_REVISION),
        (USER_MODEL, USER_REVISION),
        (STUDENT_MODEL, STUDENT_REVISION),
    )
    statement = (
        "from huggingface_hub import snapshot_download;"
        "import sys;"
        "snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2])"
    )
    for model, revision in pairs:
        python = args.train_python if model == STUDENT_MODEL else args.serve_python
        run_checked(
            [str(python), "-c", statement, model, revision],
            env=base_env(offline=False),
            log_path=args.results_root / "logs" / "model_prefetch.log",
        )


def self_test(args: argparse.Namespace) -> None:
    tests = [
        *sorted(str(path.relative_to(ROOT)) for path in ROOT.glob("tests/test_v5_3*.py")),
        "tests/test_v5_checkpoint_registry.py",
        "tests/test_v5_dynamic_audit.py",
        "tests/test_v5_sft_causal_data.py",
        "tests/test_v5_sft_causal_eval.py",
        "tests/test_v5_sft_causal_generation.py",
        "tests/test_v5_sft_causal_runner.py",
        "tests/test_v5_sft_causal_train.py",
        "tests/test_v5_stage1_manifests.py",
    ]
    run_checked(
        [str(args.train_python), "-m", "pytest", "-q", *tests],
        env=base_env(offline=True),
        log_path=args.results_root / "logs" / "self-test.log",
    )


def vllm_base_command(
    args: argparse.Namespace,
    model: str,
    revision: str,
    *,
    dtype: str = "auto",
) -> list[str]:
    return [
        str(args.vllm),
        "serve",
        model,
        "--revision",
        revision,
        "--host",
        "127.0.0.1",
        "--dtype",
        dtype,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--generation-config",
        "vllm",
    ]


def generation_runtime_preflight_path(
    args: argparse.Namespace, *, phase: str
) -> Path:
    if phase not in {"pilot", "formal", "screen"}:
        raise StageError(f"invalid generation preflight phase: {phase}")
    return (
        args.results_root
        / "runtime_preflights"
        / f"generation-{phase}.json"
    )


RUNTIME_PREFLIGHT_PROTOCOL = "v5_3_generation_runtime_preflight_v2"
RUNTIME_PREFLIGHT_TOOL_NAME = "runtime_preflight_echo"
RUNTIME_PREFLIGHT_MARKER = "v5_3_live_chat_tools"
RUNTIME_PREFLIGHT_MAX_TOKENS = 512
RUNTIME_PREFLIGHT_PROMPT_REPETITIONS = 28_000
RUNTIME_PREFLIGHT_MIN_PROMPT_TOKENS = 27_000
RUNTIME_PREFLIGHT_FATAL_LOG_PATTERNS = (
    r"cuda out of memory",
    r"outofmemoryerror",
    r"no available memory for the cache blocks",
    r"not enough kv cache memory",
    r"tool(?:-call)? parser (?:error|failed)",
    r"failed to parse (?:a )?tool",
    r"error parsing (?:a )?tool",
    r"traceback \(most recent call last\)",
    r"\bnan\b",
)


def _generation_service_specifications() -> list[dict[str, Any]]:
    return [
        {
            "role": "user_and_judge",
            "visible_gpus": "0",
            "port": 8001,
            "model": USER_MODEL,
            "revision": USER_REVISION,
            "api_key": "stage1-user-local",
            "max_num_seqs": 3,
            "memory": "0.90",
            "tensor_parallel_size": 1,
            "name": "generation-user-judge",
        },
        {
            "role": "teacher",
            "visible_gpus": "1,2",
            "port": 8011,
            "model": TEACHER_MODEL,
            "revision": TEACHER_REVISION,
            "api_key": "stage1-teacher-local",
            "max_num_seqs": 3,
            "memory": "0.90",
            "tensor_parallel_size": 2,
            "name": "generation-teacher-tp2-gpu1-2",
        },
    ]


def _generation_service_command(
    args: argparse.Namespace, specification: dict[str, Any]
) -> list[str]:
    return [
        *vllm_base_command(
            args,
            str(specification["model"]),
            str(specification["revision"]),
            dtype="float16",
        ),
        "--served-model-name",
        str(specification["model"]),
        "--port",
        str(specification["port"]),
        "--api-key",
        str(specification["api_key"]),
        "--gpu-memory-utilization",
        str(specification["memory"]),
        "--max-num-seqs",
        str(specification["max_num_seqs"]),
        "--tensor-parallel-size",
        str(specification["tensor_parallel_size"]),
        "--quantization",
        "awq",
    ]


def _runtime_gpu_identity() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 5:
            raise StageError(f"cannot parse runtime nvidia-smi row: {line!r}")
        try:
            rows.append(
                {
                    "index": int(values[0]),
                    "uuid": values[1],
                    "name": values[2],
                    "memory_mib": int(values[3]),
                    "driver": values[4],
                }
            )
        except ValueError as error:
            raise StageError(
                f"cannot parse runtime nvidia-smi row: {line!r}"
            ) from error
    if len(rows) != 4:
        raise StageError(
            f"runtime preflight requires exactly four GPUs; found {len(rows)}"
        )
    for expected_index, row in enumerate(rows):
        if (
            row["index"] != expected_index
            or not str(row["uuid"]).startswith("GPU-")
            or row["name"] != EXPECTED_GPU_MODEL
            or row["memory_mib"] < MIN_GPU_MEMORY_MIB
            or not str(row["driver"])
        ):
            raise StageError(f"invalid runtime GPU identity: {row}")
    return rows


def _snapshot_tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise StageError(f"model snapshot is absent: {path}")
    rows: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        stat = child.stat()
        rows.append(
            {
                "path": str(child.relative_to(path)),
                "bytes": stat.st_size,
                "resolved_path": str(child.resolve()),
            }
        )
    if not rows:
        raise StageError(f"model snapshot contains no files: {path}")
    return canonical_sha256(rows)


def _resolve_model_snapshot(
    args: argparse.Namespace, *, model: str, revision: str
) -> dict[str, Any]:
    statement = (
        "from huggingface_hub import snapshot_download;"
        "import sys;"
        "print(snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],"
        "local_files_only=True))"
    )
    result = subprocess.run(
        [str(args.serve_python), "-c", statement, model, revision],
        cwd=ROOT,
        env=base_env(offline=True),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise StageError(
            f"cannot resolve cached model snapshot {model}@{revision}: "
            f"{result.stderr.strip()}"
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise StageError(f"snapshot resolver returned no path for {model}")
    snapshot = Path(lines[-1]).expanduser().resolve()
    if snapshot.name != revision:
        raise StageError(
            f"resolved snapshot revision drift for {model}: {snapshot}"
        )
    return {
        "model": model,
        "requested_revision": revision,
        "resolved_revision": snapshot.name,
        "snapshot_path": str(snapshot),
        "snapshot_tree_sha256": _snapshot_tree_sha256(snapshot),
    }


def _snapshot_receipt_valid(
    value: Any, *, model: str, revision: str
) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        snapshot = Path(str(value.get("snapshot_path"))).resolve()
        return (
            value.get("model") == model
            and value.get("requested_revision") == revision
            and value.get("resolved_revision") == revision
            and snapshot.name == revision
            and value.get("snapshot_path") == str(snapshot)
            and value.get("snapshot_tree_sha256")
            == _snapshot_tree_sha256(snapshot)
        )
    except (OSError, StageError):
        return False


def _runtime_receipt_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"canonical_receipt_sha256", "receipt_pointer"}
    }


def _runtime_receipt_pointer(
    path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    invocation = payload.get("invocation") or {}
    return {
        "path": str(path.resolve()),
        "protocol": RUNTIME_PREFLIGHT_PROTOCOL,
        "phase": payload.get("phase"),
        "invocation_id": invocation.get("invocation_id"),
        "canonical_receipt_sha256": payload.get(
            "canonical_receipt_sha256"
        ),
    }


def _runtime_probe_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": RUNTIME_PREFLIGHT_TOOL_NAME,
                "description": (
                    "Echo the immutable runtime-preflight marker and request id."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "marker": {"type": "string"},
                        "request_id": {"type": "string"},
                    },
                    "required": ["marker", "request_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _chat_tool_probe(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    request_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a runtime preflight. You must make exactly one "
                    f"{RUNTIME_PREFLIGHT_TOOL_NAME} tool call and emit no text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\nCall {RUNTIME_PREFLIGHT_TOOL_NAME} exactly once "
                    f"with marker={RUNTIME_PREFLIGHT_MARKER!r} and "
                    f"request_id={request_id!r}."
                ),
            },
        ],
        "tools": _runtime_probe_tool(),
        "tool_choice": {
            "type": "function",
            "function": {"name": RUNTIME_PREFLIGHT_TOOL_NAME},
        },
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "max_tokens": RUNTIME_PREFLIGHT_MAX_TOKENS,
        "stream": False,
    }
    payload = http_post_json(
        f"{api_base}/chat/completions", api_key, request_payload
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise StageError(f"{request_id} returned a malformed choices list")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if (
        not isinstance(choice, dict)
        or choice.get("finish_reason") != "tool_calls"
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or (content is not None and str(content).strip())
        or not isinstance(calls, list)
        or len(calls) != 1
    ):
        raise StageError(f"{request_id} did not return one clean tool call")
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    arguments_text = (
        function.get("arguments") if isinstance(function, dict) else None
    )
    try:
        arguments = json.loads(arguments_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise StageError(f"{request_id} returned invalid tool arguments") from error
    expected_arguments = {
        "marker": RUNTIME_PREFLIGHT_MARKER,
        "request_id": request_id,
    }
    if (
        not isinstance(call, dict)
        or not isinstance(function, dict)
        or call.get("type") != "function"
        or not isinstance(call.get("id"), str)
        or not call["id"]
        or function.get("name") != RUNTIME_PREFLIGHT_TOOL_NAME
        or arguments != expected_arguments
    ):
        raise StageError(f"{request_id} returned the wrong tool call")
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        raise StageError(f"{request_id} returned no token usage")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or not RUNTIME_PREFLIGHT_MIN_PROMPT_TOKENS
        <= prompt_tokens
        <= MAX_MODEL_LEN - RUNTIME_PREFLIGHT_MAX_TOKENS
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or not 0 < completion_tokens <= RUNTIME_PREFLIGHT_MAX_TOKENS
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise StageError(f"{request_id} returned invalid token usage: {usage}")
    return {
        "status": "PASS",
        "request_id": request_id,
        "endpoint": "chat/completions",
        "parallel_tool_calls": False,
        "max_tokens": RUNTIME_PREFLIGHT_MAX_TOKENS,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tool_call_count": 1,
        "tool_name": RUNTIME_PREFLIGHT_TOOL_NAME,
        "tool_arguments_sha256": canonical_sha256(arguments),
        "response_sha256": canonical_sha256(payload),
    }


def _probe_evidence_valid(value: Any, *, request_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    return (
        value.get("status") == "PASS"
        and value.get("request_id") == request_id
        and value.get("endpoint") == "chat/completions"
        and value.get("parallel_tool_calls") is False
        and value.get("max_tokens") == RUNTIME_PREFLIGHT_MAX_TOKENS
        and isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and RUNTIME_PREFLIGHT_MIN_PROMPT_TOKENS
        <= prompt_tokens
        <= MAX_MODEL_LEN - RUNTIME_PREFLIGHT_MAX_TOKENS
        and isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and 0 < completion_tokens <= RUNTIME_PREFLIGHT_MAX_TOKENS
        and value.get("total_tokens") == prompt_tokens + completion_tokens
        and value.get("tool_call_count") == 1
        and value.get("tool_name") == RUNTIME_PREFLIGHT_TOOL_NAME
        and value.get("tool_arguments_sha256")
        == canonical_sha256(
            {
                "marker": RUNTIME_PREFLIGHT_MARKER,
                "request_id": request_id,
            }
        )
        and isinstance(value.get("response_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["response_sha256"]) is not None
    )


def _service_log_evidence(context: dict[str, Any]) -> dict[str, Any]:
    process = context["process"]
    if process.poll() is not None:
        raise StageError(
            f"{context['role']} vLLM process exited during runtime preflight"
        )
    context["handle"].flush()
    path = Path(context["log_path"]).resolve()
    start = int(context["log_start_byte"])
    body = path.read_bytes()
    segment = body[start:]
    text = segment.decode("utf-8", errors="replace").lower()
    matches = [
        pattern
        for pattern in RUNTIME_PREFLIGHT_FATAL_LOG_PATTERNS
        if re.search(pattern, text)
    ]
    if matches:
        raise StageError(
            f"{context['role']} runtime log contains fatal evidence: {matches}"
        )
    return {
        "path": str(path),
        "start_byte": start,
        "end_byte": len(body),
        "segment_sha256": hashlib.sha256(segment).hexdigest(),
        "fatal_pattern_matches": [],
        "process_alive_after_probes": True,
    }


def _log_evidence_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        path = Path(str(value.get("path"))).resolve()
        start = int(value.get("start_byte"))
        end = int(value.get("end_byte"))
        body = path.read_bytes()
        segment = body[start:end]
        text = segment.decode("utf-8", errors="replace").lower()
        return (
            value.get("path") == str(path)
            and 0 <= start <= end <= len(body)
            and value.get("segment_sha256")
            == hashlib.sha256(segment).hexdigest()
            and value.get("fatal_pattern_matches") == []
            and value.get("process_alive_after_probes") is True
            and not any(
                re.search(pattern, text)
                for pattern in RUNTIME_PREFLIGHT_FATAL_LOG_PATTERNS
            )
        )
    except (OSError, TypeError, ValueError):
        return False


def generation_runtime_preflight_complete(
    args: argparse.Namespace, *, phase: str
) -> bool:
    path = generation_runtime_preflight_path(args, phase=phase)
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        gpu_inventory = _runtime_gpu_identity()
        commit = source_commit()
    except (OSError, subprocess.CalledProcessError, StageError):
        return False
    specifications = {
        str(row["role"]): row for row in _generation_service_specifications()
    }
    host_preflight_path = (args.results_root / "preflight.json").resolve()
    host_preflight = payload.get("host_preflight")
    invocation = payload.get("invocation")
    roles = payload.get("roles")
    if (
        payload.get("protocol") != RUNTIME_PREFLIGHT_PROTOCOL
        or payload.get("status") != "PASS"
        or payload.get("phase") != phase
        or payload.get("source_commit") != commit
        or payload.get("max_model_len") != MAX_MODEL_LEN
        or payload.get("expected_gpu_model") != EXPECTED_GPU_MODEL
        or payload.get("official_test_used") is not False
        or not host_preflight_path.is_file()
        or host_preflight
        != {
            "path": str(host_preflight_path),
            "sha256": sha256_file(host_preflight_path),
        }
        or payload.get("gpu_inventory") != gpu_inventory
        or not isinstance(invocation, dict)
        or not isinstance(roles, dict)
        or set(roles) != set(specifications)
        or payload.get("canonical_receipt_sha256")
        != canonical_sha256(_runtime_receipt_core(payload))
        or payload.get("receipt_pointer")
        != _runtime_receipt_pointer(path, payload)
    ):
        return False
    invocation_services = invocation.get("services")
    invocation_core = {
        key: value for key, value in invocation.items() if key != "invocation_id"
    }
    if (
        not isinstance(invocation.get("started_unix_ns"), int)
        or not isinstance(invocation.get("nonce"), str)
        or re.fullmatch(r"[0-9a-f]{32}", invocation["nonce"]) is None
        or not isinstance(invocation_services, dict)
        or set(invocation_services) != set(specifications)
        or invocation.get("invocation_id") != canonical_sha256(invocation_core)
    ):
        return False
    gpu_by_index = {row["index"]: row for row in gpu_inventory}
    for role, specification in specifications.items():
        value = roles.get(role)
        service = invocation_services.get(role)
        models_endpoint = (
            value.get("models_endpoint") if isinstance(value, dict) else None
        )
        concurrency_probe = (
            value.get("concurrency_probe") if isinstance(value, dict) else None
        )
        expected_gpu_indices = [
            int(index)
            for index in str(specification["visible_gpus"]).split(",")
        ]
        expected_log = (
            args.results_root / "logs" / f"{specification['name']}.log"
        ).resolve()
        if (
            not isinstance(value, dict)
            or not isinstance(service, dict)
            or not isinstance(models_endpoint, dict)
            or not isinstance(concurrency_probe, dict)
            or value.get("model") != specification["model"]
            or value.get("revision") != specification["revision"]
            or value.get("tensor_parallel_size")
            != specification["tensor_parallel_size"]
            or value.get("api_base")
            != f"http://127.0.0.1:{specification['port']}/v1"
            or models_endpoint.get("status") != "PASS"
            or not isinstance(
                models_endpoint.get("observed_model_ids"), list
            )
            or specification["model"]
            not in models_endpoint.get("observed_model_ids", [])
            or not isinstance(models_endpoint.get("response_sha256"), str)
            or re.fullmatch(
                r"[0-9a-f]{64}", models_endpoint["response_sha256"]
            )
            is None
            or not _probe_evidence_valid(
                value.get("long_context_probe"),
                request_id=f"{phase}-{role}-long",
            )
            or concurrency_probe.get("status") != "PASS"
            or concurrency_probe.get("requests") != 3
            or concurrency_probe.get("client_barrier_size")
            != 3
            or concurrency_probe.get("parallel_tool_calls")
            is not False
            or concurrency_probe.get("max_tokens")
            != RUNTIME_PREFLIGHT_MAX_TOKENS
            or not isinstance(concurrency_probe.get("results"), list)
            or len(concurrency_probe["results"]) != 3
            or any(
                not _probe_evidence_valid(
                    result,
                    request_id=f"{phase}-{role}-concurrent-{index}",
                )
                for index, result in enumerate(
                    concurrency_probe["results"]
                )
            )
            or isinstance(service.get("pid"), bool)
            or not isinstance(service.get("pid"), int)
            or service["pid"] <= 0
            or service.get("command_sha256")
            != canonical_sha256(
                _generation_service_command(args, specification)
            )
            or service.get("visible_gpu_indices") != expected_gpu_indices
            or service.get("gpu_identity")
            != [gpu_by_index[index] for index in expected_gpu_indices]
            or not _snapshot_receipt_valid(
                service.get("model_snapshot"),
                model=str(specification["model"]),
                revision=str(specification["revision"]),
            )
            or not isinstance(service.get("log_evidence"), dict)
            or service["log_evidence"].get("path") != str(expected_log)
            or not _log_evidence_valid(service.get("log_evidence"))
        ):
            return False
    return True


def generation_runtime_preflight_pointer(
    args: argparse.Namespace, *, phase: str
) -> dict[str, Any]:
    if not generation_runtime_preflight_complete(args, phase=phase):
        raise StageError(f"{phase} runtime preflight receipt is not valid")
    path = generation_runtime_preflight_path(args, phase=phase)
    payload = read_json(path)
    pointer = payload.get("receipt_pointer")
    if not isinstance(pointer, dict):
        raise StageError(f"{phase} runtime preflight pointer is malformed")
    return dict(pointer)


def generation_runtime_contract_binding_path(
    args: argparse.Namespace, *, phase: str
) -> Path:
    if phase not in {"pilot", "formal"}:
        raise StageError(f"invalid runtime contract-binding phase: {phase}")
    return (
        args.results_root
        / "runtime_preflights"
        / f"generation-{phase}-contracts.json"
    )


def _runtime_phase_contract_paths(
    args: argparse.Namespace, *, phase: str
) -> list[Path]:
    function = (
        pilot_contract_path
        if phase == "pilot"
        else generation_contract_path
    )
    return [function(args, shard) for shard in range(GENERATION_SHARDS)]


def _runtime_binding_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "canonical_binding_sha256"
    }


def generation_runtime_contract_binding_complete(
    args: argparse.Namespace, *, phase: str
) -> bool:
    path = generation_runtime_contract_binding_path(args, phase=phase)
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
        contracts = [
            {
                "path": str(contract.resolve()),
                "sha256": sha256_file(contract),
            }
            for contract in _runtime_phase_contract_paths(args, phase=phase)
        ]
        pointer = generation_runtime_preflight_pointer(args, phase=phase)
        commit = source_commit()
    except (OSError, subprocess.CalledProcessError, StageError):
        return False
    return (
        payload.get("protocol")
        == "v5_3_generation_runtime_contract_binding_v1"
        and payload.get("status") == "COMPLETE"
        and payload.get("phase") == phase
        and payload.get("source_commit") == commit
        and payload.get("runtime_preflight") == pointer
        and payload.get("contracts") == contracts
        and payload.get("official_test_used") is False
        and payload.get("canonical_binding_sha256")
        == canonical_sha256(_runtime_binding_core(payload))
    )


def write_generation_runtime_contract_binding(
    args: argparse.Namespace, *, phase: str
) -> None:
    contracts = _runtime_phase_contract_paths(args, phase=phase)
    if not all(path.is_file() for path in contracts):
        raise StageError(f"{phase} runtime binding lacks all three contracts")
    payload = {
        "protocol": "v5_3_generation_runtime_contract_binding_v1",
        "status": "COMPLETE",
        "phase": phase,
        "source_commit": source_commit(),
        "runtime_preflight": generation_runtime_preflight_pointer(
            args, phase=phase
        ),
        "contracts": [
            {
                "path": str(contract.resolve()),
                "sha256": sha256_file(contract),
            }
            for contract in contracts
        ],
        "official_test_used": False,
    }
    payload["canonical_binding_sha256"] = canonical_sha256(payload)
    output = generation_runtime_contract_binding_path(args, phase=phase)
    if output.is_file():
        current = read_json(output)
        if current == payload:
            return
        archive = output.parent / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        os.replace(
            output,
            archive
            / f"{output.stem}.{sha256_file(output)}.{time.time_ns()}.json",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    if not generation_runtime_contract_binding_complete(args, phase=phase):
        raise StageError(f"{phase} runtime contract binding validation failed")


def run_generation_runtime_preflight(
    args: argparse.Namespace,
    *,
    phase: str,
    service_contexts: dict[str, dict[str, Any]],
    gpu_inventory: list[dict[str, Any]],
    invocation_nonce: str,
    invocation_started_unix_ns: int,
) -> None:
    output = generation_runtime_preflight_path(args, phase=phase)
    if not preflight_complete(args):
        raise StageError("live generation preflight lacks a valid host preflight")
    expected_roles = {
        str(row["role"]) for row in _generation_service_specifications()
    }
    if set(service_contexts) != expected_roles:
        raise StageError("live generation service context role drift")
    if gpu_inventory != _runtime_gpu_identity():
        raise StageError("GPU identity changed during service startup")
    if output.exists():
        archive = output.parent / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        archived = archive / (
            f"{output.stem}.{sha256_file(output)}.{time.time_ns()}.json"
        )
        os.replace(output, archived)
    specifications = {
        str(row["role"]): row for row in _generation_service_specifications()
    }
    role_receipts: dict[str, Any] = {}
    long_prompt = "x " * RUNTIME_PREFLIGHT_PROMPT_REPETITIONS
    for role, specification in specifications.items():
        context = service_contexts[role]
        process = context["process"]
        if process.poll() is not None:
            raise StageError(f"{role} service exited before live probes")
        api_base = f"http://127.0.0.1:{specification['port']}/v1"
        models_payload = http_json(
            f"{api_base}/models", str(specification["api_key"])
        )
        if (
            not isinstance(models_payload, dict)
            or not isinstance(models_payload.get("data"), list)
        ):
            raise StageError(f"{role} returned malformed /models metadata")
        observed_model_ids = sorted(
            str(row.get("id"))
            for row in models_payload["data"]
            if isinstance(row, dict) and row.get("id") is not None
        )
        if specification["model"] not in observed_model_ids:
            raise StageError(f"{role} live model alias is absent")
        long_result = _chat_tool_probe(
            api_base=api_base,
            api_key=str(specification["api_key"]),
            model=str(specification["model"]),
            prompt=long_prompt,
            request_id=f"{phase}-{role}-long",
        )
        barrier = threading.Barrier(3)

        def concurrent_probe(index: int) -> dict[str, Any]:
            barrier.wait(timeout=30)
            return _chat_tool_probe(
                api_base=api_base,
                api_key=str(specification["api_key"]),
                model=str(specification["model"]),
                prompt=long_prompt,
                request_id=f"{phase}-{role}-concurrent-{index}",
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(concurrent_probe, index): index
                for index in range(3)
            }
            concurrent_by_index = {
                futures[future]: future.result() for future in futures
            }
        concurrent = [
            concurrent_by_index[index] for index in range(3)
        ]
        if process.poll() is not None:
            raise StageError(f"{role} service exited after live probes")
        role_receipts[role] = {
            "model": specification["model"],
            "revision": specification["revision"],
            "tensor_parallel_size": specification[
                "tensor_parallel_size"
            ],
            "api_base": api_base,
            "models_endpoint": {
                "status": "PASS",
                "observed_model_ids": observed_model_ids,
                "response_sha256": canonical_sha256(models_payload),
            },
            "long_context_probe": long_result,
            "concurrency_probe": {
                "status": "PASS",
                "requests": 3,
                "client_barrier_size": 3,
                "parallel_tool_calls": False,
                "max_tokens": RUNTIME_PREFLIGHT_MAX_TOKENS,
                "results": concurrent,
            },
        }
    gpu_by_index = {row["index"]: row for row in gpu_inventory}
    invocation_services: dict[str, Any] = {}
    for role, specification in specifications.items():
        context = service_contexts[role]
        gpu_indices = [
            int(index)
            for index in str(specification["visible_gpus"]).split(",")
        ]
        invocation_services[role] = {
            "pid": context["process"].pid,
            "command_sha256": context["command_sha256"],
            "visible_gpu_indices": gpu_indices,
            "gpu_identity": [gpu_by_index[index] for index in gpu_indices],
            "model_snapshot": context["model_snapshot"],
            "log_evidence": _service_log_evidence(context),
        }
    invocation = {
        "nonce": invocation_nonce,
        "started_unix_ns": invocation_started_unix_ns,
        "services": invocation_services,
    }
    invocation["invocation_id"] = canonical_sha256(invocation)
    host_preflight_path = (args.results_root / "preflight.json").resolve()
    receipt = {
        "protocol": RUNTIME_PREFLIGHT_PROTOCOL,
        "status": "PASS",
        "phase": phase,
        "source_commit": source_commit(),
        "max_model_len": MAX_MODEL_LEN,
        "expected_gpu_model": EXPECTED_GPU_MODEL,
        "host_preflight": {
            "path": str(host_preflight_path),
            "sha256": sha256_file(host_preflight_path),
        },
        "gpu_inventory": gpu_inventory,
        "invocation": invocation,
        "roles": role_receipts,
        "official_test_used": False,
    }
    receipt["canonical_receipt_sha256"] = canonical_sha256(receipt)
    receipt["receipt_pointer"] = _runtime_receipt_pointer(output, receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    if not generation_runtime_preflight_complete(args, phase=phase):
        raise StageError("runtime generation preflight receipt validation failed")


def start_generation_services(
    args: argparse.Namespace,
    *,
    phase: str,
) -> list[tuple[subprocess.Popen[bytes], Any, Path]]:
    specifications = _generation_service_specifications()
    gpu_inventory = _runtime_gpu_identity()
    invocation_nonce = os.urandom(16).hex()
    invocation_started_unix_ns = time.time_ns()
    snapshots = {
        str(item["role"]): _resolve_model_snapshot(
            args,
            model=str(item["model"]),
            revision=str(item["revision"]),
        )
        for item in specifications
    }
    services = []
    service_contexts: dict[str, dict[str, Any]] = {}
    try:
        for item in specifications:
            if not port_is_free(item["port"]):
                raise StageError(f"generation port is occupied: {item['port']}")
            command = _generation_service_command(args, item)
            log = args.results_root / "logs" / f"{item['name']}.log"
            log_start_byte = log.stat().st_size if log.is_file() else 0
            process, handle = spawn_logged(
                command,
                env=base_env(
                    offline=True, gpu=str(item["visible_gpus"])
                ),
                log_path=log,
            )
            pid_path = args.results_root / "pids" / f"{item['name']}.pid"
            write_pid(pid_path, process)
            services.append((process, handle, pid_path))
            service_contexts[str(item["role"])] = {
                "role": str(item["role"]),
                "process": process,
                "handle": handle,
                "command_sha256": canonical_sha256(command),
                "log_path": str(log.resolve()),
                "log_start_byte": log_start_byte,
                "model_snapshot": snapshots[str(item["role"])],
            }
        for item, (process, _, _) in zip(specifications, services):
            wait_for_service(
                port=int(item["port"]),
                api_key=str(item["api_key"]),
                expected_models={str(item["model"])},
                process=process,
                timeout_seconds=args.health_timeout,
            )
        run_generation_runtime_preflight(
            args,
            phase=phase,
            service_contexts=service_contexts,
            gpu_inventory=gpu_inventory,
            invocation_nonce=invocation_nonce,
            invocation_started_unix_ns=invocation_started_unix_ns,
        )
        return services
    except Exception:
        stop_services(services)
        raise


def stop_services(
    services: Iterable[tuple[subprocess.Popen[bytes], Any, Path]]
) -> None:
    rows = list(services)
    for process, _, _ in rows:
        stop_process(process)
    for _, handle, pid_path in rows:
        handle.close()
        remove_pid(pid_path)


def pilot_contract_path(args: argparse.Namespace, shard: int) -> Path:
    return (
        pilot_files(args)["raw"]
        / f"run_contract.shard-{shard:03d}-of-{GENERATION_SHARDS:03d}.json"
    )


def _pilot_expected_shard_rows(
    args: argparse.Namespace, shard: int
) -> list[dict[str, Any]]:
    manifest = read_json(pilot_files(args)["manifest"])
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise StageError("pilot manifest rows are malformed")
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(str(row.get("pair_id")).encode("utf-8")).hexdigest(),
            str(row.get("pair_id")),
        ),
    )
    return ordered[shard::GENERATION_SHARDS]


def _validate_pilot_shard(
    args: argparse.Namespace, shard: int
) -> tuple[set[str], set[Path]]:
    path = pilot_contract_path(args, shard)
    if not path.is_file():
        raise StageError(f"pilot contract is absent: {path}")
    value = read_json(path)
    files = pilot_files(args)
    expected_rows = _pilot_expected_shard_rows(args, shard)
    expected_task_ids = [
        f"{row.get('domain')}:{row.get('task_id')}" for row in expected_rows
    ]
    generation_manifest_path = protocol_files(args)["generation_manifest"]
    generation_audit_path = protocol_files(args)["generation_audit"]
    dynamic_identity = value.get("dynamic_audit_identity") or {}
    if (
        value.get("protocol") != "v5_stage1_inner_train_generation_run"
        or value.get("subprotocol") != PILOT_PROTOCOL
        or value.get("status") != "COMPLETE"
        or value.get("source_commit") != source_commit()
        or value.get("source_split") != "derived_inner_train"
        or value.get("official_test_used") is not False
        or value.get("formal_data") is not False
        or value.get("pilot_trajectories_may_enter_formal_data") is not False
        or value.get("split_manifest_sha256") != SPLIT_SHA256
        or value.get("generation_manifest_sha256")
        != sha256_file(generation_manifest_path)
        or value.get("pilot_manifest_sha256")
        != sha256_file(files["manifest"])
        or value.get("task_ids") != expected_task_ids
        or value.get("task_universe_complete") is not False
        or value.get("shard_index") != shard
        or value.get("num_shards") != GENERATION_SHARDS
        or value.get("num_trials") != GENERATION_ATTEMPTS_PER_CONDITION
        or tuple(value.get("trial_seeds") or ()) != PILOT_TRIAL_SEEDS
        or dynamic_identity.get("protocol")
        != "v5_stage1_dynamic_injection_audit"
        or dynamic_identity.get("sha256") != sha256_file(generation_audit_path)
        or dynamic_identity.get("manifest_sha256")
        != sha256_file(generation_manifest_path)
        or dynamic_identity.get("split_manifest_sha256") != SPLIT_SHA256
        or dynamic_identity.get("source_split") != "derived_inner_train"
        or dynamic_identity.get("verified_injections") != 78
        or dynamic_identity.get("official_test_used") is not False
        or dynamic_identity.get("official_test_sealed") is not True
    ):
        raise StageError(f"pilot contract metadata/provenance drift: {path}")
    if value.get("teacher") != {
        "model": TEACHER_MODEL,
        "revision": TEACHER_REVISION,
        "api_base": "http://127.0.0.1:8011/v1",
        "tensor_parallel_size": 2,
    }:
        raise StageError(f"pilot teacher contract drift: {path}")
    if value.get("user_and_judge") != {
        "model": USER_MODEL,
        "revision": USER_REVISION,
        "api_base": "http://127.0.0.1:8001/v1",
        "strict_judge_schema": True,
        "judge_format_retries": 1,
    }:
        raise StageError(f"pilot user/judge contract drift: {path}")
    if value.get("decoding") != {
        "trajectory_temperature": GENERATION_TEMPERATURE,
        "trajectory_top_p": GENERATION_TOP_P,
        "judge_temperature": 0.0,
        "judge_top_p": 1.0,
        "max_tokens": 512,
        "max_steps": 60,
        "parallel_tool_calls": False,
    }:
        raise StageError(f"pilot decoding contract drift: {path}")
    strategy = value.get("user_loop_db0_strategy") or {}
    if (
        strategy.get("stronger_user_simulator") is not True
        or strategy.get("bounded_nonzero_fixed_seed_attempts") is not True
        or strategy.get("diagnostic_only") is not True
        or strategy.get("hidden_rescue") is not False
        or strategy.get("failed_attempts_repaired_or_relabelled") is not False
        or strategy.get("extra_attempts_after_budget") is not False
    ):
        raise StageError(f"pilot bounded-attempt strategy drift: {path}")
    result_hashes = value.get("result_sha256")
    domains = {
        str(row.get("domain"))
        for row in expected_rows
        if row.get("domain") in {"retail", "airline"}
    }
    expected_names = {
        f"{domain}_{condition}.shard-{shard:03d}-of-003.json"
        for domain in domains
        for condition in ("clean", "error")
    }
    if not isinstance(result_hashes, dict) or set(result_hashes) != expected_names:
        raise StageError(f"pilot result declaration set drift: {path}")
    result_paths: set[Path] = set()
    for name, declared_hash in result_hashes.items():
        result_path = files["raw"] / name
        if (
            not isinstance(declared_hash, str)
            or not result_path.is_file()
            or sha256_file(result_path) != declared_hash
        ):
            raise StageError(f"pilot result hash drift: {result_path}")
        result_paths.add(result_path.resolve())
        payload = read_json(result_path)
        simulations = payload.get("simulations")
        if not isinstance(simulations, list):
            raise StageError(f"pilot result lacks simulations: {result_path}")
        domain = name.split("_", 1)[0]
        expected_domain_tasks = {
            str(row.get("task_id"))
            for row in expected_rows
            if row.get("domain") == domain
        }
        expected_slots = {
            (task_id, trial)
            for task_id in expected_domain_tasks
            for trial in range(GENERATION_ATTEMPTS_PER_CONDITION)
        }
        observed_slots: set[tuple[str, int]] = set()
        for simulation in simulations:
            if not isinstance(simulation, dict):
                raise StageError(f"malformed pilot simulation: {result_path}")
            task_id = str(simulation.get("task_id"))
            trial_value = simulation.get(
                "attempt_index",
                simulation.get("attempt", simulation.get("trial")),
            )
            if isinstance(trial_value, bool):
                raise StageError(f"invalid pilot trial: {result_path}")
            try:
                trial = int(trial_value)
            except (TypeError, ValueError) as error:
                raise StageError(
                    f"invalid pilot trial: {result_path}"
                ) from error
            slot = (task_id, trial)
            if slot not in expected_slots or slot in observed_slots:
                raise StageError(
                    f"unexpected/duplicate pilot slot {slot}: {result_path}"
                )
            observed_slots.add(slot)
            if simulation.get("seed") != PILOT_TRIAL_SEEDS[trial]:
                raise StageError(
                    f"pilot trial seed drift for {slot}: {result_path}"
                )
        if observed_slots != expected_slots:
            raise StageError(f"incomplete pilot slots: {result_path}")
    try:
        observed_evidence = (
            judge_audit_contract.validate_strict_judge_evidence(
                sorted(result_paths),
                maximum_content_attempts=2,
            )
        )
    except judge_audit_contract.StrictJudgeEvidenceError as error:
        raise StageError(
            f"pilot strict-judge evidence is incomplete: {path}"
        ) from error
    if value.get("strict_judge_audit_evidence") != observed_evidence:
        raise StageError(f"pilot strict-judge evidence contract drift: {path}")
    return set(expected_task_ids), result_paths


def pilot_generation_complete(args: argparse.Namespace) -> bool:
    if not pilot_manifest_complete(args):
        return False
    expected = _validate_pilot_manifest(args)
    try:
        rows = [_validate_pilot_shard(args, shard) for shard in range(3)]
    except (OSError, ValueError, StageError):
        return False
    union: set[str] = set()
    for task_ids, _ in rows:
        if union & task_ids:
            return False
        union.update(task_ids)
    return union == expected


def _strict_judge_audit_paths(args: argparse.Namespace) -> list[Path]:
    raw = pilot_files(args)["raw"]
    return sorted(
        path.resolve()
        for path in raw.glob("logs/**/strict_nl_judge_audit_*.json")
        if path.is_file()
    )


def _validate_strict_judge_audits(
    args: argparse.Namespace,
    result_paths: list[Path],
) -> tuple[list[Path], dict[str, Any]]:
    paths = _strict_judge_audit_paths(args)
    try:
        evidence = judge_audit_contract.validate_strict_judge_evidence(
            result_paths,
            audit_paths=paths,
            maximum_content_attempts=2,
        )
    except judge_audit_contract.StrictJudgeEvidenceError as error:
        raise StageError(
            f"pilot strict-judge evidence is incomplete: {error}"
        ) from error
    return paths, evidence


def _pilot_bound_inputs(
    args: argparse.Namespace,
) -> tuple[list[Path], list[Path]]:
    result_paths: set[Path] = set()
    for shard in range(GENERATION_SHARDS):
        _, shard_paths = _validate_pilot_shard(args, shard)
        if result_paths & shard_paths:
            raise StageError("duplicate pilot result binding across shards")
    result_paths.update(shard_paths)
    inputs = sorted(result_paths)
    judge_audits, _ = _validate_strict_judge_audits(args, inputs)
    return inputs, judge_audits


def _validate_pilot_decision(args: argparse.Namespace) -> str:
    if not preflight_complete(args):
        raise StageError("pilot decision lacks a current host preflight")
    if not generation_runtime_preflight_complete(args, phase="pilot"):
        raise StageError("pilot runtime preflight receipt is absent or invalid")
    if not generation_runtime_contract_binding_complete(
        args, phase="pilot"
    ):
        raise StageError("pilot contracts are not bound to the runtime receipt")
    decision_path = pilot_files(args)["decision"]
    if not decision_path.is_file():
        raise StageError("pilot GO/NO-GO decision is absent")
    payload = read_json(decision_path)
    inputs, judge_audits = _pilot_bound_inputs(args)
    try:
        judge_evidence = judge_audit_contract.validate_strict_judge_evidence(
            inputs,
            audit_paths=judge_audits,
            maximum_content_attempts=2,
        )
    except judge_audit_contract.StrictJudgeEvidenceError as error:
        raise StageError(
            f"pilot decision strict-judge evidence is incomplete: {error}"
        ) from error
    provenance = payload.get("provenance") or {}
    if (
        payload.get("protocol") != PILOT_PROTOCOL
        or payload.get("status")
        not in {"GO_FORMAL_GENERATION", "NO_GO_STOP"}
        or provenance.get("source_commit") != source_commit()
        or provenance.get("pilot_manifest_sha256")
        != sha256_file(pilot_files(args)["manifest"])
        or provenance.get("input_sha256")
        != {str(path): sha256_file(path) for path in inputs}
        or provenance.get("strict_judge_audit_sha256")
        != {
            row["audit_path"]: row["audit_sha256"]
            for row in judge_evidence["calls"]
        }
        or provenance.get("strict_judge_evidence_mapping_sha256")
        != judge_evidence["canonical_mapping_sha256"]
        or provenance.get("official_test_used") is not False
    ):
        raise StageError("pilot decision provenance drift")
    decision = payload.get("decision") or {}
    observed = decision.get("observed") or {}
    checks = decision.get("checks") or {}
    one_pair = observed.get("tasks_with_at_least_one_pair")
    two_pair = observed.get("tasks_with_second_pair")
    if (
        isinstance(one_pair, bool)
        or not isinstance(one_pair, int)
        or isinstance(two_pair, bool)
        or not isinstance(two_pair, int)
        or not 0 <= two_pair <= one_pair <= PILOT_TASKS
        or observed.get("pilot_tasks") != PILOT_TASKS
    ):
        raise StageError("pilot decision observed-count drift")
    expected_checks = {
        "one_pair_tasks_at_least_15": (
            one_pair >= PILOT_MIN_ONE_PAIR_TASKS
        ),
        "two_pair_tasks_at_least_4": (
            two_pair >= PILOT_MIN_TWO_PAIR_TASKS
        ),
    }
    go = all(expected_checks.values())
    expected_status = "GO_FORMAL_GENERATION" if go else "NO_GO_STOP"
    claim = payload.get("claim_boundary") or {}
    if (
        checks != expected_checks
        or decision.get("all_preregistered_checks_pass") is not go
        or payload.get("status") != expected_status
        or payload.get("exit_code") != (0 if go else 20)
        or (payload.get("formal_gate_unchanged") or {}).get(
            "minimum_distinct_task_ids"
        )
        != 40
        or (payload.get("formal_gate_unchanged") or {}).get("minimum_pairs")
        != 48
        or (payload.get("formal_gate_unchanged") or {}).get(
            "maximum_pairs_per_task"
        )
        != 2
        or claim.get("training_started") is not False
        or claim.get("pilot_trajectories_enter_formal_data") is not False
        or claim.get("formal_seed_schedule_is_disjoint") is not True
        or claim.get("official_test_used") is not False
    ):
        raise StageError("pilot GO/NO-GO decision semantics drift")
    base = payload.get("base_task_level_audit") or {}
    task_audit = base.get("task_audit")
    if (
        not isinstance(task_audit, list)
        or len(task_audit) != PILOT_TASKS
        or any(
            not isinstance(row, dict)
            or row.get("observed_attempts") != {"clean": 12, "error": 12}
            for row in task_audit
        )
    ):
        raise StageError("pilot base task-level audit coverage drift")
    recomputed = pilot_runtime_module().build_pilot_audit(
        input_paths=inputs,
        pilot_manifest_path=pilot_files(args)["manifest"],
        judge_audit_paths=judge_audits,
    )
    if canonical_sha256(payload) != canonical_sha256(recomputed):
        raise StageError(
            "pilot decision differs from a fresh audit of its hash-bound raw inputs"
        )
    return expected_status


def pilot_status(args: argparse.Namespace) -> str | None:
    try:
        return _validate_pilot_decision(args)
    except (OSError, ValueError, RuntimeError):
        return None


def _pilot_partial_shard_paths(
    args: argparse.Namespace, shard: int
) -> list[Path]:
    raw = pilot_files(args)["raw"]
    if not raw.is_dir():
        return []
    token = f"shard-{shard:03d}-of-{GENERATION_SHARDS:03d}"
    return sorted(
        path
        for path in raw.rglob("*")
        if token in path.name or token in str(path.relative_to(raw))
    )


def run_pilot(args: argparse.Namespace) -> str:
    if not preflight_complete(args):
        raise StageError(
            "pilot requires a current immutable hardware/environment preflight"
        )
    existing = pilot_status(args)
    if existing is not None:
        print(f"[skip] immutable pilot decision is {existing}")
        if existing == "NO_GO_STOP":
            raise PilotNoGo(
                "pilot returned NO_GO_STOP; formal generation is prohibited"
            )
        return existing
    build_pilot_manifest(args)
    files = pilot_files(args)
    files["raw"].mkdir(parents=True, exist_ok=True)
    missing_shards: list[int] = []
    for shard in range(GENERATION_SHARDS):
        try:
            _validate_pilot_shard(args, shard)
            print(f"[skip] pilot generation shard {shard} is COMPLETE")
            continue
        except (OSError, ValueError, StageError):
            partial = _pilot_partial_shard_paths(args, shard)
            if partial:
                shown = "\n  ".join(str(path) for path in partial[:12])
                raise StageError(
                    "archive the incomplete pilot shard before retrying "
                    f"{shard}:\n  {shown}"
                )
            missing_shards.append(shard)
    services: list[tuple[subprocess.Popen[bytes], Any, Path]] = []
    workers: list[tuple[subprocess.Popen[bytes], Any, int]] = []
    try:
        if missing_shards or not generation_runtime_preflight_complete(
            args, phase="pilot"
        ):
            services = start_generation_services(args, phase="pilot")
        for shard in missing_shards:
            command = [
                str(args.serve_python),
                "scripts/run_v5_3_train_only_pilot.py",
                "generate",
                "--tau2-root",
                str(args.tau2_root),
                "--split-manifest",
                "artifacts/v5_stage0/manifests/split_manifest.json",
                "--generation-manifest",
                str(protocol_files(args)["generation_manifest"]),
                "--dynamic-audit",
                str(protocol_files(args)["generation_audit"]),
                "--pilot-manifest",
                str(files["manifest"]),
                "--output-dir",
                str(files["raw"]),
                "--shard-index",
                str(shard),
                "--teacher-api-base",
                "http://127.0.0.1:8011/v1",
                "--teacher-api-key",
                "stage1-teacher-local",
                "--user-api-base",
                "http://127.0.0.1:8001/v1",
                "--user-api-key",
                "stage1-user-local",
                "--teacher-revision",
                TEACHER_REVISION,
                "--user-judge-revision",
                USER_REVISION,
                "--expected-source-commit",
                source_commit(),
            ]
            process, handle = spawn_logged(
                command,
                env=base_env(offline=True),
                log_path=args.results_root
                / "logs"
                / f"pilot-generation-shard-{shard}.log",
            )
            workers.append((process, handle, shard))
        failures = []
        for process, _, shard in workers:
            code = process.wait()
            if code:
                failures.append((shard, code))
        if failures:
            raise StageError(f"pilot generation shard failures: {failures}")
    finally:
        for process, handle, _ in workers:
            if process.poll() is None:
                stop_process(process)
            handle.close()
        if services:
            stop_services(services)
    if not pilot_generation_complete(args):
        raise StageError("pilot generation ended without three COMPLETE shards")
    write_generation_runtime_contract_binding(args, phase="pilot")
    inputs, judge_audits = _pilot_bound_inputs(args)
    if files["decision"].exists():
        recovered = pilot_status(args)
        if recovered is None:
            raise StageError(
                f"invalid pilot decision must be archived: {files['decision']}"
            )
        print(
            "[resume] refreshed the live pilot runtime receipt and recovered "
            f"the immutable {recovered} decision"
        )
        if recovered == "NO_GO_STOP":
            raise PilotNoGo(
                "pilot returned NO_GO_STOP; formal generation is prohibited"
            )
        return recovered
    command = [
        str(args.serve_python),
        "scripts/run_v5_3_train_only_pilot.py",
        "audit",
        "--pilot-manifest",
        str(files["manifest"]),
        "--judge-audits",
        *[str(path) for path in judge_audits],
        "--output",
        str(files["decision"]),
        *[str(path) for path in inputs],
    ]
    log_path = args.results_root / "logs" / "pilot-audit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=base_env(offline=True),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode not in {0, 20}:
        raise StageError(
            f"pilot audit exited {result.returncode}; see {log_path}"
        )
    status_value = _validate_pilot_decision(args)
    if result.returncode == 0 and status_value != "GO_FORMAL_GENERATION":
        raise StageError("pilot exit-code/status disagreement")
    if result.returncode == 20 and status_value != "NO_GO_STOP":
        raise StageError("pilot exit-code/status disagreement")
    if status_value == "NO_GO_STOP":
        raise PilotNoGo(
            "pilot returned NO_GO_STOP; formal generation is prohibited"
        )
    return status_value


def generation_contract_path(args: argparse.Namespace, shard: int) -> Path:
    return (
        args.raw_root
        / f"run_contract.shard-{shard:03d}-of-{GENERATION_SHARDS:03d}.json"
    )


def _validate_generation_shard(
    args: argparse.Namespace, shard: int
) -> set[str]:
    path = generation_contract_path(args, shard)
    if not path.is_file():
        raise StageError(f"generation contract is absent: {path}")
    value = read_json(path)
    if (
        value.get("protocol") != "v5_stage1_inner_train_generation_run"
        or
        value.get("status") != "COMPLETE"
        or value.get("shard_index") != shard
        or value.get("num_shards") != GENERATION_SHARDS
        or value.get("num_trials") != GENERATION_ATTEMPTS_PER_CONDITION
        or value.get("official_test_used") is not False
        or value.get("source_split") != "derived_inner_train"
        or value.get("generation_manifest_protocol")
        != "v5_3_multifault_data_construction"
        or value.get("gt_compatibility_filter") != GT_FILTER_CONTRACT
        or value.get("manifest_sha256")
        != sha256_file(protocol_files(args)["generation_manifest"])
        or value.get("split_manifest_sha256") != SPLIT_SHA256
        or value.get("source_commit") != source_commit()
    ):
        raise StageError(f"generation contract metadata drift: {path}")
    expected_sampling = {
        "protocol": "v5_3_frozen_stochastic_attempts_v1",
        "temperature": GENERATION_TEMPERATURE,
        "top_p": GENERATION_TOP_P,
        "base_seed": SEED,
        "num_trials": GENERATION_ATTEMPTS_PER_CONDITION,
        "derived_trial_seeds": list(DERIVED_TRIAL_SEEDS),
        "derived_trial_seeds_are_distinct": True,
        "judge_temperature": 0.0,
        "judge_top_p": 1.0,
    }
    if value.get("sampling_contract") != expected_sampling:
        raise StageError(f"generation sampling contract drift: {path}")
    if value.get("teacher") != {
        "model": TEACHER_MODEL,
        "revision": TEACHER_REVISION,
        "api_base": "http://127.0.0.1:8011/v1",
        "mode": "ground_truth",
    }:
        raise StageError(f"generation teacher contract drift: {path}")
    expected_user = {
        "model": USER_MODEL,
        "revision": USER_REVISION,
        "api_base": "http://127.0.0.1:8001/v1",
    }
    if value.get("user") != expected_user:
        raise StageError(f"generation user contract drift: {path}")
    expected_judge = {
        **expected_user,
        "protocol": "v5_strict_nl_judge_v1",
        "content_attempts": 2,
        "schema_failure": "fail_closed",
        "raw_response_audit": True,
    }
    if value.get("judge") != expected_judge:
        raise StageError(f"generation judge contract drift: {path}")
    decoding = value.get("decoding")
    if decoding != {
        "temperature": GENERATION_TEMPERATURE,
        "top_p": GENERATION_TOP_P,
        "max_tokens": 512,
        "parallel_tool_calls": False,
        "parallel_tool_call_normalization": "execute_first_then_replan",
        "mixed_tool_call_content_normalization": "drop_text_preserve_sha256",
        "max_steps": 60,
        "task_timeout_seconds": 900.0,
        "seed": SEED,
        "derived_trial_seeds": list(DERIVED_TRIAL_SEEDS),
    }:
        raise StageError(f"generation decoding contract drift: {path}")
    gt_preflight = value.get("gt_compatibility_preflight")
    if (
        not isinstance(gt_preflight, dict)
        or gt_preflight.get("status") != "PASS"
        or gt_preflight.get("filter_verified") is not True
        or gt_preflight.get("checked_task_count") != 83
        or gt_preflight.get("compatible_task_count") != 78
        or gt_preflight.get("incompatible_task_ids")
        != list(GT_INCOMPATIBLE_TASK_IDS)
    ):
        raise StageError(f"generation GT preflight drift: {path}")
    task_ids = value.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or len(task_ids) != len(set(task_ids))
        or any(
            not isinstance(task_id, str)
            or task_id in GT_INCOMPATIBLE_TASK_IDS
            for task_id in task_ids
        )
    ):
        raise StageError(f"generation shard task IDs are invalid: {path}")
    result_hashes = value.get("result_sha256")
    if not isinstance(result_hashes, dict) or not result_hashes:
        raise StageError(f"generation contract has no result hashes: {path}")
    expected_names = {
        f"{domain}_{condition}.shard-{shard:03d}-of-003.json"
        for domain in ("retail", "airline")
        if any(task_id.startswith(f"{domain}:") for task_id in task_ids)
        for condition in ("clean", "error")
    }
    if set(result_hashes) != expected_names:
        raise StageError(f"generation result declaration set drift: {path}")
    result_paths: list[Path] = []
    for name, declared in result_hashes.items():
        result_path = args.raw_root / name
        if (
            not isinstance(declared, str)
            or not result_path.is_file()
            or sha256_file(result_path) != declared
        ):
            raise StageError(f"generation result hash drift: {result_path}")
        result_paths.append(result_path.resolve())
        payload = read_json(result_path)
        simulations = payload.get("simulations")
        if not isinstance(simulations, list):
            raise StageError(f"generation result lacks simulations: {result_path}")
        domain, condition = name.split("_", 1)[0], name.split("_", 1)[1].split(".")[0]
        expected_domain_tasks = {
            task_id.split(":", 1)[1]
            for task_id in task_ids
            if task_id.startswith(f"{domain}:")
        }
        expected_slots = {
            (task_id, trial)
            for task_id in expected_domain_tasks
            for trial in range(GENERATION_ATTEMPTS_PER_CONDITION)
        }
        observed_slots: set[tuple[str, int]] = set()
        for simulation in simulations:
            if not isinstance(simulation, dict):
                raise StageError(f"malformed simulation in {result_path}")
            task_id = str(simulation.get("task_id"))
            trial_value = simulation.get(
                "attempt_index",
                simulation.get("attempt", simulation.get("trial")),
            )
            if isinstance(trial_value, bool):
                raise StageError(f"invalid trial in {result_path}")
            try:
                trial = int(trial_value)
            except (TypeError, ValueError) as error:
                raise StageError(f"invalid trial in {result_path}") from error
            slot = (task_id, trial)
            if slot not in expected_slots or slot in observed_slots:
                raise StageError(
                    f"unexpected/duplicate generation slot {slot} in {result_path}"
                )
            observed_slots.add(slot)
            if simulation.get("seed") != DERIVED_TRIAL_SEEDS[trial]:
                raise StageError(
                    f"derived trial seed drift for {slot} in {result_path}"
                )
            declared_condition = simulation.get("condition")
            if declared_condition is not None and declared_condition != condition:
                raise StageError(
                    f"condition drift for {slot} in {result_path}"
                )
        if observed_slots != expected_slots:
            raise StageError(f"incomplete generation slots in {result_path}")
    try:
        observed_evidence = (
            judge_audit_contract.validate_strict_judge_evidence(
                result_paths,
                maximum_content_attempts=2,
            )
        )
    except judge_audit_contract.StrictJudgeEvidenceError as error:
        raise StageError(
            f"generation strict-judge evidence is incomplete: {path}"
        ) from error
    if value.get("strict_judge_audit_evidence") != observed_evidence:
        raise StageError(
            f"generation strict-judge evidence contract drift: {path}"
        )
    return set(task_ids)


def generation_shard_complete(args: argparse.Namespace, shard: int) -> bool:
    try:
        _validate_generation_shard(args, shard)
    except (OSError, ValueError, StageError):
        return False
    return True


def generation_complete(args: argparse.Namespace) -> bool:
    if (
        not args.raw_root.is_dir()
        or not protocol_complete(args)
        or pilot_status(args) != "GO_FORMAL_GENERATION"
        or not generation_runtime_preflight_complete(args, phase="formal")
        or not generation_runtime_contract_binding_complete(
            args, phase="formal"
        )
    ):
        return False
    manifest = read_json(protocol_files(args)["generation_manifest"])
    expected_tasks = {
        f"{row.get('domain')}:{row.get('task_id')}"
        for row in manifest.get("rows", [])
        if isinstance(row, dict)
    }
    try:
        shard_sets = [
            _validate_generation_shard(args, shard)
            for shard in range(GENERATION_SHARDS)
        ]
    except (OSError, ValueError, StageError):
        return False
    union: set[str] = set()
    for tasks in shard_sets:
        if union & tasks:
            return False
        union.update(tasks)
    return len(expected_tasks) == 78 and union == expected_tasks


def partial_generation_paths(
    args: argparse.Namespace, shard: int
) -> list[Path]:
    if not args.raw_root.is_dir():
        return []
    token = f"shard-{shard:03d}-of-{GENERATION_SHARDS:03d}"
    return sorted(
        path
        for path in args.raw_root.rglob("*")
        if token in path.name or token in str(path.relative_to(args.raw_root))
    )


def generate_trajectories(args: argparse.Namespace) -> None:
    if not preflight_complete(args):
        raise StageError(
            "formal generation requires a current immutable preflight"
        )
    if generation_complete(args):
        print("[skip] all three V5.3 generation shards are COMPLETE")
        return
    if not protocol_complete(args):
        raise StageError("protocol and dynamic-audit barrier is incomplete")
    pilot_decision = pilot_status(args)
    if pilot_decision != "GO_FORMAL_GENERATION":
        raise StageError(
            "formal generation requires an immutable "
            f"GO_FORMAL_GENERATION pilot decision; found {pilot_decision!r}"
        )
    args.raw_root.mkdir(parents=True, exist_ok=True)
    missing_shards = []
    for shard in range(GENERATION_SHARDS):
        if generation_shard_complete(args, shard):
            print(f"[skip] generation shard {shard} is COMPLETE")
            continue
        partial = partial_generation_paths(args, shard)
        if partial:
            shown = "\n  ".join(str(path) for path in partial[:12])
            raise StageError(
                "archive the incomplete files for generation shard "
                f"{shard} before retrying:\n  {shown}"
            )
        missing_shards.append(shard)
    commit = source_commit()
    files = protocol_files(args)
    services = start_generation_services(args, phase="formal")
    workers: list[tuple[subprocess.Popen[bytes], Any, int]] = []
    try:
        for shard in missing_shards:
            command = [
                str(args.serve_python),
                "scripts/run_v5_sft_causal_generate.py",
                "--tau2-root",
                str(args.tau2_root),
                "--split-manifest",
                "artifacts/v5_stage0/manifests/split_manifest.json",
                "--manifest",
                str(files["generation_manifest"]),
                "--dynamic-audit",
                str(files["generation_audit"]),
                "--output-dir",
                str(args.raw_root),
                "--teacher-model",
                TEACHER_MODEL,
                "--teacher-revision",
                TEACHER_REVISION,
                "--teacher-api-base",
                "http://127.0.0.1:8011/v1",
                "--teacher-api-key",
                "stage1-teacher-local",
                "--user-model",
                USER_MODEL,
                "--user-revision",
                USER_REVISION,
                "--user-api-base",
                "http://127.0.0.1:8001/v1",
                "--user-api-key",
                "stage1-user-local",
                "--judge-model",
                USER_MODEL,
                "--judge-revision",
                USER_REVISION,
                "--judge-api-base",
                "http://127.0.0.1:8001/v1",
                "--judge-api-key",
                "stage1-user-local",
                "--teacher-mode",
                "ground_truth",
                "--condition",
                "both",
                "--shard-index",
                str(shard),
                "--num-shards",
                str(GENERATION_SHARDS),
                "--num-trials",
                str(GENERATION_ATTEMPTS_PER_CONDITION),
                "--temperature",
                str(GENERATION_TEMPERATURE),
                "--top-p",
                str(GENERATION_TOP_P),
                "--seed",
                str(SEED),
                "--max-steps",
                "60",
                "--timeout",
                "900",
                "--max-tokens",
                "512",
                "--expected-source-commit",
                commit,
            ]
            process, handle = spawn_logged(
                command,
                env=base_env(offline=True),
                log_path=args.results_root
                / "logs"
                / f"generation-shard-{shard}.log",
            )
            workers.append((process, handle, shard))
        failures = []
        for process, _, shard in workers:
            code = process.wait()
            if code:
                failures.append((shard, code))
        if failures:
            raise StageError(f"generation shard failures: {failures}")
    finally:
        for process, handle, _ in workers:
            if process.poll() is None:
                stop_process(process)
            handle.close()
        stop_services(services)
    write_generation_runtime_contract_binding(args, phase="formal")
    if not generation_complete(args):
        raise StageError("generation ended without three COMPLETE contracts")


def data_complete(args: argparse.Namespace) -> bool:
    audit_path = args.processed_root / "audit.json"
    hashes_path = args.processed_root / "hashes.json"
    if not audit_path.is_file() or not hashes_path.is_file():
        return False
    audit = read_json(audit_path)
    hashes = read_json(hashes_path)
    if (
        not hashes
        or hashes.get("audit.json") != sha256_file(audit_path)
        or any(
            not isinstance(name, str)
            or not isinstance(declared, str)
            or not (args.processed_root / name).is_file()
            or sha256_file(args.processed_root / name) != declared
            for name, declared in hashes.items()
        )
    ):
        return False
    return (
        audit.get("status") == "PASS"
        and audit.get("design_protocol")
        == "v5_3_task_level_cross_seed_sft_screen"
        and audit.get("design_version") == "5.3"
        and audit.get("attempts_per_task_per_condition")
        == GENERATION_ATTEMPTS_PER_CONDITION
        and audit.get("official_test_used") is False
        and audit.get("derived_validation_used_for_supervision") is False
        and audit.get("ground_truth_incompatible_task_ids")
        == list(GT_INCOMPATIBLE_TASK_IDS)
        and (audit.get("generation_contracts") or {}).get("task_union") == 78
        and audit.get("cross_arm_supervised_token_relative_range", 1.0)
        <= 0.01
        and audit.get("cross_arm_nonpadding_token_relative_range", 1.0)
        <= 0.02
        and (audit.get("formal_data_gate") or {}).get(
            "observed_distinct_train_tasks", 0
        )
        >= 40
        and (audit.get("formal_data_gate") or {}).get(
            "observed_train_pairs", 0
        )
        >= 48
        and all(
            (args.processed_root / "arms" / arm / "train.jsonl").is_file()
            for arm in ARMS
        )
        and (args.processed_root / "validation_loss.jsonl").is_file()
        and (args.processed_root / "validation_manifest.json").is_file()
    )


def prepare_data(args: argparse.Namespace) -> None:
    if data_complete(args):
        print("[skip] V5.3 data constructor audit is PASS")
        return
    if args.processed_root.exists():
        if any(args.processed_root.iterdir()):
            raise StageError(
                f"partial processed data must be archived, not overwritten: "
                f"{args.processed_root}"
            )
        # The builder atomically creates its output root. Removing a verified
        # empty directory preserves that contract and cannot discard evidence.
        args.processed_root.rmdir()
    if not generation_complete(args):
        raise StageError("generation barrier is incomplete")
    files = protocol_files(args)
    commit = source_commit()
    run_checked(
        [
            str(args.train_python),
            "scripts/prepare_v5_3_sft_causal.py",
            "--split-manifest",
            "artifacts/v5_stage0/manifests/split_manifest.json",
            "--generation-manifest",
            str(files["generation_manifest"]),
            "--validation-manifest",
            str(files["evaluation_manifest"]),
            "--generation-dynamic-audit",
            str(files["generation_audit"]),
            "--validation-dynamic-audit",
            str(files["evaluation_audit"]),
            "--raw-dir",
            str(args.raw_root),
            "--tau2-root",
            str(args.tau2_root),
            "--output-dir",
            str(args.processed_root),
            "--tokenizer",
            STUDENT_MODEL,
            "--tokenizer-revision",
            STUDENT_REVISION,
            "--expected-source-commit",
            commit,
            "--expected-generation-source-commit",
            commit,
            "--local-files-only",
        ],
        env=base_env(offline=True),
        log_path=args.results_root / "logs" / "prepare-data.log",
    )
    if not data_complete(args):
        raise StageError("V5.3 data constructor did not publish PASS audit")


def _read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageError(f"invalid JSON artifact: {path}") from error


def _recompute_training_loss_audit(path: Path) -> dict[str, Any]:
    import math

    history = _read_json_value(path / "training_log.json")
    metrics = _read_json_value(path / "training_metrics.json")
    if not isinstance(history, list) or any(
        not isinstance(row, dict) for row in history
    ):
        raise StageError(f"invalid training log: {path / 'training_log.json'}")
    if not isinstance(metrics, dict):
        raise StageError(
            f"invalid training metrics: {path / 'training_metrics.json'}"
        )
    loss_values: list[float] = []
    grad_norms: list[float] = []
    eval_losses: list[float] = []
    numeric_values = 0
    for record in [*history, metrics]:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise StageError(f"non-finite training metric {key}")
                numeric_values += 1
                if "loss" in key.lower():
                    loss_values.append(numeric)
                if key == "grad_norm":
                    grad_norms.append(numeric)
                if key == "eval_loss":
                    eval_losses.append(numeric)
    train_loss = metrics.get("train_loss")
    if (
        not loss_values
        or not grad_norms
        or not eval_losses
        or isinstance(train_loss, bool)
        or not isinstance(train_loss, (int, float))
        or not math.isfinite(float(train_loss))
    ):
        raise StageError("training loss/gradient/validation audit is incomplete")
    return {
        "finite": True,
        "numeric_values_checked": numeric_values,
        "loss_values_checked": len(loss_values),
        "grad_norm_values_checked": len(grad_norms),
        "validation_loss_values_checked": len(eval_losses),
        "final_train_loss": float(train_loss),
        "final_validation_loss": eval_losses[-1],
    }


def _checkpoint_fingerprint(path: Path) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        target = path / name
        if not target.is_file() or target.stat().st_size <= 0:
            raise StageError(f"incomplete adapter checkpoint: {target}")
        observed = sha256_file(target)
        files[name] = observed
        digest.update(name.encode("utf-8"))
        digest.update(observed.encode("ascii"))
    return digest.hexdigest(), files


def _validate_training_run(
    args: argparse.Namespace,
    path: Path,
    *,
    arm: str,
    mode: str,
) -> None:
    if arm not in ARMS or mode not in {"smoke", "formal"}:
        raise StageError(f"invalid training identity: {arm}/{mode}")
    expected_path = (args.results_root / arm / mode).resolve()
    if path.resolve() != expected_path or not data_complete(args):
        raise StageError(f"training path/data barrier drift: {path}")
    manifest_path = path / "run_manifest.json"
    value = read_json(manifest_path)
    expected = {
        "protocol": "v5_stage1_message_masked_sft_7b",
        "source_commit": source_commit(),
        "arm": arm,
        "mode": mode,
        "objective": "message_masked_causal_language_model_cross_entropy",
        "model": STUDENT_MODEL,
        "model_revision": STUDENT_REVISION,
        "seed": SEED,
        "max_sequence_tokens": 8192,
        "truncation": False,
        "held_out_test_accessed": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise StageError(f"training manifest identity drift: {manifest_path}")

    hashes_path = args.processed_root / "hashes.json"
    audit_path = args.processed_root / "audit.json"
    hashes = read_json(hashes_path)
    train_path = args.processed_root / "arms" / arm / "train.jsonl"
    validation_path = args.processed_root / "validation_loss.jsonl"
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    if (
        hashes.get(f"arms/{arm}/train.jsonl") != train_sha
        or hashes.get("validation_loss.jsonl") != validation_sha
        or value.get("train_file_sha256") != train_sha
        or value.get("validation_file_sha256") != validation_sha
    ):
        raise StageError(f"training input hash drift: {manifest_path}")
    provenance = value.get("data_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("data_audit_sha256") != sha256_file(audit_path)
        or provenance.get("data_hashes_sha256") != sha256_file(hashes_path)
        or provenance.get("train_file_sha256") != train_sha
        or provenance.get("validation_file_sha256") != validation_sha
        or provenance.get("design_version") != "5.3"
        or not isinstance(provenance.get("design_provenance"), dict)
        or provenance.get("official_test_used") is not False
        or provenance.get("official_test_sealed") is not True
    ):
        raise StageError(f"training data provenance drift: {manifest_path}")

    checkpoint_path = path / "checkpoint_final"
    fingerprint, checkpoint_files = _checkpoint_fingerprint(checkpoint_path)
    adapter_config = read_json(checkpoint_path / "adapter_config.json")
    checkpoint = value.get("checkpoint")
    if (
        adapter_config.get("peft_type") != "LORA"
        or not isinstance(checkpoint, dict)
        or checkpoint.get("fingerprint") != fingerprint
        or checkpoint.get("file_sha256") != checkpoint_files
    ):
        raise StageError(f"training checkpoint identity drift: {manifest_path}")
    if value.get("loss_audit") != _recompute_training_loss_audit(path):
        raise StageError(f"training loss audit drift: {manifest_path}")


def training_run_complete(
    args: argparse.Namespace,
    path: Path,
    *,
    arm: str,
    mode: str,
) -> bool:
    try:
        _validate_training_run(args, path, arm=arm, mode=mode)
    except (OSError, ValueError, TypeError, StageError):
        return False
    return True


def train_command(
    args: argparse.Namespace,
    *,
    arm: str,
    mode: str,
    train_sha: str,
    validation_sha: str,
) -> list[str]:
    return [
        str(args.train_python),
        "scripts/train_v5_sft_causal.py",
        "--train-file",
        str(args.processed_root / "arms" / arm / "train.jsonl"),
        "--validation-file",
        str(args.processed_root / "validation_loss.jsonl"),
        "--data-audit",
        str(args.processed_root / "audit.json"),
        "--data-hashes",
        str(args.processed_root / "hashes.json"),
        "--output-dir",
        str(args.results_root / arm / mode),
        "--arm",
        arm,
        "--mode",
        mode,
        "--model-revision",
        STUDENT_REVISION,
        "--expected-source-commit",
        source_commit(),
        "--expected-train-sha256",
        train_sha,
        "--expected-validation-sha256",
        validation_sha,
        "--local-files-only",
    ]


def run_parallel_training_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    hashes: dict[str, Any],
) -> None:
    workers: list[tuple[subprocess.Popen[bytes], Any, str]] = []
    try:
        for gpu, arm in enumerate(ARMS):
            output = args.results_root / arm / mode
            if training_run_complete(
                args, output, arm=arm, mode=mode
            ):
                print(f"[skip] {arm} {mode} is complete")
                continue
            if output.exists() and any(output.iterdir()):
                raise StageError(
                    f"partial {arm} {mode} output must be archived: {output}"
                )
            command = train_command(
                args,
                arm=arm,
                mode=mode,
                train_sha=str(hashes[f"arms/{arm}/train.jsonl"]),
                validation_sha=str(hashes["validation_loss.jsonl"]),
            )
            env = base_env(offline=True, gpu=gpu)
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            process, handle = spawn_logged(
                command,
                env=env,
                log_path=args.results_root
                / "logs"
                / f"train-{mode}-{arm}-gpu{gpu}.log",
            )
            workers.append((process, handle, arm))
        failures = []
        for process, _, arm in workers:
            code = process.wait()
            if code:
                failures.append((arm, code))
        if failures:
            raise StageError(f"{mode} training failures: {failures}")
    finally:
        for process, handle, _ in workers:
            if process.poll() is None:
                stop_process(process)
            handle.close()
    missing = [
        arm
        for arm in ARMS
        if not training_run_complete(
            args,
            args.results_root / arm / mode,
            arm=arm,
            mode=mode,
        )
    ]
    if missing:
        raise StageError(f"{mode} training outputs incomplete: {missing}")


def train_arms(args: argparse.Namespace) -> None:
    if not preflight_complete(args):
        raise StageError("training requires a current immutable preflight")
    if not data_complete(args):
        raise StageError("data audit barrier is incomplete")
    hashes = read_json(args.processed_root / "hashes.json")
    required = {
        "validation_loss.jsonl",
        *(f"arms/{arm}/train.jsonl" for arm in ARMS),
    }
    missing = required - set(hashes)
    if missing:
        raise StageError(f"data hash manifest lacks: {sorted(missing)}")
    run_parallel_training_mode(args, mode="smoke", hashes=hashes)
    run_parallel_training_mode(args, mode="formal", hashes=hashes)


def _validate_registry(args: argparse.Namespace) -> None:
    path = args.results_root / "checkpoint_registry.json"
    if not path.is_file() or not data_complete(args):
        raise StageError("checkpoint registry/data barrier is incomplete")
    value = read_json(path)
    audit_path = args.processed_root / "audit.json"
    hashes_path = args.processed_root / "hashes.json"
    provenance_fields = (
        "data_audit_sha256",
        "data_hashes_sha256",
        "dynamic_audits",
        "design_version",
        "design_provenance",
        "official_test_used",
        "official_test_sealed",
    )
    formal_provenance: list[dict[str, Any]] = []
    for arm in ARMS:
        run_path = args.results_root / arm / "formal"
        if not training_run_complete(
            args, run_path, arm=arm, mode="formal"
        ):
            raise StageError(f"registry references incomplete training: {arm}")
        run_manifest = read_json(run_path / "run_manifest.json")
        row = run_manifest.get("data_provenance")
        if not isinstance(row, dict):
            raise StageError(f"training provenance absent: {arm}")
        formal_provenance.append(
            {field: row.get(field) for field in provenance_fields}
        )
    if any(row != formal_provenance[0] for row in formal_provenance[1:]):
        raise StageError("formal arms do not share one data provenance")
    expected_provenance = formal_provenance[0]
    if (
        expected_provenance["data_audit_sha256"]
        != sha256_file(audit_path)
        or expected_provenance["data_hashes_sha256"]
        != sha256_file(hashes_path)
        or expected_provenance["design_version"] != "5.3"
        or not isinstance(expected_provenance["design_provenance"], dict)
        or expected_provenance["official_test_used"] is not False
        or expected_provenance["official_test_sealed"] is not True
    ):
        raise StageError("formal registry provenance drift")
    design = expected_provenance["design_provenance"]
    dynamic = expected_provenance["dynamic_audits"]
    if (
        not isinstance(design, dict)
        or any(
            design.get(field) != expected
            for field, expected in {
                "design_version": "5.3",
                "design_protocol": (
                    "v5_3_task_level_cross_seed_sft_screen"
                ),
                "effective_generation_tasks": 78,
                "validation_tasks": 21,
            }.items()
        )
        or not isinstance(dynamic, dict)
        or (dynamic.get("generation") or {}).get("verified_injections")
        != 78
        or (dynamic.get("validation") or {}).get("verified_injections")
        != 21
    ):
        raise StageError("V5.3 registry profile identity drift")
    entries = value.get("entries")
    if (
        value.get("protocol") != "v5_stage1_checkpoint_registry"
        or value.get("provenance_profile") != "v5_3"
        or value.get("source_commit") != source_commit()
        or value.get("base_model_revision") != STUDENT_REVISION
        or value.get("training_data_provenance") != expected_provenance
        or not isinstance(entries, dict)
        or set(entries) != set(EVAL_ARMS)
    ):
        raise StageError("checkpoint registry metadata drift")
    if entries.get("base_model") != {
        "model_id": EVAL_MODEL_IDS["base_model"],
        "adapter_sha256": None,
        "adapter_config_sha256": None,
        "training_run_manifest_sha256": None,
    }:
        raise StageError("base-model registry entry drift")
    for arm in ARMS:
        run_path = args.results_root / arm / "formal"
        checkpoint = run_path / "checkpoint_final"
        expected_entry = {
            "model_id": EVAL_MODEL_IDS[arm],
            "adapter_sha256": sha256_file(
                checkpoint / "adapter_model.safetensors"
            ),
            "adapter_config_sha256": sha256_file(
                checkpoint / "adapter_config.json"
            ),
            "training_run_manifest_sha256": sha256_file(
                run_path / "run_manifest.json"
            ),
        }
        if entries.get(arm) != expected_entry:
            raise StageError(f"checkpoint registry entry drift: {arm}")


def registry_complete(args: argparse.Namespace) -> bool:
    try:
        _validate_registry(args)
    except (KeyError, OSError, ValueError, TypeError, StageError):
        return False
    return True


def build_registry(args: argparse.Namespace) -> None:
    if registry_complete(args):
        print("[skip] checkpoint registry is complete")
        return
    output = args.results_root / "checkpoint_registry.json"
    if output.exists():
        raise StageError(f"partial/invalid registry must be archived: {output}")
    for arm in ARMS:
        if not training_run_complete(
            args,
            args.results_root / arm / "formal",
            arm=arm,
            mode="formal",
        ):
            raise StageError(f"formal arm is incomplete: {arm}")
    command = [
        str(args.train_python),
        "scripts/build_v5_checkpoint_registry.py",
        "--provenance-profile",
        "v5_3",
        "--source-commit",
        source_commit(),
        "--base-revision",
        STUDENT_REVISION,
    ]
    for arm in ARMS:
        command += [
            "--arm",
            f"{arm}={args.results_root / arm / 'formal'}",
        ]
    command += ["--output", str(output)]
    run_checked(
        command,
        env=base_env(offline=True),
        log_path=args.results_root / "logs" / "checkpoint-registry.log",
    )
    if not registry_complete(args):
        raise StageError("checkpoint registry validation failed")


def start_evaluation_services(
    args: argparse.Namespace,
) -> list[tuple[subprocess.Popen[bytes], Any, Path]]:
    services = []
    try:
        for gpu in range(4):
            port = 8100 + gpu
            if not port_is_free(port):
                raise StageError(f"evaluation port is occupied: {port}")
            command = [
                *vllm_base_command(
                    args,
                    STUDENT_MODEL,
                    STUDENT_REVISION,
                    dtype="bfloat16",
                ),
                "--served-model-name",
                EVAL_MODEL_IDS["base_model"].removeprefix("openai/"),
                "--port",
                str(port),
                "--api-key",
                "stage1-eval-local",
                "--gpu-memory-utilization",
                "0.90",
                "--max-num-seqs",
                "1",
                "--enable-lora",
                "--max-lora-rank",
                "16",
                "--max-loras",
                "1",
                "--max-cpu-loras",
                "4",
                "--lora-modules",
                *[
                    f"{EVAL_MODEL_IDS[arm].removeprefix('openai/')}="
                    f"{args.results_root / arm / 'formal' / 'checkpoint_final'}"
                    for arm in ARMS
                ],
            ]
            process, handle = spawn_logged(
                command,
                env=base_env(offline=True, gpu=gpu),
                log_path=args.results_root
                / "logs"
                / f"evaluation-server-gpu{gpu}.log",
            )
            pid_path = (
                args.results_root / "pids" / f"evaluation-server-gpu{gpu}.pid"
            )
            write_pid(pid_path, process)
            services.append((process, handle, pid_path))
        for gpu, (process, _, _) in enumerate(services):
            wait_for_service(
                port=8100 + gpu,
                api_key="stage1-eval-local",
                expected_models=set(SERVED_ALIASES),
                process=process,
                timeout_seconds=args.health_timeout,
            )
        return services
    except Exception:
        stop_services(services)
        raise


def _evaluation_contract_core_sha256(value: dict[str, Any]) -> str:
    mutable = {
        "status",
        "completed_at",
        "result_sha256",
        "completion_audit",
        "strict_judge_audit_evidence",
        "contract_core_sha256",
    }
    core = {key: row for key, row in value.items() if key not in mutable}
    encoded = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_evaluation_shard(
    args: argparse.Namespace, *, arm: str, shard: int
) -> None:
    try:
        from run_v5_sft_causal_eval import (
            _expected_result_task_ids,
            audit_result_interface,
        )
    except ModuleNotFoundError:
        from scripts.run_v5_sft_causal_eval import (
            _expected_result_task_ids,
            audit_result_interface,
        )

    path = (
        args.results_root
        / "evaluation"
        / arm
        / f"run_contract.shard-{shard:03d}-of-004.json"
    )
    if (
        not path.is_file()
        or not (args.processed_root / "validation_manifest.json").is_file()
        or not registry_complete(args)
    ):
        raise StageError(f"evaluation barrier is incomplete: {path}")
    value = read_json(path)
    registry_path = args.results_root / "checkpoint_registry.json"
    registry = read_json(registry_path)
    manifest_path = args.processed_root / "validation_manifest.json"
    manifest = read_json(manifest_path)
    result_hashes = value.get("result_sha256")
    completion = value.get("completion_audit")
    task_ids = value.get("task_ids")
    preflight = value.get("runtime_preflight")
    endpoint = f"http://127.0.0.1:{8100 + shard}/v1"
    decoding = {
        "temperature": 0,
        "max_tokens": 512,
        "max_steps": 60,
        "task_timeout_seconds": 900.0,
        "seed": SEED,
        "num_trials": 1,
    }
    expected_preflight = {
            "served_context_window_tokens": MAX_MODEL_LEN,
            "request_max_tokens": 512,
            "maximum_nonoverflow_prompt_tokens": MAX_MODEL_LEN - 512,
            "max_steps": 60,
            "request_token_overflow_policy": "fail_closed",
            "longest_prompt_observation": (
                "completion_audit.max_observed_prompt_tokens"
            ),
    }
    strict_judge = {
        "module": "v5_strict_nl_judge",
        "entrypoint": "install_strict_nl_judge",
        "mode": "strict_json_schema_fail_closed",
    }
    expected_local_adapters = {
        trained_arm: {
            "adapter_sha256": registry["entries"][trained_arm][
                "adapter_sha256"
            ],
            "adapter_config_sha256": registry["entries"][trained_arm][
                "adapter_config_sha256"
            ],
        }
        for trained_arm in ARMS
    }
    expected_aliases = sorted(
        entry["model_id"].removeprefix("openai/")
        for entry in registry["entries"].values()
    )
    if (
        value.get("status") != "COMPLETE"
        or value.get("protocol")
        != "v5_stage1_sft_causal_validation_run"
        or value.get("arm") != arm
        or value.get("num_shards") != EVALUATION_SHARDS
        or value.get("shard_index") != shard
        or value.get("official_test_used") is not False
        or value.get("source_commit") != source_commit()
        or value.get("base_model_revision") != STUDENT_REVISION
        or value.get("tool_action_interface")
        != EVAL_TOOL_ACTION_INTERFACE
        or value.get("contract_core_sha256")
        != _evaluation_contract_core_sha256(value)
        or preflight != expected_preflight
        or value.get("decoding") != decoding
        or value.get("conditions") != ["clean", "error"]
        or not isinstance(task_ids, list)
        or not task_ids
        or len(task_ids) != len(set(task_ids))
        or value.get("checkpoint_registry_protocol")
        != "v5_stage1_checkpoint_registry"
        or value.get("checkpoint_registry_provenance_profile") != "v5_3"
        or value.get("checkpoint_registry_sha256")
        != sha256_file(registry_path)
        or value.get("checkpoint_entry") != registry["entries"][arm]
        or value.get("locally_verified_adapter_identity")
        != expected_local_adapters
        or value.get("served_registry_aliases") != expected_aliases
        or value.get("evaluation_manifest_protocol")
        != manifest.get("protocol")
        or value.get("evaluation_manifest_sha256")
        != sha256_file(manifest_path)
        or value.get("split_manifest_sha256") != SPLIT_SHA256
        or value.get("dynamic_audit_identity")
        != registry["training_data_provenance"]["dynamic_audits"][
            "validation"
        ]
        or value.get("agent")
        != {
            "model": EVAL_MODEL_IDS[arm],
            "revision": STUDENT_REVISION,
            "api_base": endpoint,
        }
        or value.get("user")
        != {
            "model": EVAL_MODEL_IDS["base_model"],
            "revision": STUDENT_REVISION,
            "api_base": endpoint,
        }
        or value.get("judge")
        != {
            "model": EVAL_MODEL_IDS["base_model"],
            "revision": STUDENT_REVISION,
            "api_base": endpoint,
            "strict_backend": strict_judge,
        }
    ):
        raise StageError(f"evaluation contract metadata drift: {path}")
    if not isinstance(result_hashes, dict) or not isinstance(completion, dict):
        raise StageError(f"evaluation completion metadata absent: {path}")
    try:
        expected_results = _expected_result_task_ids(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise StageError(f"evaluation result schedule drift: {path}") from error
    if set(result_hashes) != set(expected_results):
        raise StageError(f"evaluation result declaration drift: {path}")
    recomputed: dict[str, dict[str, Any]] = {}
    result_paths: list[Path] = []
    for name, expected_tasks in expected_results.items():
        result_path = path.parent / name
        if (
            Path(name).name != name
            or not result_path.is_file()
            or sha256_file(result_path) != result_hashes.get(name)
        ):
            raise StageError(f"evaluation result hash/path drift: {result_path}")
        result_paths.append(result_path.resolve())
        try:
            recomputed[name] = audit_result_interface(
                result_path,
                expected_task_ids=expected_tasks,
                num_trials=decoding["num_trials"],
                max_tokens=decoding["max_tokens"],
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise StageError(
                f"evaluation result semantic audit failed: {result_path}"
            ) from error
    expected_completion = {
        "protocol": "v5_stage1_sft_causal_interface_completion_audit",
        "tool_action_interface": EVAL_TOOL_ACTION_INTERFACE,
        "result_files": dict(sorted(recomputed.items())),
        "max_observed_prompt_tokens": max(
            row["max_observed_prompt_tokens"] for row in recomputed.values()
        ),
        "max_observed_total_request_tokens": max(
            row["max_observed_total_request_tokens"]
            for row in recomputed.values()
        ),
        "all_expected_tasks_observed_once_per_trial": True,
        "request_token_overflow_detected": False,
        "status": "PASS",
    }
    if completion != expected_completion:
        raise StageError(f"evaluation completion audit drift: {path}")
    try:
        observed_evidence = (
            judge_audit_contract.validate_strict_judge_evidence(
                result_paths,
                maximum_content_attempts=2,
            )
        )
    except judge_audit_contract.StrictJudgeEvidenceError as error:
        raise StageError(
            f"evaluation strict-judge evidence is incomplete: {path}"
        ) from error
    if value.get("strict_judge_audit_evidence") != observed_evidence:
        raise StageError(
            f"evaluation strict-judge evidence contract drift: {path}"
        )


def evaluation_shard_complete(
    args: argparse.Namespace, *, arm: str, shard: int
) -> bool:
    try:
        _validate_evaluation_shard(args, arm=arm, shard=shard)
    except (
        ImportError,
        KeyError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        StageError,
    ):
        return False
    return True


def evaluation_arm_complete(args: argparse.Namespace, arm: str) -> bool:
    task_union: set[str] = set()
    for shard in range(EVALUATION_SHARDS):
        if not evaluation_shard_complete(args, arm=arm, shard=shard):
            return False
        contract = read_json(
            args.results_root
            / "evaluation"
            / arm
            / f"run_contract.shard-{shard:03d}-of-004.json"
        )
        tasks = set(contract["task_ids"])
        if task_union & tasks:
            return False
        task_union.update(tasks)
    manifest = read_json(args.processed_root / "validation_manifest.json")
    expected = {
        f"{row.get('domain')}:{row.get('task_id')}"
        for row in manifest.get("rows", [])
        if isinstance(row, dict)
    }
    return len(expected) == 21 and task_union == expected


def evaluation_command(
    args: argparse.Namespace, *, arm: str, shard: int
) -> list[str]:
    command = [
        str(args.serve_python),
        "scripts/run_v5_sft_causal_eval.py",
        "--provenance-profile",
        "v5_3",
        "--tau2-root",
        str(args.tau2_root),
        "--split-manifest",
        "artifacts/v5_stage0/manifests/split_manifest.json",
        "--manifest",
        str(args.processed_root / "validation_manifest.json"),
        "--dynamic-audit",
        str(protocol_files(args)["evaluation_audit"]),
        "--checkpoint-registry",
        str(args.results_root / "checkpoint_registry.json"),
    ]
    for trained_arm in ARMS:
        command += [
            "--adapter-dir",
            f"{trained_arm}="
            f"{args.results_root / trained_arm / 'formal' / 'checkpoint_final'}",
        ]
    endpoint = f"http://127.0.0.1:{8100 + shard}/v1"
    command += [
        "--output-dir",
        str(args.results_root / "evaluation" / arm),
        "--arm",
        arm,
        "--agent-model",
        EVAL_MODEL_IDS[arm],
        "--agent-api-base",
        endpoint,
        "--agent-api-key",
        "stage1-eval-local",
        "--user-model",
        EVAL_MODEL_IDS["base_model"],
        "--user-api-base",
        endpoint,
        "--user-api-key",
        "stage1-eval-local",
        "--judge-model",
        EVAL_MODEL_IDS["base_model"],
        "--judge-api-base",
        endpoint,
        "--judge-api-key",
        "stage1-eval-local",
        "--condition",
        "both",
        "--shard-index",
        str(shard),
        "--num-shards",
        str(EVALUATION_SHARDS),
        "--max-steps",
        "60",
        "--timeout",
        "900",
        "--max-tokens",
        "512",
        "--seed",
        str(SEED),
        "--num-trials",
        "1",
    ]
    return command


def evaluate_one_shard(args: argparse.Namespace, shard: int) -> None:
    for arm in EVAL_ARMS:
        if evaluation_shard_complete(args, arm=arm, shard=shard):
            print(f"[skip] evaluation {arm} shard {shard} is complete")
            continue
        contract = (
            args.results_root
            / "evaluation"
            / arm
            / f"run_contract.shard-{shard:03d}-of-004.json"
        )
        if contract.exists():
            raise StageError(
                f"partial evaluation must be archived: {contract.parent}"
            )
        run_checked(
            evaluation_command(args, arm=arm, shard=shard),
            env=base_env(offline=True),
            log_path=args.results_root
            / "logs"
            / f"evaluate-{arm}-shard-{shard}.log",
        )


def evaluate(args: argparse.Namespace) -> None:
    if not preflight_complete(args):
        raise StageError("evaluation requires a current immutable preflight")
    if not registry_complete(args):
        raise StageError("checkpoint registry barrier is incomplete")
    services = start_evaluation_services(args)
    try:
        failures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(evaluate_one_shard, args, shard): shard
                for shard in range(4)
            }
            for future in as_completed(futures):
                shard = futures[future]
                try:
                    future.result()
                except Exception as error:
                    failures.append((shard, repr(error)))
        if failures:
            raise StageError(f"evaluation shard failures: {failures}")
    finally:
        stop_services(services)
    missing = [
        arm
        for arm in EVAL_ARMS
        if not evaluation_arm_complete(args, arm)
    ]
    if missing:
        raise StageError(f"evaluation contracts incomplete: {missing}")


def _validate_summary(args: argparse.Namespace) -> None:
    import tempfile

    # Reuse an already loaded canonical package module when present.  Importing
    # the same source again under a top-level name creates split mock/state
    # identities in long-lived orchestrators and test discovery.
    package_summary = sys.modules.get("scripts.summarize_v5_sft_causal")
    if package_summary is not None:
        recompute_summary = package_summary.summarize
    else:
        try:
            from summarize_v5_sft_causal import summarize as recompute_summary
        except ModuleNotFoundError:
            from scripts.summarize_v5_sft_causal import (
                summarize as recompute_summary,
            )

    output = args.results_root / "mechanism_screen_summary.json"
    if not output.is_file() or not all(
        evaluation_arm_complete(args, arm) for arm in EVAL_ARMS
    ):
        raise StageError("summary/evaluation barrier is incomplete")
    value = read_json(output)
    provenance = value.get("provenance")
    if (
        value.get("protocol") != "v5_stage1_sft_causal_validation"
        or value.get("status") != "PASS"
        or value.get("summary_contract_protocol")
        != "v5_stage1_sft_causal_validation_summary_v2"
        or not isinstance(provenance, dict)
    ):
        raise StageError("summary protocol/status drift")
    declared_binding = provenance.get("input_binding_sha256")
    binding_payload = {
        key: row
        for key, row in provenance.items()
        if key != "input_binding_sha256"
    }
    observed_binding = hashlib.sha256(
        json.dumps(
            binding_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifests = provenance.get("manifests") or {}
    source = provenance.get("source") or {}
    if (
        declared_binding != observed_binding
        or source.get("experiment_source_commit") != source_commit()
        or source.get("base_model_revision") != STUDENT_REVISION
        or manifests.get("split_sha256") != SPLIT_SHA256
        or manifests.get("evaluation_sha256")
        != sha256_file(args.processed_root / "validation_manifest.json")
        or manifests.get("dynamic_audit_sha256")
        != sha256_file(protocol_files(args)["evaluation_audit"])
        or manifests.get("checkpoint_registry_sha256")
        != sha256_file(args.results_root / "checkpoint_registry.json")
        or provenance.get("completion_audits_recomputed") is not True
        or provenance.get("tool_action_interface")
        != EVAL_TOOL_ACTION_INTERFACE
    ):
        raise StageError("summary provenance drift")
    declared_contracts = provenance.get("evaluation_contract_sha256")
    if not isinstance(declared_contracts, dict):
        raise StageError("summary lacks evaluation contract hashes")
    for arm in EVAL_ARMS:
        arm_hashes = declared_contracts.get(arm)
        if not isinstance(arm_hashes, dict) or len(arm_hashes) != 4:
            raise StageError(f"summary contract hash set drift: {arm}")
        observed = {
            path.name: sha256_file(path)
            for path in sorted(
                (args.results_root / "evaluation" / arm).glob(
                    "run_contract.shard-*-of-004.json"
                )
            )
        }
        if arm_hashes != observed:
            raise StageError(f"summary contract hashes drift: {arm}")

    arm_dirs = {
        arm: (args.results_root / "evaluation" / arm).resolve()
        for arm in EVAL_ARMS
    }
    with tempfile.TemporaryDirectory(prefix="v5-3-summary-recompute-") as temp:
        try:
            expected = recompute_summary(
                split_manifest_path=(
                    ROOT
                    / "artifacts/v5_stage0/manifests/split_manifest.json"
                ),
                evaluation_manifest_path=(
                    args.processed_root / "validation_manifest.json"
                ),
                dynamic_audit_path=protocol_files(args)["evaluation_audit"],
                checkpoint_registry_path=(
                    args.results_root / "checkpoint_registry.json"
                ),
                arm_dirs=arm_dirs,
                output_path=Path(temp) / "summary.json",
                provenance_profile="v5_3",
            )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            raise StageError("summary scientific recomputation failed") from error
    if value != expected:
        raise StageError("summary scientific fields/metrics drift")


def summary_complete(args: argparse.Namespace) -> bool:
    try:
        _validate_summary(args)
    except (
        ImportError,
        KeyError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        StageError,
    ):
        return False
    return True


def summarize(args: argparse.Namespace) -> None:
    output = args.results_root / "mechanism_screen_summary.json"
    if summary_complete(args):
        print("[skip] V5.3 mechanism screen summary is hash-bound PASS")
        return
    if output.exists():
        raise StageError(f"invalid existing summary must be archived: {output}")
    command = [
        str(args.train_python),
        "scripts/summarize_v5_sft_causal.py",
        "--provenance-profile",
        "v5_3",
        "--split-manifest",
        "artifacts/v5_stage0/manifests/split_manifest.json",
        "--evaluation-manifest",
        str(args.processed_root / "validation_manifest.json"),
        "--dynamic-audit",
        str(protocol_files(args)["evaluation_audit"]),
        "--checkpoint-registry",
        str(args.results_root / "checkpoint_registry.json"),
    ]
    for arm in EVAL_ARMS:
        command += [
            "--arm",
            f"{arm}={args.results_root / 'evaluation' / arm}",
        ]
    command += ["--output", str(output)]
    run_checked(
        command,
        env=base_env(offline=True),
        log_path=args.results_root / "logs" / "summarize.log",
    )
    if not summary_complete(args):
        raise StageError("summary protocol/provenance validation failed")


def status(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "preflight_complete": preflight_complete(args),
        "protocol_complete": protocol_complete(args),
        "pilot_manifest_complete": pilot_manifest_complete(args),
        "pilot_runtime_preflight_complete": (
            generation_runtime_preflight_complete(args, phase="pilot")
        ),
        "pilot_runtime_contract_binding_complete": (
            generation_runtime_contract_binding_complete(
                args, phase="pilot"
            )
        ),
        "pilot_status": pilot_status(args),
        "formal_generation_authorized": (
            pilot_status(args) == "GO_FORMAL_GENERATION"
        ),
        "formal_runtime_preflight_complete": (
            generation_runtime_preflight_complete(args, phase="formal")
        ),
        "formal_runtime_contract_binding_complete": (
            generation_runtime_contract_binding_complete(
                args, phase="formal"
            )
        ),
        "generation_complete": generation_complete(args),
        "data_complete": data_complete(args),
        "training": {
            mode: {
                arm: training_run_complete(
                    args,
                    args.results_root / arm / mode,
                    arm=arm,
                    mode=mode,
                )
                for arm in ARMS
            }
            for mode in ("smoke", "formal")
        },
        "registry_complete": registry_complete(args)
        if (args.results_root / "checkpoint_registry.json").exists()
        else False,
        "evaluation_complete": {
            arm: [
                evaluation_shard_complete(args, arm=arm, shard=shard)
                for shard in range(4)
            ]
            for arm in EVAL_ARMS
        },
        "summary_complete": summary_complete(args)
        if (args.results_root / "mechanism_screen_summary.json").is_file()
        else False,
        "official_test_used": False,
    }
    print(json.dumps(payload, indent=2))
    return payload


def stop_stale_services(args: argparse.Namespace) -> None:
    pid_root = args.results_root / "pids"
    if not pid_root.is_dir():
        print("[stop] no V5.3 PID directory")
        return
    for path in sorted(pid_root.glob("*.pid")):
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise StageError(f"invalid PID file: {path}") from error
        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        if not proc_cmdline.is_file():
            path.unlink()
            continue
        command = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
        if "vllm" not in command:
            raise StageError(
                f"refusing to signal PID {pid}; it is not a vLLM process: {command}"
            )
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        path.unlink()
        print(f"[stop] signalled V5.3 service PID {pid}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "preflight",
            "self-test",
            "prefetch",
            "protocol",
            "pilot",
            "generate",
            "prepare",
            "train",
            "registry",
            "evaluate",
            "summarize",
            "all",
            "status",
            "stop-services",
        ),
    )
    parser.add_argument(
        "--tau2-root",
        type=Path,
        default=ROOT / "data/raw/tau2-bench",
    )
    parser.add_argument(
        "--serve-venv",
        type=Path,
        default=Path("/workspace/venvs/v5-2-serve"),
    )
    parser.add_argument(
        "--train-venv",
        type=Path,
        default=Path("/workspace/venvs/v5-2-train"),
    )
    parser.add_argument(
        "--workspace-root", type=Path, default=Path("/workspace")
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "data/processed/v5_3_protocol",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw/v5_3_sft_causal_generation",
    )
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=ROOT / "artifacts/v5_3_train_only_pilot",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=ROOT / "data/processed/v5_3_sft_causal",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/v5_3_sft_causal",
    )
    parser.add_argument("--health-timeout", type=int, default=1800)
    args = parser.parse_args()
    args.serve_venv = args.serve_venv.expanduser().resolve()
    args.train_venv = args.train_venv.expanduser().resolve()
    args.tau2_root = args.tau2_root.expanduser().resolve()
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.protocol_root = args.protocol_root.expanduser().resolve()
    args.pilot_root = args.pilot_root.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.processed_root = args.processed_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.serve_python = args.serve_venv / "bin/python"
    args.train_python = args.train_venv / "bin/python"
    args.vllm = args.serve_venv / "bin/vllm"
    return args


def main() -> None:
    args = parse_args()
    stages = {
        "preflight": lambda: preflight(args),
        "self-test": lambda: self_test(args),
        "prefetch": lambda: prefetch_models(args),
        "protocol": lambda: build_protocol(args),
        "pilot": lambda: run_pilot(args),
        "generate": lambda: generate_trajectories(args),
        "prepare": lambda: prepare_data(args),
        "train": lambda: train_arms(args),
        "registry": lambda: build_registry(args),
        "evaluate": lambda: evaluate(args),
        "summarize": lambda: summarize(args),
        "status": lambda: status(args),
        "stop-services": lambda: stop_stale_services(args),
    }
    try:
        if args.stage == "all":
            preflight(args)
            self_test(args)
            prefetch_models(args)
            build_protocol(args)
            run_pilot(args)
            generate_trajectories(args)
            prepare_data(args)
            train_arms(args)
            build_registry(args)
            evaluate(args)
            summarize(args)
            status(args)
        else:
            stages[args.stage]()
    except PilotNoGo as error:
        print(f"V5.3 stopped by preregistered pilot: {error}", file=sys.stderr)
        raise SystemExit(20) from error
    except (StageError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"V5.3 stopped fail-closed: {error}") from error


if __name__ == "__main__":
    main()
