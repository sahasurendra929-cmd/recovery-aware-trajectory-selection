#!/usr/bin/env python3
"""Fail-closed V5.2 controller for one RunPod host with four RTX 4090 GPUs.

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
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterable
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260722
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
SPLIT_SHA256 = "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
TEACHER_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
TEACHER_REVISION = "539535859b135b0244c91f3e59816150c8056698"
USER_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
USER_REVISION = "b25037543e9394b818fdfca67ab2a00ecc7dd641"
STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ARMS = ("perfect_success", "failure_raw", "repair_50", "repair_100")
EVAL_ARMS = ("base_model", *ARMS)
EVAL_MODEL_IDS = {
    "base_model": "openai/v5-base",
    "perfect_success": "openai/v5-perfect-success",
    "failure_raw": "openai/v5-failure-raw",
    "repair_50": "openai/v5-repair-50",
    "repair_100": "openai/v5-repair-100",
}
SERVED_ALIASES = tuple(value.removeprefix("openai/") for value in EVAL_MODEL_IDS.values())
GENERATION_SHARDS = 3
GENERATION_ATTEMPTS_PER_CONDITION = 12
EVALUATION_SHARDS = 4
MAX_MODEL_LEN = 32768
MIN_GPU_MEMORY_MIB = 24_000
MIN_FREE_DISK_GIB = 140


class StageError(RuntimeError):
    """A protocol or runtime invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def base_env(*, offline: bool = True, gpu: int | None = None) -> dict[str, str]:
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
        raise StageError(f"V5.2 requires exactly 4 visible GPUs; found {len(rows)}")
    for expected_index, row in enumerate(rows):
        if row["index"] != expected_index:
            raise StageError(f"GPU indices must be 0..3; found {rows}")
        if "RTX 4090" not in row["name"]:
            raise StageError(f"GPU {expected_index} is not an RTX 4090: {row}")
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
    if kind == "train" and (
        payload.get("cuda_available") is not True
        or payload.get("bf16_supported") is not True
        or payload.get("cuda") != "12.8"
    ):
        raise StageError(f"training CUDA environment drift: {payload}")
    return payload


def static_protocol_audit() -> None:
    config = ROOT / "configs" / "v5_2_sft_causal.yaml"
    handoff = ROOT / "V5_2_SINGLE_HOST_HANDOFF.md"
    prompt = ROOT / "V5_2_RUNPOD_4X4090_AGENT_PROMPT.md"
    for path in (config, handoff, prompt):
        if not path.is_file():
            raise StageError(f"missing V5.2 protocol file: {path}")
    config_text = config.read_text(encoding="utf-8")
    required_config_fragments = {
        "every_vllm_server_max_model_len: 32768",
        "attempts_per_task_per_condition: 12",
        "minimum_distinct_task_ids: 40",
        "minimum_task_level_pairs: 48",
        "maximum_pairs_per_task: 2",
        "training_sequence_tokens_at_most: 8192",
        "num_shards: 3",
    }
    missing = {
        fragment
        for fragment in required_config_fragments
        if fragment not in config_text
    }
    if missing:
        raise StageError(
            f"V5.2 config lacks frozen controller values: {sorted(missing)}"
        )
    for path in (handoff, prompt):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "--max-model-len" in line and "32768" not in line:
                raise StageError(f"{path.name} contains non-32768 vLLM command")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
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
        raise StageError(f"reserved V5.2 ports are already in use: {occupied}")
    environments = None
    if not args.skip_environment_check:
        environments = {
            "serve": require_environment(args.serve_python, kind="serve"),
            "train": require_environment(args.train_python, kind="train"),
        }
        if not args.vllm.is_file():
            raise StageError(f"vLLM executable is absent: {args.vllm}")
    payload = {
        "status": "PASS",
        "protocol": "v5_2_single_host_preflight",
        "source_commit": source_commit(),
        "tau2_commit": observed_tau2,
        "split_manifest_sha256": SPLIT_SHA256,
        "gpus": gpu_rows,
        "free_disk_gib": round(free_gib, 2),
        "ports_free": list(ports),
        "environments": environments,
        "official_test_used": False,
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    output = args.results_root / "preflight.json"
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
    return (
        generation.get("status") == "COMPLETE"
        and evaluation.get("status") == "COMPLETE"
        and generation.get("official_test_used") is False
        and evaluation.get("official_test_used") is False
        and generation.get("verified_injections") == 83
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
            "scripts/prepare_v5_stage1_manifests.py",
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
        *sorted(str(path.relative_to(ROOT)) for path in ROOT.glob("tests/test_v5_2*.py")),
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


def start_generation_services(
    args: argparse.Namespace,
) -> list[tuple[subprocess.Popen[bytes], Any, Path]]:
    specifications = [
        {
            "gpu": 0,
            "port": 8001,
            "model": USER_MODEL,
            "revision": USER_REVISION,
            "api_key": "stage1-user-local",
            "max_num_seqs": 3,
            "memory": "0.90",
            "name": "generation-user-judge",
        },
        *[
            {
                "gpu": gpu,
                "port": 8010 + gpu,
                "model": TEACHER_MODEL,
                "revision": TEACHER_REVISION,
                "api_key": "stage1-teacher-local",
                "max_num_seqs": 1,
                "memory": "0.90",
                "name": f"generation-teacher-gpu{gpu}",
            }
            for gpu in (1, 2, 3)
        ],
    ]
    services = []
    try:
        for item in specifications:
            if not port_is_free(item["port"]):
                raise StageError(f"generation port is occupied: {item['port']}")
            command = [
                *vllm_base_command(
                    args, str(item["model"]), str(item["revision"])
                ),
                "--served-model-name",
                str(item["model"]),
                "--port",
                str(item["port"]),
                "--api-key",
                str(item["api_key"]),
                "--gpu-memory-utilization",
                str(item["memory"]),
                "--max-num-seqs",
                str(item["max_num_seqs"]),
            ]
            log = args.results_root / "logs" / f"{item['name']}.log"
            process, handle = spawn_logged(
                command,
                env=base_env(offline=True, gpu=int(item["gpu"])),
                log_path=log,
            )
            pid_path = args.results_root / "pids" / f"{item['name']}.pid"
            write_pid(pid_path, process)
            services.append((process, handle, pid_path))
        for item, (process, _, _) in zip(specifications, services):
            wait_for_service(
                port=int(item["port"]),
                api_key=str(item["api_key"]),
                expected_models={str(item["model"])},
                process=process,
                timeout_seconds=args.health_timeout,
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


def generation_contract_path(args: argparse.Namespace, shard: int) -> Path:
    return (
        args.raw_root
        / f"run_contract.shard-{shard:03d}-of-{GENERATION_SHARDS:03d}.json"
    )


def generation_shard_complete(args: argparse.Namespace, shard: int) -> bool:
    path = generation_contract_path(args, shard)
    if not path.is_file():
        return False
    value = read_json(path)
    if (
        value.get("status") != "COMPLETE"
        or value.get("shard_index") != shard
        or value.get("num_shards") != GENERATION_SHARDS
        or value.get("num_trials") != GENERATION_ATTEMPTS_PER_CONDITION
        or value.get("official_test_used") is not False
    ):
        return False
    result_hashes = value.get("result_sha256")
    if not isinstance(result_hashes, dict) or not result_hashes:
        return False
    return all(
        (args.raw_root / name).is_file()
        and sha256_file(args.raw_root / name) == declared
        for name, declared in result_hashes.items()
    )


def generation_complete(args: argparse.Namespace) -> bool:
    return args.raw_root.is_dir() and all(
        generation_shard_complete(args, shard)
        for shard in range(GENERATION_SHARDS)
    )


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
    if generation_complete(args):
        print("[skip] all three V5.2 generation shards are COMPLETE")
        return
    if not protocol_complete(args):
        raise StageError("protocol and dynamic-audit barrier is incomplete")
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
    services = start_generation_services(args)
    workers: list[tuple[subprocess.Popen[bytes], Any, int]] = []
    try:
        for shard in missing_shards:
            gpu = shard + 1
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
                f"http://127.0.0.1:{8010 + gpu}/v1",
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
    if not generation_complete(args):
        raise StageError("generation ended without three COMPLETE contracts")


def data_complete(args: argparse.Namespace) -> bool:
    audit_path = args.processed_root / "audit.json"
    hashes_path = args.processed_root / "hashes.json"
    if not audit_path.is_file() or not hashes_path.is_file():
        return False
    audit = read_json(audit_path)
    return (
        audit.get("status") == "PASS"
        and audit.get("design_protocol")
        == "v5_2_task_level_cross_seed_sft_screen"
        and audit.get("design_version") == "5.2"
        and audit.get("attempts_per_task_per_condition")
        == GENERATION_ATTEMPTS_PER_CONDITION
        and audit.get("official_test_used") is False
        and audit.get("derived_validation_used_for_supervision") is False
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
        print("[skip] V5.2 data constructor audit is PASS")
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
            "scripts/prepare_v5_2_sft_causal.py",
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
        raise StageError("V5.2 data constructor did not publish PASS audit")


def training_run_complete(path: Path, *, mode: str) -> bool:
    manifest = path / "run_manifest.json"
    adapter = path / "checkpoint_final" / "adapter_model.safetensors"
    config = path / "checkpoint_final" / "adapter_config.json"
    if not (manifest.is_file() and adapter.is_file() and config.is_file()):
        return False
    value = read_json(manifest)
    return (
        value.get("mode") == mode
        and value.get("held_out_test_accessed") is False
        and value.get("protocol") == "v5_stage1_message_masked_sft_7b"
    )


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
            if training_run_complete(output, mode=mode):
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
            args.results_root / arm / mode, mode=mode
        )
    ]
    if missing:
        raise StageError(f"{mode} training outputs incomplete: {missing}")


def train_arms(args: argparse.Namespace) -> None:
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


def registry_complete(args: argparse.Namespace) -> bool:
    path = args.results_root / "checkpoint_registry.json"
    if not path.is_file() or not data_complete(args):
        return False
    value = read_json(path)
    provenance = value.get("training_data_provenance") or {}
    return (
        value.get("protocol") == "v5_stage1_checkpoint_registry"
        and value.get("source_commit") == source_commit()
        and set((value.get("entries") or {})) == set(EVAL_ARMS)
        and provenance.get("data_audit_sha256")
        == sha256_file(args.processed_root / "audit.json")
        and provenance.get("data_hashes_sha256")
        == sha256_file(args.processed_root / "hashes.json")
    )


def build_registry(args: argparse.Namespace) -> None:
    if registry_complete(args):
        print("[skip] checkpoint registry is complete")
        return
    output = args.results_root / "checkpoint_registry.json"
    if output.exists():
        raise StageError(f"partial/invalid registry must be archived: {output}")
    for arm in ARMS:
        if not training_run_complete(
            args.results_root / arm / "formal", mode="formal"
        ):
            raise StageError(f"formal arm is incomplete: {arm}")
    command = [
        str(args.train_python),
        "scripts/build_v5_checkpoint_registry.py",
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
                "v5-base",
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


def evaluation_shard_complete(
    args: argparse.Namespace, *, arm: str, shard: int
) -> bool:
    path = (
        args.results_root
        / "evaluation"
        / arm
        / f"run_contract.shard-{shard:03d}-of-004.json"
    )
    if (
        not path.is_file()
        or not (args.results_root / "checkpoint_registry.json").is_file()
        or not (args.processed_root / "validation_manifest.json").is_file()
    ):
        return False
    value = read_json(path)
    result_hashes = value.get("result_sha256")
    return (
        value.get("status") == "COMPLETE"
        and value.get("arm") == arm
        and value.get("num_shards") == EVALUATION_SHARDS
        and value.get("official_test_used") is False
        and value.get("checkpoint_registry_sha256")
        == sha256_file(args.results_root / "checkpoint_registry.json")
        and value.get("evaluation_manifest_sha256")
        == sha256_file(args.processed_root / "validation_manifest.json")
        and isinstance(result_hashes, dict)
        and bool(result_hashes)
        and all(
            (path.parent / name).is_file()
            and sha256_file(path.parent / name) == declared
            for name, declared in result_hashes.items()
        )
    )


def evaluation_command(
    args: argparse.Namespace, *, arm: str, shard: int
) -> list[str]:
    command = [
        str(args.serve_python),
        "scripts/run_v5_sft_causal_eval.py",
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
        (arm, shard)
        for arm in EVAL_ARMS
        for shard in range(4)
        if not evaluation_shard_complete(args, arm=arm, shard=shard)
    ]
    if missing:
        raise StageError(f"evaluation contracts incomplete: {missing}")


def summarize(args: argparse.Namespace) -> None:
    output = args.results_root / "mechanism_screen_summary.json"
    if output.is_file():
        value = read_json(output)
        if value.get("status") == "PASS":
            print("[skip] V5.2 mechanism screen summary is PASS")
            return
        raise StageError(f"invalid existing summary must be archived: {output}")
    command = [
        str(args.train_python),
        "scripts/summarize_v5_sft_causal.py",
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
    if read_json(output).get("status") != "PASS":
        raise StageError("summary protocol status is not PASS")


def status(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "protocol_complete": protocol_complete(args),
        "generation_complete": generation_complete(args),
        "data_complete": data_complete(args),
        "training": {
            mode: {
                arm: training_run_complete(
                    args.results_root / arm / mode, mode=mode
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
        "summary_exists": (
            args.results_root / "mechanism_screen_summary.json"
        ).is_file(),
        "official_test_used": False,
    }
    print(json.dumps(payload, indent=2))
    return payload


def stop_stale_services(args: argparse.Namespace) -> None:
    pid_root = args.results_root / "pids"
    if not pid_root.is_dir():
        print("[stop] no V5.2 PID directory")
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
        print(f"[stop] signalled V5.2 service PID {pid}")


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
        default=ROOT / "data/processed/v5_2_protocol",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw/v5_2_sft_causal_generation",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=ROOT / "data/processed/v5_2_sft_causal",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/v5_2_sft_causal",
    )
    parser.add_argument("--health-timeout", type=int, default=1800)
    parser.add_argument("--skip-environment-check", action="store_true")
    args = parser.parse_args()
    args.serve_venv = args.serve_venv.expanduser().resolve()
    args.train_venv = args.train_venv.expanduser().resolve()
    args.tau2_root = args.tau2_root.expanduser().resolve()
    args.workspace_root = args.workspace_root.expanduser().resolve()
    args.protocol_root = args.protocol_root.expanduser().resolve()
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
            generate_trajectories(args)
            prepare_data(args)
            train_arms(args)
            build_registry(args)
            evaluate(args)
            summarize(args)
            status(args)
        else:
            stages[args.stage]()
    except (StageError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"V5.2 stopped fail-closed: {error}") from error


if __name__ == "__main__":
    main()
