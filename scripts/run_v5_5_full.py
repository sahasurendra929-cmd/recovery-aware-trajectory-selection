#!/usr/bin/env python3
"""Single-host controller for the complete V5.5 four-GPU experiment.

The controller is restart-by-phase, never overwrite-by-default:

1. ``preflight`` verifies source, tau2 and four CUDA GPUs.
2. ``data`` materializes and audits five exact recovery-dose arms.
3. ``train-smoke`` runs the longest-row two-step smoke for registered arms.
4. ``train`` queues formal QLoRA runs across four GPUs.
5. ``registry`` binds every adapter and run manifest.
6. ``evaluate`` starts three agent servers plus one fixed user/judge server,
   evaluates every registered checkpoint on all 21 derived-validation tasks,
   then stops the servers.
7. ``summarize`` rejects incomplete grids and computes task-cluster statistics.

``all`` executes data -> train-smoke -> train -> registry -> evaluate ->
summarize.  It does not open the official 60-task test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_JUDGE_PORT = 8201
SERVICE_READY_TIMEOUT_SECONDS = 1800.0

try:
    import v5_5_full_protocol as full
    from build_v5_5_checkpoint_registry import TRAINER_ARMS
except ModuleNotFoundError:
    from scripts import v5_5_full_protocol as full
    from scripts.build_v5_5_checkpoint_registry import TRAINER_ARMS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_USER_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_USER_JUDGE_REVISION = "539535859b135b0244c91f3e59816150c8056698"
SCREEN_ARMS = ("perfect_success", "repair_50", "repair_100")
BASE_DIAGNOSTIC_ARM = "base_control"
FULL_ARMS = tuple(TRAINER_ARMS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require_evaluation_inputs(args: argparse.Namespace) -> None:
    required = {
        "split_manifest": args.split_manifest,
        "validation_manifest": args.validation_manifest,
        "protocol_audit": args.protocol_audit,
        "checkpoint_registry": args.registry,
    }
    missing = [
        f"{label}={path}"
        for label, path in required.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise RuntimeError(
            "missing evaluation inputs (checked before service startup): "
            + ", ".join(missing)
        )


def git_value(*arguments: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_checkout(path: Path, *, label: str) -> None:
    for command, kind in (
        (["git", "-C", str(path), "diff", "--quiet", "--"], "unstaged"),
        (
            ["git", "-C", str(path), "diff", "--cached", "--quiet", "--"],
            "staged",
        ),
    ):
        if subprocess.run(command, check=False).returncode != 0:
            raise RuntimeError(f"{label} has {kind} tracked changes")


def log_command(path: Path, command: list[str], *, gpu: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": command,
        "gpu": gpu,
        "created_at_unix": time.time(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def run_checked(
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
) -> None:
    if stdout_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def gpu_inventory() -> list[dict[str, Any]]:
    completed = subprocess.run(
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
    for line in completed.stdout.splitlines():
        index, name, memory, driver = [part.strip() for part in line.split(",", 3)]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory),
                "driver": driver,
            }
        )
    return rows


def selected_grid(mode: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if mode == "reference-screen":
        return SCREEN_ARMS, (full.TRAINING_SEEDS[0],)
    if mode == "base-diagnostic":
        return (BASE_DIAGNOSTIC_ARM,), (full.TRAINING_SEEDS[0],)
    if mode == "full":
        return FULL_ARMS, full.TRAINING_SEEDS
    raise RuntimeError(f"unsupported mode {mode}")


def phase_preflight(args: argparse.Namespace) -> None:
    source = git_value("rev-parse", "HEAD^{commit}")
    if source != args.source_commit:
        raise RuntimeError(f"source commit drift: {source} != {args.source_commit}")
    require_clean_checkout(ROOT, label="experiment checkout")
    require_clean_checkout(args.tau2_root, label="tau2 checkout")
    tau2_commit = git_value(
        "-C",
        str(args.tau2_root),
        "rev-parse",
        "HEAD^{commit}",
    )
    if tau2_commit != full.TAU2_COMMIT:
        raise RuntimeError(f"tau2 commit drift: {tau2_commit}")
    inventory = gpu_inventory()
    if len(inventory) < 4:
        raise RuntimeError(f"V5.5 four-GPU controller found {len(inventory)} GPUs")
    for row in inventory[:4]:
        if row["memory_mib"] < 20 * 1024:
            raise RuntimeError(f"GPU {row['index']} has less than 20 GiB")
    for python, label in (
        (args.train_python, "training Python"),
        (args.serve_python, "serving Python"),
    ):
        if not python.is_file():
            raise RuntimeError(f"{label} does not exist: {python}")
    run_checked(
        [str(args.train_python), "-m", "pytest", "-q", *args.preflight_tests],
        stdout_path=args.results_root / "preflight" / "tests.log",
    )
    report = {
        "status": "PASS",
        "source_commit": source,
        "tau2_commit": tau2_commit,
        "gpus": inventory[:4],
        "official_test_used": False,
    }
    path = args.results_root / "preflight" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def phase_data(args: argparse.Namespace) -> None:
    if args.pairs is None or args.pair_audit is None or args.pair_manifest is None:
        raise RuntimeError("data phase requires --pairs/--pair-audit/--pair-manifest")
    command = [
        str(args.train_python),
        "scripts/prepare_v5_5_full_sft.py",
        "--tau2-root",
        str(args.tau2_root),
        "--pairs",
        str(args.pairs),
        "--pair-audit",
        str(args.pair_audit),
        "--pair-manifest",
        str(args.pair_manifest),
        "--output-dir",
        str(args.data_root),
        "--pair-mode",
        args.pair_mode,
        "--tokenizer-revision",
        args.base_model_revision,
        "--schedule-seed",
        str(full.TRAINING_SEEDS[0]),
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    run_checked(
        command,
        stdout_path=args.results_root / "data_prepare.log",
    )


def training_command(
    args: argparse.Namespace,
    *,
    arm: str,
    seed: int,
    mode: str,
    output: Path,
) -> list[str]:
    hashes = read_json(args.data_root / "hashes.json")
    train_relative = f"arms/{arm}/train.jsonl"
    command = [
        str(args.train_python),
        "scripts/train_v5_sft_causal.py",
        "--train-file",
        str(args.data_root / train_relative),
        "--validation-file",
        str(args.data_root / "validation_loss.jsonl"),
        "--output-dir",
        str(output),
        "--arm",
        arm,
        "--mode",
        mode,
        "--model-revision",
        args.base_model_revision,
        "--expected-source-commit",
        args.source_commit,
        "--expected-train-sha256",
        hashes[train_relative],
        "--expected-validation-sha256",
        hashes["validation_loss.jsonl"],
        "--data-audit",
        str(args.data_root / "audit.json"),
        "--data-hashes",
        str(args.data_root / "hashes.json"),
        "--training-seed",
        str(seed),
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def run_gpu_queue(
    jobs: list[tuple[str, int, list[str], Path]],
    *,
    command_log: Path,
) -> None:
    pending = deque(jobs)
    available = deque(range(4))
    running: dict[int, tuple[subprocess.Popen, Any, str, int, Path]] = {}
    try:
        while pending or running:
            while pending and available:
                gpu = available.popleft()
                name, seed, command, log_path = pending.popleft()
                if log_path.exists():
                    raise RuntimeError(f"refusing to overwrite log {log_path}")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open("w", encoding="utf-8")
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env.setdefault("TOKENIZERS_PARALLELISM", "false")
                log_command(command_log, command, gpu=gpu)
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                running[gpu] = (process, stream, name, seed, log_path)
                print(
                    json.dumps(
                        {
                            "event": "TRAIN_START",
                            "gpu": gpu,
                            "arm": name,
                            "seed": seed,
                            "pid": process.pid,
                        }
                    ),
                    flush=True,
                )
            time.sleep(2)
            for gpu, item in list(running.items()):
                process, stream, name, seed, log_path = item
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del running[gpu]
                available.append(gpu)
                if code != 0:
                    raise RuntimeError(
                        f"{name}/{seed} failed with exit {code}; see {log_path}"
                    )
                print(
                    json.dumps(
                        {
                            "event": "TRAIN_COMPLETE",
                            "gpu": gpu,
                            "arm": name,
                            "seed": seed,
                        }
                    ),
                    flush=True,
                )
    except BaseException:
        for process, stream, *_ in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            stream.close()
        raise


def phase_train(args: argparse.Namespace, *, smoke: bool) -> None:
    arms, seeds = selected_grid(args.experiment_mode)
    jobs = []
    for arm in arms:
        for seed in seeds:
            output = (
                args.results_root
                / ("smoke" if smoke else "training")
                / arm
                / str(seed)
            )
            command = training_command(
                args,
                arm=arm,
                seed=seed,
                mode="smoke" if smoke else "formal",
                output=output,
            )
            jobs.append(
                (
                    arm,
                    seed,
                    command,
                    output.parent / f"{output.name}.console.log",
                )
            )
    run_gpu_queue(
        jobs,
        command_log=args.results_root
        / ("smoke_commands.jsonl" if smoke else "training_commands.jsonl"),
    )


def phase_registry(args: argparse.Namespace) -> None:
    arms, seeds = selected_grid(args.experiment_mode)
    command = [
        str(args.train_python),
        "scripts/build_v5_5_checkpoint_registry.py",
        "--training-root",
        str(args.results_root / "training"),
        "--data-root",
        str(args.data_root),
        "--output",
        str(args.registry),
        "--source-commit",
        args.source_commit,
        "--model-revision",
        args.base_model_revision,
        "--arms",
        ",".join(arms),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--user-judge-model",
        args.user_judge_model,
        "--user-judge-revision",
        args.user_judge_revision,
        "--user-judge-alias",
        "openai/v55-user-judge",
    ]
    run_checked(
        command,
        stdout_path=args.results_root / "registry.log",
    )


def service_command(
    args: argparse.Namespace,
    *,
    gpu: int,
    port: int,
    registry: dict[str, Any],
    user_judge: bool,
) -> list[str]:
    blackwell_safe = [
        "--enforce-eager",
        "--disable-log-requests",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    if user_judge:
        return [
            str(args.serve_python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            registry["user_judge"]["model"],
            "--revision",
            registry["user_judge"]["revision"],
            "--served-model-name",
            registry["user_judge"]["model_id"].removeprefix("openai/"),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.92",
            "--generation-config",
            "vllm",
            *blackwell_safe,
        ]
    modules = []
    for arm in registry["registered_arms"]:
        for seed in registry["registered_training_seeds"]:
            entry = registry["entries"][arm][str(seed)]
            modules.append(
                f"{entry['model_id'].removeprefix('openai/')}="
                f"{entry['checkpoint']['path']}"
            )
    return [
        str(args.serve_python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        registry["base_model"],
        "--revision",
        registry["base_model_revision"],
        "--served-model-name",
        registry["base_model_alias"].removeprefix("openai/"),
        "--enable-lora",
        "--lora-modules",
        *modules,
        "--max-loras",
        "1",
        "--max-cpu-loras",
        str(len(modules)),
        "--max-lora-rank",
        "16",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        "--generation-config",
        "vllm",
        *blackwell_safe,
    ]


def wait_service(
    port: int,
    *,
    timeout: float = SERVICE_READY_TIMEOUT_SECONDS,
) -> list[str]:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            request = Request(
                url,
                headers={"Authorization": "Bearer local"},
            )
            with urlopen(request, timeout=5) as response:
                value = json.loads(response.read().decode())
            rows = value.get("data", [])
            return sorted(
                row["id"]
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("id"), str)
            )
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            last_error = str(error)
            time.sleep(5)
    raise RuntimeError(f"service {port} did not become ready: {last_error}")


def start_services(
    args: argparse.Namespace,
    registry: dict[str, Any],
) -> list[tuple[subprocess.Popen, Any, Path]]:
    services = []
    specifications = [
        (0, 8101, False),
        (1, 8102, False),
        (2, 8103, False),
        (3, USER_JUDGE_PORT, True),
    ]
    try:
        for gpu, port, user_judge in specifications:
            log = (
                args.results_root
                / "services"
                / f"gpu{gpu}-port{port}.log"
            )
            if log.exists():
                raise RuntimeError(f"refusing to overwrite service log {log}")
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("w", encoding="utf-8")
            command = service_command(
                args,
                gpu=gpu,
                port=port,
                registry=registry,
                user_judge=user_judge,
            )
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            # vLLM V1 has hung after the first token on the pinned Blackwell
            # RTX PRO 4500 + AWQ stack.  V0/eager is the validated runtime.
            env.setdefault("VLLM_USE_V1", "0")
            log_command(
                args.results_root / "service_commands.jsonl",
                command,
                gpu=gpu,
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            services.append((process, stream, log))
        for _, port, _ in specifications:
            observed = wait_service(port)
            if not observed:
                raise RuntimeError(f"service {port} exposed no models")
            print(
                json.dumps(
                    {
                        "event": "SERVICE_READY",
                        "port": port,
                        "models": observed,
                    }
                ),
                flush=True,
            )
        return services
    except BaseException:
        stop_services(services)
        raise


def stop_services(services: list[tuple[subprocess.Popen, Any, Path]]) -> None:
    for process, _, _ in services:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    for process, stream, _ in services:
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        stream.close()


def evaluation_command(
    args: argparse.Namespace,
    *,
    arm: str,
    training_seed: int,
    evaluation_seed: int,
    shard: int,
) -> tuple[list[str], Path]:
    output = (
        args.results_root
        / "evaluation"
        / arm
        / str(training_seed)
        / str(evaluation_seed)
        / f"shard-{shard}"
    )
    command = [
        str(args.serve_python),
        "scripts/run_v5_5_end_to_end_eval.py",
        "--tau2-root",
        str(args.tau2_root),
        "--split-manifest",
        str(args.split_manifest),
        "--validation-manifest",
        str(args.validation_manifest),
        "--protocol-audit",
        str(args.protocol_audit),
        "--checkpoint-registry",
        str(args.registry),
        "--evaluation-source-commit",
        args.source_commit,
        "--arm",
        arm,
        "--training-seed",
        str(training_seed),
        "--evaluation-seed",
        str(evaluation_seed),
        "--agent-api-base",
        f"http://127.0.0.1:{8101 + shard}/v1",
        "--user-api-base",
        f"http://127.0.0.1:{USER_JUDGE_PORT}/v1",
        "--output-dir",
        str(output),
        "--condition",
        "both",
        "--shard-index",
        str(shard),
        "--num-shards",
        "3",
    ]
    return command, output


def evaluation_batch_complete(
    outputs: list[Path],
    *,
    evaluation_source_commit: str,
    arm: str,
    training_seed: int,
    evaluation_seed: int,
) -> bool:
    for shard, output in enumerate(outputs):
        rows_path = output / "rows.jsonl"
        required = (
            rows_path,
            output / "metrics.json",
            output / "run_contract.json",
            output.parent / f"{output.name}.console.log",
            output / f"retail_clean.shard-{shard:03d}-of-003.json",
            output / f"retail_error.shard-{shard:03d}-of-003.json",
            output / f"airline_clean.shard-{shard:03d}-of-003.json",
            output / f"airline_error.shard-{shard:03d}-of-003.json",
        )
        if not all(path.is_file() for path in required):
            return False
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 14:
            return False
        metrics = read_json(output / "metrics.json")
        contract = read_json(output / "run_contract.json")
        expected = {
            "arm": arm,
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
        }
        if (
            metrics.get("status") != "PASS"
            or metrics.get("rows") != 14
            or metrics.get("official_test_used") is not False
            or any(metrics.get(key) != value for key, value in expected.items())
        ):
            return False
        if (
            contract.get("status") != "COMPLETE"
            or contract.get("evaluation_source_commit")
            != evaluation_source_commit
            or contract.get("official_test_used") is not False
            or any(contract.get(key) != value for key, value in expected.items())
        ):
            return False
    return True


def phase_evaluate(args: argparse.Namespace) -> None:
    require_evaluation_inputs(args)
    registry = read_json(args.registry)
    services = start_services(args, registry)
    arms, seeds = selected_grid(args.experiment_mode)
    try:
        for arm in arms:
            for training_seed in seeds:
                for evaluation_seed in full.EVALUATION_SEEDS:
                    planned = [
                        evaluation_command(
                            args,
                            arm=arm,
                            training_seed=training_seed,
                            evaluation_seed=evaluation_seed,
                            shard=shard,
                        )
                        for shard in range(3)
                    ]
                    outputs = [output for _, output in planned]
                    if evaluation_batch_complete(
                        outputs,
                        evaluation_source_commit=args.source_commit,
                        arm=arm,
                        training_seed=training_seed,
                        evaluation_seed=evaluation_seed,
                    ):
                        print(
                            json.dumps(
                                {
                                    "event": "EVAL_SKIP_COMPLETE",
                                    "arm": arm,
                                    "training_seed": training_seed,
                                    "evaluation_seed": evaluation_seed,
                                }
                            ),
                            flush=True,
                        )
                        continue
                    existing = [
                        path
                        for output in outputs
                        for path in (
                            output,
                            output.parent / f"{output.name}.console.log",
                        )
                        if path.exists()
                    ]
                    if existing:
                        raise RuntimeError(
                            "incomplete evaluation batch exists; archive it "
                            f"before retry: {existing[0]}"
                        )
                    running = []
                    for shard, (command, output) in enumerate(planned):
                        log = output.parent / f"{output.name}.console.log"
                        if log.exists():
                            raise RuntimeError(f"refusing to overwrite {log}")
                        log.parent.mkdir(parents=True, exist_ok=True)
                        stream = log.open("w", encoding="utf-8")
                        log_command(
                            args.results_root / "evaluation_commands.jsonl",
                            command,
                            gpu=shard,
                        )
                        process = subprocess.Popen(
                            command,
                            cwd=ROOT,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                        )
                        running.append((process, stream, log))
                    failed = None
                    for process, stream, log in running:
                        code = process.wait()
                        stream.close()
                        if code != 0:
                            failed = RuntimeError(
                                f"evaluation failed with exit {code}: {log}"
                            )
                    if failed is not None:
                        raise failed
                    print(
                        json.dumps(
                            {
                                "event": "EVAL_COMPLETE",
                                "arm": arm,
                                "training_seed": training_seed,
                                "evaluation_seed": evaluation_seed,
                            }
                        ),
                        flush=True,
                    )
    finally:
        stop_services(services)


def phase_summarize(args: argparse.Namespace) -> None:
    arms, seeds = selected_grid(args.experiment_mode)
    command = [
        str(args.serve_python),
        "scripts/summarize_v5_5_results.py",
        "--results-root",
        str(args.results_root / "evaluation"),
        "--validation-manifest",
        str(args.validation_manifest),
        "--checkpoint-registry",
        str(args.registry),
        "--output-dir",
        str(args.results_root / "summary"),
        "--arms",
        ",".join(arms),
        "--training-seeds",
        ",".join(str(seed) for seed in seeds),
        "--evaluation-seeds",
        ",".join(str(seed) for seed in full.EVALUATION_SEEDS),
        "--bootstrap-replicates",
        str(full.BOOTSTRAP_REPLICATES),
    ]
    run_checked(
        command,
        stdout_path=args.results_root / "summary.log",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "data",
            "train-smoke",
            "train",
            "registry",
            "evaluate",
            "summarize",
            "all",
        ),
        required=True,
    )
    parser.add_argument(
        "--experiment-mode",
        choices=("reference-screen", "base-diagnostic", "full"),
        default="reference-screen",
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--serve-python", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "processed" / "v5_5_full",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results" / "v5_5_full",
    )
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--pair-audit", type=Path)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument(
        "--pair-mode",
        choices=("reference", "natural"),
        default="reference",
    )
    parser.add_argument(
        "--base-model-revision",
        default=DEFAULT_BASE_REVISION,
    )
    parser.add_argument(
        "--user-judge-model",
        default=DEFAULT_USER_JUDGE_MODEL,
    )
    parser.add_argument(
        "--user-judge-revision",
        default=DEFAULT_USER_JUDGE_REVISION,
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "artifacts" / "v5_stage0" / "manifests" / "split_manifest.json",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "v5_stage1_protocol" / "validation_manifest.json",
    )
    parser.add_argument(
        "--protocol-audit",
        type=Path,
        default=ROOT / "data" / "processed" / "v5_stage1_protocol" / "audit.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "results" / "v5_5_full" / "checkpoint_registry.json",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.set_defaults(
        preflight_tests=[
            "tests/test_v5_5_protocol.py",
            "tests/test_v5_5_manifest.py",
            "tests/test_v5_5_full_protocol.py",
            "tests/test_v5_5_full_sft_data.py",
            "tests/test_v5_5_eval_and_summary.py",
            "tests/test_v5_5_completeness_fixes.py",
            "tests/test_v5_sft_causal_train.py",
        ]
    )
    args = parser.parse_args()
    for name in (
        "tau2_root",
        "train_python",
        "serve_python",
        "data_root",
        "results_root",
        "split_manifest",
        "validation_manifest",
        "protocol_audit",
        "registry",
        "pairs",
        "pair_audit",
        "pair_manifest",
    ):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    return args


def main() -> None:
    args = parse_args()
    phases = (
        (
            "preflight",
            "data",
            "train-smoke",
            "train",
            "registry",
            "evaluate",
            "summarize",
        )
        if args.phase == "all"
        else (args.phase,)
    )
    handlers = {
        "preflight": phase_preflight,
        "data": phase_data,
        "train-smoke": lambda value: phase_train(value, smoke=True),
        "train": lambda value: phase_train(value, smoke=False),
        "registry": phase_registry,
        "evaluate": phase_evaluate,
        "summarize": phase_summarize,
    }
    for phase in phases:
        print(json.dumps({"event": "PHASE_START", "phase": phase}), flush=True)
        handlers[phase](args)
        print(json.dumps({"event": "PHASE_COMPLETE", "phase": phase}), flush=True)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "phases": list(phases),
                "official_test_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
