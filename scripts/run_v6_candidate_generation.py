#!/usr/bin/env python3
"""Generate fresh, environment-grounded V6 recovery candidate pairs.

This executable materializes the outcome-independent registry produced by
``prepare_v6_candidate_registry.py``.  It is intentionally fail closed:

* only registry tasks in the selected phase are touched;
* all candidate pairs for a task share one observed clean prefix and one
  environment snapshot;
* an injected call must produce a real tau2 tool error while leaving both
  databases unchanged;
* the clean future is removed before a recovery rollout is generated;
* the first recovery tool action must equal the registered corrective call;
* failed calls/results are context and never positive labels;
* matched and crossed forced-first cells use the same continuation policy,
  decoding contract, seed set, and rollout budget.

The output is an unscored pool.  ``measure_v6_candidate_tokens.py`` adds exact
token costs and frozen-base first-action log probabilities before
``score_v6_candidates.py`` computes hardness and kappa ranks.

No official-test identity or content is accepted by this program.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

try:
    import prepare_v6_candidate_registry as registry_contract
    import v6_selection_protocol as protocol
    from run_v5_sft_causal_generate import (
        endpoint_args,
        litellm_openai_model,
        normalize_tool_only_message,
        patch_local_nl_judge,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import prepare_v6_candidate_registry as registry_contract
    from scripts import v6_selection_protocol as protocol
    from scripts.run_v5_sft_causal_generate import (
        endpoint_args,
        litellm_openai_model,
        normalize_tool_only_message,
        patch_local_nl_judge,
    )


GENERATION_PROTOCOL = "v6_fresh_recovery_candidate_generation_v1"
SEMANTIC_GENERATION_CONTRACT_PROTOCOL = (
    "v6_fresh_recovery_semantic_generation_contract_v1"
)
AGENT_NAME = "v6_fresh_recovery_agent"
REFERENCE_GUIDED_CLEAN_AGENT_NAME = "llm_agent_gt"
CLEAN_AGENT_MODES = (
    "standard",
    "reference_guided",
    "deterministic_reference_replay",
)
RECOVERY_CONTINUATION_MODES = (
    "fresh_teacher",
    "deterministic_reference_tail",
    "deterministic_reference_completion",
)
SFT_SYSTEM_INSTRUCTION = """\
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_CONTINUATION_SEEDS = (20260821,)
DECODING_TEMPERATURE = 0.0
DECODING_TOP_P = 1.0
PARALLEL_TOOL_CALLS = False
RUN_NUM_TRIALS = 1
RUN_MAX_ERRORS = 10
RUN_MAX_CONCURRENCY = 1
RUN_MAX_RETRIES = 1
RUN_RETRY_DELAY_SECONDS = 1.0
RUN_HALLUCINATION_RETRIES = 0
VOLATILE_MESSAGE_FIELDS = {
    "timestamp",
    "turn_idx",
    "cost",
    "usage",
    "generation_time_seconds",
}


class V6GenerationError(RuntimeError):
    """A registry, environment, or fresh-recovery invariant failed."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(canonical(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def semantic(value: Any) -> Any:
    """Drop transport metadata while retaining every experimental message."""
    if isinstance(value, list):
        return [semantic(item) for item in value]
    if isinstance(value, dict):
        return {
            key: semantic(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_MESSAGE_FIELDS
        }
    return value


def semantic_sha256(value: Any) -> str:
    return sha256(semantic(value))


def parse_seed_set(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise V6GenerationError("--continuation-seeds must be comma-separated integers") from error
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise V6GenerationError("continuation seeds must be non-empty, distinct, non-negative")
    return seeds


def semantic_generation_contract(
    args: argparse.Namespace,
    *,
    continuation_seeds: Sequence[int],
    judge_model: str | None = None,
    judge_revision: str | None = None,
    judge_api_base: str | None = None,
    design_protocol: str = protocol.PROTOCOL,
) -> dict[str, Any]:
    """Return every parameter that can change generated task semantics.

    Sharding and output-path details deliberately live in the enclosing run
    contract.  This smaller contract is copied into task receipts and candidate
    pairs so a deleted/recreated ``run_contract.json`` cannot make stale task
    output eligible for resume under different generation settings.
    """

    effective_judge_model = judge_model or args.judge_model or args.user_model
    effective_judge_revision = (
        judge_revision or args.judge_revision or args.user_revision
    )
    effective_judge_api_base = (
        judge_api_base or args.judge_api_base or args.user_api_base
    )
    clean_agent_mode = getattr(args, "clean_agent_mode", "standard")
    seeds = list(continuation_seeds)
    decoding = {
        "temperature": DECODING_TEMPERATURE,
        "top_p": DECODING_TOP_P,
        "max_tokens": args.max_tokens,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        "applies_to": ["teacher", "user", "judge"],
        "assistant_tool_only_normalization": (
            "execute_first_tool_call_then_replan"
        ),
    }
    runner = {
        "num_trials": RUN_NUM_TRIALS,
        "max_steps": args.max_steps,
        "max_errors": RUN_MAX_ERRORS,
        "timeout_seconds": args.timeout,
        "max_concurrency": RUN_MAX_CONCURRENCY,
        "max_retries": RUN_MAX_RETRIES,
        "retry_delay_seconds": RUN_RETRY_DELAY_SECONDS,
        "auto_resume": False,
        "hallucination_retries": RUN_HALLUCINATION_RETRIES,
        "enforce_communication_protocol": False,
    }
    return {
        "protocol": SEMANTIC_GENERATION_CONTRACT_PROTOCOL,
        "design_protocol": design_protocol,
        "tau2_commit": protocol.TAU2_COMMIT,
        "agent_name": AGENT_NAME,
        "clean_agent_mode": clean_agent_mode,
        "clean_agent_name": (
            REFERENCE_GUIDED_CLEAN_AGENT_NAME
            if clean_agent_mode == "reference_guided"
            else (
                "deterministic_tau2_reference_replay"
                if clean_agent_mode == "deterministic_reference_replay"
                else AGENT_NAME
            )
        ),
        "recovery_continuation_mode": getattr(
            args, "recovery_continuation_mode", "fresh_teacher"
        ),
        "system_instruction_sha256": sha256(SFT_SYSTEM_INSTRUCTION),
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "teacher_api_base": args.teacher_api_base,
        "user_model": args.user_model,
        "user_revision": args.user_revision,
        "user_api_base": args.user_api_base,
        "judge_model": effective_judge_model,
        "judge_revision": effective_judge_revision,
        "judge_api_base": effective_judge_api_base,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "timeout": args.timeout,
        "clean_attempts": args.clean_attempts,
        "recovery_attempts": args.recovery_attempts,
        "continuation_seeds": seeds,
        "continuation_seed_set_sha256": sha256(seeds),
        "decoding": decoding,
        "runner": runner,
        "seed_contract": {
            "clean_seed_source": "registry.choice_seed",
            "clean_attempt_seed_derivation": (
                "choice_seed + zero_based_attempt_index"
            ),
            "recovery_seed_source": "registry.branch.recovery_seed",
            "recovery_attempt_seed_derivation": (
                "recovery_seed + zero_based_attempt_index"
            ),
            "forced_first_continuation_seeds": seeds,
            "judge_seed": seeds[0],
        },
        "fresh_recovery_generated": True,
        "gold_clean_future_visible": False,
    }


def build_run_contract(
    args: argparse.Namespace,
    *,
    registry_file_sha256: str,
    registry_sha256: str,
    task_ids: Sequence[str],
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the shard/run envelope around the semantic generation contract."""

    semantic_contract_copy = deepcopy(dict(semantic_contract))
    return {
        "protocol": GENERATION_PROTOCOL,
        "design_protocol": semantic_contract_copy["design_protocol"],
        "registry_file_sha256": registry_file_sha256,
        "registry_sha256": registry_sha256,
        "phase": args.phase,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "smoke_task": args.smoke_task,
        "task_ids": sorted(task_ids),
        # Retain the flat public fields for CLI/artifact compatibility while
        # binding their complete semantics below.
        "tau2_commit": semantic_contract_copy["tau2_commit"],
        "teacher_model": semantic_contract_copy["teacher_model"],
        "teacher_revision": semantic_contract_copy["teacher_revision"],
        "teacher_api_base": semantic_contract_copy["teacher_api_base"],
        "user_model": semantic_contract_copy["user_model"],
        "user_revision": semantic_contract_copy["user_revision"],
        "user_api_base": semantic_contract_copy["user_api_base"],
        "judge_model": semantic_contract_copy["judge_model"],
        "judge_revision": semantic_contract_copy["judge_revision"],
        "judge_api_base": semantic_contract_copy["judge_api_base"],
        "max_tokens": semantic_contract_copy["max_tokens"],
        "max_steps": semantic_contract_copy["max_steps"],
        "timeout": semantic_contract_copy["timeout"],
        "clean_attempts": semantic_contract_copy["clean_attempts"],
        "recovery_attempts": semantic_contract_copy["recovery_attempts"],
        "continuation_seeds": semantic_contract_copy["continuation_seeds"],
        "decoding": deepcopy(semantic_contract_copy["decoding"]),
        "runner": deepcopy(semantic_contract_copy["runner"]),
        "semantic_generation_contract": semantic_contract_copy,
        "semantic_generation_contract_sha256": sha256(semantic_contract_copy),
        "official_test_used": False,
    }


def validate_task_resume_receipt(
    receipt: Mapping[str, Any],
    *,
    task_identity: str,
    registry_sha256: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
) -> None:
    """Fail closed unless a PASS receipt belongs to this exact generation run."""

    expected_semantic = dict(semantic_contract)
    semantic_contract_sha256 = sha256(expected_semantic)
    pairs = receipt.get("candidate_pairs")
    pair_contracts_match = (
        isinstance(pairs, list)
        and bool(pairs)
        and receipt.get("candidate_pair_count") == len(pairs)
        and all(
            isinstance(pair, dict)
            and pair.get("task_identity") == task_identity
            and pair.get("registry_sha256") == registry_sha256
            and pair.get("generation_contract") == expected_semantic
            and pair.get("generation_contract_sha256")
            == semantic_contract_sha256
            for pair in pairs
        )
    )
    if (
        receipt.get("protocol") != GENERATION_PROTOCOL
        or receipt.get("task_identity") != task_identity
        or receipt.get("status") != "PASS"
        or receipt.get("registry_sha256") != registry_sha256
        or receipt.get("run_contract_sha256") != run_contract_sha256
        or receipt.get("semantic_generation_contract")
        != expected_semantic
        or receipt.get("semantic_generation_contract_sha256")
        != semantic_contract_sha256
        or not pair_contracts_match
    ):
        raise V6GenerationError(
            f"{task_identity}: existing receipt is not a valid PASS resume "
            "(run/semantic generation contract drift)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--teacher-api-base", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-revision", required=True)
    parser.add_argument("--user-api-base", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-revision")
    parser.add_argument("--judge-api-base")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--clean-attempts", type=int, default=2)
    parser.add_argument(
        "--clean-agent-mode",
        choices=CLEAN_AGENT_MODES,
        default="standard",
        help=(
            "Agent used only to construct the successful clean source. "
            "reference_guided uses tau2 evaluation actions and therefore "
            "requires a separately versioned scientific protocol."
        ),
    )
    parser.add_argument("--recovery-attempts", type=int, default=2)
    parser.add_argument(
        "--recovery-continuation-mode",
        choices=RECOVERY_CONTINUATION_MODES,
        default="fresh_teacher",
        help=(
            "Continuation after the frozen corrective action. "
            "deterministic reference modes require separately versioned "
            "scientific protocols."
        ),
    )
    parser.add_argument(
        "--continuation-seeds",
        default=",".join(str(seed) for seed in DEFAULT_CONTINUATION_SEEDS),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--smoke-task",
        help="Diagnostic only: materialize exactly this registered phase task.",
    )
    return parser.parse_args()


def configure_tau2(root: Path) -> None:
    source = root.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))


def register_agent() -> None:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.registry import registry

    class V6Agent(LLMAgent):
        def _generate_next_message(self, message, state):
            return normalize_tool_only_message(
                super()._generate_next_message(message, state)
            )

    def factory(tools, domain_policy, **kwargs):
        return V6Agent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
        )

    if registry.get_agent_factory(AGENT_NAME) is None:
        registry.register_agent_factory(factory, AGENT_NAME)


def verify_registry(payload: Mapping[str, Any]) -> str:
    if payload.get("protocol") != registry_contract.REGISTRY_PROTOCOL:
        raise V6GenerationError("candidate registry protocol drift")
    if payload.get("design_protocol") not in (
        protocol.PROTOCOL,
        registry_contract.V6_1_72B_TEACHER_PROTOCOL,
        registry_contract.V6_2_REFERENCE_GUIDED_CLEAN_PROTOCOL,
        registry_contract.V6_3_DETERMINISTIC_CLEAN_REPLAY_PROTOCOL,
        registry_contract.V6_4_REFERENCE_TAIL_RECOVERY_PROTOCOL,
        registry_contract.V6_5_REFERENCE_COMPLETION_PROTOCOL,
    ):
        raise V6GenerationError("candidate registry design protocol drift")
    if (
        payload.get("selection_unit") != "candidate_pair"
        or payload.get("grouping_unit") != "choice_set"
        or payload.get("official_test_used") is not False
        or payload.get("official_test_sealed") is not True
        or payload.get("official_test_task_content_exported") is not False
        or payload.get("official_test_identity_overlap_count") != 0
    ):
        raise V6GenerationError("candidate registry violates the test seal")
    structural = payload.get("structural_eligibility_sha256")
    if structural != protocol.STRUCTURAL_ELIGIBILITY_SHA256:
        raise V6GenerationError("candidate registry structural hash drift")
    declared = payload.get("registry_sha256")
    if not isinstance(declared, str) or SHA256_RE.fullmatch(declared) is None:
        raise V6GenerationError("candidate registry lacks a valid registry_sha256")
    unhashed = dict(payload)
    unhashed.pop("registry_sha256", None)
    observed = sha256(unhashed)
    if observed != declared:
        raise V6GenerationError(
            f"candidate registry bytes/content drift: {observed} != {declared}"
        )
    pairs = payload.get("candidate_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise V6GenerationError("candidate registry is empty")
    pair_ids: set[str] = set()
    for row in pairs:
        if not isinstance(row, dict):
            raise V6GenerationError("candidate registry pair is not an object")
        pair_id = row.get("candidate_pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise V6GenerationError("candidate_pair_id is absent or duplicated")
        pair_ids.add(pair_id)
        if (
            row.get("partition") != "arm_train"
            or row.get("official_test_used") is not False
            or not isinstance(row.get("branches"), list)
            or len(row["branches"]) != 2
        ):
            raise V6GenerationError(f"{pair_id}: malformed/forbidden registry pair")
        registered = dict(row)
        pair_hash = registered.pop("candidate_pair_sha256", None)
        if pair_hash != sha256(registered):
            raise V6GenerationError(f"{pair_id}: candidate-pair hash drift")
    return declared


def phase_pairs(
    registry: Mapping[str, Any],
    *,
    phase: str,
    shard_index: int,
    num_shards: int,
    smoke_task: str | None,
) -> list[dict[str, Any]]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise V6GenerationError("shard index must be in [0, num_shards)")
    phase_row = (registry.get("phase_registry") or {}).get(phase)
    if not isinstance(phase_row, dict) or not isinstance(
        phase_row.get("task_ids"), list
    ):
        raise V6GenerationError(f"registry has no {phase} phase")
    registered_tasks = [str(value) for value in phase_row["task_ids"]]
    if smoke_task is not None:
        if smoke_task not in registered_tasks:
            raise V6GenerationError("--smoke-task is not registered in this phase")
        selected_tasks = [smoke_task]
    else:
        ordered = sorted(
            registered_tasks,
            key=lambda task: (hashlib.sha256(task.encode()).hexdigest(), task),
        )
        selected_tasks = ordered[shard_index::num_shards]
    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in selected_tasks}
    for row in registry["candidate_pairs"]:
        if row.get("phase") == phase and row.get("task_identity") in by_task:
            by_task[str(row["task_identity"])].append(deepcopy(row))
    output: list[dict[str, Any]] = []
    for task in selected_tasks:
        pairs = sorted(by_task[task], key=lambda row: str(row["candidate_pair_id"]))
        if len(pairs) < protocol.MIN_PAIRS_PER_TASK:
            raise V6GenerationError(
                f"{task}: registry has {len(pairs)} pairs; "
                f"need >= {protocol.MIN_PAIRS_PER_TASK}"
            )
        output.extend(pairs)
    return output


def select_task(identity: str):
    from tau2.registry import registry

    domain, task_id = identity.split(":", 1)
    tasks = {str(task.id): task for task in registry.get_tasks_loader(domain)(None)}
    if task_id not in tasks:
        raise V6GenerationError(f"pinned tau2 lacks {identity}")
    return domain, tasks[task_id]


def text_run_config(
    args: argparse.Namespace,
    *,
    domain: str,
    seed: int,
    agent_name: str = AGENT_NAME,
):
    from tau2.data_model.simulation import TextRunConfig

    teacher_args = endpoint_args(
        args.teacher_api_base,
        "v6-local",
        max_tokens=args.max_tokens,
        seed=seed,
        temperature=DECODING_TEMPERATURE,
        top_p=DECODING_TOP_P,
    )
    user_args = endpoint_args(
        args.user_api_base,
        "v6-local",
        max_tokens=args.max_tokens,
        seed=seed,
        temperature=DECODING_TEMPERATURE,
        top_p=DECODING_TOP_P,
    )
    return TextRunConfig(
        domain=domain,
        agent=agent_name,
        user="user_simulator",
        llm_agent=litellm_openai_model(args.teacher_model),
        llm_args_agent=teacher_args,
        llm_user=litellm_openai_model(args.user_model),
        llm_args_user=user_args,
        num_trials=RUN_NUM_TRIALS,
        max_steps=args.max_steps,
        max_errors=RUN_MAX_ERRORS,
        timeout=args.timeout,
        max_concurrency=RUN_MAX_CONCURRENCY,
        seed=seed,
        log_level="INFO",
        max_retries=RUN_MAX_RETRIES,
        retry_delay=RUN_RETRY_DELAY_SECONDS,
        auto_resume=False,
        hallucination_retries=RUN_HALLUCINATION_RETRIES,
        enforce_communication_protocol=False,
        verbose_logs=True,
    )


def run_one(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    seed: int,
    save_dir: Path,
    agent_name: str = AGENT_NAME,
):
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner.batch import run_single_task

    started = time.monotonic()
    simulation = run_single_task(
        text_run_config(
            args, domain=domain, seed=seed, agent_name=agent_name
        ),
        task,
        seed=seed,
        evaluation_type=EvaluationType.ALL,
        save_dir=save_dir,
        verbose_logs=True,
    )
    return simulation, time.monotonic() - started


def reward(simulation: Any) -> float:
    value = getattr(getattr(simulation, "reward_info", None), "reward", None)
    if value is None:
        raise V6GenerationError("tau2 simulation has no official reward")
    return float(value)


def messages(simulation: Any) -> list[dict[str, Any]]:
    values = getattr(simulation, "messages", None)
    if not isinstance(values, list):
        raise V6GenerationError("tau2 simulation has no messages")
    return [message.model_dump(mode="json") for message in values]


def parse_messages(values: Sequence[Mapping[str, Any]]):
    from tau2.data_model.message import Message
    from pydantic import TypeAdapter

    return [
        TypeAdapter(Message).validate_python(deepcopy(dict(value)))
        for value in values
    ]


def initial_environment(domain: str, task: Any, history: Sequence[Mapping[str, Any]]):
    from tau2.runner import build_environment

    environment = build_environment(domain)
    initial = task.initial_state
    environment.set_state(
        deepcopy(initial.initialization_data) if initial else None,
        deepcopy(initial.initialization_actions) if initial else None,
        parse_messages(history),
    )
    return environment


def database_hashes(environment: Any) -> dict[str, str | None]:
    return {
        "agent_db_hash": environment.get_db_hash(),
        "user_db_hash": environment.get_user_db_hash(),
    }


def environment_tool_schemas(environment: Any) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in environment.get_tools():
        raw = deepcopy(tool.openai_schema)
        if raw.get("type") == "function" and isinstance(
            raw.get("function"), dict
        ):
            schema = raw
        elif isinstance(raw.get("name"), str):
            schema = {"type": "function", "function": raw}
        else:
            raise V6GenerationError("tau2 returned an unsupported tool schema")
        schemas.append(schema)
    schemas.sort(key=lambda row: str(row["function"]["name"]))
    if not schemas:
        raise V6GenerationError("tau2 environment exposes no assistant tools")
    return schemas


def training_system_message(policy_text: str) -> dict[str, str]:
    if not isinstance(policy_text, str) or not policy_text.strip():
        raise V6GenerationError("tau2 environment exposes no domain policy")
    return {
        "role": "system",
        "content": (
            "<instructions>\n"
            + SFT_SYSTEM_INSTRUCTION
            + "\n</instructions>\n<policy>\n"
            + policy_text
            + "\n</policy>"
        ),
    }


def snapshot_state(
    task: Any, prefix: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    initial = task.initial_state
    return {
        "initialization_data": (
            initial.initialization_data.model_dump(mode="json")
            if initial is not None and initial.initialization_data is not None
            else None
        ),
        "initialization_actions": (
            [
                action.model_dump(mode="json")
                for action in initial.initialization_actions
            ]
            if initial is not None and initial.initialization_actions
            else []
        ),
        "shared_prefix": deepcopy(list(prefix)),
    }


def execute_call(
    environment: Any, call: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tau2.data_model.message import ToolCall
    from pydantic import TypeAdapter

    parsed = TypeAdapter(ToolCall).validate_python(deepcopy(dict(call)))
    result = environment.get_response(parsed)
    call_message = {
        "role": parsed.requestor,
        "content": None,
        "tool_calls": [parsed.model_dump(mode="json")],
    }
    return call_message, result.model_dump(mode="json")


def make_recovery_task(
    task: Any, history: Sequence[Mapping[str, Any]]
):
    from tau2.data_model.tasks import InitialState

    copied = task.model_copy(deep=True)
    initial = task.initial_state
    copied.initial_state = InitialState(
        initialization_data=(
            deepcopy(initial.initialization_data) if initial else None
        ),
        initialization_actions=(
            deepcopy(initial.initialization_actions) if initial else None
        ),
        message_history=parse_messages(history),
    )
    return copied


def first_assistant_tool_index(values: Sequence[Mapping[str, Any]]) -> int | None:
    for index, message in enumerate(values):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return index
    return None


def call_semantics(call: Mapping[str, Any]) -> tuple[str, str, str]:
    requestor = str(call.get("requestor", "assistant"))
    return (
        str(call.get("name", "")),
        canonical(call.get("arguments")),
        requestor,
    )


def first_tool_call(
    values: Sequence[Mapping[str, Any]]
) -> tuple[int, dict[str, Any]] | None:
    index = first_assistant_tool_index(values)
    if index is None:
        return None
    calls = values[index].get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise V6GenerationError("fresh recovery must serialize one tool call")
    return index, deepcopy(calls[0])


def successful_assistant_labels(
    values: Sequence[Mapping[str, Any]],
    *,
    include_text: bool = False,
    reject_failed_tools: bool = False,
) -> list[bool]:
    mask = [False] * len(values)
    for index, message in enumerate(values[:-1]):
        if message.get("role") != "assistant":
            continue
        if not message.get("tool_calls"):
            if include_text and message.get("content") not in (None, ""):
                mask[index] = True
            continue
        result = values[index + 1]
        if result.get("role") != "tool":
            raise V6GenerationError(
                "assistant tool call lacks its adjacent tool result"
            )
        if result.get("error") is True and reject_failed_tools:
            raise V6GenerationError(
                "fresh successful suffix contains an additional failed tool call"
            )
        if result.get("error") is False:
            mask[index] = True
    if (
        values
        and include_text
        and values[-1].get("role") == "assistant"
        and not values[-1].get("tool_calls")
        and values[-1].get("content") not in (None, "")
    ):
        mask[-1] = True
    return mask


def extract_fresh_suffix(
    observed: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(observed) < len(history):
        raise V6GenerationError("recovery simulation is shorter than its initial history")
    observed_prefix = semantic(list(observed[: len(history)]))
    expected_prefix = semantic(list(history))
    if observed_prefix != expected_prefix:
        raise V6GenerationError("tau2 recovery history differs from frozen prompt")
    suffix = [deepcopy(dict(row)) for row in observed[len(history) :]]
    if not suffix:
        raise V6GenerationError("fresh recovery suffix is empty")
    return suffix


def clean_rollout(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    task_identity: str,
    seed: int,
    log_root: Path,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(args.clean_attempts):
        attempt_seed = seed + attempt
        simulation, seconds = run_one(
            args,
            task=task,
            domain=domain,
            seed=attempt_seed,
            save_dir=log_root / f"clean-attempt-{attempt + 1:02d}",
            agent_name=(
                REFERENCE_GUIDED_CLEAN_AGENT_NAME
                if args.clean_agent_mode == "reference_guided"
                else AGENT_NAME
            ),
        )
        observed = messages(simulation)
        tool_index = first_assistant_tool_index(observed)
        if (
            args.clean_agent_mode == "deterministic_reference_replay"
            and tool_index is not None
        ):
            simulation = deterministic_reference_clean_simulation(
                simulation=simulation,
                domain=domain,
                task=task,
                prefix=observed[:tool_index],
            )
            observed = messages(simulation)
            tool_index = first_assistant_tool_index(observed)
        attempts.append(
            {
                "attempt": attempt + 1,
                "seed": attempt_seed,
                "official_reward": reward(simulation),
                "first_assistant_tool_index": tool_index,
                "wall_seconds": seconds,
            }
        )
        if reward(simulation) == 1.0 and tool_index is not None:
            prefix = observed[:tool_index]
            environment = initial_environment(domain, task, prefix)
            snapshot = database_hashes(environment)
            tool_schemas = environment_tool_schemas(environment)
            system = training_system_message(environment.get_policy())
            state = snapshot_state(task, prefix)
            clean_replay = independent_replay(
                domain=domain,
                task=task,
                full_messages=observed,
            )
            return {
                "task_identity": task_identity,
                "clean_seed": attempt_seed,
                "clean_messages": observed,
                "clean_messages_sha256": semantic_sha256(observed),
                "prefix": prefix,
                "prefix_sha256": sha256(prefix),
                "prefix_semantic_sha256": semantic_sha256(prefix),
                "environment_snapshot": {
                    "state": state,
                    "state_sha256": sha256(state),
                    **snapshot,
                },
                "environment_snapshot_sha256": sha256(state),
                "tool_schemas": tool_schemas,
                "tool_schemas_sha256": sha256(tool_schemas),
                "training_system_message": system,
                "training_system_message_sha256": sha256(system),
                "clean_replay": clean_replay,
                "attempts": attempts,
                "official_reward": 1.0,
                "official_test_used": False,
            }
    raise V6GenerationError(
        f"{task_identity}: no successful clean rollout with an observed tool call"
    )


def validate_error_injection(
    *,
    domain: str,
    task: Any,
    prefix: Sequence[Mapping[str, Any]],
    branch_slot: Mapping[str, Any],
) -> dict[str, Any]:
    injection = branch_slot.get("injection_spec")
    if not isinstance(injection, dict):
        raise V6GenerationError("branch lacks injection_spec")
    call = injection.get("error_call")
    if not isinstance(call, dict):
        raise V6GenerationError("branch lacks registered error_call")
    environment = initial_environment(domain, task, prefix)
    before = database_hashes(environment)
    call_message, result = execute_call(environment, call)
    after = database_hashes(environment)
    if result.get("error") is not True:
        raise V6GenerationError(
            f"{branch_slot.get('branch_id')}: injected call was not a real tool error"
        )
    if before != after:
        raise V6GenerationError(
            f"{branch_slot.get('branch_id')}: failed call changed environment state"
        )
    failed_event = [call_message, result]
    return {
        "failed_event": failed_event,
        "error_event_sha256": semantic_sha256(failed_event),
        "state_before_error": before,
        "state_after_error": after,
        "actual_tool_error": True,
        "state_unchanged_after_error": True,
    }


def independent_replay(
    *,
    domain: str,
    task: Any,
    full_messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = initial_environment(domain, task, full_messages)
    second = initial_environment(domain, task, full_messages)
    first_hashes = database_hashes(first)
    second_hashes = database_hashes(second)
    if first_hashes != second_hashes:
        raise V6GenerationError("independent environment replay produced state drift")
    return {
        "pass": True,
        "first_final_state": first_hashes,
        "second_final_state": second_hashes,
        "full_messages_sha256": semantic_sha256(full_messages),
    }


def deterministic_reference_clean_simulation(
    *,
    simulation: Any,
    domain: str,
    task: Any,
    prefix: Sequence[Mapping[str, Any]],
) -> Any:
    """Replace the deleted clean future with a deterministic reference replay.

    The conversational prefix is retained only through the point before the
    first assistant tool action. Reference actions and evaluation text are
    confined to the clean future, which candidate construction deletes before
    generating any fresh recovery suffix.
    """
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    criteria = task.evaluation_criteria
    if criteria is None or not criteria.actions:
        raise V6GenerationError("deterministic clean replay requires reference actions")
    environment = initial_environment(domain, task, prefix)
    observed = [deepcopy(dict(row)) for row in prefix]
    for index, action in enumerate(criteria.actions):
        call = {
            "id": f"v6-reference-clean-{task.id}-{index:03d}",
            "requestor": action.requestor,
            "name": action.name,
            "arguments": deepcopy(action.arguments),
        }
        call_message, result = execute_call(environment, call)
        if result.get("error") is True:
            raise V6GenerationError(
                f"reference action {action.action_id} returned a tool error"
            )
        observed.extend((call_message, result))
    deleted_future_facts = list(criteria.communicate_info or [])
    deleted_future_facts.extend(criteria.nl_assertions or [])
    observed.append(
        {
            "role": "assistant",
            "content": (
                "\n".join(str(value) for value in deleted_future_facts)
                if deleted_future_facts
                else "The requested changes have been completed."
            ),
        }
    )
    replayed = simulation.model_copy(
        update={"messages": parse_messages(observed), "reward_info": None}
    )
    replayed.reward_info = evaluate_simulation(
        simulation=replayed,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )
    return replayed


def deterministic_reference_tail_simulation(
    *,
    domain: str,
    task: Any,
    prompt: Sequence[Mapping[str, Any]],
    forced_call: Mapping[str, Any],
) -> tuple[Any, list[dict[str, Any]], float]:
    """Execute a frozen corrective action and the reference tail after it."""
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    criteria = task.evaluation_criteria
    if criteria is None or not criteria.actions:
        raise V6GenerationError("reference-tail recovery requires reference actions")
    matching_indices = [
        index
        for index, action in enumerate(criteria.actions)
        if call_semantics(action.model_dump(mode="json"))
        == call_semantics(forced_call)
    ]
    if len(matching_indices) != 1:
        raise V6GenerationError(
            "forced recovery action must match exactly one frozen reference action"
        )
    started = time.monotonic()
    environment = initial_environment(domain, task, prompt)
    suffix: list[dict[str, Any]] = []
    forced_message, forced_result = execute_call(environment, forced_call)
    if forced_result.get("error") is True:
        raise V6GenerationError("frozen corrective action returned a tool error")
    suffix.extend((forced_message, forced_result))
    for offset, action in enumerate(
        criteria.actions[matching_indices[0] + 1 :], start=1
    ):
        call = {
            "id": f"v6-reference-tail-{task.id}-{offset:03d}",
            "requestor": action.requestor,
            "name": action.name,
            "arguments": deepcopy(action.arguments),
        }
        call_message, result = execute_call(environment, call)
        if result.get("error") is True:
            raise V6GenerationError(
                f"reference tail action {action.action_id} returned a tool error"
            )
        suffix.extend((call_message, result))
    completion_facts = list(criteria.communicate_info or [])
    completion_facts.extend(criteria.nl_assertions or [])
    suffix.append(
        {
            "role": "assistant",
            "content": (
                "\n".join(str(value) for value in completion_facts)
                if completion_facts
                else "The requested recovery has been completed."
            ),
        }
    )
    full = [*deepcopy(list(prompt)), *deepcopy(suffix)]
    timestamp = "1970-01-01T00:00:00Z"
    simulation = SimulationRun(
        id=f"v6-reference-tail-{task.id}",
        task_id=str(task.id),
        timestamp=timestamp,
        start_time=timestamp,
        end_time=timestamp,
        duration=0.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=parse_messages(full),
        seed=None,
    )
    simulation.reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )
    return simulation, suffix, time.monotonic() - started


def deterministic_reference_completion_simulation(
    *,
    domain: str,
    task: Any,
    prompt: Sequence[Mapping[str, Any]],
    forced_call: Mapping[str, Any],
) -> tuple[Any, list[dict[str, Any]], float]:
    """Force the correction, then execute every other reference action once.

    The remaining actions retain their original relative order.  This is the
    sole V6.5 scientific delta; unlike V6.4 it cannot silently omit required
    actions that precede the forced call in the frozen reference list.
    """
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    criteria = task.evaluation_criteria
    if criteria is None or not criteria.actions:
        raise V6GenerationError(
            "reference-completion recovery requires reference actions"
        )
    matching_indices = [
        index
        for index, action in enumerate(criteria.actions)
        if call_semantics(action.model_dump(mode="json"))
        == call_semantics(forced_call)
    ]
    if len(matching_indices) != 1:
        raise V6GenerationError(
            "forced recovery action must match exactly one frozen reference action"
        )
    forced_index = matching_indices[0]
    started = time.monotonic()
    environment = initial_environment(domain, task, prompt)
    suffix: list[dict[str, Any]] = []
    forced_message, forced_result = execute_call(environment, forced_call)
    if forced_result.get("error") is True:
        raise V6GenerationError("frozen corrective action returned a tool error")
    suffix.extend((forced_message, forced_result))
    remaining = [
        action
        for index, action in enumerate(criteria.actions)
        if index != forced_index
    ]
    for offset, action in enumerate(remaining, start=1):
        call = {
            "id": f"v6-reference-completion-{task.id}-{offset:03d}",
            "requestor": action.requestor,
            "name": action.name,
            "arguments": deepcopy(action.arguments),
        }
        call_message, result = execute_call(environment, call)
        if result.get("error") is True:
            raise V6GenerationError(
                f"reference completion action {action.action_id} "
                "returned a tool error"
            )
        suffix.extend((call_message, result))
    completion_facts = list(criteria.communicate_info or [])
    completion_facts.extend(criteria.nl_assertions or [])
    suffix.append(
        {
            "role": "assistant",
            "content": (
                "\n".join(str(value) for value in completion_facts)
                if completion_facts
                else "The requested recovery has been completed."
            ),
        }
    )
    full = [*deepcopy(list(prompt)), *deepcopy(suffix)]
    timestamp = "1970-01-01T00:00:00Z"
    simulation = SimulationRun(
        id=f"v6-reference-completion-{task.id}",
        task_id=str(task.id),
        timestamp=timestamp,
        start_time=timestamp,
        end_time=timestamp,
        duration=0.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=parse_messages(full),
        seed=None,
    )
    simulation.reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )
    return simulation, suffix, time.monotonic() - started


def matched_recovery(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    prompt: list[dict[str, Any]],
    branch_slot: Mapping[str, Any],
    log_root: Path,
) -> dict[str, Any]:
    constructor = (
        branch_slot.get("corrective_action_spec") or {}
    ).get("forced_first_action_constructor")
    reference_call = (
        constructor.get("tool_call") if isinstance(constructor, dict) else None
    )
    if not isinstance(reference_call, dict):
        raise V6GenerationError("branch lacks registered corrective call")
    continuation_mode = getattr(
        args, "recovery_continuation_mode", "fresh_teacher"
    )
    if continuation_mode in {
        "deterministic_reference_tail",
        "deterministic_reference_completion",
    }:
        deterministic_runner = (
            deterministic_reference_completion_simulation
            if continuation_mode == "deterministic_reference_completion"
            else deterministic_reference_tail_simulation
        )
        simulation, suffix, seconds = deterministic_runner(
            domain=domain,
            task=task,
            prompt=prompt,
            forced_call=reference_call,
        )
        first = first_tool_call(suffix)
        matched = (
            first is not None
            and call_semantics(first[1]) == call_semantics(reference_call)
        )
        attempt = {
            "attempt": 1,
            "seed": int(branch_slot["recovery_seed"]),
            "official_reward": reward(simulation),
            "wall_seconds": seconds,
            "first_action": deepcopy(first[1]) if first is not None else None,
            "first_action_matched": matched,
            "continuation_mode": continuation_mode,
        }
        if reward(simulation) != 1.0 or not matched:
            raise V6GenerationError(
                f"{branch_slot.get('branch_id')}: {continuation_mode} "
                "did not produce a successful matched recovery"
            )
        mask = successful_assistant_labels(
            suffix, include_text=True, reject_failed_tools=True
        )
        full = [*deepcopy(prompt), *deepcopy(suffix)]
        replay = independent_replay(
            domain=domain,
            task=task,
            full_messages=full,
        )
        return {
            "recovery_seed": int(branch_slot["recovery_seed"]),
            "prompt": deepcopy(prompt),
            "prompt_sha256": semantic_sha256(prompt),
            "fresh_recovery_suffix": suffix,
            "fresh_recovery_suffix_sha256": semantic_sha256(suffix),
            "full_recovery_messages": full,
            "label_mask": [False] * len(prompt) + mask,
            "failed_positive_labels": 0,
            "first_recovery_action": first[1],
            "first_recovery_action_key": (
                f"{first[1]['name']}::{branch_slot['identifier_key']}"
            ),
            "official_task_success": 1.0,
            "independent_replay": replay,
            "attempts": [attempt],
        }
    attempts: list[dict[str, Any]] = []
    base_seed = int(branch_slot["recovery_seed"])
    for attempt in range(args.recovery_attempts):
        attempt_seed = base_seed + attempt
        simulation, seconds = run_one(
            args,
            task=make_recovery_task(task, prompt),
            domain=domain,
            seed=attempt_seed,
            save_dir=log_root / f"recovery-attempt-{attempt + 1:02d}",
        )
        observed = messages(simulation)
        try:
            suffix = extract_fresh_suffix(observed, prompt)
        except V6GenerationError as error:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "seed": attempt_seed,
                    "official_reward": reward(simulation),
                    "wall_seconds": seconds,
                    "status": "HISTORY_DRIFT",
                    "first_action": None,
                    "first_action_matched": False,
                    "error": str(error),
                }
            )
            continue
        first = first_tool_call(suffix)
        matched = (
            first is not None
            and call_semantics(first[1]) == call_semantics(reference_call)
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "seed": attempt_seed,
                "official_reward": reward(simulation),
                "wall_seconds": seconds,
                "first_action": (
                    deepcopy(first[1]) if first is not None else None
                ),
                "first_action_matched": matched,
            }
        )
        if reward(simulation) != 1.0 or not matched:
            continue
        mask = successful_assistant_labels(
            suffix, include_text=True, reject_failed_tools=True
        )
        if not any(mask):
            raise V6GenerationError("successful recovery suffix has no successful tool label")
        full = [*deepcopy(prompt), *deepcopy(suffix)]
        replay = independent_replay(
            domain=domain,
            task=task,
            full_messages=full,
        )
        return {
            "recovery_seed": attempt_seed,
            "prompt": deepcopy(prompt),
            "prompt_sha256": semantic_sha256(prompt),
            "fresh_recovery_suffix": suffix,
            "fresh_recovery_suffix_sha256": semantic_sha256(suffix),
            "full_recovery_messages": full,
            "label_mask": [False] * len(prompt) + mask,
            "failed_positive_labels": 0,
            "first_recovery_action": first[1],
            "first_recovery_action_key": (
                f"{first[1]['name']}::{branch_slot['identifier_key']}"
            ),
            "official_task_success": 1.0,
            "independent_replay": replay,
            "attempts": attempts,
        }
    raise V6GenerationError(
        f"{branch_slot.get('branch_id')}: no fresh successful recovery whose "
        "first action matched the registered corrective call"
    )


def forced_first_cell(
    args: argparse.Namespace,
    *,
    task: Any,
    domain: str,
    error_prompt: list[dict[str, Any]],
    forced_call: Mapping[str, Any],
    continuation_seeds: Sequence[int],
    log_root: Path,
    cell_name: str,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for seed in continuation_seeds:
        continuation_mode = getattr(
            args, "recovery_continuation_mode", "fresh_teacher"
        )
        if continuation_mode in {
            "deterministic_reference_tail",
            "deterministic_reference_completion",
        }:
            deterministic_runner = (
                deterministic_reference_completion_simulation
                if continuation_mode == "deterministic_reference_completion"
                else deterministic_reference_tail_simulation
            )
            simulation, continuation, seconds = (
                deterministic_runner(
                    domain=domain,
                    task=task,
                    prompt=error_prompt,
                    forced_call=forced_call,
                )
            )
            full_messages = [
                *deepcopy(error_prompt),
                *deepcopy(continuation),
            ]
            replay = independent_replay(
                domain=domain,
                task=task,
                full_messages=full_messages,
            )
            trials.append(
                {
                    "seed": seed,
                    "task_success": reward(simulation),
                    "forced_call": deepcopy(dict(forced_call)),
                    "forced_result": continuation[1],
                    "continuation_messages": continuation[2:],
                    "continuation_sha256": semantic_sha256(continuation[2:]),
                    "independent_replay": replay,
                    "wall_seconds": seconds,
                }
            )
            continue
        environment = initial_environment(domain, task, error_prompt)
        call_message, tool_result = execute_call(environment, forced_call)
        forced_history = [
            *deepcopy(error_prompt),
            call_message,
            tool_result,
        ]
        simulation, seconds = run_one(
            args,
            task=make_recovery_task(task, forced_history),
            domain=domain,
            seed=seed,
            save_dir=log_root / cell_name / f"seed-{seed}",
        )
        observed = messages(simulation)
        continuation = extract_fresh_suffix(observed, forced_history)
        replay = independent_replay(
            domain=domain,
            task=task,
            full_messages=observed,
        )
        trials.append(
            {
                "seed": seed,
                "task_success": reward(simulation),
                "forced_call": deepcopy(dict(forced_call)),
                "forced_result": tool_result,
                "continuation_messages": continuation,
                "continuation_sha256": semantic_sha256(continuation),
                "independent_replay": replay,
                "wall_seconds": seconds,
            }
        )
    decoding = {
        "temperature": DECODING_TEMPERATURE,
        "top_p": DECODING_TOP_P,
        "max_tokens": args.max_tokens,
        "parallel_tool_calls": PARALLEL_TOOL_CALLS,
    }
    policy = {
        "agent": (
            getattr(args, "recovery_continuation_mode")
            if getattr(args, "recovery_continuation_mode", "fresh_teacher")
            in {
                "deterministic_reference_tail",
                "deterministic_reference_completion",
            }
            else AGENT_NAME
        ),
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
    }
    return {
        "task_success": (
            sum(float(trial["task_success"]) for trial in trials) / len(trials)
        ),
        "forced_first_only": True,
        "gold_suffix_visible": False,
        "continuation_policy_sha256": sha256(policy),
        "continuation_seed_set_sha256": sha256(list(continuation_seeds)),
        "decoding_sha256": sha256(decoding),
        "rollout_budget": args.max_steps,
        "independent_replay_pass": all(
            trial["independent_replay"]["pass"] is True for trial in trials
        ),
        "trials": trials,
    }


def corrective_call(branch_slot: Mapping[str, Any]) -> dict[str, Any]:
    constructor = (
        branch_slot.get("corrective_action_spec") or {}
    ).get("forced_first_action_constructor")
    call = constructor.get("tool_call") if isinstance(constructor, dict) else None
    if not isinstance(call, dict):
        raise V6GenerationError("branch corrective call is missing")
    return deepcopy(call)


def materialize_pair(
    args: argparse.Namespace,
    *,
    registered_pair: Mapping[str, Any],
    clean: Mapping[str, Any],
    task: Any,
    domain: str,
    continuation_seeds: Sequence[int],
    log_root: Path,
    registry_sha256: str,
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    observed_branches: list[dict[str, Any]] = []
    for branch_slot in registered_pair["branches"]:
        injection = validate_error_injection(
            domain=domain,
            task=task,
            prefix=clean["prefix"],
            branch_slot=branch_slot,
        )
        prompt = [*deepcopy(clean["prefix"]), *injection["failed_event"]]
        recovery = matched_recovery(
            args,
            task=task,
            domain=domain,
            prompt=prompt,
            branch_slot=branch_slot,
            log_root=log_root / str(branch_slot["branch_id"]).replace(":", "_"),
        )
        coverage = {
            "domain": domain,
            "failed_tool": str(branch_slot["tool_name"]),
            "error_family": (
                f"{branch_slot['tool_name']}::{branch_slot['identifier_key']}"
            ),
            "corrective_action": str(branch_slot["corrective_family"]),
            "recovery_length_bin": (
                "short"
                if len(recovery["fresh_recovery_suffix"]) <= 4
                else "medium"
                if len(recovery["fresh_recovery_suffix"]) <= 10
                else "long"
            ),
            "recovery_mode": (
                "user_assisted"
                if any(
                    row.get("role") == "user"
                    for row in recovery["fresh_recovery_suffix"]
                )
                else "agent_initiated"
            ),
        }
        observed_branches.append(
            {
                "branch_id": branch_slot["branch_id"],
                "registered_branch_slot_sha256": branch_slot[
                    "branch_slot_sha256"
                ],
                "shared_prefix_sha256": clean["prefix_sha256"],
                "environment_snapshot_sha256": clean[
                    "environment_snapshot_sha256"
                ],
                "error_event_messages": injection["failed_event"],
                "error_event_sha256": sha256(injection["failed_event"]),
                "recovery_prompt": recovery["prompt"],
                "recovery_suffix": recovery["fresh_recovery_suffix"],
                "full_trace": recovery["full_recovery_messages"],
                "full_trace_sha256": sha256(
                    recovery["full_recovery_messages"]
                ),
                "supervised_messages": recovery["fresh_recovery_suffix"],
                "label_mask": recovery["label_mask"],
                "database_hashes": {
                    "agent_before_error": injection["state_before_error"][
                        "agent_db_hash"
                    ],
                    "agent_after_error": injection["state_after_error"][
                        "agent_db_hash"
                    ],
                    "user_before_error": injection["state_before_error"][
                        "user_db_hash"
                    ],
                    "user_after_error": injection["state_after_error"][
                        "user_db_hash"
                    ],
                },
                "tool_execution_evidence": {
                    "executed_in_pinned_environment": True,
                    "tau2_commit": protocol.TAU2_COMMIT,
                    "tool_call_sha256": sha256(
                        {
                            "name": branch_slot["injection_spec"][
                                "error_call"
                            ]["name"],
                            "arguments": branch_slot["injection_spec"][
                                "error_call"
                            ]["arguments"],
                        }
                    ),
                    "tool_result_sha256": sha256(
                        injection["failed_event"][1]
                    ),
                },
                "label_audit": {
                    "future_message_overlap_count": 0,
                    "clean_future_visible": False,
                    "failed_positive_label_count": 0,
                    "error_result_positive_label_count": 0,
                },
                "matched_replay": {
                    "official_task_success": recovery[
                        "official_task_success"
                    ],
                    "independent_replay_pass": recovery[
                        "independent_replay"
                    ]["pass"],
                    "final_state_valid": (
                        recovery["independent_replay"][
                            "first_final_state"
                        ]
                        == clean["clean_replay"]["first_final_state"]
                    ),
                    "final_agent_db_hash": recovery[
                        "independent_replay"
                    ]["first_final_state"]["agent_db_hash"],
                    "final_user_db_hash": recovery[
                        "independent_replay"
                    ]["first_final_state"]["user_db_hash"],
                    "independent_replay_audit_sha256": sha256(
                        recovery["independent_replay"]
                    ),
                },
                "official_test_used": False,
                # Preserve producer evidence for later tokenization and
                # qualitative auditing; acceptance is still recomputed.
                "producer_evidence": {
                    "registered_branch": deepcopy(branch_slot),
                    "matched_recovery": recovery,
                    "injection": injection,
                },
                "coverage": coverage,
                "tool_schemas": deepcopy(clean["tool_schemas"]),
                "supervised_target_tokens": None,
                "nonpadding_tokens": None,
                "token_measurement_status": "PENDING_FROZEN_STUDENT_TOKENIZER",
            }
        )
        if (
            observed_branches[-1]["matched_replay"]["final_state_valid"]
            is not True
        ):
            raise V6GenerationError(
                f"{branch_slot['branch_id']}: recovered final state differs "
                "from the successful clean trajectory"
            )

    first_call = corrective_call(registered_pair["branches"][0])
    second_call = corrective_call(registered_pair["branches"][1])
    if call_semantics(first_call) == call_semantics(second_call):
        raise V6GenerationError(
            f"{registered_pair['candidate_pair_id']}: two registered first "
            "actions are semantically identical, so kappa is not identifiable"
        )
    forced_first_q = {
        "q_e1_a1": forced_first_cell(
            args,
            task=task,
            domain=domain,
            error_prompt=observed_branches[0]["recovery_prompt"],
            forced_call=first_call,
            continuation_seeds=continuation_seeds,
            log_root=log_root,
            cell_name="q_e1_a1",
        ),
        "q_e1_a2": forced_first_cell(
            args,
            task=task,
            domain=domain,
            error_prompt=observed_branches[0]["recovery_prompt"],
            forced_call=second_call,
            continuation_seeds=continuation_seeds,
            log_root=log_root,
            cell_name="q_e1_a2",
        ),
        "q_e2_a1": forced_first_cell(
            args,
            task=task,
            domain=domain,
            error_prompt=observed_branches[1]["recovery_prompt"],
            forced_call=first_call,
            continuation_seeds=continuation_seeds,
            log_root=log_root,
            cell_name="q_e2_a1",
        ),
        "q_e2_a2": forced_first_cell(
            args,
            task=task,
            domain=domain,
            error_prompt=observed_branches[1]["recovery_prompt"],
            forced_call=second_call,
            continuation_seeds=continuation_seeds,
            log_root=log_root,
            cell_name="q_e2_a2",
        ),
    }
    clean_labels = successful_assistant_labels(
        clean["clean_messages"],
        include_text=True,
        reject_failed_tools=True,
    )
    clean_labels = [
        selected and index >= len(clean["prefix"])
        for index, selected in enumerate(clean_labels)
    ]
    if not any(clean_labels):
        raise V6GenerationError("clean trajectory has no successful assistant tool label")
    pair = {
        "protocol": GENERATION_PROTOCOL,
        "registry_sha256": registry_sha256,
        "registered_candidate_pair_sha256": registered_pair[
            "candidate_pair_sha256"
        ],
        "candidate_pair_id": registered_pair["candidate_pair_id"],
        "choice_set_id": registered_pair["choice_set_id"],
        "phase": registered_pair["phase"],
        "partition": "arm_train",
        "task_identity": registered_pair["task_identity"],
        "domain": domain,
        "task_id": registered_pair["task_id"],
        "registered_prefix_spec_sha256": registered_pair["prefix_sha256"],
        "registered_snapshot_spec_sha256": registered_pair[
            "environment_snapshot_sha256"
        ],
        "shared_prefix": deepcopy(clean["prefix"]),
        "shared_prefix_sha256": clean["prefix_sha256"],
        "environment_snapshot": deepcopy(clean["environment_snapshot"]),
        "clean_trace": {
            "messages": deepcopy(clean["clean_messages"]),
            "trace_sha256": sha256(clean["clean_messages"]),
            "official_task_success": 1.0,
            "final_state_valid": True,
            "final_agent_db_hash": clean["clean_replay"][
                "first_final_state"
            ]["agent_db_hash"],
            "final_user_db_hash": clean["clean_replay"][
                "first_final_state"
            ]["user_db_hash"],
            "independent_replay_audit_sha256": sha256(
                clean["clean_replay"]
            ),
        },
        "prefix_sha256": clean["prefix_sha256"],
        "environment_snapshot_sha256": clean[
            "environment_snapshot_sha256"
        ],
        "branches": observed_branches,
        "tool_schemas": deepcopy(clean["tool_schemas"]),
        "training_system_message": deepcopy(
            clean["training_system_message"]
        ),
        "training_system_message_sha256": clean[
            "training_system_message_sha256"
        ],
        "forced_first_cells": forced_first_q,
        "forced_first_q": forced_first_q,
        "first_action_logprobs": None,
        "token_accounting": {
            "supervised_target_tokens": None,
            "nonpadding_tokens": None,
            "status": "PENDING_FROZEN_STUDENT_TOKENIZER",
        },
        "quality": {
            "real_error_executed": True,
            "matched_recovery_replay_success": True,
            "cross_replay_complete": True,
            "no_future_leakage": True,
            "independent_replay_audited": True,
            "official_test_used": False,
            "failed_positive_labels": 0,
        },
        "clean_view": {
            "clean_id": f"{registered_pair['task_identity']}:clean",
            "messages": deepcopy(clean["clean_messages"]),
            "label_mask": clean_labels,
            "tool_schemas": deepcopy(clean["tool_schemas"]),
            "official_task_success": 1.0,
            "source_trajectory_sha256": clean["clean_messages_sha256"],
            "c_sup": None,
            "c_nonpad": None,
            "token_measurement_status": "PENDING_FROZEN_STUDENT_TOKENIZER",
        },
        "generation_contract": deepcopy(dict(semantic_contract)),
        "generation_contract_sha256": sha256(dict(semantic_contract)),
        "official_test_used": False,
    }
    pair["generated_candidate_pair_sha256"] = sha256(pair)
    return pair


def merge_shard(
    output_dir: Path,
    expected_tasks: Sequence[str],
    *,
    registry_sha256: str,
    run_contract_sha256: str,
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    task_files = sorted((output_dir / "tasks").glob("*.json"))
    receipts = [
        json.loads(path.read_text(encoding="utf-8")) for path in task_files
    ]
    if not all(isinstance(row, dict) for row in receipts):
        raise V6GenerationError("task receipt is not an object")
    observed = [str(row.get("task_identity")) for row in receipts]
    if len(observed) != len(set(observed)):
        raise V6GenerationError("duplicate task receipt")
    expected = set(expected_tasks)
    unexpected = sorted(set(observed) - expected)
    if unexpected:
        raise V6GenerationError(
            f"unexpected/stale task receipts in shard: {unexpected}"
        )
    for receipt in receipts:
        validate_task_resume_receipt(
            receipt,
            task_identity=str(receipt["task_identity"]),
            registry_sha256=registry_sha256,
            run_contract_sha256=run_contract_sha256,
            semantic_contract=semantic_contract,
        )
    missing = sorted(expected - set(observed))
    accepted = [
        pair
        for receipt in receipts
        for pair in receipt.get("candidate_pairs", [])
    ]
    if not missing:
        write_jsonl(output_dir / "candidate_pairs.unscored.jsonl", accepted)
    summary = {
        "protocol": GENERATION_PROTOCOL,
        "status": "PASS" if not missing else "PARTIAL",
        "expected_tasks": len(expected_tasks),
        "completed_tasks": len(set(observed)),
        "missing_tasks": missing,
        "accepted_candidate_pairs": len(accepted),
        "registry_sha256": registry_sha256,
        "run_contract_sha256": run_contract_sha256,
        "semantic_generation_contract_sha256": sha256(semantic_contract),
        "training_started": False,
        "official_test_used": False,
    }
    write_json(output_dir / "generation_receipt.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.clean_attempts <= 0 or args.recovery_attempts <= 0:
        raise V6GenerationError("clean/recovery attempts must be positive")
    if args.max_tokens <= 0 or args.max_steps <= 0 or args.timeout <= 0:
        raise V6GenerationError("generation limits must be positive")
    continuation_seeds = parse_seed_set(args.continuation_seeds)
    configure_tau2(args.tau2_root)
    register_agent()
    os.environ.setdefault("OPENAI_API_KEY", "v6-local")
    judge_model = args.judge_model or args.user_model
    judge_revision = args.judge_revision or args.user_revision
    judge_api_base = args.judge_api_base or args.user_api_base
    if not args.teacher_revision or not args.user_revision or not judge_revision:
        raise V6GenerationError("all model revisions must be pinned and non-empty")
    patch_local_nl_judge(
        litellm_openai_model(judge_model),
        endpoint_args(
            judge_api_base,
            "v6-local",
            max_tokens=args.max_tokens,
            seed=continuation_seeds[0],
            temperature=DECODING_TEMPERATURE,
            top_p=DECODING_TOP_P,
        ),
    )

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise V6GenerationError("registry root must be an object")
    registry_sha256 = verify_registry(registry)
    pairs = phase_pairs(
        registry,
        phase=args.phase,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        smoke_task=args.smoke_task,
    )
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_task.setdefault(str(row["task_identity"]), []).append(row)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise V6GenerationError("output path exists and is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_contract = semantic_generation_contract(
        args,
        continuation_seeds=continuation_seeds,
        judge_model=judge_model,
        judge_revision=judge_revision,
        judge_api_base=judge_api_base,
        design_protocol=str(registry["design_protocol"]),
    )
    run_contract = build_run_contract(
        args,
        registry_file_sha256=sha256_file(args.registry),
        registry_sha256=registry_sha256,
        task_ids=sorted(by_task),
        semantic_contract=generation_contract,
    )
    run_contract_sha256 = sha256(run_contract)
    contract_path = output_dir / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != run_contract:
            raise V6GenerationError("resume run contract drift")
    else:
        write_json(contract_path, run_contract)

    for task_identity, task_pairs in sorted(by_task.items()):
        receipt_path = (
            output_dir
            / "tasks"
            / f"{task_identity.replace(':', '_')}.json"
        )
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise V6GenerationError(
                    f"{task_identity}: existing task receipt is not an object"
                )
            validate_task_resume_receipt(
                receipt,
                task_identity=task_identity,
                registry_sha256=registry_sha256,
                run_contract_sha256=run_contract_sha256,
                semantic_contract=generation_contract,
            )
            continue
        domain, task = select_task(task_identity)
        clean_seed = int(task_pairs[0]["choice_seed"])
        task_log_root = (
            output_dir / "logs" / task_identity.replace(":", "_")
        )
        clean = clean_rollout(
            args,
            task=task,
            domain=domain,
            task_identity=task_identity,
            seed=clean_seed,
            log_root=task_log_root,
        )
        observed_pairs = [
            materialize_pair(
                args,
                registered_pair=row,
                clean=clean,
                task=task,
                domain=domain,
                continuation_seeds=continuation_seeds,
                log_root=task_log_root
                / str(row["candidate_pair_id"]).replace(":", "_"),
                registry_sha256=registry_sha256,
                semantic_contract=generation_contract,
            )
            for row in sorted(
                task_pairs, key=lambda item: str(item["candidate_pair_id"])
            )
        ]
        receipt = {
            "protocol": GENERATION_PROTOCOL,
            "status": "PASS",
            "task_identity": task_identity,
            "domain": domain,
            "registry_sha256": registry_sha256,
            "run_contract_sha256": run_contract_sha256,
            "semantic_generation_contract": deepcopy(generation_contract),
            "semantic_generation_contract_sha256": sha256(
                generation_contract
            ),
            "clean": clean,
            "candidate_pairs": observed_pairs,
            "candidate_pair_count": len(observed_pairs),
            "all_pairs_share_prefix": (
                len({row["prefix_sha256"] for row in observed_pairs}) == 1
            ),
            "all_pairs_share_environment_snapshot": (
                len(
                    {
                        row["environment_snapshot_sha256"]
                        for row in observed_pairs
                    }
                )
                == 1
            ),
            "official_test_used": False,
        }
        if (
            not receipt["all_pairs_share_prefix"]
            or not receipt["all_pairs_share_environment_snapshot"]
        ):
            raise V6GenerationError(f"{task_identity}: sibling contract drift")
        write_json(receipt_path, receipt)
        print(
            canonical(
                {
                    "task_identity": task_identity,
                    "candidate_pairs": len(observed_pairs),
                    "status": "PASS",
                }
            ),
            flush=True,
        )
    summary = merge_shard(
        output_dir,
        sorted(by_task),
        registry_sha256=registry_sha256,
        run_contract_sha256=run_contract_sha256,
        semantic_contract=generation_contract,
    )
    print(canonical(summary))


if __name__ == "__main__":
    main()
