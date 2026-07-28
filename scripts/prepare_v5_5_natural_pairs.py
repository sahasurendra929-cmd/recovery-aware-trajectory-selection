#!/usr/bin/env python3
"""Build and independently audit natural-conversation V5.5 pairs.

``build`` consumes successful tau2 inner-train simulations.  For each eligible
task it deterministically selects one successful natural conversation and one
read-only identifier lookup already present in that conversation.  It inserts
two deterministic invalid-identifier calls immediately before the correct
call, deletes the original future from the recovery prompt, and retains the
original successful suffix as the target.

``audit`` reconstructs fresh tau2 environments and independently replays both
the clean path and every counterfactual path.  It authorizes SFT only when at
least 24 distinct tasks and 48 pairs across airline and retail pass, with at
most two pairs per task and zero failed positive labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

try:
    import v5_5_protocol as protocol
    import v5_5_full_protocol as full
except ModuleNotFoundError:
    from scripts import v5_5_protocol as protocol
    from scripts import v5_5_full_protocol as full


NATURAL_PROTOCOL = "v5_5_natural_counterfactual_pairs_v1"
PAIR_LIMIT_PER_TASK = 2


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def configure_tau2(root: Path) -> None:
    source = root.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))


def initialize_environment(domain: str, task: Any) -> Any:
    from tau2.registry import registry

    environment = registry.get_env_constructor(domain)()
    initial = task.initial_state
    environment.set_state(
        initialization_data=(
            deepcopy(initial.initialization_data) if initial else None
        ),
        initialization_actions=(
            deepcopy(initial.initialization_actions) if initial else None
        ),
        message_history=(
            deepcopy(initial.message_history)
            if initial and initial.message_history
            else []
        ),
    )
    return environment


def execute(environment: Any, call: dict[str, Any]) -> dict[str, Any]:
    from tau2.data_model.message import ToolCall

    return environment.get_response(
        ToolCall.model_validate(call)
    ).model_dump(mode="json")


def compact_call(call: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise RuntimeError(f"{where}: tool call is not an object")
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = call.get("name")
        arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{where}: arguments are not JSON") from error
    call_id = call.get("id")
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(name, str)
        or not name
        or not isinstance(arguments, dict)
    ):
        raise RuntimeError(f"{where}: incomplete tool call")
    return {
        "id": call_id,
        "name": name,
        "arguments": deepcopy(arguments),
        "requestor": call.get("requestor", "assistant"),
    }


def tool_link_id(message: dict[str, Any]) -> str | None:
    value = message.get("tool_call_id", message.get("id"))
    return value if isinstance(value, str) and value else None


def simulation_reward_one(simulation: dict[str, Any]) -> bool:
    value = (simulation.get("reward_info") or {}).get("reward")
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) == 1.0
    )


def analyze_simulation(
    simulation: dict[str, Any],
    *,
    domain: str,
    allowed_task_ids: set[str],
) -> dict[str, Any] | None:
    task_id = str(simulation.get("task_id"))
    messages = simulation.get("messages")
    if (
        task_id not in allowed_task_ids
        or not simulation_reward_one(simulation)
        or not isinstance(messages, list)
        or not messages
    ):
        return None
    calls: list[dict[str, Any]] = []
    call_sites: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return None
        raw_calls = message.get("tool_calls") or []
        if not raw_calls:
            continue
        if (
            message.get("role") != "assistant"
            or len(raw_calls) != 1
            or message.get("content") not in (None, "")
            or index + 1 >= len(messages)
            or messages[index + 1].get("role") != "tool"
            or messages[index + 1].get("error") is not False
        ):
            return None
        try:
            call = compact_call(
                raw_calls[0],
                where=f"{domain}:{task_id}:message-{index}",
            )
        except RuntimeError:
            return None
        if (
            call["requestor"] != "assistant"
            or tool_link_id(messages[index + 1]) not in (None, call["id"])
        ):
            return None
        calls.append(call)
        call_sites.append(
            {
                "message_index": index,
                "action_index": len(calls) - 1,
                "requestor": "assistant",
                "name": call["name"],
                "arguments": call["arguments"],
            }
        )
    if not calls or not any(message.get("role") == "user" for message in messages):
        return None
    eligible = protocol.eligible_reference_actions(domain, call_sites)
    if not eligible:
        return None
    task_identity = f"{domain}:{task_id}"
    site = protocol.select_site(task_identity, eligible)
    selected = call_sites[int(site["action_index"])]
    return {
        "domain": domain,
        "task_id": task_id,
        "task_identity": task_identity,
        "messages": deepcopy(messages),
        "calls": calls,
        "selected_message_index": selected["message_index"],
        "source_simulation_sha256": sha256(simulation),
        **site,
    }


def load_candidates(
    inputs: list[tuple[str, Path]],
    *,
    split: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], Counter]:
    candidates: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    exclusions: Counter = Counter()
    for domain, path in inputs:
        if domain not in {"retail", "airline"}:
            raise RuntimeError(f"unsupported domain {domain}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        simulations = payload.get("simulations") if isinstance(payload, dict) else None
        if not isinstance(simulations, list):
            raise RuntimeError(f"{path}: simulations are missing")
        hashes[str(path.resolve())] = sha256_file(path)
        allowed = {
            str(value)
            for value in split["domains"][domain]["inner_train_ids"]
        }
        for simulation in simulations:
            if not isinstance(simulation, dict):
                exclusions["malformed_simulation"] += 1
                continue
            analyzed = analyze_simulation(
                simulation,
                domain=domain,
                allowed_task_ids=allowed,
            )
            if analyzed is None:
                exclusions["not_successful_natural_pair_candidate"] += 1
            else:
                candidates.append(analyzed)
    return candidates, hashes, exclusions


def choose_one_per_task(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["task_identity"]].append(row)
    selected = []
    for identity, rows in grouped.items():
        selected.append(
            min(
                rows,
                key=lambda row: (
                    hashlib.sha256(
                        (
                            f"{NATURAL_PROTOCOL}|{full.TRAINING_SEEDS[0]}|"
                            f"{identity}|{row['source_simulation_sha256']}"
                        ).encode()
                    ).hexdigest(),
                    row["source_simulation_sha256"],
                ),
            )
        )
    return sorted(selected, key=lambda row: row["task_identity"])


def task_map(domains: set[str]) -> dict[str, Any]:
    from tau2.registry import registry

    result = {}
    for domain in domains:
        for task in registry.get_tasks_loader(domain)(None):
            result[f"{domain}:{task.id}"] = task
    return result


def replay(
    *,
    domain: str,
    task: Any,
    calls: list[dict[str, Any]],
    injection_index: int | None = None,
    injected_call: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    environment = initialize_environment(domain, task)
    results = []
    for index, call in enumerate(calls):
        if index == injection_index:
            if injected_call is None:
                raise RuntimeError("injected call is missing")
            results.append(execute(environment, injected_call))
        results.append(execute(environment, call))
    return results, environment


def build_pair(
    source: dict[str, Any],
    *,
    task: Any,
    mutation_variant: int,
) -> dict[str, Any]:
    domain = source["domain"]
    action_index = int(source["action_index"])
    clean_call = deepcopy(source["calls"][action_index])
    injected_call = deepcopy(clean_call)
    injected_call["id"] = (
        f"v55-natural-error-{source['task_id']}-{mutation_variant}"
    )
    key = source["identifier_key"]
    injected_call["arguments"][key] = protocol.mutate_identifier(
        str(clean_call["arguments"][key]),
        mutation_variant,
    )
    clean_results, clean_env = replay(
        domain=domain,
        task=task,
        calls=source["calls"],
    )
    recovery_results, recovery_env = replay(
        domain=domain,
        task=task,
        calls=source["calls"],
        injection_index=action_index,
        injected_call=injected_call,
    )
    if any(result.get("error") is True for result in clean_results):
        raise RuntimeError("natural clean action path is not replayable")
    injected_result = recovery_results[action_index]
    correction_result = recovery_results[action_index + 1]
    if (
        injected_result.get("error") is not True
        or correction_result.get("error") is not False
        or sum(result.get("error") is True for result in recovery_results) != 1
    ):
        raise RuntimeError("natural counterfactual does not create one safe error")
    message_index = int(source["selected_message_index"])
    clean_prefix = deepcopy(source["messages"][:message_index])
    supervised = deepcopy(source["messages"][message_index:])
    failed_event = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [injected_call],
        },
        deepcopy(injected_result),
    ]
    clean_agent = clean_env.get_db_hash()
    clean_user = clean_env.get_user_db_hash()
    recovery_agent = recovery_env.get_db_hash()
    recovery_user = recovery_env.get_user_db_hash()
    pair_id = (
        f"{source['task_identity']}:natural-counterfactual:"
        f"{mutation_variant + 1}"
    )
    return {
        "protocol": NATURAL_PROTOCOL,
        "pair_id": pair_id,
        "task_identity": source["task_identity"],
        "domain": domain,
        "task_id": source["task_id"],
        "mutation_variant": mutation_variant,
        "identifier_key": key,
        "fault_family": f"{domain}_natural_identifier_lookup",
        "clean_call": clean_call,
        "injected_call": injected_call,
        "injected_result": injected_result,
        "correction_call": clean_call,
        "correction_result": correction_result,
        "clean_prefix": clean_prefix,
        "recovery_prompt": [*deepcopy(clean_prefix), *failed_event],
        "failed_event": failed_event,
        "supervised_messages": supervised,
        "source_call_sequence": deepcopy(source["calls"]),
        "selected_action_index": action_index,
        "source_simulation_sha256": source["source_simulation_sha256"],
        "supervision_starts_after_error": True,
        "clean_future_present_in_recovery_prompt": False,
        "clean_prefix_sha256": protocol.semantic_sha256(clean_prefix),
        "recovery_prefix_sha256": protocol.semantic_sha256(clean_prefix),
        "error_event_sha256": protocol.semantic_sha256(failed_event),
        "supervised_suffix_sha256": protocol.semantic_sha256(supervised),
        "clean_end_state_matches_reference": True,
        "recovery_end_state_matches_reference": (
            clean_agent == recovery_agent and clean_user == recovery_user
        ),
        "clean_agent_db_hash": clean_agent,
        "clean_user_db_hash": clean_user,
        "recovery_agent_db_hash": recovery_agent,
        "recovery_user_db_hash": recovery_user,
        "reference_actions_are_natural_dialogue": True,
        "independent_environment_replay_pass": False,
        "official_test_used": False,
    }


def build(
    *,
    tau2_root: Path,
    split_manifest: Path,
    inputs: list[tuple[str, Path]],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"output directory must be absent: {output_dir}")
    configure_tau2(tau2_root)
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    candidates, input_hashes, exclusions = load_candidates(inputs, split=split)
    selected = choose_one_per_task(candidates)
    tasks = task_map({row["domain"] for row in selected})
    pairs = []
    build_failures = []
    for source in selected:
        task_pairs = []
        try:
            for variant in range(PAIR_LIMIT_PER_TASK):
                task_pairs.append(
                    build_pair(
                        source,
                        task=tasks[source["task_identity"]],
                        mutation_variant=variant,
                    )
                )
        except Exception as error:
            build_failures.append(
                {
                    "task_identity": source["task_identity"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        pairs.extend(task_pairs)
    output_dir.mkdir(parents=True)
    pairs_path = output_dir / "pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    manifest = {
        "protocol": NATURAL_PROTOCOL,
        "design_protocol": full.PROTOCOL,
        "design_version": full.DESIGN_VERSION,
        "selection": (
            "successful inner-train natural conversations only; one "
            "hash-ranked conversation per task; structural site selection"
        ),
        "source_split_manifest_sha256": sha256_file(split_manifest),
        "input_file_sha256": input_hashes,
        "candidate_simulations": len(candidates),
        "selected_candidate_tasks": len(selected),
        # Preserve the independently selected clean source in the immutable
        # manifest.  The auditor uses this copy, rather than pair-owned fields,
        # to reconstruct and compare the complete supervised suffix.
        "selected_sources": [
            {
                "task_identity": source["task_identity"],
                "domain": source["domain"],
                "task_id": source["task_id"],
                "messages": source["messages"],
                "calls": source["calls"],
                "selected_message_index": source["selected_message_index"],
                "action_index": source["action_index"],
                "source_simulation_sha256": source["source_simulation_sha256"],
            }
            for source in selected
        ],
        "built_tasks": len({pair["task_identity"] for pair in pairs}),
        "built_pairs": len(pairs),
        "pairs_per_task": PAIR_LIMIT_PER_TASK,
        "task_ids": sorted({pair["task_identity"] for pair in pairs}),
        "pair_ids": [pair["pair_id"] for pair in pairs],
        "exclusions": dict(exclusions),
        "build_failures": build_failures,
        "pairs_jsonl_sha256": sha256_file(pairs_path),
        "future_leakage": False,
        "failed_action_positive_labels": 0,
        "official_test_used": False,
        "official_test_sealed": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def manifest_source_checks(
    pair: dict[str, Any], source: dict[str, Any]
) -> dict[str, bool]:
    message_index = source.get("selected_message_index")
    if not isinstance(message_index, int) or message_index < 0:
        return {"manifest_source_well_formed": False}
    expected_prefix = source.get("messages", [])[:message_index]
    expected_suffix = source.get("messages", [])[message_index:]
    return {
        "manifest_source_well_formed": True,
        "pair_identity_matches_manifest_source": (
            pair.get("task_identity") == source.get("task_identity")
            and pair.get("domain") == source.get("domain")
            and str(pair.get("task_id")) == str(source.get("task_id"))
            and pair.get("source_simulation_sha256")
            == source.get("source_simulation_sha256")
        ),
        "source_call_sequence_matches_manifest": (
            pair.get("source_call_sequence") == source.get("calls")
            and pair.get("selected_action_index") == source.get("action_index")
        ),
        "stored_clean_prefix_matches_manifest": (
            pair.get("clean_prefix") == expected_prefix
        ),
        "stored_supervised_suffix_matches_manifest": (
            pair.get("supervised_messages") == expected_suffix
        ),
        "stored_supervised_suffix_hash_matches_manifest": (
            pair.get("supervised_suffix_sha256")
            == protocol.semantic_sha256(expected_suffix)
        ),
    }


def audit_pair_replay(
    pair: dict[str, Any], task: Any, source: dict[str, Any]
) -> dict[str, Any]:
    calls = pair.get("source_call_sequence")
    index = pair.get("selected_action_index")
    if not isinstance(calls, list) or not isinstance(index, int):
        return {"well_formed_replay_contract": False}
    try:
        clean_results, clean_env = replay(
            domain=pair["domain"],
            task=task,
            calls=calls,
        )
        recovery_results, recovery_env = replay(
            domain=pair["domain"],
            task=task,
            calls=calls,
            injection_index=index,
            injected_call=pair["injected_call"],
        )
    except Exception:
        return {"environment_replay_completed": False}
    checks = {
        "environment_replay_completed": True,
        "clean_path_has_no_errors": not any(
            result.get("error") is True for result in clean_results
        ),
        "injection_actually_errors": recovery_results[index].get("error") is True,
        "correction_succeeds": recovery_results[index + 1].get("error") is False,
        "exactly_one_recovery_error": (
            sum(result.get("error") is True for result in recovery_results) == 1
        ),
        "stored_error_matches": (
            pair.get("injected_result", {}).get("content")
            == recovery_results[index].get("content")
            and pair.get("injected_result", {}).get("error")
            == recovery_results[index].get("error")
        ),
        "stored_correction_matches": (
            pair.get("correction_result", {}).get("content")
            == recovery_results[index + 1].get("content")
            and pair.get("correction_result", {}).get("error")
            == recovery_results[index + 1].get("error")
        ),
        "end_state_matches": (
            clean_env.get_db_hash() == recovery_env.get_db_hash()
            and clean_env.get_user_db_hash() == recovery_env.get_user_db_hash()
        ),
        "stored_end_state_hashes_match": (
            pair.get("clean_agent_db_hash") == clean_env.get_db_hash()
            and pair.get("clean_user_db_hash") == clean_env.get_user_db_hash()
            and pair.get("recovery_agent_db_hash") == recovery_env.get_db_hash()
            and pair.get("recovery_user_db_hash")
            == recovery_env.get_user_db_hash()
        ),
    }
    checks.update(manifest_source_checks(pair, source))
    static_view = deepcopy(pair)
    static_view["independent_environment_replay_pass"] = True
    static = protocol.audit_pair(static_view)
    checks.update({f"static_{key}": value for key, value in static.items()})
    return checks


def audit(
    *,
    tau2_root: Path,
    manifest_path: Path,
    pairs_path: Path,
    output: Path,
) -> dict[str, Any]:
    configure_tau2(tau2_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = read_jsonl(pairs_path)
    if (
        manifest.get("protocol") != NATURAL_PROTOCOL
        or manifest.get("pairs_jsonl_sha256") != sha256_file(pairs_path)
        or manifest.get("official_test_used") is not False
    ):
        raise RuntimeError("natural pair manifest/input binding drift")
    tasks = task_map({str(pair["domain"]) for pair in pairs})
    raw_sources = manifest.get("selected_sources")
    if not isinstance(raw_sources, list):
        raise RuntimeError("natural pair manifest lacks immutable selected sources")
    sources = {
        str(source.get("task_identity")): source
        for source in raw_sources
        if isinstance(source, dict)
    }
    failures = []
    accepted = []
    for pair in pairs:
        identity = str(pair.get("task_identity"))
        if identity not in tasks or identity not in sources:
            checks = {"registered_manifest_source": False}
        else:
            checks = audit_pair_replay(pair, tasks[identity], sources[identity])
        failed = sorted(key for key, value in checks.items() if not value)
        if failed:
            failures.append({"pair_id": pair.get("pair_id"), "failed_checks": failed})
        else:
            accepted.append(pair)
    counts = Counter(str(pair["task_identity"]) for pair in accepted)
    domains = {str(pair["domain"]) for pair in accepted}
    gate_checks = {
        "minimum_distinct_tasks": len(counts) >= full.MIN_NATURAL_TRAIN_TASKS,
        "minimum_pairs": len(accepted) >= full.MIN_NATURAL_PAIRS,
        "maximum_two_pairs_per_task": (
            bool(counts) and max(counts.values()) <= full.MAX_PAIRS_PER_TASK
        ),
        "exactly_two_pairs_per_accepted_task": (
            bool(counts) and set(counts.values()) == {PAIR_LIMIT_PER_TASK}
        ),
        "both_domains": domains == {"airline", "retail"},
        "zero_pair_failures": not failures,
        "unique_pair_ids": (
            len({pair["pair_id"] for pair in accepted}) == len(accepted)
        ),
        "official_test_sealed": all(
            pair.get("official_test_used") is False for pair in pairs
        ),
        "failed_action_positive_labels": all(
            pair.get("supervision_starts_after_error") is True
            and pair.get("clean_future_present_in_recovery_prompt") is False
            for pair in accepted
        ),
    }
    result = {
        "protocol": NATURAL_PROTOCOL,
        "status": (
            "PASS_TRAINING_AUTHORIZED"
            if all(gate_checks.values())
            else "FAIL_CLOSED"
        ),
        "checks": gate_checks,
        "observed": {
            "input_pairs": len(pairs),
            "audited_pairs": len(accepted),
            "distinct_tasks": len(counts),
            "domains": sorted(domains),
            "paper_target_tasks_reached": (
                len(counts) >= full.PAPER_TARGET_NATURAL_TASKS
            ),
            "paper_target_pairs_reached": (
                len(accepted) >= full.PAPER_TARGET_NATURAL_PAIRS
            ),
        },
        "pair_failures": failures,
        "manifest_sha256": sha256_file(manifest_path),
        "pairs_jsonl_sha256": sha256_file(pairs_path),
        "official_test_used": False,
        "official_test_sealed": True,
        "claim_boundary": (
            "A pass authorizes natural V5.5 training. It does not establish "
            "improved end-to-end task success."
        ),
    }
    write_json(output, result)
    return result


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be DOMAIN=PATH")
    domain, raw_path = value.split("=", 1)
    if domain not in {"retail", "airline"} or not raw_path:
        raise argparse.ArgumentTypeError("--input domain must be retail or airline")
    return domain, Path(raw_path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--tau2-root", type=Path, required=True)
    build_parser.add_argument("--split-manifest", type=Path, required=True)
    build_parser.add_argument(
        "--input",
        action="append",
        type=parse_input,
        required=True,
        help="Repeat DOMAIN=PATH for tau2 result files.",
    )
    build_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--tau2-root", type=Path, required=True)
    audit_parser.add_argument("--manifest", type=Path, required=True)
    audit_parser.add_argument("--pairs", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(
            tau2_root=args.tau2_root.resolve(),
            split_manifest=args.split_manifest.resolve(),
            inputs=args.input,
            output_dir=args.output_dir.resolve(),
        )
        print(
            json.dumps(
                {
                    "status": "BUILD_COMPLETE",
                    "tasks": result["built_tasks"],
                    "pairs": result["built_pairs"],
                    "output_dir": str(args.output_dir.resolve()),
                },
                indent=2,
            )
        )
        return
    result = audit(
        tau2_root=args.tau2_root.resolve(),
        manifest_path=args.manifest.resolve(),
        pairs_path=args.pairs.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS_TRAINING_AUTHORIZED" else 2)


if __name__ == "__main__":
    main()
