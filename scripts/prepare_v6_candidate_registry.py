#!/usr/bin/env python3
"""Build the outcome-independent V6 candidate-pair registry.

The registry is a *structural* contract, not generated recovery data.  It
reads only the frozen 70-task arm-train partition and pinned tau2 reference
actions.  A task is choice-eligible only when it contains at least two
distinct reference actions with strictly allowlisted identifier arguments.

Each ``candidate_pair`` contains exactly two sibling branch slots from one
shared prefix/snapshot specification.  The failed calls are deterministic,
type-preserving identifier mutations; runtime generation must still prove
that each call produced a real tool error and left both databases unchanged.
The matched first corrective action is the exact registered reference call.

No rollout, reward, validation, model, or official-test outcome is consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import v6_selection_protocol as protocol
except ModuleNotFoundError:
    from scripts import v6_selection_protocol as protocol


REGISTRY_PROTOCOL = "v6_candidate_pair_registry_v1"
V6_1_72B_TEACHER_PROTOCOL = "v6_1_causal_recovery_selection_72b_teacher_v1"
V6_2_REFERENCE_GUIDED_CLEAN_PROTOCOL = "v6_2_reference_guided_clean_source_v1"
ALLOWED_DESIGN_PROTOCOLS = frozenset(
    {
        protocol.PROTOCOL,
        V6_1_72B_TEACHER_PROTOCOL,
        V6_2_REFERENCE_GUIDED_CLEAN_PROTOCOL,
    }
)
REGISTRY_VERSION = "1.0"
PAIRS_PER_CHOICE_SET = 3
STRUCTURAL_PAYLOAD_SHA256 = (
    "002b5a2c5d83a4dab6dc8ce398d81bb8541bf7bfc6b25c04e62ce9ed179587f7"
)
LEGACY_STRUCTURAL_PAYLOAD_SHA256 = (
    "93cc882757b677c049e725a96540244d9988e79a326eb42a659a342729904dc3"
)
PILOT_RANK_SEED = 20260806
PILOT_DOMAIN_COUNTS = {"retail": 18, "airline": 6}
ACTION_IDENTIFIABLE_PROTOCOL = "v6_action_identifiable_candidate_pair_v1"
EXPECTED_ACTION_IDENTIFIABLE_COUNTS = {"retail": 39, "airline": 11}
EXPECTED_ACTION_IDENTIFIABLE_PILOT_TASK_IDS = (
    "airline:21",
    "airline:40",
    "airline:33",
    "airline:12",
    "airline:4",
    "airline:14",
    "retail:16",
    "retail:98",
    "retail:47",
    "retail:31",
    "retail:92",
    "retail:99",
    "retail:11",
    "retail:104",
    "retail:107",
    "retail:37",
    "retail:21",
    "retail:19",
    "retail:35",
    "retail:30",
    "retail:1",
    "retail:87",
    "retail:72",
    "retail:7",
)
EXPECTED_ACTION_IDENTIFIABLE_PILOT_SHA256 = (
    "5170b139c6307a3f46e29ab9db2573b864181c7b8ff85e27452e14f591bf27c4"
)
PHASE_SEED_SALTS = {
    "pilot": 20260816,
    "formal": 20260817,
}
EXPECTED_ARM_TRAIN_DOMAIN_COUNTS = {"retail": 52, "airline": 18}
EXPECTED_OFFICIAL_TEST_DOMAIN_COUNTS = {"retail": 40, "airline": 20}


class V6RegistryError(RuntimeError):
    """The candidate registry cannot be frozen without violating protocol."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(domain: str, task_id: Any) -> str:
    return f"{domain}:{task_id}"


def _numeric_task_key(identity: str) -> tuple[str, int, str]:
    domain, task_id = identity.split(":", 1)
    try:
        number = int(task_id)
    except ValueError as error:
        raise V6RegistryError(f"non-numeric task identity: {identity}") from error
    return domain, number, identity


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        "|".join(str(value) for value in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 1_000_001


def _task_rank(identity: str) -> tuple[str, str]:
    return (
        hashlib.sha256(
            (
                f"{protocol.PROTOCOL}|pilot|{PILOT_RANK_SEED}|"
                f"{identity}"
            ).encode("utf-8")
        ).hexdigest(),
        identity,
    )


def _config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only preregistered facts used by this registry."""
    if config.get("protocol") not in ALLOWED_DESIGN_PROTOCOLS:
        raise V6RegistryError("config protocol is not an explicitly frozen design")
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise V6RegistryError("config benchmark section is missing")
    if benchmark.get("commit") != protocol.TAU2_COMMIT:
        raise V6RegistryError("config tau2 commit drift")
    arm_train = benchmark.get("arm_train")
    if (
        not isinstance(arm_train, Mapping)
        or arm_train.get("tasks") != protocol.ARM_TRAIN_TASKS
    ):
        raise V6RegistryError("config does not freeze the 70-task arm-train set")
    official = benchmark.get("official_test")
    if not isinstance(official, Mapping):
        raise V6RegistryError("config official-test seal is missing")
    checks = {
        "used_false": (
            official.get(
                "used_during_generation_scoring_selection_or_model_selection"
            )
            is False
        ),
        "sealed_true": official.get("sealed") is True,
        "content_not_exported": (
            official.get("ids_or_content_exported_to_generation_workers")
            is False
        ),
    }
    if not all(checks.values()):
        raise V6RegistryError(
            f"official test is not fail-closed in config: {checks}"
        )
    return {
        "protocol": str(config["protocol"]),
        "tau2_commit": str(benchmark["commit"]),
        "arm_train_tasks": int(arm_train["tasks"]),
        "official_test_checks": checks,
    }


def _minimal_yaml_contract(text: str) -> dict[str, Any]:
    """Load the small frozen subset when PyYAML is unavailable.

    The normal path uses ``yaml.safe_load``.  This fallback intentionally
    recognizes only the exact V6 contract and refuses ambiguous YAML.
    """
    protocol_match = re.search(r"(?m)^protocol:\s*([^\s#]+)\s*$", text)
    commit_match = re.search(
        r"(?m)^\s{2}commit:\s*([0-9a-f]{40})\s*$", text
    )
    arm_match = re.search(
        r"(?ms)^\s{2}arm_train:\s*\n\s{4}tasks:\s*(\d+)\s*$", text
    )
    used_match = re.search(
        r"(?m)^\s{4}used_during_generation_scoring_selection_or_model_selection:"
        r"\s*(true|false)\s*$",
        text,
    )
    sealed_match = re.search(
        r"(?m)^\s{4}sealed:\s*(true|false)\s*$", text
    )
    exported_match = re.search(
        r"(?m)^\s{4}ids_or_content_exported_to_generation_workers:"
        r"\s*(true|false)\s*$",
        text,
    )
    if not all(
        (
            protocol_match,
            commit_match,
            arm_match,
            used_match,
            sealed_match,
            exported_match,
        )
    ):
        raise V6RegistryError(
            "PyYAML is unavailable and the config is not the exact frozen "
            "V6 YAML shape"
        )
    to_bool = lambda match: match.group(1) == "true"
    return {
        "protocol": protocol_match.group(1),
        "benchmark": {
            "commit": commit_match.group(1),
            "arm_train": {"tasks": int(arm_match.group(1))},
            "official_test": {
                "used_during_generation_scoring_selection_or_model_selection": (
                    to_bool(used_match)
                ),
                "sealed": to_bool(sealed_match),
                "ids_or_content_exported_to_generation_workers": to_bool(
                    exported_match
                ),
            },
        },
    }


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            value = _minimal_yaml_contract(text)
        else:
            value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise V6RegistryError(f"{path}: config root must be an object")
    return value


def frozen_arm_train(
    split: Mapping[str, Any],
    *,
    strict: bool = True,
) -> tuple[list[str], dict[str, list[str]], dict[str, set[str]]]:
    """Recompute the V5.3 70-task arm partition from identity metadata."""
    domains = split.get("domains")
    if not isinstance(domains, Mapping):
        raise V6RegistryError("split manifest domains are missing")
    excluded = set(protocol.EXCLUDED_TASK_IDS)
    loss_validation = set(protocol.LOSS_VALIDATION_IDS)
    arm_by_domain: dict[str, list[str]] = {}
    sealed_sets: dict[str, set[str]] = {}
    validation_sets: dict[str, set[str]] = {}
    all_arm: list[str] = []
    for domain in ("retail", "airline"):
        row = domains.get(domain)
        if not isinstance(row, Mapping):
            raise V6RegistryError(f"split manifest lacks {domain}")
        required = ("inner_train_ids", "validation_ids", "sealed_test_ids")
        if any(not isinstance(row.get(field), list) for field in required):
            raise V6RegistryError(f"{domain}: incomplete identity-only split")
        inner = {_identity(domain, value) for value in row["inner_train_ids"]}
        validation = {
            _identity(domain, value) for value in row["validation_ids"]
        }
        sealed = {_identity(domain, value) for value in row["sealed_test_ids"]}
        if (
            len(inner) != len(row["inner_train_ids"])
            or len(validation) != len(row["validation_ids"])
            or len(sealed) != len(row["sealed_test_ids"])
            or inner & validation
            or inner & sealed
            or validation & sealed
        ):
            raise V6RegistryError(f"{domain}: split is not a disjoint identity set")
        arm = sorted(
            inner - excluded - loss_validation,
            key=_numeric_task_key,
        )
        arm_by_domain[domain] = arm
        sealed_sets[domain] = sealed
        validation_sets[domain] = validation
        all_arm.extend(arm)
    if set(all_arm) & (
        set().union(*sealed_sets.values())
        | set().union(*validation_sets.values())
        | excluded
        | loss_validation
    ):
        raise V6RegistryError("arm-train overlaps a forbidden identity set")
    if strict:
        observed = {
            domain: len(arm_by_domain[domain])
            for domain in ("retail", "airline")
        }
        if observed != EXPECTED_ARM_TRAIN_DOMAIN_COUNTS:
            raise V6RegistryError(f"arm-train domain count drift: {observed}")
        sealed_counts = {
            domain: len(sealed_sets[domain])
            for domain in ("retail", "airline")
        }
        if sealed_counts != EXPECTED_OFFICIAL_TEST_DOMAIN_COUNTS:
            raise V6RegistryError(
                f"official-test identity count drift: {sealed_counts}"
            )
        if len(all_arm) != protocol.ARM_TRAIN_TASKS:
            raise V6RegistryError("arm-train task count is not 70")
    return (
        sorted(all_arm, key=_numeric_task_key),
        arm_by_domain,
        {
            "sealed": set().union(*sealed_sets.values()),
            "validation": set().union(*validation_sets.values()),
        },
    )


def load_task_catalog(
    tau2_root: Path,
    split: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    catalog: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for domain in ("retail", "airline"):
        path = tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json"
        if not path.is_file():
            raise V6RegistryError(f"missing pinned tau2 task file: {path}")
        observed_hash = sha256_file(path)
        expected_hash = (
            split["domains"][domain]
            .get("source_files", {})
            .get("tasks_json_sha256")
        )
        if expected_hash and observed_hash != expected_hash:
            raise V6RegistryError(
                f"{domain}: tasks.json hash drift "
                f"observed={observed_hash} expected={expected_hash}"
            )
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise V6RegistryError(f"{path}: task catalog must be a list")
        for task in rows:
            if not isinstance(task, dict) or "id" not in task:
                raise V6RegistryError(f"{path}: malformed task")
            identity = _identity(domain, task["id"])
            if identity in catalog:
                raise V6RegistryError(f"duplicate task identity {identity}")
            catalog[identity] = task
        hashes[domain] = observed_hash
    return catalog, hashes


def _reference_actions(
    *,
    domain: str,
    task: Mapping[str, Any],
) -> list[dict[str, Any]]:
    criteria = task.get("evaluation_criteria")
    actions = criteria.get("actions") if isinstance(criteria, Mapping) else None
    if not isinstance(actions, list):
        return []
    allowed_keys = set(protocol.SAFE_ON_ERROR_IDENTIFIER_KEYS.get(domain, ()))
    result: list[dict[str, Any]] = []
    for action_index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            continue
        tool_name = action.get("name")
        arguments = action.get("arguments")
        if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
            continue
        for argument_key in sorted(allowed_keys & set(arguments)):
            value = arguments.get(argument_key)
            if not isinstance(value, str) or len(value) < 2:
                continue
            family = f"{tool_name}::{argument_key}"
            result.append(
                {
                    "family": family,
                    "tool_name": tool_name,
                    "argument_key": argument_key,
                    "original_identifier": value,
                    "reference_action_index": action_index,
                    "reference_action_kind": (
                        "READ"
                        if tool_name.lower().startswith(
                            ("get_", "find_", "list_", "search_", "lookup_", "check_")
                        )
                        else "WRITE"
                    ),
                    "reference_call": {
                        "id": str(
                            action.get(
                                "action_id",
                                f"v6-reference-{action_index:03d}",
                            )
                        ),
                        "name": tool_name,
                        "arguments": deepcopy(dict(arguments)),
                        "requestor": str(action.get("requestor", "assistant")),
                    },
                }
            )
    # Retain distinct reference-call semantics within one family.  Repeated
    # identical product lookups do not create new actions, while two calls in
    # the same family with different arguments remain valid alternatives to a
    # *different* family.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in result:
        family = str(row["family"])
        call = row["reference_call"]
        semantics = sha256(
            {"name": call["name"], "arguments": call["arguments"]}
        )
        key = (family, semantics)
        incumbent = unique.get(key)
        if incumbent is None or (
            int(row["reference_action_index"]),
            sha256(row["reference_call"]),
        ) < (
            int(incumbent["reference_action_index"]),
            sha256(incumbent["reference_call"]),
        ):
            unique[key] = row
    return [
        unique[key]
        for key in sorted(
            unique,
            key=lambda value: (value[0], value[1]),
        )
    ]


def structural_payload(
    *,
    arm_train_task_ids: Iterable[str],
    task_catalog: Mapping[str, Mapping[str, Any]],
    strict: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    actions_by_task: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    domains: dict[str, Any] = {}
    arm_set = set(arm_train_task_ids)
    for domain in ("retail", "airline"):
        families_by_task: dict[str, list[str]] = {}
        eligible_ids: list[int] = []
        for identity in sorted(
            (value for value in arm_set if value.startswith(f"{domain}:")),
            key=_numeric_task_key,
        ):
            task = task_catalog.get(identity)
            if not isinstance(task, Mapping):
                raise V6RegistryError(f"task catalog lacks {identity}")
            actions = _reference_actions(domain=domain, task=task)
            families = sorted({str(row["family"]) for row in actions})
            if len(families) < 2:
                exclusions.append(
                    {
                        "task_identity": identity,
                        "domain": domain,
                        "reason_code": (
                            "FEWER_THAN_TWO_SAFE_ON_ERROR_ACTION_FAMILIES"
                        ),
                        "observed_corrective_families": families,
                    }
                )
                continue
            actions_by_task[identity] = actions
            families_by_task[identity] = families
            eligible_ids.append(int(identity.split(":", 1)[1]))
        domains[domain] = {
            "identifier_argument_keys": sorted(
                protocol.SAFE_ON_ERROR_IDENTIFIER_KEYS[domain]
            ),
            "eligible_task_ids": sorted(eligible_ids),
            "families_by_task": dict(sorted(families_by_task.items())),
        }
    payload = {
        "protocol": protocol.STRUCTURAL_PROTOCOL,
        "tau2_commit": protocol.TAU2_COMMIT,
        "split_partition_sha256": protocol.PARTITION_SHA256,
        "canonical_error_family": "tool_name::argument_key",
        "minimum_distinct_families_per_task": 2,
        "failed_injection_requires_database_unchanged": True,
        "domains": domains,
    }
    observed_hash = sha256(payload)
    counts = {
        domain: len(domains[domain]["eligible_task_ids"])
        for domain in ("retail", "airline")
    }
    if strict:
        if counts != protocol.STRUCTURAL_ELIGIBLE_COUNTS:
            raise V6RegistryError(
                f"structural eligible count drift: {counts}"
            )
        if observed_hash != STRUCTURAL_PAYLOAD_SHA256:
            raise V6RegistryError(
                "structural payload hash drift: "
                f"observed={observed_hash} expected={STRUCTURAL_PAYLOAD_SHA256}"
            )
    return payload, actions_by_task, exclusions


def mutate_identifier(value: str, *, salt: str) -> str:
    """Deterministically mutate one alphanumeric character, preserving type."""
    if not isinstance(value, str) or len(value) < 2:
        raise V6RegistryError("identifier mutation requires a string")
    chars = list(value)
    positions = [
        index
        for index, char in enumerate(chars)
        if char.isalnum()
        and not (
            "@" in value
            and index > value.index("@")
        )
    ]
    if not positions:
        raise V6RegistryError(f"identifier has no safe mutable character: {value!r}")
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    position = positions[int.from_bytes(digest[:4], "big") % len(positions)]
    original = chars[position]
    if original.isdigit():
        offset = digest[4] % 9 + 1
        chars[position] = str((int(original) + offset) % 10)
    elif original.isupper():
        offset = digest[4] % 25 + 1
        chars[position] = chr((ord(original) - 65 + offset) % 26 + 65)
    else:
        offset = digest[4] % 25 + 1
        chars[position] = chr(
            (ord(original.lower()) - 97 + offset) % 26 + 97
        )
    mutated = "".join(chars)
    if mutated == value or type(mutated) is not type(value):
        raise V6RegistryError("identifier mutation was not type-preserving")
    return mutated


def _action_pairs(
    identity: str,
    actions: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    def action_semantics(row: Mapping[str, Any]) -> str:
        call = row["reference_call"]
        return sha256(
            {"name": call["name"], "arguments": call["arguments"]}
        )

    pairs = list(combinations(actions, 2))
    pairs = [
        pair
        for pair in pairs
        if pair[0]["family"] != pair[1]["family"]
        and action_semantics(pair[0]) != action_semantics(pair[1])
    ]
    if not pairs:
        raise V6RegistryError(f"{identity}: no distinct action-family pair")
    return sorted(
        pairs,
        key=lambda pair: (
            hashlib.sha256(
                (
                    f"{REGISTRY_PROTOCOL}|family-pair|{identity}|"
                    f"{pair[0]['family']}|{action_semantics(pair[0])}|"
                    f"{pair[1]['family']}|{action_semantics(pair[1])}"
                ).encode("utf-8")
            ).hexdigest(),
            str(pair[0]["family"]),
            action_semantics(pair[0]),
            str(pair[1]["family"]),
            action_semantics(pair[1]),
        ),
    )


def _branch_slot(
    *,
    phase: str,
    choice_set_id: str,
    candidate_pair_id: str,
    pair_index: int,
    branch_index: int,
    task_identity: str,
    action: Mapping[str, Any],
    prefix_spec_sha256: str,
    snapshot_spec_sha256: str,
) -> dict[str, Any]:
    mutation_salt = (
        f"{REGISTRY_PROTOCOL}|{phase}|{task_identity}|"
        f"{pair_index}|{branch_index}|{action['family']}"
    )
    mutated = mutate_identifier(
        str(action["original_identifier"]),
        salt=mutation_salt,
    )
    branch_id = f"{candidate_pair_id}:branch:{branch_index + 1}"
    reference_call = deepcopy(action["reference_call"])
    error_call = deepcopy(reference_call)
    error_call["id"] = (
        f"v6-error-{sha256([branch_id, mutation_salt])[:16]}"
    )
    error_call["arguments"][str(action["argument_key"])] = mutated
    canonical_family = str(action["family"])
    injection_spec = {
        "status": "FROZEN_RUNTIME_VERIFICATION_REQUIRED",
        "site_tool_allowlist": [str(action["tool_name"])],
        "selected_site_tool": str(action["tool_name"]),
        "operator": "replace_one_allowlisted_identifier_argument",
        "argument_key": str(action["argument_key"]),
        "original_identifier": str(action["original_identifier"]),
        "replacement": {
            "source": "deterministic_type_preserving_mutation",
            "value": mutated,
            "mutation_salt_sha256": hashlib.sha256(
                mutation_salt.encode("utf-8")
            ).hexdigest(),
        },
        "error_call": error_call,
        "expected_error_predicate": {
            "json_path": "$.error",
            "operator": "is",
            "value": True,
        },
        "expected_state_mutating": False,
        "required_runtime_evidence": [
            "actual_tool_error",
            "agent_db_hash_before_equals_after_error",
            "user_db_hash_before_equals_after_error",
        ],
    }
    corrective_action_spec = {
        "canonical_family": canonical_family,
        "forced_first_action_constructor": {
            "kind": "exact_registered_reference_tool_call",
            "tool_call": reference_call,
        },
        "reference_action_index": int(action["reference_action_index"]),
        "reference_action_kind": str(action["reference_action_kind"]),
        "gold_future_visible": False,
    }
    row = {
        "branch_id": branch_id,
        "candidate_pair_id": candidate_pair_id,
        "choice_set_id": choice_set_id,
        "phase": phase,
        "task_identity": task_identity,
        "branch_index": branch_index,
        "prefix_spec_sha256": prefix_spec_sha256,
        "snapshot_spec_sha256": snapshot_spec_sha256,
        "tool_name": str(action["tool_name"]),
        "identifier_key": str(action["argument_key"]),
        "original_identifier": str(action["original_identifier"]),
        "mutated_identifier": mutated,
        "corrective_family": canonical_family,
        "canonical_corrective_family": canonical_family,
        "reference_action_kind": str(action["reference_action_kind"]),
        "reference_arguments": deepcopy(reference_call["arguments"]),
        "reference_action_key": protocol.canonical_reference_action_key(
            str(action["tool_name"]), reference_call["arguments"]
        ),
        "identifier_field_allowlisted": True,
        "identifier_mutation_type_preserving": True,
        "recovery_seed": _seed(
            REGISTRY_PROTOCOL,
            PHASE_SEED_SALTS[phase],
            branch_id,
        ),
        "injection_spec": injection_spec,
        "corrective_action_spec": corrective_action_spec,
        "status": "REGISTERED_RUNTIME_PREFLIGHT_REQUIRED",
        "official_test_used": False,
    }
    row["branch_slot_sha256"] = sha256(row)
    return row


def build_registry(
    *,
    split: Mapping[str, Any],
    config: Mapping[str, Any],
    task_catalog: Mapping[str, Mapping[str, Any]],
    split_file_sha256: str | None = None,
    config_file_sha256: str | None = None,
    task_file_sha256: Mapping[str, str] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    protocol.validate_constants()
    config_contract = _config_contract(config)
    arm_train, arm_by_domain, forbidden = frozen_arm_train(
        split, strict=strict
    )
    structural, actions_by_task, exclusions = structural_payload(
        arm_train_task_ids=arm_train,
        task_catalog=task_catalog,
        strict=strict,
    )
    structural_hash = sha256(structural)
    action_pair_options: dict[
        str, list[tuple[dict[str, Any], dict[str, Any]]]
    ] = {}
    action_exclusions: list[dict[str, Any]] = []
    for identity, actions in sorted(
        actions_by_task.items(), key=lambda item: _numeric_task_key(item[0])
    ):
        try:
            action_pair_options[identity] = _action_pairs(identity, actions)
        except V6RegistryError:
            action_exclusions.append(
                {
                    "task_identity": identity,
                    "domain": identity.split(":", 1)[0],
                    "reason_code": "FEWER_THAN_TWO_DISTINCT_REFERENCE_ACTIONS",
                    "observed_corrective_families": sorted(
                        {str(row["family"]) for row in actions}
                    ),
                    "distinct_reference_action_semantics": len(
                        {
                            sha256(
                                {
                                    "name": row["reference_call"]["name"],
                                    "arguments": row["reference_call"][
                                        "arguments"
                                    ],
                                }
                            )
                            for row in actions
                        }
                    ),
                }
            )
    eligible_by_domain = {
        domain: sorted(
            (
                identity
                for identity in action_pair_options
                if identity.startswith(f"{domain}:")
            ),
            key=_numeric_task_key,
        )
        for domain in ("retail", "airline")
    }
    action_payload = {
        "protocol": ACTION_IDENTIFIABLE_PROTOCOL,
        "source_structural_eligibility_sha256": structural_hash,
        "reference_action_semantics": "tool_name+all_arguments",
        "candidate_pair_requires_distinct_canonical_families": True,
        "candidate_pair_requires_distinct_reference_action_semantics": True,
        "domains": {
            domain: {
                "eligible_task_ids": [
                    int(identity.split(":", 1)[1])
                    for identity in eligible_by_domain[domain]
                ],
                "candidate_pair_semantics_by_task": {
                    identity: [
                        {
                            "left_family": left["family"],
                            "left_reference_action_sha256": sha256(
                                {
                                    "name": left["reference_call"]["name"],
                                    "arguments": left["reference_call"][
                                        "arguments"
                                    ],
                                }
                            ),
                            "right_family": right["family"],
                            "right_reference_action_sha256": sha256(
                                {
                                    "name": right["reference_call"]["name"],
                                    "arguments": right["reference_call"][
                                        "arguments"
                                    ],
                                }
                            ),
                        }
                        for left, right in action_pair_options[identity]
                    ]
                    for identity in eligible_by_domain[domain]
                },
            }
            for domain in ("retail", "airline")
        },
    }
    action_counts = {
        domain: len(eligible_by_domain[domain])
        for domain in ("retail", "airline")
    }
    if strict and action_counts != EXPECTED_ACTION_IDENTIFIABLE_COUNTS:
        raise V6RegistryError(
            f"action-identifiable task count drift: {action_counts}"
        )
    pilot_tasks: list[str] = []
    for domain in ("airline", "retail"):
        ranked = sorted(eligible_by_domain[domain], key=_task_rank)
        count = (
            PILOT_DOMAIN_COUNTS[domain]
            if strict
            else min(PILOT_DOMAIN_COUNTS[domain], len(ranked))
        )
        if len(ranked) < count:
            raise V6RegistryError(
                f"{domain}: only {len(ranked)} eligible tasks; need {count}"
            )
        pilot_tasks.extend(ranked[:count])
    if strict:
        if tuple(pilot_tasks) != EXPECTED_ACTION_IDENTIFIABLE_PILOT_TASK_IDS:
            raise V6RegistryError(
                "action-identifiable pilot registry drift: "
                f"observed={pilot_tasks} "
                f"expected={list(EXPECTED_ACTION_IDENTIFIABLE_PILOT_TASK_IDS)}"
            )
        if sha256(pilot_tasks) != EXPECTED_ACTION_IDENTIFIABLE_PILOT_SHA256:
            raise V6RegistryError("action-identifiable pilot digest drift")
    formal_tasks = sorted(
        [*eligible_by_domain["retail"], *eligible_by_domain["airline"]],
        key=_numeric_task_key,
    )

    choice_sets: list[dict[str, Any]] = []
    candidate_pairs: list[dict[str, Any]] = []
    for phase, task_ids in (("pilot", pilot_tasks), ("formal", formal_tasks)):
        for task_identity in task_ids:
            domain, task_id = task_identity.split(":", 1)
            choice_set_id = f"v6:{phase}:{task_identity}:prefix:initial"
            prefix_spec = {
                "prefix_slot_id": f"{choice_set_id}:prefix-slot",
                "source": "task_initial_state_and_user_request",
                "reference_action_prefix_count": 0,
                "clean_future_available_to_generator": False,
                "shared_across_candidate_pairs": True,
                "materialization_seed": _seed(
                    REGISTRY_PROTOCOL, phase, task_identity, "prefix"
                ),
            }
            snapshot_spec = {
                "snapshot_slot_id": f"{choice_set_id}:snapshot-slot",
                "source": "fresh_pinned_tau2_task_initial_state",
                "agent_and_user_database_hashes_required": True,
                "shared_across_candidate_pairs": True,
            }
            prefix_hash = sha256(prefix_spec)
            snapshot_hash = sha256(snapshot_spec)
            pair_options = action_pair_options[task_identity]
            pair_ids: list[str] = []
            for pair_index in range(PAIRS_PER_CHOICE_SET):
                selected = pair_options[pair_index % len(pair_options)]
                candidate_pair_id = (
                    f"{choice_set_id}:candidate-pair:{pair_index + 1:02d}"
                )
                branches = [
                    _branch_slot(
                        phase=phase,
                        choice_set_id=choice_set_id,
                        candidate_pair_id=candidate_pair_id,
                        pair_index=pair_index,
                        branch_index=branch_index,
                        task_identity=task_identity,
                        action=action,
                        prefix_spec_sha256=prefix_hash,
                        snapshot_spec_sha256=snapshot_hash,
                    )
                    for branch_index, action in enumerate(selected)
                ]
                families = [
                    str(branch["canonical_corrective_family"])
                    for branch in branches
                ]
                if len(set(families)) != 2:
                    raise V6RegistryError(
                        f"{candidate_pair_id}: corrective families are not distinct"
                    )
                row = {
                    "candidate_pair_id": candidate_pair_id,
                    "choice_set_id": choice_set_id,
                    "phase": phase,
                    "partition": "arm_train",
                    "task_identity": task_identity,
                    "domain": domain,
                    "task_id": task_id,
                    "pair_index": pair_index,
                    "selection_unit": True,
                    "prefix_slot": prefix_spec,
                    "snapshot_slot": snapshot_spec,
                    "prefix_sha256": prefix_hash,
                    "environment_snapshot_sha256": snapshot_hash,
                    "choice_seed": _seed(
                        REGISTRY_PROTOCOL,
                        PHASE_SEED_SALTS[phase],
                        candidate_pair_id,
                    ),
                    "branches": branches,
                    "expected_canonical_corrective_families": families,
                    "status": "REGISTERED_RUNTIME_PREFLIGHT_REQUIRED",
                    "official_test_used": False,
                }
                structural_check = protocol.structural_eligibility(row)
                if structural_check.get("eligible") is not True:
                    raise V6RegistryError(
                        f"{candidate_pair_id}: protocol structural audit failed: "
                        f"{structural_check}"
                    )
                row["structural_audit"] = structural_check
                row["candidate_pair_sha256"] = sha256(row)
                candidate_pairs.append(row)
                pair_ids.append(candidate_pair_id)
            choice = {
                "choice_set_id": choice_set_id,
                "phase": phase,
                "task_identity": task_identity,
                "domain": domain,
                "task_id": task_id,
                "prefix_slot_id": prefix_spec["prefix_slot_id"],
                "prefix_spec_sha256": prefix_hash,
                "snapshot_slot_id": snapshot_spec["snapshot_slot_id"],
                "snapshot_spec_sha256": snapshot_hash,
                "candidate_pair_ids": pair_ids,
                "minimum_complete_pairs": PAIRS_PER_CHOICE_SET,
                "status": "REGISTERED",
                "official_test_used": False,
            }
            choice["choice_set_sha256"] = sha256(choice)
            choice_sets.append(choice)

    forbidden_hash = sha256(
        {
            "validation": sorted(forbidden["validation"]),
            "sealed": sorted(forbidden["sealed"]),
        }
    )
    registry: dict[str, Any] = {
        "protocol": REGISTRY_PROTOCOL,
        "design_protocol": config_contract["protocol"],
        "design_version": protocol.DESIGN_VERSION,
        "registry_version": REGISTRY_VERSION,
        "registry_status": "STRUCTURALLY_FROZEN_RUNTIME_PREFLIGHT_REQUIRED",
        "generation_authorized_before_runtime_preflight": False,
        "selection_unit": "candidate_pair",
        "grouping_unit": "choice_set",
        "accounting_unit": "sibling_branch",
        "selection_uses_outcomes": False,
        "selection_inputs": [
            "frozen_arm_train_identity",
            "pinned_reference_action_structure",
            "strict_identifier_allowlist",
            "protocol_seed",
        ],
        "source": {
            "tau2_commit": protocol.TAU2_COMMIT,
            "partition_sha256": protocol.PARTITION_SHA256,
            "split_file_sha256": split_file_sha256 or sha256(split),
            "config_file_sha256": config_file_sha256 or sha256(config),
            "task_file_sha256": dict(sorted((task_file_sha256 or {}).items())),
            "config_contract": config_contract,
            "arm_train_task_ids_sha256": sha256(arm_train),
            "forbidden_identity_sets_sha256": forbidden_hash,
        },
        "official_test_used": False,
        "official_test_sealed": True,
        "official_test_task_content_exported": False,
        "official_test_identity_overlap_count": 0,
        "arm_train_task_ids": arm_train,
        "arm_train_domain_counts": {
            domain: len(arm_by_domain[domain])
            for domain in ("retail", "airline")
        },
        "structural_eligibility_payload": structural,
        "structural_eligibility_sha256": structural_hash,
        "structural_eligibility_expected_sha256": (
            STRUCTURAL_PAYLOAD_SHA256
        ),
        "legacy_expected_hash": LEGACY_STRUCTURAL_PAYLOAD_SHA256,
        "legacy_hash_match": (
            structural_hash == LEGACY_STRUCTURAL_PAYLOAD_SHA256
        ),
        "structural_exclusions": sorted(
            exclusions, key=lambda row: _numeric_task_key(row["task_identity"])
        ),
        "action_identifiability_payload": action_payload,
        "action_identifiability_sha256": sha256(action_payload),
        "action_identifiable_domain_counts": action_counts,
        "action_identifiability_exclusions": sorted(
            action_exclusions,
            key=lambda row: _numeric_task_key(row["task_identity"]),
        ),
        "phase_registry": {
            "pilot": {
                "task_ids": pilot_tasks,
                "task_ids_sha256": sha256(pilot_tasks),
                "task_count": len(pilot_tasks),
                "candidate_pair_count": (
                    len(pilot_tasks) * PAIRS_PER_CHOICE_SET
                ),
                "seed_salt": PHASE_SEED_SALTS["pilot"],
                "formal_seed_overlap": False,
            },
            "formal": {
                "task_ids": formal_tasks,
                "task_ids_sha256": sha256(formal_tasks),
                "task_count": len(formal_tasks),
                "candidate_pair_count": (
                    len(formal_tasks) * PAIRS_PER_CHOICE_SET
                ),
                "seed_salt": PHASE_SEED_SALTS["formal"],
                "pilot_seed_overlap": False,
            },
        },
        "gates": {
            "pilot": {
                "minimum_tasks_with_three_accepted_pairs": (
                    protocol.PILOT_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_accepted_pairs": protocol.PILOT_MIN_ACCEPTED_PAIRS,
                "minimum_domains": protocol.MIN_DOMAINS,
                "minimum_error_families": protocol.MIN_ERROR_FAMILIES,
            },
            "formal": {
                "minimum_tasks_with_three_accepted_pairs": (
                    protocol.FORMAL_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_accepted_pairs": protocol.FORMAL_MIN_ACCEPTED_PAIRS,
                "screen_minimum_tasks": (
                    protocol.SCREEN_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_domains": protocol.MIN_DOMAINS,
                "minimum_error_families": protocol.MIN_ERROR_FAMILIES,
            },
        },
        "choice_sets": choice_sets,
        "candidate_pairs": candidate_pairs,
        "hashes": {
            "structural_payload_sha256": structural_hash,
            "action_identifiability_sha256": sha256(action_payload),
            "choice_sets_sha256": sha256(
                [row["choice_set_sha256"] for row in choice_sets]
            ),
            "candidate_pairs_sha256": sha256(
                [row["candidate_pair_sha256"] for row in candidate_pairs]
            ),
        },
    }
    if strict and structural_hash != STRUCTURAL_PAYLOAD_SHA256:
        raise V6RegistryError("structural eligibility did not reproduce")
    if set(arm_train) & forbidden["sealed"]:
        raise V6RegistryError("official-test identity entered arm-train")
    registry["registry_sha256"] = sha256(registry)
    return registry


def _verify_tau2_commit(tau2_root: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(tau2_root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise V6RegistryError("cannot verify pinned tau2 git commit") from error
    observed = result.stdout.strip()
    if observed != protocol.TAU2_COMMIT:
        raise V6RegistryError(
            f"tau2 commit drift: observed={observed} expected={protocol.TAU2_COMMIT}"
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare(
    *,
    tau2_root: Path,
    split_manifest: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise V6RegistryError(f"refusing to overwrite registry: {output}")
    _verify_tau2_commit(tau2_root)
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    if not isinstance(split, dict):
        raise V6RegistryError("split manifest root must be an object")
    config = load_config(config_path)
    catalog, task_hashes = load_task_catalog(tau2_root, split)
    registry = build_registry(
        split=split,
        config=config,
        task_catalog=catalog,
        split_file_sha256=sha256_file(split_manifest),
        config_file_sha256=sha256_file(config_path),
        task_file_sha256=task_hashes,
        strict=True,
    )
    write_json(output, registry)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = prepare(
        tau2_root=args.tau2_root.resolve(),
        split_manifest=args.split_manifest.resolve(),
        config_path=args.config.resolve(),
        output=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": registry["registry_status"],
                "registry_sha256": registry["registry_sha256"],
                "structural_eligibility_sha256": (
                    registry["structural_eligibility_sha256"]
                ),
                "pilot_tasks": registry["phase_registry"]["pilot"]["task_count"],
                "formal_tasks": registry["phase_registry"]["formal"]["task_count"],
                "candidate_pairs": len(registry["candidate_pairs"]),
                "official_test_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
