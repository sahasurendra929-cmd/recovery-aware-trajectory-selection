#!/usr/bin/env python3
"""Measure V6 candidate cost and frozen-student first-action hardness evidence.

The input is the raw ``candidate_pairs.unscored.jsonl`` written by
``run_v6_candidate_generation.py``.  This stage does not use validation or
official-test outcomes.  It performs two train-pool-only measurements with one
pinned frozen base checkpoint:

* exact non-padding and supervised-target token counts under the student's
  chat template; and
* the four matched/crossed first-corrective-action log probabilities used by
  the downstream length-normalized hardness score.

The injected failed call and its error result are prompt context and receive
zero label tokens.  Every non-failed assistant output in the *fresh* recovery
suffix is labeled, including ordinary assistant text (not only tool calls).
The clean control likewise labels all of its non-failed assistant outputs.

PyTorch, Transformers, bitsandbytes, and tau2 are imported only by runtime
loaders.  Contract helpers therefore remain importable and testable on a
CPU-only coordinator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import train_v5_sft_causal as v5_train
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import train_v5_sft_causal as v5_train


PROTOCOL = "v6_frozen_student_candidate_measurement_v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PINNED_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
Q_KEYS = ("q_e1_a1", "q_e1_a2", "q_e2_a1", "q_e2_a2")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class V6CandidateMeasurementError(RuntimeError):
    """The raw pool or frozen-student measurement contract is invalid."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_revision(value: str | None, label: str) -> str:
    """Return a non-empty pinned revision or fail before loading a model."""
    if not isinstance(value, str) or not value.strip():
        raise V6CandidateMeasurementError(f"{label} is required and must be non-empty")
    return value.strip()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V6CandidateMeasurementError(f"{label} must be a non-empty string")
    return value.strip()


def _messages(value: Any, label: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(row, dict) for row in value)
    ):
        raise V6CandidateMeasurementError(
            f"{label} must be a non-empty message-object list"
        )
    return deepcopy(value)


def normalize_messages(
    messages: Sequence[Mapping[str, Any]], *, label: str
) -> list[dict[str, Any]]:
    """Canonicalize tau2/compact/OpenAI messages exactly like the V5 trainer."""
    canonical_messages = json.loads(canonical(list(messages)))
    try:
        return [
            v5_train.normalize_message(
                message, where=f"{label}[{index}]"
            )
            for index, message in enumerate(canonical_messages)
        ]
    except RuntimeError as error:
        raise V6CandidateMeasurementError(
            f"{label}: message normalization failed: {error}"
        ) from error


def normalize_tool_schema(schema: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        raise V6CandidateMeasurementError(f"{label} must be an object")
    if schema.get("type") == "function" and isinstance(
        schema.get("function"), Mapping
    ):
        normalized = deepcopy(dict(schema))
    elif isinstance(schema.get("name"), str):
        normalized = {
            "type": "function",
            "function": deepcopy(dict(schema)),
        }
    else:
        raise V6CandidateMeasurementError(
            f"{label} is not an OpenAI function schema"
        )
    function = normalized["function"]
    if not isinstance(function.get("name"), str) or not function["name"]:
        raise V6CandidateMeasurementError(f"{label} has no function name")
    return json.loads(canonical(normalized))


def normalize_tool_schemas(
    schemas: Any, *, label: str
) -> list[dict[str, Any]]:
    if (
        not isinstance(schemas, list)
        or not schemas
        or any(not isinstance(row, Mapping) for row in schemas)
    ):
        raise V6CandidateMeasurementError(
            f"{label} must be a non-empty tool-schema list"
        )
    normalized = [
        normalize_tool_schema(schema, label=f"{label}[{index}]")
        for index, schema in enumerate(schemas)
    ]
    names = [row["function"]["name"] for row in normalized]
    if len(names) != len(set(names)):
        raise V6CandidateMeasurementError(f"{label} has duplicate tool names")
    return sorted(normalized, key=lambda row: row["function"]["name"])


def _tool_result_is_error(
    messages: Sequence[Mapping[str, Any]], assistant_index: int
) -> bool:
    if assistant_index + 1 >= len(messages):
        return False
    result = messages[assistant_index + 1]
    return result.get("role") == "tool" and result.get("error") is True


def assistant_label_mask(
    messages: Sequence[Mapping[str, Any]],
    *,
    assistant_start: int,
) -> list[bool]:
    """Label every non-failed assistant output at/after ``assistant_start``.

    The explicit boundary is what keeps the injected failed call in the
    recovery prompt at zero loss.  A later assistant tool call whose adjacent
    result is also an error is conservatively masked as well.
    """
    if not 0 <= assistant_start <= len(messages):
        raise V6CandidateMeasurementError("assistant_start is outside messages")
    mask: list[bool] = []
    for index, message in enumerate(messages):
        selected = (
            index >= assistant_start
            and message.get("role") == "assistant"
            and (
                message.get("content") not in (None, "")
                or bool(message.get("tool_calls"))
            )
            and not _tool_result_is_error(messages, index)
        )
        mask.append(bool(selected))
    if not any(mask):
        raise V6CandidateMeasurementError(
            "message sequence has no non-failed assistant target"
        )
    return mask


def tokenize_labeled_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    label_mask: Sequence[bool],
    tool_schemas: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    """Map message labels to exact prefix-stable chat-template token spans."""
    if len(messages) != len(label_mask) or any(
        type(value) is not bool for value in label_mask
    ):
        raise V6CandidateMeasurementError(
            f"{label}: label_mask must contain one boolean per message"
        )
    normalized = normalize_messages(messages, label=f"{label}.messages")
    tools = normalize_tool_schemas(
        list(tool_schemas), label=f"{label}.tool_schemas"
    )
    try:
        full_ids = v5_train._token_ids(
            tokenizer, normalized, tools=tools, generation=False
        )
    except RuntimeError as error:
        raise V6CandidateMeasurementError(
            f"{label}: complete chat-template rendering failed: {error}"
        ) from error
    if not full_ids:
        raise V6CandidateMeasurementError(
            f"{label}: chat template produced no tokens"
        )
    spans: list[dict[str, int]] = []
    for message_index, selected in enumerate(label_mask):
        if not selected:
            continue
        if normalized[message_index].get("role") != "assistant":
            raise V6CandidateMeasurementError(
                f"{label}: selected message {message_index} is not assistant"
            )
        try:
            before = v5_train._token_ids(
                tokenizer,
                normalized[:message_index],
                tools=tools,
                generation=True,
            )
            through = v5_train._token_ids(
                tokenizer,
                normalized[: message_index + 1],
                tools=tools,
                generation=False,
            )
        except RuntimeError as error:
            raise V6CandidateMeasurementError(
                f"{label}: target-span rendering failed: {error}"
            ) from error
        if len(before) >= len(through) or through[: len(before)] != before:
            raise V6CandidateMeasurementError(
                f"{label}: chat template is not prefix-stable at message "
                f"{message_index}"
            )
        if full_ids[: len(through)] != through:
            raise V6CandidateMeasurementError(
                f"{label}: future messages alter target prefix at message "
                f"{message_index}"
            )
        spans.append(
            {
                "message_index": message_index,
                "token_start": len(before),
                "token_end": len(through),
            }
        )
    supervised = sum(
        span["token_end"] - span["token_start"] for span in spans
    )
    if supervised <= 0 or supervised > len(full_ids):
        raise V6CandidateMeasurementError(
            f"{label}: invalid supervised-token total {supervised}"
        )
    return {
        "input_ids": full_ids,
        "sequence_tokens": len(full_ids),
        "supervised_tokens": supervised,
        "label_spans": spans,
        "normalized_messages_sha256": canonical_sha256(normalized),
        "tool_schemas_sha256": canonical_sha256(tools),
    }


def _compact_action(call: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(call, Mapping):
        raise V6CandidateMeasurementError(f"{label} must be an object")
    function = call.get("function")
    if function is not None:
        if not isinstance(function, Mapping):
            raise V6CandidateMeasurementError(f"{label}.function must be an object")
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", {})
    name = _nonempty(name, f"{label}.name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise V6CandidateMeasurementError(
                f"{label}.arguments is not valid JSON"
            ) from error
    if not isinstance(arguments, dict):
        raise V6CandidateMeasurementError(
            f"{label}.arguments must encode an object"
        )
    # Provider call IDs and requestor metadata are transport evidence, not
    # semantic action tokens.  The raw registered call remains untouched.
    return {
        "name": name,
        "arguments": json.loads(canonical(arguments)),
    }


def registered_corrective_action(
    branch: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    registered = (
        (branch.get("producer_evidence") or {}).get("registered_branch")
        if isinstance(branch.get("producer_evidence"), Mapping)
        else None
    )
    owners = [branch]
    if isinstance(registered, Mapping):
        owners.insert(0, registered)
    for owner in owners:
        specification = owner.get("corrective_action_spec")
        constructor = (
            specification.get("forced_first_action_constructor")
            if isinstance(specification, Mapping)
            else None
        )
        call = (
            constructor.get("tool_call")
            if isinstance(constructor, Mapping)
            else None
        )
        if isinstance(call, Mapping):
            return _compact_action(call, label=label)
    fallback = branch.get("first_recovery_action")
    if isinstance(fallback, Mapping):
        return _compact_action(fallback, label=label)
    raise V6CandidateMeasurementError(
        f"{label}: registered corrective action is absent"
    )


def action_target_message(action: Mapping[str, Any]) -> dict[str, Any]:
    compact = _compact_action(action, label="corrective_action")
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "name": compact["name"],
                "arguments": deepcopy(compact["arguments"]),
            }
        ],
    }


def make_logprob_cell(
    sum_logprob: float,
    token_count: int,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        isinstance(sum_logprob, bool)
        or not isinstance(sum_logprob, (int, float))
        or not math.isfinite(float(sum_logprob))
        or float(sum_logprob) > 1e-8
    ):
        raise V6CandidateMeasurementError(
            "sum_logprob must be a finite non-positive number"
        )
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise V6CandidateMeasurementError("token_count must be a positive integer")
    total = float(sum_logprob)
    cell: dict[str, Any] = {
        "sum_logprob": total,
        "token_count": token_count,
        "mean_logprob": total / token_count,
        "length_normalized": True,
    }
    if evidence:
        cell.update(deepcopy(dict(evidence)))
    return cell


ActionScorer = Callable[
    [Sequence[Mapping[str, Any]], Mapping[str, Any], Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]


def four_cell_first_action_logprobs(
    branches: Sequence[Mapping[str, Any]],
    tool_schemas: Sequence[Mapping[str, Any]],
    *,
    scorer: ActionScorer,
) -> dict[str, dict[str, Any]]:
    """Score both registered sibling actions under both observed error prompts."""
    if (
        len(branches) != 2
        or any(not isinstance(branch, Mapping) for branch in branches)
    ):
        raise V6CandidateMeasurementError(
            "candidate pair must contain exactly two sibling branches"
        )
    prompts = [
        _messages(
            branch.get(
                "training_prompt",
                branch.get("recovery_prompt", branch.get("prompt")),
            ),
            f"branches[{index}].training_prompt",
        )
        for index, branch in enumerate(branches)
    ]
    actions = [
        registered_corrective_action(
            branch, label=f"branches[{index}].corrective_action"
        )
        for index, branch in enumerate(branches)
    ]
    if canonical(actions[0]) == canonical(actions[1]):
        raise V6CandidateMeasurementError(
            "sibling registered corrective actions must be distinct"
        )
    cells = {
        "q_e1_a1": (prompts[0], actions[0]),
        "q_e1_a2": (prompts[0], actions[1]),
        "q_e2_a1": (prompts[1], actions[0]),
        "q_e2_a2": (prompts[1], actions[1]),
    }
    output: dict[str, dict[str, Any]] = {}
    for key in Q_KEYS:
        prompt, action = cells[key]
        measured = dict(scorer(prompt, action, tool_schemas))
        output[key] = make_logprob_cell(
            measured.get("sum_logprob"),
            measured.get("token_count"),
            evidence={
                **{
                    name: deepcopy(value)
                    for name, value in measured.items()
                    if name not in {"sum_logprob", "token_count", "mean_logprob"}
                },
                "error_prompt_sha256": canonical_sha256(prompt),
                "corrective_action_sha256": canonical_sha256(action),
            },
        )
    if set(output) != set(Q_KEYS):
        raise V6CandidateMeasurementError("four-cell log-probability grid is incomplete")
    return output


def _candidate_tool_schema_sources(
    row: Mapping[str, Any]
) -> list[Any]:
    candidates: list[Any] = []
    for owner in (row, row.get("clean_view"), row.get("generation_contract")):
        if isinstance(owner, Mapping) and "tool_schemas" in owner:
            candidates.append(owner["tool_schemas"])
    branches = row.get("branches")
    if isinstance(branches, list):
        for branch in branches:
            if isinstance(branch, Mapping) and "tool_schemas" in branch:
                candidates.append(branch["tool_schemas"])
    return candidates


def resolve_tool_schemas(
    row: Mapping[str, Any],
    *,
    tau2_schemas: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> list[dict[str, Any]]:
    candidates = _candidate_tool_schema_sources(row)
    if not candidates and tau2_schemas is not None:
        domain = _nonempty(row.get("domain"), "candidate.domain")
        if domain in tau2_schemas:
            candidates.append(list(tau2_schemas[domain]))
    if not candidates:
        raise V6CandidateMeasurementError(
            "candidate has no immutable tool schemas; provide --tau2-root"
        )
    normalized = [
        normalize_tool_schemas(value, label=f"tool_schema_source[{index}]")
        for index, value in enumerate(candidates)
    ]
    identities = {canonical(value) for value in normalized}
    if len(identities) != 1:
        raise V6CandidateMeasurementError(
            "tool schemas differ across candidate evidence layers"
        )
    return normalized[0]


def load_tau2_tool_schemas(
    tau2_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load immutable domain schemas without importing tau2 at module import."""
    root = tau2_root.resolve()
    source = root / "src"
    if not source.is_dir():
        raise V6CandidateMeasurementError(
            f"tau2 source directory does not exist: {source}"
        )
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise V6CandidateMeasurementError(
            f"cannot resolve tau2 commit at {root}"
        ) from error
    sys.path.insert(0, str(source))
    try:
        from tau2.domains.airline.environment import get_environment as airline_env
        from tau2.domains.retail.environment import get_environment as retail_env

        environments = {"retail": retail_env(), "airline": airline_env()}
        schemas = {
            domain: normalize_tool_schemas(
                [tool.openai_schema for tool in environment.get_tools()],
                label=f"tau2.{domain}.tool_schemas",
            )
            for domain, environment in environments.items()
        }
    except Exception as error:
        raise V6CandidateMeasurementError(
            f"failed to load tau2 tool schemas: {error}"
        ) from error
    finally:
        if sys.path and sys.path[0] == str(source):
            sys.path.pop(0)
    provenance = {
        "tau2_root": str(root),
        "tau2_commit": commit,
        "domain_tool_schema_sha256": {
            domain: canonical_sha256(value)
            for domain, value in sorted(schemas.items())
        },
    }
    return schemas, provenance


class FrozenHFActionScorer:
    """Exact target-span scorer for one frozen Hugging Face causal LM."""

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str,
        tokenizer_name: str,
        tokenizer_revision: str,
        load_in_4bit: bool,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> None:
        model_revision = require_revision(model_revision, "model_revision")
        tokenizer_revision = require_revision(
            tokenizer_revision, "tokenizer_revision"
        )
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise V6CandidateMeasurementError(
                "runtime scoring requires torch and transformers"
            ) from error
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if load_in_4bit and device != "cuda":
            raise V6CandidateMeasurementError(
                "--load-in-4bit requires a CUDA device"
            )
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype == "auto":
            model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            model_dtype = dtype_map[dtype]
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=tokenizer_revision,
            local_files_only=local_files_only,
            use_fast=True,
        )
        model_kwargs: dict[str, Any] = {
            "revision": model_revision,
            "local_files_only": local_files_only,
            "torch_dtype": model_dtype,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=model_dtype,
            )
            model_kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        if not load_in_4bit:
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        model_config = self.model.config.to_dict()
        tokenizer_identity = {
            "class": type(self.tokenizer).__name__,
            "name_or_path": self.tokenizer.name_or_path,
            "vocab_size": int(len(self.tokenizer)),
            "model_max_length": int(self.tokenizer.model_max_length),
            "special_tokens_map": self.tokenizer.special_tokens_map,
            "chat_template": self.tokenizer.chat_template,
            "requested_name": tokenizer_name,
            "requested_revision": tokenizer_revision,
            "resolved_commit": self.tokenizer.init_kwargs.get("_commit_hash"),
        }
        self.provenance = {
            "protocol": PROTOCOL,
            "model_name": model_name,
            "model_revision": model_revision,
            "resolved_model_commit": getattr(
                self.model.config, "_commit_hash", None
            ),
            "tokenizer_name": tokenizer_name,
            "tokenizer_revision": tokenizer_revision,
            "load_in_4bit": load_in_4bit,
            "quantization": "bitsandbytes_nf4_double_quant" if load_in_4bit else "none",
            "requested_dtype": dtype,
            "resolved_torch_dtype": str(model_dtype),
            "device_type": self.device.type,
            "model_config_sha256": canonical_sha256(model_config),
            "tokenizer_identity_sha256": canonical_sha256(tokenizer_identity),
            "transformers_version": importlib.metadata.version("transformers"),
            "torch_version": importlib.metadata.version("torch"),
            "official_test_used": False,
            "validation_used": False,
        }
        self.provenance["frozen_checkpoint_identity_sha256"] = canonical_sha256(
            {
                key: self.provenance[key]
                for key in (
                    "model_name",
                    "model_revision",
                    "resolved_model_commit",
                    "model_config_sha256",
                    "tokenizer_name",
                    "tokenizer_revision",
                    "tokenizer_identity_sha256",
                    "load_in_4bit",
                    "quantization",
                )
            }
        )

    def __call__(
        self,
        prompt: Sequence[Mapping[str, Any]],
        action: Mapping[str, Any],
        tool_schemas: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        target = action_target_message(action)
        complete = [*deepcopy(list(prompt)), target]
        mask = [False] * len(prompt) + [True]
        contract = tokenize_labeled_messages(
            self.tokenizer,
            complete,
            mask,
            tool_schemas,
            label="first_action",
        )
        span = contract["label_spans"][0]
        start = span["token_start"]
        end = span["token_end"]
        if start <= 0:
            raise V6CandidateMeasurementError(
                "first-action target starts before a causal prediction position"
            )
        input_ids = self.torch.tensor(
            [contract["input_ids"]],
            dtype=self.torch.long,
            device=self.device,
        )
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            target_logits = logits[:, start - 1 : end - 1, :].float()
            target_ids = input_ids[:, start:end]
            values = self.torch.log_softmax(target_logits, dim=-1).gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
        total = float(values.sum().item())
        count = int(target_ids.numel())
        return {
            "sum_logprob": min(total, 0.0),
            "token_count": count,
            "token_start": start,
            "token_end": end,
            "target_span_sha256": canonical_sha256(
                contract["input_ids"][start:end]
            ),
            "frozen_checkpoint_identity_sha256": self.provenance[
                "frozen_checkpoint_identity_sha256"
            ],
        }


def _branch_trace(
    branch: Mapping[str, Any], *, label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    prompt = _messages(
        branch.get("recovery_prompt", branch.get("prompt")),
        f"{label}.recovery_prompt",
    )
    suffix = _messages(
        branch.get(
            "recovery_suffix", branch.get("fresh_recovery_suffix")
        ),
        f"{label}.fresh_recovery_suffix",
    )
    combined = [*deepcopy(prompt), *deepcopy(suffix)]
    supplied = branch.get("full_trace", branch.get("full_recovery_messages"))
    if supplied is not None and canonical(supplied) != canonical(combined):
        raise V6CandidateMeasurementError(
            f"{label}: full trace differs from recovery prompt plus fresh suffix"
        )
    return combined, suffix, len(prompt)


def enrich_candidate(
    raw: Mapping[str, Any],
    *,
    tokenizer: Any,
    scorer: ActionScorer,
    model_provenance: Mapping[str, Any],
    tau2_schemas: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    tau2_provenance: Mapping[str, Any] | None = None,
    source_pool_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a deep-copied raw candidate with exact measurement fields."""
    pair_id = _nonempty(raw.get("candidate_pair_id"), "candidate_pair_id")
    if raw.get("official_test_used") is not False:
        raise V6CandidateMeasurementError(
            f"{pair_id}: official-test records are forbidden"
        )
    if raw.get("partition") not in (None, "arm_train"):
        raise V6CandidateMeasurementError(
            f"{pair_id}: only arm_train candidates may be measured"
        )
    branches = raw.get("branches")
    if (
        not isinstance(branches, list)
        or len(branches) != 2
        or any(not isinstance(branch, Mapping) for branch in branches)
    ):
        raise V6CandidateMeasurementError(
            f"{pair_id}: exactly two sibling branches are required"
        )
    schemas = resolve_tool_schemas(raw, tau2_schemas=tau2_schemas)
    enriched = deepcopy(dict(raw))
    enriched["tool_schemas"] = deepcopy(schemas)
    system_message = raw.get("training_system_message")
    if (
        not isinstance(system_message, Mapping)
        or system_message.get("role") != "system"
        or not isinstance(system_message.get("content"), str)
        or not system_message["content"].strip()
        or raw.get("training_system_message_sha256")
        != canonical_sha256(system_message)
    ):
        raise V6CandidateMeasurementError(
            f"{pair_id}: missing or drifted training system/policy message"
        )
    system_message = deepcopy(dict(system_message))
    branch_sup = 0
    branch_nonpad = 0
    for index, branch in enumerate(enriched["branches"]):
        label = f"{pair_id}.branches[{index}]"
        full_messages, suffix, suffix_start = _branch_trace(
            branch, label=label
        )
        if any(message.get("role") == "system" for message in full_messages):
            raise V6CandidateMeasurementError(
                f"{label}: raw tau2 trace must not contain a system message; "
                "the frozen policy message is injected exactly once"
            )
        training_prompt = [deepcopy(system_message), *full_messages[:suffix_start]]
        training_messages = [deepcopy(system_message), *full_messages]
        mask = assistant_label_mask(
            training_messages, assistant_start=1 + suffix_start
        )
        contract = tokenize_labeled_messages(
            tokenizer,
            training_messages,
            mask,
            schemas,
            label=label,
        )
        suffix_mask = mask[1 + suffix_start :]
        if len(suffix_mask) != len(suffix):
            raise V6CandidateMeasurementError(
                f"{label}: fresh suffix label-mask length drift"
            )
        branch["tool_schemas"] = deepcopy(schemas)
        branch["prompt"] = deepcopy(training_prompt)
        branch["training_prompt"] = deepcopy(training_prompt)
        branch["training_full_trace"] = deepcopy(training_messages)
        branch["fresh_recovery_label_mask"] = suffix_mask
        branch["full_assistant_label_mask"] = mask
        branch["supervised_target_tokens"] = contract["supervised_tokens"]
        branch["nonpadding_tokens"] = contract["sequence_tokens"]
        branch["token_contract"] = {
            "sequence_tokens": contract["sequence_tokens"],
            "supervised_tokens": contract["supervised_tokens"],
            "label_spans": contract["label_spans"],
        }
        branch["token_measurement_status"] = "MEASURED_FROZEN_STUDENT"
        branch_sup += contract["supervised_tokens"]
        branch_nonpad += contract["sequence_tokens"]
    clean_view = enriched.get("clean_view")
    if not isinstance(clean_view, dict):
        raise V6CandidateMeasurementError(
            f"{pair_id}: complete clean_view is required"
        )
    clean_raw_messages = _messages(
        clean_view.get("messages"), f"{pair_id}.clean_view.messages"
    )
    if any(message.get("role") == "system" for message in clean_raw_messages):
        raise V6CandidateMeasurementError(
            f"{pair_id}: raw clean trace unexpectedly contains a system message"
        )
    shared_prefix = _messages(
        enriched.get("shared_prefix"), f"{pair_id}.shared_prefix"
    )
    if clean_raw_messages[: len(shared_prefix)] != shared_prefix:
        raise V6CandidateMeasurementError(
            f"{pair_id}: clean view does not begin with the frozen shared prefix"
        )
    clean_messages = [deepcopy(system_message), *clean_raw_messages]
    clean_mask = assistant_label_mask(
        clean_messages, assistant_start=1 + len(shared_prefix)
    )
    clean_contract = tokenize_labeled_messages(
        tokenizer,
        clean_messages,
        clean_mask,
        schemas,
        label=f"{pair_id}.clean_view",
    )
    if "label_mask" in clean_view:
        clean_view["producer_label_mask"] = deepcopy(clean_view["label_mask"])
    clean_view["raw_messages"] = deepcopy(clean_raw_messages)
    clean_view["messages"] = deepcopy(clean_messages)
    clean_view["label_mask"] = clean_mask
    clean_view["tool_schemas"] = deepcopy(schemas)
    clean_view["c_sup"] = clean_contract["supervised_tokens"]
    clean_view["c_nonpad"] = clean_contract["sequence_tokens"]
    clean_view["token_contract"] = {
        "sequence_tokens": clean_contract["sequence_tokens"],
        "supervised_tokens": clean_contract["supervised_tokens"],
        "label_spans": clean_contract["label_spans"],
    }
    clean_view["token_measurement_status"] = "MEASURED_FROZEN_STUDENT"
    enriched["first_action_logprobs"] = four_cell_first_action_logprobs(
        enriched["branches"], schemas, scorer=scorer
    )
    enriched["token_accounting"] = {
        "supervised_target_tokens": branch_sup,
        "nonpadding_tokens": branch_nonpad,
        "branch_count": 2,
        "selector_atom": "complete_candidate_pair_two_sibling_branches",
        "status": "MEASURED_FROZEN_STUDENT",
    }
    raw_hash = canonical_sha256(raw)
    provenance = {
        **deepcopy(dict(model_provenance)),
        "protocol": PROTOCOL,
        "raw_candidate_record_sha256": raw_hash,
        "source_pool_sha256": source_pool_sha256,
        "tool_schemas_sha256": canonical_sha256(schemas),
        "training_system_message_sha256": canonical_sha256(system_message),
        "tau2": deepcopy(dict(tau2_provenance or {})),
        "measurement_uses_train_pool_only": True,
        "validation_used": False,
        "official_test_used": False,
        "failed_calls_supervised": False,
        "full_fresh_assistant_suffix_supervised": True,
    }
    provenance["measurement_contract_sha256"] = canonical_sha256(
        {
            key: provenance.get(key)
            for key in (
                "protocol",
                "model_name",
                "model_revision",
                "tokenizer_name",
                "tokenizer_revision",
                "frozen_checkpoint_identity_sha256",
                "tool_schemas_sha256",
                "failed_calls_supervised",
                "full_fresh_assistant_suffix_supervised",
            )
        }
    )
    enriched["token_measurement_provenance"] = provenance
    return enriched


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise V6CandidateMeasurementError(f"input JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise V6CandidateMeasurementError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(row, dict):
                raise V6CandidateMeasurementError(
                    f"{path}:{line_number}: row is not an object"
                )
            rows.append(row)
    if not rows:
        raise V6CandidateMeasurementError(f"{path}: input pool is empty")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise V6CandidateMeasurementError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(canonical(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument(
        "--tau2-root",
        type=Path,
        help=(
            "Pinned tau2 checkout used to recover immutable domain tool "
            "schemas when raw generation rows do not embed them."
        ),
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.model_revision = require_revision(
        args.model_revision, "--model-revision"
    )
    args.tokenizer_revision = require_revision(
        args.tokenizer_revision, "--tokenizer-revision"
    )
    args.model = _nonempty(args.model, "--model")
    args.tokenizer = _nonempty(
        args.tokenizer or args.model, "--tokenizer"
    )
    if (
        args.model != DEFAULT_MODEL
        or args.tokenizer != DEFAULT_MODEL
        or args.model_revision != PINNED_MODEL_REVISION
        or args.tokenizer_revision != PINNED_MODEL_REVISION
    ):
        raise V6CandidateMeasurementError(
            "V6 freezes Qwen2.5-7B model and tokenizer at revision "
            f"{PINNED_MODEL_REVISION}"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise V6CandidateMeasurementError(
            f"output already exists: {args.output}"
        )
    rows = read_jsonl(args.input)
    source_pool_sha256 = sha256_file(args.input)
    tau2_schemas = None
    tau2_provenance = None
    if args.tau2_root is not None:
        tau2_schemas, tau2_provenance = load_tau2_tool_schemas(
            args.tau2_root
        )
    scorer = FrozenHFActionScorer(
        model_name=args.model,
        model_revision=args.model_revision,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        load_in_4bit=args.load_in_4bit,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    enriched = [
        enrich_candidate(
            row,
            tokenizer=scorer.tokenizer,
            scorer=scorer,
            model_provenance=scorer.provenance,
            tau2_schemas=tau2_schemas,
            tau2_provenance=tau2_provenance,
            source_pool_sha256=source_pool_sha256,
        )
        for row in rows
    ]
    write_jsonl(args.output, enriched)


if __name__ == "__main__":
    main()
