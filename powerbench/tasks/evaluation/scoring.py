from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import jsonschema

from ..models import GoldRecord, PublicQuestion, answer_schema
from ..records import expand_gold_records
from ..selector import json_selector_matches, markdown_selector_matches
from ..task_specs import normalize_task_family
from ..util import file_sha256, parse_timestamp, write_json, write_jsonl
from .manifest import (
    ATTEMPT_IDENTITY_FIELDS,
    load_manifest_binding,
    validate_attempt_identities,
)

INFRASTRUCTURE_STATUSES = frozenset({"infrastructure_error", "api_error"})
RETRYABLE_ATTEMPT_STATUSES = INFRASTRUCTURE_STATUSES
ERROR_TAXONOMY = (
    "retrieval",
    "window_calculation",
    "rule_execution",
    "cross_device_reduction",
    "format",
)
TERMINAL_ATTEMPT_STATUSES = frozenset(
    {
        "completed",
        "format_error",
        "tool_budget_exhausted",
        "response_incomplete",
        "refusal",
        "time_budget_exhausted",
    }
)
_MISSING = object()
SCORING_PROTOCOL = "partial_fields_v1"


def _per_device_by_alias(value: Any, *, partial: bool = False) -> dict[str, dict[str, Any]]:
    """Normalize legacy mappings and v0.2 record lists to alias-keyed records."""

    if isinstance(value, dict):
        return {
            str(alias): record
            for alias, record in value.items()
            if isinstance(alias, str) and isinstance(record, dict)
        }
    if not isinstance(value, list):
        return {}
    records: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            if partial:
                continue
            return {}
        alias = item.get("alias")
        if not isinstance(alias, str) or not alias:
            if partial:
                continue
            return {}
        if alias in records or alias in ambiguous:
            if partial:
                records.pop(alias, None)
                ambiguous.add(alias)
                continue
            return {}
        records[alias] = {key: field for key, field in item.items() if key != "alias"}
    return records


def classify_error(row: Mapping[str, Any]) -> str:
    """Classify one scored run using only observable evaluation fields.

    This is intentionally the single implementation used both while scoring a
    run and while auditing the resulting score file for the pilot report.
    """

    status = str(row.get("status", "completed"))
    if status in INFRASTRUCTURE_STATUSES:
        return "infrastructure"
    if status == "tool_budget_exhausted":
        return "retrieval"
    if status == "time_budget_exhausted":
        return "time_budget"
    if status in {"response_incomplete", "refusal"}:
        return "unclassified"
    if status == "format_error" or row.get("format_valid") is False:
        return "format"
    if float(row.get("joint_accuracy") or 0.0) == 1.0:
        return "none"
    if float(row.get("available_evidence_recall") or 0.0) == 0.0:
        return "retrieval"
    level = row.get("task_level")
    if level == "L1":
        return "window_calculation"
    if level == "L2":
        return "rule_execution"
    if level == "L3":
        local_joint = row.get("l3_local_joint_accuracy")
        if isinstance(local_joint, (int, float)) and float(local_joint) < 1.0:
            return "rule_execution"
        return "cross_device_reduction"
    return "unclassified"


class _SourceSnapshots:
    """Parse and hash each evaluator input from the same byte stream."""

    def __init__(self) -> None:
        self._digests: dict[Path, str] = {}
        self._json_objects: dict[Path, dict[str, Any]] = {}
        self._jsonl_rows: dict[Path, list[dict[str, Any]]] = {}

    @staticmethod
    def _key(path: str | Path) -> Path:
        return Path(path).resolve()

    def _record_digest(self, path: Path, digest: str) -> None:
        previous = self._digests.get(path)
        if previous is not None and previous != digest:
            raise RuntimeError(f"source changed while it was being captured: {path}")
        self._digests[path] = digest

    def load_json_object(self, path: str | Path, *, label: str) -> dict[str, Any]:
        key = self._key(path)
        if key in self._json_objects:
            return self._json_objects[key]
        try:
            payload = key.read_bytes()
            value = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load {label} {key}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object: {key}")
        self._record_digest(key, hashlib.sha256(payload).hexdigest())
        self._json_objects[key] = value
        return value

    def read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        key = self._key(path)
        if key in self._jsonl_rows:
            return self._jsonl_rows[key]
        digest = hashlib.sha256()
        rows: list[dict[str, Any]] = []
        with key.open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if raw_line.strip():
                    rows.append(json.loads(raw_line.decode("utf-8")))
        self._record_digest(key, digest.hexdigest())
        self._jsonl_rows[key] = rows
        return rows

    def digest(self, path: str | Path) -> str:
        key = self._key(path)
        if key not in self._digests:
            self._digests[key] = file_sha256(key)
        return self._digests[key]

    def changed_sources(self) -> list[str]:
        errors: list[str] = []
        for path, expected in sorted(self._digests.items()):
            try:
                actual = file_sha256(path)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                continue
            if actual != expected:
                errors.append(str(path))
        return errors


def _parse_prediction(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _compare_scalar(path: str, expected: Any, actual: Any, tolerances: dict[str, float]) -> bool:
    if actual is _MISSING:
        return False
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        tolerance = float(tolerances.get(path, tolerances.get(path.rsplit(".", 1)[-1], 0.0)))
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance)
    return expected == actual


def _leaf_scores(
    expected: Any,
    actual: Any,
    tolerances: dict[str, float],
    *,
    path: str = "",
    partial: bool = False,
) -> dict[str, bool]:
    """In partial mode, keep a fixed Gold-defined denominator despite missing values."""
    if path.endswith("per_device") and isinstance(expected, (dict, list)):
        expected_by_alias = _per_device_by_alias(expected)
        actual_by_alias = _per_device_by_alias(actual, partial=partial)
        if not expected_by_alias or (
            not partial and set(expected_by_alias) != set(actual_by_alias)
        ):
            return {path: False}
        result: dict[str, bool] = {}
        for alias, record in expected_by_alias.items():
            result.update(
                _leaf_scores(
                    record,
                    actual_by_alias.get(alias, _MISSING),
                    tolerances,
                    path=f"{path}.{alias}",
                    partial=partial,
                )
            )
        return result
    if isinstance(expected, dict):
        actual_dict = actual if isinstance(actual, dict) else {}
        result: dict[str, bool] = {}
        for key, value in expected.items():
            child = f"{path}.{key}" if path else key
            child_actual = actual_dict.get(key, _MISSING)
            result.update(_leaf_scores(value, child_actual, tolerances, path=child, partial=partial))
        return result
    if isinstance(expected, list):
        if path.endswith("triggered_devices"):
            valid = (
                isinstance(actual, list)
                and all(isinstance(value, str) for value in actual)
                and set(expected) == set(actual)
                and len(expected) == len(actual)
            )
            return {path: valid}
        if not partial and (not isinstance(actual, list) or len(expected) != len(actual)):
            return {path: False}
        actual_list = actual if isinstance(actual, list) else []
        result = {}
        for index, value in enumerate(expected):
            child_actual = actual_list[index] if index < len(actual_list) else _MISSING
            result.update(_leaf_scores(
                value, child_actual, tolerances, path=f"{path}[{index}]", partial=partial
            ))
        return result
    return {path: _compare_scalar(path, expected, actual, tolerances)}


def _set_f1(expected: Iterable[str], actual: Iterable[str]) -> float:
    gold = set(expected)
    predicted = set(actual)
    if not gold and not predicted:
        return 1.0
    if not predicted:
        return 0.0
    overlap = len(gold & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(gold) if gold else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _absolute_error(expected: Any, actual: Any) -> float | None:
    """Return an absolute numeric error, excluding booleans and missing values."""

    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return abs(float(expected) - float(actual))
    return None


def _numeric_observed(expected: Any, actual: Any, *, format_valid: bool) -> float:
    """Expose the denominator for an optional absolute-error observation."""

    return float(format_valid and _absolute_error(expected, actual) is not None)


def _mean_available(values: Iterable[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def _all_leaf_scores(leaves: dict[str, bool], path: str) -> bool:
    selected = [value for key, value in leaves.items() if key == path or key.startswith(f"{path}[")]
    return bool(selected) and all(selected)


def _is_model_event(event: dict[str, Any]) -> bool:
    """Return whether an access event was performed by the evaluated model."""

    actor = str(event.get("actor", "model")).casefold()
    return actor not in {"oracle", "evaluator"} and event.get("event") != "oracle_prefetch"


def _event_step(event: dict[str, Any], fallback: int) -> int:
    raw = event.get("step")
    try:
        return int(raw) if raw is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _event_paths(event: dict[str, Any]) -> list[str]:
    values = event.get("paths")
    if not isinstance(values, list):
        value = event.get("path")
        values = [value] if isinstance(value, str) else []
    return [path for path in values if isinstance(path, str)]


def _trace_paths(trace: list[dict[str, Any]], stage: str) -> tuple[set[str], dict[str, int]]:
    paths: set[str] = set()
    first_step: dict[str, int] = {}
    for fallback_step, event in enumerate(trace, start=1):
        if (
            not _is_model_event(event)
            or str(event.get("access_stage", "")).casefold() != stage.casefold()
        ):
            continue
        step = _event_step(event, fallback_step)
        for path in _event_paths(event):
            paths.add(path)
            first_step.setdefault(path, step)
    return paths, first_step


def _read_observations(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge model tool results and corpus access events into auditable reads."""

    observations: list[dict[str, Any]] = []
    for fallback_step, event in enumerate(trace, start=1):
        if (
            not _is_model_event(event)
            or event.get("event") != "tool_call"
            or event.get("tool") not in {"read_file", "read_timeseries"}
        ):
            continue
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        path = result.get("path") or arguments.get("path")
        observations.append(
            {
                "step": _event_step(event, fallback_step),
                "tool": event.get("tool"),
                "path": path,
                "error": event.get("error") or result.get("error"),
                "status": result.get("status"),
                "supported": result.get("supported"),
                "row_count": result.get("row_count", result.get("line_count")),
                "byte_count": result.get("byte_count"),
                "fields": result.get("fields", result.get("columns")),
                "start": result.get("start"),
                "end": result.get("end"),
                "content": result.get("content"),
            }
        )

    for fallback_step, event in enumerate(trace, start=1):
        if not _is_model_event(event) or str(event.get("access_stage", "")).casefold() != "read":
            continue
        step = _event_step(event, fallback_step)
        for path in _event_paths(event):
            matching = next(
                (
                    item
                    for item in reversed(observations)
                    if item.get("step") == step
                    and item.get("path") == path
                    and (
                        not event.get("tool")
                        or not item.get("tool")
                        or event.get("tool") == item.get("tool")
                    )
                ),
                None,
            )
            if matching is None:
                matching = {
                    "step": step,
                    "tool": event.get("tool"),
                    "path": path,
                }
                observations.append(matching)
            for key in (
                "status",
                "row_count",
                "byte_count",
                "fields",
                "start",
                "end",
                "content",
                "supported",
                "error",
            ):
                if event.get(key) is not None:
                    matching[key] = event[key]
    return observations


def _snippet_observations(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return factual snippets that a search call actually exposed to the model."""

    observations: list[dict[str, Any]] = []
    for fallback_step, event in enumerate(trace, start=1):
        if (
            not _is_model_event(event)
            or event.get("event") != "tool_call"
            or event.get("tool") != "search_corpus"
        ):
            continue
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        rows = result.get("results") if isinstance(result.get("results"), list) else []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                continue
            observations.append(
                {
                    "step": _event_step(event, fallback_step),
                    "tool": "search_corpus",
                    "path": row["path"],
                    "content": row.get("snippet"),
                    "status": "ok",
                    "supported": True,
                }
            )
    return observations


def _successful_read(observation: dict[str, Any]) -> bool:
    if observation.get("error") or observation.get("supported") is False:
        return False
    status = str(observation.get("status") or "").casefold()
    if status and status not in {"ok", "success", "succeeded", "completed"}:
        return False
    count = observation.get("row_count")
    if count is not None:
        try:
            if int(count) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    return isinstance(observation.get("path"), str)


def _required_csv_fields(selector: dict[str, Any]) -> set[str]:
    raw = selector.get("fields", selector.get("field"))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(value) for value in raw if str(value)}
    return set()


def _csv_selector_matches(observation: dict[str, Any], selector: dict[str, Any]) -> bool:
    if observation.get("tool") not in {None, "read_timeseries"}:
        return False
    raw_fields = observation.get("fields")
    observed_fields = (
        {str(value) for value in raw_fields}
        if isinstance(raw_fields, (list, tuple, set))
        else set()
    )
    observed_fields.discard("timestamp")
    if not _required_csv_fields(selector).issubset(observed_fields):
        return False

    required_start = selector.get("start")
    required_end = selector.get("end_exclusive")
    if required_start is None and required_end is None:
        return True
    if required_start is None or required_end is None:
        return False
    try:
        start = parse_timestamp(str(required_start)) - timedelta(
            hours=int(selector.get("lookback_hours") or 0)
        )
        end = parse_timestamp(str(required_end))
        observed_start = parse_timestamp(str(observation.get("start")))
        observed_end = parse_timestamp(str(observation.get("end")))
    except (TypeError, ValueError):
        return False
    return observed_start <= start and observed_end >= end


def _source_matches_read(source: dict[str, Any], observation: dict[str, Any]) -> bool:
    if not _successful_read(observation) or observation.get("path") != source.get("path"):
        return False
    selector = source.get("selector") if isinstance(source.get("selector"), dict) else {}
    suffix = Path(str(source.get("path", ""))).suffix.casefold()
    if suffix == ".csv":
        return _csv_selector_matches(observation, selector)
    if suffix == ".md":
        return markdown_selector_matches(observation.get("content"), selector)
    if suffix == ".json":
        content = observation.get("content")
        return isinstance(content, str) and json_selector_matches(content, selector)
    return not selector


def _provided_atom_steps(trace: list[dict[str, Any]]) -> dict[str, int]:
    provided: dict[str, int] = {}
    for fallback_step, event in enumerate(trace, start=1):
        stage = str(event.get("access_stage", "")).casefold()
        kind = str(event.get("event", "")).casefold()
        if stage != "provided" and kind not in {
            "provided",
            "evidence_provided",
            "oracle_evidence_provided",
        }:
            continue
        values: list[Any] = []
        if event.get("atom_id") is not None:
            values.append(event["atom_id"])
        if isinstance(event.get("atom_ids"), list):
            values.extend(event["atom_ids"])
        step = _event_step(event, fallback_step)
        for value in values:
            if isinstance(value, str):
                provided.setdefault(value, step)
    return provided


def _completion_steps(atom_steps: list[int | None]) -> tuple[int | None, int | None]:
    found = [step for step in atom_steps if step is not None]
    first = min(found) if found else None
    complete = max(found) if len(found) == len(atom_steps) and atom_steps else None
    return first, complete


def _evidence_metrics(
    atoms: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    cited: list[str],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    retrieved, retrieved_steps = _trace_paths(trace, "Retrieved")
    successful_observations = [item for item in observations if _successful_read(item)]
    snippet_observations = _snippet_observations(trace)
    read_paths = {str(item["path"]) for item in successful_observations}
    provided_steps = _provided_atom_steps(trace)
    cited_set = set(cited)

    def path_covered(paths: set[str], atom: dict[str, Any]) -> bool:
        acceptable = {source["path"] for source in atom["acceptable_sources"]}
        return bool(paths & acceptable)

    retrieved_atom_steps: list[int | None] = []
    read_atom_steps: list[int | None] = []
    snippet_atom_steps: list[int | None] = []
    provided_atom_steps: list[int | None] = []
    effective_atom_steps: list[int | None] = []
    cited_flags: list[bool] = []
    for atom in atoms:
        retrieved_matches = [
            retrieved_steps[source["path"]]
            for source in atom["acceptable_sources"]
            if source["path"] in retrieved_steps
        ]
        read_matches = [
            int(observation["step"])
            for source in atom["acceptable_sources"]
            for observation in successful_observations
            if _source_matches_read(source, observation)
        ]
        snippet_matches = [
            int(observation["step"])
            for source in atom["acceptable_sources"]
            for observation in snippet_observations
            if _source_matches_read(source, observation)
        ]
        provided = provided_steps.get(str(atom.get("atom_id")))
        retrieved_atom_steps.append(min(retrieved_matches) if retrieved_matches else None)
        read_atom_steps.append(min(read_matches) if read_matches else None)
        snippet_atom_steps.append(min(snippet_matches) if snippet_matches else None)
        provided_atom_steps.append(provided)
        available = [
            step
            for step in (read_atom_steps[-1], snippet_atom_steps[-1], provided)
            if step is not None
        ]
        effective_atom_steps.append(min(available) if available else None)
        cited_flags.append(path_covered(cited_set, atom))

    retrieved_flags = [step is not None for step in retrieved_atom_steps]
    read_flags = [step is not None for step in read_atom_steps]
    snippet_flags = [step is not None for step in snippet_atom_steps]
    provided_flags = [step is not None for step in provided_atom_steps]
    effective_flags = [step is not None for step in effective_atom_steps]
    first_retrieved, complete_retrieved = _completion_steps(retrieved_atom_steps)
    first_read, complete_read = _completion_steps(read_atom_steps)
    first_snippet, complete_snippet = _completion_steps(snippet_atom_steps)
    first_provided, complete_provided = _completion_steps(provided_atom_steps)
    first_evidence, complete_evidence = _completion_steps(effective_atom_steps)
    denominator = len(atoms) or 1
    metrics = {
        "retrieved_evidence_recall": sum(retrieved_flags) / denominator,
        "read_evidence_recall": sum(read_flags) / denominator,
        "snippet_evidence_recall": sum(snippet_flags) / denominator,
        "provided_evidence_recall": sum(provided_flags) / denominator,
        "available_evidence_recall": sum(effective_flags) / denominator,
        "cited_evidence_recall": sum(cited_flags) / denominator,
        "read_complete_evidence_rate": float(bool(atoms) and all(read_flags)),
        "complete_evidence_rate": float(bool(atoms) and all(effective_flags)),
        "steps_to_first_retrieved": first_retrieved,
        "steps_to_complete_retrieved": complete_retrieved,
        "steps_to_first_read": first_read,
        "steps_to_complete_read": complete_read,
        "steps_to_first_snippet": first_snippet,
        "steps_to_complete_snippet": complete_snippet,
        "steps_to_first_provided": first_provided,
        "steps_to_complete_provided": complete_provided,
        "steps_to_first_evidence": first_evidence,
        "steps_to_complete_evidence": complete_evidence,
        "retrieved_paths": sorted(retrieved),
        "read_paths": sorted(read_paths),
    }
    return metrics


def _retrieval_cost(
    trace: list[dict[str, Any]],
    run: dict[str, Any],
    atoms: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    tools = [
        event for event in trace if event.get("event") == "tool_call" and _is_model_event(event)
    ]
    list_count = sum(event.get("tool") == "list_corpus" for event in tools)
    search_count = sum(event.get("tool") == "search_corpus" for event in tools)
    read_count = sum(event.get("tool") in {"read_file", "read_timeseries"} for event in tools)
    python_count = sum(event.get("tool") == "python" for event in tools)
    read_events = [
        event
        for event in trace
        if str(event.get("access_stage", "")).casefold() == "read" and _is_model_event(event)
    ]
    successful_paths = {str(item["path"]) for item in observations if _successful_read(item)}
    acceptable_paths = {
        str(source["path"]) for atom in atoms for source in atom.get("acceptable_sources", [])
    }
    over_read_count = len(successful_paths.difference(acceptable_paths))
    source_sets = [
        {str(source["path"]) for source in atom.get("acceptable_sources", [])} for atom in atoms
    ]
    source_universe = sorted(set().union(*source_sets)) if source_sets else []
    minimum_file_count = 0
    if source_sets and all(source_sets):
        for size in range(1, len(source_universe) + 1):
            if any(
                all(set(choice).intersection(sources) for sources in source_sets)
                for choice in itertools.combinations(source_universe, size)
            ):
                minimum_file_count = size
                break
    read_to_minimum = len(successful_paths) / minimum_file_count if minimum_file_count else None
    excess_read_ratio = (
        max(0, len(successful_paths) - minimum_file_count) / minimum_file_count
        if minimum_file_count
        else None
    )
    usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
    return {
        "tool_call_count": len(tools),
        "retrieval_tool_call_count": list_count + search_count + read_count,
        "discovery_call_count": list_count + search_count,
        "list_call_count": list_count,
        "search_call_count": search_count,
        "read_call_count": read_count,
        "python_call_count": python_count,
        "distinct_read_files": len(successful_paths),
        "minimum_evidence_file_count": minimum_file_count,
        "read_to_minimum_file_ratio": read_to_minimum,
        "over_read_file_ratio": excess_read_ratio,
        "irrelevant_read_file_ratio": over_read_count / len(successful_paths)
        if successful_paths
        else 0.0,
        "read_bytes": sum(int(event.get("byte_count") or 0) for event in read_events),
        "read_rows": sum(int(event.get("row_count") or 0) for event in read_events),
        "latency_seconds": float(run.get("latency_seconds") or 0.0),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def score_run(
    run: dict[str, Any],
    gold: dict[str, Any],
    question: dict[str, Any],
) -> dict[str, Any]:
    GoldRecord.model_validate(gold)
    PublicQuestion.model_validate(question)
    family = normalize_task_family(gold["task_level"], gold.get("task_family"))
    raw_prediction = run.get("prediction", run.get("response_text"))
    prediction = _parse_prediction(raw_prediction)
    format_error = None
    if prediction is None:
        format_error = "response is not a JSON object"
    else:
        try:
            jsonschema.validate(prediction, question["answer_schema"])
        except jsonschema.ValidationError as exc:
            format_error = exc.message
    # Score every recoverable answer field independently of schema validity or
    # termination status. Do not infer values from prose or repair malformed JSON.
    actual_answer = prediction.get("answer", {}) if prediction else {}
    if not isinstance(actual_answer, dict):
        actual_answer = {}
    leaves = _leaf_scores(gold["answer"], actual_answer, gold.get("tolerances", {}), partial=True)
    format_valid = format_error is None
    strict_leaves = _leaf_scores(gold["answer"], actual_answer, gold.get("tolerances", {}))
    joint = bool(format_valid and strict_leaves and all(strict_leaves.values()))
    raw_cited = (
        prediction.get("evidence", [])
        if prediction and isinstance(prediction.get("evidence"), list)
        else []
    )
    cited = [value for value in raw_cited if isinstance(value, str)]
    trace = run.get("tool_trace") if isinstance(run.get("tool_trace"), list) else []
    observations = _read_observations(trace)
    evidence = _evidence_metrics(gold["evidence_atoms"], trace, cited, observations)
    cost = _retrieval_cost(trace, run, gold["evidence_atoms"], observations)

    l3_local_scores: dict[str, bool] = {}
    l3_reduction_scores: dict[str, bool] = {}
    l3_local_joint_accuracy = None
    l3_local_field_accuracy = None
    l3_reduction_joint_accuracy = None
    l3_reduction_field_accuracy = None
    l3_reduction_joint_given_local_correct = None
    if gold["task_level"] == "L3":
        reduction_gold = dict(gold["answer"])
        reduction_actual = actual_answer if isinstance(actual_answer, dict) else {}
        l3_reduction_scores = _leaf_scores(
            reduction_gold,
            reduction_actual,
            gold.get("tolerances", {}),
            partial=True,
        )
        l3_reduction_joint_accuracy = float(joint)
        l3_reduction_field_accuracy = sum(l3_reduction_scores.values()) / (
            len(l3_reduction_scores) or 1
        )

    run_status = str(run.get("status", "completed"))
    if run_status in INFRASTRUCTURE_STATUSES:
        run_failure_mode = "api"
    elif run_status == "response_incomplete":
        run_failure_mode = "incomplete"
    elif run_status == "refusal":
        run_failure_mode = "refusal"
    elif run_status == "tool_budget_exhausted":
        run_failure_mode = "tool_budget"
    elif run_status == "time_budget_exhausted":
        run_failure_mode = "time_budget"
    elif run_status == "format_error" or not format_valid:
        run_failure_mode = "format"
    else:
        run_failure_mode = "none"

    error_category = classify_error(
        {
            "status": run_status,
            "format_valid": format_valid,
            "joint_accuracy": float(joint),
            "available_evidence_recall": evidence["available_evidence_recall"],
            "task_level": gold["task_level"],
            "l3_local_joint_accuracy": l3_local_joint_accuracy,
        }
    )

    triggered_f1 = None
    if (
        format_valid
        and gold["task_level"] == "L3"
        and "triggered_devices" in gold["answer"]
        and isinstance(actual_answer, dict)
    ):
        actual_devices = actual_answer.get("triggered_devices", [])
        if isinstance(actual_devices, list) and all(
            isinstance(value, str) for value in actual_devices
        ):
            triggered_f1 = _set_f1(gold["answer"]["triggered_devices"], actual_devices)

    level_metrics: dict[str, float | None] = {
        "l1_numeric_joint_accuracy": None,
        "l1_window_mean_absolute_error": None,
        "l1_window_mean_absolute_error_observed": None,
        "l1_window_max_absolute_error": None,
        "l1_window_max_absolute_error_observed": None,
        "l1_first_max_time_accuracy": None,
        "l1_hours_above_accuracy": None,
        "l2_triggered_accuracy": None,
        "l2_first_trigger_time_accuracy": None,
        "l2_trigger_hours_accuracy": None,
        "l2_trigger_hours_absolute_error": None,
        "l2_trigger_hours_absolute_error_observed": None,
        "l2_margin_accuracy": None,
        "l2_margin_absolute_error": None,
        "l2_margin_absolute_error_observed": None,
        "l3_triggered_device_exact_match": None,
        "l3_earliest_trigger_device_accuracy": None,
        "l3_trigger_offset_span_accuracy": None,
        "l3_trigger_offset_span_absolute_error": None,
        "l3_trigger_offset_span_absolute_error_observed": None,
        "l3_margin_order_accuracy": None,
        "l3_margin_span_accuracy": None,
        "l3_margin_span_absolute_error": None,
        "l3_margin_span_absolute_error_observed": None,
        "l3_local_triggered_accuracy": None,
        "l3_local_trigger_hours_mean_absolute_error": None,
        "l3_local_trigger_hours_absolute_error_observed_count": None,
        "l3_local_margin_mean_absolute_error": None,
        "l3_local_margin_absolute_error_observed_count": None,
    }
    expected_answer = gold["answer"]
    if gold["task_level"] == "L1" and family == "post_window":
        actual_mean = actual_answer.get("window_mean", _MISSING)
        actual_max = actual_answer.get("window_max", _MISSING)
        level_metrics.update(
            {
                "l1_numeric_joint_accuracy": float(
                    format_valid
                    and bool(leaves.get("window_mean"))
                    and bool(leaves.get("window_max"))
                ),
                "l1_window_mean_absolute_error": _absolute_error(
                    expected_answer.get("window_mean"), actual_mean
                ),
                "l1_window_mean_absolute_error_observed": _numeric_observed(
                    expected_answer.get("window_mean"), actual_mean, format_valid=format_valid
                ),
                "l1_window_max_absolute_error": _absolute_error(
                    expected_answer.get("window_max"), actual_max
                ),
                "l1_window_max_absolute_error_observed": _numeric_observed(
                    expected_answer.get("window_max"), actual_max, format_valid=format_valid
                ),
                "l1_first_max_time_accuracy": float(
                    format_valid and bool(leaves.get("first_max_time"))
                ),
                "l1_hours_above_accuracy": float(
                    format_valid and bool(leaves.get("hours_above_anchor_value"))
                ),
            }
        )
    elif gold["task_level"] == "L2":
        actual_trigger_hours = actual_answer.get("trigger_hours", _MISSING)
        level_metrics.update(
            {
                "l2_triggered_accuracy": float(format_valid and bool(leaves.get("triggered"))),
                "l2_first_trigger_time_accuracy": float(
                    format_valid and bool(leaves.get("first_trigger_time"))
                ),
                "l2_trigger_hours_accuracy": float(
                    format_valid and bool(leaves.get("trigger_hours"))
                ),
                "l2_trigger_hours_absolute_error": _absolute_error(
                    expected_answer.get("trigger_hours"), actual_trigger_hours
                ),
                "l2_trigger_hours_absolute_error_observed": _numeric_observed(
                    expected_answer.get("trigger_hours"),
                    actual_trigger_hours,
                    format_valid=format_valid,
                ),
            }
        )
    elif gold["task_level"] == "L3" and family == "multi_device_replay":
        actual_offset_span = actual_answer.get("trigger_offset_span_hours", _MISSING)
        level_metrics.update(
            {
                "l3_triggered_device_exact_match": float(
                    format_valid and bool(leaves.get("triggered_devices"))
                ),
                "l3_earliest_trigger_device_accuracy": float(
                    format_valid and bool(leaves.get("earliest_trigger_device"))
                ),
                "l3_trigger_offset_span_accuracy": float(
                    format_valid and bool(leaves.get("trigger_offset_span_hours"))
                ),
                "l3_trigger_offset_span_absolute_error": _absolute_error(
                    expected_answer.get("trigger_offset_span_hours"),
                    actual_offset_span,
                ),
                "l3_trigger_offset_span_absolute_error_observed": _numeric_observed(
                    expected_answer.get("trigger_offset_span_hours"),
                    actual_offset_span,
                    format_valid=format_valid,
                ),
            }
        )
    complexity = dict(gold["complexity"])
    return {
        "scoring_protocol": SCORING_PROTOCOL,
        "run_id": run.get("run_id"),
        "experiment_id": run.get("experiment_id"),
        "qa_id": gold["qa_id"],
        "pair_id": gold["pair_id"],
        "task_level": gold["task_level"],
        "task_family": family,
        "anchor_type": gold["anchor_type"],
        "model": run.get("model"),
        "reasoning_effort": run.get("reasoning_effort"),
        "resolved_models": run.get("resolved_models", []),
        "condition": run.get("condition"),
        "N": run.get("N"),
        "scope_seed": run.get("scope_seed"),
        "replicate": run.get("replicate", "main"),
        "status": run.get("status", "completed"),
        "tool_budget_reached": bool(
            run.get("tool_budget_reached", run.get("status") == "tool_budget_exhausted")
        ),
        "budget_finalization_used": bool(run.get("budget_finalization_used", False)),
        "budget_finalization_succeeded": bool(run.get("budget_finalization_succeeded", False)),
        "time_budget_reached": bool(
            run.get("time_budget_reached", run.get("status") == "time_budget_exhausted")
        ),
        "time_finalization_used": bool(run.get("time_finalization_used", False)),
        "time_finalization_succeeded": bool(run.get("time_finalization_succeeded", False)),
        "max_job_seconds": run.get("max_job_seconds"),
        "format_valid": format_valid,
        "format_error": format_error,
        "field_scores": leaves,
        "field_accuracy": sum(leaves.values()) / (len(leaves) or 1),
        "joint_accuracy": float(joint),
        "triggered_device_f1": triggered_f1,
        "l3_local_field_scores": l3_local_scores,
        "l3_reduction_field_scores": l3_reduction_scores,
        "l3_local_joint_accuracy": l3_local_joint_accuracy,
        "l3_local_field_accuracy": l3_local_field_accuracy,
        "l3_reduction_joint_accuracy": l3_reduction_joint_accuracy,
        "l3_reduction_field_accuracy": l3_reduction_field_accuracy,
        "l3_reduction_joint_given_local_correct": l3_reduction_joint_given_local_correct,
        "complexity": complexity,
        **complexity,
        **level_metrics,
        "run_failure_mode": run_failure_mode,
        "error_category": error_category,
        **evidence,
        **cost,
    }


def _aggregate(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "model",
        "condition",
        "anchor_type",
        "task_level",
        "task_family",
        "N",
        "scope_seed",
        "replicate",
        "max_job_seconds",
    )
    grouped: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        if score.get("error_category") == "infrastructure":
            continue
        grouped[tuple(score.get(key) for key in keys)].append(score)
    rows = []
    numeric_metrics = (
        "format_valid",
        "joint_accuracy",
        "field_accuracy",
        "triggered_device_f1",
        "l3_local_joint_accuracy",
        "l3_local_field_accuracy",
        "l3_reduction_joint_accuracy",
        "l3_reduction_field_accuracy",
        "l3_reduction_joint_given_local_correct",
        "l1_numeric_joint_accuracy",
        "l1_window_mean_absolute_error",
        "l1_window_mean_absolute_error_observed",
        "l1_window_max_absolute_error",
        "l1_window_max_absolute_error_observed",
        "l1_first_max_time_accuracy",
        "l1_hours_above_accuracy",
        "l2_triggered_accuracy",
        "l2_first_trigger_time_accuracy",
        "l2_trigger_hours_accuracy",
        "l2_trigger_hours_absolute_error",
        "l2_trigger_hours_absolute_error_observed",
        "l2_margin_accuracy",
        "l2_margin_absolute_error",
        "l2_margin_absolute_error_observed",
        "l3_triggered_device_exact_match",
        "l3_earliest_trigger_device_accuracy",
        "l3_trigger_offset_span_accuracy",
        "l3_trigger_offset_span_absolute_error",
        "l3_trigger_offset_span_absolute_error_observed",
        "l3_margin_order_accuracy",
        "l3_margin_span_accuracy",
        "l3_margin_span_absolute_error",
        "l3_margin_span_absolute_error_observed",
        "l3_local_triggered_accuracy",
        "l3_local_trigger_hours_mean_absolute_error",
        "l3_local_trigger_hours_absolute_error_observed_count",
        "l3_local_margin_mean_absolute_error",
        "l3_local_margin_absolute_error_observed_count",
        "retrieved_evidence_recall",
        "read_evidence_recall",
        "snippet_evidence_recall",
        "provided_evidence_recall",
        "available_evidence_recall",
        "cited_evidence_recall",
        "read_complete_evidence_rate",
        "complete_evidence_rate",
        "tool_call_count",
        "retrieval_tool_call_count",
        "discovery_call_count",
        "list_call_count",
        "search_call_count",
        "read_call_count",
        "python_call_count",
        "distinct_read_files",
        "minimum_evidence_file_count",
        "read_to_minimum_file_ratio",
        "over_read_file_ratio",
        "irrelevant_read_file_ratio",
        "read_bytes",
        "read_rows",
        "steps_to_first_retrieved",
        "steps_to_complete_retrieved",
        "steps_to_first_read",
        "steps_to_complete_read",
        "steps_to_first_snippet",
        "steps_to_complete_snippet",
        "steps_to_first_provided",
        "steps_to_complete_provided",
        "steps_to_first_evidence",
        "steps_to_complete_evidence",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "tool_budget_reached",
        "budget_finalization_used",
        "budget_finalization_succeeded",
        "time_budget_reached",
        "time_finalization_used",
        "time_finalization_succeeded",
    )
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = {name: value for name, value in zip(keys, key, strict=True)}
        row["count"] = len(group)
        for metric in numeric_metrics:
            values = [float(item[metric]) for item in group if item.get(metric) is not None]
            row[metric] = mean(values) if values else None
        budget_reached = [item for item in group if item.get("tool_budget_reached") is True]
        row["tool_budget_reached_rate"] = len(budget_reached) / len(group)
        row["budget_finalization_success_given_reached"] = (
            sum(item.get("budget_finalization_succeeded") is True for item in budget_reached)
            / len(budget_reached)
            if budget_reached
            else None
        )
        time_reached = [item for item in group if item.get("time_budget_reached") is True]
        row["time_budget_reached_rate"] = len(time_reached) / len(group)
        row["time_finalization_success_given_reached"] = (
            sum(item.get("time_finalization_succeeded") is True for item in time_reached)
            / len(time_reached)
            if time_reached
            else None
        )
        row["l3_reduction_given_local_correct_count"] = sum(
            item.get("l3_reduction_joint_given_local_correct") is not None for item in group
        )
        row["error_counts"] = dict(
            sorted(Counter(item["error_category"] for item in group).items())
        )
        row["run_failure_mode_counts"] = dict(
            sorted(Counter(item["run_failure_mode"] for item in group).items())
        )
        rows.append(row)
    return rows


def _select_latest_attempts(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse append-only retry history without hiding terminal duplicates.

    Runner may append another attempt only after an infrastructure/API failure.
    Therefore every non-latest row for a run must have a retryable status. The
    selected rows are ordered by the file position of their latest attempt.
    """

    grouped: defaultdict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, run in enumerate(runs):
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Result attempt at line {index + 1} has no valid run_id")
        grouped[run_id].append((index, run))

    selected: list[tuple[int, dict[str, Any]]] = []
    valid_statuses = RETRYABLE_ATTEMPT_STATUSES | TERMINAL_ATTEMPT_STATUSES
    for run_id, attempts in grouped.items():
        reference = attempts[0][1]
        for index, attempt in attempts[1:]:
            mismatches = [
                field
                for field in ATTEMPT_IDENTITY_FIELDS
                if attempt.get(field) != reference.get(field)
            ]
            if mismatches:
                raise ValueError(
                    f"Attempts for run_id {run_id} disagree at line {index + 1} on: "
                    f"{sorted(mismatches)}"
                )

        for index, attempt in attempts[:-1]:
            status = attempt.get("status")
            if status in TERMINAL_ATTEMPT_STATUSES:
                raise ValueError(
                    f"Duplicate terminal attempt for run_id {run_id}: line {index + 1} "
                    f"already has terminal status {status!r}"
                )
            if status not in RETRYABLE_ATTEMPT_STATUSES:
                raise ValueError(
                    f"Invalid retry history for run_id {run_id}: line {index + 1} has "
                    f"non-retryable status {status!r}"
                )

        latest_index, latest = attempts[-1]
        latest_status = latest.get("status")
        if latest_status not in valid_statuses:
            raise ValueError(
                f"Latest attempt for run_id {run_id} has unsupported status "
                f"{latest_status!r} at line {latest_index + 1}"
            )
        selected.append(
            (
                latest_index,
                {
                    "run": latest,
                    "attempt_count": len(attempts),
                    "prior_attempt_statuses": [
                        str(attempt.get("status")) for _, attempt in attempts[:-1]
                    ],
                },
            )
        )
    selected.sort(key=lambda item: item[0])
    return [record for _, record in selected]


def _validate_manifest_binding(
    *,
    runs: list[dict[str, Any]],
    manifest_path: Path,
    questions_path: Path,
    gold_path: Path,
    snapshots: _SourceSnapshots,
) -> dict[str, Any]:
    """Bind result attempts and evaluator inputs to one immutable run manifest."""

    binding = load_manifest_binding(
        manifest_path,
        artifact_paths={"questions": questions_path, "gold": gold_path},
        required_path_identities=("manifest", "questions", "gold"),
        json_object_loader=lambda path: snapshots.load_json_object(
            path, label="run manifest binding summary"
        ),
        jsonl_loader=snapshots.read_jsonl,
        digest_loader=snapshots.digest,
        require_valid_experiment_id=True,
        index_jobs=True,
    )
    validate_attempt_identities(runs, binding)
    return binding.provenance({"questions": questions_path, "gold": gold_path})


def evaluate_runs(
    *,
    results_path: str | Path,
    gold_path: str | Path,
    questions_path: str | Path,
    manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    results_source = Path(results_path).resolve()
    gold_source = Path(gold_path).resolve()
    questions_source = Path(questions_path).resolve()
    manifest_source = Path(manifest_path).resolve()
    snapshots = _SourceSnapshots()
    runs = snapshots.read_jsonl(results_source)
    gold_rows = expand_gold_records(snapshots.read_jsonl(gold_source))
    question_rows = snapshots.read_jsonl(questions_source)
    results_sha256 = snapshots.digest(results_source)
    provenance = _validate_manifest_binding(
        runs=runs,
        manifest_path=manifest_source,
        questions_path=questions_source,
        gold_path=gold_source,
        snapshots=snapshots,
    )
    selected_attempts = _select_latest_attempts(runs)

    def unique_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        values: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows, start=1):
            qa_id = row.get("qa_id")
            if not isinstance(qa_id, str) or not qa_id:
                raise ValueError(f"{label} row {index} has no valid qa_id")
            if qa_id in values:
                raise ValueError(f"Duplicate {label} qa_id: {qa_id}")
            values[qa_id] = row
        return values

    gold_by_id = unique_by_id(gold_rows, "Gold")
    questions_by_id = unique_by_id(question_rows, "question")
    if set(gold_by_id) != set(questions_by_id):
        raise ValueError("Gold and question qa_id sets differ")
    for qa_id, gold in gold_by_id.items():
        question = questions_by_id[qa_id]
        GoldRecord.model_validate(gold)
        PublicQuestion.model_validate(question)
        if gold.get("schema_version") != question.get("schema_version"):
            raise ValueError(f"Gold/question {qa_id} disagree on schema_version")
        family = normalize_task_family(question["task_level"], question.get("task_family"))
        expected_schema = answer_schema(question["task_level"], family)
        if question.get("answer_schema") != expected_schema:
            raise ValueError(f"Question {qa_id} answer schema disagrees with its task level")
        jsonschema.Draft202012Validator.check_schema(expected_schema)
        try:
            jsonschema.validate({"answer": gold["answer"], "evidence": []}, expected_schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(
                f"Gold answer violates question schema for {qa_id}: {exc.message}"
            ) from exc
        for field in ("pair_id", "task_level", "task_family"):
            if gold.get(field) != question.get(field):
                raise ValueError(f"Gold/question {qa_id} disagree on {field}")
    scores = []
    for selected in selected_attempts:
        run = selected["run"]
        qa_id = run.get("qa_id")
        if qa_id not in gold_by_id or qa_id not in questions_by_id:
            raise KeyError(f"Run references unknown qa_id: {qa_id}")
        for field in ("pair_id", "task_level", "anchor_type"):
            if run.get(field) != gold_by_id[qa_id].get(field):
                raise ValueError(f"Run {run.get('run_id')} disagrees with {qa_id} on {field}")
        score = score_run(run, gold_by_id[qa_id], questions_by_id[qa_id])
        score["attempt_count"] = selected["attempt_count"]
        score["prior_attempt_statuses"] = selected["prior_attempt_statuses"]
        scores.append(score)
    aggregates = _aggregate(scores)
    changed_inputs = snapshots.changed_sources()
    if changed_inputs:
        raise ValueError(f"Evaluation inputs changed while scoring: {changed_inputs}")
    output = Path(output_root)
    scores_path = output / "per_run_scores.jsonl"
    write_jsonl(scores_path, scores)
    write_json(output / "aggregate_scores.json", aggregates)
    summary = {
        "scoring_protocol": SCORING_PROTOCOL,
        "scorer_sha256": file_sha256(Path(__file__)),
        **provenance,
        "results_path": str(results_source),
        "results_sha256": results_sha256,
        "per_run_scores_path": str(scores_path.resolve()),
        "per_run_scores_sha256": file_sha256(scores_path),
        # Compatibility alias: historical run_count counted physical JSONL rows.
        "run_count": len(runs),
        "raw_attempt_count": len(runs),
        "unique_run_count": len(selected_attempts),
        "retried_run_count": sum(
            int(selected["attempt_count"] > 1) for selected in selected_attempts
        ),
        "superseded_attempt_count": len(runs) - len(selected_attempts),
        "scored_count": len(scores),
        "infrastructure_error_count": sum(
            score["error_category"] == "infrastructure" for score in scores
        ),
        "aggregate_eligible_count": sum(
            score["error_category"] != "infrastructure" for score in scores
        ),
        "aggregate_group_count": len(aggregates),
        "primary_metric": "joint_accuracy",
    }
    write_json(output / "evaluation_summary.json", summary)
    return summary