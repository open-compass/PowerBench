
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..models import (
    CorpusInventory,
    QuestionGenerationRecord,
    QuestionTextParts,
)
from ..task_specs import TASK_LEVELS, answer_schema, normalize_task_family, question_parts
from ..util import canonical_json
from .metric_eligibility import UNIT_ELIGIBILITY_CLAUSE

STYLE_BRIEFS: tuple[tuple[str, str], ...] = (
    ("operations_query", "Write a concise operations question; do not describe the solution workflow."),
    (
        "analysis_request",
        "Write a natural data-analysis request that states only the subject, conditions, and required results.",
    ),
    (
        "result_focused",
        "Center the question on the business conclusion to be answered; avoid manual-style phrasing.",
    ),
    (
        "incident_review",
        "Write a natural post-incident review question without hinting at data files or tools.",
    ),
    (
        "comparison_query",
        "Connect business constraints and the final comparison goal naturally without decomposing intermediate steps.",
    ),
)

_FORBIDDEN_GENERATION_PHRASES = (
    "fault injection succeeded",
    "injection modified field",
    "discovery-time strategy",
    "near_miss",
    "per_device annotation",
)
_CONTEXT_KEYS = (
    "Device name",
    "Model / type",
    "Equipment level",
    "Device type",
    "Project",
    "Station / line",
    "Installation location",
    "Phase",
    "Primary/standby",
    "Cooling",
    "Communication interface",
    "Uplink interface",
)


def _style(pair_id: str) -> tuple[str, str]:
    match = re.search(r"(\d+)$", pair_id)
    index = (
        int(match.group(1)) - 1 if match else int(hashlib.sha256(pair_id.encode()).hexdigest(), 16)
    )
    return STYLE_BRIEFS[index % len(STYLE_BRIEFS)]


def _device_facts(task: Mapping[str, Any], inventory: CorpusInventory) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for alias, uid in task["device_aliases"].items():
        device = inventory.device(uid)
        result.append(
            {
                "alias": alias,
                "uid": uid,
                "name": device.name,
                "model_key": device.model_key,
                "context": {
                    key: device.metadata[key] for key in _CONTEXT_KEYS if device.metadata.get(key)
                },
                "report_path": task["report_paths"][alias],
                "indirect_locator": list(task["locators"][alias]["fields"]),
            }
        )
    return result


def build_question_facts(task: Mapping[str, Any], inventory: CorpusInventory) -> dict[str, Any]:
    """Create the answer-free source of truth supplied to the polisher."""

    level = str(task["task_level"])
    family = normalize_task_family(level, task.get("task_family"))
    schema = answer_schema(level, family)["properties"]["answer"]
    facts: dict[str, Any] = {
        "pair_id": task["pair_id"],
        "task_level": level,
        "task_family": family,
        "devices": _device_facts(task, inventory),
        "answer_contract": {
            "required_fields": list(schema["required"]),
            "evidence_required": True,
        },
    }
    if family == "document_routed_window":
        facts.update(
            {
                "report_selector": {
                    "kind": "inspection report",
                    "inspection_start_date": str(task["anchor"]["inspection_start"]).split(
                        " ", 1
                    )[0],
                },
                "routing_policy": {
                    "candidate_scope": "numeric raw metrics from the inspection-end key-metrics excerpt",
                    "catalog_requirement": "metric catalog provides both normal mean and standard deviation",
                    "selection": "maximize |inspection-end reading − normal mean| / standard deviation",
                },
                "computation": {
                    "window_hours": task["window"]["hours"],
                    "window_semantics": task["window"]["semantics"],
                    "operations": [
                        "return selected metric",
                        "standardized deviation at inspection end",
                        "post-window mean",
                        "post-window maximum and first occurrence time",
                    ],
                    "round_digits": int(task.get("answer_round_digits", 4)),
                },
            }
        )
    elif family in {"post_window", "threshold_episode", "metadata_normalized"}:
        facts.update(
            {
                "report_selector": {
                    "kind": "inspection report",
                    "inspection_start_date": str(task["anchor"]["inspection_start"]).split(" ", 1)[
                        0
                    ],
                    "inspection_end_source": "inspection report",
                },
                "computation": {
                    "metric_name": task["metric"]["name"],
                    "unit": task["metric"]["unit"],
                    "baseline": "metric value at inspection end",
                    "window_hours": task["window"]["hours"],
                    "window_semantics": task["window"]["semantics"],
                    "window_boundary": (
                        "target window excludes inspection end and uses consecutive hourly samples after it"
                    ),
                    "operations": [
                        *(
                            [
                                "total exceedance hours",
                                "longest consecutive exceedance run",
                                "first exceedance time",
                            ]
                            if family == "threshold_episode"
                            else [
                                "mean",
                                "maximum",
                                "first maximum time",
                                "hours strictly above inspection-end value",
                            ]
                        ),
                    ],
                    "round_digits": int(task.get("answer_round_digits", 4)),
                },
            }
        )
        if family == "metadata_normalized":
            facts["metadata_parameter"] = {
                "field": "Uplink interface",
                "interface": task["capacity"]["interface"],
                "capacity_mbps": task["capacity"]["value_mbps"],
                "utilization_threshold_percent": task["utilization_threshold_percent"],
            }
    elif family == "pre_post_comparison":
        facts.update(
            {
                "report_selector": {
                    "kind": "fault report",
                    "discovery_date": str(
                        task["anchor"].get("discovery_time") or task["anchor"]["fault_end"]
                    ).split(" ", 1)[0],
                    "report_number": task["anchor"].get("report_number"),
                    "fault_end_source": "fault report",
                },
                "computation": {
                    "metric_name": task["metric"]["name"],
                    "unit": task["metric"]["unit"],
                    "window_hours": task["window"]["hours"],
                    "window_semantics": task["window"]["semantics"],
                    "window_boundary": "neither pre- nor post-window includes fault-window end",
                    "operations": [
                        "pre-window mean",
                        "post-window mean",
                        "change rate",
                        "post-window maximum and first occurrence time",
                    ],
                    "round_digits": int(task.get("answer_round_digits", 4)),
                },
            }
        )
    elif family == "context_conditioned_replay":
        facts.update(
            {
                "report_selector": {
                    "kind": "fault report",
                    "per_device": {
                        alias: {
                            "discovery_date": str(
                                values.get("discovery_time") or values["fault_end"]
                            ).split(" ", 1)[0],
                            "report_number": values.get("report_number"),
                        }
                        for alias, values in task["anchors"].items()
                    },
                },
                "context_policy": task["context_policy"],
                "computation": {
                    "window_hours": task["window"]["hours"],
                    "window_offset_hours": int(task["window"].get("offset_hours", 24)),
                    "window_semantics": task["window"]["semantics"],
                    "continuity_at_window_start": (
                        "evaluation window boundary does not reset consecutive state accumulated before it"
                    ),
                    "outputs": [
                        "load category",
                        "effective threshold",
                        "trigger result",
                        "first trigger time",
                        "total trigger hours",
                    ],
                },
            }
        )
    elif family in {
        "threshold_replay",
        "duration_replay",
        "shifted_window_replay",
        "metadata_bound_replay",
        "multi_device_replay",
    }:
        facts.update(
            {
                "report_selector": {
                    "kind": "fault report",
                    "per_device": {
                        alias: {
                            "discovery_date": str(
                                values.get("discovery_time") or values["fault_end"]
                            ).split(" ", 1)[0],
                            "report_number": values.get("report_number"),
                            "abnormal_object": task["rule"]["metric_name"],
                        }
                        for alias, values in task["anchors"].items()
                    },
                },
                "rule": {
                    "rule_id": task["rule"]["rule_id"],
                    "metric_name": task["rule"]["metric_name"],
                    "unit": task["rule"]["unit"],
                    "operator": task["rule"]["operator"],
                    "consecutive_hours": task["rule"]["consecutive_hours"],
                    "counterfactual_threshold": task["rule"]["counterfactual_threshold_text"],
                    **(
                        {
                            "counterfactual_consecutive_hours": task["rule"][
                                "counterfactual_consecutive_hours"
                            ]
                        }
                        if family == "duration_replay"
                        else {}
                    ),
                },
                "computation": {
                    "window_hours": task["window"]["hours"],
                    "window_offset_hours": int(task["window"].get("offset_hours", 24)),
                    "window_semantics": task["window"]["semantics"],
                    "strict_comparison": True,
                    "requires_lookback": True,
                    "continuity_at_window_start": (
                        "evaluation window boundary does not reset consecutive state accumulated before it"
                    ),
                },
            }
        )
        if family == "metadata_bound_replay":
            facts["metadata_parameter"] = {
                "field": "Uplink interface",
                "capacity_mapping": "GE=1000 Mbps; 10GE/XGSPON=10000 Mbps",
            }
        if family == "multi_device_replay":
            facts["cross_device_reductions"] = [
                "triggered device set",
                "earliest first-trigger device (by relative offset)",
                "span of first-trigger offsets across triggered devices",
            ]
    elif family == "heterogeneous_metric_comparison":
        facts.update(
            {
                "report_selector": {
                    "kind": "inspection report",
                    "per_device": {
                        alias: {"inspection_start": values["inspection_start"]}
                        for alias, values in task["anchors"].items()
                    },
                },
                "routing_policy": {
                    "candidate_scope": "numeric raw metrics from each report's inspection-end excerpt",
                    "catalog_requirement": "each metric catalog provides both normal mean and standard deviation",
                    "selection": "per device, maximize |inspection-end reading − normal mean| / standard deviation",
                },
                "computation": {
                    "window_hours": task["window"]["hours"],
                    "window_semantics": task["window"]["semantics"],
                    "local_operation": (
                        "post-window maximum absolute standardized deviation for selected metric and first time"
                    ),
                    "group_operation": "device with largest deviation and span between highest and lowest",
                    "round_digits": int(task.get("answer_round_digits", 4)),
                },
            }
        )
    elif family in {"aligned_window_comparison", "filter_then_compute"}:
        facts.update(
            {
                "report_selector": {
                    "kind": "inspection report",
                    "per_device": {
                        alias: {
                            "inspection_start": values["inspection_start"],
                            "inspection_end": values["inspection_end"],
                        }
                        for alias, values in task["anchors"].items()
                    },
                },
                "computation": {
                    "metric_name": task["metric"]["name"],
                    "unit": task["metric"]["unit"],
                    "window_hours": task["window"]["hours"],
                    "window_semantics": task["window"]["semantics"],
                    "window_boundary": (
                        "target window excludes inspection end and uses consecutive hourly samples after it"
                    ),
                    "filter": task.get("filter"),
                    "round_digits": int(task.get("answer_round_digits", 4)),
                },
            }
        )
    else:  # pragma: no cover - family registry validation rejects this first.
        raise ValueError(f"Unsupported task family: {family!r}")
    return facts


def build_generation_record(
    task: Mapping[str, Any],
    inventory: CorpusInventory,
    *,
    schema_version: str,
) -> dict[str, Any]:
    facts = build_question_facts(task, inventory)
    deterministic = question_parts(dict(task))
    draft = QuestionTextParts(question=deterministic.question)
    style_id, _ = _style(str(task["pair_id"]))
    record = QuestionGenerationRecord(
        schema_version=schema_version,
        pair_id=str(task["pair_id"]),
        task_level=task["task_level"],
        task_family=task.get("task_family"),
        style_id=style_id,
        facts=facts,
        facts_sha256=hashlib.sha256(canonical_json(facts).encode("utf-8")).hexdigest(),
        draft=draft,
        final=draft,
        status="draft",
        provenance=None,
        validation={},
    ).model_dump(mode="json")
    report = validate_generation_record(record, task, inventory)
    if not report["valid"]:
        raise ValueError(
            f"Deterministic question draft failed for {task['pair_id']}: {report['errors']}"
        )
    record["validation"] = report
    return QuestionGenerationRecord.model_validate(record).model_dump(mode="json")


def question_texts(record: Mapping[str, Any]) -> str:
    parsed = QuestionGenerationRecord.model_validate(record)
    return parsed.final.question.strip()


def _contains_all(text: str, values: Iterable[object]) -> list[str]:
    return [str(value) for value in values if str(value) and str(value) not in text]


def _flatten_scalars(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in _flatten_scalars(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _flatten_scalars(nested)]
    return [value]


def _answer_leaks(task: Mapping[str, Any], facts: Mapping[str, Any], text: str) -> list[str]:
    input_literals = {str(value) for value in _flatten_scalars(facts) if value is not None}
    leaks: list[str] = []
    for value in _flatten_scalars(task.get("answer", {})):
        if value is None or isinstance(value, bool):
            continue
        literal = str(value)
        if literal in input_literals or len(literal) < 4:
            continue
        if literal in text:
            leaks.append(literal)
    return sorted(set(leaks))


def _linguistic_signature(text: str, facts: Mapping[str, Any]) -> str:
    """Mask task-specific values so repeated surface templates remain visible."""

    masked = text.casefold()
    literals = sorted(
        {
            str(value).casefold()
            for value in _flatten_scalars(facts)
            if value is not None and len(str(value)) >= 2
        },
        key=len,
        reverse=True,
    )
    for literal in literals:
        masked = masked.replace(literal, "<fact>")
    masked = re.sub(r"\d+(?:\.\d+)?", "<n>", masked)
    return re.sub(r"[^a-z<>]+", "", masked)


def validate_generation_record(
    raw_record: Mapping[str, Any],
    task: Mapping[str, Any],
    inventory: CorpusInventory,
) -> dict[str, Any]:
    """Check provenance, facts, pair isolation, required semantics and leakage."""

    errors: list[str] = []
    try:
        record = QuestionGenerationRecord.model_validate(raw_record)
    except ValueError as exc:
        return {"valid": False, "errors": [f"record contract: {exc}"]}

    expected_facts = build_question_facts(task, inventory)
    expected_hash = hashlib.sha256(canonical_json(expected_facts).encode("utf-8")).hexdigest()
    family = normalize_task_family(str(task.get("task_level")), task.get("task_family"))
    if (
        record.pair_id != task.get("pair_id")
        or record.task_level != task.get("task_level")
        or normalize_task_family(record.task_level, record.task_family) != family
    ):
        errors.append("record identity disagrees with semantic task")
    if record.style_id not in dict(STYLE_BRIEFS):
        errors.append(f"unknown language style: {record.style_id}")
    if record.facts != expected_facts or record.facts_sha256 != expected_hash:
        errors.append("question facts disagree with the deterministic semantic task")

    expected_draft = question_parts(dict(task))
    if record.draft.model_dump() != {"question": expected_draft.question}:
        errors.append("deterministic draft was changed")
    if (record.status == "draft") != (record.provenance is None):
        errors.append("polish status and provenance are inconsistent")

    full_text = record.final.question
    devices = expected_facts["devices"]
    requirements: list[object] = [
        item["value"] for device in devices for item in device["indirect_locator"]
    ]
    if family == "document_routed_window":
        requirements.extend(
            [
                expected_facts["report_selector"]["inspection_start_date"],
                str(task["window"]["hours"]),
            ]
        )
        concept_groups = [
            ("inspection end", "inspection-end"),
            ("normal mean",),
            ("standard deviation", "std"),
            ("deviation",),
            ("metric",),
            ("mean", "average"),
            ("max", "maximum"),
            ("first",),
            ("excluding", "exclude", "not include"),
            ("hourly", "sample"),
            ("round", "decimal"),
        ]
    elif family in {"post_window", "threshold_episode", "metadata_normalized"}:
        requirements.extend(
            [
                expected_facts["report_selector"]["inspection_start_date"],
                task["metric"]["name"],
                str(task["window"]["hours"]),
            ]
        )
        concept_groups: list[tuple[str, ...]] = (
            [("total", "cumulative"), ("longest", "consecutive"), ("first",), ("above", "exceed")]
            if family == "threshold_episode"
            else [("mean", "average"), ("longest",), ("first",), ("utilization",)]
            if family == "metadata_normalized"
            else [("mean", "average"), ("max", "maximum"), ("first",), ("above", "exceed")]
        )
        concept_groups.extend(
            [
                ("inspection end", "inspection-end"),
                ("hour",),
                ("excluding", "exclude", "not include"),
                ("hourly", "sample"),
            ]
        )
        if family != "threshold_episode":
            concept_groups.append(("round", "decimal"))
    elif family == "pre_post_comparison":
        requirements.extend(
            [
                expected_facts["report_selector"]["discovery_date"],
                task["metric"]["name"],
                str(task["window"]["hours"]),
            ]
        )
        if expected_facts["report_selector"].get("report_number"):
            requirements.append(expected_facts["report_selector"]["report_number"])
        concept_groups = [
            ("before", "pre"),
            ("after", "post"),
            ("mean", "average"),
            ("change rate", "rate of change"),
            ("max", "maximum"),
            ("excluding", "exclude", "not include"),
            ("hourly", "sample"),
            ("round", "decimal"),
        ]
    elif family == "context_conditioned_replay":
        requirements.extend(
            item["discovery_date"]
            for item in expected_facts["report_selector"]["per_device"].values()
        )
        requirements.extend(
            item["report_number"]
            for item in expected_facts["report_selector"]["per_device"].values()
            if item.get("report_number")
        )
        requirements.extend(
            [
                str(task["window"]["hours"]),
                str(task["window"].get("offset_hours", 24)),
                *(f"{value:g}" for value in task["context_policy"]["factors"].values()),
            ]
        )
        concept_groups = [
            ("load level", "load category"),
            ("high load", "high-load"),
            ("low load", "low-load"),
            ("regular load", "normal load"),
            ("threshold",),
            ("trigger",),
            ("first",),
            ("total", "cumulative"),
            ("does not reset", "not reset", "carry over", "continue"),
        ]
    elif family in {
        "threshold_replay",
        "duration_replay",
        "shifted_window_replay",
        "metadata_bound_replay",
        "multi_device_replay",
    }:
        requirements.extend(
            item["discovery_date"]
            for item in expected_facts["report_selector"]["per_device"].values()
        )
        requirements.extend(
            item["report_number"]
            for item in expected_facts["report_selector"]["per_device"].values()
            if item.get("report_number")
        )
        requirements.extend(
            [
                task["rule"]["metric_name"],
                str(task["window"]["hours"]),
                str(task["window"].get("offset_hours", 24)),
            ]
        )
        if family == "metadata_bound_replay":
            requirements.append(f"{float(task['rule']['counterfactual_threshold']) * 100:g}")
        elif family != "duration_replay":
            requirements.append(task["rule"]["counterfactual_threshold_text"])
        if family not in {"duration_replay", "metadata_bound_replay"} and task["rule"]["unit"]:
            requirements.append(task["rule"]["unit"])
        if family == "multi_device_replay":
            requirements.extend(task["device_aliases"])
            concept_groups = [
                ("threshold",),
                ("unchanged", "remain", "keep"),
                ("trigger",),
                ("earliest", "first"),
                ("offset", "relative"),
                ("span", "difference", "gap"),
                ("does not reset", "not reset", "carry over", "continue"),
            ]
        elif family == "duration_replay":
            requirements.append(str(task["rule"]["counterfactual_consecutive_hours"]))
            concept_groups = [
                ("consecutive", "duration"),
                ("unchanged", "remain", "keep"),
                ("trigger",),
                ("first",),
                ("total", "cumulative"),
                ("does not reset", "not reset", "carry over", "continue"),
            ]
        else:
            concept_groups = [
                ("threshold",),
                ("unchanged", "remain", "keep"),
                ("trigger",),
                ("first",),
                ("total", "cumulative"),
                ("does not reset", "not reset", "carry over", "continue"),
            ]
    elif family == "heterogeneous_metric_comparison":
        requirements.extend(task["device_aliases"])
        requirements.extend(
            [
                str(values["inspection_start"]).split(" ", 1)[0]
                for values in task["anchors"].values()
            ]
        )
        requirements.append(str(task["window"]["hours"]))
        concept_groups = [
            ("each", "per device", "respectively"),
            ("normal mean",),
            ("standard deviation", "std"),
            ("deviation",),
            ("max", "maximum"),
            ("first",),
            ("highest", "largest"),
            ("span", "difference", "gap"),
            ("excluding", "exclude", "not include"),
            ("hourly", "sample"),
            ("round", "decimal"),
        ]
    else:
        requirements.extend(task["device_aliases"])
        requirements.extend([task["metric"]["name"], str(task["window"]["hours"])])
        if family == "filter_then_compute":
            requirements.extend([task["filter"]["field"], task["filter"]["value"]])
            concept_groups = [
                ("filter", "select"),
                ("mean", "average"),
                ("highest", "maximum"),
                ("span", "difference", "gap"),
                ("excluding", "exclude", "not include"),
                ("hourly", "sample"),
                ("round", "decimal"),
            ]
        else:
            concept_groups = [
                ("baseline", "reference"),
                ("increase", "change"),
                ("max", "maximum", "largest"),
                ("median",),
                ("excluding", "exclude", "not include"),
                ("hourly", "sample"),
                ("round", "decimal"),
            ]

    missing = _contains_all(full_text, requirements)
    if missing:
        errors.append(f"Question omitted protected semantics: {sorted(set(missing))}")
    for alternatives in concept_groups:
        if not any(value in full_text for value in alternatives):
            errors.append(f"Question omitted semantic concept: {'/'.join(alternatives)}")

    hidden_execution_values = [
        item for device in devices for item in (device["uid"], device["report_path"])
    ]
    if family in {
        "context_conditioned_replay",
        "threshold_replay",
        "duration_replay",
        "shifted_window_replay",
        "metadata_bound_replay",
        "multi_device_replay",
    }:
        hidden_execution_values.append(task["rule"]["rule_id"])
        hidden_execution_values.extend(
            str(value)
            for values in task["anchors"].values()
            for value in (values.get("discovery_time"), values.get("fault_end"))
            if value
        )
    elif family == "pre_post_comparison":
        hidden_execution_values.extend(
            str(value)
            for value in (task["anchor"].get("discovery_time"), task["anchor"].get("fault_end"))
            if value
        )
    exposed = sorted({value for value in hidden_execution_values if value in full_text})
    if exposed:
        errors.append(f"Question exposed hidden execution facts: {exposed}")

    if record.status == "polished":
        original = record.draft.question
        rewritten = record.final.question
        # This one deterministic contract sentence is added after LLM
        # polishing and therefore must not count as model verbosity.
        rewritten_for_length = rewritten.replace(
            f" {UNIT_ELIGIBILITY_CLAUSE}", ""
        ).replace(UNIT_ELIGIBILITY_CLAUSE, "")
        limit = max(len(original) + 30, int(len(original) * 1.25))
        if len(rewritten_for_length) > limit:
            errors.append(
                "Polished question is too verbose: "
                f"{len(rewritten_for_length)} characters, limit {limit}"
            )

    forbidden = [phrase for phrase in _FORBIDDEN_GENERATION_PHRASES if phrase in full_text]
    if forbidden:
        errors.append(f"Question contains generation-process language: {forbidden}")
    answer_leaks = _answer_leaks(task, expected_facts, full_text)
    if answer_leaks:
        errors.append(f"Question contains answer-only literals: {answer_leaks}")
    return {
        "valid": not errors,
        "errors": errors,
        "autonomous_single_question": True,
        "answer_supplied_to_polisher": False,
        "forbidden_phrase_count": len(forbidden),
        "answer_literal_leak_count": len(answer_leaks),
    }


def validate_generation_records(
    records: Iterable[Mapping[str, Any]],
    tasks: Iterable[Mapping[str, Any]],
    inventory: CorpusInventory,
    *,
    require_polished: bool = False,
) -> dict[str, Any]:
    record_rows = list(records)
    task_rows = list(tasks)
    by_pair = {str(task["pair_id"]): task for task in task_rows}
    errors: list[str] = []
    if len(by_pair) != len(task_rows):
        errors.append("duplicate semantic task pair_id")
    seen: set[str] = set()
    status_counts: dict[str, int] = {"draft": 0, "polished": 0}
    style_counts: dict[str, dict[str, int]] = {
        level: {style_id: 0 for style_id, _ in STYLE_BRIEFS} for level in TASK_LEVELS
    }
    signatures: dict[str, dict[str, list[str]]] = {level: {} for level in TASK_LEVELS}
    for raw in record_rows:
        pair_id = str(raw.get("pair_id", ""))
        if pair_id in seen:
            errors.append(f"duplicate question-generation pair_id: {pair_id}")
            continue
        seen.add(pair_id)
        task = by_pair.get(pair_id)
        if task is None:
            errors.append(f"question generation references unknown pair_id: {pair_id}")
            continue
        report = validate_generation_record(raw, task, inventory)
        errors.extend(f"{pair_id}: {error}" for error in report["errors"])
        try:
            parsed = QuestionGenerationRecord.model_validate(raw)
        except ValueError:
            continue
        status_counts[parsed.status] += 1
        style_counts[parsed.task_level][parsed.style_id] = (
            style_counts[parsed.task_level].get(parsed.style_id, 0) + 1
        )
        signature = _linguistic_signature(parsed.final.question, parsed.facts)
        signatures[parsed.task_level].setdefault(signature, []).append(parsed.pair_id)
        if require_polished and parsed.final == parsed.draft:
            errors.append(f"{pair_id}: polished wording is identical to its deterministic draft")
        if require_polished and parsed.status != "polished":
            errors.append(f"{pair_id}: final benchmark requires polished wording")
    missing = sorted(set(by_pair).difference(seen))
    if missing:
        errors.append(f"missing question-generation records: {missing[:10]}")
    duplicate_surface_groups = {
        level: [pairs for pairs in groups.values() if len(pairs) > 1]
        for level, groups in signatures.items()
    }
    if require_polished:
        for level, groups in duplicate_surface_groups.items():
            for pairs in groups:
                if len(pairs) > 2:
                    errors.append(
                        f"{level}: repeated normalized shared wording across {len(pairs)} pairs: "
                        f"{pairs[:5]}"
                    )
    return {
        "valid": not errors,
        "errors": errors,
        "record_count": len(record_rows),
        "status_counts": status_counts,
        "style_counts": style_counts,
        "duplicate_surface_groups": duplicate_surface_groups,
    }


def polished_public_texts(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        parsed = QuestionGenerationRecord.model_validate(record)
        if parsed.pair_id in result:
            raise ValueError(f"Duplicate question-generation pair_id: {parsed.pair_id}")
        result[parsed.pair_id] = question_texts(parsed.model_dump(mode="json"))
    return result


__all__ = [
    "STYLE_BRIEFS",
    "build_generation_record",
    "build_question_facts",
    "polished_public_texts",
    "question_texts",
    "validate_generation_record",
    "validate_generation_records",
]