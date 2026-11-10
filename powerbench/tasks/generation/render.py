from __future__ import annotations

import itertools
import json
from typing import Any

from ..models import (
    Complexity,
    CorpusInventory,
    EvidenceAtom,
    EvidenceSource,
    GoldRecord,
    PublicQuestion,
)
from ..task_specs import public_answer, task_spec
from ..util import safe_join
from .gold import execute_gold, trace_complexity


def _source(path: str, **selector: Any) -> EvidenceSource:
    return EvidenceSource(path=path, selector=selector)


def _derived_metric_record(
    inventory: CorpusInventory, relative_path: str, derived_key: str
) -> dict[str, Any]:
    path = safe_join(inventory.raw_root, relative_path)
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    active = payload.get("derived_metrics") if isinstance(payload, dict) else None
    matches = (
        [
            item
            for item in active
            if isinstance(item, dict) and str(item.get("derived_key")) == derived_key
        ]
        if isinstance(active, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one active derived metric {derived_key!r} in {relative_path}"
        )
    record = matches[0]
    atomic_metrics = record.get("atomic_metrics")
    atomic_fields = (
        sorted(
            {
                str(item.get("db_field"))
                for item in atomic_metrics
                if isinstance(item, dict) and str(item.get("db_field") or "")
            }
        )
        if isinstance(atomic_metrics, list)
        else []
    )
    required = {
        "derived_name": str(record.get("derived_name") or ""),
        "formula": str(record.get("formula") or ""),
        "output_unit": str(record.get("output_unit") or ""),
    }
    if not all(required.values()) or not atomic_fields:
        raise ValueError(f"Incomplete active derived metric {derived_key!r} in {relative_path}")
    return {
        "kind": "derived_metric",
        "derived_key": derived_key,
        **required,
        "atomic_fields": atomic_fields,
    }


def _archive_sources(
    inventory: CorpusInventory, uid: str, locator: dict[str, Any]
) -> list[EvidenceSource]:
    device = inventory.device(uid)
    # Bind every locator key to its value inside the same archive section. This
    # prevents unrelated mentions elsewhere in a long document from satisfying
    # an indirect locator atom.
    key_values = {
        str(item["field"]): {"equals": str(item["value"])} for item in locator.get("fields", [])
    }
    return [
        _source(
            device.archive_path,
            section="Basic Information",
            key_values=key_values,
        )
    ]


def _routing_atoms(
    *,
    prefix: str,
    report: str,
    selection: dict[str, Any],
) -> list[EvidenceAtom]:
    atoms: list[EvidenceAtom] = []
    for index, candidate in enumerate(selection["candidate_metrics"], start=1):
        atom_prefix = f"{prefix}_candidate_{index}" if prefix else f"candidate_{index}"
        atoms.extend(
            [
                EvidenceAtom(
                    atom_id=f"{atom_prefix}_report_value",
                    fact=f"Inspection-end value for candidate metric {candidate['name']}",
                    modality="document",
                    acceptable_sources=[
                        _source(
                            report,
                            section="Key metrics at inspection end",
                            table_row={
                                "match": {
                                    "Metric": {"equals": candidate["name"]},
                                    "Unit": {"equals": candidate["unit"]},
                                },
                                "numeric": {
                                    "column": "Value",
                                    "value": candidate["report_value"],
                                    "abs_tolerance": 0.000001,
                                },
                            },
                        )
                    ],
                ),
                EvidenceAtom(
                    atom_id=f"{atom_prefix}_normal_reference",
                    fact=f"Normal mean and standard deviation for {candidate['name']}",
                    modality="metadata",
                    acceptable_sources=[
                        _source(
                            candidate["metric_catalog_path"],
                            field=candidate["field"],
                            json_record={
                                "kind": "metric",
                                "field": candidate["field"],
                                "name": candidate["name"],
                                "unit": candidate["unit"],
                                "normal_mean": candidate["normal_mean"],
                                "normal_std": candidate["normal_std"],
                            },
                        )
                    ],
                ),
            ]
        )
    return atoms


def _evidence(
    task: dict[str, Any],
    inventory_or_legacy_anchor: CorpusInventory | str,
    legacy_inventory: CorpusInventory | None = None,
) -> list[EvidenceAtom]:
    """Build evidence atoms; accept the removed anchor argument for old callers."""

    inventory = (
        legacy_inventory
        if isinstance(inventory_or_legacy_anchor, str)
        else inventory_or_legacy_anchor
    )
    if inventory is None:  # pragma: no cover - guarded by the typed public call.
        raise TypeError("inventory is required")
    atoms: list[EvidenceAtom] = []
    for alias, uid in task["device_aliases"].items():
        atoms.append(
            EvidenceAtom(
                atom_id=f"device_locator_{alias}",
                fact=f"The full business locator resolves uniquely to device {alias}",
                modality="metadata",
                acceptable_sources=_archive_sources(inventory, uid, task["locators"][alias]),
            )
        )

    family = (
        task.get("task_family")
        or {
            "L1": "post_window",
            "L2": "threshold_replay",
            "L3": "multi_device_replay",
        }[task["task_level"]]
    )

    if family == "document_routed_window":
        uid = task["device_aliases"]["A"]
        device = inventory.device(uid)
        report = task["report_paths"]["A"]
        atoms.append(
            EvidenceAtom(
                atom_id="inspection_anchor",
                fact="Inspection end time defining the evaluation window",
                modality="document",
                acceptable_sources=[
                    _source(
                        report,
                        section="Basic Information",
                        key_values={
                            "Inspection window": {
                                "contains": [
                                    task["anchor"]["inspection_start"],
                                    task["anchor"]["inspection_end"],
                                ]
                            }
                        },
                    )
                ],
            )
        )
        atoms.extend(_routing_atoms(prefix="", report=report, selection=task["selection"]))
        atoms.append(
            EvidenceAtom(
                atom_id="selected_metric_window",
                fact="Hourly values for the document-selected metric",
                modality="timeseries",
                acceptable_sources=[
                    _source(
                        device.csv_path,
                        field=task["metric"]["field"],
                        start=task["window"]["start"],
                        end_exclusive=task["window"]["end_exclusive"],
                    )
                ],
            )
        )
        return atoms

    if family == "heterogeneous_metric_comparison":
        for alias, uid in task["device_aliases"].items():
            device = inventory.device(uid)
            report = task["report_paths"][alias]
            anchor = task["anchors"][alias]
            window = task["window"]["per_device"][alias]
            atoms.append(
                EvidenceAtom(
                    atom_id=f"inspection_anchor_{alias}",
                    fact=f"Inspection end time defining the branch window for {alias}",
                    modality="document",
                    acceptable_sources=[
                        _source(
                            report,
                            section="Basic Information",
                            key_values={
                                "Inspection window": {
                                    "contains": [
                                        anchor["inspection_start"],
                                        anchor["inspection_end"],
                                    ]
                                }
                            },
                        )
                    ],
                )
            )
            atoms.extend(
                _routing_atoms(
                    prefix=alias,
                    report=report,
                    selection=task["selections"][alias],
                )
            )
            atoms.append(
                EvidenceAtom(
                    atom_id=f"selected_metric_window_{alias}",
                    fact=f"Hourly values for the metric independently selected for {alias}",
                    modality="timeseries",
                    acceptable_sources=[
                        _source(
                            device.csv_path,
                            field=task["metrics"][alias]["field"],
                            start=window["start"],
                            end_exclusive=window["end_exclusive"],
                        )
                    ],
                )
            )
        return atoms

    if family == "pre_post_comparison":
        uid = task["device_aliases"]["A"]
        device = inventory.device(uid)
        report = task["report_paths"]["A"]
        metric = task["metric"]
        atoms.extend(
            [
                EvidenceAtom(
                    atom_id="fault_anchor",
                    fact="Fault-window end and abnormal metric defining both windows",
                    modality="document",
                    acceptable_sources=[
                        _source(
                            report,
                            section="Basic Information",
                            key_values={"Fault window": {"contains": [task["anchor"]["fault_end"]]}},
                        )
                    ],
                ),
                EvidenceAtom(
                    atom_id="metric_semantics",
                    fact=f"CSV field and unit for {metric['name']}",
                    modality="metadata",
                    acceptable_sources=[
                        _source(
                            metric["metric_catalog_path"],
                            field=metric["field"],
                            json_record={
                                "kind": "metric",
                                "field": metric["field"],
                                "name": metric["name"],
                                "unit": metric["unit"],
                            },
                        )
                    ],
                ),
                EvidenceAtom(
                    atom_id="pre_window_values",
                    fact="Hourly values before the fault-window end",
                    modality="timeseries",
                    acceptable_sources=[
                        _source(
                            device.csv_path,
                            field=metric["field"],
                            start=task["window"]["pre_start"],
                            end_exclusive=task["window"]["pre_end_exclusive"],
                        )
                    ],
                ),
                EvidenceAtom(
                    atom_id="post_window_values",
                    fact="Hourly values after the fault-window end",
                    modality="timeseries",
                    acceptable_sources=[
                        _source(
                            device.csv_path,
                            field=metric["field"],
                            start=task["window"]["post_start"],
                            end_exclusive=task["window"]["post_end_exclusive"],
                        )
                    ],
                ),
            ]
        )
        return atoms

    if family in {"aligned_window_comparison", "filter_then_compute"}:
        metric = task["metric"]
        for alias, uid in task["device_aliases"].items():
            device = inventory.device(uid)
            report = task["report_paths"][alias]
            window = task["window"]["per_device"][alias]
            atoms.extend(
                [
                    EvidenceAtom(
                        atom_id=f"inspection_anchor_{alias}",
                        fact=f"Inspection window defining the relative anchor for {alias}",
                        modality="document",
                        acceptable_sources=[
                            _source(
                                report,
                                section="Basic Information",
                                key_values={
                                    "Inspection window": {
                                        "contains": [
                                            task["anchors"][alias]["inspection_start"],
                                            task["anchors"][alias]["inspection_end"],
                                        ]
                                    }
                                },
                            )
                        ],
                    ),
                    EvidenceAtom(
                        atom_id=f"timeseries_{alias}",
                        fact=f"Hourly values for {alias}",
                        modality="timeseries",
                        acceptable_sources=[
                            _source(
                                device.csv_path,
                                field=metric["field"],
                                start=window["start"],
                                end_exclusive=window["end_exclusive"],
                            )
                        ],
                    ),
                ]
            )
        atoms.append(
            EvidenceAtom(
                atom_id="metric_semantics",
                fact=f"CSV field and unit for {metric['name']}",
                modality="metadata",
                acceptable_sources=[
                    _source(
                        metric["metric_catalog_path"],
                        field=metric["field"],
                        json_record={
                            "kind": "metric",
                            "field": metric["field"],
                            "name": metric["name"],
                            "unit": metric["unit"],
                        },
                    )
                ],
            )
        )
        if family == "filter_then_compute":
            criterion = task["filter"]
            for alias, uid in task["device_aliases"].items():
                atoms.append(
                    EvidenceAtom(
                        atom_id=f"filter_value_{alias}",
                        fact=f"Filter field value for {alias}",
                        modality="metadata",
                        acceptable_sources=[
                            _source(
                                inventory.device(uid).archive_path,
                                section="Basic Information",
                                key_values={
                                    criterion["field"]: {
                                        "equals": inventory.device(uid).metadata.get(
                                            criterion["field"], ""
                                        )
                                    }
                                },
                            )
                        ],
                    )
                )
        return atoms

    if task["task_level"] == "L1":
        uid = task["device_aliases"]["A"]
        device = inventory.device(uid)
        report = task["report_paths"]["A"]
        metric = task["metric"]
        atoms.extend(
            [
                EvidenceAtom(
                    atom_id="inspection_anchor",
                    fact="Inspection end time defining the new window",
                    modality="document",
                    acceptable_sources=[
                        _source(
                            report,
                            section="Basic Information",
                            key_values={
                                "Inspection window": {
                                    "contains": [
                                        task["anchor"]["inspection_start"],
                                        task["anchor"]["inspection_end"],
                                    ]
                                }
                            },
                        ),
                    ],
                ),
                EvidenceAtom(
                    atom_id="anchor_metric_value",
                    fact="Metric value at the inspection end timestamp",
                    modality="document",
                    acceptable_sources=[
                        _source(
                            report,
                            section="Key metrics at inspection end",
                            table_row={
                                "match": {
                                    "Metric": {"equals": metric["name"]},
                                    "Unit": {"equals": metric["unit"]},
                                },
                                "numeric": {
                                    "column": "Value",
                                    "value": task["anchor"]["anchor_value"],
                                    "abs_tolerance": 0.000001,
                                },
                            },
                        ),
                        _source(
                            device.csv_path,
                            field=metric["field"],
                            start=task["anchor"]["inspection_end"],
                            end_exclusive=task["window"]["start"],
                        ),
                    ],
                ),
                EvidenceAtom(
                    atom_id="metric_semantics",
                    fact=f"Mapping and unit for {metric['name']}",
                    modality="metadata",
                    acceptable_sources=[
                        _source(
                            metric["metric_catalog_path"],
                            field=metric["field"],
                            json_record={
                                "kind": "metric",
                                "field": metric["field"],
                                "name": metric["name"],
                                "unit": metric["unit"],
                            },
                        ),
                        _source(
                            device.archive_path,
                            section="Metric Dictionary",
                            table_row={
                                "match": {
                                    "metric_name": {"equals": metric["name"]},
                                    "Database field": {"equals": metric["field"]},
                                    "Unit": {"equals": metric["unit"]},
                                }
                            },
                        ),
                    ],
                ),
                EvidenceAtom(
                    atom_id="target_window_values",
                    fact="All hourly values used by the target-window computation",
                    modality="timeseries",
                    acceptable_sources=[
                        _source(
                            device.csv_path,
                            field=metric["field"],
                            start=task["window"]["start"],
                            end_exclusive=task["window"]["end_exclusive"],
                        )
                    ],
                ),
            ]
        )
        if family == "metadata_normalized":
            atoms.append(
                EvidenceAtom(
                    atom_id="interface_capacity",
                    fact="Interface type and its nominal capacity",
                    modality="metadata",
                    acceptable_sources=[
                        _source(
                            device.archive_path,
                            section="Basic Information",
                            key_values={"Uplink interface": {"equals": task["capacity"]["interface"]}},
                        )
                    ],
                )
            )
        return atoms

    rule = task["rule"]
    for alias, uid in task["device_aliases"].items():
        device = inventory.device(uid)
        report = task["report_paths"][alias]
        window = (
            task["window"] if task["task_level"] == "L2" else task["window"]["per_device"][alias]
        )
        atoms.append(
            EvidenceAtom(
                atom_id=f"fault_anchor_{alias}",
                fact=f"Fault-window end and associated rule for device {alias}",
                modality="document",
                acceptable_sources=[
                    _source(
                        report,
                        section="Basic Information",
                        key_values={
                            "Linked rule": {"equals": rule["rule_id"]},
                            "Fault window": {"contains": [task["anchors"][alias]["fault_end"]]},
                        },
                    )
                ],
            )
        )
        atoms.append(
            EvidenceAtom(
                atom_id=f"timeseries_{alias}",
                fact=f"Counterfactual input series and lookback for device {alias}",
                modality="timeseries",
                acceptable_sources=[
                    _source(
                        device.csv_path,
                        field=rule["metric_field"],
                        start=window["start"],
                        end_exclusive=window["end_exclusive"],
                        lookback_hours=(
                            int(rule.get("counterfactual_consecutive_hours")) - 1
                            if rule.get("counterfactual_consecutive_hours") is not None
                            else rule["consecutive_hours"] - 1
                        ),
                    )
                ],
            )
        )

    atoms.append(
        EvidenceAtom(
            atom_id="rule_semantics",
            fact="Original expression, parameter binding and consecutive-window semantics",
            modality="rule",
            acceptable_sources=[
                _source(
                    rule["rule_path"],
                    rule_id=rule["rule_id"],
                    json_record={
                        "kind": "rule",
                        "rule_id": rule["rule_id"],
                        "expression": rule["expression"],
                        "consecutive_hours": rule["consecutive_hours"],
                        "parameter": {
                            "name": rule["parameter_name"],
                            "value": rule["original_threshold"],
                        },
                    },
                )
            ],
        )
    )
    if family == "metadata_bound_replay":
        uid = task["device_aliases"]["A"]
        atoms.append(
            EvidenceAtom(
                atom_id="interface_capacity",
                fact="Device-specific interface capacity bound to C_if",
                modality="metadata",
                acceptable_sources=[
                    _source(
                        inventory.device(uid).archive_path,
                        section="Basic Information",
                        key_values={"Uplink interface": {"equals": rule["interface"]}},
                    )
                ],
            )
        )
    if family == "context_conditioned_replay":
        uid = task["device_aliases"]["A"]
        atoms.append(
            EvidenceAtom(
                atom_id="load_context",
                fact="Fault-report load context used to choose the threshold branch",
                modality="document",
                acceptable_sources=[
                    _source(
                        task["report_paths"]["A"],
                        section="Basic Information",
                        key_values={
                            "Load level": {"equals": task["context_decision"]["source_value"]}
                        },
                    )
                ],
            )
        )
    first_uid = next(iter(task["device_aliases"].values()))
    atoms.append(
        EvidenceAtom(
            atom_id="metric_semantics",
            fact=f"CSV field and unit for {rule['metric_name']}",
            modality="metadata",
            acceptable_sources=[
                _source(
                    rule["metric_catalog_path"],
                    field=rule["metric_field"],
                    json_record={
                        "kind": "metric",
                        "field": rule["metric_field"],
                        "name": rule["metric_name"],
                        "unit": rule["unit"],
                    },
                ),
                _source(
                    inventory.device(first_uid).archive_path,
                    section="Metric Dictionary",
                    table_row={
                        "match": {
                            "metric_name": {"equals": rule["metric_name"]},
                            "Database field": {"equals": rule["metric_field"]},
                            "Unit": {"equals": rule["unit"]},
                        }
                    },
                ),
            ],
        )
    )
    if rule.get("derived_metrics_path"):
        definition = _derived_metric_record(
            inventory,
            rule["derived_metrics_path"],
            rule["metric_field"],
        )
        atoms.append(
            EvidenceAtom(
                atom_id="derived_metric_definition",
                fact="Public definition of the derived rule input",
                modality="derived_metric",
                acceptable_sources=[
                    _source(
                        rule["derived_metrics_path"],
                        derived_key=rule["metric_field"],
                        json_record=definition,
                    ),
                ],
            )
        )
    return atoms


def _minimum_source_cover(atoms: list[EvidenceAtom]) -> int:
    source_sets = [{source.path for source in atom.acceptable_sources} for atom in atoms]
    universe = sorted(set().union(*source_sets))
    for size in range(1, len(universe) + 1):
        for choice in itertools.combinations(universe, size):
            selected = set(choice)
            if all(selected.intersection(sources) for sources in source_sets):
                return size
    return 0


def _complexity(task: dict[str, Any], atoms: list[EvidenceAtom]) -> Complexity:
    modalities = {atom.modality for atom in atoms}
    operator_count, depth = trace_complexity(task["gold_trace"])
    return Complexity(
        target_device_count=len(task["target_device_uids"]),
        evidence_atom_count=len(atoms),
        modality_count=len(modalities),
        operator_count=operator_count,
        critical_path_depth=depth,
    )


def render_task_pair(
    task: dict[str, Any],
    inventory: CorpusInventory,
    *,
    schema_version: str,
    question_texts: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if "gold_trace" not in task:
        execution = execute_gold(task, inventory)
        task = {**task, "answer": execution.answer, "gold_trace": execution.trace}
    definition = task_spec(task["task_level"])
    definition.validate_target_count(len(task["target_device_uids"]))
    family = task.get("task_family")
    qa_id = str(task["pair_id"])
    question = question_texts if question_texts is not None else definition.render_question(task)
    answer = public_answer(task["task_level"], task["answer"], family)
    tolerances = {key: value for key, value in task.get("tolerances", {}).items() if key in answer}
    public = PublicQuestion(
        schema_version=schema_version,
        qa_id=qa_id,
        pair_id=task["pair_id"],
        task_level=task["task_level"],
        task_family=family,
        question=question,
        answer_schema=definition.envelope_schema(family),
        scope_key=task["pair_id"],
    )
    atoms = _evidence(task, inventory)
    minimum_cover = _minimum_source_cover(atoms)
    diagnostics = dict(task.get("diagnostics", {}))
    diagnostics.update(
        {
            "minimum_distinct_source_files": minimum_cover,
            "single_file_sufficient": minimum_cover == 1,
            "raw_corpus_unmodified": True,
        }
    )
    gold = GoldRecord(
        schema_version=schema_version,
        qa_id=qa_id,
        pair_id=task["pair_id"],
        task_level=task["task_level"],
        task_family=family,
        anchor_type="Indirect",
        target_device_uids=task["target_device_uids"],
        answer=answer,
        tolerances=tolerances,
        evidence_atoms=atoms,
        gold_trace=task["gold_trace"],
        complexity=_complexity(task, atoms),
        diagnostics=diagnostics,
    )
    return [public.model_dump(mode="json")], [gold.model_dump(mode="json")]