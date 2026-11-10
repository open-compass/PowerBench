from __future__ import annotations
import copy
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

from ..documents import parse_document
from ..models import CorpusInventory, Metric
from ..task_specs import normalize_task_family
from ..timeseries import decimal_value

UNIT_ELIGIBILITY_CLAUSE = (
    "Eligible candidates are numeric raw metrics whose catalog unit field is non-empty and "
    "whose unit in the inspection report exactly matches the catalog unit."
)

_DIMENSIONLESS_UNITS = frozenset({"", "-", "—", "n/a", "dimensionless"})
_CELSIUS_UNITS = frozenset({"℃", "°C"})
_NUMERIC_TYPES = frozenset({"float", "double", "number", "numeric"})


def _units_semantically_compatible(left: str, right: str) -> bool:
    left = left.strip()
    right = right.strip()
    return (
        left == right
        or {left, right}.issubset(_DIMENSIONLESS_UNITS)
        or {left, right}.issubset(_CELSIUS_UNITS)
    )


def _report_rows(inventory: CorpusInventory, report_path: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    parsed = parse_document(inventory.raw_root / report_path)
    for table in parsed.get("tables", []):
        if "Key metrics at inspection end" not in str(table.get("section", "")):
            continue
        for row in table.get("rows", []):
            name = str(row.get("Metric", "")).strip()
            if not name:
                continue
            if name in rows:
                duplicates.add(name)
            rows[name] = row
    return {name: row for name, row in rows.items() if name not in duplicates}


def _is_numeric_original(metric: Metric, csv_columns: tuple[str, ...]) -> bool:
    return (
        metric.field in csv_columns
        and metric.data_type.lower() in _NUMERIC_TYPES
        and "raw" in metric.metric_type.lower()
        and metric.normal_mean is not None
        and metric.normal_std is not None
        and metric.normal_std > 0
    )


def _literal_candidates(
    inventory: CorpusInventory,
    *,
    device_uid: str,
    report_path: str,
) -> list[dict[str, Any]]:
    """Candidates admitted by the old wording, including equivalent blank units."""

    device = inventory.device(device_uid)
    rows = _report_rows(inventory, report_path)
    candidates: list[dict[str, Any]] = []
    for metric in inventory.metrics_by_model.get(device.model_key, {}).values():
        if not _is_numeric_original(metric, device.csv_columns):
            continue
        row = rows.get(metric.name)
        if row is None:
            continue
        report_unit = str(row.get("Unit", "")).strip()
        if not _units_semantically_compatible(report_unit, metric.unit):
            continue
        try:
            report_value = decimal_value(
                row.get("Value"), context=f"report metric {metric.name!r}"
            )
        except ValueError:
            continue
        mean = Decimal(str(metric.normal_mean))
        std = Decimal(str(metric.normal_std))
        candidates.append(
            {
                "field": metric.field,
                "name": metric.name,
                "catalog_unit": metric.unit,
                "report_unit": report_unit,
                "anchor_deviation_z": float(abs(report_value - mean) / std),
                "admitted_by_generator": bool(metric.unit) and report_unit == metric.unit,
            }
        )
    return sorted(
        candidates,
        key=lambda item: (-float(item["anchor_deviation_z"]), str(item["field"])),
    )


def audit_metric_eligibility(
    tasks: Iterable[Mapping[str, Any]], inventory: CorpusInventory
) -> list[dict[str, Any]]:
    """Return the exact frozen tasks whose old wording admits a different winner."""

    findings: list[dict[str, Any]] = []
    for task in tasks:
        family = normalize_task_family(str(task.get("task_level")), task.get("task_family"))
        if family == "document_routed_window":
            selections = {"A": task["selection"]}
        elif family == "heterogeneous_metric_comparison":
            selections = task["selections"]
        else:
            continue

        branches: list[dict[str, Any]] = []
        for alias, selection in selections.items():
            candidates = _literal_candidates(
                inventory,
                device_uid=str(task["device_aliases"][alias]),
                report_path=str(task["report_paths"][alias]),
            )
            if not candidates:
                continue
            selected_field = str(selection["selected_field"])
            literal_winner = candidates[0]
            if literal_winner["field"] == selected_field:
                continue
            branches.append(
                {
                    "alias": alias,
                    "generator_selected_field": selected_field,
                    "wording_admitted_winner": literal_winner,
                }
            )
        if branches:
            findings.append(
                {
                    "pair_id": str(task["pair_id"]),
                    "task_level": str(task["task_level"]),
                    "task_family": family,
                    "affected_branches": branches,
                }
            )
    return sorted(findings, key=lambda item: str(item["pair_id"]))


def clarify_affected_records(
    records: Iterable[Mapping[str, Any]],
    findings: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Append the missing candidate-set contract to affected private records only."""

    affected = {str(item["pair_id"]) for item in findings}
    updated: list[dict[str, Any]] = []
    changed: list[str] = []
    for source in records:
        record = copy.deepcopy(dict(source))
        pair_id = str(record["pair_id"])
        if pair_id in affected:
            if UNIT_ELIGIBILITY_CLAUSE not in record["final"]["question"]:
                record["final"]["question"] = (
                    record["final"]["question"].rstrip() + " " + UNIT_ELIGIBILITY_CLAUSE
                )
                changed.append(pair_id)
            validation = record.setdefault("validation", {})
            clarifications = validation.setdefault("deterministic_clarifications", [])
            if "metric_unit_eligibility" not in clarifications:
                clarifications.append("metric_unit_eligibility")
        updated.append(record)
    return updated, sorted(changed)


__all__ = [
    "UNIT_ELIGIBILITY_CLAUSE",
    "audit_metric_eligibility",
    "clarify_affected_records",
]