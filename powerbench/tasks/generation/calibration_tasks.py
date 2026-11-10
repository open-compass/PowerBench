from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..documents import parse_document
from ..models import CorpusInventory, Locator
from ..rules import compile_inventory_rule, quantize_threshold, replay_rule
from ..timeseries import (
    decimal_value,
    numeric_window,
    read_numeric_series,
    validate_complete_hourly_window,
)
from ..util import format_timestamp, parse_timestamp
from .anchors import check_anchor, fault_anchor
from .eligibility import EligibilityRejection
from .l1_tasks import _build_candidate, _eligible_metrics

LOAD_FACTORS = {"high": 0.8, "normal": 0.9, "low": 1.2}
CONTEXT_WINDOW_OFFSETS = (0,)
HIGH_LOAD_TERMS = (
    "high load",
    "heavy load",
    "overload",
    "full load",
    "peak load",
    "elevated load",
    "above rated",
)
LOW_LOAD_TERMS = (
    "low load",
    "light load",
    "standby",
    "no load",
    "out of service",
)


def _order_key(seed: int, *parts: object) -> str:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _round(value: Decimal, digits: int) -> float:
    rounded = value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return 0.0 if rounded.is_zero() else float(rounded)


def classify_load_context(text: str) -> str:
    """Map free-form report context to the public three-way policy."""

    normalized = str(text).strip()
    if any(term in normalized for term in LOW_LOAD_TERMS):
        return "low"
    if any(term in normalized for term in HIGH_LOAD_TERMS):
        return "high"
    return "normal"


def inspection_metric_candidates(
    inventory: CorpusInventory,
    device_uid: str,
    report_path: str,
) -> list[dict[str, Any]]:
    """Rank report metrics by absolute z-score using public catalog statistics."""

    parsed = parse_document(inventory.raw_root / report_path)
    report_rows: dict[str, dict[str, Any]] = {}
    duplicate_names: set[str] = set()
    for table in parsed.get("tables", []):
        if "Key metrics at inspection end" not in str(table.get("section", "")):
            continue
        for row in table.get("rows", []):
            name = str(row.get("Metric", "")).strip()
            if not name:
                continue
            if name in report_rows:
                duplicate_names.add(name)
            report_rows[name] = row
    candidates: list[dict[str, Any]] = []
    for metric in _eligible_metrics(inventory, device_uid):
        if metric.normal_mean is None or metric.normal_std is None or metric.normal_std <= 0:
            continue
        row = report_rows.get(metric.name)
        if row is None or metric.name in duplicate_names:
            continue
        if str(row.get("Unit", "")).strip() != metric.unit:
            continue
        try:
            report_value = decimal_value(
                row.get("Value"), context=f"report metric {metric.name!r}"
            )
        except ValueError:
            continue
        mean = Decimal(str(metric.normal_mean))
        std = Decimal(str(metric.normal_std))
        deviation = abs(report_value - mean) / std
        candidates.append(
            {
                "field": metric.field,
                "name": metric.name,
                "unit": metric.unit,
                "report_value": float(report_value),
                "normal_mean": float(mean),
                "normal_std": float(std),
                "anchor_deviation_z": float(deviation),
                "metric_catalog_path": (
                    f"{inventory.device(device_uid).model_key}/metric_catalog.json"
                ),
            }
        )
    return sorted(
        candidates,
        key=lambda item: (-float(item["anchor_deviation_z"]), str(item["field"])),
    )


def build_document_routed_pool(
    inventory: CorpusInventory,
    locators: dict[str, Locator],
    *,
    window_hours: list[int],
    digits: int,
    seed: int,
    max_reports_per_device: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    series_cache: dict[tuple[str, str], tuple[Any, ...]] = {}
    for uid in sorted(locators, key=lambda value: _order_key(seed, "routed-device", value)):
        device = inventory.device(uid)
        reports = sorted(
            device.check_reports,
            key=lambda path: _order_key(seed, "routed-report", uid, path),
        )[:max_reports_per_device]
        for report_path in reports:
            candidates = inspection_metric_candidates(inventory, uid, report_path)
            if len(candidates) < 3:
                continue
            if float(candidates[0]["anchor_deviation_z"]) - float(
                candidates[1]["anchor_deviation_z"]
            ) < 0.05:
                continue
            selected = candidates[0]
            metric = inventory.metrics_by_model[device.model_key][str(selected["field"])]
            cache_key = (uid, metric.field)
            if cache_key not in series_cache:
                series_cache[cache_key] = read_numeric_series(
                    inventory.raw_root / device.csv_path,
                    field=metric.field,
                )
            try:
                anchor = check_anchor(inventory, uid, report_path)
            except ValueError:
                continue
            for hours in sorted(set(window_hours)):
                try:
                    base = _build_candidate(
                        inventory,
                        locators[uid],
                        uid,
                        report_path,
                        metric,
                        hours,
                        digits,
                        anchor=anchor,
                        series_points=series_cache[cache_key],
                        stated_anchor_value=Decimal(str(selected["report_value"])),
                    )
                except (EligibilityRejection, ValueError):
                    continue
                answer = {
                    "selected_metric": selected["name"],
                    "anchor_deviation_z": _round(
                        Decimal(str(selected["anchor_deviation_z"])), digits
                    ),
                    "window_mean": base["answer"]["window_mean"],
                    "window_max": base["answer"]["window_max"],
                    "first_max_time": base["answer"]["first_max_time"],
                    "unit": selected["unit"],
                }
                base.update(
                    {
                        "task_family": "document_routed_window",
                        "selection": {
                            "method": "max_absolute_standardized_deviation",
                            "source_section": "Key metrics at inspection end",
                            "candidate_metrics": candidates,
                            "selected_field": selected["field"],
                        },
                        "answer": answer,
                        "tolerances": {
                            "anchor_deviation_z": 10 ** (-digits),
                            "window_mean": 10 ** (-digits),
                            "window_max": 10 ** (-digits),
                        },
                        "diagnostics": {
                            "routing_candidate_count": len(candidates),
                            "routing_gap_z": round(
                                float(candidates[0]["anchor_deviation_z"])
                                - float(candidates[1]["anchor_deviation_z"]),
                                6,
                            ),
                        },
                    }
                )
                pool.append(base)
                if len(pool) >= max_candidates:
                    return pool
    return pool


def build_context_conditioned_pool(
    inventory: CorpusInventory,
    replay_pool: list[dict[str, Any]],
    *,
    threshold_factors: list[float],
) -> list[dict[str, Any]]:
    """Enumerate executable context-conditioned counterfactuals.

    Outcome balance is imposed later by the deterministic selector.  This
    builder only expands a fixed, public intervention grid: configured
    threshold factors and one common observation offset.  It never
    changes a computed label or invents a replay result to fill a stratum.
    """

    pool: list[dict[str, Any]] = []
    base_items: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in replay_pool:
        uid = item["device_uid"]
        report_path = item["report_path"]
        signature = (uid, report_path, int(item["window_hours"]))
        base_items.setdefault(signature, item)

    series_cache: dict[tuple[str, str], tuple[Any, ...]] = {}
    factors = tuple(Decimal(str(value)) for value in threshold_factors)
    for item in base_items.values():
        uid = item["device_uid"]
        report_path = item["report_path"]
        anchor = fault_anchor(inventory, uid, report_path)
        load_text = anchor.metadata.get("Load level", "").strip()
        if not load_text or load_text in {"unknown", "unrecorded", "TBD", "pending confirmation", "not applicable"}:
            continue
        category = classify_load_context(load_text)
        device = inventory.device(uid)
        rule_record = inventory.rules_by_model[device.model_key][item["rule_id"]]
        compiled = compile_inventory_rule(rule_record)
        cache_key = (uid, compiled.field)
        if cache_key not in series_cache:
            series_cache[cache_key] = read_numeric_series(
                inventory.raw_root / device.csv_path,
                field=compiled.field,
            )

        original = Decimal(str(item["original_threshold"]))
        for offset_hours in CONTEXT_WINDOW_OFFSETS:
            start = anchor.end + timedelta(hours=offset_hours)
            end = start + timedelta(hours=int(item["window_hours"]))
            try:
                window = numeric_window(
                    series_cache[cache_key],
                    field=compiled.field,
                    start=start,
                    end=end,
                    lookback_hours=compiled.lookback_hours,
                )
                validate_complete_hourly_window(
                    window,
                    required_lookback_hours=compiled.lookback_hours,
                )
            except ValueError:
                continue

            for factor_decimal in factors:
                expected = quantize_threshold(
                    original * factor_decimal,
                    decimal_places=6,
                )
                if expected == original:
                    continue
                boundary_margin = max(abs(expected) * Decimal("1e-10"), Decimal("1e-9"))
                if any(abs(point.value - expected) <= boundary_margin for point in window.points):
                    continue
                result = replay_rule(compiled, window, threshold=expected)
                # All-hour positives leak almost no temporal information and
                # are poor calibration questions even though they are valid.
                if result.triggered and result.trigger_hours == int(item["window_hours"]):
                    continue

                factor = float(factor_decimal)
                policy_factors = dict(LOAD_FACTORS)
                policy_factors[category] = factor
                task = {
                    "task_level": "L2",
                    "task_family": "context_conditioned_replay",
                    "target_device_uids": [uid],
                    "device_aliases": {"A": uid},
                    "locators": {"A": copy.deepcopy(item["locator"])},
                    "report_paths": {"A": report_path},
                    "anchors": {
                        "A": {
                            "event_id": item["anchor_event_id"],
                            "fault_end": item["fault_end"],
                            "discovery_time": item["discovery_time"],
                            "report_number": item["report_number"],
                        }
                    },
                    "rule": {
                        "rule_id": item["rule_id"],
                        "rule_path": item["rule_path"],
                        "metric_catalog_path": item["metric_catalog_path"],
                        "derived_metrics_path": item["derived_metrics_path"],
                        "metric_field": item["metric_field"],
                        "metric_name": item["metric_name"],
                        "unit": item["unit"],
                        "expression": item["expression"],
                        "operator": item["operator"],
                        "consecutive_hours": item["consecutive_hours"],
                        "parameter_name": item["parameter_name"],
                        "original_threshold": item["original_threshold"],
                        "counterfactual_threshold": float(expected),
                        "counterfactual_threshold_text": (
                            format(expected, "f").rstrip("0").rstrip(".")
                        ),
                    },
                    "window": {
                        "start": format_timestamp(start),
                        "end_exclusive": format_timestamp(end),
                        "hours": item["window_hours"],
                        "offset_hours": offset_hours,
                        "semantics": (
                            f"[fault_end + {offset_hours}h, "
                            f"fault_end + {offset_hours}h + W)"
                        ),
                    },
                }
                task["context_policy"] = {
                    "source_field": "Load level",
                    "factors": policy_factors,
                    "high_description": "high, heavy load, overload, full load, peak load, elevated load, or above rated",
                    "low_description": "low, light load, standby, no load, or out of service",
                    "low_precedence": True,
                }
                task["context_decision"] = {
                    "source_value": load_text,
                    "category": category,
                    "factor": factor,
                }
                task["answer"] = {
                    "load_category": category,
                    "effective_threshold": float(expected),
                    "unit": task["rule"]["unit"],
                    **result.as_answer(),
                }
                task["tolerances"] = {"effective_threshold": 0.000001}
                pool.append(task)
    return pool


def build_heterogeneous_metric_pool(
    inventory: CorpusInventory,
    routed_pool: list[dict[str, Any]],
    *,
    device_counts: tuple[int, ...],
    digits: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_hours: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in routed_pool:
        by_hours[int(task["window"]["hours"])].append(task)

    pool: list[dict[str, Any]] = []
    series_cache: dict[tuple[str, str], tuple[Any, ...]] = {}
    for hours, candidates in sorted(by_hours.items()):
        ordered = sorted(
            candidates,
            key=lambda task: _order_key(
                seed,
                "heterogeneous-order",
                task["target_device_uids"][0],
                task["report_paths"]["A"],
            ),
        )
        for size in sorted(set(device_counts)):
            if len(ordered) < size:
                continue
            for offset in range(min(len(ordered), 120)):
                locals_: list[dict[str, Any]] = []
                used_uids: set[str] = set()
                used_models: set[str] = set()
                for step in range(len(ordered)):
                    local = ordered[(offset + step) % len(ordered)]
                    uid = local["target_device_uids"][0]
                    model_key = inventory.device(uid).model_key
                    if uid in used_uids:
                        continue
                    if model_key in used_models and len(used_models) < size:
                        continue
                    locals_.append(local)
                    used_uids.add(uid)
                    used_models.add(model_key)
                    if len(locals_) == size:
                        break
                if len(locals_) != size:
                    continue
                if len({local["metric"]["field"] for local in locals_}) < 2:
                    continue
                aliases = [f"D{index}" for index in range(1, size + 1)]
                local_results: list[dict[str, Any]] = []
                raw_scores: dict[str, Decimal] = {}
                for alias, local in zip(aliases, locals_, strict=True):
                    uid = local["target_device_uids"][0]
                    device = inventory.device(uid)
                    field = local["metric"]["field"]
                    cache_key = (uid, field)
                    if cache_key not in series_cache:
                        series_cache[cache_key] = read_numeric_series(
                            inventory.raw_root / device.csv_path,
                            field=field,
                        )
                    points = validate_complete_hourly_window(
                        numeric_window(
                            series_cache[cache_key],
                            field=field,
                            start=parse_timestamp(local["window"]["start"]),
                            end=parse_timestamp(local["window"]["end_exclusive"]),
                        )
                    )
                    selected = local["selection"]["candidate_metrics"][0]
                    mean = Decimal(str(selected["normal_mean"]))
                    std = Decimal(str(selected["normal_std"]))
                    scores = [abs(point.value - mean) / std for point in points]
                    peak = max(scores)
                    first = points[scores.index(peak)].timestamp
                    raw_scores[alias] = peak
                    local_results.append(
                        {
                            "alias": alias,
                            "selected_metric": selected["name"],
                            "peak_deviation_z": _round(peak, digits),
                            "first_peak_time": format_timestamp(first),
                        }
                    )
                if len(set(raw_scores.values())) != len(raw_scores):
                    continue
                highest = max(raw_scores, key=raw_scores.get)
                span = max(raw_scores.values()) - min(raw_scores.values())
                task = {
                    "task_level": "L3",
                    "task_family": "heterogeneous_metric_comparison",
                    "target_device_uids": [
                        local["target_device_uids"][0] for local in locals_
                    ],
                    "device_aliases": {
                        alias: local["target_device_uids"][0]
                        for alias, local in zip(aliases, locals_, strict=True)
                    },
                    "locators": {
                        alias: copy.deepcopy(local["locators"]["A"])
                        for alias, local in zip(aliases, locals_, strict=True)
                    },
                    "report_paths": {
                        alias: local["report_paths"]["A"]
                        for alias, local in zip(aliases, locals_, strict=True)
                    },
                    "anchors": {
                        alias: copy.deepcopy(local["anchor"])
                        for alias, local in zip(aliases, locals_, strict=True)
                    },
                    "metrics": {
                        alias: copy.deepcopy(local["metric"])
                        for alias, local in zip(aliases, locals_, strict=True)
                    },
                    "selections": {
                        alias: copy.deepcopy(local["selection"])
                        for alias, local in zip(aliases, locals_, strict=True)
                    },
                    "window": {
                        "hours": hours,
                        "per_device": {
                            alias: copy.deepcopy(local["window"])
                            for alias, local in zip(aliases, locals_, strict=True)
                        },
                        "semantics": "per-device post-inspection windows",
                    },
                    "rule": None,
                    "answer": {
                        "per_device": local_results,
                        "highest_risk_device": highest,
                        "risk_score_span": _round(span, digits),
                    },
                    "tolerances": {
                        "peak_deviation_z": 10 ** (-digits),
                        "risk_score_span": 10 ** (-digits),
                    },
                    "answer_round_digits": digits,
                    "diagnostics": {
                        "branch_metric_count": len(
                            {local["metric"]["field"] for local in locals_}
                        ),
                        "branch_model_count": len(used_models),
                    },
                }
                pool.append(task)
                if len(pool) >= 500:
                    return pool
    return pool


__all__ = [
    "HIGH_LOAD_TERMS",
    "LOAD_FACTORS",
    "LOW_LOAD_TERMS",
    "build_context_conditioned_pool",
    "build_document_routed_pool",
    "build_heterogeneous_metric_pool",
    "classify_load_context",
    "inspection_metric_candidates",
]