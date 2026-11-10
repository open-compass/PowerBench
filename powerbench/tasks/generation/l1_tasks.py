from __future__ import annotations

import hashlib
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..documents import parse_document
from ..models import CorpusInventory, DocumentAnchor, Locator, Metric
from ..timeseries import (
    NumericPoint,
    decimal_value,
    numeric_window,
    read_numeric_series,
    read_numeric_window,
    validate_complete_hourly_window,
)
from ..util import format_timestamp
from .anchors import check_anchor
from .eligibility import EligibilityRejection


def _order_key(seed: int, *parts: object) -> str:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _round_decimal(value: Decimal, digits: int) -> float:
    quantum = Decimal(1).scaleb(-digits)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    return float(Decimal(0) if rounded.is_zero() else rounded)


def _report_anchor_value(report_path: Any, metric: Metric) -> Decimal:
    """Read the stated end-of-inspection value from the public report.

    The task uses a document value as its baseline, so candidate construction
    must verify that the report really contains the requested metric and that
    it agrees exactly with the frozen CSV at the report endpoint.
    """

    parsed = parse_document(report_path)
    matching_rows: list[dict[str, Any]] = []
    for table in parsed.get("tables", []):
        if "Key metrics at inspection end" not in str(table.get("section", "")):
            continue
        for row in table.get("rows", []):
            if str(row.get("Metric", "")).strip() == metric.name:
                matching_rows.append(row)
    if len(matching_rows) != 1:
        raise EligibilityRejection(
            "report_metric_missing_or_ambiguous",
            f"expected one {metric.name!r} row in inspection-end excerpt, "
            f"found {len(matching_rows)}",
        )
    row = matching_rows[0]
    report_unit = str(row.get("Unit", "")).strip()
    if report_unit != metric.unit:
        raise EligibilityRejection(
            "report_metric_unit_mismatch",
            f"inspection report unit {report_unit!r} disagrees with catalog unit {metric.unit!r}",
        )
    return decimal_value(row.get("Value"), context=f"report metric {metric.name!r}")


def _eligible_metrics(inventory: CorpusInventory, device_uid: str) -> list[Metric]:
    device = inventory.device(device_uid)
    result = []
    for metric in inventory.metrics_by_model.get(device.model_key, {}).values():
        if metric.field not in device.csv_columns:
            continue
        if metric.data_type.lower() not in {"float", "double", "number", "numeric"}:
            continue
        if "raw" not in metric.metric_type.lower():
            continue
        if not metric.unit:
            continue
        result.append(metric)
    return sorted(result, key=lambda item: item.field)


def _build_candidate(
    inventory: CorpusInventory,
    locator: Locator,
    device_uid: str,
    report_path: str,
    metric: Metric,
    window_hours: int,
    digits: int,
    anchor: DocumentAnchor | None = None,
    series_points: tuple[NumericPoint, ...] | None = None,
    stated_anchor_value: Decimal | None = None,
) -> dict[str, Any]:
    device = inventory.device(device_uid)
    anchor = anchor or check_anchor(inventory, device_uid, report_path)
    csv_path = inventory.raw_root / device.csv_path
    anchor_window = (
        numeric_window(
            series_points,
            field=metric.field,
            start=anchor.end,
            end=anchor.end + timedelta(hours=1),
        )
        if series_points is not None
        else read_numeric_window(
            csv_path,
            field=metric.field,
            start=anchor.end,
            end=anchor.end + timedelta(hours=1),
        )
    )
    anchor_points = validate_complete_hourly_window(anchor_window)
    anchor_value = anchor_points[0].value
    stated_anchor_value = (
        stated_anchor_value
        if stated_anchor_value is not None
        else _report_anchor_value(inventory.raw_root / report_path, metric)
    )
    if stated_anchor_value != anchor_value:
        raise EligibilityRejection(
            "report_csv_baseline_mismatch",
            f"inspection report baseline {stated_anchor_value} disagrees with CSV {anchor_value}",
        )

    start = anchor.end + timedelta(hours=1)
    end = start + timedelta(hours=window_hours)
    target_window = (
        numeric_window(series_points, field=metric.field, start=start, end=end)
        if series_points is not None
        else read_numeric_window(csv_path, field=metric.field, start=start, end=end)
    )
    points = validate_complete_hourly_window(target_window)
    values = [point.value for point in points]
    count_above = sum(value > anchor_value for value in values)
    if count_above in {0, len(values)}:
        raise EligibilityRejection(
            "degenerate_window",
            f"{count_above}/{len(values)} target values exceed the report baseline",
        )
    maximum = max(values)
    first_max = next(point.timestamp for point in points if point.value == maximum)
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    return {
        "task_level": "L1",
        "target_device_uids": [device_uid],
        "locators": {"A": locator.as_dict()},
        "device_aliases": {"A": device_uid},
        "report_paths": {"A": report_path},
        "rule": None,
        "metric": {
            "field": metric.field,
            "name": metric.name,
            "unit": metric.unit,
            "metric_catalog_path": f"{device.model_key}/metric_catalog.json",
        },
        "window": {
            "anchor_end": format_timestamp(anchor.end),
            "start": format_timestamp(start),
            "end_exclusive": format_timestamp(end),
            "hours": window_hours,
            "semantics": "anchor_end < timestamp <= anchor_end + W hours",
        },
        "anchor": {
            "event_id": anchor.event_id,
            "inspection_start": format_timestamp(anchor.start),
            "inspection_end": format_timestamp(anchor.end),
            "anchor_value": _round_decimal(anchor_value, digits + 2),
        },
        "answer": {
            "window_mean": _round_decimal(mean, digits),
            "window_max": _round_decimal(maximum, digits),
            "first_max_time": format_timestamp(first_max),
            "hours_above_anchor_value": count_above,
            "unit": metric.unit,
        },
        "tolerances": {
            "window_mean": 10 ** (-digits),
            "window_max": 10 ** (-digits),
        },
        "answer_round_digits": digits,
    }


def generate_l1_tasks(
    inventory: CorpusInventory,
    locators: dict[str, Locator],
    *,
    count: int,
    window_hours: list[int],
    digits: int,
    seed: int,
    max_reports_per_device: int | None = None,
    max_metrics_per_device: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count % len(window_hours):
        raise ValueError("L1 count must be divisible by the number of window sizes")
    quota = count // len(window_hours)
    selected: list[dict[str, Any]] = []
    used_devices: set[str] = set()
    used_models: set[str] = set()
    rejection_counts: dict[str, int] = {
        "missing_locator": len(inventory.devices) - len(locators),
        "no_eligible_metric": 0,
    }
    rejection_examples: dict[str, dict[str, Any]] = {}

    eligible_uids = []
    for uid in sorted(locators):
        device = inventory.device(uid)
        if device.model_key == "0015_300Mvar_dual_water_inner_cooled_condenser":
            continue
        if not device.check_reports or not _eligible_metrics(inventory, uid):
            rejection_counts["no_eligible_metric"] += 1
            continue
        eligible_uids.append(uid)

    for window_index, hours in enumerate(window_hours):
        made = 0
        # First preserve model diversity; relax only if the frozen data needs it.
        for diversity_pass in (True, False):
            ordered_devices = sorted(
                eligible_uids,
                key=lambda uid: _order_key(seed, "L1", hours, uid),
            )
            for uid in ordered_devices:
                if made >= quota:
                    break
                device = inventory.device(uid)
                if uid in used_devices or (diversity_pass and device.model_key in used_models):
                    continue
                target_event = f"E{((window_index * quota + made) % 12) + 1:02d}"
                reports = sorted(
                    device.check_reports,
                    key=lambda path: (
                        target_event not in path,
                        _order_key(seed, "report", uid, hours, path),
                    ),
                )
                metrics = sorted(
                    _eligible_metrics(inventory, uid),
                    key=lambda metric: _order_key(seed, "metric", uid, hours, metric.field),
                )
                candidate = None
                report_candidates = (
                    reports[:max_reports_per_device]
                    if max_reports_per_device is not None
                    else reports
                )
                metric_candidates = (
                    metrics[:max_metrics_per_device]
                    if max_metrics_per_device is not None
                    else metrics
                )
                for report_path in report_candidates:
                    for metric in metric_candidates:
                        try:
                            candidate = _build_candidate(
                                inventory,
                                locators[uid],
                                uid,
                                report_path,
                                metric,
                                hours,
                                digits,
                            )
                        except EligibilityRejection as rejection:
                            rejection_counts[rejection.reason] = (
                                rejection_counts.get(rejection.reason, 0) + 1
                            )
                            rejection_examples.setdefault(
                                rejection.reason,
                                {
                                    "device_uid": uid,
                                    "report_path": report_path,
                                    "metric_field": metric.field,
                                    "window_hours": hours,
                                    "detail": rejection.detail,
                                },
                            )
                            continue
                        break
                    if candidate is not None:
                        break
                if candidate is None:
                    continue
                selected.append(candidate)
                used_devices.add(uid)
                used_models.add(device.model_key)
                made += 1
            if made >= quota:
                break
        if made != quota:
            raise RuntimeError(f"Could generate only {made}/{quota} L1 tasks for W={hours}")

    selected.sort(
        key=lambda task: _order_key(
            seed,
            "L1-number",
            task["target_device_uids"][0],
            task["report_paths"]["A"],
            task["metric"]["field"],
            task["window"]["hours"],
        )
    )
    for index, task in enumerate(selected, start=1):
        task["pair_id"] = f"PB-L1-{index:03d}"
    return selected, {
        **rejection_counts,
        "rejection_examples": rejection_examples,
    }


def build_l1_candidate_pool(
    inventory: CorpusInventory,
    locators: dict[str, Locator],
    *,
    window_hours: list[int],
    digits: int,
    seed: int,
    max_reports_per_device: int | None = None,
    max_metrics_per_device: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate the bounded, executable L1 generation space.

    The census examines every locatable device.  For each device/window pair it
    records the first valid report/metric combination under the configured
    deterministic search caps.  This gives a reproducible candidate population
    without materializing every equivalent report/metric permutation.
    """

    pool: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {
        "missing_locator": len(inventory.devices) - len(locators),
        "no_eligible_metric": 0,
    }
    rejection_examples: dict[str, dict[str, Any]] = {}
    for uid in sorted(locators):
        device = inventory.device(uid)
        metrics = _eligible_metrics(inventory, uid)
        if not device.check_reports or not metrics:
            rejection_counts["no_eligible_metric"] += 1
            continue
        reports = sorted(
            device.check_reports,
            key=lambda path: _order_key(seed, "pool-report", uid, path),
        )
        metrics = sorted(
            metrics,
            key=lambda metric: _order_key(seed, "pool-metric", uid, metric.field),
        )
        if max_reports_per_device is not None:
            reports = reports[:max_reports_per_device]
        if max_metrics_per_device is not None:
            metrics = metrics[:max_metrics_per_device]
        anchor_cache: dict[str, DocumentAnchor] = {}
        stated_cache: dict[tuple[str, str], Decimal] = {}
        series_cache: dict[str, tuple[NumericPoint, ...]] = {}
        for hours in sorted(set(window_hours)):
            candidate = None
            for report_path in reports:
                for metric in metrics:
                    try:
                        if report_path not in anchor_cache:
                            anchor_cache[report_path] = check_anchor(inventory, uid, report_path)
                        anchor = anchor_cache[report_path]
                        cache_key = (report_path, metric.field)
                        if cache_key not in stated_cache:
                            stated_cache[cache_key] = _report_anchor_value(
                                inventory.raw_root / report_path, metric
                            )
                        if metric.field not in series_cache:
                            series_cache[metric.field] = read_numeric_series(
                                inventory.raw_root / device.csv_path, field=metric.field
                            )
                        candidate = _build_candidate(
                            inventory,
                            locators[uid],
                            uid,
                            report_path,
                            metric,
                            hours,
                            digits,
                            anchor=anchor,
                            series_points=series_cache[metric.field],
                            stated_anchor_value=stated_cache[cache_key],
                        )
                    except EligibilityRejection as rejection:
                        rejection_counts[rejection.reason] = (
                            rejection_counts.get(rejection.reason, 0) + 1
                        )
                        rejection_examples.setdefault(
                            rejection.reason,
                            {
                                "device_uid": uid,
                                "report_path": report_path,
                                "metric_field": metric.field,
                                "window_hours": hours,
                                "detail": rejection.detail,
                            },
                        )
                        continue
                    break
                if candidate is not None:
                    break
            if candidate is not None:
                pool.append(candidate)
    return pool, {
        **rejection_counts,
        "pool_size": len(pool),
        "device_count": len({task["target_device_uids"][0] for task in pool}),
        "rejection_examples": rejection_examples,
    }