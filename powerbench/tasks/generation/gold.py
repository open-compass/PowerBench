from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..models import CorpusInventory
from ..rules import compile_inventory_rule, quantize_threshold, replay_rule
from ..task_specs import task_spec
from ..timeseries import NumericPoint, read_numeric_window, validate_complete_hourly_window
from ..util import canonical_json, format_timestamp, parse_timestamp
from .anchors import check_anchor, fault_anchor


class GoldExecutionError(ValueError):
    """Raised when a serialized semantic task cannot be executed from raw data."""


@dataclass(frozen=True)
class GoldExecution:
    """The answer and auditable operation trace produced by one Gold execution."""

    answer: dict[str, Any]
    trace: list[dict[str, Any]]


def _round(value: Decimal, digits: int = 4) -> float:
    quantum = Decimal(1).scaleb(-digits)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    return 0.0 if rounded.is_zero() else float(rounded)


def _point_payload(point: NumericPoint) -> dict[str, str]:
    return {
        "timestamp": format_timestamp(point.timestamp),
        "value": format(point.value, "f"),
    }


def _points_summary(points: Sequence[NumericPoint]) -> dict[str, Any]:
    payload = [_point_payload(point) for point in points]
    return {
        "row_count": len(payload),
        "first_timestamp": payload[0]["timestamp"] if payload else None,
        "last_timestamp": payload[-1]["timestamp"] if payload else None,
        "values_sha256": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
    }


def _step(
    step_id: str,
    op: str,
    *,
    depends_on: Iterable[str] = (),
    inputs: Mapping[str, Any] | None = None,
    output: Any = None,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "op": op,
        "depends_on": list(depends_on),
        "inputs": dict(inputs or {}),
        "output": output,
    }


def validate_trace(trace: Sequence[Mapping[str, Any]]) -> None:
    """Validate that a Gold trace is a topologically ordered dependency graph."""

    if not trace:
        raise GoldExecutionError("Gold trace must contain at least one step")
    seen: set[str] = set()
    for index, raw in enumerate(trace):
        step_id = raw.get("step_id")
        op = raw.get("op")
        dependencies = raw.get("depends_on")
        if not isinstance(step_id, str) or not step_id:
            raise GoldExecutionError(f"Gold trace step {index} has no step_id")
        if step_id in seen:
            raise GoldExecutionError(f"Duplicate Gold trace step_id: {step_id}")
        if not isinstance(op, str) or not op:
            raise GoldExecutionError(f"Gold trace step {step_id} has no op")
        if not isinstance(dependencies, list) or any(
            not isinstance(value, str) for value in dependencies
        ):
            raise GoldExecutionError(f"Gold trace step {step_id} has invalid depends_on")
        missing = sorted(set(dependencies).difference(seen))
        if missing:
            raise GoldExecutionError(
                f"Gold trace step {step_id} depends on missing or later steps: {missing}"
            )
        seen.add(step_id)


def trace_complexity(trace: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Return actual operator count and longest dependency-chain depth."""

    validate_trace(trace)
    depth: dict[str, int] = {}
    for step in trace:
        step_id = str(step["step_id"])
        dependencies = [str(value) for value in step["depends_on"]]
        depth[step_id] = 1 + max((depth[value] for value in dependencies), default=0)
    return len(trace), max(depth.values())


def _aliases(task: Mapping[str, Any]) -> list[str]:
    aliases = task.get("device_aliases")
    if not isinstance(aliases, Mapping) or not aliases:
        raise GoldExecutionError("Task has no device_aliases")
    result = [str(alias) for alias in aliases]
    if len(result) != len(set(result)):
        raise GoldExecutionError("Task has duplicate device aliases")
    return result


def _device_uid(task: Mapping[str, Any], alias: str) -> str:
    aliases = task["device_aliases"]
    value = aliases.get(alias)
    if not isinstance(value, str) or not value:
        raise GoldExecutionError(f"Task has no device UID for {alias}")
    return value


def _report_path(task: Mapping[str, Any], alias: str) -> str:
    reports = task.get("report_paths")
    value = reports.get(alias) if isinstance(reports, Mapping) else None
    if not isinstance(value, str) or not value:
        raise GoldExecutionError(f"Task has no report path for {alias}")
    return value


def _execute_l1(task: Mapping[str, Any], inventory: CorpusInventory) -> GoldExecution:
    aliases = _aliases(task)
    if len(aliases) != 1:
        raise GoldExecutionError("L1 requires exactly one target device")
    alias = aliases[0]
    uid = _device_uid(task, alias)
    device = inventory.device(uid)
    report_path = _report_path(task, alias)
    anchor = check_anchor(inventory, uid, report_path)
    metric = task.get("metric")
    if not isinstance(metric, Mapping):
        raise GoldExecutionError("L1 task has no metric")
    field = str(metric.get("field") or "")
    name = str(metric.get("name") or "")
    unit = str(metric.get("unit") or "")
    catalog_metric = inventory.metrics_by_model.get(device.model_key, {}).get(field)
    if catalog_metric is None:
        raise GoldExecutionError(f"Metric {field!r} is not available for {device.model_key}")
    if (catalog_metric.name, catalog_metric.unit) != (name, unit):
        raise GoldExecutionError(f"Metric semantics disagree for {device.model_key}/{field}")

    window = task.get("window")
    if not isinstance(window, Mapping):
        raise GoldExecutionError("L1 task has no window")
    start = parse_timestamp(window.get("start"))
    end = parse_timestamp(window.get("end_exclusive"))
    if start <= anchor.end:
        raise GoldExecutionError("L1 target window must begin after the inspection end")
    csv_path = inventory.raw_root / device.csv_path
    anchor_window = read_numeric_window(
        csv_path,
        field=field,
        start=anchor.end,
        end=start,
    )
    anchor_points = validate_complete_hourly_window(anchor_window)
    if len(anchor_points) != 1:
        raise GoldExecutionError("L1 anchor window must contain exactly one hourly point")
    anchor_value = anchor_points[0].value
    target_window = read_numeric_window(csv_path, field=field, start=start, end=end)
    target_points = validate_complete_hourly_window(target_window)
    values = [point.value for point in target_points]
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    maximum = max(values)
    first_max = next(point.timestamp for point in target_points if point.value == maximum)
    count_above = sum(value > anchor_value for value in values)
    digits = int(task.get("answer_round_digits", 4))
    answer = {
        "window_mean": _round(mean, digits),
        "window_max": _round(maximum, digits),
        "first_max_time": format_timestamp(first_max),
        "hours_above_anchor_value": count_above,
        "unit": unit,
    }
    trace = [
        _step(
            "resolve_device",
            "RESOLVE_DEVICE",
            inputs={"alias": alias, "locator": task.get("locators", {}).get(alias)},
            output={"device_uid": uid, "model_key": device.model_key},
        ),
        _step(
            "read_report",
            "READ_INSPECTION_REPORT",
            depends_on=("resolve_device",),
            inputs={"path": report_path, "event_id": anchor.event_id},
            output={
                "inspection_start": format_timestamp(anchor.start),
                "inspection_end": format_timestamp(anchor.end),
            },
        ),
        _step(
            "resolve_metric",
            "RESOLVE_METRIC",
            depends_on=("resolve_device",),
            inputs={"catalog_path": metric.get("metric_catalog_path"), "field": field},
            output={"field": field, "name": name, "unit": unit},
        ),
        _step(
            "read_anchor",
            "READ_TIMESERIES",
            depends_on=("read_report", "resolve_metric"),
            inputs={
                "path": device.csv_path,
                "field": field,
                "start": format_timestamp(anchor.end),
                "end_exclusive": format_timestamp(start),
            },
            output={**_points_summary(anchor_points), "value": format(anchor_value, "f")},
        ),
        _step(
            "read_window",
            "READ_TIMESERIES",
            depends_on=("read_report", "resolve_metric"),
            inputs={
                "path": device.csv_path,
                "field": field,
                "start": format_timestamp(start),
                "end_exclusive": format_timestamp(end),
            },
            output=_points_summary(target_points),
        ),
        _step(
            "mean",
            "MEAN",
            depends_on=("read_window",),
            inputs={"row_count": len(values)},
            output={"value": format(mean, "f")},
        ),
        _step(
            "max",
            "MAX_AND_FIRST_ARGMAX",
            depends_on=("read_window",),
            output={"value": format(maximum, "f"), "first_time": format_timestamp(first_max)},
        ),
        _step(
            "count_above",
            "COUNT_STRICTLY_ABOVE",
            depends_on=("read_anchor", "read_window"),
            inputs={"threshold": format(anchor_value, "f")},
            output={"count": count_above},
        ),
        _step(
            "answer",
            "FORMAT_ANSWER",
            depends_on=("mean", "max", "count_above"),
            inputs={"round_digits": digits},
            output=answer,
        ),
    ]
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _execute_document_routed_window(
    task: Mapping[str, Any], inventory: CorpusInventory
) -> GoldExecution:
    from .calibration_tasks import inspection_metric_candidates

    uid = _device_uid(task, "A")
    report = _report_path(task, "A")
    ranked = inspection_metric_candidates(inventory, uid, report)
    if len(ranked) < 3:
        raise GoldExecutionError("Document-routed task has fewer than three eligible metrics")
    selected = ranked[0]
    metric = task.get("metric")
    selection = task.get("selection")
    if not isinstance(metric, Mapping) or not isinstance(selection, Mapping):
        raise GoldExecutionError("Document-routed task has no metric-selection binding")
    if metric.get("field") != selected["field"] or selection.get("selected_field") != selected[
        "field"
    ]:
        raise GoldExecutionError("Document-routed metric is not the reproducible top-ranked metric")

    base = _execute_l1(task, inventory)
    digits = int(task.get("answer_round_digits", 4))
    answer = {
        "selected_metric": selected["name"],
        "anchor_deviation_z": _round(
            Decimal(str(selected["anchor_deviation_z"])), digits
        ),
        "window_mean": base.answer["window_mean"],
        "window_max": base.answer["window_max"],
        "first_max_time": base.answer["first_max_time"],
        "unit": selected["unit"],
    }
    trace: list[dict[str, Any]] = []
    for raw_step in base.trace:
        if raw_step["step_id"] == "answer":
            continue
        step = {**raw_step, "depends_on": list(raw_step["depends_on"])}
        trace.append(step)
        if step["step_id"] == "resolve_metric":
            trace.append(
                _step(
                    "select_metric",
                    "SELECT_MAX_STANDARDIZED_REPORT_DEVIATION",
                    depends_on=("read_report", "resolve_metric"),
                    inputs={
                        "method": "abs(report_value-normal_mean)/normal_std",
                        "candidate_count": len(ranked),
                    },
                    output={
                        "field": selected["field"],
                        "name": selected["name"],
                        "anchor_deviation_z": answer["anchor_deviation_z"],
                    },
                )
            )
    for step in trace:
        if step["step_id"] in {"read_anchor", "read_window"}:
            step["depends_on"] = ["select_metric"]
    trace.append(
        _step(
            "answer",
            "FORMAT_ANSWER",
            depends_on=("select_metric", "mean", "max"),
            inputs={"round_digits": digits},
            output=answer,
        )
    )
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _metric_binding(
    task: Mapping[str, Any], inventory: CorpusInventory, alias: str
) -> tuple[str, Any, Mapping[str, Any]]:
    uid = _device_uid(task, alias)
    device = inventory.device(uid)
    metric = task.get("metric")
    if not isinstance(metric, Mapping):
        raise GoldExecutionError("Task has no metric")
    field = str(metric.get("field") or "")
    catalog = inventory.metrics_by_model.get(device.model_key, {}).get(field)
    if catalog is None or (catalog.name, catalog.unit) != (
        str(metric.get("name") or ""),
        str(metric.get("unit") or ""),
    ):
        raise GoldExecutionError(f"Metric binding disagrees for {uid}/{field}")
    return uid, device, metric


def _execute_l1_threshold_episode(
    task: Mapping[str, Any], inventory: CorpusInventory
) -> GoldExecution:
    uid, device, metric = _metric_binding(task, inventory, "A")
    report = _report_path(task, "A")
    anchor = check_anchor(inventory, uid, report)
    field = str(metric["field"])
    start = parse_timestamp(task["window"]["start"])
    end = parse_timestamp(task["window"]["end_exclusive"])
    baseline_window = read_numeric_window(
        inventory.raw_root / device.csv_path,
        field=field,
        start=anchor.end,
        end=anchor.end + timedelta(hours=1),
    )
    baseline = validate_complete_hourly_window(baseline_window)[0].value
    points = validate_complete_hourly_window(
        read_numeric_window(inventory.raw_root / device.csv_path, field=field, start=start, end=end)
    )
    flags = [point.value > baseline for point in points]
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    first = next((point.timestamp for point, flag in zip(points, flags, strict=True) if flag), None)
    answer = {
        "exceedance_hours": sum(flags),
        "longest_exceedance_run_hours": longest,
        "first_exceedance_time": format_timestamp(first) if first else None,
    }
    trace = [
        _step(
            "resolve_device",
            "RESOLVE_DEVICE",
            inputs={"alias": "A", "locator": task["locators"]["A"]},
            output={"device_uid": uid},
        ),
        _step(
            "read_report",
            "READ_INSPECTION_REPORT",
            depends_on=("resolve_device",),
            inputs={"path": report},
            output={"inspection_end": format_timestamp(anchor.end)},
        ),
        _step(
            "read_baseline",
            "READ_TIMESERIES",
            depends_on=("read_report",),
            inputs={"path": device.csv_path, "field": field, "start": format_timestamp(anchor.end)},
            output={"value": format(baseline, "f")},
        ),
        _step(
            "read_window",
            "READ_TIMESERIES",
            depends_on=("read_report",),
            inputs={
                "path": device.csv_path,
                "field": field,
                "start": format_timestamp(start),
                "end_exclusive": format_timestamp(end),
            },
            output=_points_summary(points),
        ),
        _step(
            "episodes",
            "REDUCE_THRESHOLD_EPISODES",
            depends_on=("read_baseline", "read_window"),
            inputs={"strict": True},
            output=answer,
        ),
        _step("answer", "FORMAT_ANSWER", depends_on=("episodes",), output=answer),
    ]
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _execute_l1_pre_post(task: Mapping[str, Any], inventory: CorpusInventory) -> GoldExecution:
    uid, device, metric = _metric_binding(task, inventory, "A")
    report = _report_path(task, "A")
    anchor = fault_anchor(inventory, uid, report)
    field = str(metric["field"])
    window = task["window"]
    pre = validate_complete_hourly_window(
        read_numeric_window(
            inventory.raw_root / device.csv_path,
            field=field,
            start=parse_timestamp(window["pre_start"]),
            end=parse_timestamp(window["pre_end_exclusive"]),
        )
    )
    post = validate_complete_hourly_window(
        read_numeric_window(
            inventory.raw_root / device.csv_path,
            field=field,
            start=parse_timestamp(window["post_start"]),
            end=parse_timestamp(window["post_end_exclusive"]),
        )
    )
    pre_mean = sum((point.value for point in pre), Decimal(0)) / Decimal(len(pre))
    post_mean = sum((point.value for point in post), Decimal(0)) / Decimal(len(post))
    if pre_mean == 0:
        raise GoldExecutionError("Cannot compute a relative change from zero pre-window mean")
    change = (post_mean - pre_mean) / abs(pre_mean) * Decimal(100)
    maximum = max(point.value for point in post)
    first_max = next(point.timestamp for point in post if point.value == maximum)
    digits = int(task.get("answer_round_digits", 4))
    answer = {
        "pre_mean": _round(pre_mean, digits),
        "post_mean": _round(post_mean, digits),
        "change_rate_percent": _round(change, digits),
        "post_max": _round(maximum, digits),
        "first_post_max_time": format_timestamp(first_max),
        "unit": str(metric["unit"]),
    }
    trace = [
        _step(
            "resolve_device",
            "RESOLVE_DEVICE",
            inputs={"alias": "A", "locator": task["locators"]["A"]},
            output={"device_uid": uid},
        ),
        _step(
            "read_report",
            "READ_FAULT_REPORT",
            depends_on=("resolve_device",),
            inputs={"path": report},
            output={"fault_end": format_timestamp(anchor.end)},
        ),
        _step(
            "read_pre",
            "READ_TIMESERIES",
            depends_on=("read_report",),
            inputs={
                "path": device.csv_path,
                "field": field,
                "start": window["pre_start"],
                "end_exclusive": window["pre_end_exclusive"],
            },
            output=_points_summary(pre),
        ),
        _step(
            "read_post",
            "READ_TIMESERIES",
            depends_on=("read_report",),
            inputs={
                "path": device.csv_path,
                "field": field,
                "start": window["post_start"],
                "end_exclusive": window["post_end_exclusive"],
            },
            output=_points_summary(post),
        ),
        _step(
            "compare_windows",
            "COMPARE_PRE_POST_WINDOWS",
            depends_on=("read_pre", "read_post"),
            output=answer,
        ),
        _step("answer", "FORMAT_ANSWER", depends_on=("compare_windows",), output=answer),
    ]
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _execute_l1_metadata_normalized(
    task: Mapping[str, Any], inventory: CorpusInventory
) -> GoldExecution:
    from .family_tasks import INTERFACE_CAPACITY_MBPS

    uid, device, metric = _metric_binding(task, inventory, "A")
    report = _report_path(task, "A")
    anchor = check_anchor(inventory, uid, report)
    interface = device.metadata.get("Uplink interface", "").strip().upper()
    capacity = INTERFACE_CAPACITY_MBPS.get(interface)
    if capacity is None or float(task["capacity"]["value_mbps"]) != capacity:
        raise GoldExecutionError(f"Invalid interface capacity binding for {uid}")
    start = parse_timestamp(task["window"]["start"])
    end = parse_timestamp(task["window"]["end_exclusive"])
    points = validate_complete_hourly_window(
        read_numeric_window(
            inventory.raw_root / device.csv_path,
            field=str(metric["field"]),
            start=start,
            end=end,
        )
    )
    values = [point.value / Decimal(str(capacity)) * Decimal(100) for point in points]
    threshold = Decimal(str(task["utilization_threshold_percent"]))
    flags = [value > threshold for value in values]
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    first = next((point.timestamp for point, flag in zip(points, flags, strict=True) if flag), None)
    digits = int(task.get("answer_round_digits", 4))
    answer = {
        "capacity_mbps": capacity,
        "mean_utilization_percent": _round(sum(values, Decimal(0)) / Decimal(len(values)), digits),
        "longest_above_utilization_run_hours": longest,
        "first_above_utilization_time": format_timestamp(first) if first else None,
    }
    trace = [
        _step(
            "resolve_device",
            "RESOLVE_DEVICE",
            inputs={"alias": "A", "locator": task["locators"]["A"]},
            output={"device_uid": uid},
        ),
        _step(
            "read_archive",
            "BIND_METADATA_PARAMETER",
            depends_on=("resolve_device",),
            inputs={"path": device.archive_path, "field": "Uplink interface"},
            output={"interface": interface, "capacity_mbps": capacity},
        ),
        _step(
            "read_report",
            "READ_INSPECTION_REPORT",
            depends_on=("resolve_device",),
            inputs={"path": report},
            output={"inspection_end": format_timestamp(anchor.end)},
        ),
        _step(
            "read_window",
            "READ_TIMESERIES",
            depends_on=("read_report",),
            inputs={
                "path": device.csv_path,
                "field": metric["field"],
                "start": format_timestamp(start),
                "end_exclusive": format_timestamp(end),
            },
            output=_points_summary(points),
        ),
        _step(
            "normalize",
            "NORMALIZE_BY_CAPACITY",
            depends_on=("read_archive", "read_window"),
            inputs={"threshold_percent": float(threshold)},
            output=answer,
        ),
        _step("answer", "FORMAT_ANSWER", depends_on=("normalize",), output=answer),
    ]
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _rule_result_payload(result: Any) -> dict[str, Any]:
    return {
        "triggered": bool(result.triggered),
        "first_trigger_time": (
            format_timestamp(result.first_trigger_time) if result.first_trigger_time else None
        ),
        "first_trigger_offset_hours": result.first_trigger_offset_hours,
        "trigger_hours": int(result.trigger_hours),
        "best_consecutive_margin": _round(result.best_consecutive_margin),
    }


def _execute_local_replay(
    task: Mapping[str, Any],
    inventory: CorpusInventory,
    alias: str,
    *,
    shared_dependencies: Sequence[str] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uid = _device_uid(task, alias)
    device = inventory.device(uid)
    report_path = _report_path(task, alias)
    anchor = fault_anchor(inventory, uid, report_path)
    rule_payload = task.get("rule")
    if not isinstance(rule_payload, Mapping):
        raise GoldExecutionError("Replay task has no rule")
    rule_id = str(rule_payload.get("rule_id") or "")
    engine = str(rule_payload.get("engine") or "consecutive")
    inventory_rule = inventory.rules_by_model.get(device.model_key, {}).get(rule_id)
    if inventory_rule is None and engine != "rolling_mean_utilization":
        raise GoldExecutionError(f"Rule {device.model_key}/{rule_id} is not executable")
    if anchor.rule_id != rule_id:
        raise GoldExecutionError(f"Report {report_path} does not bind to rule {rule_id}")
    if engine == "rolling_mean_utilization":
        return _execute_metadata_bound_replay(
            task,
            inventory,
            alias,
            anchor=anchor,
            shared_dependencies=shared_dependencies,
        )
    assert inventory_rule is not None
    compiled = compile_inventory_rule(inventory_rule)
    duration = rule_payload.get("counterfactual_consecutive_hours")
    if duration is not None:
        duration = int(duration)
        compiled = type(compiled)(
            expression=compiled.expression,
            source_field=compiled.source_field,
            field=compiled.field,
            operator=compiled.operator,
            consecutive_hours=duration,
            parameter_name=compiled.parameter_name,
        )
    threshold = rule_payload.get("counterfactual_threshold")
    task_window = task.get("window")
    if not isinstance(task_window, Mapping):
        raise GoldExecutionError("Replay task has no window")
    if task.get("task_level") == "L3":
        windows = task_window.get("per_device")
        raw_window = windows.get(alias) if isinstance(windows, Mapping) else None
    else:
        raw_window = task_window
    if not isinstance(raw_window, Mapping):
        raise GoldExecutionError(f"Replay task has no window for {alias}")
    start = parse_timestamp(raw_window.get("start"))
    end = parse_timestamp(raw_window.get("end_exclusive"))
    csv_path = inventory.raw_root / device.csv_path
    numeric = read_numeric_window(
        csv_path,
        field=compiled.field,
        start=start,
        end=end,
        lookback_hours=compiled.lookback_hours,
    )
    target_points = validate_complete_hourly_window(
        numeric, required_lookback_hours=compiled.lookback_hours
    )
    result = replay_rule(compiled, numeric, threshold=threshold)
    local = _rule_result_payload(result)
    values = numeric.values_by_timestamp()
    atomic_true = 0
    for point in numeric.points:
        value = point.value
        if (compiled.operator == ">" and value > result.threshold) or (
            compiled.operator == "<" and value < result.threshold
        ):
            atomic_true += 1
    prefix = alias.casefold()
    resolve_id = f"{prefix}_resolve"
    report_id = f"{prefix}_report"
    window_id = f"{prefix}_window"
    compare_id = f"{prefix}_compare"
    consecutive_id = f"{prefix}_consecutive"
    local_id = f"{prefix}_local"
    trace = [
        _step(
            resolve_id,
            "RESOLVE_DEVICE",
            inputs={"alias": alias, "locator": task.get("locators", {}).get(alias)},
            output={"device_uid": uid, "model_key": device.model_key},
        ),
        _step(
            report_id,
            "READ_FAULT_REPORT",
            depends_on=(resolve_id,),
            inputs={"path": report_path, "event_id": anchor.event_id},
            output={"fault_end": format_timestamp(anchor.end), "rule_id": anchor.rule_id},
        ),
        _step(
            window_id,
            "READ_TIMESERIES",
            depends_on=(report_id, *shared_dependencies),
            inputs={
                "path": device.csv_path,
                "field": compiled.field,
                "start": format_timestamp(start),
                "end_exclusive": format_timestamp(end),
                "lookback_hours": compiled.lookback_hours,
            },
            output={
                **_points_summary(numeric.points),
                "target_row_count": len(target_points),
            },
        ),
        _step(
            compare_id,
            "ATOMIC_COMPARE",
            depends_on=(window_id, *shared_dependencies),
            inputs={"operator": compiled.operator, "threshold": format(result.threshold, "f")},
            output={"satisfied_point_count": atomic_true, "evaluated_point_count": len(values)},
        ),
        _step(
            consecutive_id,
            "APPLY_CONSECUTIVE",
            depends_on=(compare_id,),
            inputs={"consecutive_hours": compiled.consecutive_hours},
            output={
                "trigger_timestamps": [
                    format_timestamp(value) for value in result.trigger_timestamps
                ],
                "trigger_hours": result.trigger_hours,
            },
        ),
        _step(
            local_id,
            "REDUCE_LOCAL_REPLAY",
            depends_on=(consecutive_id,),
            output=local,
        ),
    ]
    return local, trace


def _execute_metadata_bound_replay(
    task: Mapping[str, Any],
    inventory: CorpusInventory,
    alias: str,
    *,
    anchor: Any,
    shared_dependencies: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute MA5600T K10 using its published six-hour utilization rule."""

    from .family_tasks import INTERFACE_CAPACITY_MBPS

    uid = _device_uid(task, alias)
    device = inventory.device(uid)
    rule = task["rule"]
    interface = device.metadata.get("Uplink interface", "").strip().upper()
    capacity = INTERFACE_CAPACITY_MBPS.get(interface)
    if capacity is None or float(rule.get("capacity_mbps")) != capacity:
        raise GoldExecutionError(f"Cannot bind interface capacity for {uid}")
    raw_window = task["window"]
    start = parse_timestamp(raw_window["start"])
    end = parse_timestamp(raw_window["end_exclusive"])
    rolling_hours = 6
    numeric = read_numeric_window(
        inventory.raw_root / device.csv_path,
        field="uplink_interface_traffic",
        start=start,
        end=end,
        lookback_hours=rolling_hours - 1,
    )
    targets = validate_complete_hourly_window(numeric, required_lookback_hours=rolling_hours - 1)
    values = numeric.values_by_timestamp()
    threshold = Decimal(str(rule["counterfactual_threshold"]))
    triggered: list[Any] = []
    for point in targets:
        rolling = [values[point.timestamp - timedelta(hours=offset)] for offset in range(6)]
        utilization = sum(rolling, Decimal(0)) / Decimal(6) / Decimal(str(capacity))
        if utilization > threshold:
            triggered.append(point.timestamp)
    local = {
        "triggered": bool(triggered),
        "first_trigger_time": format_timestamp(triggered[0]) if triggered else None,
        "first_trigger_offset_hours": (
            int((triggered[0] - start).total_seconds() // 3600) if triggered else None
        ),
        "trigger_hours": len(triggered),
        "best_consecutive_margin": 0.0,
    }
    prefix = alias.casefold()
    trace = [
        _step(
            f"{prefix}_resolve",
            "RESOLVE_DEVICE",
            inputs={"alias": alias, "locator": task.get("locators", {}).get(alias)},
            output={"device_uid": uid, "model_key": device.model_key},
        ),
        _step(
            f"{prefix}_report",
            "READ_FAULT_REPORT",
            depends_on=(f"{prefix}_resolve",),
            inputs={"path": _report_path(task, alias), "event_id": anchor.event_id},
            output={"fault_end": format_timestamp(anchor.end), "rule_id": anchor.rule_id},
        ),
        _step(
            f"{prefix}_capacity",
            "BIND_METADATA_PARAMETER",
            depends_on=(f"{prefix}_resolve",),
            inputs={"archive_path": device.archive_path, "field": "Uplink interface"},
            output={"interface": interface, "capacity_mbps": capacity},
        ),
        _step(
            f"{prefix}_window",
            "READ_TIMESERIES",
            depends_on=(f"{prefix}_report", f"{prefix}_capacity", *shared_dependencies),
            inputs={
                "path": device.csv_path,
                "field": "uplink_interface_traffic",
                "start": format_timestamp(start),
                "end_exclusive": format_timestamp(end),
                "lookback_hours": 5,
            },
            output={**_points_summary(numeric.points), "target_row_count": len(targets)},
        ),
        _step(
            f"{prefix}_rolling",
            "ROLLING_MEAN_UTILIZATION_COMPARE",
            depends_on=(f"{prefix}_window",),
            inputs={"rolling_hours": 6, "threshold": format(threshold, "f")},
            output={"trigger_timestamps": [format_timestamp(value) for value in triggered]},
        ),
        _step(
            f"{prefix}_local",
            "REDUCE_LOCAL_REPLAY",
            depends_on=(f"{prefix}_rolling",),
            output=local,
        ),
    ]
    return local, trace


def _shared_rule_steps(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    rule = task.get("rule")
    if not isinstance(rule, Mapping):
        raise GoldExecutionError("Replay task has no rule")
    load = _step(
        "load_rule",
        "LOAD_RULE",
        inputs={"path": rule.get("rule_path"), "rule_id": rule.get("rule_id")},
        output={
            "expression": rule.get("expression"),
            "field": rule.get("metric_field"),
            "operator": rule.get("operator"),
            "consecutive_hours": rule.get("consecutive_hours"),
            "parameter_name": rule.get("parameter_name"),
            "original_threshold": rule.get("original_threshold"),
        },
    )
    override = _step(
        "override_threshold",
        "OVERRIDE_PARAMETER",
        depends_on=("load_rule",),
        inputs={
            "parameter_name": rule.get("parameter_name"),
            "counterfactual_threshold": rule.get("counterfactual_threshold"),
        },
        output={"effective_threshold": rule.get("counterfactual_threshold")},
    )
    return [load, override]


def _execute_l2(task: Mapping[str, Any], inventory: CorpusInventory) -> GoldExecution:
    aliases = _aliases(task)
    if len(aliases) != 1:
        raise GoldExecutionError("L2 requires exactly one target device")
    trace = _shared_rule_steps(task)
    local, local_trace = _execute_local_replay(
        task,
        inventory,
        aliases[0],
        shared_dependencies=("override_threshold",),
    )
    trace.extend(local_trace)
    if task.get("task_family"):
        answer = {key: local[key] for key in ("triggered", "first_trigger_time", "trigger_hours")}
    else:
        answer = {key: value for key, value in local.items() if key != "first_trigger_offset_hours"}
    trace.append(
        _step(
            "answer",
            "FORMAT_ANSWER",
            depends_on=(f"{aliases[0].casefold()}_local",),
            output=answer,
        )
    )
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _execute_context_conditioned_replay(
    task: Mapping[str, Any], inventory: CorpusInventory
) -> GoldExecution:
    from .calibration_tasks import classify_load_context

    uid = _device_uid(task, "A")
    report_path = _report_path(task, "A")
    anchor = fault_anchor(inventory, uid, report_path)
    load_text = anchor.metadata.get("Load level", "").strip()
    category = classify_load_context(load_text)
    policy = task.get("context_policy")
    if not isinstance(policy, Mapping) or not isinstance(policy.get("factors"), Mapping):
        raise GoldExecutionError("Context-conditioned replay has no threshold policy")
    factor = policy["factors"].get(category)
    if factor is None:
        raise GoldExecutionError(f"Threshold policy has no branch for {category!r}")
    rule = task.get("rule")
    decision = task.get("context_decision")
    if not isinstance(rule, Mapping) or not isinstance(decision, Mapping):
        raise GoldExecutionError("Context-conditioned replay has no decision binding")
    effective = quantize_threshold(
        Decimal(str(rule["original_threshold"])) * Decimal(str(factor)),
        decimal_places=6,
    )
    if Decimal(str(rule["counterfactual_threshold"])) != effective:
        raise GoldExecutionError("Context-conditioned threshold is not reproducible")
    if (
        decision.get("source_value") != load_text
        or decision.get("category") != category
        or Decimal(str(decision.get("factor"))) != Decimal(str(factor))
    ):
        raise GoldExecutionError("Stored context decision disagrees with the fault report")

    base = _execute_l2(task, inventory)
    answer = {
        "load_category": category,
        "effective_threshold": float(effective),
        "unit": str(rule.get("unit") or ""),
        **base.answer,
    }
    steps = {step["step_id"]: {**step, "depends_on": list(step["depends_on"])} for step in base.trace}
    steps["a_report"]["output"] = {
        **steps["a_report"]["output"],
        "load_level": load_text,
    }
    derive = _step(
        "derive_threshold",
        "BRANCH_ON_REPORT_CONTEXT",
        depends_on=("load_rule", "a_report"),
        inputs={"source_field": "Load level", "policy": task.get("context_policy")},
        output={
            "load_category": category,
            "factor": factor,
            "effective_threshold": float(effective),
        },
    )
    steps["override_threshold"]["depends_on"] = ["derive_threshold"]
    steps["override_threshold"]["inputs"] = {
        **steps["override_threshold"]["inputs"],
        "counterfactual_threshold": float(effective),
    }
    ordered_ids = [
        "load_rule",
        "a_resolve",
        "a_report",
        "derive_threshold",
        "override_threshold",
        "a_window",
        "a_compare",
        "a_consecutive",
        "a_local",
    ]
    trace = [derive if step_id == "derive_threshold" else steps[step_id] for step_id in ordered_ids]
    trace.append(
        _step(
            "answer",
            "FORMAT_ANSWER",
            depends_on=("derive_threshold", "a_local"),
            output=answer,
        )
    )
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _execute_l3(task: Mapping[str, Any], inventory: CorpusInventory) -> GoldExecution:
    aliases = _aliases(task)
    if not 3 <= len(aliases) <= 5:
        raise GoldExecutionError("L3 requires between three and five target devices")
    trace = _shared_rule_steps(task)
    local_by_alias: dict[str, dict[str, Any]] = {}
    for alias in aliases:
        local, local_trace = _execute_local_replay(
            task,
            inventory,
            alias,
            shared_dependencies=("override_threshold",),
        )
        local_by_alias[alias] = local
        trace.extend(local_trace)

    local_dependencies = tuple(f"{alias.casefold()}_local" for alias in aliases)
    triggered = [alias for alias in aliases if local_by_alias[alias]["triggered"]]
    earliest = (
        min(
            triggered,
            key=lambda alias: int(local_by_alias[alias]["first_trigger_offset_hours"]),
        )
        if triggered
        else None
    )
    offsets = [int(local_by_alias[alias]["first_trigger_offset_hours"]) for alias in triggered]
    offset_span = max(offsets) - min(offsets) if offsets else None
    margins = {
        alias: Decimal(str(local_by_alias[alias]["best_consecutive_margin"])) for alias in aliases
    }
    margin_order = sorted(aliases, key=lambda alias: margins[alias], reverse=True)
    margin_span = _round(max(margins.values()) - min(margins.values()))
    per_device = [{"alias": alias, **local_by_alias[alias]} for alias in aliases]
    answer = {
        "per_device": per_device,
        "triggered_devices": triggered,
        "earliest_trigger_device": earliest,
        "trigger_offset_span_hours": offset_span,
        "margin_order": margin_order,
        "margin_span": margin_span,
    }
    trace.extend(
        [
            _step(
                "triggered_devices",
                "FILTER_TRIGGERED_DEVICES",
                depends_on=local_dependencies,
                output={"aliases": triggered},
            ),
            _step(
                "earliest_trigger",
                "ARGMIN_FIRST_TRIGGER_OFFSET",
                depends_on=("triggered_devices", *local_dependencies),
                output={"alias": earliest},
            ),
            _step(
                "trigger_offset_span",
                "RANGE_FIRST_TRIGGER_OFFSETS",
                depends_on=("triggered_devices", *local_dependencies),
                output={"hours": offset_span},
            ),
            _step(
                "margin_order",
                "ORDER_BEST_CONSECUTIVE_MARGINS",
                depends_on=local_dependencies,
                output={"aliases": margin_order},
            ),
            _step(
                "margin_span",
                "RANGE_BEST_CONSECUTIVE_MARGINS",
                depends_on=local_dependencies,
                output={"value": margin_span},
            ),
            _step(
                "answer",
                "FORMAT_ANSWER",
                depends_on=(
                    "triggered_devices",
                    "earliest_trigger",
                    "trigger_offset_span",
                    "margin_order",
                    "margin_span",
                ),
                output=answer,
            ),
        ]
    )
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _execute_l3_window_family(task: Mapping[str, Any], inventory: CorpusInventory) -> GoldExecution:
    family = str(task["task_family"])
    aliases = _aliases(task)
    if not 3 <= len(aliases) <= 5:
        raise GoldExecutionError("L3 window families require three to five devices")
    metric = task["metric"]
    local_means: dict[str, Decimal] = {}
    increases: dict[str, Decimal] = {}
    trace: list[dict[str, Any]] = []
    for alias in aliases:
        uid, device, bound_metric = _metric_binding(task, inventory, alias)
        if dict(bound_metric) != dict(metric):
            raise GoldExecutionError(f"Metric semantics differ for {alias}")
        report = _report_path(task, alias)
        anchor = check_anchor(inventory, uid, report)
        window = task["window"]["per_device"][alias]
        start = parse_timestamp(window["start"])
        end = parse_timestamp(window["end_exclusive"])
        points = validate_complete_hourly_window(
            read_numeric_window(
                inventory.raw_root / device.csv_path,
                field=str(metric["field"]),
                start=start,
                end=end,
            )
        )
        mean_value = sum((point.value for point in points), Decimal(0)) / Decimal(len(points))
        local_means[alias] = mean_value
        baseline = Decimal(str(task["anchors"][alias]["anchor_value"]))
        increases[alias] = mean_value - baseline
        prefix = alias.casefold()
        trace.extend(
            [
                _step(
                    f"{prefix}_resolve",
                    "RESOLVE_DEVICE",
                    inputs={"alias": alias, "locator": task["locators"][alias]},
                    output={"device_uid": uid},
                ),
                _step(
                    f"{prefix}_report",
                    "READ_INSPECTION_REPORT",
                    depends_on=(f"{prefix}_resolve",),
                    inputs={"path": report},
                    output={"inspection_end": format_timestamp(anchor.end)},
                ),
                _step(
                    f"{prefix}_window",
                    "READ_TIMESERIES",
                    depends_on=(f"{prefix}_report",),
                    inputs={
                        "path": device.csv_path,
                        "field": metric["field"],
                        "start": format_timestamp(start),
                        "end_exclusive": format_timestamp(end),
                    },
                    output=_points_summary(points),
                ),
                _step(
                    f"{prefix}_mean",
                    "MEAN",
                    depends_on=(f"{prefix}_window",),
                    output={"value": format(mean_value, "f")},
                ),
            ]
        )
    dependencies = tuple(f"{alias.casefold()}_mean" for alias in aliases)
    if family == "aligned_window_comparison":
        best = max(increases, key=increases.get)
        ordered = sorted(increases.values())
        middle = len(ordered) // 2
        med = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        answer = {
            "largest_increase_device": best,
            "largest_increase_percentage_points": _round(increases[best]),
            "group_median_increase_percentage_points": _round(med),
            "lead_over_median_percentage_points": _round(increases[best] - med),
        }
        trace.append(
            _step("reduce_group", "ARGMAX_AND_MEDIAN", depends_on=dependencies, output=answer)
        )
    elif family == "filter_then_compute":
        criterion = task["filter"]
        selected = [
            alias
            for alias in aliases
            if inventory.device(_device_uid(task, alias)).metadata.get(criterion["field"])
            == criterion["value"]
        ]
        if not selected:
            raise GoldExecutionError("L3 metadata filter selected no devices")
        highest = max(selected, key=lambda alias: local_means[alias])
        values = [local_means[alias] for alias in selected]
        answer = {
            "selected_devices": selected,
            "highest_mean_device": highest,
            "mean_span": _round(max(values) - min(values)),
            "unit": str(metric["unit"]),
        }
        trace.append(
            _step(
                "reduce_group",
                "FILTER_AND_COMPARE_MEANS",
                depends_on=dependencies,
                inputs={"filter": criterion},
                output=answer,
            )
        )
    else:  # pragma: no cover - caller dispatches known families.
        raise GoldExecutionError(f"Unsupported L3 window family: {family}")
    trace.append(_step("answer", "FORMAT_ANSWER", depends_on=("reduce_group",), output=answer))
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def _execute_heterogeneous_metric_comparison(
    task: Mapping[str, Any], inventory: CorpusInventory
) -> GoldExecution:
    from .calibration_tasks import inspection_metric_candidates

    aliases = _aliases(task)
    if not 3 <= len(aliases) <= 5:
        raise GoldExecutionError("Heterogeneous comparison requires three to five devices")
    metrics = task.get("metrics")
    selections = task.get("selections")
    windows = task.get("window", {}).get("per_device")
    anchors = task.get("anchors")
    if not all(isinstance(value, Mapping) for value in (metrics, selections, windows, anchors)):
        raise GoldExecutionError("Heterogeneous comparison has incomplete branch bindings")

    digits = int(task.get("answer_round_digits", 4))
    trace: list[dict[str, Any]] = []
    per_device: list[dict[str, Any]] = []
    raw_scores: dict[str, Decimal] = {}
    local_dependencies: list[str] = []
    for alias in aliases:
        uid = _device_uid(task, alias)
        device = inventory.device(uid)
        report_path = _report_path(task, alias)
        report_anchor = check_anchor(inventory, uid, report_path)
        ranked = inspection_metric_candidates(inventory, uid, report_path)
        if len(ranked) < 3:
            raise GoldExecutionError(f"{alias} has fewer than three eligible routing metrics")
        selected = ranked[0]
        metric = metrics[alias]
        selection = selections[alias]
        if not isinstance(metric, Mapping) or not isinstance(selection, Mapping):
            raise GoldExecutionError(f"{alias} has an invalid metric-selection record")
        if metric.get("field") != selected["field"] or selection.get("selected_field") != selected[
            "field"
        ]:
            raise GoldExecutionError(f"{alias} does not bind the reproducible selected metric")
        window = windows[alias]
        if not isinstance(window, Mapping):
            raise GoldExecutionError(f"{alias} has no evaluation window")
        start = parse_timestamp(window.get("start"))
        end = parse_timestamp(window.get("end_exclusive"))
        if start != report_anchor.end + timedelta(hours=1):
            raise GoldExecutionError(f"{alias} window does not start after its inspection end")
        points = validate_complete_hourly_window(
            read_numeric_window(
                inventory.raw_root / device.csv_path,
                field=str(selected["field"]),
                start=start,
                end=end,
            )
        )
        normal_mean = Decimal(str(selected["normal_mean"]))
        normal_std = Decimal(str(selected["normal_std"]))
        scores = [abs(point.value - normal_mean) / normal_std for point in points]
        peak = max(scores)
        first_peak = points[scores.index(peak)].timestamp
        raw_scores[alias] = peak
        local = {
            "alias": alias,
            "selected_metric": selected["name"],
            "peak_deviation_z": _round(peak, digits),
            "first_peak_time": format_timestamp(first_peak),
        }
        per_device.append(local)
        prefix = alias.casefold()
        local_id = f"{prefix}_peak"
        local_dependencies.append(local_id)
        trace.extend(
            [
                _step(
                    f"{prefix}_resolve",
                    "RESOLVE_DEVICE",
                    inputs={"alias": alias, "locator": task.get("locators", {}).get(alias)},
                    output={"device_uid": uid, "model_key": device.model_key},
                ),
                _step(
                    f"{prefix}_report",
                    "READ_INSPECTION_REPORT",
                    depends_on=(f"{prefix}_resolve",),
                    inputs={"path": report_path},
                    output={"inspection_end": format_timestamp(report_anchor.end)},
                ),
                _step(
                    f"{prefix}_select",
                    "SELECT_MAX_STANDARDIZED_REPORT_DEVIATION",
                    depends_on=(f"{prefix}_report",),
                    inputs={
                        "catalog_path": selected["metric_catalog_path"],
                        "method": "abs(report_value-normal_mean)/normal_std",
                        "candidate_count": len(ranked),
                    },
                    output={"field": selected["field"], "name": selected["name"]},
                ),
                _step(
                    f"{prefix}_window",
                    "READ_TIMESERIES",
                    depends_on=(f"{prefix}_select",),
                    inputs={
                        "path": device.csv_path,
                        "field": selected["field"],
                        "start": format_timestamp(start),
                        "end_exclusive": format_timestamp(end),
                    },
                    output=_points_summary(points),
                ),
                _step(
                    local_id,
                    "MAX_ABSOLUTE_STANDARDIZED_DEVIATION",
                    depends_on=(f"{prefix}_window",),
                    inputs={"normal_mean": float(normal_mean), "normal_std": float(normal_std)},
                    output=local,
                ),
            ]
        )
    if len(set(raw_scores.values())) != len(raw_scores):
        raise GoldExecutionError("Heterogeneous comparison contains a tied risk score")
    highest = max(raw_scores, key=raw_scores.get)
    answer = {
        "per_device": per_device,
        "highest_risk_device": highest,
        "risk_score_span": _round(max(raw_scores.values()) - min(raw_scores.values()), digits),
    }
    trace.append(
        _step(
            "reduce_group",
            "ARGMAX_AND_RANGE",
            depends_on=tuple(local_dependencies),
            output={
                "highest_risk_device": highest,
                "risk_score_span": answer["risk_score_span"],
            },
        )
    )
    trace.append(_step("answer", "FORMAT_ANSWER", depends_on=("reduce_group",), output=answer))
    validate_trace(trace)
    return GoldExecution(answer=answer, trace=trace)


def execute_gold(task: Mapping[str, Any], inventory: CorpusInventory) -> GoldExecution:
    """Execute one L1/L2/L3 semantic task against the frozen raw corpus."""

    level = task.get("task_level")
    definition = task_spec(str(level))
    targets = task.get("target_device_uids")
    if not isinstance(targets, list):
        raise GoldExecutionError("Task has no target_device_uids list")
    try:
        definition.validate_target_count(len(targets))
    except ValueError as exc:
        raise GoldExecutionError(str(exc)) from exc
    family = str(
        task.get("task_family")
        or {"L1": "post_window", "L2": "threshold_replay", "L3": "multi_device_replay"}.get(
            str(level)
        )
    )
    if family == "threshold_episode":
        return _execute_l1_threshold_episode(task, inventory)
    if family == "pre_post_comparison":
        return _execute_l1_pre_post(task, inventory)
    if family == "metadata_normalized":
        return _execute_l1_metadata_normalized(task, inventory)
    if family == "document_routed_window":
        return _execute_document_routed_window(task, inventory)
    if family == "context_conditioned_replay":
        return _execute_context_conditioned_replay(task, inventory)
    if family in {"aligned_window_comparison", "filter_then_compute"}:
        return _execute_l3_window_family(task, inventory)
    if family == "heterogeneous_metric_comparison":
        return _execute_heterogeneous_metric_comparison(task, inventory)
    if level == "L1":
        return _execute_l1(task, inventory)
    if level == "L2":
        return _execute_l2(task, inventory)
    if level == "L3":
        return _execute_l3(task, inventory)
    raise GoldExecutionError(f"Unsupported task level: {level!r}")


def _answers_match(
    candidate: Any,
    replayed: Any,
    tolerances: Mapping[str, Any],
    *,
    field: str | None = None,
) -> bool:
    """Compare provisional and replayed answers using declared numeric tolerances."""

    if isinstance(candidate, Mapping) and isinstance(replayed, Mapping):
        return set(candidate) == set(replayed) and all(
            _answers_match(
                candidate[key],
                replayed[key],
                tolerances,
                field=str(key),
            )
            for key in candidate
        )
    if isinstance(candidate, list) and isinstance(replayed, list):
        return len(candidate) == len(replayed) and all(
            _answers_match(left, right, tolerances, field=field)
            for left, right in zip(candidate, replayed, strict=True)
        )
    if (
        isinstance(candidate, (int, float, Decimal))
        and not isinstance(candidate, bool)
        and isinstance(replayed, (int, float, Decimal))
        and not isinstance(replayed, bool)
    ):
        tolerance = Decimal(str(tolerances.get(field or "", 0)))
        return abs(Decimal(str(candidate)) - Decimal(str(replayed))) <= tolerance
    return candidate == replayed


def attach_gold_execution(
    tasks: Iterable[dict[str, Any]], inventory: CorpusInventory
) -> list[dict[str, Any]]:
    """Replace provisional candidate answers with independently replayed Gold outputs."""

    materialized: list[dict[str, Any]] = []
    for source in tasks:
        task = dict(source)
        provisional = task.get("answer")
        execution = execute_gold(task, inventory)
        if provisional is not None:
            normalized = provisional
            if task.get("task_level") == "L3" and isinstance(provisional, Mapping):
                per_device = provisional.get("per_device")
                if isinstance(per_device, Mapping):
                    normalized = dict(provisional)
                    normalized["per_device"] = [
                        {"alias": alias, **dict(per_device[alias])} for alias in _aliases(task)
                    ]
            if not _answers_match(
                normalized,
                execution.answer,
                task.get("tolerances", {}),
            ):
                raise GoldExecutionError(
                    f"Candidate answer disagrees with raw replay for {task.get('pair_id')}: "
                    f"candidate={normalized!r}, replay={execution.answer!r}"
                )
        task["answer"] = execution.answer
        task["gold_trace"] = execution.trace
        materialized.append(task)
    return materialized