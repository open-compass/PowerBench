from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from pathlib import Path, PurePosixPath
from typing import Any

from ..models import CorpusInventory, Device
from ..util import canonical_json
from .gold import execute_gold, validate_trace

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_TIMESTAMP_PATTERN = r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
_RANGE_RE = re.compile(
    rf"(?P<start>{_TIMESTAMP_PATTERN})\s*(?:~|to|—|–)\s*"
    rf"(?P<end>{_TIMESTAMP_PATTERN})"
)
_KEY_VALUE_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?:\*\*|__)(?P<key>.+?)(?:\*\*|__)"
    r"\s*[:：]\s*(?P<value>.*?)\s*$"
)
_RULE_ID_RE = re.compile(r"(?<![A-Za-z0-9])K\s*0*(\d+)(?!\d)", re.IGNORECASE)
_RULE_EXPRESSION_RE = re.compile(
    r"^\s*consec_(?P<hours>[1-9]\d*)h\s*\(\s*"
    r"(?P<field>.+?)\s*(?P<operator>[<>])\s*"
    r"(?P<parameter>[^\W\d]\w*)\s*\)\s*=\s*true\s*$",
    re.IGNORECASE,
)
_CHECK_EVENT_RE = re.compile(r"_(E\d+)_check_report\.md$", re.IGNORECASE)
_FAULT_EVENT_RE = re.compile(r"_(E\d+)_\d{14}_fault_report\.md$", re.IGNORECASE)
_ONE_HOUR = timedelta(hours=1)
_ALLOWED_LEVELS = {"L1", "L2", "L3"}
_FORBIDDEN_PATH_PARTS = {"gen_qa"}


class GoldVerificationError(ValueError):
    """Raised when a semantic task cannot be reproduced from the raw corpus."""


@dataclass(frozen=True)
class _IndependentExecution:
    """Answer and trace reconstructed without the Gold-producing implementation."""

    answer: dict[str, Any]
    trace: list[dict[str, Any]]


@dataclass(frozen=True)
class _ReplayAudit:
    """Primitive replay state needed to reconstruct every serialized trace step."""

    local_result: dict[str, Any]
    points: tuple[tuple[datetime, Decimal], ...]
    target_row_count: int
    satisfied_point_count: int
    trigger_timestamps: tuple[datetime, ...]


def _timestamp(value: Any, *, context: str) -> datetime:
    if not isinstance(value, str):
        raise GoldVerificationError(f"{context} must be a timestamp string")
    try:
        return datetime.strptime(value.strip(), _TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise GoldVerificationError(f"Invalid {context}: {value!r}") from exc


def _format_timestamp(value: datetime) -> str:
    return value.strftime(_TIMESTAMP_FORMAT)


def _decimal(value: Any, *, context: str) -> Decimal:
    if isinstance(value, bool):
        raise GoldVerificationError(f"{context} must be numeric, not boolean")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise GoldVerificationError(f"Invalid decimal {context}: {value!r}") from exc
    if not number.is_finite():
        raise GoldVerificationError(f"Non-finite decimal {context}: {value!r}")
    return number


def _json_object(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GoldVerificationError(f"{context} must be an object")
    return value


def _json_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GoldVerificationError(f"{context} must be a list")
    return value


def _trace_step(
    step_id: str,
    op: str,
    *,
    depends_on: Sequence[str] = (),
    inputs: Mapping[str, Any] | None = None,
    output: Any = None,
) -> dict[str, Any]:
    """Build the independently specified serialized form of one trace step."""

    return {
        "step_id": step_id,
        "op": op,
        "depends_on": list(depends_on),
        "inputs": dict(inputs or {}),
        "output": output,
    }


def _trace_locator(task: Mapping[str, Any], alias: str) -> Any:
    locators = task.get("locators")
    if locators is None:
        return None
    if not isinstance(locators, Mapping):
        raise GoldVerificationError("locators must be an object when present")
    return locators.get(alias)


def _trace_points_summary(points: Sequence[tuple[datetime, Decimal]]) -> dict[str, Any]:
    payload = [
        {"timestamp": _format_timestamp(timestamp), "value": format(value, "f")}
        for timestamp, value in points
    ]
    return {
        "row_count": len(payload),
        "first_timestamp": payload[0]["timestamp"] if payload else None,
        "last_timestamp": payload[-1]["timestamp"] if payload else None,
        "values_sha256": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
    }


class _RawVerifier:
    def __init__(self, inventory: CorpusInventory) -> None:
        self.inventory = inventory
        self.raw_root = inventory.raw_root.resolve()
        self._text_cache: dict[str, str] = {}
        self._json_cache: dict[str, Any] = {}
        self._series_cache: dict[tuple[str, str], dict[datetime, Decimal]] = {}

    def path(self, relative: Any, *, context: str) -> Path:
        if not isinstance(relative, str) or not relative.strip():
            raise GoldVerificationError(f"{context} must be a non-empty raw-relative path")
        normalized = relative.strip().replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise GoldVerificationError(f"Unsafe {context}: {relative!r}")
        if any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in pure.parts):
            raise GoldVerificationError(f"Forbidden {context}: {relative!r}")
        resolved = (self.raw_root / pure).resolve()
        try:
            resolved.relative_to(self.raw_root)
        except ValueError as exc:
            raise GoldVerificationError(f"{context} escapes raw root: {relative!r}") from exc
        if not resolved.is_file():
            raise GoldVerificationError(f"Missing {context}: {normalized}")
        return resolved

    def text(self, relative: str, *, context: str) -> str:
        if relative not in self._text_cache:
            self._text_cache[relative] = self.path(relative, context=context).read_text(
                encoding="utf-8-sig"
            )
        return self._text_cache[relative]

    def json(self, relative: str, *, context: str) -> Any:
        if relative not in self._json_cache:
            try:
                self._json_cache[relative] = json.loads(self.text(relative, context=context))
            except json.JSONDecodeError as exc:
                raise GoldVerificationError(f"Invalid JSON in {context}: {relative}") from exc
        return self._json_cache[relative]

    def document_fields(self, relative: str) -> tuple[dict[str, str], str]:
        text = self.text(relative, context="report path")
        fields: dict[str, str] = {}
        for line in text.splitlines():
            match = _KEY_VALUE_RE.match(line)
            if match:
                fields.setdefault(match.group("key").strip(), match.group("value").strip())
        return fields, text

    def series(self, device: Device, field: str) -> dict[datetime, Decimal]:
        key = (device.uid, field)
        if key in self._series_cache:
            return self._series_cache[key]
        source = self.path(device.csv_path, context=f"CSV for {device.uid}")
        values: dict[datetime, Decimal] = {}
        previous: datetime | None = None
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            missing = [column for column in ("timestamp", field) if column not in columns]
            if missing:
                raise GoldVerificationError(
                    f"{device.uid} CSV is missing column(s) {missing}: {device.csv_path}"
                )
            for line_number, row in enumerate(reader, start=2):
                current = _timestamp(
                    row.get("timestamp"), context=f"{device.csv_path}:{line_number} timestamp"
                )
                if previous is not None and current <= previous:
                    raise GoldVerificationError(
                        f"CSV timestamps are not strictly increasing at "
                        f"{device.csv_path}:{line_number}"
                    )
                previous = current
                values[current] = _decimal(
                    row.get(field),
                    context=f"{device.csv_path}:{line_number} column {field!r}",
                )
        self._series_cache[key] = values
        return values


def _device_for_alias(task: Mapping[str, Any], inventory: CorpusInventory, alias: str) -> Device:
    aliases = _json_object(task.get("device_aliases"), context="device_aliases")
    uid = aliases.get(alias)
    if not isinstance(uid, str):
        raise GoldVerificationError(f"Missing device alias {alias}")
    try:
        return inventory.device(uid)
    except KeyError as exc:
        raise GoldVerificationError(f"Unknown target device {uid!r}") from exc


def _report_for_alias(task: Mapping[str, Any], device: Device, alias: str, *, kind: str) -> str:
    report_paths = _json_object(task.get("report_paths"), context="report_paths")
    relative = report_paths.get(alias)
    if not isinstance(relative, str):
        raise GoldVerificationError(f"Missing report path for alias {alias}")
    allowed = device.check_reports if kind == "check" else device.fault_reports
    if relative not in allowed:
        raise GoldVerificationError(
            f"Report {relative!r} is not a {kind} report indexed for {device.uid}"
        )
    return relative


def _document_window(
    fields: Mapping[str, str], keys: tuple[str, ...], *, context: str
) -> tuple[datetime, datetime]:
    raw = next((fields[key] for key in keys if fields.get(key)), None)
    if raw is None:
        raise GoldVerificationError(f"No {context} found in report")
    match = _RANGE_RE.search(raw)
    if match is None:
        raise GoldVerificationError(f"Invalid {context}: {raw!r}")
    start = _timestamp(match.group("start"), context=f"{context} start")
    end = _timestamp(match.group("end"), context=f"{context} end")
    if end < start:
        raise GoldVerificationError(f"{context} ends before it starts")
    return start, end


def _event_id(relative: str, *, kind: str) -> str:
    pattern = _CHECK_EVENT_RE if kind == "check" else _FAULT_EVENT_RE
    match = pattern.search(PurePosixPath(relative).name)
    if match is None:
        raise GoldVerificationError(f"Cannot parse event ID from {relative}")
    return match.group(1).upper()


def _window_points(
    values: Mapping[datetime, Decimal],
    *,
    start: datetime,
    end: datetime,
    lookback_hours: int,
) -> list[tuple[datetime, Decimal]]:
    if isinstance(lookback_hours, bool) or not isinstance(lookback_hours, int):
        raise GoldVerificationError("lookback_hours must be an integer")
    if lookback_hours < 0:
        raise GoldVerificationError("lookback_hours must be non-negative")
    seconds = (end - start).total_seconds()
    if seconds <= 0 or seconds % 3600:
        raise GoldVerificationError(
            "Evaluation window must contain a whole positive number of hours"
        )
    hours = int(seconds // 3600)
    required = [start - offset * _ONE_HOUR for offset in range(lookback_hours, 0, -1)]
    required.extend(start + offset * _ONE_HOUR for offset in range(hours))
    missing = [value for value in required if value not in values]
    if missing:
        rendered = ", ".join(_format_timestamp(value) for value in missing[:3])
        raise GoldVerificationError(f"Hourly window or lookback is incomplete at {rendered}")
    return [(timestamp, values[timestamp]) for timestamp in required]


def _window_values(
    values: Mapping[datetime, Decimal],
    *,
    start: datetime,
    end: datetime,
    lookback_hours: int,
) -> list[Decimal]:
    points = _window_points(
        values,
        start=start,
        end=end,
        lookback_hours=lookback_hours,
    )
    return [value for _, value in points[lookback_hours:]]


def _round_decimal(value: Decimal, digits: int) -> float:
    quantum = Decimal(1).scaleb(-digits)
    with localcontext() as context:
        context.prec = max(64, len(value.as_tuple().digits) + digits + 8)
        rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    return float(Decimal(0) if rounded.is_zero() else rounded)


def _round_digits(task: Mapping[str, Any]) -> int:
    digits = task.get("answer_round_digits")
    if isinstance(digits, int) and not isinstance(digits, bool) and digits >= 0:
        return digits
    raise GoldVerificationError("L1 answer_round_digits must be a non-negative integer")


def _verify_target_uids(task: Mapping[str, Any], aliases: tuple[str, ...]) -> None:
    alias_map = _json_object(task.get("device_aliases"), context="device_aliases")
    expected = [alias_map.get(alias) for alias in aliases]
    targets = _json_list(task.get("target_device_uids"), context="target_device_uids")
    if targets != expected:
        raise GoldVerificationError(
            f"target_device_uids {targets!r} disagree with alias order {expected!r}"
        )
    if len(set(targets)) != len(targets):
        raise GoldVerificationError("Target devices must be distinct")


def _verify_l1(task: Mapping[str, Any], verifier: _RawVerifier) -> _IndependentExecution:
    _verify_target_uids(task, ("A",))
    device = _device_for_alias(task, verifier.inventory, "A")
    report = _report_for_alias(task, device, "A", kind="check")
    fields, _ = verifier.document_fields(report)
    report_start, report_end = _document_window(
        fields,
        ("Inspection window",),
        context="inspection window",
    )
    anchor = _json_object(task.get("anchor"), context="L1 anchor")
    if anchor.get("event_id") != _event_id(report, kind="check"):
        raise GoldVerificationError("L1 report event ID disagrees with task anchor")
    if _timestamp(anchor.get("inspection_start"), context="inspection_start") != report_start:
        raise GoldVerificationError("L1 inspection start disagrees with the report")
    if _timestamp(anchor.get("inspection_end"), context="inspection_end") != report_end:
        raise GoldVerificationError("L1 inspection end disagrees with the report")

    metric = _json_object(task.get("metric"), context="L1 metric")
    field = metric.get("field")
    if not isinstance(field, str) or field not in device.csv_columns:
        raise GoldVerificationError(f"Invalid L1 CSV metric field: {field!r}")
    metric_path = _model_relative_path(
        metric.get("metric_catalog_path"), device.model_key, context="L1 metric catalog path"
    )
    _, metric_name, metric_unit, _ = _catalog_metric(verifier, metric_path=metric_path, field=field)
    if str(metric.get("name") or "") != metric_name:
        raise GoldVerificationError("L1 metric name disagrees with the metric catalog")
    if str(metric.get("unit") or "") != metric_unit:
        raise GoldVerificationError("L1 metric unit disagrees with the metric catalog")
    window = _json_object(task.get("window"), context="L1 window")
    start = _timestamp(window.get("start"), context="L1 window start")
    end = _timestamp(window.get("end_exclusive"), context="L1 window end")
    hours = window.get("hours")
    if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0:
        raise GoldVerificationError("L1 window hours must be a positive integer")
    if _timestamp(window.get("anchor_end"), context="L1 anchor_end") != report_end:
        raise GoldVerificationError("L1 anchor_end disagrees with the report")
    if start != report_end + _ONE_HOUR or end != start + hours * _ONE_HOUR:
        raise GoldVerificationError(
            "L1 must use the W hourly points with t_end < timestamp <= t_end + W hours"
        )

    values = verifier.series(device, field)
    target_points = _window_points(values, start=start, end=end, lookback_hours=0)
    target_values = [value for _, value in target_points]
    if report_end not in values:
        raise GoldVerificationError("L1 CSV has no value at the inspection end timestamp")
    anchor_value = values[report_end]
    digits = _round_digits(task)
    stored_anchor = _decimal(anchor.get("anchor_value"), context="stored L1 anchor value")
    if abs(stored_anchor - anchor_value) > Decimal(1).scaleb(-(digits + 2)):
        raise GoldVerificationError("Stored L1 anchor value disagrees with the CSV")

    with localcontext() as context:
        context.prec = 64
        window_mean = sum(target_values, Decimal(0)) / Decimal(len(target_values))
    # The serialized operation output follows Decimal's specified default
    # precision used by the producer; the final answer is rounded separately.
    with localcontext() as context:
        context.prec = 28
        trace_mean = sum(target_values, Decimal(0)) / Decimal(len(target_values))
    maximum = max(target_values)
    first_max_offset = target_values.index(maximum)
    first_max = start + first_max_offset * _ONE_HOUR
    count_above = sum(value > anchor_value for value in target_values)
    answer = {
        "window_mean": _round_decimal(window_mean, digits),
        "window_max": _round_decimal(maximum, digits),
        "first_max_time": _format_timestamp(first_max),
        "hours_above_anchor_value": count_above,
        "unit": metric_unit,
    }
    trace = [
        _trace_step(
            "resolve_device",
            "RESOLVE_DEVICE",
            inputs={"alias": "A", "locator": _trace_locator(task, "A")},
            output={"device_uid": device.uid, "model_key": device.model_key},
        ),
        _trace_step(
            "read_report",
            "READ_INSPECTION_REPORT",
            depends_on=("resolve_device",),
            inputs={"path": report, "event_id": _event_id(report, kind="check")},
            output={
                "inspection_start": _format_timestamp(report_start),
                "inspection_end": _format_timestamp(report_end),
            },
        ),
        _trace_step(
            "resolve_metric",
            "RESOLVE_METRIC",
            depends_on=("resolve_device",),
            inputs={"catalog_path": metric.get("metric_catalog_path"), "field": field},
            output={"field": field, "name": metric_name, "unit": metric_unit},
        ),
        _trace_step(
            "read_anchor",
            "READ_TIMESERIES",
            depends_on=("read_report", "resolve_metric"),
            inputs={
                "path": device.csv_path,
                "field": field,
                "start": _format_timestamp(report_end),
                "end_exclusive": _format_timestamp(start),
            },
            output={
                **_trace_points_summary(((report_end, anchor_value),)),
                "value": format(anchor_value, "f"),
            },
        ),
        _trace_step(
            "read_window",
            "READ_TIMESERIES",
            depends_on=("read_report", "resolve_metric"),
            inputs={
                "path": device.csv_path,
                "field": field,
                "start": _format_timestamp(start),
                "end_exclusive": _format_timestamp(end),
            },
            output=_trace_points_summary(target_points),
        ),
        _trace_step(
            "mean",
            "MEAN",
            depends_on=("read_window",),
            inputs={"row_count": len(target_values)},
            output={"value": format(trace_mean, "f")},
        ),
        _trace_step(
            "max",
            "MAX_AND_FIRST_ARGMAX",
            depends_on=("read_window",),
            output={"value": format(maximum, "f"), "first_time": _format_timestamp(first_max)},
        ),
        _trace_step(
            "count_above",
            "COUNT_STRICTLY_ABOVE",
            depends_on=("read_anchor", "read_window"),
            inputs={"threshold": format(anchor_value, "f")},
            output={"count": count_above},
        ),
        _trace_step(
            "answer",
            "FORMAT_ANSWER",
            depends_on=("mean", "max", "count_above"),
            inputs={"round_digits": digits},
            output=answer,
        ),
    ]
    return _IndependentExecution(answer=answer, trace=trace)


def _model_relative_path(relative: Any, model_key: str, *, context: str) -> str:
    if not isinstance(relative, str):
        raise GoldVerificationError(f"{context} must be a path")
    parts = PurePosixPath(relative).parts
    if not parts or parts[0] != model_key:
        raise GoldVerificationError(f"{context} does not belong to model {model_key}: {relative}")
    return relative


def _metric_rows(payload: Any, *, context: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("metrics") or payload.get("metric_catalog") or payload.get("metrics")
    else:
        rows = None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise GoldVerificationError(f"Unsupported {context} structure")
    return rows


def _catalog_metric(
    verifier: _RawVerifier, *, metric_path: str, field: str
) -> tuple[Mapping[str, Any], str, str, str]:
    metrics = _metric_rows(
        verifier.json(metric_path, context="metric catalog"), context="metric catalog"
    )
    matches = [
        row
        for row in metrics
        if str(row.get("database_field") or row.get("variable_name") or "").strip() == field
    ]
    if len(matches) != 1:
        raise GoldVerificationError(
            f"Expected one public metric definition for {field!r}, found {len(matches)}"
        )
    metric = matches[0]
    metric_name = str(metric.get("metric_name") or field).strip()
    normal = metric.get("normal_range") if isinstance(metric.get("normal_range"), Mapping) else {}
    unit = str(metric.get("unit") or normal.get("unit") or "").strip()
    data_type = str(metric.get("data_type") or "").strip().lower()
    if data_type not in {"float", "double", "number", "numeric"}:
        raise GoldVerificationError(f"Metric {field!r} is not declared numeric")
    return metric, metric_name, unit, str(metric.get("metric_type") or "").strip()


def _compiled_rule(
    task: Mapping[str, Any], verifier: _RawVerifier, devices: tuple[Device, ...]
) -> dict[str, Any]:
    rule = _json_object(task.get("rule"), context="rule")
    model_keys = {device.model_key for device in devices}
    if len(model_keys) != 1:
        raise GoldVerificationError(
            "A shared replay rule requires all target devices to share a model"
        )
    model_key = next(iter(model_keys))
    rule_id = rule.get("rule_id")
    if not isinstance(rule_id, str) or not re.fullmatch(r"K[1-9]\d*", rule_id):
        raise GoldVerificationError(f"Invalid rule ID: {rule_id!r}")

    rule_path = _model_relative_path(rule.get("rule_path"), model_key, context="rule path")
    payload = verifier.json(rule_path, context="rule catalog")
    rows = payload.get("anomaly_judgment_rules") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise GoldVerificationError(f"Unsupported rule catalog structure: {rule_path}")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("rule_id") or "").strip().upper() == rule_id
    ]
    if len(matches) != 1:
        raise GoldVerificationError(
            f"Expected one {rule_id} row in {rule_path}, found {len(matches)}"
        )
    source_rule = matches[0]
    expression = str(source_rule.get("rule_expression") or "").strip()
    if expression != rule.get("expression"):
        raise GoldVerificationError("Task rule expression disagrees with the rule catalog")
    match = _RULE_EXPRESSION_RE.fullmatch(expression)
    if match is None:
        raise GoldVerificationError(
            "Rule is outside consec_Hh(single_metric >/< single_parameter) = true"
        )
    source_field = match.group("field").strip()
    if any(
        token in f" {source_field.upper()} "
        for token in (" AND ", " OR ", "+", "*", "=", "!", "<", ">")
    ):
        raise GoldVerificationError(f"Rule contains a compound field expression: {source_field!r}")

    parameters = source_rule.get("threshold_params")
    if (
        not isinstance(parameters, list)
        or len(parameters) != 1
        or not isinstance(parameters[0], Mapping)
    ):
        raise GoldVerificationError(f"Rule {rule_id} must have exactly one numeric parameter")
    source_parameter = parameters[0]
    parameter_name = str(source_parameter.get("param_name") or "").strip()
    original_threshold = _decimal(
        source_parameter.get("param_value"), context=f"{rule_id} original threshold"
    )
    if parameter_name != match.group("parameter") or parameter_name != rule.get("parameter_name"):
        raise GoldVerificationError("Rule parameter binding is inconsistent")
    if original_threshold != _decimal(
        rule.get("original_threshold"), context="task original threshold"
    ):
        raise GoldVerificationError("Task original threshold disagrees with the rule catalog")

    field = rule.get("metric_field")
    if not isinstance(field, str) or any(field not in device.csv_columns for device in devices):
        raise GoldVerificationError(f"Rule field {field!r} is not materialized in every target CSV")
    metric_path = _model_relative_path(
        rule.get("metric_catalog_path"), model_key, context="metric catalog path"
    )
    _, metric_name, unit, metric_type = _catalog_metric(
        verifier, metric_path=metric_path, field=field
    )
    if source_field not in {field, metric_name}:
        raise GoldVerificationError(
            f"Rule source field {source_field!r} does not bind to CSV field {field!r}"
        )
    if str(rule.get("metric_name") or "") != metric_name:
        raise GoldVerificationError("Task metric name disagrees with the metric catalog")
    if str(rule.get("unit") or "") != unit:
        raise GoldVerificationError("Task metric unit disagrees with the metric catalog")

    hours = int(match.group("hours"))
    if rule.get("consecutive_hours") != hours or rule.get("operator") != match.group("operator"):
        raise GoldVerificationError("Task consecutive-rule metadata disagrees with the expression")
    point_requirement = source_rule.get("pointwise_criteria")
    if not isinstance(point_requirement, Mapping):
        raise GoldVerificationError("Rule has no structured point-trigger requirement")
    stated_points = _decimal(
        point_requirement.get("consecutive_points"), context="stated consecutive point count"
    )
    if stated_points != stated_points.to_integral_value() or int(stated_points) != hours:
        raise GoldVerificationError("Rule expression and stated consecutive point count disagree")
    single_point = point_requirement.get("single_point_trigger")
    if not isinstance(single_point, bool):
        raise GoldVerificationError("Rule single-point trigger flag must be boolean")
    if single_point is not (hours == 1):
        raise GoldVerificationError("Rule expression and stated single-point trigger flag disagree")

    threshold_text = rule.get("counterfactual_threshold_text")
    counterfactual = _decimal(threshold_text, context="counterfactual threshold text")
    if counterfactual != _decimal(
        rule.get("counterfactual_threshold"), context="counterfactual threshold"
    ):
        raise GoldVerificationError("Counterfactual threshold text and numeric value disagree")
    if counterfactual == original_threshold:
        raise GoldVerificationError("Counterfactual threshold must differ from the public original")
    derived_path = rule.get("derived_metrics_path")
    if "derived" in metric_type:
        if derived_path is None:
            raise GoldVerificationError(f"Derived metric {field!r} has no public definition path")
        relative = _model_relative_path(derived_path, model_key, context="derived metric path")
        derived_payload = verifier.json(relative, context="derived metric definition")
        derived_rows = (
            derived_payload.get("derived_metrics") if isinstance(derived_payload, Mapping) else None
        )
        if not isinstance(derived_rows, list):
            raise GoldVerificationError(f"Unsupported derived metric structure: {relative}")
        matches = [
            row
            for row in derived_rows
            if isinstance(row, Mapping) and str(row.get("derived_key") or "") == field
        ]
        if len(matches) != 1:
            raise GoldVerificationError(
                f"Expected one derived definition for {field!r}, found {len(matches)}"
            )
    elif derived_path is not None:
        raise GoldVerificationError(f"Raw metric {field!r} unexpectedly declares a derived path")
    return {
        "rule_id": rule_id,
        "field": field,
        "operator": match.group("operator"),
        "hours": hours,
        "original_threshold": original_threshold,
        "counterfactual_threshold": counterfactual,
    }


def _verify_fault_anchor(
    task: Mapping[str, Any], verifier: _RawVerifier, device: Device, alias: str, rule_id: str
) -> datetime:
    report = _report_for_alias(task, device, alias, kind="fault")
    fields, _ = verifier.document_fields(report)
    associated = []
    for match in _RULE_ID_RE.finditer(fields.get("Linked rule", "")):
        normalized = f"K{int(match.group(1))}"
        if normalized not in associated:
            associated.append(normalized)
    if associated != [rule_id]:
        raise GoldVerificationError(
            f"{alias} report must bind exactly one rule {rule_id}; found {associated}"
        )
    _, fault_end = _document_window(
        fields, ("Fault window",), context="fault window"
    )
    anchors = _json_object(task.get("anchors"), context="fault anchors")
    anchor = _json_object(anchors.get(alias), context=f"fault anchor {alias}")
    if anchor.get("event_id") != _event_id(report, kind="fault"):
        raise GoldVerificationError(f"{alias} report event ID disagrees with task anchor")
    if _timestamp(anchor.get("fault_end"), context=f"{alias} fault_end") != fault_end:
        raise GoldVerificationError(f"{alias} fault_end disagrees with the report")
    discovery = str(fields.get("Discovery time") or "").strip()
    stored_discovery = str(anchor.get("discovery_time") or "").strip()
    if stored_discovery and discovery != stored_discovery:
        raise GoldVerificationError(f"{alias} discovery time disagrees with the report")
    return fault_end


def _replay_audit(
    values: Mapping[datetime, Decimal],
    *,
    start: datetime,
    end: datetime,
    hours: int,
    operator: str,
    threshold: Decimal,
) -> _ReplayAudit:
    lookback_hours = hours - 1
    points = _window_points(
        values,
        start=start,
        end=end,
        lookback_hours=lookback_hours,
    )
    targets = [value for _, value in points[lookback_hours:]]
    trigger_times: list[datetime] = []
    best_margin: Decimal | None = None
    for index in range(len(targets)):
        target = start + index * _ONE_HOUR
        trailing = [values[target - offset * _ONE_HOUR] for offset in range(hours)]
        if operator == ">":
            margin = min(value - threshold for value in trailing)
        elif operator == "<":
            margin = min(threshold - value for value in trailing)
        else:  # pragma: no cover - the parser rejects this first.
            raise GoldVerificationError(f"Unsupported rule operator {operator!r}")
        if best_margin is None or margin > best_margin:
            best_margin = margin
        if margin > 0:
            trigger_times.append(target)
    if best_margin is None:  # pragma: no cover - _window_values requires a non-empty window.
        raise GoldVerificationError("Replay window contains no target rows")
    first = trigger_times[0] if trigger_times else None
    local_result = {
        "triggered": bool(trigger_times),
        "first_trigger_time": _format_timestamp(first) if first else None,
        "first_trigger_offset_hours": (
            int((first - start).total_seconds() // 3600) if first else None
        ),
        "trigger_hours": len(trigger_times),
        "best_consecutive_margin": _round_decimal(best_margin, 4),
    }
    satisfied = sum(
        (value > threshold if operator == ">" else value < threshold) for _, value in points
    )
    return _ReplayAudit(
        local_result=local_result,
        points=tuple(points),
        target_row_count=len(targets),
        satisfied_point_count=satisfied,
        trigger_timestamps=tuple(trigger_times),
    )


def _replay(
    values: Mapping[datetime, Decimal],
    *,
    start: datetime,
    end: datetime,
    hours: int,
    operator: str,
    threshold: Decimal,
) -> dict[str, Any]:
    return _replay_audit(
        values,
        start=start,
        end=end,
        hours=hours,
        operator=operator,
        threshold=threshold,
    ).local_result


def _window_for_replay(
    raw_window: Any, *, fault_end: datetime, context: str
) -> tuple[datetime, datetime, int]:
    window = _json_object(raw_window, context=context)
    start = _timestamp(window.get("start"), context=f"{context} start")
    end = _timestamp(window.get("end_exclusive"), context=f"{context} end")
    hours = window.get("hours")
    if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0:
        raise GoldVerificationError(f"{context} hours must be a positive integer")
    if start != fault_end + 24 * _ONE_HOUR or end != start + hours * _ONE_HOUR:
        raise GoldVerificationError(f"{context} must be [fault_end + 24h, fault_end + 24h + W)")
    return start, end, hours


def _shared_rule_trace(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    rule = _json_object(task.get("rule"), context="rule")
    return [
        _trace_step(
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
        ),
        _trace_step(
            "override_threshold",
            "OVERRIDE_PARAMETER",
            depends_on=("load_rule",),
            inputs={
                "parameter_name": rule.get("parameter_name"),
                "counterfactual_threshold": rule.get("counterfactual_threshold"),
            },
            output={"effective_threshold": rule.get("counterfactual_threshold")},
        ),
    ]


def _local_replay_trace(
    task: Mapping[str, Any],
    *,
    alias: str,
    device: Device,
    report: str,
    fault_end: datetime,
    start: datetime,
    end: datetime,
    rule: Mapping[str, Any],
    replay: _ReplayAudit,
) -> list[dict[str, Any]]:
    prefix = alias.casefold()
    resolve_id = f"{prefix}_resolve"
    report_id = f"{prefix}_report"
    window_id = f"{prefix}_window"
    compare_id = f"{prefix}_compare"
    consecutive_id = f"{prefix}_consecutive"
    local_id = f"{prefix}_local"
    threshold = _decimal(
        _json_object(task.get("rule"), context="rule").get("counterfactual_threshold"),
        context="counterfactual threshold",
    )
    return [
        _trace_step(
            resolve_id,
            "RESOLVE_DEVICE",
            inputs={"alias": alias, "locator": _trace_locator(task, alias)},
            output={"device_uid": device.uid, "model_key": device.model_key},
        ),
        _trace_step(
            report_id,
            "READ_FAULT_REPORT",
            depends_on=(resolve_id,),
            inputs={"path": report, "event_id": _event_id(report, kind="fault")},
            output={"fault_end": _format_timestamp(fault_end), "rule_id": rule["rule_id"]},
        ),
        _trace_step(
            window_id,
            "READ_TIMESERIES",
            depends_on=(report_id, "override_threshold"),
            inputs={
                "path": device.csv_path,
                "field": rule["field"],
                "start": _format_timestamp(start),
                "end_exclusive": _format_timestamp(end),
                "lookback_hours": int(rule["hours"]) - 1,
            },
            output={
                **_trace_points_summary(replay.points),
                "target_row_count": replay.target_row_count,
            },
        ),
        _trace_step(
            compare_id,
            "ATOMIC_COMPARE",
            depends_on=(window_id, "override_threshold"),
            inputs={"operator": rule["operator"], "threshold": format(threshold, "f")},
            output={
                "satisfied_point_count": replay.satisfied_point_count,
                "evaluated_point_count": len(replay.points),
            },
        ),
        _trace_step(
            consecutive_id,
            "APPLY_CONSECUTIVE",
            depends_on=(compare_id,),
            inputs={"consecutive_hours": rule["hours"]},
            output={
                "trigger_timestamps": [
                    _format_timestamp(timestamp) for timestamp in replay.trigger_timestamps
                ],
                "trigger_hours": replay.local_result["trigger_hours"],
            },
        ),
        _trace_step(
            local_id,
            "REDUCE_LOCAL_REPLAY",
            depends_on=(consecutive_id,),
            output=replay.local_result,
        ),
    ]


def _verify_l2(task: Mapping[str, Any], verifier: _RawVerifier) -> _IndependentExecution:
    _verify_target_uids(task, ("A",))
    device = _device_for_alias(task, verifier.inventory, "A")
    rule = _compiled_rule(task, verifier, (device,))
    report = _report_for_alias(task, device, "A", kind="fault")
    fault_end = _verify_fault_anchor(task, verifier, device, "A", rule["rule_id"])
    start, end, _ = _window_for_replay(task.get("window"), fault_end=fault_end, context="L2 window")
    values = verifier.series(device, rule["field"])
    original = _replay(
        values,
        start=start,
        end=end,
        hours=rule["hours"],
        operator=rule["operator"],
        threshold=rule["original_threshold"],
    )
    counterfactual_audit = _replay_audit(
        values,
        start=start,
        end=end,
        hours=rule["hours"],
        operator=rule["operator"],
        threshold=rule["counterfactual_threshold"],
    )
    counterfactual = counterfactual_audit.local_result
    diagnostics = _json_object(task.get("diagnostics"), context="L2 diagnostics")
    if diagnostics.get("original_result") != original:
        raise GoldVerificationError("Stored L2 original replay result is not reproducible")
    replay_fields = ("triggered", "first_trigger_time", "trigger_hours")
    changed = any(original[field] != counterfactual[field] for field in replay_fields)
    if bool(diagnostics.get("counterfactual_changed_output")) != changed:
        raise GoldVerificationError("Stored L2 intervention-effect diagnostic is incorrect")
    answer = {
        key: counterfactual[key]
        for key in (
            "triggered",
            "first_trigger_time",
            "trigger_hours",
            "best_consecutive_margin",
        )
    }
    trace = _shared_rule_trace(task)
    trace.extend(
        _local_replay_trace(
            task,
            alias="A",
            device=device,
            report=report,
            fault_end=fault_end,
            start=start,
            end=end,
            rule=rule,
            replay=counterfactual_audit,
        )
    )
    trace.append(
        _trace_step(
            "answer",
            "FORMAT_ANSWER",
            depends_on=("a_local",),
            output=answer,
        )
    )
    return _IndependentExecution(answer=answer, trace=trace)


def _verify_l3(task: Mapping[str, Any], verifier: _RawVerifier) -> _IndependentExecution:
    alias_map = _json_object(task.get("device_aliases"), context="device_aliases")
    aliases = tuple(str(alias) for alias in alias_map)
    if not 3 <= len(aliases) <= 5:
        raise GoldVerificationError("L3 must contain between three and five device aliases")
    if aliases != tuple(f"D{index}" for index in range(1, len(aliases) + 1)):
        raise GoldVerificationError("L3 aliases must be contiguous D1...Dk values")
    _verify_target_uids(task, aliases)
    devices = tuple(_device_for_alias(task, verifier.inventory, alias) for alias in aliases)
    rule = _compiled_rule(task, verifier, devices)
    task_window = _json_object(task.get("window"), context="L3 window")
    window_hours = task_window.get("hours")
    if isinstance(window_hours, bool) or not isinstance(window_hours, int) or window_hours <= 0:
        raise GoldVerificationError("L3 window hours must be a positive integer")
    per_device_windows = _json_object(
        task_window.get("per_device"), context="L3 per-device windows"
    )

    per_device: dict[str, dict[str, Any]] = {}
    original_results: dict[str, dict[str, Any]] = {}
    trace = _shared_rule_trace(task)
    for alias, device in zip(aliases, devices, strict=True):
        report = _report_for_alias(task, device, alias, kind="fault")
        fault_end = _verify_fault_anchor(task, verifier, device, alias, rule["rule_id"])
        raw_window = _json_object(per_device_windows.get(alias), context=f"L3 window for {alias}")
        local_window = {**raw_window, "hours": window_hours}
        start, end, hours = _window_for_replay(
            local_window, fault_end=fault_end, context=f"L3 window for {alias}"
        )
        values = verifier.series(device, rule["field"])
        original_results[alias] = _replay(
            values,
            start=start,
            end=end,
            hours=rule["hours"],
            operator=rule["operator"],
            threshold=rule["original_threshold"],
        )
        replay = _replay_audit(
            values,
            start=start,
            end=end,
            hours=rule["hours"],
            operator=rule["operator"],
            threshold=rule["counterfactual_threshold"],
        )
        per_device[alias] = replay.local_result
        trace.extend(
            _local_replay_trace(
                task,
                alias=alias,
                device=device,
                report=report,
                fault_end=fault_end,
                start=start,
                end=end,
                rule=rule,
                replay=replay,
            )
        )
        if hours != window_hours:  # Defensive against future contract changes.
            raise GoldVerificationError(f"L3 {alias} window length is inconsistent")

    triggered_aliases = [alias for alias in aliases if per_device[alias]["triggered"]]
    if triggered_aliases:
        offsets = {
            alias: int(per_device[alias]["first_trigger_offset_hours"])
            for alias in triggered_aliases
        }
        minimum = min(offsets.values())
        earliest = [alias for alias, offset in offsets.items() if offset == minimum]
        if len(earliest) != 1:
            raise GoldVerificationError(f"L3 earliest-trigger reduction is tied: {earliest}")
        earliest_alias: str | None = earliest[0]
        span: int | None = max(offsets.values()) - min(offsets.values())
    else:
        earliest_alias = None
        span = None

    diagnostics = _json_object(task.get("diagnostics"), context="L3 diagnostics")
    if diagnostics.get("original_results") != original_results:
        raise GoldVerificationError("Stored L3 original replay results are not reproducible")
    replay_fields = ("triggered", "first_trigger_time", "trigger_hours")
    changed_branches = sum(
        any(per_device[alias][field] != original_results[alias][field] for field in replay_fields)
        for alias in aliases
    )
    if diagnostics.get("changed_branches") != changed_branches:
        raise GoldVerificationError("Stored L3 changed_branches diagnostic is incorrect")
    if diagnostics.get("triggered_branch_count") != len(triggered_aliases):
        raise GoldVerificationError("Stored L3 triggered_branch_count diagnostic is incorrect")
    margins = {
        alias: _decimal(
            per_device[alias]["best_consecutive_margin"],
            context=f"{alias} best consecutive margin",
        )
        for alias in aliases
    }
    if len(set(margins.values())) != len(margins):
        raise GoldVerificationError("L3 best-consecutive-margin ordering is tied")
    margin_order = sorted(aliases, key=lambda alias: margins[alias], reverse=True)
    margin_span = _round_decimal(max(margins.values()) - min(margins.values()), 4)
    answer = {
        "per_device": [{"alias": alias, **per_device[alias]} for alias in aliases],
        "triggered_devices": triggered_aliases,
        "earliest_trigger_device": earliest_alias,
        "trigger_offset_span_hours": span,
        "margin_order": margin_order,
        "margin_span": margin_span,
    }
    local_dependencies = tuple(f"{alias.casefold()}_local" for alias in aliases)
    trace.extend(
        [
            _trace_step(
                "triggered_devices",
                "FILTER_TRIGGERED_DEVICES",
                depends_on=local_dependencies,
                output={"aliases": triggered_aliases},
            ),
            _trace_step(
                "earliest_trigger",
                "ARGMIN_FIRST_TRIGGER_OFFSET",
                depends_on=("triggered_devices", *local_dependencies),
                output={"alias": earliest_alias},
            ),
            _trace_step(
                "trigger_offset_span",
                "RANGE_FIRST_TRIGGER_OFFSETS",
                depends_on=("triggered_devices", *local_dependencies),
                output={"hours": span},
            ),
            _trace_step(
                "margin_order",
                "ORDER_BEST_CONSECUTIVE_MARGINS",
                depends_on=local_dependencies,
                output={"aliases": margin_order},
            ),
            _trace_step(
                "margin_span",
                "RANGE_BEST_CONSECUTIVE_MARGINS",
                depends_on=local_dependencies,
                output={"value": margin_span},
            ),
            _trace_step(
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
    return _IndependentExecution(answer=answer, trace=trace)


def _tolerance_for(path: str, tolerances: Mapping[str, Any]) -> Decimal:
    raw = tolerances.get(path, tolerances.get(path.rsplit(".", 1)[-1], 0))
    tolerance = _decimal(raw, context=f"tolerance for {path}")
    if tolerance < 0:
        raise GoldVerificationError(f"Negative tolerance for {path}")
    return tolerance


def _answer_mismatches(
    expected: Any,
    actual: Any,
    tolerances: Mapping[str, Any],
    *,
    path: str = "answer",
) -> list[str]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path}: expected an object, recomputed {type(actual).__name__}"]
        errors: list[str] = []
        if set(expected) != set(actual):
            errors.append(
                f"{path}: field sets differ (stored={sorted(expected)}, recomputed={sorted(actual)})"
            )
        for key in sorted(set(expected).intersection(actual)):
            errors.extend(
                _answer_mismatches(expected[key], actual[key], tolerances, path=f"{path}.{key}")
            )
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected a list, recomputed {type(actual).__name__}"]
        if path.endswith("triggered_devices"):
            if len(expected) == len(actual) and set(expected) == set(actual):
                return []
            return [f"{path}: stored={expected!r}, recomputed={actual!r}"]
        if len(expected) != len(actual):
            return [f"{path}: list lengths differ ({len(expected)} != {len(actual)})"]
        errors = []
        for index, (stored, recomputed) in enumerate(zip(expected, actual, strict=True)):
            errors.extend(
                _answer_mismatches(stored, recomputed, tolerances, path=f"{path}[{index}]")
            )
        return errors
    if isinstance(expected, bool):
        equal = isinstance(actual, bool) and expected is actual
    elif isinstance(expected, (int, float, Decimal)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float, Decimal)) or isinstance(actual, bool):
            equal = False
        else:
            stored = _decimal(expected, context=f"stored {path}")
            recomputed = _decimal(actual, context=f"recomputed {path}")
            equal = abs(stored - recomputed) <= _tolerance_for(
                path.removeprefix("answer."), tolerances
            )
    else:
        equal = expected == actual
    return [] if equal else [f"{path}: stored={expected!r}, recomputed={actual!r}"]


def _first_structure_mismatch(stored: Any, expected: Any, *, path: str) -> str | None:
    """Return the first exact JSON-structure mismatch, including scalar types."""

    if isinstance(expected, Mapping):
        if not isinstance(stored, Mapping):
            return f"{path}: expected object, stored {type(stored).__name__}"
        stored_keys = set(stored)
        expected_keys = set(expected)
        if stored_keys != expected_keys:
            return (
                f"{path}: field sets differ "
                f"(stored={sorted(map(str, stored_keys))}, "
                f"expected={sorted(map(str, expected_keys))})"
            )
        for key in expected:
            mismatch = _first_structure_mismatch(stored[key], expected[key], path=f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if not isinstance(stored, list):
            return f"{path}: expected list, stored {type(stored).__name__}"
        if len(stored) != len(expected):
            return f"{path}: list lengths differ ({len(stored)} != {len(expected)})"
        for index, (stored_item, expected_item) in enumerate(zip(stored, expected, strict=True)):
            mismatch = _first_structure_mismatch(
                stored_item,
                expected_item,
                path=f"{path}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None
    if type(stored) is not type(expected) or stored != expected:
        return f"{path}: stored={stored!r}, expected={expected!r}"
    return None


def verify_semantic_tasks(
    tasks: Iterable[Mapping[str, Any]], inventory: CorpusInventory
) -> dict[str, Any]:
    """Independently reproduce semantic-task answers from the frozen raw corpus.

    Legacy families are reproduced by the original independent parser. New
    family-tagged tasks are re-executed by their family-aware deterministic
    executor; their provisional candidate answers were already checked against
    this execution before publication. A task counts in ``verified_by_level``
    only when its answer and trace reproduce successfully.
    """

    verifier = _RawVerifier(inventory)
    verified: Counter[str] = Counter()
    errors: list[str] = []
    seen_pair_ids: set[str] = set()
    for index, raw_task in enumerate(tasks, start=1):
        if not isinstance(raw_task, Mapping):
            errors.append(f"task[{index}]: semantic task must be an object")
            continue
        pair_id = str(raw_task.get("pair_id") or f"task[{index}]")
        level = str(raw_task.get("task_level") or "")
        try:
            if pair_id in seen_pair_ids:
                raise GoldVerificationError(f"Duplicate semantic pair_id {pair_id}")
            seen_pair_ids.add(pair_id)
            if level not in _ALLOWED_LEVELS:
                raise GoldVerificationError(f"Unsupported task level {level!r}")
            execution = (
                execute_gold(raw_task, inventory)
                if raw_task.get("task_family")
                else {
                    "L1": _verify_l1,
                    "L2": _verify_l2,
                    "L3": _verify_l3,
                }[level](raw_task, verifier)
            )
            tolerances = _json_object(
                raw_task.get("tolerances", {}), context=f"{pair_id} tolerances"
            )
            mismatches = _answer_mismatches(raw_task.get("answer"), execution.answer, tolerances)
            if mismatches:
                raise GoldVerificationError("; ".join(mismatches))
            trace = _json_list(raw_task.get("gold_trace"), context=f"{pair_id} gold_trace")
            if not all(isinstance(step, Mapping) for step in trace):
                raise GoldVerificationError("Gold trace steps must be objects")
            validate_trace(trace)
            validate_trace(execution.trace)
            trace_mismatch = _first_structure_mismatch(
                trace,
                execution.trace,
                path="gold_trace",
            )
            if trace_mismatch is not None:
                raise GoldVerificationError(
                    "Gold trace disagrees with the independently reconstructed trace: "
                    f"{trace_mismatch}"
                )
            verified[level] += 1
        except (GoldVerificationError, OSError, UnicodeError, ValueError, TypeError) as exc:
            errors.append(f"{pair_id}: {type(exc).__name__}: {exc}")

    return {
        "valid": not errors,
        "errors": errors,
        "verified_by_level": {level: verified.get(level, 0) for level in sorted(_ALLOWED_LEVELS)},
    }