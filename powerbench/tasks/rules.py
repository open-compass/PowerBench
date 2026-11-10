from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from pathlib import Path

from .inventory import load_metrics
from .models import CorpusInventory, Rule
from .models import Rule as InventoryRule
from .timeseries import (
    IncompleteHourlySeriesError,
    NumericWindow,
    decimal_value,
    validate_complete_hourly_window,
)
from .util import finite_float, format_timestamp, relative_posix

_CONSECUTIVE_EXPRESSION = re.compile(
    r"^\s*consec_(?P<hours>[1-9]\d*)h\s*\(\s*"
    r"(?P<field>.+?)\s*(?P<operator>[<>])\s*"
    r"(?P<parameter>[^\W\d]\w*)\s*\)\s*=\s*true\s*$",
    flags=re.IGNORECASE,
)


class RuleError(ValueError):
    """Base class for public-semantics rule errors."""


class UnsupportedRuleExpression(RuleError):
    """Raised when an expression is outside the audited pilot grammar."""


class RuleBindingError(RuleError):
    """Raised when an expression cannot be bound to its public metric metadata."""


class ThresholdError(RuleError):
    """Raised when a threshold or threshold factor is invalid."""


@dataclass(frozen=True)
class ConsecutiveRule:
    """Compiled form of the audited single-field, single-threshold grammar."""

    expression: str
    source_field: str
    field: str
    operator: str
    consecutive_hours: int
    parameter_name: str

    @property
    def lookback_hours(self) -> int:
        return self.consecutive_hours - 1


@dataclass(frozen=True)
class ReplayResult:
    """Deterministic hourly result for one counterfactual rule replay."""

    window_start: datetime
    window_end: datetime
    threshold: Decimal
    evaluated_hours: int
    triggered: bool
    first_trigger_time: datetime | None
    first_trigger_offset_hours: int | None
    trigger_hours: int
    trigger_timestamps: tuple[datetime, ...]
    best_consecutive_margin: Decimal

    def as_answer(self) -> dict[str, object]:
        """Return the L2-compatible JSON answer fields."""

        return {
            "triggered": self.triggered,
            "first_trigger_time": (
                format_timestamp(self.first_trigger_time) if self.first_trigger_time else None
            ),
            "trigger_hours": self.trigger_hours,
        }


def parse_consecutive_expression(
    expression: str,
    *,
    canonical_field: str | None = None,
    field_aliases: Iterable[str] = (),
    expected_parameter: str | None = None,
) -> ConsecutiveRule:
    """Parse only ``consec_Hh(field >/< parameter) = true`` expressions.

    ``canonical_field`` binds a human-readable expression field to its CSV
    column. When supplied, the expression field must equal that column or one
    of ``field_aliases``. Boolean composition, direct comparisons, arithmetic,
    rolling statistics and non-strict comparisons are deliberately rejected.
    """

    if not isinstance(expression, str):
        raise UnsupportedRuleExpression("Rule expression must be text")
    match = _CONSECUTIVE_EXPRESSION.fullmatch(expression)
    if match is None:
        raise UnsupportedRuleExpression(
            "Supported grammar is only consec_Hh(field > parameter) = true "
            "or consec_Hh(field < parameter) = true"
        )

    source_field = match.group("field").strip()
    parameter = match.group("parameter").strip()
    operator = match.group("operator")
    hours = int(match.group("hours"))

    # The permissive field capture is constrained here so it cannot smuggle in
    # an arithmetic or boolean subexpression.
    forbidden = (" AND ", " OR ", "+", "*", "=", "!", "<", ">")
    upper_field = f" {source_field.upper()} "
    looks_like_function = re.match(r"^[A-Z_]\w*\s*\(", source_field, flags=re.IGNORECASE)
    if any(token in upper_field for token in forbidden) or looks_like_function:
        raise UnsupportedRuleExpression(f"Unsupported field expression: {source_field!r}")

    aliases = {alias.strip() for alias in field_aliases if alias and alias.strip()}
    if canonical_field is not None:
        canonical = canonical_field.strip()
        if not canonical:
            raise RuleBindingError("canonical_field must be non-empty")
        if source_field not in aliases | {canonical}:
            raise RuleBindingError(
                f"Expression field {source_field!r} does not match CSV field {canonical!r} "
                f"or its aliases"
            )
    else:
        canonical = source_field

    if expected_parameter is not None and parameter != expected_parameter:
        raise RuleBindingError(
            f"Expression parameter {parameter!r} does not match {expected_parameter!r}"
        )

    return ConsecutiveRule(
        expression=expression.strip(),
        source_field=source_field,
        field=canonical,
        operator=operator,
        consecutive_hours=hours,
        parameter_name=parameter,
    )


def compile_inventory_rule(rule: InventoryRule) -> ConsecutiveRule:
    """Bind a corpus ``Rule`` record to the intentionally small pilot engine."""

    compiled = parse_consecutive_expression(
        rule.expression,
        canonical_field=rule.field,
        field_aliases=(rule.metric_name,),
        expected_parameter=rule.parameter_name,
    )
    if compiled.operator != rule.operator:
        raise RuleBindingError(
            f"Parsed operator {compiled.operator!r} disagrees with inventory operator "
            f"{rule.operator!r}"
        )
    if compiled.consecutive_hours != rule.consecutive_hours:
        raise RuleBindingError(
            f"Parsed consecutive hours {compiled.consecutive_hours} disagrees with "
            f"inventory value {rule.consecutive_hours}"
        )
    return compiled


def quantize_threshold(value: object, *, decimal_places: int = 6) -> Decimal:
    """Convert and round a threshold deterministically using ``ROUND_HALF_UP``."""

    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
        raise ThresholdError("decimal_places must be an integer")
    if decimal_places < 0:
        raise ThresholdError("decimal_places must be non-negative")
    try:
        number = decimal_value(value, context="threshold")
    except ValueError as exc:
        raise ThresholdError(str(exc)) from exc

    quantum = Decimal(1).scaleb(-decimal_places)
    digit_count = max(1, len(number.as_tuple().digits))
    try:
        with localcontext() as context:
            context.prec = max(28, digit_count + decimal_places + 4)
            rounded = number.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ThresholdError(f"Cannot quantize threshold {value!r}") from exc
    if rounded.is_zero():
        return Decimal(0).quantize(quantum)
    return rounded


def generate_threshold_candidates(
    original: object,
    factors: Iterable[object],
    *,
    decimal_places: int = 6,
) -> tuple[Decimal, ...]:
    """Generate stable, de-duplicated counterfactual thresholds.

    Multiplication happens entirely in decimal arithmetic. Factors must be
    finite and positive; candidates that quantize back to the original value
    are removed while caller order is preserved.
    """

    base = quantize_threshold(original, decimal_places=decimal_places)
    candidates: list[Decimal] = []
    seen: set[Decimal] = set()
    for raw_factor in factors:
        try:
            factor = decimal_value(raw_factor, context="threshold factor")
        except ValueError as exc:
            raise ThresholdError(str(exc)) from exc
        if factor <= 0:
            raise ThresholdError(f"Threshold factor must be positive: {raw_factor!r}")
        candidate = quantize_threshold(base * factor, decimal_places=decimal_places)
        if candidate == base or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return tuple(candidates)


def replay_rule(
    rule: ConsecutiveRule,
    window: NumericWindow,
    *,
    threshold: object,
    threshold_decimal_places: int | None = None,
) -> ReplayResult:
    """Execute a counterfactual rule on a complete hourly ``NumericWindow``.

    Rule outputs are evaluated only for timestamps in the left-closed,
    right-open interval ``[window.start, window.end)``. Each output at time
    ``t`` uses exactly the hourly observations from ``t-H+1`` through ``t``.
    A missing target or lookback row is a data error, never silently treated as
    a non-trigger.
    """

    if window.field != rule.field:
        raise RuleBindingError(
            f"Rule expects CSV field {rule.field!r}, received window field {window.field!r}"
        )
    if threshold_decimal_places is None:
        try:
            effective_threshold = decimal_value(threshold, context="threshold")
        except ValueError as exc:
            raise ThresholdError(str(exc)) from exc
    else:
        effective_threshold = quantize_threshold(threshold, decimal_places=threshold_decimal_places)

    targets = validate_complete_hourly_window(window, required_lookback_hours=rule.lookback_hours)
    values = window.values_by_timestamp()
    trigger_times: list[datetime] = []
    first_offset: int | None = None
    best_consecutive_margin: Decimal | None = None

    for target_index, target in enumerate(targets):
        signed_margins: list[Decimal] = []
        for offset in range(rule.consecutive_hours):
            value = values[target.timestamp - timedelta(hours=offset)]
            if rule.operator == ">":
                signed_margins.append(value - effective_threshold)
            elif rule.operator == "<":
                signed_margins.append(effective_threshold - value)
            else:  # Defensive: parser never constructs another operator.
                raise UnsupportedRuleExpression(f"Unsupported operator: {rule.operator!r}")
        # The weakest point in the H-hour trailing window is its signed
        # consecutive margin. Taking the best such window gives a useful,
        # deterministic score for both triggered and non-triggered replays:
        # the rule triggers iff this value is strictly positive.
        consecutive_margin = min(signed_margins)
        if best_consecutive_margin is None or consecutive_margin > best_consecutive_margin:
            best_consecutive_margin = consecutive_margin
        if consecutive_margin > 0:
            trigger_times.append(target.timestamp)
            if first_offset is None:
                first_offset = target_index

    if best_consecutive_margin is None:  # pragma: no cover - targets are validated non-empty.
        raise IncompleteHourlySeriesError("Evaluation window contains no target rows")
    first_trigger = trigger_times[0] if trigger_times else None
    return ReplayResult(
        window_start=window.start,
        window_end=window.end,
        threshold=effective_threshold,
        evaluated_hours=len(targets),
        triggered=bool(trigger_times),
        first_trigger_time=first_trigger,
        first_trigger_offset_hours=first_offset,
        trigger_hours=len(trigger_times),
        trigger_timestamps=tuple(trigger_times),
        best_consecutive_margin=best_consecutive_margin,
    )


def load_supported_rules(
    model_key: str,
    rule_path: str | Path,
    metric_catalog_path: str | Path,
    *,
    raw_root: str | Path | None = None,
    derived_metrics_path: str | Path | None = None,
) -> dict[str, Rule]:
    """Load only rules that bind exactly to the audited executable grammar.

    Unsupported or ambiguous rows are intentionally skipped. The returned
    mapping is therefore a capability declaration, not a lossy attempt to
    reinterpret all rule expressions in the corpus.
    """

    source = Path(rule_path)
    metric_source = Path(metric_catalog_path)
    with source.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    rows = payload.get("anomaly_judgment_rules") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported stage-2 rule structure: {source}")
    metrics = load_metrics(metric_source)
    root = Path(raw_root).resolve() if raw_root is not None else None

    def public_path(path: Path | None) -> str | None:
        if path is None:
            return None
        return relative_posix(path, root) if root is not None else str(path)

    result: dict[str, Rule] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        rule_id = str(row.get("rule_id") or "").strip().upper()
        expression = str(row.get("rule_expression") or "").strip()
        parameters = row.get("threshold_params")
        if not rule_id or not expression:
            continue
        if not isinstance(parameters, list) or len(parameters) != 1:
            continue
        parameter = parameters[0]
        if not isinstance(parameter, dict):
            continue
        parameter_name = str(parameter.get("param_name") or "").strip()
        parameter_value = finite_float(parameter.get("param_value"))
        if not parameter_name or parameter_value is None:
            continue

        try:
            unbound = parse_consecutive_expression(expression)
        except RuleError:
            continue
        object_name = str(row.get("object_name") or "").strip()
        possible_metrics = [
            metric
            for metric in metrics.values()
            if metric.field == unbound.source_field
            or metric.name == unbound.source_field
            or (object_name and metric.name == object_name)
        ]
        possible_metrics = list({metric.field: metric for metric in possible_metrics}.values())
        if len(possible_metrics) != 1:
            continue
        metric = possible_metrics[0]
        if metric.data_type.lower() not in {"float", "double", "number", "numeric"}:
            continue
        try:
            compiled = parse_consecutive_expression(
                expression,
                canonical_field=metric.field,
                field_aliases=(metric.name,),
                expected_parameter=parameter_name,
            )
        except RuleError:
            continue

        # Fail closed when parallel public semantics disagree. A benchmark
        # question must not force the model to guess whether the formal
        # expression or the point-trigger metadata is authoritative.
        point_requirement = row.get("pointwise_criteria")
        if not isinstance(point_requirement, dict):
            continue
        consecutive_points = finite_float(point_requirement.get("consecutive_points"))
        single_point = point_requirement.get("single_point_trigger")
        if (
            consecutive_points is None
            or not float(consecutive_points).is_integer()
            or int(consecutive_points) != compiled.consecutive_hours
            or not isinstance(single_point, bool)
            or single_point is not (compiled.consecutive_hours == 1)
        ):
            continue

        derived_path = Path(derived_metrics_path) if derived_metrics_path else None
        record = Rule(
            model_key=model_key,
            rule_id=rule_id,
            expression=expression,
            field=metric.field,
            metric_name=metric.name,
            unit=metric.unit,
            operator=compiled.operator,
            consecutive_hours=compiled.consecutive_hours,
            parameter_name=parameter_name,
            parameter_value=parameter_value,
            parameter_description=str(parameter.get("meaning") or "").strip(),
            rule_path=public_path(source) or str(source),
            metric_catalog_path=public_path(metric_source) or str(metric_source),
            derived_metrics_path=(
                public_path(derived_path)
                if derived_path is not None
                and derived_path.is_file()
                and "derived" in metric.metric_type
                else None
            ),
            description=str(row.get("judgment_description") or row.get("text_description") or "").strip(),
        )
        # Duplicate IDs are invalid even if one happens to be unsupported.
        if rule_id in result:
            raise ValueError(f"Duplicate supported rule {model_key}/{rule_id} in {source}")
        result[rule_id] = record
    return result


def hydrate_supported_rules(inventory: CorpusInventory) -> int:
    """Populate an inventory with the executable rule subset in place."""

    total = 0
    for model_key in sorted(inventory.metrics_by_model):
        model_root = inventory.raw_root / model_key
        derived = model_root / "derived_metrics.json"
        rules = load_supported_rules(
            model_key,
            model_root / "rule_metric_stage2.json",
            model_root / "metric_catalog.json",
            raw_root=inventory.raw_root,
            derived_metrics_path=derived if derived.is_file() else None,
        )
        inventory.rules_by_model[model_key] = rules
        total += len(rules)
    return total