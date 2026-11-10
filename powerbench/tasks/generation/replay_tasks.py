from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from collections.abc import Iterable
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..models import CorpusInventory, Locator
from ..rules import (
    compile_inventory_rule,
    generate_threshold_candidates,
    replay_rule,
)
from ..timeseries import IncompleteHourlySeriesError, numeric_window, read_numeric_series
from ..util import format_timestamp
from .anchors import fault_anchor
from .eligibility import EligibilityRejection

_MARGIN_ROUND_DIGITS = 4


def _order_key(seed: int, *parts: object) -> str:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f").rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def _rounded_decimal(value: Decimal, digits: int = _MARGIN_ROUND_DIGITS) -> float:
    quantum = Decimal(1).scaleb(-digits)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    return 0.0 if rounded.is_zero() else float(rounded)


def _result_payload(result: Any) -> dict[str, Any]:
    answer = result.as_answer()
    answer["first_trigger_offset_hours"] = result.first_trigger_offset_hours
    answer["best_consecutive_margin"] = _rounded_decimal(result.best_consecutive_margin)
    return answer


def _meaningfully_changed(original: Any, counterfactual: Any) -> bool:
    # Keep this diagnostic about the ordinary replay outputs. The signed
    # margin necessarily shifts when the threshold shifts and would otherwise
    # make every intervention look outcome-changing by construction.
    return original.as_answer() != counterfactual.as_answer()


def _intervention_direction(
    operator: str, original_threshold: Decimal, counterfactual_threshold: Decimal
) -> str:
    if operator == ">":
        return "relax" if counterfactual_threshold < original_threshold else "tighten"
    if operator == "<":
        return "relax" if counterfactual_threshold > original_threshold else "tighten"
    raise ValueError(f"Unsupported threshold operator: {operator}")


def _away_from_boundary(points: Iterable[Any], threshold: Decimal) -> bool:
    margin = max(abs(threshold) * Decimal("1e-10"), Decimal("1e-9"))
    return all(abs(point.value - threshold) > margin for point in points)


def _require_source_coverage(
    series: tuple[Any, ...],
    *,
    start: Any,
    end: Any,
    lookback_hours: int,
) -> None:
    """Reject only windows that extend beyond an otherwise valid series.

    Missing points inside the available extent are data errors and remain the
    replay validator's responsibility. They must never be silently skipped.
    """

    if not series:
        raise IncompleteHourlySeriesError("Time-series CSV contains no rows")
    required_start = start - timedelta(hours=lookback_hours)
    required_end = end - timedelta(hours=1)
    available_start = series[0].timestamp
    available_end = series[-1].timestamp
    if required_start < available_start or required_end > available_end:
        raise EligibilityRejection(
            "insufficient_time_coverage",
            f"required [{format_timestamp(required_start)}, {format_timestamp(required_end)}], "
            f"available [{format_timestamp(available_start)}, {format_timestamp(available_end)}]",
        )


def build_replay_pool(
    inventory: CorpusInventory,
    locators: dict[str, Locator],
    *,
    window_hours: Iterable[int],
    threshold_factors: Iterable[float],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize all reliable counterfactual replay candidates once.

    The pool includes unchanged local outcomes because both L2 and L3 ask for
    results under a genuinely new threshold but do not condition selection on
    whether that intervention changes the hidden original-threshold result.
    """

    windows = tuple(sorted(set(int(value) for value in window_hours)))
    counters = defaultdict(int)
    rejection_examples: dict[str, dict[str, Any]] = {}
    pool: list[dict[str, Any]] = []

    def reject(
        reason: str,
        detail: str,
        *,
        device_uid: str,
        report_path: str | None = None,
        window: int | None = None,
    ) -> None:
        counters[reason] += 1
        rejection_examples.setdefault(
            reason,
            {
                "device_uid": device_uid,
                **({"report_path": report_path} if report_path is not None else {}),
                **({"window_hours": window} if window is not None else {}),
                "detail": detail,
            },
        )

    for uid in sorted(locators, key=lambda value: _order_key(seed, "device", value)):
        device = inventory.device(uid)
        series_cache: dict[str, tuple[Any, ...]] = {}
        rules = inventory.rules_by_model.get(device.model_key, {})
        if not rules:
            reject(
                "device_without_supported_rule",
                "device model has no rule in the audited executable subset",
                device_uid=uid,
            )
            continue
        anchor_cache: dict[str, Any] = {}
        anchor_rejections: dict[str, EligibilityRejection] = {}
        report_selectors: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
        for report_path in sorted(device.fault_reports):
            try:
                anchor = fault_anchor(inventory, uid, report_path)
            except EligibilityRejection as rejection:
                anchor_rejections[report_path] = rejection
                continue
            anchor_cache[report_path] = anchor
            discovery_date = anchor.metadata.get("Discovery time", "").split(" ", 1)[0]
            report_number = str(anchor.report_number or "").strip()
            report_selectors[(discovery_date, report_number)].append(report_path)
        for report_path in sorted(device.fault_reports):
            if report_path in anchor_rejections:
                rejection = anchor_rejections[report_path]
                reject(
                    rejection.reason,
                    rejection.detail,
                    device_uid=uid,
                    report_path=report_path,
                )
                continue
            anchor = anchor_cache[report_path]
            rule = rules.get(anchor.rule_id or "")
            if rule is None:
                reject(
                    "unsupported_report_rule",
                    f"associated rule {anchor.rule_id!r} is outside the executable subset",
                    device_uid=uid,
                    report_path=report_path,
                )
                continue
            discovery_date = anchor.metadata.get("Discovery time", "").split(" ", 1)[0]
            report_number = str(anchor.report_number or "").strip()
            if not report_number or report_number == "TBD":
                reject(
                    "missing_public_report_number",
                    "fault report has no usable report number",
                    device_uid=uid,
                    report_path=report_path,
                )
                continue
            selector_key = (discovery_date, report_number)
            selector_matches = report_selectors[selector_key]
            if selector_matches != [report_path]:
                reject(
                    "ambiguous_public_report_selector",
                    f"date and report number match {selector_matches}",
                    device_uid=uid,
                    report_path=report_path,
                )
                continue
            if rule.field not in device.csv_columns:
                raise ValueError(
                    f"Supported rule {rule.rule_id} requires missing CSV field "
                    f"{rule.field!r} for {uid}"
                )
            compiled = compile_inventory_rule(rule)
            thresholds = generate_threshold_candidates(
                rule.parameter_value,
                threshold_factors,
                decimal_places=6,
            )
            original_threshold = Decimal(str(rule.parameter_value))
            for hours in windows:
                start = anchor.end + timedelta(hours=24)
                end = start + timedelta(hours=hours)
                if rule.field not in series_cache:
                    series_cache[rule.field] = read_numeric_series(
                        inventory.raw_root / device.csv_path, field=rule.field
                    )
                try:
                    _require_source_coverage(
                        series_cache[rule.field],
                        start=start,
                        end=end,
                        lookback_hours=compiled.lookback_hours,
                    )
                except EligibilityRejection as rejection:
                    reject(
                        rejection.reason,
                        rejection.detail,
                        device_uid=uid,
                        report_path=report_path,
                        window=hours,
                    )
                    continue
                window = numeric_window(
                    series_cache[rule.field],
                    field=rule.field,
                    start=start,
                    end=end,
                    lookback_hours=compiled.lookback_hours,
                )
                original = replay_rule(compiled, window, threshold=rule.parameter_value)
                for threshold in thresholds:
                    if not _away_from_boundary(window.points, threshold):
                        reject(
                            "threshold_boundary",
                            f"one or more values are numerically indistinguishable from {threshold}",
                            device_uid=uid,
                            report_path=report_path,
                            window=hours,
                        )
                        continue
                    counterfactual = replay_rule(compiled, window, threshold=threshold)
                    if _rounded_decimal(counterfactual.best_consecutive_margin) == 0:
                        reject(
                            "rounded_zero_counterfactual_margin",
                            "counterfactual margin rounds to zero at answer precision",
                            device_uid=uid,
                            report_path=report_path,
                            window=hours,
                        )
                        continue
                    if counterfactual.triggered and counterfactual.trigger_hours in {0, hours}:
                        reject(
                            "degenerate_counterfactual",
                            f"triggered for {counterfactual.trigger_hours}/{hours} target hours",
                            device_uid=uid,
                            report_path=report_path,
                            window=hours,
                        )
                        continue
                    pool.append(
                        {
                            "device_uid": uid,
                            "model_key": device.model_key,
                            "report_path": report_path,
                            "anchor_event_id": anchor.event_id,
                            "fault_start": format_timestamp(anchor.start),
                            "fault_end": format_timestamp(anchor.end),
                            "discovery_time": anchor.metadata.get("Discovery time", ""),
                            "report_number": anchor.report_number,
                            "rule_id": rule.rule_id,
                            "rule_path": rule.rule_path,
                            "metric_catalog_path": rule.metric_catalog_path,
                            "derived_metrics_path": rule.derived_metrics_path,
                            "metric_field": rule.field,
                            "metric_name": rule.metric_name,
                            "unit": rule.unit,
                            "expression": rule.expression,
                            "operator": rule.operator,
                            "consecutive_hours": rule.consecutive_hours,
                            "parameter_name": rule.parameter_name,
                            "parameter_description": rule.parameter_description,
                            "original_threshold": rule.parameter_value,
                            "counterfactual_threshold": float(threshold),
                            "counterfactual_threshold_text": _decimal_text(threshold),
                            "relative_threshold_change": round(
                                float(abs((threshold - original_threshold) / original_threshold)),
                                6,
                            ),
                            "intervention_direction": _intervention_direction(
                                compiled.operator, original_threshold, threshold
                            ),
                            "window_start": format_timestamp(start),
                            "window_end_exclusive": format_timestamp(end),
                            "window_hours": hours,
                            "original_result": _result_payload(original),
                            "result": _result_payload(counterfactual),
                            "changed": _meaningfully_changed(original, counterfactual),
                            "locator": locators[uid].as_dict(),
                        }
                    )
    counters["pool_size"] = len(pool)
    counters["changed_pool_size"] = sum(bool(item["changed"]) for item in pool)
    counters["triggered_pool_size"] = sum(bool(item["result"]["triggered"]) for item in pool)
    return pool, {
        **dict(counters),
        "rejection_examples": rejection_examples,
    }


def _rule_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": candidate["rule_id"],
        "rule_path": candidate["rule_path"],
        "metric_catalog_path": candidate["metric_catalog_path"],
        "derived_metrics_path": candidate["derived_metrics_path"],
        "metric_field": candidate["metric_field"],
        "metric_name": candidate["metric_name"],
        "unit": candidate["unit"],
        "expression": candidate["expression"],
        "operator": candidate["operator"],
        "consecutive_hours": candidate["consecutive_hours"],
        "parameter_name": candidate["parameter_name"],
        "original_threshold": candidate["original_threshold"],
        "counterfactual_threshold": candidate["counterfactual_threshold"],
        "counterfactual_threshold_text": candidate["counterfactual_threshold_text"],
    }


def generate_l2_tasks(
    pool: list[dict[str, Any]],
    *,
    count: int,
    window_hours: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    if count % len(window_hours):
        raise ValueError("L2 count must be divisible by the number of window sizes")
    per_window = count // len(window_hours)
    selected: list[dict[str, Any]] = []
    used_devices: set[str] = set()
    used_signatures: set[tuple[str, str, int, str]] = set()
    device_counts: defaultdict[str, int] = defaultdict(int)
    model_counts: defaultdict[str, int] = defaultdict(int)
    strata: list[tuple[int, bool, int]] = []
    for window_index, hours in enumerate(window_hours):
        # Alternate the odd item between outcomes so the complete pilot is
        # exactly balanced without conditioning on whether the intervention
        # changed the unseen original-threshold replay result.
        true_quota = per_window // 2 + (1 if window_index % 2 == 0 else 0)
        for desired, quota in ((True, true_quota), (False, per_window - true_quota)):
            strata.append((hours, desired, quota))

    # Keep intervention magnitude constant across answer classes. Otherwise a
    # model can partly infer triggered/non-triggered from how aggressively the
    # threshold was changed without reading the target time series.
    relaxed = [item for item in pool if item["intervention_direction"] == "relax"]
    strengths = sorted({float(item["relative_threshold_change"]) for item in relaxed})
    feasible_strengths = [
        strength
        for strength in strengths
        if all(
            len(
                {
                    item["device_uid"]
                    for item in relaxed
                    if item["window_hours"] == hours
                    and item["result"]["triggered"] is desired
                    and float(item["relative_threshold_change"]) == strength
                }
            )
            >= quota
            for hours, desired, quota in strata
        )
    ]
    if not feasible_strengths:
        raise RuntimeError("No single relaxed-threshold magnitude can fill every L2 stratum")
    standardized_strength = max(feasible_strengths)

    requests = []
    for hours, desired, quota in strata:
        candidates = [
            item
            for item in relaxed
            if item["window_hours"] == hours
            and item["result"]["triggered"] is desired
            and float(item["relative_threshold_change"]) == standardized_strength
        ]
        requests.append(
            (
                len({item["device_uid"] for item in candidates}),
                hours,
                desired,
                quota,
                candidates,
            )
        )
    # Allocate scarce strata first so abundant true cases cannot consume every
    # device that is capable of producing a meaningful false counterfactual.
    for _, hours, desired, quota, candidates in sorted(requests, key=lambda item: item[0]):
        made = 0
        ordered = sorted(
            candidates,
            key=lambda item: _order_key(
                seed,
                "L2",
                hours,
                desired,
                item["device_uid"],
                item["report_path"],
                item["counterfactual_threshold_text"],
            ),
        )
        # Pass 1 keeps every target device unique. Pass 2 permits at most
        # two distinct report/window instances from a device if the frozen
        # corpus has a genuinely scarce outcome stratum.
        for allow_reuse, model_cap in ((False, 2), (False, 4), (True, 6), (True, count)):
            for item in ordered:
                if made >= quota:
                    break
                signature = (
                    item["device_uid"],
                    item["report_path"],
                    int(item["window_hours"]),
                    item["counterfactual_threshold_text"],
                )
                if signature in used_signatures:
                    continue
                if not allow_reuse and item["device_uid"] in used_devices:
                    continue
                if allow_reuse and device_counts[item["device_uid"]] >= 2:
                    continue
                if model_counts[item["model_key"]] >= model_cap:
                    continue
                task = {
                    "task_level": "L2",
                    "target_device_uids": [item["device_uid"]],
                    "device_aliases": {"A": item["device_uid"]},
                    "locators": {"A": item["locator"]},
                    "report_paths": {"A": item["report_path"]},
                    "anchors": {
                        "A": {
                            "event_id": item["anchor_event_id"],
                            "fault_end": item["fault_end"],
                            "discovery_time": item["discovery_time"],
                            "report_number": item["report_number"],
                        }
                    },
                    "rule": _rule_payload(item),
                    "window": {
                        "start": item["window_start"],
                        "end_exclusive": item["window_end_exclusive"],
                        "hours": hours,
                        "semantics": "[fault_end + 24h, fault_end + 24h + W)",
                    },
                    "answer": {
                        key: item["result"][key]
                        for key in (
                            "triggered",
                            "first_trigger_time",
                            "trigger_hours",
                            "best_consecutive_margin",
                        )
                    },
                    "tolerances": {"best_consecutive_margin": 0.0001},
                    "diagnostics": {
                        "original_result": item["original_result"],
                        "counterfactual_changed_output": bool(item["changed"]),
                        "intervention_direction": item["intervention_direction"],
                        "relative_threshold_change": item["relative_threshold_change"],
                    },
                }
                selected.append(task)
                used_devices.add(item["device_uid"])
                used_signatures.add(signature)
                device_counts[item["device_uid"]] += 1
                model_counts[item["model_key"]] += 1
                made += 1
            if made >= quota:
                break
        if made != quota:
            raise RuntimeError(
                f"Could generate only {made}/{quota} L2 candidates for W={hours}, "
                f"triggered={desired}"
            )
    selected.sort(
        key=lambda task: _order_key(
            seed,
            "L2-number",
            task["target_device_uids"][0],
            task["report_paths"]["A"],
            task["window"]["hours"],
            task["rule"]["counterfactual_threshold_text"],
        )
    )
    for index, task in enumerate(selected, start=1):
        task["pair_id"] = f"PB-L2-{index:03d}"
    return selected


def _pick_local_options(
    options: list[dict[str, Any]],
    *,
    seed: int,
    role: str,
    limit: int,
) -> list[dict[str, Any]]:
    return sorted(
        options,
        key=lambda item: _order_key(
            seed,
            "L3-local",
            role,
            item["device_uid"],
            item["report_path"],
            item["counterfactual_threshold_text"],
        ),
    )[:limit]


def _l3_candidate_from_locals(
    locals_: tuple[dict[str, Any], ...],
    *,
    seed: int,
) -> dict[str, Any] | None:
    device_count = len(locals_)
    if not 3 <= device_count <= 5:
        return None
    if len({item["device_uid"] for item in locals_}) != device_count:
        return None
    # Every branch executes a threshold that differs from the public original.
    # We record intervention effects diagnostically instead of conditioning the
    # entire L3 sample on them, which would introduce answer-selection bias.
    triggered = [item for item in locals_ if item["result"]["triggered"]]
    # Require a mixed outcome so the item cannot be solved by copying one local
    # replay result to every device.
    if not 1 <= len(triggered) < device_count:
        return None
    if any(item["result"]["trigger_hours"] in {0, item["window_hours"]} for item in triggered):
        return None
    offsets = [int(item["result"]["first_trigger_offset_hours"]) for item in triggered]
    if offsets.count(min(offsets)) != 1:
        return None
    if len(offsets) > 1 and max(offsets) == min(offsets):
        return None

    ordered = sorted(
        locals_,
        key=lambda item: _order_key(
            seed,
            "L3-alias",
            item["model_key"],
            item["rule_id"],
            item["counterfactual_threshold_text"],
            item["device_uid"],
        ),
    )
    alias_names = [f"D{index}" for index in range(1, device_count + 1)]
    aliases = {alias: item for alias, item in zip(alias_names, ordered, strict=True)}
    triggered_aliases = [alias for alias, item in aliases.items() if item["result"]["triggered"]]
    earliest = min(
        triggered_aliases,
        key=lambda alias: aliases[alias]["result"]["first_trigger_offset_hours"],
    )
    triggered_offsets = [
        int(aliases[alias]["result"]["first_trigger_offset_hours"]) for alias in triggered_aliases
    ]
    margins = {
        alias: Decimal(str(item["result"]["best_consecutive_margin"]))
        for alias, item in aliases.items()
    }
    # A complete ordering forces the solver to execute every local
    # branches even when only one device triggers. Four-decimal ties are
    # rejected rather than resolved by an arbitrary alias tie-break.
    if len(set(margins.values())) != len(margins):
        return None
    margin_order = sorted(aliases, key=lambda alias: margins[alias], reverse=True)
    representative = ordered[0]
    per_device = []
    for alias, item in aliases.items():
        per_device.append(
            {
                "alias": alias,
                **{
                    key: item["result"][key]
                    for key in (
                        "triggered",
                        "first_trigger_time",
                        "first_trigger_offset_hours",
                        "trigger_hours",
                        "best_consecutive_margin",
                    )
                },
            }
        )
    task = {
        "task_level": "L3",
        "target_device_uids": [item["device_uid"] for item in ordered],
        "device_aliases": {alias: item["device_uid"] for alias, item in aliases.items()},
        "locators": {alias: item["locator"] for alias, item in aliases.items()},
        "report_paths": {alias: item["report_path"] for alias, item in aliases.items()},
        "anchors": {
            alias: {
                "event_id": item["anchor_event_id"],
                "fault_end": item["fault_end"],
                "discovery_time": item["discovery_time"],
                "report_number": item["report_number"],
            }
            for alias, item in aliases.items()
        },
        "rule": _rule_payload(representative),
        "window": {
            "hours": representative["window_hours"],
            "semantics": "each [fault_end + 24h, fault_end + 24h + W)",
            "per_device": {
                alias: {
                    "start": item["window_start"],
                    "end_exclusive": item["window_end_exclusive"],
                }
                for alias, item in aliases.items()
            },
        },
        "answer": {
            "per_device": per_device,
            "triggered_devices": triggered_aliases,
            "earliest_trigger_device": earliest,
            "trigger_offset_span_hours": max(triggered_offsets) - min(triggered_offsets),
            "margin_order": margin_order,
            "margin_span": _rounded_decimal(max(margins.values()) - min(margins.values())),
        },
        "tolerances": {
            "best_consecutive_margin": 0.0001,
            "margin_span": 0.0001,
        },
        "diagnostics": {
            "changed_branches": sum(bool(item["changed"]) for item in ordered),
            "triggered_branch_count": len(triggered),
            "original_results": {alias: item["original_result"] for alias, item in aliases.items()},
            "target_device_count": device_count,
        },
    }
    return task


def generate_l3_tasks(
    pool: list[dict[str, Any]],
    *,
    count: int,
    window_hours: int,
    seed: int,
    device_counts: Iterable[int] = (3,),
    max_local_options_per_device: int = 4,
    max_candidates_per_group_pattern: int = 4,
) -> list[dict[str, Any]]:
    requested_counts = tuple(sorted(set(int(value) for value in device_counts)))
    if not requested_counts or any(value < 3 or value > 5 for value in requested_counts):
        raise ValueError("L3 device_counts must contain values between three and five")
    if count % len(requested_counts):
        raise ValueError("L3 task count must be divisible by the number of device counts")
    if max_local_options_per_device < 1 or max_candidates_per_group_pattern < 1:
        raise ValueError("L3 candidate-search budgets must be positive")
    quota = count // len(requested_counts)
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in pool:
        if item["window_hours"] != window_hours:
            continue
        key = (item["model_key"], item["rule_id"], item["counterfactual_threshold_text"])
        grouped[key].append(item)

    selected: list[dict[str, Any]] = []
    reserves: dict[int, list[dict[str, Any]]] = {}
    selected_counts: dict[int, int] = {}
    for device_count in requested_counts:
        candidates: list[dict[str, Any]] = []
        for group_key, items in sorted(
            grouped.items(),
            key=lambda pair: _order_key(seed, "L3-group", device_count, *pair[0]),
        ):
            by_outcome: dict[bool, defaultdict[str, list[dict[str, Any]]]] = {
                True: defaultdict(list),
                False: defaultdict(list),
            }
            for item in items:
                by_outcome[bool(item["result"]["triggered"])][item["device_uid"]].append(item)

            trigger_counts = sorted(
                range(1, device_count),
                key=lambda value: (abs(value - device_count / 2), -value),
            )
            for triggered_count in trigger_counts:
                false_count = device_count - triggered_count
                if len(by_outcome[True]) < triggered_count or len(by_outcome[False]) < false_count:
                    continue
                pattern_candidates = 0
                true_groups = itertools.combinations(sorted(by_outcome[True]), triggered_count)
                false_uids = sorted(by_outcome[False])
                for true_uids in true_groups:
                    for selected_false in itertools.combinations(false_uids, false_count):
                        uid_outcomes = [
                            *((True, uid) for uid in true_uids),
                            *((False, uid) for uid in selected_false),
                        ]
                        option_lists = [
                            _pick_local_options(
                                by_outcome[outcome][uid],
                                seed=seed,
                                role=f"{device_count}:{index}",
                                limit=max_local_options_per_device,
                            )
                            for index, (outcome, uid) in enumerate(uid_outcomes)
                        ]
                        found = None
                        found_locals = None
                        for local_tuple in itertools.product(*option_lists):
                            found = _l3_candidate_from_locals(local_tuple, seed=seed)
                            if found is not None:
                                found_locals = local_tuple
                                break
                        if found is not None and found_locals is not None:
                            signature = "|".join(
                                sorted(
                                    f"{item['device_uid']}:{item['report_path']}"
                                    for item in found_locals
                                )
                            )
                            found["candidate_signature"] = (
                                f"k={device_count}|{'/'.join(group_key)}|{signature}"
                            )
                            candidates.append(found)
                            pattern_candidates += 1
                        if pattern_candidates >= max_candidates_per_group_pattern:
                            break
                    if pattern_candidates >= max_candidates_per_group_pattern:
                        break

        ordered_candidates = sorted(
            candidates,
            key=lambda task: (
                -int(task["diagnostics"]["triggered_branch_count"]),
                _order_key(seed, "L3-select", task["candidate_signature"]),
            ),
        )
        stratum: list[dict[str, Any]] = []
        model_counts: defaultdict[str, int] = defaultdict(int)
        group_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        signatures: set[str] = set()
        for model_cap, group_cap in ((3, 2), (6, 3), (quota, quota)):
            for task in ordered_candidates:
                if len(stratum) >= quota:
                    break
                signature = task["candidate_signature"]
                if signature in signatures:
                    continue
                first_uid = task["target_device_uids"][0]
                model_key = first_uid.split("/", 1)[0]
                group_key = (model_key, task["rule"]["rule_id"])
                if model_counts[model_key] >= model_cap or group_counts[group_key] >= group_cap:
                    continue
                signatures.add(signature)
                model_counts[model_key] += 1
                group_counts[group_key] += 1
                stratum.append(task)
            if len(stratum) >= quota:
                break
        chosen_signatures = {task["candidate_signature"] for task in stratum}
        reserves[device_count] = [
            task
            for task in ordered_candidates
            if task["candidate_signature"] not in chosen_signatures
        ]
        selected_counts[device_count] = len(stratum)
        selected.extend(stratum)

    # A stricter public report selector can make one device-count stratum
    # slightly smaller than its equal-share target. Preserve the requested
    # total by drawing deterministic reserves from the least represented
    # remaining stratum instead of failing the whole benchmark build.
    while len(selected) < count:
        available = [device_count for device_count in requested_counts if reserves[device_count]]
        if not available:
            raise RuntimeError(f"Could generate only {len(selected)}/{count} L3 tasks")
        device_count = min(available, key=lambda value: (selected_counts[value], value))
        selected.append(reserves[device_count].pop(0))
        selected_counts[device_count] += 1

    selected.sort(
        key=lambda task: (
            len(task["target_device_uids"]),
            _order_key(seed, "L3-number", task["candidate_signature"]),
        )
    )
    for index, task in enumerate(selected, start=1):
        task.pop("candidate_signature", None)
        task["pair_id"] = f"PB-L3-{index:03d}"
    return selected