from __future__ import annotations

import copy
import hashlib
import itertools
from collections import defaultdict
from collections.abc import Iterable
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal
from statistics import median
from typing import Any

from ..models import CorpusInventory, Locator
from ..rules import ConsecutiveRule, compile_inventory_rule, replay_rule
from ..timeseries import (
    numeric_window,
    read_numeric_series,
    read_numeric_window,
    validate_complete_hourly_window,
)
from ..util import format_timestamp, parse_timestamp
from .anchors import check_anchor, fault_anchor
from .eligibility import EligibilityRejection
from .l1_tasks import _build_candidate
from .locators import metadata_key_group
from .replay_tasks import _rule_payload

INTERFACE_CAPACITY_MBPS = {"GE": 1000.0, "10GE": 10000.0, "XGSPON": 10000.0}


def _order_key(seed: int, *parts: object) -> str:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _round(value: Decimal | float, digits: int = 4) -> float:
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    quantum = Decimal(1).scaleb(-digits)
    result = number.quantize(quantum, rounding=ROUND_HALF_UP)
    return float(Decimal(0) if result.is_zero() else result)


def _longest_true_run(flags: Iterable[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _locator_hides_filter(locator: dict[str, Any], *, filter_field: str, filter_value: str) -> bool:
    """Reject prompts whose device locator already reveals the filter decision."""

    filter_group = metadata_key_group(filter_field)
    return all(
        metadata_key_group(str(item["field"])) != filter_group
        and str(item["value"]) != filter_value
        for item in locator.get("fields", [])
    )


def _fault_report_matches_date_and_number(
    inventory: CorpusInventory,
    uid: str,
    *,
    discovery_date: str,
    report_number: str,
) -> list[str]:
    """Resolve the deliberately non-leaking public fault-report selector."""

    matches: list[str] = []
    for report_path in sorted(inventory.device(uid).fault_reports):
        try:
            candidate = fault_anchor(inventory, uid, report_path)
        except ValueError:
            continue
        candidate_date = candidate.metadata.get("Discovery time", "").split(" ", 1)[0]
        if candidate_date == discovery_date and candidate.report_number == report_number:
            matches.append(report_path)
    return matches


def _family_copy(task: dict[str, Any], family: str) -> dict[str, Any]:
    result = copy.deepcopy(task)
    result.pop("pair_id", None)
    result["task_family"] = family
    return result


def l1_family_pools(
    inventory: CorpusInventory,
    locators: dict[str, Locator],
    base_pool: list[dict[str, Any]],
    replay_pool: list[dict[str, Any]],
    *,
    digits: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {
        "post_window": [],
        "threshold_episode": [],
        "pre_post_comparison": [],
        "metadata_normalized": [],
    }
    cached_uid: str | None = None
    cached_series: dict[str, tuple[Any, ...]] = {}
    for source in base_pool:
        base = _family_copy(source, "post_window")
        pools["post_window"].append(base)

        episode = _family_copy(source, "threshold_episode")
        uid = episode["target_device_uids"][0]
        device = inventory.device(uid)
        if uid != cached_uid:
            cached_uid = uid
            cached_series.clear()
        field = episode["metric"]["field"]
        start = parse_timestamp(episode["window"]["start"])
        end = parse_timestamp(episode["window"]["end_exclusive"])
        if field not in cached_series:
            cached_series[field] = read_numeric_series(
                inventory.raw_root / device.csv_path, field=field
            )
        points = validate_complete_hourly_window(
            numeric_window(cached_series[field], field=field, start=start, end=end)
        )
        anchor_end = parse_timestamp(episode["window"]["anchor_end"])
        threshold = next(
            point.value for point in cached_series[field] if point.timestamp == anchor_end
        )
        flags = [point.value > threshold for point in points]
        if 0 < sum(flags) < len(flags):
            first = next(point.timestamp for point, flag in zip(points, flags, strict=True) if flag)
            episode["threshold"] = {
                "source": "inspection_end_value",
                "value": float(threshold),
            }
            episode["answer"] = {
                "exceedance_hours": sum(flags),
                "longest_exceedance_run_hours": _longest_true_run(flags),
                "first_exceedance_time": format_timestamp(first),
            }
            episode["tolerances"] = {}
            pools["threshold_episode"].append(episode)

    # A report/rule/threshold tuple appears several times in the replay pool.
    # For a pre/post task only the report, metric and window size matter.
    seen_pre_post: set[tuple[str, str, str, int]] = set()
    cached_uid = None
    cached_series.clear()
    for item in sorted(
        replay_pool,
        key=lambda value: _order_key(
            seed,
            "pre-post",
            value["device_uid"],
            value["report_path"],
            value["metric_field"],
            value["window_hours"],
        ),
    ):
        signature = (
            item["device_uid"],
            item["report_path"],
            item["metric_field"],
            int(item["window_hours"]),
        )
        if signature in seen_pre_post:
            continue
        seen_pre_post.add(signature)
        uid = item["device_uid"]
        device = inventory.device(uid)
        if uid != cached_uid:
            cached_uid = uid
            cached_series.clear()
        if item["metric_field"] not in cached_series:
            cached_series[item["metric_field"]] = read_numeric_series(
                inventory.raw_root / device.csv_path, field=item["metric_field"]
            )
        hours = min(48, int(item["window_hours"]))
        anchor_end = parse_timestamp(item["fault_end"])
        pre_start, pre_end = anchor_end - timedelta(hours=hours), anchor_end
        post_start, post_end = (
            anchor_end + timedelta(hours=1),
            anchor_end + timedelta(hours=hours + 1),
        )
        try:
            pre = validate_complete_hourly_window(
                numeric_window(
                    cached_series[item["metric_field"]],
                    field=item["metric_field"],
                    start=pre_start,
                    end=pre_end,
                )
            )
            post = validate_complete_hourly_window(
                numeric_window(
                    cached_series[item["metric_field"]],
                    field=item["metric_field"],
                    start=post_start,
                    end=post_end,
                )
            )
        except ValueError:
            continue
        pre_mean = sum((point.value for point in pre), Decimal(0)) / Decimal(len(pre))
        post_mean = sum((point.value for point in post), Decimal(0)) / Decimal(len(post))
        if pre_mean == 0:
            continue
        maximum = max(point.value for point in post)
        first_max = next(point.timestamp for point in post if point.value == maximum)
        change = (post_mean - pre_mean) / abs(pre_mean) * Decimal(100)
        task = {
            "task_level": "L1",
            "task_family": "pre_post_comparison",
            "target_device_uids": [uid],
            "device_aliases": {"A": uid},
            "locators": {"A": item["locator"]},
            "report_paths": {"A": item["report_path"]},
            "rule": None,
            "metric": {
                "field": item["metric_field"],
                "name": item["metric_name"],
                "unit": item["unit"],
                "metric_catalog_path": item["metric_catalog_path"],
            },
            "anchor": {
                "event_id": item["anchor_event_id"],
                "fault_end": item["fault_end"],
                "discovery_time": item["discovery_time"],
                "report_number": item["report_number"],
            },
            "window": {
                "hours": hours,
                "pre_start": format_timestamp(pre_start),
                "pre_end_exclusive": format_timestamp(pre_end),
                "post_start": format_timestamp(post_start),
                "post_end_exclusive": format_timestamp(post_end),
                "semantics": "W hourly points before and after fault_end; fault_end excluded",
            },
            "answer": {
                "pre_mean": _round(pre_mean, digits),
                "post_mean": _round(post_mean, digits),
                "change_rate_percent": _round(change, digits),
                "post_max": _round(maximum, digits),
                "first_post_max_time": format_timestamp(first_max),
                "unit": item["unit"],
            },
            "tolerances": {
                "pre_mean": 10 ** (-digits),
                "post_mean": 10 ** (-digits),
                "change_rate_percent": 10 ** (-digits),
                "post_max": 10 ** (-digits),
            },
            "answer_round_digits": digits,
        }
        pools["pre_post_comparison"].append(task)

    # This family needs one specific metric and one archive parameter, so it
    # must be enumerated directly rather than inherited from the generic L1
    # pool's single canonical metric per device.
    for uid in sorted(locators):
        device = inventory.device(uid)
        if device.model_key != "0054_MA5600T":
            continue
        interface = device.metadata.get("Uplink interface", "").strip().upper()
        capacity = INTERFACE_CAPACITY_MBPS.get(interface)
        metric = inventory.metrics_by_model.get(device.model_key, {}).get(
            "uplink_interface_traffic"
        )
        if capacity is None or metric is None or metric.field not in device.csv_columns:
            continue
        series = read_numeric_series(inventory.raw_root / device.csv_path, field=metric.field)
        for report_path in device.check_reports:
            try:
                anchor = check_anchor(inventory, uid, report_path)
                for hours in (48, 72, 168):
                    source = _build_candidate(
                        inventory,
                        locators[uid],
                        uid,
                        report_path,
                        metric,
                        hours,
                        digits,
                        anchor=anchor,
                        series_points=series,
                    )
                    start = parse_timestamp(source["window"]["start"])
                    end = parse_timestamp(source["window"]["end_exclusive"])
                    points = validate_complete_hourly_window(
                        numeric_window(series, field=metric.field, start=start, end=end)
                    )
                    utilization = [float(point.value) / capacity * 100 for point in points]
                    for threshold in (20.0, 30.0, 40.0, 50.0, 70.0):
                        flags = [value > threshold for value in utilization]
                        if not 0 < sum(flags) < len(flags):
                            continue
                        task = _family_copy(source, "metadata_normalized")
                        task["capacity"] = {
                            "metadata_field": "Uplink interface",
                            "interface": interface,
                            "value_mbps": capacity,
                        }
                        task["utilization_threshold_percent"] = threshold
                        task["answer"] = {
                            "capacity_mbps": capacity,
                            "mean_utilization_percent": _round(
                                sum(utilization) / len(utilization), digits
                            ),
                            "longest_above_utilization_run_hours": _longest_true_run(flags),
                            "first_above_utilization_time": format_timestamp(
                                next(
                                    point.timestamp
                                    for point, flag in zip(points, flags, strict=True)
                                    if flag
                                )
                            ),
                        }
                        task["tolerances"] = {
                            "capacity_mbps": 0.0,
                            "mean_utilization_percent": 10 ** (-digits),
                        }
                        pools["metadata_normalized"].append(task)
            except (EligibilityRejection, ValueError):
                continue
    return pools


def _replay_task(item: dict[str, Any], family: str) -> dict[str, Any]:
    return {
        "task_level": "L2",
        "task_family": family,
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
            "hours": item["window_hours"],
            "offset_hours": 24,
            "semantics": "[fault_end + offset, fault_end + offset + W)",
        },
        "answer": {
            key: item["result"][key] for key in ("triggered", "first_trigger_time", "trigger_hours")
        },
        "tolerances": {},
    }


def l2_family_pools(
    inventory: CorpusInventory,
    replay_pool: list[dict[str, Any]],
    locators: dict[str, Locator],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {
        "threshold_replay": [],
        "duration_replay": [],
        "shifted_window_replay": [],
        "metadata_bound_replay": [],
    }
    seen: set[tuple[Any, ...]] = set()
    cached_uid: str | None = None
    cached_series: dict[str, tuple[Any, ...]] = {}
    for item in replay_pool:
        signature = (
            item["device_uid"],
            item["report_path"],
            item["window_hours"],
            item["counterfactual_threshold_text"],
        )
        if signature in seen:
            continue
        seen.add(signature)
        pools["threshold_replay"].append(_replay_task(item, "threshold_replay"))

        device = inventory.device(item["device_uid"])
        if item["device_uid"] != cached_uid:
            cached_uid = item["device_uid"]
            cached_series.clear()
        inventory_rule = inventory.rules_by_model[item["model_key"]][item["rule_id"]]
        compiled = compile_inventory_rule(inventory_rule)
        if compiled.field not in cached_series:
            cached_series[compiled.field] = read_numeric_series(
                inventory.raw_root / device.csv_path, field=compiled.field
            )
        start = parse_timestamp(item["window_start"])
        end = parse_timestamp(item["window_end_exclusive"])
        for duration in sorted(
            {max(1, compiled.consecutive_hours - 1), compiled.consecutive_hours + 2}
        ):
            if duration == compiled.consecutive_hours:
                continue
            changed_rule = ConsecutiveRule(
                expression=compiled.expression,
                source_field=compiled.source_field,
                field=compiled.field,
                operator=compiled.operator,
                consecutive_hours=duration,
                parameter_name=compiled.parameter_name,
            )
            try:
                window = numeric_window(
                    cached_series[compiled.field],
                    field=compiled.field,
                    start=start,
                    end=end,
                    lookback_hours=changed_rule.lookback_hours,
                )
                validate_complete_hourly_window(
                    window, required_lookback_hours=changed_rule.lookback_hours
                )
                result = replay_rule(changed_rule, window, threshold=inventory_rule.parameter_value)
            except ValueError:
                continue
            task = _replay_task(item, "duration_replay")
            task["rule"]["counterfactual_threshold"] = inventory_rule.parameter_value
            task["rule"]["counterfactual_threshold_text"] = str(inventory_rule.parameter_value)
            task["rule"]["counterfactual_consecutive_hours"] = duration
            task["answer"] = {
                key: result.as_answer()[key]
                for key in ("triggered", "first_trigger_time", "trigger_hours")
            }
            pools["duration_replay"].append(task)

        for offset in (48, 72):
            shifted_start = parse_timestamp(item["fault_end"]) + timedelta(hours=offset)
            shifted_end = shifted_start + timedelta(hours=int(item["window_hours"]))
            try:
                window = numeric_window(
                    cached_series[compiled.field],
                    field=compiled.field,
                    start=shifted_start,
                    end=shifted_end,
                    lookback_hours=compiled.lookback_hours,
                )
                validate_complete_hourly_window(
                    window, required_lookback_hours=compiled.lookback_hours
                )
                result = replay_rule(compiled, window, threshold=item["counterfactual_threshold"])
            except ValueError:
                continue
            task = _replay_task(item, "shifted_window_replay")
            task["window"].update(
                {
                    "start": format_timestamp(shifted_start),
                    "end_exclusive": format_timestamp(shifted_end),
                    "offset_hours": offset,
                }
            )
            task["answer"] = {
                key: result.as_answer()[key]
                for key in ("triggered", "first_trigger_time", "trigger_hours")
            }
            pools["shifted_window_replay"].append(task)

    # K10 for MA5600T is intentionally outside the simple consecutive-rule
    # grammar.  This family implements its published six-hour mean utilization
    # semantics explicitly and binds C_if from the device archive.
    for uid in sorted(locators):
        device = inventory.device(uid)
        if device.model_key != "0054_MA5600T":
            continue
        interface = device.metadata.get("Uplink interface", "").strip().upper()
        capacity = INTERFACE_CAPACITY_MBPS.get(interface)
        if capacity is None:
            continue
        for report_path in device.fault_reports:
            try:
                anchor = fault_anchor(inventory, uid, report_path)
            except ValueError:
                continue
            if anchor.rule_id != "K10":
                continue
            discovery_date = anchor.metadata.get("Discovery time", "").split(" ", 1)[0]
            report_number = str(anchor.report_number or "").strip()
            if not report_number or report_number == "TBD":
                continue
            if _fault_report_matches_date_and_number(
                inventory,
                uid,
                discovery_date=discovery_date,
                report_number=report_number,
            ) != [report_path]:
                continue
            start = anchor.end + timedelta(hours=24)
            end = start + timedelta(hours=72)
            try:
                window = read_numeric_window(
                    inventory.raw_root / device.csv_path,
                    field="uplink_interface_traffic",
                    start=start,
                    end=end,
                    lookback_hours=5,
                )
                targets = validate_complete_hourly_window(window, required_lookback_hours=5)
            except ValueError:
                continue
            values = window.values_by_timestamp()
            rolling_utilization: list[Decimal] = []
            for point in targets:
                six = [values[point.timestamp - timedelta(hours=offset)] for offset in range(6)]
                rolling_utilization.append(
                    sum(six, Decimal(0)) / Decimal(6) / Decimal(str(capacity))
                )
            for threshold in (Decimal("0.025"), Decimal("0.25"), Decimal("0.65")):
                flags = [value > threshold for value in rolling_utilization]
                first = next(
                    (point.timestamp for point, flag in zip(targets, flags, strict=True) if flag),
                    None,
                )
                pools["metadata_bound_replay"].append(
                    {
                        "task_level": "L2",
                        "task_family": "metadata_bound_replay",
                        "target_device_uids": [uid],
                        "device_aliases": {"A": uid},
                        "locators": {"A": locators[uid].as_dict()},
                        "report_paths": {"A": report_path},
                        "anchors": {
                            "A": {
                                "event_id": anchor.event_id,
                                "fault_end": format_timestamp(anchor.end),
                                "discovery_time": anchor.metadata.get("Discovery time", ""),
                                "report_number": anchor.report_number,
                            }
                        },
                        "rule": {
                            "rule_id": "K10",
                            "rule_path": "0054_MA5600T/rule_metric_stage2.json",
                            "metric_catalog_path": "0054_MA5600T/metric_catalog.json",
                            "derived_metrics_path": None,
                            "metric_field": "uplink_interface_traffic",
                            "metric_name": "Uplink interface traffic",
                            "unit": "Mbps/Gbps",
                            "expression": "avg_6h(uplink_interface_traffic / C_if) > T_if_util",
                            "operator": ">",
                            "consecutive_hours": 1,
                            "rolling_hours": 6,
                            "parameter_name": "T_if_util",
                            "original_threshold": 0.8,
                            "counterfactual_threshold": float(threshold),
                            "counterfactual_threshold_text": format(threshold, "f"),
                            "capacity_mbps": capacity,
                            "interface": interface,
                            "engine": "rolling_mean_utilization",
                        },
                        "window": {
                            "start": format_timestamp(start),
                            "end_exclusive": format_timestamp(end),
                            "hours": 72,
                            "offset_hours": 24,
                            "semantics": "[fault_end + 24h, fault_end + 96h)",
                        },
                        "answer": {
                            "triggered": bool(first),
                            "first_trigger_time": format_timestamp(first) if first else None,
                            "trigger_hours": sum(flags),
                        },
                        "tolerances": {},
                    }
                )
    return pools


def l3_family_pools(
    inventory: CorpusInventory,
    base_l3: list[dict[str, Any]],
    l1_pool: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {
        "multi_device_replay": [_family_copy(task, "multi_device_replay") for task in base_l3],
        "aligned_window_comparison": [],
        "filter_then_compute": [],
    }
    grouped: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for task in l1_pool:
        uid = task["target_device_uids"][0]
        device = inventory.device(uid)
        key = (
            device.model_key,
            task["metric"]["field"],
            task["metric"]["unit"],
            task["window"]["hours"],
        )
        grouped[key].append(task)

    for key, members in sorted(grouped.items()):
        if len({item["target_device_uids"][0] for item in members}) < 3:
            continue
        ordered = sorted(
            members,
            key=lambda item: _order_key(seed, "l3-group", key, item["target_device_uids"][0]),
        )
        unique: list[dict[str, Any]] = []
        seen_uids: set[str] = set()
        for item in ordered:
            uid = item["target_device_uids"][0]
            if uid not in seen_uids:
                unique.append(item)
                seen_uids.add(uid)
        for size in range(3, min(5, len(unique)) + 1):
            locals_ = unique[:size]
            aliases = [f"D{index}" for index in range(1, size + 1)]
            if key[2] == "%":
                increases = {
                    alias: float(local["answer"]["window_mean"])
                    - float(local["anchor"]["anchor_value"])
                    for alias, local in zip(aliases, locals_, strict=True)
                }
                if len(set(increases.values())) == 1:
                    continue
                best = max(increases, key=increases.get)
                med = float(median(increases.values()))
                task = _combine_l1_locals(locals_, aliases, "aligned_window_comparison")
                task["metric"] = copy.deepcopy(locals_[0]["metric"])
                task["anchors"] = {
                    alias: copy.deepcopy(local["anchor"])
                    for alias, local in zip(aliases, locals_, strict=True)
                }
                task["window"] = {
                    "hours": key[3],
                    "per_device": {
                        alias: copy.deepcopy(local["window"])
                        for alias, local in zip(aliases, locals_, strict=True)
                    },
                    "semantics": "per-device post-inspection windows",
                }
                task["answer"] = {
                    "largest_increase_device": best,
                    "largest_increase_percentage_points": _round(increases[best]),
                    "group_median_increase_percentage_points": _round(med),
                    "lead_over_median_percentage_points": _round(increases[best] - med),
                }
                task["tolerances"] = {
                    key: 0.0001
                    for key in (
                        "largest_increase_percentage_points",
                        "group_median_increase_percentage_points",
                        "lead_over_median_percentage_points",
                    )
                }
                pools["aligned_window_comparison"].append(task)

            metadata_fields: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
            for index, local in enumerate(locals_):
                device = inventory.device(local["target_device_uids"][0])
                for field, value in device.metadata.items():
                    if field in {"Device name", "Device ID", "Device tag"} or not value:
                        continue
                    metadata_fields[(field, value)].append(index)
            viable = [
                (field, value, indices)
                for (field, value), indices in metadata_fields.items()
                if 2 <= len(indices) < size
                and all(
                    _locator_hides_filter(
                        local["locators"]["A"],
                        filter_field=field,
                        filter_value=value,
                    )
                    for local in locals_
                )
            ]
            if not viable:
                continue
            field, value, indices = min(
                viable,
                key=lambda item: _order_key(seed, "filter", key, item[0], item[1]),
            )
            selected_aliases = [aliases[index] for index in indices]
            means = {
                aliases[index]: float(locals_[index]["answer"]["window_mean"]) for index in indices
            }
            if len(set(means.values())) == 1:
                continue
            highest = max(means, key=means.get)
            task = _combine_l1_locals(locals_, aliases, "filter_then_compute")
            task["metric"] = copy.deepcopy(locals_[0]["metric"])
            task["filter"] = {"field": field, "value": value}
            task["anchors"] = {
                alias: copy.deepcopy(local["anchor"])
                for alias, local in zip(aliases, locals_, strict=True)
            }
            task["window"] = {
                "hours": key[3],
                "per_device": {
                    alias: copy.deepcopy(local["window"])
                    for alias, local in zip(aliases, locals_, strict=True)
                },
                "semantics": "filter archives, then compute per-device post-inspection mean",
            }
            task["answer"] = {
                "selected_devices": selected_aliases,
                "highest_mean_device": highest,
                "mean_span": _round(max(means.values()) - min(means.values())),
                "unit": key[2],
            }
            task["tolerances"] = {"mean_span": 0.0001}
            pools["filter_then_compute"].append(task)
    return pools


def _combine_l1_locals(
    locals_: list[dict[str, Any]], aliases: list[str], family: str
) -> dict[str, Any]:
    return {
        "task_level": "L3",
        "task_family": family,
        "target_device_uids": [local["target_device_uids"][0] for local in locals_],
        "device_aliases": {
            alias: local["target_device_uids"][0]
            for alias, local in zip(aliases, locals_, strict=True)
        },
        "locators": {
            alias: copy.deepcopy(local["locators"]["A"])
            for alias, local in zip(aliases, locals_, strict=True)
        },
        "report_paths": {
            alias: local["report_paths"]["A"] for alias, local in zip(aliases, locals_, strict=True)
        },
        "rule": None,
    }


def _calibration_priority_order(
    family: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose diverse decision branches before the general diversity passes."""

    if family == "context_conditioned_replay":
        strata = [
            [task for task in candidates if task["answer"]["triggered"]],
            [
                task
                for task in candidates
                if not task["answer"]["triggered"]
                and task["answer"]["load_category"] == "low"
            ],
            [
                task
                for task in candidates
                if not task["answer"]["triggered"]
                and task["answer"]["load_category"] == "high"
            ],
            [
                task
                for task in candidates
                if not task["answer"]["triggered"]
                and task["answer"]["load_category"] == "normal"
            ],
        ]
    elif family == "heterogeneous_metric_comparison":
        sizes = sorted({len(task["target_device_uids"]) for task in candidates})
        strata = [
            [task for task in candidates if len(task["target_device_uids"]) == size]
            for size in sizes
        ]
    else:
        return candidates
    return [
        task
        for group in itertools.zip_longest(*strata)
        for task in group
        if task is not None
    ]


def _selection_signature(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(task["target_device_uids"]),
        tuple(task["report_paths"].values()),
        str(task.get("window")),
        str(task.get("rule")),
    )


def _select_calibration_tasks(
    family: str,
    candidates: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]] | None:
    """Select a small calibration set that actually covers its designed branches."""

    predicates: list[Any]
    if family == "context_conditioned_replay":
        if count % 6 == 0:
            quota = count // 6
            strata = [
                (category, triggered)
                for category in ("high", "normal", "low")
                for triggered in (True, False)
            ]
            availability = {
                (category, triggered): sum(
                    task["answer"]["load_category"] == category
                    and bool(task["answer"]["triggered"]) is triggered
                    for task in candidates
                )
                for category, triggered in strata
            }
            insufficient = {
                f"{category}/{str(triggered).lower()}": available
                for (category, triggered), available in availability.items()
                if available < quota
            }
            if insufficient:
                raise RuntimeError(
                    "Context-conditioned replay cannot fill its balanced strata: "
                    f"needs {quota} per cell, available {insufficient}"
                )

            chosen: list[dict[str, Any]] = []
            signatures: set[tuple[Any, ...]] = set()
            used_devices: set[str] = set()
            used_models: set[str] = set()
            factor_counts: defaultdict[float, int] = defaultdict(int)
            offset_counts: defaultdict[int, int] = defaultdict(int)
            # Scarce cells are allocated first. Within each cell, prefer new
            # devices/models and underrepresented intervention settings.
            for category, triggered in sorted(
                strata,
                key=lambda key: (availability[key], key[0], key[1]),
            ):
                for _ in range(quota):
                    eligible = [
                        task
                        for task in candidates
                        if task["answer"]["load_category"] == category
                        and bool(task["answer"]["triggered"]) is triggered
                        and _selection_signature(task) not in signatures
                    ]
                    if not eligible:
                        raise RuntimeError(
                            f"Context-conditioned replay exhausted {category}/{triggered} "
                            f"after selecting {len(chosen)} tasks"
                        )
                    indexed = list(enumerate(eligible))
                    _, task = max(
                        indexed,
                        key=lambda pair: (
                            pair[1]["target_device_uids"][0] not in used_devices,
                            pair[1]["target_device_uids"][0].split("/", 1)[0]
                            not in used_models,
                            -factor_counts[float(pair[1]["context_decision"]["factor"])],
                            -offset_counts[int(pair[1]["window"]["offset_hours"])],
                            -pair[0],
                        ),
                    )
                    chosen.append(copy.deepcopy(task))
                    signatures.add(_selection_signature(task))
                    uid = task["target_device_uids"][0]
                    used_devices.add(uid)
                    used_models.add(uid.split("/", 1)[0])
                    factor_counts[float(task["context_decision"]["factor"])] += 1
                    offset_counts[int(task["window"]["offset_hours"])] += 1
            return chosen

        predicates = [
            lambda task: bool(task["answer"]["triggered"]),
            lambda task: bool(task["answer"]["triggered"]),
            lambda task: (
                not task["answer"]["triggered"]
                and task["answer"]["load_category"] == "low"
            ),
            lambda task: (
                not task["answer"]["triggered"]
                and task["answer"]["load_category"] == "high"
            ),
            lambda task: (
                not task["answer"]["triggered"]
                and task["answer"]["load_category"] == "normal"
            ),
        ]
    elif family == "heterogeneous_metric_comparison":
        desired_sizes = [3, 4, 5, 3, 4]
        predicates = [
            lambda task, size=size: len(task["target_device_uids"]) == size
            for size in desired_sizes
        ]
    else:
        return None

    chosen: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...]] = set()
    used_devices: set[str] = set()
    for predicate in predicates[:count]:
        eligible = [
            task
            for task in candidates
            if predicate(task) and _selection_signature(task) not in signatures
        ]
        if not eligible:
            continue
        # Prefer a task that introduces the most new devices.  Stable candidate
        # ordering remains the tie-breaker, so selection is deterministic.
        task = max(
            eligible,
            key=lambda item: len(set(item["target_device_uids"]) - used_devices),
        )
        chosen.append(copy.deepcopy(task))
        signatures.add(_selection_signature(task))
        used_devices.update(task["target_device_uids"])

    for task in candidates:
        if len(chosen) >= count:
            break
        signature = _selection_signature(task)
        if signature in signatures:
            continue
        chosen.append(copy.deepcopy(task))
        signatures.add(signature)
    return chosen


def select_family_tasks(
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    quotas: dict[str, dict[str, int]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for level in ("L1", "L2", "L3"):
        for family, count in quotas[level].items():
            candidates = pools[level].get(family, [])
            if len(candidates) < count:
                raise RuntimeError(
                    f"Candidate pool {level}/{family} has {len(candidates)} items, needs {count}"
                )
            ordered = sorted(
                candidates,
                key=lambda task: _order_key(
                    seed,
                    "select-family",
                    level,
                    family,
                    *task["target_device_uids"],
                    *task["report_paths"].values(),
                    task.get("window"),
                    task.get("rule"),
                ),
            )
            ordered = _calibration_priority_order(family, ordered)
            calibration_choice = _select_calibration_tasks(family, ordered, count)
            if calibration_choice is not None:
                selected.extend(calibration_choice)
                continue
            if level == "L2":
                by_outcome = {
                    outcome: [
                        task
                        for task in ordered
                        if bool(task.get("answer", {}).get("triggered")) is outcome
                    ]
                    for outcome in (True, False)
                }
                ordered = [
                    task
                    for pair in itertools.zip_longest(by_outcome[True], by_outcome[False])
                    for task in pair
                    if task is not None
                ]
            chosen: list[dict[str, Any]] = []
            used_models: set[str] = set()
            used_devices: set[str] = set()
            for require_new_models, require_new_devices in (
                (True, True),
                (False, True),
                (False, False),
            ):
                for task in ordered:
                    if len(chosen) >= count:
                        break
                    signature = _selection_signature(task)
                    if any(
                        existing.get("_selection_signature") == signature for existing in chosen
                    ):
                        continue
                    model_keys = {
                        inventory_key.split("/", 1)[0]
                        for inventory_key in task["target_device_uids"]
                    }
                    target_uids = set(task["target_device_uids"])
                    if require_new_models and model_keys & used_models:
                        continue
                    if require_new_devices and target_uids & used_devices:
                        continue
                    item = copy.deepcopy(task)
                    item["_selection_signature"] = signature
                    chosen.append(item)
                    used_models.update(model_keys)
                    used_devices.update(target_uids)
                if len(chosen) >= count:
                    break
            for item in chosen:
                item.pop("_selection_signature", None)
            selected.extend(chosen)
    counters = defaultdict(int)
    for task in selected:
        counters[task["task_level"]] += 1
        task["pair_id"] = f"PB-{task['task_level']}-{counters[task['task_level']]:03d}"
    return selected


def compact_candidate_manifest(
    pools: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in ("L1", "L2", "L3"):
        for family, candidates in pools[level].items():
            for index, task in enumerate(candidates, start=1):
                rows.append(
                    {
                        "candidate_id": f"{level}-{family}-{index:06d}",
                        "task_level": level,
                        "task_family": family,
                        "target_device_uids": task["target_device_uids"],
                        "report_paths": task["report_paths"],
                        "window": task.get("window"),
                        "rule_id": (task.get("rule") or {}).get("rule_id"),
                        "answer_signature": task.get("answer"),
                    }
                )
    return rows


__all__ = [
    "compact_candidate_manifest",
    "l1_family_pools",
    "l2_family_pools",
    "l3_family_pools",
    "select_family_tasks",
]