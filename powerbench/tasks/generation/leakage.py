from __future__ import annotations

import csv
import itertools
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ..documents import parse_document
from ..models import CorpusInventory
from ..selector import (
    contains_selector_token,
    json_record_matches,
    markdown_section_chunks,
    matching_json_objects,
    matching_markdown_key_values,
    matching_markdown_table_rows,
    required_text_matches_one_block,
)
from ..selector import (
    normalize_text as _normalized_text,
)
from ..selector import (
    normalized_rule_id as _normalized_rule_id,
)
from ..selector import (
    selector_strings as _selector_strings,
)
from ..util import canonical_json, parse_timestamp, safe_join

_MARKDOWN_SELECTOR_KEYS = {
    "field",
    "fields",
    "key_values",
    "metric",
    "required_text",
    "section",
    "table_row",
}
_JSON_SELECTOR_KEYS = {
    "derived_key",
    "field",
    "fields",
    "json_record",
    "required_text",
    "rule_id",
}
_CSV_SELECTOR_KEYS = {"end_exclusive", "field", "fields", "lookback_hours", "start"}
_NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])")
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.IGNORECASE | re.DOTALL)

_ANSWER_LABELS: dict[str, tuple[str, ...]] = {
    "window_mean": ("window_mean", "window mean", "mean"),
    "window_max": ("window_max", "window max", "maximum"),
    "first_max_time": ("first_max_time", "first max time", "first maximum time"),
    "hours_above_anchor_value": (
        "hours_above_anchor_value",
        "hours above anchor",
        "hours above baseline",
    ),
    "unit": ("unit",),
    "triggered": ("triggered", "trigger status"),
    "first_trigger_time": ("first_trigger_time", "first trigger time"),
    "first_trigger_offset_hours": (
        "first_trigger_offset_hours",
        "first trigger offset hours",
        "first offset",
    ),
    "trigger_hours": ("trigger_hours", "trigger hours"),
    "best_consecutive_margin": (
        "best_consecutive_margin",
        "best consecutive margin",
        "consecutive margin",
    ),
    "triggered_devices": ("triggered_devices", "triggered devices"),
    "earliest_trigger_device": (
        "earliest_trigger_device",
        "earliest trigger device",
        "smallest offset device",
    ),
    "trigger_offset_span_hours": (
        "trigger_offset_span_hours",
        "trigger offset span",
        "first offset span",
    ),
    "exceedance_hours": ("exceedance_hours", "total exceedance hours"),
    "longest_exceedance_run_hours": (
        "longest_exceedance_run_hours",
        "longest consecutive exceedance hours",
    ),
    "first_exceedance_time": ("first_exceedance_time", "first exceedance time"),
    "pre_mean": ("pre_mean", "pre-window mean"),
    "post_mean": ("post_mean", "post-window mean"),
    "change_rate_percent": ("change_rate_percent", "change rate"),
    "post_max": ("post_max", "post-window max"),
    "first_post_max_time": ("first_post_max_time", "first post-window max time"),
    "capacity_mbps": ("capacity_mbps", "rated bandwidth"),
    "mean_utilization_percent": ("mean_utilization_percent", "mean utilization"),
    "longest_above_utilization_run_hours": (
        "longest_above_utilization_run_hours",
        "longest consecutive high-utilization hours",
    ),
    "first_above_utilization_time": (
        "first_above_utilization_time",
        "first high-utilization time",
    ),
    "largest_increase_device": ("largest_increase_device", "largest increase device"),
    "largest_increase_percentage_points": (
        "largest_increase_percentage_points",
        "largest increase",
    ),
    "group_median_increase_percentage_points": (
        "group_median_increase_percentage_points",
        "group median increase",
    ),
    "lead_over_median_percentage_points": (
        "lead_over_median_percentage_points",
        "lead over group median",
    ),
    "selected_devices": ("selected_devices", "filtered devices"),
    "highest_mean_device": ("highest_mean_device", "highest mean device"),
    "mean_span": ("mean_span", "mean span"),
    "margin_order": (
        "margin_order",
        "consecutive margin order",
        "margin order",
    ),
    "margin_span": (
        "margin_span",
        "consecutive margin span",
        "max-min margin span",
    ),
}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    raise TypeError(f"Expected a mapping-like record, received {type(value).__name__}")


def _iter_nodes(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_nodes(child)


class _SourceValidator:
    def __init__(self, inventory: CorpusInventory) -> None:
        self.inventory = inventory
        self.root = inventory.raw_root.resolve()
        self._markdown: dict[str, tuple[str, dict[str, Any]]] = {}
        self._json: dict[str, Any] = {}
        self._csv_checks: dict[tuple[str, str], list[str]] = {}

    def validate(self, source: Mapping[str, Any]) -> list[str]:
        relative = str(source.get("path") or "")
        selector = source.get("selector")
        if not isinstance(selector, dict):
            return ["selector_is_not_an_object"]
        if not selector:
            return ["empty_selector"]
        try:
            path = safe_join(self.root, relative)
        except (TypeError, ValueError):
            return ["unsafe_or_invalid_path"]
        if not path.is_file():
            return ["path_missing"]
        artifact = self.inventory.artifacts.get(relative)
        if artifact is None:
            return ["path_not_in_public_inventory"]
        if artifact.get("public") is False:
            return ["source_is_not_public"]

        suffix = path.suffix.casefold()
        if suffix == ".md":
            return self._validate_markdown(relative, path, selector)
        if suffix == ".json":
            return self._validate_json(relative, path, selector)
        if suffix == ".csv":
            return self._validate_csv(relative, path, selector)
        return [f"unsupported_source_type:{suffix or '<none>'}"]

    @staticmethod
    def _unsupported(selector: Mapping[str, Any], allowed: set[str]) -> list[str]:
        return [f"unsupported_selector_key:{key}" for key in sorted(set(selector) - allowed)]

    def _validate_markdown(
        self, relative: str, path: Path, selector: Mapping[str, Any]
    ) -> list[str]:
        reasons = self._unsupported(selector, _MARKDOWN_SELECTOR_KEYS)
        try:
            if relative not in self._markdown:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                self._markdown[relative] = (text, parse_document(path))
            text, parsed = self._markdown[relative]
        except OSError as exc:
            return [*reasons, f"markdown_read_error:{type(exc).__name__}"]

        section = selector.get("section")
        if section is not None and not markdown_section_chunks(text, section):
            reasons.append("markdown_section_missing")

        field = selector.get("field")
        if field is not None:
            keys = {_normalized_text(key) for key in parsed.get("key_values", {})}
            if _normalized_text(field) not in keys:
                reasons.append("markdown_field_missing")

        fields = selector.get("fields")
        if fields is not None:
            wanted_fields = [fields] if isinstance(fields, str) else fields
            if not isinstance(wanted_fields, (list, tuple, set)):
                reasons.append("markdown_fields_is_not_a_list")
            else:
                keys = {_normalized_text(key) for key in parsed.get("key_values", {})}
                missing = [
                    str(item) for item in wanted_fields if _normalized_text(item) not in keys
                ]
                if missing:
                    reasons.append(f"markdown_fields_missing:{','.join(missing)}")

        metric = selector.get("metric")
        if metric is not None:
            wanted_metric = _normalized_text(metric)
            matching_tables = []
            for table in parsed.get("tables", []):
                table_section = _normalized_text(table.get("section", ""))
                if section is None or _normalized_text(section) in table_section:
                    matching_tables.append(table)
            found = any(
                wanted_metric == _normalized_text(cell)
                for table in matching_tables
                for row in table.get("rows", [])
                for cell in row.values()
            )
            if not found:
                reasons.append("markdown_metric_row_missing")

        key_values = selector.get("key_values")
        if key_values is not None:
            if not isinstance(key_values, dict) or not key_values:
                reasons.append("markdown_key_values_invalid")
            else:
                matched = any(
                    matching_markdown_key_values(chunk, key_values) is not None
                    for chunk in markdown_section_chunks(text, section)
                )
                if not matched:
                    reasons.append(
                        f"markdown_key_values_missing:{','.join(str(key) for key in key_values)}"
                    )

        table_row = selector.get("table_row")
        if table_row is not None:
            if not isinstance(table_row, dict) or not isinstance(table_row.get("match"), dict):
                reasons.append("markdown_table_row_invalid")
            elif not matching_markdown_table_rows(text, table_row, section=section):
                reasons.append("markdown_table_row_missing")

        required = _selector_strings(selector.get("required_text"))
        required_matches = any(
            required_text_matches_one_block(chunk, required)
            for chunk in markdown_section_chunks(text, section)
        )
        if "required_text" in selector and not required:
            reasons.append("required_text_is_empty")
        elif required and not required_matches:
            reasons.append(f"required_text_missing:{','.join(required)}")
        return reasons

    def _validate_json(self, relative: str, path: Path, selector: Mapping[str, Any]) -> list[str]:
        reasons = self._unsupported(selector, _JSON_SELECTOR_KEYS)
        try:
            if relative not in self._json:
                with path.open("r", encoding="utf-8-sig") as handle:
                    self._json[relative] = json.load(handle)
            payload = self._json[relative]
        except (OSError, json.JSONDecodeError) as exc:
            return [*reasons, f"json_read_error:{type(exc).__name__}"]

        field = selector.get("field")
        if field is not None:
            wanted = str(field)
            found = any(
                isinstance(node, dict)
                and (
                    wanted in node
                    or str(node.get("database_field", "")) == wanted
                    or str(node.get("variable_name", "")) == wanted
                )
                for node in _iter_nodes(payload)
            )
            if not found:
                reasons.append("json_field_missing")

        fields = selector.get("fields")
        if fields is not None:
            wanted_fields = [fields] if isinstance(fields, str) else fields
            if not isinstance(wanted_fields, (list, tuple, set)):
                reasons.append("json_fields_is_not_a_list")
            else:
                dictionary_keys = {
                    str(key)
                    for node in _iter_nodes(payload)
                    if isinstance(node, dict)
                    for key in node
                }
                missing = [str(item) for item in wanted_fields if str(item) not in dictionary_keys]
                if missing:
                    reasons.append(f"json_fields_missing:{','.join(missing)}")

        rule_id = selector.get("rule_id")
        if rule_id is not None:
            wanted_rule = _normalized_rule_id(rule_id)
            found = any(
                isinstance(node, dict)
                and _normalized_rule_id(node.get("rule_id", "")) == wanted_rule
                for node in _iter_nodes(payload)
            )
            if not found:
                reasons.append("json_rule_id_missing")

        derived_key = selector.get("derived_key")
        if derived_key is not None:
            # A key listed only under `removed` is not an available definition.
            active = payload.get("derived_metrics") if isinstance(payload, dict) else None
            found = isinstance(active, list) and any(
                isinstance(item, dict) and str(item.get("derived_key", "")) == str(derived_key)
                for item in active
            )
            if not found:
                reasons.append("json_active_derived_key_missing")

        json_record = selector.get("json_record")
        if json_record is not None:
            if not isinstance(json_record, dict) or not json_record:
                reasons.append("json_record_invalid")
            elif not any(
                isinstance(node, dict) and json_record_matches(node, json_record)
                for node in _iter_nodes(payload)
            ):
                reasons.append("json_record_missing")

        if "required_text" in selector:
            required = _selector_strings(selector.get("required_text"))
            if not required:
                reasons.append("required_text_is_empty")
            elif not matching_json_objects(payload, selector):
                missing = [
                    token
                    for token in required
                    if not contains_selector_token(json.dumps(payload, ensure_ascii=False), token)
                ]
                reasons.append(f"required_text_missing:{','.join(missing or required)}")
        return reasons

    def _validate_csv(self, relative: str, path: Path, selector: Mapping[str, Any]) -> list[str]:
        signature = (relative, canonical_json(dict(selector)))
        if signature in self._csv_checks:
            return list(self._csv_checks[signature])
        reasons = self._unsupported(selector, _CSV_SELECTOR_KEYS)

        raw_fields = selector.get("fields", selector.get("field"))
        if isinstance(raw_fields, str):
            required_fields = [raw_fields]
        elif isinstance(raw_fields, (list, tuple, set)):
            required_fields = [str(value) for value in raw_fields if str(value)]
        else:
            required_fields = []
        if not required_fields:
            reasons.append("csv_field_selector_missing")

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = list(reader.fieldnames or [])
                missing_fields = [field for field in required_fields if field not in headers]
                if missing_fields:
                    reasons.append(f"csv_fields_missing:{','.join(missing_fields)}")

                has_start = "start" in selector
                has_end = "end_exclusive" in selector
                if has_start != has_end:
                    reasons.append("csv_interval_is_incomplete")
                elif has_start and has_end and not missing_fields:
                    try:
                        lookback = int(selector.get("lookback_hours") or 0)
                        if lookback < 0:
                            raise ValueError("negative lookback")
                        required_start = parse_timestamp(str(selector["start"])) - timedelta(
                            hours=lookback
                        )
                        required_end = parse_timestamp(str(selector["end_exclusive"]))
                        seconds = (required_end - required_start).total_seconds()
                        if required_start >= required_end or seconds % 3600:
                            raise ValueError("non-hourly interval")
                    except (TypeError, ValueError):
                        reasons.append("csv_interval_is_invalid")
                    else:
                        expected_count = int(seconds // 3600)
                        expected = {
                            required_start + timedelta(hours=offset)
                            for offset in range(expected_count)
                        }
                        observed: Counter[Any] = Counter()
                        blank_values: set[Any] = set()
                        for row in reader:
                            try:
                                timestamp = parse_timestamp(str(row.get("timestamp", "")))
                            except (TypeError, ValueError):
                                continue
                            if required_start <= timestamp < required_end:
                                observed[timestamp] += 1
                                if any(
                                    not str(row.get(field, "")).strip() for field in required_fields
                                ):
                                    blank_values.add(timestamp)
                        missing_times = expected.difference(observed)
                        duplicate_times = {time for time, count in observed.items() if count != 1}
                        unexpected_times = set(observed).difference(expected)
                        if missing_times:
                            reasons.append(f"csv_interval_missing_hours:{len(missing_times)}")
                        if duplicate_times:
                            reasons.append(f"csv_interval_duplicate_hours:{len(duplicate_times)}")
                        if unexpected_times:
                            reasons.append(f"csv_interval_non_hourly_rows:{len(unexpected_times)}")
                        if blank_values:
                            reasons.append(f"csv_interval_blank_values:{len(blank_values)}")
        except (OSError, csv.Error) as exc:
            reasons.append(f"csv_read_error:{type(exc).__name__}")

        self._csv_checks[signature] = list(reasons)
        return reasons


def _minimum_cover(source_sets: list[set[str]]) -> tuple[int | None, list[str]]:
    if not source_sets or any(not paths for paths in source_sets):
        return None, []
    universe = sorted(set().union(*source_sets))
    for size in range(1, len(universe) + 1):
        for choice in itertools.combinations(universe, size):
            selected = set(choice)
            if all(selected.intersection(paths) for paths in source_sets):
                return size, list(choice)
    return None, []


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _values_equal(expected: Any, actual: Any, *, path: str = "") -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if expected is None:
        return actual is None
    if isinstance(expected, (int, float, Decimal)) and not isinstance(expected, bool):
        left = _decimal(expected)
        right = _decimal(actual)
        return left is not None and right is not None and left == right
    if isinstance(expected, str):
        return isinstance(actual, str) and _normalized_text(expected) == _normalized_text(actual)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        if path.endswith("triggered_devices"):
            return Counter(map(str, expected)) == Counter(map(str, actual))
        return all(
            _values_equal(left, right, path=f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
        )
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual
            and _values_equal(value, actual[key], path=f"{path}.{key}" if path else key)
            for key, value in expected.items()
        )
    return expected == actual


def _json_answer_block(payload: Any, answer: dict[str, Any]) -> Any | None:
    for node in _iter_nodes(payload):
        if isinstance(node, dict) and _values_equal(answer, node):
            return node
    return None


def _contains_scalar(text: str, value: Any) -> bool:
    normalized = _normalized_text(text)
    if isinstance(value, bool):
        tokens = ("true", "yes", "triggered") if value else ("false", "no", "not triggered")
        return any(token in normalized for token in tokens)
    if value is None:
        return any(token in normalized for token in ("null", "none", "n/a", "not triggered", "not applicable"))
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        wanted = _decimal(value)
        if wanted is None:
            return False
        return any(_decimal(match.group(0)) == wanted for match in _NUMBER_RE.finditer(normalized))
    if isinstance(value, list):
        return all(_contains_scalar(normalized, item) for item in value)
    if isinstance(value, str) and re.fullmatch(r"D[1-5]", value, flags=re.IGNORECASE):
        alias = re.escape(value.casefold())
        return re.search(rf"(?<![a-z0-9]){alias}(?![a-z0-9])", normalized) is not None
    return _normalized_text(value) in normalized


def _has_label_value(block: str, labels: Iterable[str], value: Any) -> bool:
    normalized = _normalized_text(block)
    for label in labels:
        wanted = _normalized_text(label)
        start = 0
        while wanted and (position := normalized.find(wanted, start)) >= 0:
            tail = normalized[position + len(wanted) : position + len(wanted) + 240]
            if _contains_scalar(tail, value):
                return True
            start = position + len(wanted)
    return False


def _markdown_answer_block(
    block: str, level: str, answer: dict[str, Any], _task: Mapping[str, Any]
) -> bool:
    if answer and all(field in _ANSWER_LABELS for field in answer):
        return all(
            _has_label_value(block, _ANSWER_LABELS[field], value) for field, value in answer.items()
        )
    if level == "L1":
        fields = (
            "window_mean",
            "window_max",
            "first_max_time",
            "hours_above_anchor_value",
            "unit",
        )
        return all(
            field in answer and _has_label_value(block, _ANSWER_LABELS[field], answer[field])
            for field in fields
        )
    if level == "L2":
        fields = (
            "triggered",
            "first_trigger_time",
            "trigger_hours",
            "best_consecutive_margin",
        )
        return all(
            field in answer and _has_label_value(block, _ANSWER_LABELS[field], answer[field])
            for field in fields
        )
    if level != "L3":
        return False

    aggregate_fields = (
        "triggered_devices",
        "earliest_trigger_device",
        "trigger_offset_span_hours",
        "margin_order",
        "margin_span",
    )
    if not all(
        field in answer and _has_label_value(block, _ANSWER_LABELS[field], answer[field])
        for field in aggregate_fields
    ):
        return False
    local = answer.get("per_device")
    if isinstance(local, list):
        local_by_alias = {
            str(row.get("alias")): row
            for row in local
            if isinstance(row, dict) and isinstance(row.get("alias"), str)
        }
    elif isinstance(local, dict):
        local_by_alias = local
    else:
        return False
    local_fields = (
        "triggered",
        "first_trigger_time",
        "first_trigger_offset_hours",
        "trigger_hours",
        "best_consecutive_margin",
    )
    normalized_block = _normalized_text(block)
    if not all(
        any(_normalized_text(label) in normalized_block for label in _ANSWER_LABELS[field])
        for field in local_fields
    ):
        return False
    lines = block.splitlines()
    for alias, values in local_by_alias.items():
        if not isinstance(values, dict):
            return False
        alias_re = re.compile(rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])")
        if not any(
            alias_re.search(_normalized_text(line))
            and all(
                field in values and _contains_scalar(line, values[field]) for field in local_fields
            )
            for line in lines
        ):
            return False
    return True


def _markdown_blocks(text: str) -> Iterable[str]:
    for match in re.finditer(r"(?:\A|\n\s*\n)(.*?)(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        block = match.group(1).strip()
        if block:
            yield block


def _embedded_json_blocks(block: str) -> Iterable[dict[str, Any]]:
    candidates = [match.group(1) for match in _FENCED_JSON_RE.finditer(block)]
    stripped = block.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _excerpt(value: Any, *, limit: int = 1800) -> str:
    text = (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    return text if len(text) <= limit else f"{text[:limit]}…"


def _target_text_artifacts(
    gold: Mapping[str, Any], inventory: CorpusInventory
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    targets = {str(uid) for uid in gold.get("target_device_uids", [])}
    unknown = sorted(targets.difference(inventory.devices))
    if unknown:
        warnings.append(f"unknown_target_device_uids:{','.join(unknown)}")
    model_keys = {inventory.device(uid).model_key for uid in targets if uid in inventory.devices}
    paths = []
    for path, artifact in inventory.artifacts.items():
        if artifact.get("public") is False or Path(path).suffix.casefold() not in {".md", ".json"}:
            continue
        device_uid = artifact.get("device_uid")
        model_key = artifact.get("model_key")
        if device_uid in targets or (device_uid is None and model_key in model_keys):
            paths.append(path)
    return sorted(set(paths)), warnings


def _answer_hits_for_record(
    gold: Mapping[str, Any],
    task: Mapping[str, Any],
    inventory: CorpusInventory,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    answer = gold.get("answer")
    if not isinstance(answer, dict) or not answer:
        return [], [
            {
                "qa_id": gold.get("qa_id"),
                "code": "answer_scan_skipped",
                "detail": "Gold answer is empty or is not an object",
            }
        ]
    paths, path_warnings = _target_text_artifacts(gold, inventory)
    warnings = [
        {"qa_id": gold.get("qa_id"), "code": "target_artifact_warning", "detail": value}
        for value in path_warnings
    ]
    hits: list[dict[str, Any]] = []
    level = str(gold.get("task_level") or task.get("task_level") or "")
    for relative in paths:
        try:
            path = safe_join(inventory.raw_root.resolve(), relative)
            if path.suffix.casefold() == ".json":
                with path.open("r", encoding="utf-8-sig") as handle:
                    payload = json.load(handle)
                block = _json_answer_block(payload, answer)
                if block is not None:
                    hits.append(
                        {
                            "qa_id": gold.get("qa_id"),
                            "pair_id": gold.get("pair_id"),
                            "path": relative,
                            "block_type": "json_object",
                            "block": _excerpt(block),
                        }
                    )
                continue

            text = path.read_text(encoding="utf-8-sig", errors="replace")
            for block in _markdown_blocks(text):
                embedded = next(
                    (
                        found
                        for payload in _embedded_json_blocks(block)
                        if (found := _json_answer_block(payload, answer)) is not None
                    ),
                    None,
                )
                if embedded is not None:
                    hits.append(
                        {
                            "qa_id": gold.get("qa_id"),
                            "pair_id": gold.get("pair_id"),
                            "path": relative,
                            "block_type": "embedded_json_object",
                            "block": _excerpt(embedded),
                        }
                    )
                    break
                if _markdown_answer_block(block, level, answer, task):
                    hits.append(
                        {
                            "qa_id": gold.get("qa_id"),
                            "pair_id": gold.get("pair_id"),
                            "path": relative,
                            "block_type": "labelled_markdown_tuple",
                            "block": _excerpt(block),
                        }
                    )
                    break
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(
                {
                    "qa_id": gold.get("qa_id"),
                    "code": "answer_scan_read_error",
                    "path": relative,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    return hits, warnings


def audit_shortcuts(
    tasks: Iterable[dict[str, Any]],
    gold_records: Iterable[dict[str, Any]],
    inventory: CorpusInventory,
) -> dict[str, Any]:
    """Independently audit evidence closure and direct-answer shortcuts.

    CSV files are inspected only as primitive inputs: a raw value is never
    considered a final-answer leak. Hard answer hits require a complete Gold
    object or a complete, labelled tuple within one Markdown block.
    """

    task_rows = [_as_dict(task) for task in tasks]
    gold_rows = [_as_dict(record) for record in gold_records]
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    tasks_by_pair: dict[str, dict[str, Any]] = {}
    for task in task_rows:
        pair_id = str(task.get("pair_id") or "")
        if not pair_id:
            errors.append({"code": "task_missing_pair_id"})
        elif pair_id in tasks_by_pair:
            errors.append({"code": "duplicate_task_pair_id", "pair_id": pair_id})
        else:
            tasks_by_pair[pair_id] = task

    validator = _SourceValidator(inventory)
    invalid_alternatives: list[dict[str, Any]] = []
    minimum_covers: list[dict[str, Any]] = []
    single_file_records: list[dict[str, Any]] = []
    exact_answer_hits: list[dict[str, Any]] = []

    # Direct/Indirect variants share targets and answers, so cache raw scanning
    # by semantic pair and copy hits with the variant's qa_id.
    answer_scan_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for record in gold_rows:
        qa_id = str(record.get("qa_id") or "")
        pair_id = str(record.get("pair_id") or "")
        task = tasks_by_pair.get(pair_id)
        if task is None:
            errors.append(
                {"code": "gold_without_semantic_task", "qa_id": qa_id, "pair_id": pair_id}
            )
            task = {}

        atom_sets: list[set[str]] = []
        atoms = record.get("evidence_atoms")
        if not isinstance(atoms, list) or not atoms:
            errors.append({"code": "gold_has_no_evidence_atoms", "qa_id": qa_id})
            atoms = []
        seen_atom_ids: set[str] = set()
        for atom in atoms:
            atom_id = str(atom.get("atom_id") or "") if isinstance(atom, dict) else ""
            if not isinstance(atom, dict):
                errors.append({"code": "evidence_atom_is_not_an_object", "qa_id": qa_id})
                atom_sets.append(set())
                continue
            if not atom_id or atom_id in seen_atom_ids:
                errors.append(
                    {"code": "missing_or_duplicate_atom_id", "qa_id": qa_id, "atom_id": atom_id}
                )
            seen_atom_ids.add(atom_id)
            valid_paths: set[str] = set()
            sources = atom.get("acceptable_sources")
            if not isinstance(sources, list) or not sources:
                errors.append(
                    {"code": "atom_has_no_alternatives", "qa_id": qa_id, "atom_id": atom_id}
                )
                atom_sets.append(set())
                continue
            for index, source in enumerate(sources):
                if not isinstance(source, dict):
                    reasons = ["source_is_not_an_object"]
                    path = ""
                    selector: Any = None
                else:
                    reasons = validator.validate(source)
                    path = str(source.get("path") or "")
                    selector = source.get("selector")
                if reasons:
                    invalid_alternatives.append(
                        {
                            "qa_id": qa_id,
                            "pair_id": pair_id,
                            "atom_id": atom_id,
                            "alternative_index": index,
                            "path": path,
                            "selector": selector,
                            "reasons": reasons,
                        }
                    )
                else:
                    valid_paths.add(path)
            atom_sets.append(valid_paths)

        cover_size, cover_paths = _minimum_cover(atom_sets)
        cover_record = {
            "qa_id": qa_id,
            "pair_id": pair_id,
            "validated_primitive_count": len(atom_sets),
            "minimum_distinct_source_files": cover_size,
            "witness_paths": cover_paths,
        }
        minimum_covers.append(cover_record)
        if cover_size == 1:
            single_file_records.append(cover_record)

        answer_key = (pair_id, canonical_json(record.get("answer", {})))
        if answer_key not in answer_scan_cache:
            answer_scan_cache[answer_key] = _answer_hits_for_record(record, task, inventory)
        cached_hits, cached_warnings = answer_scan_cache[answer_key]
        exact_answer_hits.extend([{**hit, "qa_id": qa_id} for hit in cached_hits])
        for warning in cached_warnings:
            issue = {**warning, "qa_id": qa_id}
            if issue.get("code") in {"answer_scan_read_error", "answer_scan_skipped"}:
                errors.append(issue)
            else:
                warnings.append(issue)

    answer_scan_complete = not any(
        error.get("code") in {"answer_scan_read_error", "answer_scan_skipped"} for error in errors
    )
    valid = (
        not errors
        and not invalid_alternatives
        and not single_file_records
        and not exact_answer_hits
    )
    return {
        "valid": valid,
        "gold_record_count": len(gold_rows),
        "invalid_alternative_count": len(invalid_alternatives),
        "invalid_alternatives": invalid_alternatives,
        "minimum_source_cover": minimum_covers,
        "single_file_sufficient_count": len(single_file_records),
        "single_file_sufficient": single_file_records,
        "exact_answer_hit_count": len(exact_answer_hits),
        "exact_answer_hits": exact_answer_hits,
        "answer_scan_complete": answer_scan_complete,
        "errors": errors,
        "warnings": warnings,
    }