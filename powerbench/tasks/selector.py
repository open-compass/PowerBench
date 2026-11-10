from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from .documents import parse_markdown_key_values, parse_markdown_tables

_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
_RULE_ID_RE = re.compile(r"^K\s*0*(\d+)$", flags=re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"[a-z0-9_./:-]+")


def normalize_text(value: Any) -> str:
    """Normalize selector text without erasing token boundaries."""

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def selector_strings(value: Any) -> list[str]:
    """Flatten only non-empty string leaves from a selector value."""

    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [token for item in value for token in selector_strings(item)]
    if isinstance(value, Mapping):
        return [token for item in value.values() for token in selector_strings(item)]
    return []


def contains_selector_token(haystack: Any, token: Any) -> bool:
    """Match one literal selector token while rejecting common false positives.

    Compact ASCII identifiers use explicit boundaries, so ``K6`` does not
    match ``K60`` and ``temp`` does not match ``temp-rate``.  State words also
    reject common negation prefixes (for example, ``not running`` cannot
    establish ``running``).
    """

    normalized_haystack = normalize_text(haystack)
    normalized_token = normalize_text(token)
    if not normalized_token:
        return False
    if _IDENTIFIER_RE.fullmatch(normalized_token):
        identifier_character = r"a-z0-9_./:-"
        pattern = (
            rf"(?<![{identifier_character}]){re.escape(normalized_token)}"
            rf"(?![{identifier_character}])"
        )
        return re.search(pattern, normalized_haystack) is not None
    if re.search(rf"(?<![a-z])not\s+{re.escape(normalized_token)}\b", normalized_haystack):
        return False
    return normalized_token in normalized_haystack


def numeric_equal(left: Any, right: Any, tolerance: Any = 0) -> bool:
    try:
        left_decimal = Decimal(str(left).strip())
        right_decimal = Decimal(str(right).strip())
        tolerance_decimal = Decimal(str(tolerance))
    except (InvalidOperation, TypeError, ValueError):
        return False
    return (
        left_decimal.is_finite()
        and right_decimal.is_finite()
        and tolerance_decimal.is_finite()
        and tolerance_decimal >= 0
        and abs(left_decimal - right_decimal) <= tolerance_decimal
    )


def normalized_rule_id(value: Any) -> str:
    match = _RULE_ID_RE.fullmatch(str(value).strip())
    return f"K{int(match.group(1))}" if match else str(value).strip().upper()


def structured_value_matches(actual: Any, specification: Any) -> bool:
    """Apply the small value predicate language used by evidence selectors."""

    if isinstance(specification, Mapping):
        if set(specification) == {"equals"}:
            return normalize_text(actual) == normalize_text(specification["equals"])
        if set(specification) == {"contains"}:
            tokens = selector_strings(specification["contains"])
            return bool(tokens) and all(contains_selector_token(actual, token) for token in tokens)
        return False
    return normalize_text(actual) == normalize_text(specification)


def iter_json_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_json_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_dicts(child)


def json_record_matches(node: Mapping[str, Any], specification: Mapping[str, Any]) -> bool:
    """Match one complete structured metric/rule definition record."""

    kind = specification.get("kind")
    if kind == "rule":
        if normalized_rule_id(node.get("rule_id")) != normalized_rule_id(
            specification.get("rule_id")
        ):
            return False
        if normalize_text(node.get("rule_expression", "")) != normalize_text(
            specification.get("expression", "")
        ):
            return False
        point = node.get("pointwise_criteria")
        if not isinstance(point, Mapping) or not numeric_equal(
            point.get("consecutive_points"), specification.get("consecutive_hours")
        ):
            return False
        wanted_parameter = specification.get("parameter")
        parameters = node.get("threshold_params")
        return (
            isinstance(wanted_parameter, Mapping)
            and isinstance(parameters, list)
            and any(
                isinstance(parameter, Mapping)
                and str(parameter.get("param_name", "")) == str(wanted_parameter.get("name", ""))
                and numeric_equal(parameter.get("param_value"), wanted_parameter.get("value"))
                for parameter in parameters
            )
        )
    if kind == "metric":
        identity_matches = (
            str(node.get("database_field", "")) == str(specification.get("field", ""))
            and normalize_text(node.get("metric_name", ""))
            == normalize_text(specification.get("name", ""))
            and normalize_text(node.get("unit", ""))
            == normalize_text(specification.get("unit", ""))
        )
        if not identity_matches:
            return False
        normal = node.get("normal_range")
        for source_key, specification_key in (
            ("mean", "normal_mean"),
            ("std", "normal_std"),
        ):
            if specification_key not in specification:
                continue
            if not isinstance(normal, Mapping) or not numeric_equal(
                normal.get(source_key), specification.get(specification_key)
            ):
                return False
        return True
    if kind == "derived_metric":
        atomic_metrics = node.get("atomic_metrics")
        observed_fields = (
            sorted(
                {
                    str(item.get("db_field"))
                    for item in atomic_metrics
                    if isinstance(item, Mapping) and str(item.get("db_field") or "")
                }
            )
            if isinstance(atomic_metrics, list)
            else []
        )
        return (
            str(node.get("derived_key", "")) == str(specification.get("derived_key", ""))
            and normalize_text(node.get("derived_name", ""))
            == normalize_text(specification.get("derived_name", ""))
            and normalize_text(node.get("formula", ""))
            == normalize_text(specification.get("formula", ""))
            and normalize_text(node.get("output_unit", ""))
            == normalize_text(specification.get("output_unit", ""))
            and observed_fields
            == sorted(str(item) for item in specification.get("atomic_fields", []))
        )
    return False


def json_object_matches_selector(node: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    """Match all JSON selector constraints against one coherent object."""

    structured = selector.get("json_record")
    if structured is not None and (
        not isinstance(structured, Mapping) or not json_record_matches(node, structured)
    ):
        return False

    rule_id = selector.get("rule_id")
    if rule_id is not None and normalized_rule_id(node.get("rule_id")) != normalized_rule_id(
        rule_id
    ):
        return False
    field = selector.get("field")
    if field is not None and not (
        str(node.get("database_field", "")) == str(field)
        or str(node.get("variable_name", "")) == str(field)
        or str(field) in node
    ):
        return False
    derived_key = selector.get("derived_key")
    if derived_key is not None and str(node.get("derived_key", "")) != str(derived_key):
        return False

    required = selector_strings(selector.get("required_text"))
    if "required_text" in selector and not required:
        return False
    if required:
        discriminators = (structured, rule_id, field, derived_key)
        direct_only = not any(value is not None for value in discriminators)
        selected = (
            {key: value for key, value in node.items() if not isinstance(value, (dict, list))}
            if direct_only
            else node
        )
        serialized = json.dumps(selected, ensure_ascii=False)
        if not all(contains_selector_token(serialized, token) for token in required):
            return False

    return bool(
        structured is not None
        or rule_id is not None
        or field is not None
        or derived_key is not None
        or required
    )


def matching_json_objects(payload: Any, selector: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        node for node in iter_json_dicts(payload) if json_object_matches_selector(node, selector)
    ]


def json_values_from_content(content: str) -> list[Any]:
    """Parse a complete JSON value or self-contained values in a file slice."""

    try:
        return [json.loads(content)]
    except (json.JSONDecodeError, TypeError):
        pass
    values: list[Any] = []
    stack: list[tuple[str, int]] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index, character in enumerate(content):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append((character, index))
        elif character in "}]":
            if not stack or stack[-1][0] != pairs[character]:
                stack.clear()
                continue
            _, start = stack.pop()
            try:
                values.append(json.loads(content[start : index + 1]))
            except json.JSONDecodeError:
                continue
    return values


def json_selector_matches(content: str, selector: Mapping[str, Any]) -> bool:
    payloads = json_values_from_content(content)
    if not payloads:
        return False
    if not selector:
        return True
    return any(matching_json_objects(payload, selector) for payload in payloads)


def markdown_section_chunks(content: str, section: Any) -> list[str]:
    if section is None:
        return [content]
    wanted = normalize_text(section)
    if not wanted:
        return []
    lines = content.splitlines()
    chunks: list[str] = []
    start: int | None = None
    level: int | None = None
    for index, line in enumerate(lines):
        match = _MARKDOWN_HEADING_RE.match(line)
        if not match:
            continue
        current_level = len(match.group("marks"))
        if start is not None and level is not None and current_level <= level:
            chunks.append("\n".join(lines[start:index]))
            start = None
            level = None
        if start is None and wanted in normalize_text(match.group("title")):
            start = index
            level = current_level
    if start is not None:
        chunks.append("\n".join(lines[start:]))
    return chunks


def matching_markdown_key_values(content: str, specification: Any) -> dict[str, str] | None:
    if not isinstance(specification, Mapping) or not specification:
        return None
    values = parse_markdown_key_values(content)
    normalized = {normalize_text(key): value for key, value in values.items()}
    selected: dict[str, str] = {}
    for key, expected in specification.items():
        normalized_key = normalize_text(key)
        if normalized_key not in normalized or not structured_value_matches(
            normalized[normalized_key], expected
        ):
            return None
        selected[str(key)] = normalized[normalized_key]
    return selected


def matching_markdown_table_rows(
    content: str,
    specification: Any,
    *,
    section: Any = None,
) -> list[tuple[list[str], dict[str, Any]]]:
    if not isinstance(specification, Mapping):
        return []
    match_specification = specification.get("match")
    if not isinstance(match_specification, Mapping) or not match_specification:
        return []
    numeric = specification.get("numeric")
    if numeric is not None and not (
        isinstance(numeric, Mapping) and isinstance(numeric.get("column"), str)
    ):
        return []
    wanted_section = normalize_text(section) if section is not None else ""
    matches: list[tuple[list[str], dict[str, Any]]] = []
    for table in parse_markdown_tables(content):
        if wanted_section and wanted_section not in normalize_text(table.get("section", "")):
            continue
        for row in table.get("rows", []):
            if not all(
                key in row and structured_value_matches(row[key], expected)
                for key, expected in match_specification.items()
            ):
                continue
            if numeric is not None and not numeric_equal(
                row.get(str(numeric["column"])),
                numeric.get("value"),
                numeric.get("abs_tolerance", 0),
            ):
                continue
            matches.append((list(table.get("headers", [])), row))
    return matches


def required_text_matches_one_block(content: str, required: Any) -> bool:
    tokens = selector_strings(required)
    if not tokens:
        return False
    blocks = [block for block in re.split(r"\n\s*\n", content) if block.strip()]
    return any(all(contains_selector_token(block, token) for token in tokens) for block in blocks)


def markdown_selector_matches(content: Any, selector: Mapping[str, Any]) -> bool:
    """Match a Markdown selector inside one selected section and coherent block."""

    if not selector:
        return True
    if not isinstance(content, str) or not content:
        return False
    chunks = markdown_section_chunks(content, selector.get("section"))
    if not chunks:
        return False
    for chunk in chunks:
        if (
            "key_values" in selector
            and matching_markdown_key_values(chunk, selector.get("key_values")) is None
        ):
            continue
        if "table_row" in selector and not matching_markdown_table_rows(
            chunk, selector.get("table_row")
        ):
            continue
        if "required_text" in selector and not required_text_matches_one_block(
            chunk, selector.get("required_text")
        ):
            continue
        if any(key in selector for key in ("key_values", "table_row", "required_text")):
            return True
        # Legacy scalar selectors still have to co-occur in one coherent block.
        if required_text_matches_one_block(chunk, selector):
            return True
    return False