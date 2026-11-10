from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..corpus import CorpusTools
from ..models import CorpusInventory
from ..selector import (
    contains_selector_token,
    markdown_section_chunks,
    markdown_selector_matches,
    matching_json_objects,
    matching_markdown_key_values,
    matching_markdown_table_rows,
    selector_strings,
)
from ..util import format_timestamp, parse_timestamp


def _find_json_object(payload: Any, selector: dict[str, Any]) -> Any:
    if not selector:
        return payload
    matches = matching_json_objects(payload, selector)
    return matches[0] if matches else None


def _read_complete_text(tools: CorpusTools, path: str) -> str:
    chunks: list[str] = []
    start_line = 1
    while True:
        result = tools.read_file(path, start_line=start_line, end_line=None)
        if not result.get("supported"):
            raise ValueError(f"Oracle source is not readable text: {path}")
        chunks.append(str(result.get("content", "")))
        if not result.get("truncated"):
            return "".join(chunks)
        end_line = int(result.get("end_line") or 0)
        if end_line < start_line:
            raise ValueError(f"Oracle reader made no progress for {path}")
        start_line = end_line + 1


def _json_excerpt(text: str, path: str, selector: dict[str, Any]) -> str:
    payload = json.loads(text)
    selected = _find_json_object(payload, selector)
    if selected is None:
        raise ValueError(f"Oracle selector did not match any JSON object in {path}: {selector}")
    return json.dumps(selected, ensure_ascii=False, sort_keys=True, indent=2)


def _markdown_excerpt(text: str, path: str, selector: dict[str, Any]) -> str:
    section = str(selector.get("section") or "Evidence excerpt")
    key_values = selector.get("key_values")
    if key_values is not None:
        if not isinstance(key_values, dict) or not key_values:
            raise ValueError(f"Oracle key-value selector is invalid for {path}")
        selections: list[dict[str, str]] = []
        for chunk in markdown_section_chunks(text, selector.get("section")):
            selected = matching_markdown_key_values(chunk, key_values)
            if selected is not None:
                selections.append(selected)
        if len(selections) != 1:
            raise ValueError(f"Oracle key-value selector did not uniquely match in {path}")
        selected = selections[0]
        lines = [f"## {section}", ""]
        lines.extend(f"- **{key}**: {value}" for key, value in selected.items())
        return "\n".join(lines)

    table_row = selector.get("table_row")
    if table_row is not None:
        if not isinstance(table_row, dict) or not isinstance(table_row.get("match"), dict):
            raise ValueError(f"Oracle table-row selector is invalid for {path}")
        matches = matching_markdown_table_rows(
            text,
            table_row,
            section=selector.get("section"),
        )
        if len(matches) != 1:
            raise ValueError(f"Oracle table-row selector did not uniquely match in {path}")
        headers, row = matches[0]
        return "\n".join(
            [
                f"## {section}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |",
            ]
        )

    required = list(dict.fromkeys(selector_strings(selector.get("required_text"))))
    if not required:
        required = [
            str(value)
            for key, value in selector.items()
            if key != "required_text" and isinstance(value, (str, int, float)) and str(value)
        ]
    missing = [token for token in required if not contains_selector_token(text, token)]
    if missing or not markdown_selector_matches(text, selector):
        raise ValueError(f"Oracle selector tokens missing from {path}: {missing}")
    lines = text.splitlines()
    occurrences = sorted(
        (index, token)
        for index, line in enumerate(lines)
        for token in required
        if contains_selector_token(line, token)
    )
    if not occurrences:
        raise ValueError(f"Oracle selector produced no Markdown excerpt for {path}")

    # Use the smallest line interval covering every selector token. This avoids
    # attaching unrelated sections when a common token appears many times.
    counts: dict[str, int] = {}
    covered = 0
    left = 0
    best: tuple[int, int] | None = None
    for right_line, token in occurrences:
        counts[token] = counts.get(token, 0) + 1
        if counts[token] == 1:
            covered += 1
        while covered == len(required):
            left_line, left_token = occurrences[left]
            candidate = (left_line, right_line)
            if best is None or candidate[1] - candidate[0] < best[1] - best[0]:
                best = candidate
            counts[left_token] -= 1
            if counts[left_token] == 0:
                covered -= 1
            left += 1
    if best is None:
        raise ValueError(f"Oracle selector could not cover every token in {path}")
    start = max(0, best[0] - 3)
    end = min(len(lines), best[1] + 4)
    return "\n".join(lines[start:end])


def build_oracle_bundle(
    inventory: CorpusInventory,
    tools: CorpusTools,
    gold: dict[str, Any],
) -> dict[str, Any]:
    """Materialize one public source for every Gold evidence atom.

    This function is evaluator-side and never includes the Gold answer or Gold
    program. It only extracts public raw evidence and gives time-series slices
    the same materialized handles used by the normal Python tool.
    """

    evidence: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for atom in gold["evidence_atoms"]:
        source = atom["acceptable_sources"][0]
        path = str(source["path"])
        selector = dict(source.get("selector") or {})
        signature = (path, json.dumps(selector, ensure_ascii=False, sort_keys=True))
        if signature in seen:
            audit.append({"atom_id": atom["atom_id"], "path": path, "selector": selector})
            continue
        seen.add(signature)
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            artifact = inventory.artifacts.get(path)
            if not artifact or not artifact.get("device_uid"):
                raise ValueError(f"Oracle timeseries source is not device-scoped: {path}")
            device_uid = str(artifact["device_uid"])
            if inventory.device(device_uid).csv_path != path:
                raise ValueError(f"Oracle timeseries source is not the device CSV: {path}")
            start = parse_timestamp(selector["start"])
            lookback = int(selector.get("lookback_hours") or 0)
            start -= timedelta(hours=lookback)
            result = tools.read_timeseries(
                device_uid,
                [str(selector["field"])],
                format_timestamp(start),
                str(selector["end_exclusive"]),
            )
            if result.get("path") != path:
                raise ValueError(f"Oracle timeseries reader returned the wrong source for {path}")
            evidence.append(
                {
                    "path": path,
                    "materialized_handle": result["handle"],
                    "rows": result["rows"],
                }
            )
        elif suffix == ".json":
            text = _read_complete_text(tools, path)
            evidence.append(
                {
                    "path": path,
                    "content": _json_excerpt(text, path, selector),
                }
            )
        elif suffix == ".md":
            text = _read_complete_text(tools, path)
            evidence.append(
                {
                    "path": path,
                    "content": _markdown_excerpt(text, path, selector),
                }
            )
        else:
            raise ValueError(f"Unsupported Oracle evidence source: {path}")
        audit.append({"atom_id": atom["atom_id"], "path": path, "selector": selector})
    return {
        "notice": "Only public raw evidence is included; no Gold answer or Gold program is present.",
        "evidence": evidence,
        "materialized_handles": sorted(tools.materialized_handles()),
        "_audit": audit,
    }