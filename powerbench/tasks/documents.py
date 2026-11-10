from __future__ import annotations

import re
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

_TIMESTAMP_PATTERN = r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
TIMESTAMP_RE = re.compile(_TIMESTAMP_PATTERN)
TIME_RANGE_RE = re.compile(
    rf"(?P<start>{_TIMESTAMP_PATTERN})\s*(?:~|to|—|–)\s*"
    rf"(?P<end>{_TIMESTAMP_PATTERN})"
)
RULE_ID_RE = re.compile(r"(?<![A-Za-z0-9])K\s*0*(\d+)(?!\d)", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")

_BOLD_KEY_VALUE_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?:\*\*|__)(?P<key>.+?)(?:\*\*|__)"
    r"\s*[:：]\s*(?P<value>.*?)\s*$"
)
_PLAIN_KEY_VALUE_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?P<key>[^|:：#]{1,80}?)\s*[:：]\s*(?P<value>.+?)\s*$"
)

_INSPECTION_WINDOW_KEYS = ("Inspection window",)
_FAULT_WINDOW_KEYS = ("Fault window",)
_KEY_HEADERS = {"item", "field", "property", "key", "Item", "Field", "Property", "Key"}
_VALUE_HEADERS = {"content", "value", "Content", "Value", "information"}


def _unique_in_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _clean_inline_markdown(value: str) -> str:
    value = value.strip()
    if len(value) >= 4 and (
        (value.startswith("**") and value.endswith("**"))
        or (value.startswith("__") and value.endswith("__"))
    ):
        value = value[2:-2].strip()
    return value.replace("\\|", "|")


def _split_table_row(line: str) -> list[str]:
    """Split a Markdown pipe row while preserving escaped pipes."""

    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in text:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append(_clean_inline_markdown("".join(current)))
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append(_clean_inline_markdown("".join(current)))
    return cells


def _is_table_separator(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(
        TABLE_SEPARATOR_CELL_RE.fullmatch(cell.replace(" ", "")) for cell in cells
    )


def _unique_headers(headers: list[str]) -> list[str]:
    unique: list[str] = []
    counts: dict[str, int] = {}
    for index, raw_header in enumerate(headers, start=1):
        header = raw_header or f"column_{index}"
        counts[header] = counts.get(header, 0) + 1
        suffix = counts[header]
        unique.append(header if suffix == 1 else f"{header}#{suffix}")
    return unique


def parse_markdown_tables(text: str) -> list[dict[str, Any]]:
    """Parse GitHub-style pipe tables without assuming a document template.

    Each result contains the closest preceding heading, the normalized headers,
    row dictionaries, and one-based source line numbers. Malformed short rows are
    padded; long rows are retained under generated column names.
    """

    lines = text.splitlines()
    headings: list[str] = []
    tables: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        heading_match = HEADING_RE.match(lines[index])
        if heading_match:
            stripped = lines[index].lstrip()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = _clean_inline_markdown(heading_match.group("title"))
            headings = headings[: max(0, level - 1)]
            headings.append(title)
            index += 1
            continue

        if (
            "|" not in lines[index]
            or index + 1 >= len(lines)
            or not _is_table_separator(lines[index + 1])
        ):
            index += 1
            continue

        raw_headers = _split_table_row(lines[index])
        headers = _unique_headers(raw_headers)
        start_line = index + 1
        index += 2
        rows: list[dict[str, str]] = []
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            if HEADING_RE.match(lines[index]):
                break
            values = _split_table_row(lines[index])
            if len(values) > len(headers):
                extra_headers = [
                    f"column_{column}" for column in range(len(headers) + 1, len(values) + 1)
                ]
                headers.extend(extra_headers)
                for existing in rows:
                    existing.update({header: "" for header in extra_headers})
            values.extend([""] * (len(headers) - len(values)))
            rows.append(dict(zip(headers, values, strict=True)))
            index += 1

        tables.append(
            {
                "section": " / ".join(headings),
                "headers": headers,
                "rows": rows,
                "start_line": start_line,
                "end_line": index,
                "format": "markdown",
            }
        )
    return tables


class _SimpleHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def parse_html_tables(text: str) -> list[dict[str, Any]]:
    parser = _SimpleHTMLTableParser()
    parser.feed(text)
    results: list[dict[str, Any]] = []
    for table in parser.tables:
        headers = _unique_headers(table[0])
        rows: list[dict[str, str]] = []
        for raw_values in table[1:]:
            values = list(raw_values[: len(headers)])
            values.extend([""] * (len(headers) - len(values)))
            rows.append(dict(zip(headers, values, strict=True)))
        results.append(
            {
                "section": "",
                "headers": headers,
                "rows": rows,
                "start_line": None,
                "end_line": None,
                "format": "html",
            }
        )
    return results


def parse_markdown_key_values(
    text: str, tables: list[dict[str, Any]] | None = None
) -> dict[str, str]:
    """Extract bullet/plain key-value pairs and conventional two-column tables."""

    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.lstrip().startswith(("#", "|")):
            continue
        match = _BOLD_KEY_VALUE_RE.match(line) or _PLAIN_KEY_VALUE_RE.match(line)
        if not match:
            continue
        key = _clean_inline_markdown(match.group("key"))
        value = _clean_inline_markdown(match.group("value"))
        if key:
            values.setdefault(key, value)

    for table in tables if tables is not None else parse_markdown_tables(text):
        headers = table.get("headers", [])
        if len(headers) != 2:
            continue
        left, right = (str(header).strip() for header in headers)
        if left.lower() not in _KEY_HEADERS or right.lower() not in _VALUE_HEADERS:
            continue
        for row in table.get("rows", []):
            key = _clean_inline_markdown(str(row.get(left, "")))
            value = _clean_inline_markdown(str(row.get(right, "")))
            if key:
                values.setdefault(key, value)
    return values


def extract_rule_ids(text: str) -> list[str]:
    return _unique_in_order(f"K{int(match.group(1))}" for match in RULE_ID_RE.finditer(text))


def extract_time_ranges(text: str) -> list[dict[str, str]]:
    pairs = ((match.group("start"), match.group("end")) for match in TIME_RANGE_RE.finditer(text))
    unique = _unique_in_order(f"{start}\x1f{end}" for start, end in pairs)
    return [
        {"start": value.split("\x1f", 1)[0], "end": value.split("\x1f", 1)[1]} for value in unique
    ]


def _ranges_for_keys(key_values: dict[str, str], keywords: tuple[str, ...]) -> list[dict[str, str]]:
    ranges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, value in key_values.items():
        if not any(keyword in key for keyword in keywords):
            continue
        for item in extract_time_ranges(value):
            pair = (item["start"], item["end"])
            if pair not in seen:
                seen.add(pair)
                ranges.append(item)
    return ranges


def _document_type(path: Path, title: str) -> str:
    parent = path.parent.name.lower()
    filename = path.name.lower()
    if parent == "device_archive" or "device_archive" in filename or "Device Archive" in title:
        return "device_archive"
    if parent == "check_report" or "check_report" in filename or "Inspection Report" in title:
        return "check_report"
    if parent == "fault_report" or "fault_report" in filename or "Fault Report" in title:
        return "fault_report"
    return "document"


def parse_document(
    path: str | Path,
    *,
    max_chars: int | None = None,
    include_tables: bool = True,
) -> dict[str, Any]:
    """Parse a PowerBench Markdown document into JSON-serializable structure.

    ``max_chars`` is intended for inventory summaries: all report linkage fields
    occur in the leading basic-information section, so callers need not retain
    large embedded time-series tables. The default parses the complete document.
    """

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
        if max_chars is None:
            text = handle.read()
            truncated = False
        else:
            prefix = handle.read(max_chars + 1)
            truncated = len(prefix) > max_chars
            text = prefix[:max_chars]

    headings = [
        _clean_inline_markdown(match.group("title"))
        for line in text.splitlines()
        if (match := HEADING_RE.match(line))
    ]
    title = headings[0] if headings else ""
    tables = parse_markdown_tables(text) if include_tables else []
    if include_tables and "<table" in text.lower():
        tables.extend(parse_html_tables(text))
    key_values = parse_markdown_key_values(text, tables)
    timestamps = _unique_in_order(match.group(0) for match in TIMESTAMP_RE.finditer(text))
    time_ranges = extract_time_ranges(text)
    inspection_windows = _ranges_for_keys(key_values, _INSPECTION_WINDOW_KEYS)
    fault_windows = _ranges_for_keys(key_values, _FAULT_WINDOW_KEYS)

    return {
        "path": str(source),
        "document_type": _document_type(source, title),
        "title": title,
        "headings": headings,
        "key_values": key_values,
        "metadata": dict(key_values),
        "tables": tables,
        "timestamps": timestamps,
        "time_ranges": time_ranges,
        "inspection_windows": inspection_windows,
        "inspection_window": inspection_windows[0] if inspection_windows else None,
        "fault_windows": fault_windows,
        "fault_window": fault_windows[0] if fault_windows else None,
        "rule_ids": extract_rule_ids(text),
        "truncated": truncated,
    }