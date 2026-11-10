from __future__ import annotations

import re
from pathlib import Path

from ..documents import parse_document
from ..models import CorpusInventory, DocumentAnchor
from ..util import parse_timestamp
from .eligibility import EligibilityRejection

_CHECK_EVENT_RE = re.compile(r"_(E\d+)_check_report\.md$", re.IGNORECASE)
_FAULT_EVENT_RE = re.compile(r"_(E\d+)_\d{14}_fault_report\.md$", re.IGNORECASE)
_RULE_RE = re.compile(r"K\s*0*(\d+)", re.IGNORECASE)


def _event_id(path: str, kind: str) -> str:
    pattern = _CHECK_EVENT_RE if kind == "check_report" else _FAULT_EVENT_RE
    match = pattern.search(Path(path).name)
    return match.group(1).upper() if match else ""


def _associated_rule_id(value: str) -> str:
    rule_ids = {f"K{int(number)}" for number in _RULE_RE.findall(value)}
    if len(rule_ids) > 1:
        raise EligibilityRejection(
            "multiple_associated_rules",
            "a replay anchor must name exactly one associated rule; "
            f"found {sorted(rule_ids)} in {value!r}",
        )
    if not rule_ids:
        raise ValueError(f"A replay anchor must name an associated rule; found none in {value!r}")
    return next(iter(rule_ids))


def check_anchor(
    inventory: CorpusInventory,
    device_uid: str,
    relative_path: str,
) -> DocumentAnchor:
    parsed = parse_document(
        inventory.raw_root / relative_path,
        max_chars=80_000,
        include_tables=False,
    )
    window = parsed.get("inspection_window")
    if not window:
        raise ValueError(f"No inspection window in {relative_path}")
    return DocumentAnchor(
        path=relative_path,
        device_uid=device_uid,
        kind="check_report",
        start=parse_timestamp(window["start"]),
        end=parse_timestamp(window["end"]),
        event_id=_event_id(relative_path, "check_report"),
        metadata={str(k): str(v) for k, v in parsed.get("key_values", {}).items()},
    )


def fault_anchor(
    inventory: CorpusInventory,
    device_uid: str,
    relative_path: str,
) -> DocumentAnchor:
    parsed = parse_document(
        inventory.raw_root / relative_path,
        max_chars=80_000,
        include_tables=False,
    )
    window = parsed.get("fault_window")
    if not window:
        raise ValueError(f"No fault window in {relative_path}")
    key_values = {str(k): str(v) for k, v in parsed.get("key_values", {}).items()}
    try:
        rule_id = _associated_rule_id(key_values.get("Linked rule", ""))
    except EligibilityRejection as exc:
        raise EligibilityRejection(exc.reason, f"{relative_path}: {exc.detail}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid associated rule in {relative_path}: {exc}") from exc
    metric_field = key_values.get("Input metric field")
    return DocumentAnchor(
        path=relative_path,
        device_uid=device_uid,
        kind="fault_report",
        start=parse_timestamp(window["start"]),
        end=parse_timestamp(window["end"]),
        event_id=_event_id(relative_path, "fault_report"),
        rule_id=rule_id,
        metric_field=metric_field,
        report_number=key_values.get("Report ID"),
        metadata=key_values,
    )