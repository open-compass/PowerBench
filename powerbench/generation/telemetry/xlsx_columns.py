from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# Logical xlsx column/sheet names -> candidate physical headers (English only).
XLSX_COLUMN_ALIASES: Dict[str, List[str]] = {
    "device_list_sheet": ["device_list"],
    "device_name": ["device_name"],
    "applicable_rule_ids_all": ["applicable_rule_ids_all"],
    "applicable_rules_with_objects": ["applicable_rules_with_objects"],
    "maintenance_metric_db_fields_all": ["maintenance_metric_db_fields_all"],
    "maintenance_metrics_all": ["maintenance_metrics_all"],
    "derived_metric_keys_all": ["derived_metric_keys_all"],
    "project_name": ["project_name", "project"],
    "station_name": ["station_name"],
    "insulation_type": ["insulation_type"],
    "installation_location": ["installation_location"],
    "bay_number": ["bay_number"],
    "rated_voltage": ["rated_voltage"],
    "insulation_medium": ["insulation_medium"],
    "structure_type": ["structure_type"],
}


def load_xlsx_column_aliases() -> Dict[str, List[str]]:
    return XLSX_COLUMN_ALIASES


def resolve_column_name(columns: Sequence[str], logical_name: str) -> Optional[str]:
    """Pick the first matching physical column for a logical xlsx field."""
    cols = set(columns)
    candidates = XLSX_COLUMN_ALIASES.get(logical_name, [logical_name])
    for name in candidates:
        if name in cols:
            return name
    return None


def row_get(row: Any, logical_name: str, default: str = "") -> str:
    """Read a cell by logical xlsx column name."""
    index = getattr(row, "index", None)
    columns = list(index) if index is not None else (list(row.keys()) if isinstance(row, dict) else [])
    col = resolve_column_name(columns, logical_name)
    if col is None:
        return default
    val = row.get(col) if hasattr(row, "get") else row[col]
    if val is None:
        return default
    s = str(val).strip()
    if s.lower() in {"nan", "none", "null"}:
        return default
    return s


def read_device_sheet(xlsx_path, *, sheet_logical: str = "device_list_sheet"):
    """Return (resolved_sheet_name, dataframe) or raise on failure."""
    import pandas as pd
    from pathlib import Path

    xlsx_path = Path(xlsx_path)
    aliases = XLSX_COLUMN_ALIASES.get(sheet_logical, [sheet_logical])
    last_err: Optional[Exception] = None
    for sheet in aliases:
        try:
            return sheet, pd.read_excel(xlsx_path, sheet_name=sheet)
        except Exception as exc:
            last_err = exc
    raise last_err or ValueError(f"Cannot read sheet for {sheet_logical!r} from {xlsx_path}")
