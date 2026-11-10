from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from powerbench.generation.telemetry.xlsx_columns import read_device_sheet, resolve_column_name, row_get
import requests
from requests import HTTPError


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _md_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [len(c) for c in rows[0]]
    for r in rows[1:]:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    out: List[str] = []

    def esc(x: str) -> str:
        return str(x).replace("\r", "").replace("|", "\\|")

    out.append("| " + " | ".join(esc(c).ljust(widths[i]) for i, c in enumerate(rows[0])) + " |")
    out.append("| " + " | ".join("---".ljust(widths[i]) for i in range(len(widths))) + " |")
    for r in rows[1:]:
        out.append("| " + " | ".join(esc(r[i]).ljust(widths[i]) for i in range(len(widths))) + " |")
    return "\n".join(out) + "\n"


def _extract_first_json_object(text: str) -> Any:
    t = text.strip()
    if not t:
        raise ValueError("empty LLM output")

    candidates: List[str] = [t]
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.I):
        block = _compact_str(m.group(1))
        if block:
            candidates.append(block)

    def iter_json_substrings(src: str) -> List[str]:
        subs: List[str] = []
        starts = [i for i, ch in enumerate(src) if ch in "{["]
        for start in starts:
            open_ch = src[start]
            close_ch = "}" if open_ch == "{" else "]"
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(src)):
                ch = src[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        subs.append(src[start : i + 1].strip())
                        break
        return subs

    parse_errs: List[str] = []
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception as e:
            parse_errs.append(str(e))
        for sub in iter_json_substrings(cand):
            try:
                return json.loads(sub)
            except Exception as e:
                parse_errs.append(str(e))

    preview = t[:300].replace("\n", "\\n")
    raise ValueError(f"cannot parse JSON object from LLM output; preview={preview}; errs={parse_errs[:3]}")


@dataclass
class OpenAICompatClient:
    api_key: str
    base_url: str
    model: str
    timeout_s: int = 1800

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_new_tokens: Optional[int] = None) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": float(temperature)}
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            payload["max_tokens"] = int(max_new_tokens)
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except HTTPError as e:
            body = ""
            try:
                body = r.text  # type: ignore[name-defined]
            except Exception:
                body = "<no response body>"
            raise RuntimeError(f"HTTP error calling LLM: {e}; response_body={body[:2000]}") from e
        data = r.json()
        return data["choices"][0]["message"]["content"]


def _load_metric_catalog(model_root: Path) -> List[Dict[str, Any]]:
    items = _read_json(model_root / "metric_catalog.json")
    return [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []


def _catalog_maps(catalog: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    db2cn: Dict[str, str] = {}
    db2unit: Dict[str, str] = {}
    by_cn: Dict[str, Dict[str, Any]] = {}
    for it in catalog:
        db = _compact_str(it.get("database_field"))
        cn = _compact_str(it.get("metric_name"))
        unit = _compact_str(it.get("unit"))
        if db:
            db2cn[db] = cn or db
            db2unit[db] = unit
        if cn:
            by_cn[cn] = it
    return db2cn, db2unit, by_cn


def _load_rules_list(model_root: Path) -> List[Dict[str, Any]]:
    path = model_root / "rule_metric_stage2.json"
    if not path.is_file():
        return []
    raw = _read_json(path)
    out: List[Dict[str, Any]] = []
    for it in raw.get("anomaly_judgment_rules", []) if isinstance(raw, dict) else []:
        if isinstance(it, dict):
            out.append(it)
    return out


def _rule_id_to_text(rules: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in rules:
        rid = _compact_str(it.get("rule_id")).upper()
        text = _compact_str(it.get("text_description")) or _compact_str(it.get("judgment_description"))
        if rid and text:
            out[rid] = text
    return out


def _find_xlsx(model_root: Path) -> Optional[Path]:
    cands = sorted(model_root.glob("*_with_metrics.xlsx"))
    return cands[0] if cands else None


def _load_xlsx_device_row(model_root: Path, device_name: str) -> Dict[str, Any]:
    xlsx = _find_xlsx(model_root)
    if xlsx is None:
        return {}
    try:
        df = pd.read_excel(xlsx, sheet_name="device_list")
    except Exception:
        return {}
    name_col = resolve_column_name(df.columns, "device_name")
    if not name_col:
        return {}
    hit = df[df[name_col].astype(str).str.strip() == device_name]
    if hit.empty:
        # fuzzy: stem containment
        hit = df[df[name_col].astype(str).str.contains(re.escape(device_name[:20]), na=False)]
    if hit.empty:
        return {}
    row = hit.iloc[0].to_dict()
    return {str(k): v for k, v in row.items() if _compact_str(v)}

def _row_field(row: Dict[str, Any], logical: str) -> str:
    if not isinstance(row, dict):
        return ""
    val = row_get(row, logical) if logical else ""
    return val or _compact_str(row.get(logical))


def _extract_unit_from_meaning(meaning: str) -> str:
    ms = re.findall(r"\(([^)]+)\)", meaning or "")
    return ms[-1].strip() if ms else ""


def _param_rows(rule: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for p in rule.get("threshold_params", []) or []:
        if not isinstance(p, dict):
            continue
        pn = _compact_str(p.get("param_name"))
        m = _compact_str(p.get("meaning"))
        v = _compact_str(p.get("param_value"))
        if pn or m:
            rows.append((pn, m, v))
    return rows


def _param_dict(rule: Dict[str, Any]) -> Dict[str, Tuple[str, str, str]]:
    d: Dict[str, Tuple[str, str, str]] = {}
    for pn, m, v in _param_rows(rule):
        if pn:
            d[pn] = (pn, m, v)
    return d


EXPR_LEFT_ALIASES: Dict[str, str] = {
    "oil_dissolved_gas": "oil_dissolved_gas",
    "hotspot_temperature_rise": "hotspot_temperature_rise",
    "winding_to_oil_temp_differential": "winding_to_oil_temp_differential",
    "apparent_power": "apparent_power",
    "I_rated": "i_rated",
    "U_rated": "u_rated",
    "T_in": "cooling_water_temp",
    "P_water": "cooling_water_pressure",
    "flow": "cooling_water_flow",
    "P_SF6": "sf6_gas_pressure_density",
    "cooling_status": "cooling_system_status",
    "oltc_status": "oltc_overall_status",
}

# CSV rule alias column -> canonical metric_name for stage2 matching
RULE_ALIAS_CANON_CN: Dict[str, str] = {
    "P_SF6": "sf6_gas_pressure_density",
    "cooling_status": "cooling_system_status",
    "oltc_status": "oltc_overall_status",
    "oltc_tap_position": "oltc_overall_status",
    "oltc_motor_status": "oltc_overall_status",
    "oltc_operation_count": "oltc_overall_status",
}

_EXPR_SKIP_HINTS = ("ABS(", "abs(", " in {", " in{", "='")


def _unwrap_consec_duration(expr: str) -> str:
    s = expr
    funcs = ("consec_3h", "duration_1h", "consec_1h")
    changed = True
    while changed:
        changed = False
        for fn in funcs:
            key = fn + "("
            i = 0
            while True:
                i = s.find(key, i)
                if i == -1:
                    break
                if i > 0 and (s[i - 1].isalnum() or s[i - 1] == "_"):
                    i += 1
                    continue
                j = i + len(key)
                depth = 1
                k = j
                while k < len(s) and depth > 0:
                    if s[k] == "(":
                        depth += 1
                    elif s[k] == ")":
                        depth -= 1
                    k += 1
                inner = s[j : k - 1]
                s = s[:i] + inner + s[k:]
                changed = True
                break
    return s


def _split_top_level(expr: str, sep: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    i = 0
    n = len(expr)
    m = len(sep)
    while i < n:
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and i + m <= n and expr[i : i + m].upper() == sep.upper():
            parts.append(expr[start:i].strip())
            start = i + m
            i += m
            continue
        i += 1
    parts.append(expr[start:].strip())
    return [p for p in parts if p]


def _flatten_atomic_clauses(expr: str) -> List[str]:
    s = _unwrap_consec_duration(expr)
    s = re.sub(r"\s*=\s*true\s*", " ", s, flags=re.I)
    out: List[str] = []

    def push_leaf(t: str) -> None:
        tt = t.strip().strip("()").strip()
        if not tt:
            return
        sub_ors = _split_top_level(tt, " OR ")
        if len(sub_ors) > 1:
            for o in sub_ors:
                walk(o)
            return
        sub_ands = _split_top_level(tt, " AND ")
        if len(sub_ands) > 1:
            for a in sub_ands:
                walk(a)
            return
        if tt not in out:
            out.append(tt)

    def walk(x: str) -> None:
        x = x.strip()
        if not x:
            return
        x = _unwrap_consec_duration(x)
        ors = _split_top_level(x, " OR ")
        if len(ors) > 1:
            for o in ors:
                walk(o)
            return
        ands = _split_top_level(x, " AND ")
        if len(ands) > 1:
            for a in ands:
                walk(a)
            return
        push_leaf(x)

    walk(s)
    return out


_COMP_RE = re.compile(r"^\s*(.+?)\s*(>=|<=|>|<)\s*([A-Za-z_][\w]*)\s*$", re.DOTALL)


def _parse_plain_comparison(leaf: str) -> Optional[Tuple[str, str, str]]:
    t = leaf.strip()
    while t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    m = _COMP_RE.match(t)
    if not m:
        return None
    left, op, pname = m.group(1).strip(), m.group(2), m.group(3)
    if "(" in left or ")" in left:
        return None
    return left, op, pname


def _left_refs_column(left: str, db_col: str, cn: str) -> bool:
    L = left.strip().replace(" ", "")
    if EXPR_LEFT_ALIASES.get(left.strip(), "") == db_col:
        return True
    if db_col and (db_col == left.strip() or db_col in left):
        return True
    if cn:
        cn0 = cn.replace(" ", "")
        if cn in left or cn0 in L:
            return True
    return False


def _try_expr_driven_design(rule: Dict[str, Any], db_col: str, cn: str) -> Optional[str]:
    expr = _compact_str(rule.get("rule_expression"))
    if not expr or any(h in expr for h in _EXPR_SKIP_HINTS):
        return None
    pdict = _param_dict(rule)
    atoms: List[Tuple[str, str, str]] = []
    for leaf in _flatten_atomic_clauses(expr):
        pc = _parse_plain_comparison(leaf)
        if pc:
            atoms.append(pc)
    matched: List[Tuple[str, str, str]] = []
    for left, op, pname in atoms:
        if not _left_refs_column(left, db_col, cn):
            continue
        if pname not in pdict:
            continue
        matched.append((left, op, pname))
    if not matched:
        return None

    ups: List[Tuple[float, str]] = []
    los: List[Tuple[float, str]] = []
    for _, op, pname in matched:
        _, mean, vs = pdict[pname]
        fv = _safe_float(vs)
        if fv is None:
            continue
        if op in (">", ">="):
            ups.append((fv, mean))
        elif op in ("<", "<="):
            los.append((fv, mean))

    def unit_from(ms: List[str]) -> str:
        for mm in ms:
            u = _extract_unit_from_meaning(mm)
            if u:
                return u
        return ""

    def fmt_v(v: float) -> str:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return str(v)

    if ups and los:
        lo_v = max(x[0] for x in los)
        hi_v = min(x[0] for x in ups)
        if lo_v > hi_v:
            lo_v, hi_v = hi_v, lo_v
        u = unit_from([pdict[p][1] for _, _, p in matched if p in pdict])
        return f"between{fmt_v(lo_v)}~{fmt_v(hi_v)}{u}"
    if ups and not los:
        v = min(x[0] for x in ups)
        u = unit_from([x[1] for x in ups])
        return f"less than{fmt_v(v)}{u}"
    if los and not ups:
        v = max(x[0] for x in los)
        u = unit_from([x[1] for x in los])
        return f"not less than{fmt_v(v)}{u}"
    return None


def _rule_match_score(rule: Dict[str, Any], cn: str) -> int:
    obj = _compact_str(rule.get("object_name"))
    inputs = [_compact_str(x) for x in (rule.get("input_metrics") or []) if _compact_str(x)]
    if cn == obj:
        return 100
    if cn in inputs:
        return 95
    if obj and cn in obj:
        return 80
    for inp in inputs:
        if inp and (cn in inp or inp in cn):
            return 55
    return 0


def _rule_strength(rule: Dict[str, Any]) -> int:
    s = 0
    blob = " ".join(
        f"{pn} {m}".lower() for pn, m, _ in _param_rows(rule)
    )
    if "alarm" in blob or "alarm" in blob or "alarm" in blob:
        s += 5
    if "trip" in blob or "trip" in blob or "lockout" in blob:
        s += 5
    return s


def _metric_design_value(
    cn: str,
    rules: List[Dict[str, Any]],
    db_col: str,
    preferred_rule_ids: Optional[List[str]] = None,
) -> str:
    """Threshold / design hint from this model's stage2 only."""
    pref = {x.upper() for x in (preferred_rule_ids or [])}
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for r in rules:
        ms = _rule_match_score(r, cn)
        expr = _compact_str(r.get("rule_expression"))
        if db_col and db_col in expr:
            ms += 6
        inputs = r.get("input_metrics") or []
        if isinstance(inputs, list) and cn in [str(x) for x in inputs]:
            ms += 4
        obj = _compact_str(r.get("object_name"))
        if obj == cn:
            ms += 20  # exact object wins over loosely related multi-metric rules
        rid = _compact_str(r.get("rule_id")).upper()
        if pref and rid in pref:
            ms += 3
        if ms > 0:
            scored.append((ms + _rule_strength(r), r))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return "TBD (no stage2 threshold match for this model)"

    # Prefer first rule that yields a clean expr-driven bound.
    for _, r in scored:
        driven = _try_expr_driven_design(r, db_col, cn)
        if driven:
            return driven

    best = scored[0][1]
    parts: List[str] = []
    rid = _compact_str(best.get("rule_id"))
    for name, meaning, val in _param_rows(best):
        if not val:
            continue
        # Drop params that clearly belong to another metric when object ≠ cn
        blob = f"{name} {meaning}"
        if _compact_str(best.get("object_name")) != cn and cn and cn not in blob and db_col and db_col not in blob:
            # keep only if meaning/name vaguely related via shared tokens
            if not any(tok and tok in blob for tok in re.findall(r"[\u4e00-\u9fff]{2,}", cn)):
                continue
        label = meaning or name
        parts.append(f"{label}={val}")
    if parts:
        prefix = f"{rid}：" if rid else ""
        return prefix + "；".join(parts[:4])
    desc = _compact_str(best.get("judgment_description"))
    return desc or "TBD"


def _pick_station(device_name: str, xlsx_row: Dict[str, Any], override: Dict[str, Any]) -> str:
    for src in (xlsx_row, override):
        for key in ("project_name", "station_name"):
            v = _compact_str(src.get(key))
            if v:
                return v
    m = re.search(r"^(.+? station)", device_name)
    if m:
        return m.group(1)
    m2 = re.search(r"^(.+? project)", device_name)
    return m2.group(1) if m2 else "TBD"


def _pick_device_pairs(model_root: Path) -> List[Tuple[int, str, Path, Path, Optional[Path]]]:
    per = model_root / "per_device"
    json_files = sorted([p for p in per.glob("*.json") if not p.name.endswith("_report.json")], key=lambda p: p.name)
    out: List[Tuple[int, str, Path, Path, Optional[Path]]] = []
    for j in json_files:
        idx_text = j.name.split("_", 1)[0]
        if not idx_text.isdigit():
            continue
        idx = int(idx_text)
        base = j.stem
        csv_path = per / f"{base}.csv"
        rep_path = per / f"{base}_report.json"
        if not csv_path.is_file():
            continue
        name = _compact_str(_read_json(j).get("device_name")) or base
        out.append((idx, name, j, csv_path, rep_path if rep_path.is_file() else None))
    return out


def _pick_commission_date(df: pd.DataFrame, cfg: Dict[str, Any]) -> str:
    tr = cfg.get("time_range") if isinstance(cfg, dict) else None
    if isinstance(tr, dict):
        start = _compact_str(tr.get("start"))
        if start:
            try:
                d = (pd.to_datetime(start) - pd.Timedelta(days=1)).date()
                return str(d)
            except Exception:
                pass
    if "timestamp" not in df.columns:
        return "TBD"
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if len(ts) == 0:
        return "TBD"
    return str((ts.min() - pd.Timedelta(days=1)).date())


def _parse_rule_ids(blob: Any) -> List[str]:
    s = _compact_str(blob)
    if not s:
        return []
    parts = re.split(r"[;；,\s]+", s)
    out: List[str] = []
    for p in parts:
        p = p.strip().upper()
        if not p:
            continue
        # "K1:xxx" → K1
        if ":" in p:
            p = p.split(":", 1)[0].strip().upper()
        if re.fullmatch(r"K\d+", p):
            out.append(p)
    return out


def _applicable_rule_ids(cfg: Dict[str, Any], xlsx_row: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ("eligible_rules_with_injector", "chosen_rules_for_injection"):
        raw = cfg.get(key) if isinstance(cfg, dict) else None
        if isinstance(raw, list):
            for x in raw:
                rid = _compact_str(x).upper()
                if rid and rid not in ids:
                    ids.append(rid)
    for key in ("applicable_rule_ids_all", "applicable_rules_with_objects"):
        for rid in _parse_rule_ids(xlsx_row.get(key)):
            if rid not in ids:
                ids.append(rid)
    return ids

def _build_basic_info(
    model_root: Path,
    device_idx: int,
    device_name: str,
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    xlsx_row: Dict[str, Any],
) -> List[List[str]]:
    model_short = model_root.name.split("_", 1)[1] if "_" in model_root.name else model_root.name
    override = cfg.get("device_fields_override", {}) if isinstance(cfg, dict) else {}
    if not isinstance(override, dict):
        override = {}
    merged = {**xlsx_row, **{k: v for k, v in override.items() if _compact_str(v)}}

    station = _pick_station(device_name, xlsx_row, override)
    run_status = _compact_str(merged.get("primary_standby_role")) or "running"
    for col in ("energy_storage_status", "density_alarm", "heater_dehumidifier_status"):
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                run_status = str(s.iloc[-1])
                break

    device_type = "+/-800kV high-end converter transformer"
    insulation = _row_field(merged, "insulation_type")
    install = _row_field(merged, "installation_location") or device_name
    bay = _row_field(merged, "bay_number")
    if bay and bay not in install:
        install = f"{install} {bay}".strip()

    return [
        ["item", "content"],
        ["archive_id", f"SB-DA-{model_root.name}-{device_idx:04d}"],
        ["device_id", f"{model_root.name}-{device_idx:04d}"],
        ["device_name", device_name],
        ["model_spec", model_short],
        ["equipment_level", "primary equipment (transformer class)"],
        ["device_type", device_type],
        ["insulation_type", insulation or "TBD"],
        ["station_or_line", station],
        ["installation_location", install],
        ["commission_date", _pick_commission_date(df, cfg)],
        ["status", run_status],
        ["rated_voltage", _row_field(merged, "rated_voltage") or _compact_str(merged.get("grid_side_voltage")) or "±800kV"],
        ["insulation_medium", _row_field(merged, "insulation_medium") or insulation or "transformer oil"],
        ["structure_type", _row_field(merged, "structure_type") or (
            f"single-phase converter transformer（{_compact_str(merged.get('connection_type'))}）"
            if _compact_str(merged.get("connection_type"))
            else "single-phase converter transformer"
        )],
        ["primary_standby_role", _compact_str(merged.get("primary_standby_role")) or "TBD"],
        ["grid_side_voltage", _compact_str(merged.get("grid_side_voltage")) or "TBD"],
        ["tap_changing_method", _compact_str(merged.get("tap_changing_method")) or "TBD"],
    ]


def _build_metric_dictionary(
    catalog: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
    db2cn: Dict[str, str],
    csv_cols: List[str],
    preferred_rule_ids: Optional[List[str]] = None,
) -> List[List[str]]:
    """Monitoring metric dictionary: CSV persisted fields for this device; catalog fills metadata."""
    rows: List[List[str]] = [["metric_name", "database_field", "unit", "metric_type", "normal_range_or_discrete", "threshold_or_design_ref"]]
    skip = {"device_name", "timestamp"}
    csv_set = {c for c in csv_cols if c not in skip}
    by_db = {
        _compact_str(it.get("database_field")): it
        for it in catalog
        if isinstance(it, dict) and _compact_str(it.get("database_field"))
    }
    # Prefer CSV column order for consistency with per_device timeseries.
    for db in [c for c in csv_cols if c not in skip]:
        it = by_db.get(db) or {}
        cn = _compact_str(it.get("metric_name")) or db2cn.get(db, db) or db
        unit = _compact_str(it.get("unit")) or "-"
        mtype = _compact_str(it.get("metric_type")) or ("derived" if db not in by_db else "-")
        rule_cn = RULE_ALIAS_CANON_CN.get(db, cn)
        discrete = it.get("discrete_values") or []
        nr = it.get("normal_range")
        if isinstance(discrete, list) and discrete:
            range_txt = " / ".join(str(x) for x in discrete)
        elif isinstance(nr, dict) and nr:
            parts = []
            for k in ("min", "mean", "max"):
                if k in nr and nr[k] is not None:
                    parts.append(f"{k}={nr[k]}")
            u = _compact_str(nr.get("unit"))
            range_txt = ("；".join(parts) + (f"（{u}）" if u else "")) if parts else (_compact_str(nr.get("note")) or "-")
        else:
            range_txt = "-"
        design = _metric_design_value(rule_cn, rules, db, preferred_rule_ids)
        rows.append([cn, db, unit, mtype, range_txt, design])
    # Note catalog-only metrics that did not land on this device (do not present as active).
    for it in catalog:
        if not isinstance(it, dict):
            continue
        db = _compact_str(it.get("database_field"))
        if not db or db in csv_set:
            continue
        cn = _compact_str(it.get("metric_name")) or db
        appl = _compact_str(it.get("applicability_conditions")) or "not generated for this device"
        rows.append(
            [
                cn,
                db,
                _compact_str(it.get("unit")) or "-",
                "not_persisted_conditional",
                appl,
                "-",
            ]
        )
    return rows


def _build_rules_overview(
    rules: List[Dict[str, Any]],
    applicable_ids: List[str],
) -> List[List[str]]:
    by_id = {_compact_str(r.get("rule_id")).upper(): r for r in rules}
    rows: List[List[str]] = [["rule_id", "object_name", "applies_to_device", "judgment_summary", "primary_thresholds"]]
    order = applicable_ids + [rid for rid in by_id if rid not in applicable_ids]
    for rid in order:
        r = by_id.get(rid)
        if not r:
            continue
        obj = _compact_str(r.get("object_name")) or "-"
        desc = _compact_str(r.get("judgment_description")) or "-"
        if len(desc) > 80:
            desc = desc[:77] + "…"
        th_parts = []
        for pn, m, v in _param_rows(r)[:3]:
            label = (m.split("(")[0].strip() if m else pn)[:20]
            if v:
                th_parts.append(f"{label}={v}")
        th = "；".join(th_parts) if th_parts else "-"
        appl = "yes" if rid in applicable_ids else "model rule (not in this device applicable set)"
        rows.append([rid, obj, appl, desc, th])
    if len(rows) == 1:
        rows.append(["-", "-", "-", "no stage2 rules for this model", "-"])
    return rows


def _build_runtime_stats(df: pd.DataFrame, db2cn: Dict[str, str], db2unit: Dict[str, str]) -> List[List[str]]:
    rows: List[List[str]] = [["metric", "unit", "sample_count", "mean_or_mode", "min", "max", "last_value"]]
    skip = {"device_name", "timestamp"}
    n_total = len(df)
    for col in [c for c in df.columns if c not in skip]:
        cn = db2cn.get(col, col)
        unit = db2unit.get(col, "") or "-"
        s = df[col]
        valid = s.dropna()
        n = len(valid)
        if n == 0:
            rows.append([cn, unit, "0", "-", "-", "-", "-"])
            continue
        last = str(valid.iloc[-1])
        if pd.api.types.is_numeric_dtype(s):
            mean_v = float(valid.astype(float).mean())
            min_v = float(valid.astype(float).min())
            max_v = float(valid.astype(float).max())
            rows.append(
                [
                    cn,
                    unit,
                    str(n),
                    f"{mean_v:.4g}",
                    f"{min_v:.4g}",
                    f"{max_v:.4g}",
                    f"{float(valid.iloc[-1]):.4g}" if _safe_float(valid.iloc[-1]) is not None else last,
                ]
            )
        else:
            mode = valid.astype(str).value_counts()
            mode_txt = f"{mode.index[0]}（{int(mode.iloc[0])}/{n_total}）" if len(mode) else "-"
            rows.append([cn, unit, str(n), mode_txt, "-", "-", last])
    return rows


def _pick_fault_source(cfg_json: Dict[str, Any], report_json: Dict[str, Any]) -> Dict[str, Any]:
    for src in (cfg_json, report_json):
        sc = src.get("scenarios") if isinstance(src, dict) else None
        if isinstance(sc, list) and len(sc) > 0:
            return src
    return report_json if isinstance(report_json, dict) else {}


def _build_fault_summary(
    report_json: Dict[str, Any],
    rule_text: Dict[str, str],
) -> List[List[str]]:
    """Only non-skipped fault scenarios from per_device; do not invent events."""
    rows: List[List[str]] = [["time", "rule_id", "description", "related_fields", "action_result", "closure_status"]]
    scenarios = report_json.get("scenarios", []) if isinstance(report_json, dict) else []
    items: List[Dict[str, Any]] = []
    for sc in scenarios:
        if not isinstance(sc, dict):
            continue
        if _compact_str(sc.get("type")) != "fault":
            continue
        if bool(sc.get("skipped")):
            continue
        items.append(sc)
    items.sort(key=lambda x: _compact_str(x.get("window_end")))
    for sc in items[-20:]:
        rid = _compact_str(sc.get("rule_id")).upper() or "unknown"
        text_desc = rule_text.get(rid, "") or _compact_str(sc.get("note_cn")) or "rule text description missing"
        ts = _compact_str(sc.get("window_end")) or _compact_str(sc.get("window_start")) or "TBD"
        fields = sc.get("input_database_fields") or sc.get("changed_cols") or []
        if isinstance(fields, list):
            field_txt = ",".join(str(x) for x in fields) if fields else "-"
        else:
            field_txt = _compact_str(fields) or "-"
        rows.append([ts, rid, text_desc, field_txt, "TBD", "annotated"])
    if len(rows) == 1:
        rows.append(["—", "—", "no fault scenarios annotated in per_device for this device", "—", "—", "—"])
    return rows


def _llm_fill_fault_actions(
    client: OpenAICompatClient,
    device_name: str,
    history_rows: List[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
) -> List[str]:
    payload = {"device_name": device_name, "rows": history_rows}
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = (
        "Fill action_result for each row in the historical anomaly summary table of a +/-800kV converter transformer archive.\n"
        "Each action_result should summarize remediation/maintenance points plus closure wording.\n"
        "Output exactly one JSON object (no markdown fences, no extra prose).\n"
        'JSON shape: {"actions": ["...", ...]} with array length matching input rows in order.\n'
        "\n"
        "Hard constraints:\n"
        "1) Do not invent fault times, rules, or metrics not in the input.\n"
        "2) Use converter-transformer O&M knowledge consistent with the row description.\n"
        "3) Do not use wording from unrelated equipment types.\n"
        "4) Each entry 20-90 words; if sparse, use generic monitor-and-retest closure.\n"
        "\n"
        f"[Input JSON]\n{payload_json}\n"
    )
    raw = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    obj = _extract_first_json_object(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("actions"), list):
        raise ValueError("invalid llm actions payload")
    acts = [_compact_str(x) for x in obj["actions"]]
    if len(acts) != len(history_rows):
        raise ValueError("llm actions size mismatch")
    return [x if x else "TBD" for x in acts]


def _maybe_llm_fill_fault_actions(
    history_rows: List[List[str]],
    llm_client: Optional[OpenAICompatClient],
    device_name: str,
    temperature: float,
    max_new_tokens: int,
) -> List[List[str]]:
    if llm_client is None or len(history_rows) <= 1:
        return history_rows
    if history_rows[1][0] == "—":
        return history_rows
    body = history_rows[1:]
    llm_input = [{"time": r[0], "rule_id": r[1], "description": r[2], "related_fields": r[3]} for r in body]
    try:
        actions = _llm_fill_fault_actions(
            llm_client,
            device_name=device_name,
            history_rows=llm_input,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        for i, act in enumerate(actions, start=1):
            history_rows[i][4] = act
    except Exception:
        pass
    return history_rows


def _llm_fill_sections(
    client: OpenAICompatClient,
    facts: Dict[str, Any],
    temperature: float,
    max_new_tokens: int,
) -> Dict[str, str]:
    fact_json = json.dumps(facts, ensure_ascii=False, indent=2)
    sec3_spec = (
        "sec3 is the Manufacturing and procurement table Markdown (with header). "
        "Illustrative manufacturer/serial/date fields must align with model, project, and commission_date; "
        "do not write concrete fault events.\n"
        "| item | content |\n| :--- | :--- |\n"
        "| manufacturer | … |\n| factory_serial | … |\n| factory_date | … |\n"
        "| purchase_contract | … |\n| receiving_inspection_date | … |\n| receiving_inspection_conclusion | … |\n| warranty_period | … |\n"
    )
    sec4_spec = (
        "sec4 is Installation/commissioning/acceptance for converter transformer (with header). "
        "Phases such as body placement, cooler installation, insulation/temperature-rise tests, tap-changer commissioning; "
        "do not invent test values or fault events.\n"
        "| phase | date | main_work | conclusion | participants |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
    )
    sec6_spec = (
        "sec6 is Design notes and O&M guidance body (no '## 6' heading). Must include:\n"
        "### Design highlights\n"
        "Summarize design/setpoint notes from metric dictionary and applicable rule thresholds (from fact JSON).\n"
        "### O&M notes\n"
        "Executable O&M advice; may summarize annotated rule types from historical anomaly summary only.\n"
        "### Current status\n"
        "Line 1: status is one of running/abnormal/retired.\n"
        "Line 2: Assessment:\n"
        "Line 3+: assessment body.\n"
        "### Next focused inspection recommendations\n"
        "Executable inspection recommendations (optionally with cadence).\n"
    )
    prompt = (
        "Complete sections 3, 4, and 6 of the +/-800kV converter transformer archive (section titles already rendered).\n"
        "Output exactly one JSON object with keys sec3, sec4, sec6.\n"
        "\n"
        f"[Section 3]\n{sec3_spec}\n"
        f"[Section 4]\n{sec4_spec}\n"
        f"[Section 6]\n{sec6_spec}\n"
        "Constraints: formal O&M tone; no unannotated faults; no unrelated equipment wording;"
        "Avoid data-analysis/model/CSV/LLM wording.\n"
        f"[Fact JSON]\n{fact_json}\n"
    )
    raw = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    obj = _extract_first_json_object(raw)
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    return {
        "sec3": _compact_str(obj.get("sec3")) or "TBD",
        "sec4": _compact_str(obj.get("sec4")) or "TBD",
        "sec6": _compact_str(obj.get("sec6")) or "TBD",
    }


def _default_sec6() -> str:
    return (
        "### Design highlights\nTBD\n\n"
        "### O&M notes\nTBD\n\n"
        "### Current status\n"
        "status：running\n"
        "Assessment：\n"
        "TBD\n\n"
        "### Next focused inspection recommendations\nTBD\n"
    )


def _build_markdown(
    basic_rows: List[List[str]],
    metric_rows: List[List[str]],
    rules_rows: List[List[str]],
    stats_rows: List[List[str]],
    sec3_md: str,
    sec4_md: str,
    fault_rows: List[List[str]],
    sec6_md: str,
) -> str:
    parts: List[str] = []
    parts.append("# Power Equipment Archive\n")
    parts.append("> Archive ledger / design and O&M notes (not a single-incident memo). Facts from model metadata and per_device annotations.\n")
    parts.append("## 1. Basic archive information\n")
    parts.append(_md_table(basic_rows))
    parts.append("## 2. Monitoring metric dictionary\n")
    parts.append(_md_table(metric_rows))
    parts.append("## 3. Manufacturing and procurement\n")
    parts.append(sec3_md.strip() + "\n")
    parts.append("\n## 4. Installation, commissioning and acceptance\n")
    parts.append(sec4_md.strip() + "\n")
    parts.append("\n## 5. Operating statistics and anomaly summary\n")
    parts.append("### 5.1 Operating statistics summary\n")
    parts.append(_md_table(stats_rows))
    parts.append("### 5.2 Applicable anomaly rules overview\n")
    parts.append(_md_table(rules_rows))
    parts.append("### 5.3 Historical anomaly summary (per_device annotations only)\n")
    parts.append(_md_table(fault_rows))
    parts.append("## 6. Design notes and O&M guidance\n")
    parts.append(sec6_md.strip() + "\n")
    return "\n".join(parts)


CHAPTER_DATA_SOURCES = [
    ["section", "data_source"],
    ["1 Basic archive information", "per_device JSON + xlsx device_list row; last status from CSV status column"],
    ["2 Monitoring metric dictionary", "metric_catalog.json; threshold_or_design_ref from rule_metric_stage2.json"],
    ["3 Manufacturing and procurement", "no structured source -> TBD; --use_llm may polish illustrative fields"],
    ["4 Installation, commissioning and acceptance", "no structured source -> TBD; --use_llm may polish commissioning steps"],
    ["5.1 Operating statistics summary", "per_device CSV aggregates (count/mean-or-mode/min/max/last)"],
    ["5.2 Applicable anomaly rules overview", "rule_metric_stage2.json intersect JSON eligible/chosen or xlsx applicable_rule_ids_all"],
    ["5.3 Historical anomaly summary", "per_device scenarios[type=fault, skipped=false]; description from stage2 judgment_description/note_cn"],
    ["6 Design notes and O&M guidance", "default placeholder; --use_llm polishes design/O&M from facts (no new faults)"],
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate device archive (ledger) markdown from catalog/stage2/xlsx/per_device."
    )
    ap.add_argument("--device_index", type=int, default=None, help="only generate one device by index")
    ap.add_argument("--out_dir", type=str, default="", help="default: <model_root>/device_archive")
    ap.add_argument(
        "--use_llm",
        action="store_true",
        help="polish sec3/4/6 and fault action_result (requires API key); must not invent unannotated faults",
    )
    ap.add_argument("--base_url", type=str, default="https://api.your-provider.com/v1")
    ap.add_argument("--model", type=str, default="deepseek-v3.2")
    ap.add_argument("--api_key", "--api-key", dest="api_key", default="", help="or OPENAI_API_KEY / LLM_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument(
        "--print_data_sources",
        action="store_true",
        help="print chapter→data-source table to stdout and exit",
    )
    ap.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="when generating all devices, spawn up to N parallel subprocesses (default 1)",
    )
    ap.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="model directory under data/demo/models (default: demo type)",
    )
    args = ap.parse_args()

    if args.print_data_sources:
        print(_md_table(CHAPTER_DATA_SOURCES))
        return 0

    from powerbench.paths import demo_models_dir, DEFAULT_DEMO_TYPE, repo_root
    root = repo_root(Path(__file__))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    model_root = Path(args.model_dir).resolve() if args.model_dir else (demo_models_dir() / DEFAULT_DEMO_TYPE)
    out_dir = Path(args.out_dir) if args.out_dir else (model_root / "device_archive")
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = _load_metric_catalog(model_root)
    db2cn, db2unit, _by_cn = _catalog_maps(catalog)
    rules = _load_rules_list(model_root)
    rid2text = _rule_id_to_text(rules)
    pairs = _pick_device_pairs(model_root)
    if args.device_index is not None:
        pairs = [x for x in pairs if x[0] == int(args.device_index)]
    if not pairs:
        raise SystemExit("no matched per_device csv/json pairs found")

    jobs = max(1, int(args.jobs))
    # Full-batch parallel: one subprocess per device (LLM-bound; overlapping I/O helps).
    if args.device_index is None and jobs > 1:
        import subprocess
        import sys
        from concurrent.futures import ThreadPoolExecutor, as_completed

        ok = failed = 0
        cmds: List[List[str]] = []
        for idx, *_rest in pairs:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--device_index", str(idx), "--model_dir", str(model_root)]
            if args.use_llm:
                cmd.append("--use_llm")
            cmd += ["--base_url", str(args.base_url), "--model", str(args.model)]
            cmd += ["--temperature", str(args.temperature), "--max_new_tokens", str(args.max_new_tokens)]
            if args.out_dir:
                cmd += ["--out_dir", str(args.out_dir)]
            cmds.append(cmd)

        child_env = os.environ.copy()
        api_key = _compact_str(args.api_key)
        if api_key:
            child_env["OPENAI_API_KEY"] = api_key
            child_env["LLM_API_KEY"] = api_key

        print(f"[PARALLEL] archive devices={len(cmds)} jobs={jobs}", flush=True)

        def _run(cmd: List[str]) -> int:
            return int(subprocess.run(cmd, env=child_env).returncode)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = [pool.submit(_run, c) for c in cmds]
            for fut in as_completed(futs):
                if fut.result() == 0:
                    ok += 1
                else:
                    failed += 1
        print(f"[PARALLEL] archive done. success={ok}, failed={failed}", flush=True)
        return 0 if failed == 0 else 1

    llm_client: Optional[OpenAICompatClient] = None
    if args.use_llm:
        api_key = _compact_str(args.api_key) or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
        if not api_key:
            raise SystemExit("--use_llm requires --api-key (or OPENAI_API_KEY / LLM_API_KEY)")
        llm_client = OpenAICompatClient(api_key=api_key, base_url=args.base_url, model=args.model)
    else:
        print(
            "[INFO] Sections 3/4/6 and fault-action cells are placeholders unless "
            "you pass --use_llm --api-key <KEY>."
        )

    for idx, name, cfg_path, csv_path, rep_path in pairs:
        cfg = _read_json(cfg_path)
        df = pd.read_csv(csv_path)
        xlsx_row = _load_xlsx_device_row(model_root, name)

        basic_rows = _build_basic_info(model_root, idx, name, cfg, df, xlsx_row)
        appl_ids = _applicable_rule_ids(cfg, xlsx_row)
        metric_rows = _build_metric_dictionary(catalog, rules, db2cn, list(df.columns), appl_ids)
        rules_rows = _build_rules_overview(rules, appl_ids)
        stats_rows = _build_runtime_stats(df, db2cn, db2unit)

        rep = _read_json(rep_path) if rep_path and rep_path.is_file() else {}
        history_src = _pick_fault_source(cfg, rep)
        fault_rows = _build_fault_summary(history_src, rule_text=rid2text)
        fault_rows = _maybe_llm_fill_fault_actions(
            fault_rows, llm_client, name, args.temperature, args.max_new_tokens
        )

        sec3_md = "TBD"
        sec4_md = "TBD"
        sec6_md = _default_sec6()

        if llm_client is not None:
            facts = {
                "model_name": model_root.name,
                "device_index": idx,
                "device_name": name,
                "basic_info_table": basic_rows[1:],
                "metric_dictionary": metric_rows[1:],
                "rules_overview": rules_rows[1:],
                "runtime_stats": stats_rows[1:],
                "fault_summary_from_per_device_only": fault_rows[1:],
                "available_override_fields": cfg.get("device_fields_override", {}),
                "xlsx_device_fields": xlsx_row,
                "time_range": cfg.get("time_range", {}),
                "writing_policy": (
                    "Sec 3/4 may add illustrative manufacturing/commissioning; sec 6 polishes design/O&M;"
                    "Do not add faults absent from fault_summary; no unrelated equipment wording."
                ),
            }
            try:
                filled = _llm_fill_sections(
                    llm_client,
                    facts=facts,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
                sec3_md = filled["sec3"]
                sec4_md = filled["sec4"]
                sec6_md = filled["sec6"]
                print(f"[OK] LLM filled sec3/4/6 for device {idx:04d}_{name}")
            except Exception as e:
                print(f"[WARN] LLM fill sections failed for device {idx:04d}: {e}")
                sec3_md, sec4_md, sec6_md = "TBD", "TBD", _default_sec6()

        md = _build_markdown(
            basic_rows=basic_rows,
            metric_rows=metric_rows,
            rules_rows=rules_rows,
            stats_rows=stats_rows,
            sec3_md=sec3_md,
            sec4_md=sec4_md,
            fault_rows=fault_rows,
            sec6_md=sec6_md,
        )
        safe_name = name
        try:
            from powerbench.generation.documents._local_imports import safe_fs_stem

            safe_name = safe_fs_stem(name)
        except Exception:
            safe_name = str(name).replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{idx:04d}_{safe_name}_device_archive.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"[OK] wrote: {out_path}")

    print(f"[DONE] total={len(pairs)} -> {out_dir}")
    # Avoid spamming this table on every single-device parallel worker.
    if args.device_index is None:
        print("\n# sectiondata_source\n")
        print(_md_table(CHAPTER_DATA_SOURCES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
