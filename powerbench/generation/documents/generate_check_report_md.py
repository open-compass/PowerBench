from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import re
import math
import requests
from requests import HTTPError


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _pick_applicable_rules(rep: Dict[str, Any]) -> List[str]:
    """Compatible rule list loader for different report schemas."""
    candidates = (
        rep.get("applicable_rules"),
        rep.get("eligible_rules_with_injector"),
        rep.get("chosen_rules_for_injection"),
    )
    for src in candidates:
        if not isinstance(src, list):
            continue
        out: List[str] = []
        seen = set()
        for x in src:
            rid = _compact_str(x)
            if not rid or rid in seen:
                continue
            seen.add(rid)
            out.append(rid)
        if out:
            return out
    return []


def _ensure_shared_pkg_on_path(model_root: Path | None = None) -> None:
    """Ensure repo root is on sys.path so `powerbench` is importable."""
    from powerbench.paths import repo_root
    root = repo_root(Path(model_root) if model_root else Path(__file__))
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_model_rule_functions(model_root: Path):
    """Load rule_functions via file path (model dir names like ZF27-1100(L) are not packages)."""
    import importlib.util as _ilu

    helper = Path(__file__).resolve().parent / "_local_imports.py"
    spec = _ilu.spec_from_file_location("m0002_report_local_imports", str(helper))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {helper}")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.load_rule_functions(Path(model_root))


def _safe_device_fs(name: str) -> str:
    try:
        import importlib.util as _ilu

        helper = Path(__file__).resolve().parent / "_local_imports.py"
        spec = _ilu.spec_from_file_location("m0002_report_safe_fs", str(helper))
        if spec is None or spec.loader is None:
            raise ImportError("no helper")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return str(mod.safe_fs_stem(name))
    except Exception:
        return str(name or "device").replace("/", "_").replace("\\", "_")


def _get_rule_registry(rf: Any) -> Dict[str, Any]:
    """Some models expose RULE_REGISTRY; others (e.g. 0022) use RULES."""
    for name in ("RULE_REGISTRY", "RULES"):
        reg = getattr(rf, name, None)
        if isinstance(reg, dict):
            return reg
    return {}


def _md_already_llm_filled(path: Path) -> bool:
    """True if md exists and no longer has LLM placeholders / markers."""
    if not path.is_file():
        return False
    try:
        t = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return ("TBD_LLM_FILL" not in t) and ("LLM_BEGIN" not in t)


def _extract_first_json_object(text: str) -> Any:
    """
    Extract first JSON object/array from possibly wrapped LLM output.
    """
    if not isinstance(text, str):
        raise ValueError("LLM output is not a string")
    t = text.strip()
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        return json.loads(t)
    start_obj = t.find("{")
    start_arr = t.find("[")
    starts = [x for x in (start_obj, start_arr) if x != -1]
    if not starts:
        raise ValueError("no JSON start found in output")
    start = min(starts)
    open_ch = t[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("could not find a complete JSON payload in output")


def _load_metric_catalog(model_root: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      - db2cn: db_field -> cn
      - cn2db: cn -> db_field
      - db2unit: db_field -> unit
    """
    items = _read_json(model_root / "metric_catalog.json")
    db2cn: Dict[str, str] = {}
    cn2db: Dict[str, str] = {}
    db2unit: Dict[str, str] = {}
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            cn = _compact_str(it.get("metric_name"))
            db = _compact_str(it.get("database_field"))
            unit = _compact_str(it.get("unit"))
            if cn and db:
                db2cn[db] = cn
                cn2db[cn] = db
            if db:
                db2unit[db] = unit
    return db2cn, cn2db, db2unit


def _pick_per_device_paths(model_root: Path, device_index: Optional[int]) -> Tuple[Path, Path]:
    per = model_root / "per_device"
    if device_index is None:
        rep_files = sorted(per.glob("*.json"), key=lambda p: p.name)
        if not rep_files:
            raise SystemExit(f"no per_device *.json found under: {per}")
        report_path = rep_files[0]
        prefix = report_path.name.split("_", 1)[0] + "_"
        csv_files = sorted([p for p in per.glob("*.csv") if p.name.startswith(prefix)], key=lambda p: p.name)
        if not csv_files:
            csv_files = sorted(per.glob("*.csv"), key=lambda p: p.name)
        if not csv_files:
            raise SystemExit(f"no per_device *.csv found under: {per}")
        return report_path, csv_files[0]
    rep_files = sorted(per.glob(f"{int(device_index):04d}_*.json"))
    csv_files = sorted(per.glob(f"{int(device_index):04d}_*.csv"))
    if not rep_files:
        raise SystemExit(f"missing report.json for device_index={device_index} under: {per}")
    if not csv_files:
        raise SystemExit(f"missing csv for device_index={device_index} under: {per}")
    return rep_files[0], csv_files[0]


def _md_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [len(x) for x in rows[0]]
    for r in rows[1:]:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    out = []
    def esc(x: str) -> str:
        return str(x).replace("\r", "").replace("|", "\\|")

    header = "| " + " | ".join(esc(c).ljust(widths[i]) for i, c in enumerate(rows[0])) + " |"
    sep = "| " + " | ".join("---".ljust(widths[i]) for i in range(len(widths))) + " |"
    out.append(header)
    out.append(sep)
    for r in rows[1:]:
        out.append("| " + " | ".join(esc(r[i]).ljust(widths[i]) for i in range(len(widths))) + " |")
    return "\n".join(out) + "\n"


@dataclass
class AnomalyHit:
    timestamp: str
    rule_id: str

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


def _extract_block(md: str, block_name: str) -> Tuple[str, str, str]:
    begin = f"<!-- LLM_BEGIN:{block_name} -->"
    end = f"<!-- LLM_END:{block_name} -->"
    i = md.find(begin)
    j = md.find(end)
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"missing LLM block: {block_name}")
    before = md[: i + len(begin)]
    inner = md[i + len(begin) : j]
    after = md[j:]
    return before, inner, after


def _replace_block(md: str, block_name: str, new_inner: str) -> str:
    begin = f"<!-- LLM_BEGIN:{block_name} -->"
    end = f"<!-- LLM_END:{block_name} -->"
    i = md.find(begin)
    j = md.find(end)
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"missing LLM block: {block_name}")
    return md[: i + len(begin)] + "\n" + new_inner.strip() + "\n" + md[j:]


def _strip_llm_markers(md: str) -> str:
    out_lines: List[str] = []
    for line in md.splitlines():
        if line.strip().startswith("<!-- LLM_BEGIN:") and line.strip().endswith("-->"):
            continue
        if line.strip().startswith("<!-- LLM_END:") and line.strip().endswith("-->"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if md.endswith("\n") else "")


def _remove_rule_ids_in_text(text: str) -> str:
    """
    Remove internal rule IDs like K1/K12 from human-facing summary text.
    """
    s = _compact_str(text)
    if not s:
        return s
    s = re.sub(r"\b[Kk]\s*\d+\b", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"[ \t]+([,，。；;:：\n])", r"\1", s)
    return s


def _enforce_inspection_tone(text: str) -> str:
    """
    Make prose read like a conventional inspection record, not a data analysis.
    Applies conservative phrasing replacements only for clearly analytical terms.
    """
    s = _compact_str(text)
    if not s:
        return s
    # Targeted replacements (keep minimal to avoid unintended changes)
    repl = [
        (r"data analysis", "inspection"),
        (r"analysis", "review"),
        (r"database", "system records"),
        (r"\bCSV\b", "system records"),
        (r"\bLLM\b", ""),
        (r"model", ""),
        (r"algorithm", ""),
        (r"dataset", "records"),
        (r"assessment", "assessment"),
        (r"judgment", "judgment"),
    ]
    for pat, rep in repl:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


def _renumber_md_table_rows(md_rows: str, start: int) -> str:
    """
    Renumber markdown table rows (lines starting with '|') by overwriting the first column.
    Intended for appending extra rows to an existing table where numbering must continue.
    """
    s = _compact_str(md_rows)
    if not s:
        return s
    out_lines: List[str] = []
    n = int(start)
    for line in s.splitlines():
        ln = line.rstrip("\r")
        if not ln.strip().startswith("|"):
            # keep non-row lines as-is (though we instruct LLM not to output them)
            if ln.strip():
                out_lines.append(ln)
            continue
        parts = [p.strip() for p in ln.strip().strip("|").split("|")]
        if len(parts) < 4:
            continue
        parts[0] = str(n)
        n += 1
        out_lines.append("| " + " | ".join(parts[:4]) + " |")
    return "\n".join(out_lines).strip()


def _pick_section3_check_fields(df: pd.DataFrame, raw_csv_cols: List[str]) -> List[str]:
    """
    Section-3 candidates = columns present on this device CSV (header order),
    excluding timestamp/device_name. Includes numeric + discrete status fields.
    Do not invent foreign-model columns.
    """
    skip = {"timestamp", "device_name"}
    out: List[str] = []
    for c in raw_csv_cols:
        if c in skip or c not in df.columns:
            continue
        out.append(c)
    return out


def _field_match_tokens(col: str, db2cn: Dict[str, str]) -> List[str]:
    """
    Tokens linking CSV db field to stage2 rule_expression / judgment_description.
    Built from catalog metric_name + local expr aliases for THIS model only.
    """
    toks = [col]
    cn = _compact_str(db2cn.get(col, ""))
    if cn:
        toks.append(cn)
    # stage2 sometimes uses symbolic lhs; map only 0007-local aliases
    local_rev = {
        "top_oil_temp": ["T_oil", "top_oil_temp"],
        "winding_temp": ["T_winding"],
        "winding_hotspot_temp_fiber": ["T_hotspot"],
        "oil_dissolved_gas": ["oil_dissolved_gas"],
        "i_rated": ["I_rated"],
        "u_rated": ["U_rated"],
    }
    toks.extend(local_rev.get(col, []))
    seen: set[str] = set()
    out: List[str] = []
    for t in toks:
        t2 = _compact_str(t)
        if not t2 or t2 in seen:
            continue
        seen.add(t2)
        out.append(t2)
    return out


def _llm_fill_blocks_json(
    *,
    client: OpenAICompatClient,
    facts: Dict[str, Any],
    blocks: Dict[str, str],
    temperature: float,
    max_new_tokens: Optional[int],
) -> Dict[str, str]:
    fact_json = json.dumps(facts, ensure_ascii=False, indent=2)
    blocks_json = json.dumps(blocks, ensure_ascii=False, indent=2)
    prompt = (
        "Fill placeholder blocks for a power-equipment inspection report.\n"
        "Output one JSON object: keys are block names, values are Markdown content (no BEGIN/END markers).\n"
        "\n"
        "Hard constraints:\n"
        "1) JSON only; no extra text.\n"
        "2) Do not invent values, timestamps, rule ids, or thresholds.\n"
        "3) Quote device_name, time range, and rules only from the fact JSON.\n"
        "4) You may add safety measures and conclusions without conflicting facts.\n"
        "5) In section_6_summary, do not use rule ids (K1, K2, K3).\n"
        "6) Neutral inspection-record tone; avoid data-analysis/model/algorithm/database/CSV/LLM wording.\n"
        "7) section_2_safety: bulleted executable safety steps for converter-transformer sites.\n"
        "8) section_3_extra_checks: qualitative extra checks as Markdown table rows (4 columns), blank seq column.\n"
        "\n"
        f"[Fact JSON]\n{fact_json}\n\n"
        f"[Blocks JSON]\n{blocks_json}\n"
    )
    raw = client.chat([{"role": "user", "content": prompt}], temperature=temperature, max_new_tokens=max_new_tokens)
    js = _extract_first_json_object(raw)
    if not isinstance(js, dict):
        raise ValueError("LLM did not return a JSON object")
    out: Dict[str, str] = {}
    for k in blocks.keys():
        v = js.get(k)
        if not isinstance(v, str):
            raise ValueError(f"LLM JSON missing string for key={k}")
        out[k] = v
    return out


def _build_check_facts(
    *,
    model_root: Path,
    report_path: Path,
    csv_path: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    device_name: str,
    hits: List[AnomalyHit],
) -> Dict[str, Any]:
    rep = _read_json(report_path)
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    # small snapshot at end
    snap: Dict[str, Any] = {}
    if end_ts in df.index:
        row = df.loc[end_ts]
        for k, v in row.items():
            if v is None:
                continue
            try:
                if not isinstance(v, str) and pd.isna(v):
                    continue
            except Exception:
                pass
            snap[k] = v if isinstance(v, str) else float(v) if isinstance(v, (int, float)) else str(v)
    nm_f = _filter_near_miss_items_for_inspection_end(rep.get("near_miss"), end_ts)
    nmc_f = _filter_near_miss_cluster_for_inspection_end(rep.get("near_miss_cluster"), end_ts)
    return {
        "device_name": device_name,
        "device_fields_override": rep.get("device_fields_override") or {},
        "dimension_cols": rep.get("dimension_cols") or [],
        "window_start": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "window_end": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": {"csv": str(csv_path), "device_report_json": str(report_path)},
        # 0007: no applicable_rules key — expose eligible/chosen for LLM facts
        "applicable_rules": _pick_applicable_rules(rep),
        "eligible_rules_with_injector": rep.get("eligible_rules_with_injector") or [],
        "near_miss_cluster": nmc_f,
        "near_miss": nm_f,
        "near_miss_count": len(nm_f) if isinstance(nm_f, list) else 0,
        "point_only_rule_hits_in_window": [{"rule_id": h.rule_id, "timestamp": h.timestamp} for h in hits],
        "snapshot_at_window_end": snap,
    }


def _load_stage2_rules(model_root: Path) -> Dict[str, Dict[str, Any]]:
    stage2 = _read_json(model_root / "rule_metric_stage2.json")
    out: Dict[str, Dict[str, Any]] = {}
    for r in stage2.get("anomaly_judgment_rules", []) or []:
        if isinstance(r, dict) and r.get("rule_id"):
            out[_compact_str(r["rule_id"])] = r
    return out


def _is_point_only_rule(rule_obj: Dict[str, Any]) -> bool:
    """
    Pointwise: rule_expression contains no window/temporal functions.
    Strictly by expression semantics whether a history window is required.
    """
    expr = _compact_str(rule_obj.get("rule_expression"))
    if not expr:
        return False
    bad_tokens = [
        "consec_",
        "duration_",
        "avg_",
        "std_",
        "slope_",
        "count_",
        "max_",
        "min_",
    ]
    s = expr.replace(" ", "")
    return not any(tok in s for tok in bad_tokens)


def _alias_to_db_field(token: str) -> str:
    """
    Map stage2 symbolic identifiers to per_device CSV db fields.
    0007: prefer identity for db field names already used in exprs;
    keep a few local aliases. Do NOT map to 0002 converter oil/DGA columns.
    """
    t = _compact_str(token)
    m = {
        # 0002 ±800kV converter transformer — keep identity for catalog db fields;
        # only map a few stage2 symbolic extras used in expressions.
        "oil_dissolved_gas": "oil_dissolved_gas",
        "hotspot_temperature_rise": "hotspot_temperature_rise",
        "winding_to_oil_temp_differential": "winding_to_oil_temp_differential",
        "apparent_power": "apparent_power",
    }
    return m.get(t, t)


def _parse_point_comparisons(expr: str) -> List[Tuple[str, str, str]]:
    """
    Extract simple point comparisons like X > T_H2 or oil_dga_h2 < L_min.
    Returns list of (lhs, op, rhs) as raw tokens.
    """
    s = _compact_str(expr)
    if not s:
        return []
    # normalize a bit
    s = s.replace("AND", "and").replace("OR", "or").replace("tanδ", "tan_delta")
    # very lightweight regex (we only use for "near threshold" hints)
    out: List[Tuple[str, str, str]] = []
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(>=|<=|>|<)\s*\b([A-Za-z_][A-Za-z0-9_]*)\b", s):
        out.append((m.group(1), m.group(2), m.group(3)))
    return out


def _near_threshold_hints(
    *,
    df: pd.DataFrame,
    ts: pd.Timestamp,
    applicable_point_rules: List[str],
    stage2_rules: Dict[str, Dict[str, Any]],
    params_by_rule: Dict[str, Dict[str, float]],
    db2cn: Dict[str, str],
    db2unit: Dict[str, str],
    ratio: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    If no anomaly triggered, still highlight metrics close to thresholds.
    Only for point-only rules and simple comparisons.
    """
    hints: List[Dict[str, Any]] = []
    snap = df.loc[ts] if ts in df.index else None
    if snap is None:
        return hints
    for rid in applicable_point_rules:
        r = stage2_rules.get(rid) or {}
        expr = _compact_str(r.get("rule_expression"))
        comps = _parse_point_comparisons(expr)
        pmap = params_by_rule.get(rid, {})
        for lhs, op, rhs in comps:
            if rhs not in pmap:
                continue
            thr = pmap.get(rhs)
            if thr is None or (isinstance(thr, float) and math.isnan(thr)):
                continue
            col = _alias_to_db_field(lhs)
            if col not in df.columns:
                continue
            v = snap.get(col)
            try:
                v_f = float(v)
            except Exception:
                continue
            thr_f = float(thr)
            # compute "distance to trigger" for the comparison
            if op in (">", ">="):
                dist = thr_f - v_f  # <=0 means already triggered
            else:
                dist = v_f - thr_f
            if dist <= 0:
                continue
            denom = max(1e-9, abs(thr_f))
            if dist / denom <= ratio:
                hints.append(
                    {
                        "rule_id": rid,
                        "metric": db2cn.get(col, col),
                        "unit": db2unit.get(col, ""),
                        "observed_value": v_f,
                        "threshold": thr_f,
                        "distance_to_threshold": dist,
                        "comparison": f"{lhs} {op} {rhs}",
                    }
                )
    return hints


def _select_ts_near_threshold(
    *,
    df: pd.DataFrame,
    ts_candidates: List[pd.Timestamp],
    applicable_point_rules: List[str],
    stage2_rules: Dict[str, Dict[str, Any]],
    params_by_rule: Dict[str, Dict[str, float]],
    prefer_no_trigger: bool = True,
) -> pd.Timestamp:
    """
    Auto-pick an inspection timestamp that is as close as possible to point-rule thresholds.

    Strategy (high level):
    - For each candidate ts, compute the smallest "distance to trigger" across all simple comparisons.
      - For 'X > THR': distance = THR - X (>=0 means not triggered yet, closer to 0 is nearer)
      - For 'X < THR': distance = X - THR
    - Use *relative* distance: dist / max(|THR|, 1) so different-unit thresholds are comparable.
    - Prefer timestamps that do NOT trigger any point-only rules (strict check via rule_functions) when enabled.
    - Among preferred set, pick the minimal positive relative distance; fallback to minimal absolute relative distance.
    """
    if not ts_candidates:
        raise SystemExit("no timestamp candidates")

    # Pre-parse comparisons per rule (used for scoring only)
    comps_by_rule: Dict[str, List[Tuple[str, str, str]]] = {}
    for rid in applicable_point_rules:
        r = stage2_rules.get(rid) or {}
        comps_by_rule[rid] = _parse_point_comparisons(_compact_str(r.get("rule_expression")))

    best_ts: Optional[pd.Timestamp] = None
    best_key: Optional[Tuple[int, float, float]] = None

    any_comparison_seen = False

    # We'll do a strict trigger check at candidate-level (point-only, so it stays fast for reasonable ranges)
    # Cache: timestamp -> whether any rule triggers
    trigger_cache: Dict[pd.Timestamp, bool] = {}

    # prepare strict registry once
    # (done lazily inside helper to avoid circular import at module import time)
    def _any_trigger_strict(ts: pd.Timestamp) -> bool:
        if ts in trigger_cache:
            return trigger_cache[ts]
        pkg = model_root_name = None  # satisfy type checker; actual values resolved below
        # This inner function is called from main() where sys.path is already set.
        # We'll import rule_functions here.
        try:
            # model_root isn't available here; but __package__ is set in main().
            pkg = globals().get("__package__", None)
            if not pkg:
                # fall back: infer from cwd by scanning df? can't; so assume no trigger
                trigger_cache[ts] = False
                return False
            rf = _import_model_rule_functions(Path(__file__).resolve().parent.parent)
            registry = _get_rule_registry(rf)
        except Exception:
            trigger_cache[ts] = False
            return False

        snap_hits = 0
        mr = Path(__file__).resolve().parent.parent
        for rid in applicable_point_rules:
            fn = registry.get(rid)
            if fn is None:
                continue
            if _rule_fn_triggered(fn, mr, df, ts, params_by_rule.get(rid, {})):
                snap_hits += 1
                break
        trigger_cache[ts] = snap_hits > 0
        return trigger_cache[ts]

    for ts in ts_candidates:
        if ts not in df.index:
            continue
        snap = df.loc[ts]

        # Score: nearest positive *relative* distance to any threshold; and nearest absolute relative distance as fallback.
        best_pos = float("inf")
        best_abs = float("inf")

        for rid in applicable_point_rules:
            pmap = params_by_rule.get(rid, {})
            for lhs, op, rhs in comps_by_rule.get(rid, []):
                if rhs not in pmap:
                    continue
                thr = pmap.get(rhs)
                if thr is None or (isinstance(thr, float) and math.isnan(thr)):
                    continue
                col = _alias_to_db_field(lhs)
                if col not in df.columns:
                    continue
                v = snap.get(col)
                try:
                    v_f = float(v)
                    thr_f = float(thr)
                except Exception:
                    continue

                if op in (">", ">="):
                    dist = thr_f - v_f
                else:
                    dist = v_f - thr_f

                any_comparison_seen = True
                denom = max(1.0, abs(thr_f))
                rel = dist / denom
                best_abs = min(best_abs, abs(rel))
                if rel >= 0:
                    best_pos = min(best_pos, rel)

        any_trigger = _any_trigger_strict(ts) if prefer_no_trigger else False
        key = (1 if any_trigger else 0, best_pos, best_abs)
        if best_key is None or key < best_key:
            best_key = key
            best_ts = ts

    if not any_comparison_seen:
        # Nothing to score on (no parseable comparisons / missing threshold params / missing cols)
        # Fall back to earliest candidate and let the report show near-threshold hints table (likely empty).
        for ts in ts_candidates:
            if ts in df.index:
                return ts
        return ts_candidates[0]

    if best_ts is None:
        # fallback: first available
        for ts in ts_candidates:
            if ts in df.index:
                return ts
        return ts_candidates[0]

    return best_ts


def _load_params_by_rule(model_root: Path) -> Dict[str, Dict[str, float]]:
    # Use existing per-rule parameter loader from pipeline_generate.rule_functions
    mod = _import_model_rule_functions(model_root)
    return mod.load_params_by_rule(model_root)  # type: ignore[attr-defined]


def _rule_fn_triggered(
    fn: Any,
    model_root: Path,
    df: pd.DataFrame,
    ts: pd.Timestamp,
    params: Dict[str, Any],
) -> bool:
    """
    Call a rule from RULE_REGISTRY.

    - Some models: fn(df, ts, p) -> object with .triggered
    - Others: fn(model_dir, df, ts, fields_override) -> (bool, details_dict)
    """
    try:
        res = fn(model_root, df, ts, params)  # type: ignore[misc]
    except TypeError:
        res = fn(df, ts, params)  # type: ignore[misc]
    if isinstance(res, tuple):
        return bool(res[0]) if len(res) > 0 else False
    return bool(getattr(res, "triggered", False))


def _eval_rules_in_range(
    model_root: Path,
    df: pd.DataFrame,
    ts_list: List[pd.Timestamp],
    applicable_rules: List[str],
) -> List[AnomalyHit]:
    rf = _import_model_rule_functions(model_root)
    registry = _get_rule_registry(rf)
    params_by_rule = _load_params_by_rule(model_root)
    hits: List[AnomalyHit] = []
    for ts in ts_list:
        for rid in applicable_rules:
            fn = registry.get(rid)
            if fn is None:
                continue
            if _rule_fn_triggered(fn, model_root, df, ts, params_by_rule.get(rid, {})):
                hits.append(AnomalyHit(ts.strftime("%Y-%m-%d %H:%M:%S"), rid))
    return hits


def _eval_point_rules_first_hit_in_window(
    model_root: Path,
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    applicable_rules: List[str],
) -> List[AnomalyHit]:
    """
    Walk timestamps in [start_ts, end_ts] in order; for each point-only rule, record the first
    timestep where it triggers. Avoids section 5 reflecting only the window-end snapshot (often
    dominated by one ratio rule at end_ts).
    """
    rf = _import_model_rule_functions(model_root)
    registry = _get_rule_registry(rf)
    params_by_rule = _load_params_by_rule(model_root)
    hits: List[AnomalyHit] = []
    seen: set[str] = set()
    try:
        sub = df.loc[start_ts:end_ts]
    except Exception:
        return hits
    for ts in sub.index:
        ts = pd.Timestamp(ts)
        for rid in applicable_rules:
            if rid in seen:
                continue
            fn = registry.get(rid)
            if fn is None:
                continue
            if _rule_fn_triggered(fn, model_root, df, ts, params_by_rule.get(rid, {})):
                hits.append(AnomalyHit(ts.strftime("%Y-%m-%d %H:%M:%S"), rid))
                seen.add(rid)
        if len(seen) >= len(applicable_rules):
            break
    hits.sort(key=lambda h: (h.timestamp, h.rule_id))
    return hits


def _filter_near_miss_items_for_inspection_end(items: Any, inspection_end: pd.Timestamp) -> List[Dict[str, Any]]:
    """Keep near_miss rows whose episode ends on or before the inspection window end (no future episodes)."""
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        we_s = _compact_str(it.get("window_end"))
        ws_s = _compact_str(it.get("window_start"))
        if we_s:
            t_end = pd.Timestamp(we_s)
            if pd.isna(t_end) or t_end > inspection_end:
                continue
        elif ws_s:
            t_start = pd.Timestamp(ws_s)
            if pd.isna(t_start) or t_start > inspection_end:
                continue
        else:
            continue
        out.append(it)
    return out


def _filter_near_miss_cluster_for_inspection_end(nm_cl: Any, inspection_end: pd.Timestamp) -> Dict[str, Any]:
    if not isinstance(nm_cl, dict):
        return {}
    end_s = _compact_str(nm_cl.get("end"))
    if not end_s:
        return {}
    t_end = pd.Timestamp(end_s)
    if pd.isna(t_end) or t_end > inspection_end:
        return {}
    return dict(nm_cl)


def _snap_timestamp_to_csv_index(ts: pd.Timestamp, idx: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """Use ts if it is an index label; otherwise pick the nearest timestamp in idx."""
    ts = pd.Timestamp(ts)
    if len(idx) == 0:
        return None
    if ts in idx:
        return ts
    pos = int(idx.searchsorted(ts, side="left"))
    best: Optional[pd.Timestamp] = None
    best_sec: Optional[float] = None
    for p in (pos - 1, pos):
        if 0 <= p < len(idx):
            cand = pd.Timestamp(idx[p])
            d = abs((cand - ts).total_seconds())
            if best_sec is None or d < best_sec:
                best_sec = d
                best = cand
    return best


def _inspection_day_windows(rep: Dict[str, Any], df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Build inspection windows for report generation.

    Strategy (logged):
      1) Prefer per_device JSON ``inspection_days`` (one window per entry:
         start=snapped day ts, end=start+24h snapped). 0002-compatible.
      2) Else degrade to unique ``checks[].timestamp`` (same start/end+24h rule).
      3) Else empty → caller must pass CLI --ts / --start/--end.

    This model exports inspection_days from pipeline_generate; the checks fallback
    is only for older JSON without that field.
    """
    idx = pd.DatetimeIndex(df.index).sort_values()

    def _windows_from_ts_list(raw_items: List[Any], *, strategy: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        seen: set[str] = set()
        out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for item in raw_items:
            s = _compact_str(item)
            if not s:
                continue
            t = pd.Timestamp(s)
            if pd.isna(t):
                continue
            start_ts = _snap_timestamp_to_csv_index(t, idx)
            if start_ts is None:
                print(f"[INSPECTION] skip ({strategy}): cannot map {s!r} onto CSV timestamps")
                continue
            end_target = pd.Timestamp(start_ts) + pd.Timedelta(hours=24)
            end_ts = _snap_timestamp_to_csv_index(end_target, idx)
            if end_ts is None:
                print(f"[INSPECTION] skip ({strategy}): cannot map end for {s!r}")
                continue
            if end_ts < start_ts:
                print(f"[INSPECTION] skip ({strategy}): end before start for {s!r}")
                continue
            key = f"{start_ts.strftime('%Y-%m-%d %H:%M:%S')}|{end_ts.strftime('%Y-%m-%d %H:%M:%S')}"
            if key in seen:
                continue
            seen.add(key)
            out.append((start_ts, end_ts))
        if out:
            print(f"[INSPECTION] strategy={strategy} windows={len(out)}")
        return out

    raw = rep.get("inspection_days") or []
    if isinstance(raw, list) and raw:
        wins = _windows_from_ts_list(raw, strategy="inspection_days")
        if wins:
            return wins

    # Fallback: checks[].timestamp (document-side degrade; not a substitute for planned calendar)
    checks = rep.get("checks") or []
    check_ts: List[Any] = []
    if isinstance(checks, list):
        for c in checks:
            if isinstance(c, dict):
                check_ts.append(c.get("timestamp"))
            else:
                check_ts.append(c)
    if check_ts:
        wins = _windows_from_ts_list(check_ts, strategy="checks[].timestamp(fallback)")
        if wins:
            return wins

    print("[INSPECTION] strategy=none (no inspection_days / usable checks timestamps)")
    return []


def _render_check_report_markdown(
    model_root: Path,
    device_name: str,
    df: pd.DataFrame,
    raw_csv_cols: List[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    applicable_rules: List[str],
    applicable_rules_all: List[str],
    hits: List[AnomalyHit],
    db2cn: Dict[str, str],
    db2unit: Dict[str, str],
    *,
    rep: Optional[Dict[str, Any]] = None,
    debug_section3: bool = False,
) -> str:
    df_win = df.loc[start_ts:end_ts]
    snap = df.loc[end_ts] if end_ts in df.index else None
    rep = rep if isinstance(rep, dict) else {}
    override = rep.get("device_fields_override") if isinstance(rep.get("device_fields_override"), dict) else {}
    nm_filtered = _filter_near_miss_items_for_inspection_end(rep.get("near_miss"), end_ts)

    md: List[str] = []
    md.append("# Inspection Report\n\n")

    md.append("## 1. Basic information\n\n")
    md.append(f"- **Device**: {device_name}\n")
    md.append(f"- **Model directory**: {model_root.name}\n")
    md.append(f"- **Inspection period**: {start_ts.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_ts.strftime('%Y-%m-%d %H:%M:%S')}（24h）\n")
    for k in (rep.get("dimension_cols") or list(override.keys())):
        kk = _compact_str(k)
        if not kk:
            continue
        vv = _compact_str(override.get(kk))
        if vv:
            md.append(f"- **{kk}**: {vv}\n")
    md.append("\n")

    md.append("## 2. Pre-job safety measures\n\n")
    md.append("<!-- LLM_BEGIN:section_2_safety -->\n")
    md.append("TBD_LLM_FILL (do not invent values/timestamps/rule ids).\n")
    md.append("<!-- LLM_END:section_2_safety -->\n\n")

    md.append("## 3. Patrol inspection checklist\n\n")
    # One row per check_item result table
    # Header: seq | check_item | standard_or_criterion | conclusion (normal/abnormal) | notes
    check_fields = _pick_section3_check_fields(df, raw_csv_cols)

    # Conclusion at inspection end from pointwise rule hits
    # - Abnormal if a hit rule references the check item, else normal
    # - standard_or_criterion from rule text when available, else empty
    field_to_criterion: Dict[str, str] = {}
    field_to_abnormal: Dict[str, bool] = {f: False for f in check_fields}

    # Link via judgment_description and expr field/alias tokens (not numeric formulas)
    try:
        stage2_rules = _load_stage2_rules(model_root)
        if debug_section3:
            print("[SECTION3_DEBUG] check_fields(db) =", check_fields)
            print("[SECTION3_DEBUG] applicable_rules(point_only) =", applicable_rules)
            print("[SECTION3_DEBUG] applicable_rules_all =", applicable_rules_all)

        # Build match tokens for each candidate field (catalog-driven, no foreign-model columns).
        field_tokens: Dict[str, List[str]] = {col: _field_match_tokens(col, db2cn) for col in check_fields}
        if debug_section3:
            print("[SECTION3_DEBUG] field_tokens =")
            for k in check_fields:
                print(f"[SECTION3_DEBUG]   - {k}: {field_tokens.get(k, [])}")
        # Collect judgment_description from related rules per field
        for rid in applicable_rules_all:
            r = stage2_rules.get(rid) or {}
            desc = _compact_str(r.get("judgment_description"))
            expr = _compact_str(r.get("rule_expression"))
            if debug_section3:
                print(f"[SECTION3_DEBUG] rule={rid} expr={expr!r}")
                print(f"[SECTION3_DEBUG] rule={rid} judgment_description={desc!r}")
            if not expr or not desc:
                continue
            for col in check_fields:
                # Rule expr containing field name/alias counts as related
                matched = False
                for tok in field_tokens.get(col, [col]):
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
                        pat = r"\b" + re.escape(tok) + r"\b"
                        m = re.search(pat, expr)
                        if debug_section3:
                            print(f"[SECTION3_DEBUG]   - try field={col} tok={tok} pat={pat!r} matched={bool(m)}")
                        if m:
                            matched = True
                            break
                    else:
                        m2 = tok in expr
                        if debug_section3:
                            print(f"[SECTION3_DEBUG]   - try field={col} tok={tok} (substr) matched={bool(m2)}")
                        if m2:
                            matched = True
                            break
                if debug_section3:
                    print(f"[SECTION3_DEBUG]     => field={col} matched_any={matched}")
                if matched:
                    prev = field_to_criterion.get(col, "")
                    items = [x for x in [p.strip() for p in prev.split("；")] if x] if prev else []
                    if desc not in items:
                        items.append(desc)
                    field_to_criterion[col] = "；".join(items)
                    if debug_section3:
                        print(f"[SECTION3_DEBUG]     -> field={col} add judgment_description; now={field_to_criterion[col]!r}")
        # Mark abnormal if end_ts hits a rule whose expr references the field/alias
        hits_set = {h.rule_id for h in hits}
        if debug_section3:
            print("[SECTION3_DEBUG] hits(rule_id) =", sorted(list(hits_set)))
        for rid in hits_set:
            r = stage2_rules.get(rid) or {}
            expr = _compact_str(r.get("rule_expression"))
            if not expr:
                continue
            for col in check_fields:
                matched = False
                for tok in field_tokens.get(col, [col]):
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
                        pat = r"\b" + re.escape(tok) + r"\b"
                        if re.search(pat, expr):
                            matched = True
                            break
                    else:
                        if tok in expr:
                            matched = True
                            break
                if matched and col in field_to_abnormal:
                    field_to_abnormal[col] = True
                    if debug_section3:
                        print(f"[SECTION3_DEBUG]     -> abnormal field={col} due to hit rule={rid}")
    except Exception as e:
        if debug_section3:
            print("[SECTION3_DEBUG][ERROR] exception while building section 3 criteria:", repr(e))
            traceback.print_exc()

    if debug_section3:
        print("[SECTION3_DEBUG] field_to_criterion(final) =")
        for k in check_fields:
            print(f"[SECTION3_DEBUG]   - {k}: {field_to_criterion.get(k,'')!r}")

    # Keep only rows with non-empty criterion.
    kept_fields: List[str] = [db for db in check_fields if _compact_str(field_to_criterion.get(db, ""))]
    rows = [["seq", "check_item", "standard_or_criterion", "conclusion（normal/abnormal）"]]
    if kept_fields:
        for idx_i, db in enumerate(kept_fields, start=1):
            concl = "abnormal" if field_to_abnormal.get(db, False) else "normal"
            crit = field_to_criterion.get(db, "")
            rows.append([str(idx_i), db2cn.get(db, db), crit, concl])
    else:
        rows.append(["1", "(no items matching criteria)", "", "normal"])
    md.append(_md_table(rows))
    # Append extra qualitative checks as additional table rows (LLM only outputs rows, no header)
    md.append("<!-- LLM_BEGIN:section_3_extra_checks -->\n")
    md.append("|  | Body appearance and fasteners | No obvious rust, deformation, leakage (oil) or cracks (dry-type); fasteners complete | normal |\n")
    md.append("<!-- LLM_END:section_3_extra_checks -->\n")
    md.append("\n")

    # Key metric snapshot at inspection end (device CSV columns only)
    md.append("## 3.1 Key metric snapshot at inspection end\n\n")
    snap_rows = [["metric", "unit", "value"]]
    if snap is not None:
        for c in raw_csv_cols:
            if c in {"timestamp", "device_name"} or c not in df.columns:
                continue
            v = snap.get(c) if hasattr(snap, "get") else snap[c]
            if not _compact_str(v):
                continue
            try:
                if not isinstance(v, str) and pd.isna(v):
                    continue
            except Exception:
                pass
            unit = _compact_str(db2unit.get(c, ""))
            snap_rows.append([db2cn.get(c, c), unit or "-", _compact_str(v)])
    if len(snap_rows) == 1:
        snap_rows.append(["(no snapshot available)", "-", "-"])
    md.append(_md_table(snap_rows))
    md.append("\n")

    md.append("## 4. Key metrics 24h time series\n\n")
    rows4: List[List[str]] = []
    # Strictly use columns from the original CSV header to avoid including
    # derived/intermediate columns added during rule evaluation.
    metric_cols = [c for c in raw_csv_cols if c not in {"timestamp", "device_name"} and c in df.columns]
    # Drop columns that have no usable value in the 24h window.
    if not df_win.empty:
        kept_metric_cols: List[str] = []
        for c in metric_cols:
            s = df_win[c]
            has_value = False
            for v in s.tolist():
                if _compact_str(v):
                    has_value = True
                    break
            if has_value:
                kept_metric_cols.append(c)
        metric_cols = kept_metric_cols
    header = ["timestamp"]
    for c in metric_cols:
        cn = db2cn.get(c, c)
        unit = _compact_str(db2unit.get(c, ""))
        header.append(f"{cn}({unit})" if unit else cn)
    rows4.append(header)
    if not df_win.empty:
        for ts, row in df_win.iterrows():
            line = [pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")]
            for c in metric_cols:
                v = row.get(c)
                if v is None:
                    line.append("")
                elif not isinstance(v, str) and pd.isna(v):
                    line.append("")
                else:
                    line.append(_compact_str(v))
            rows4.append(line)
    else:
        rows4.append(["(no data)"] + [""] * len(metric_cols))
    md.append(_md_table(rows4))
    md.append("\n")

    md.append("## 5. Defect and hazard register\n\n")
    stage2_rules_s5 = _load_stage2_rules(model_root)
    if hits:
        md.append("### 5.1 Pointwise rule triggers in this inspection window\n\n")
        hrows = [["rule_id", "trigger_time", "object_or_judgment"]]
        for h in hits:
            r = stage2_rules_s5.get(h.rule_id) or {}
            desc = _compact_str(r.get("judgment_description")) or _compact_str(r.get("object_name")) or "-"
            hrows.append([h.rule_id, h.timestamp, desc])
        md.append(_md_table(hrows))
        md.append("\n")
    else:
        md.append("No pointwise anomaly rule triggers in this inspection window.\n\n")

    md.append("### 5.2 Near threshold (near_miss, not after inspection end)\n\n")
    if nm_filtered:
        nm_rows = [["rule_id", "window_start", "window_end", "description"]]
        for it in nm_filtered:
            if not isinstance(it, dict) or bool(it.get("skipped")):
                continue
            nm_rows.append(
                [
                    _compact_str(it.get("rule_id")),
                    _compact_str(it.get("window_start")),
                    _compact_str(it.get("window_end")),
                    _compact_str(it.get("note_cn")),
                ]
            )
        if len(nm_rows) == 1:
            md.append("No valid near_miss entries after filtering.\n\n")
        else:
            md.append(_md_table(nm_rows))
            md.append("\n")
    else:
        md.append("No near_miss records before inspection end (future events excluded).\n\n")

    md.append("## 6. Patrol conclusions and recommendations\n\n")
    md.append("<!-- LLM_BEGIN:section_6_summary -->\n")
    md.append(
        "TBD_LLM_FILL (no invented values/timestamps/rule ids; conclusion must match sections 3-5; "
        f"near_miss count={len(nm_filtered)}, write to actual count).\n"
    )
    md.append("<!-- LLM_END:section_6_summary -->\n\n")
    return "".join(md)


def main() -> int:
    from powerbench.paths import demo_models_dir, DEFAULT_DEMO_TYPE, repo_root

    this_dir = Path(__file__).resolve().parent
    model_root = demo_models_dir() / DEFAULT_DEMO_TYPE
    _ensure_shared_pkg_on_path(model_root)
    root = repo_root(Path(__file__))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(description="Generate inspection report from per_device CSV + rules.")
    ap.add_argument(
        "--model_dir",
        type=str,
        default=str(model_root),
        help="model directory under data/demo/models",
    )
    ap.add_argument("--device_index", type=int, default=None)
    ap.add_argument("--ts", type=str, default="", help="inspection end timestamp. If provided, window is [ts-23h, ts].")
    ap.add_argument(
        "--start",
        type=str,
        default="",
        help="inspection window start (inclusive); requires --end unless using default inspection_days.",
    )
    ap.add_argument(
        "--end",
        type=str,
        default="",
        help="inspection window end (inclusive). If omitted together with --ts/--start, inspection_days from per_device JSON is used.",
    )
    ap.add_argument(
        "--prefer_no_trigger",
        action="store_true",
        help="when auto-picking, prefer timestamps that do NOT trigger any point-only rules",
    )
    ap.add_argument("--out", type=str, default="", help="output md path; default report/<idx>_<device>_check_report.md")
    ap.add_argument("--use_llm", action="store_true", help="fill LLM placeholder blocks in the SAME run")
    ap.add_argument("--save_placeholders_copy", action="store_true", help="also save a *_placeholders.md copy before LLM filling")
    ap.add_argument("--debug_section3", action="store_true", help="print debug info for section 3 criterion matching")
    ap.add_argument("--base_url", type=str, default="https://api.your-provider.com/v1")
    ap.add_argument("--model", type=str, default="deepseek-v3.2")
    ap.add_argument("--api-key", type=str, default="", help="or OPENAI_API_KEY / LLM_API_KEY (required when --use_llm)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--jobs", "-j", type=int, default=1, help="parallel device workers in full-batch mode")
    args = ap.parse_args()
    model_root = Path(args.model_dir).resolve()
    _ensure_shared_pkg_on_path(model_root)

    # Batch mode: when device_index is omitted, generate reports for all per_device indices.
    if args.device_index is None:
        if _compact_str(args.out):
            raise SystemExit("When generating all devices, please do not pass --out (single output path).")
        per = model_root / "per_device"
        rep_files = sorted(per.glob("*.json"), key=lambda p: p.name)
        if not rep_files:
            raise SystemExit(f"no per_device *.json found under: {per}")
        cmds: List[List[str]] = []
        for rp in rep_files:
            idx_s = rp.name.split("_", 1)[0]
            try:
                idx = int(idx_s)
            except Exception:
                continue
            cmd: List[str] = [sys.executable, str(Path(__file__).resolve()), "--device_index", str(idx)]
            if _compact_str(args.ts):
                cmd.extend(["--ts", args.ts])
            if _compact_str(args.start):
                cmd.extend(["--start", args.start])
            if _compact_str(args.end):
                cmd.extend(["--end", args.end])
            if args.prefer_no_trigger:
                cmd.append("--prefer_no_trigger")
            if args.use_llm:
                cmd.append("--use_llm")
            if args.save_placeholders_copy:
                cmd.append("--save_placeholders_copy")
            if args.debug_section3:
                cmd.append("--debug_section3")
            cmd.extend(["--base_url", str(args.base_url), "--model", str(args.model)])
            cmd.extend(["--temperature", str(args.temperature), "--max_new_tokens", str(args.max_new_tokens)])
            cmds.append(cmd)

        child_env = os.environ.copy()
        api_key = _compact_str(getattr(args, "api_key", ""))
        if api_key:
            child_env["OPENAI_API_KEY"] = api_key
            child_env["LLM_API_KEY"] = api_key

        jobs = max(1, int(args.jobs))
        ok = failed = 0
        if jobs <= 1:
            for cmd in cmds:
                print(f"[BATCH] generating device_index={cmd[cmd.index('--device_index')+1]}")
                r = subprocess.run(cmd, env=child_env)
                if r.returncode == 0:
                    ok += 1
                else:
                    failed += 1
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            print(f"[PARALLEL] check devices={len(cmds)} jobs={jobs}", flush=True)

            def _run(cmd: List[str]) -> int:
                idx = cmd[cmd.index("--device_index") + 1]
                print(f"[PARALLEL] start device_index={int(idx):04d}", flush=True)
                rc = int(subprocess.run(cmd, env=child_env).returncode)
                print(f"[PARALLEL] done  device_index={int(idx):04d} rc={rc}", flush=True)
                return rc

            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futs = [pool.submit(_run, c) for c in cmds]
                for fut in as_completed(futs):
                    if fut.result() == 0:
                        ok += 1
                    else:
                        failed += 1
        print(f"[BATCH] done. success={ok}, failed={failed}")
        return 0 if failed == 0 else 1

    report_path, csv_path = _pick_per_device_paths(model_root, args.device_index)
    rep = _read_json(report_path)
    device_name = _compact_str(rep.get("device_name"))
    applicable_rules = _pick_applicable_rules(rep)
    df = pd.read_csv(csv_path)
    raw_csv_cols = list(df.columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    cli_ts = _compact_str(args.ts)
    cli_start = _compact_str(args.start)
    cli_end = _compact_str(args.end)
    # Default: one report per inspection_days entry (start=inspection day, end=start+24h).
    if not cli_ts and not cli_start and not cli_end:
        windows = _inspection_day_windows(rep, df)
        if not windows:
            raise SystemExit(
                "per_device JSON has no usable inspection windows: prefer 'inspection_days', "
                "else fallback 'checks[].timestamp'. Pass --ts or --end/--start explicitly, "
                "or re-export inspection_days from pipeline_generate."
            )
        ok = 0
        failed = 0
        prefix = report_path.name.split("_", 1)[0]
        for i, (s_ts, e_ts) in enumerate(windows, start=1):
            tag = f"E{i:02d}"
            cmd: List[str] = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--device_index",
                str(int(args.device_index)),
                "--start",
                s_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "--end",
                e_ts.strftime("%Y-%m-%d %H:%M:%S"),
            ]
            if _compact_str(args.out):
                out_base = Path(args.out)
                out_i = out_base.with_name(f"{out_base.stem}_{tag}{out_base.suffix}")
            else:
                out_i = model_root / "check_report" / f"{prefix}_{_safe_device_fs(device_name)}_{tag}_check_report.md"
            cmd.extend(["--out", str(out_i)])
            if args.prefer_no_trigger:
                cmd.append("--prefer_no_trigger")
            if args.use_llm:
                cmd.append("--use_llm")
            if args.save_placeholders_copy:
                cmd.append("--save_placeholders_copy")
            if args.debug_section3:
                cmd.append("--debug_section3")
            cmd.extend(["--base_url", str(args.base_url), "--model", str(args.model)])
            cmd.extend(["--temperature", str(args.temperature), "--max_new_tokens", str(args.max_new_tokens)])
            print(
                f"[INSPECTION_DAYS] generating {tag} window={s_ts.strftime('%Y-%m-%d %H:%M:%S')} ~ {e_ts.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            child_env = os.environ.copy()
            api_key = _compact_str(getattr(args, "api_key", ""))
            if api_key:
                child_env["OPENAI_API_KEY"] = api_key
                child_env["LLM_API_KEY"] = api_key
            r = subprocess.run(cmd, env=child_env)
            if r.returncode == 0:
                ok += 1
            else:
                failed += 1
        print(f"[INSPECTION_DAYS] done. success={ok}, failed={failed}")
        return 0 if failed == 0 else 1

    # Only evaluate rules that are point-only by stage2 expression
    stage2_rules = _load_stage2_rules(model_root)
    point_only = [rid for rid in applicable_rules if rid in stage2_rules and _is_point_only_rule(stage2_rules[rid])]
    params_by_rule = _load_params_by_rule(model_root)

    if cli_ts:
        end_ts = pd.Timestamp(args.ts)
        start_ts = end_ts - pd.Timedelta(hours=23)
    elif cli_end:
        end_ts = pd.Timestamp(args.end)
        if cli_start:
            start_ts = pd.Timestamp(args.start)
        else:
            start_ts = end_ts - pd.Timedelta(hours=23)
    else:
        raise SystemExit("--start requires --end, or use --ts. Omit all three to use per_device JSON inspection_days.")
    if end_ts < start_ts:
        raise SystemExit("--end must be >= --start")
    # --ts -> [ts-23h, ts]. --end with optional --start -> explicit or [end-23h, end].

    if end_ts not in df.index:
        raise SystemExit(f"window end timestamp not found in CSV index: {end_ts}")

    # Print some near-threshold hints at end_ts to help you confirm it is a near-miss day
    try:
        db2cn_dbg, _cn2db_dbg, db2unit_dbg = _load_metric_catalog(model_root)
        hints_dbg = _near_threshold_hints(
            df=df,
            ts=end_ts,
            applicable_point_rules=point_only,
            stage2_rules=stage2_rules,
            params_by_rule=params_by_rule,
            db2cn=db2cn_dbg,
            db2unit=db2unit_dbg,
            ratio=0.20,
        )
        hints_dbg = sorted(hints_dbg, key=lambda x: float(x.get("distance_to_threshold", 1e18)))[:5]
        print(f"[CHECK_24H] window: {start_ts.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
        for h in hints_dbg:
            print(
                f"  - {h.get('metric')}={h.get('observed_value')} {h.get('unit','')}  "
                f"thr={h.get('threshold')}  dist={h.get('distance_to_threshold')}  ({h.get('comparison')}, rule={h.get('rule_id')})"
            )
    except Exception:
        pass

    hits = _eval_point_rules_first_hit_in_window(model_root, df, start_ts, end_ts, point_only)
    db2cn, _cn2db, db2unit = _load_metric_catalog(model_root)
    md = _render_check_report_markdown(
        model_root,
        device_name,
        df,
        raw_csv_cols,
        start_ts,
        end_ts,
        point_only,
        applicable_rules,
        hits,
        db2cn,
        db2unit,
        rep=rep,
    )
    if args.debug_section3:
        # Re-render with debug prints (no change to output content).
        md = _render_check_report_markdown(
            model_root,
            device_name,
            df,
            raw_csv_cols,
            start_ts,
            end_ts,
            point_only,
            applicable_rules,
            hits,
            db2cn,
            db2unit,
            rep=rep,
            debug_section3=True,
        )

    # Post-clean: remove any accidental leaked intermediate metric artifacts and empty-value table rows.
    md = md.replace("|| rel_change_capacitance |         |                    |", "")
    cleaned_lines: List[str] = []
    empty_row_re = re.compile(r"^\|\s*[^|]+\s*\|\s*[^|]*\s*\|\s*\|\s*$")
    for line in md.splitlines():
        if empty_row_re.match(line):
            continue
        cleaned_lines.append(line)
    md = "\n".join(cleaned_lines) + ("\n" if md.endswith("\n") else "")

    out_path: Path
    if _compact_str(args.out):
        out_path = Path(args.out)
    else:
        prefix = report_path.name.split("_", 1)[0]
        out_path = model_root / "check_report" / f"{prefix}_{_safe_device_fs(device_name)}_check_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --use_llm: never overwrite an already-filled report.
    if args.use_llm and _md_already_llm_filled(out_path):
        print(f"[SKIP] already LLM-filled: {out_path}")
        return 0

    # Only save placeholders when explicitly requested.
    if args.save_placeholders_copy:
        placeholders_path = out_path.with_name(out_path.stem + "_placeholders.md")
        placeholders_path.write_text(md, encoding="utf-8")
    else:
        out_path.write_text(md, encoding="utf-8")

    if args.use_llm:
        api_key = (
            _compact_str(getattr(args, "api_key", ""))
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("LLM_API_KEY", "")
        )
        if not api_key:
            raise SystemExit("--use_llm requires --api-key (or OPENAI_API_KEY / LLM_API_KEY)")
        facts = _build_check_facts(
            model_root=model_root,
            report_path=report_path,
            csv_path=csv_path,
            start_ts=start_ts,
            end_ts=end_ts,
            device_name=device_name,
            hits=hits,
        )
        client = OpenAICompatClient(api_key=api_key, base_url=args.base_url, model=args.model)
        md2 = md
        block_names = ["section_2_safety", "section_3_extra_checks", "section_6_summary"]
        blocks_current: Dict[str, str] = {}
        for b in block_names:
            _before, inner, _after = _extract_block(md2, b)
            blocks_current[b] = inner.strip()
        max_tokens = int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else None
        try:
            filled_map = _llm_fill_blocks_json(
                client=client,
                facts=facts,
                blocks=blocks_current,
                temperature=float(args.temperature),
                max_new_tokens=max_tokens,
            )
        except Exception as e:
            print(f"[WARN] LLM fill failed for {out_path.name}: {e}")
            print(f"[OK] wrote placeholders (LLM failed): {out_path}")
            return 1
        if "section_3_extra_checks" in filled_map:
            # Continue numbering after the program-generated rows in section 3.
            # (Rows without criterion are filtered out in section 3 render logic.)
            check_fields = _pick_section3_check_fields(df, raw_csv_cols)
            stage2_rules_s3 = _load_stage2_rules(model_root)
            db2cn_s3, _cn2db_s3, _db2unit_s3 = _load_metric_catalog(model_root)
            kept_count = 0
            for col in check_fields:
                toks = _field_match_tokens(col, db2cn_s3)
                has_criterion = False
                for rid in applicable_rules:
                    r = stage2_rules_s3.get(rid) or {}
                    desc = _compact_str(r.get("judgment_description"))
                    expr = _compact_str(r.get("rule_expression"))
                    if not desc or not expr:
                        continue
                    for tok in toks:
                        t = _compact_str(tok)
                        if not t:
                            continue
                        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t):
                            if re.search(r"\b" + re.escape(t) + r"\b", expr):
                                has_criterion = True
                                break
                        else:
                            if t in expr:
                                has_criterion = True
                                break
                    if has_criterion:
                        break
                if has_criterion:
                    kept_count += 1
            filled_map["section_3_extra_checks"] = _renumber_md_table_rows(
                filled_map["section_3_extra_checks"], start=kept_count + 1
            )
        if "section_6_summary" in filled_map:
            txt = filled_map["section_6_summary"]
            txt = _remove_rule_ids_in_text(txt)
            txt = _enforce_inspection_tone(txt)
            filled_map["section_6_summary"] = txt
        for b in block_names:
            md2 = _replace_block(md2, b, filled_map[b])
        md2 = _strip_llm_markers(md2)
        out_path.write_text(md2, encoding="utf-8")
    else:
        if args.save_placeholders_copy:
            out_path.write_text(md, encoding="utf-8")

    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

