from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from requests import HTTPError
import os

# Allow direct script execution.
_THIS_DIR = Path(__file__).resolve().parent


def _ensure_shared_pkg_on_path(model_root: Path | None = None) -> None:
    """Ensure repo root is on sys.path so `powerbench` is importable."""
    from powerbench.paths import repo_root
    root = repo_root(Path(model_root) if model_root else Path(__file__))
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _safe_device_fs(name: str) -> str:
    try:
        import importlib.util as _ilu

        helper = Path(__file__).resolve().parent / "_local_imports.py"
        spec = _ilu.spec_from_file_location("m0002_fault_safe_fs", str(helper))
        if spec is None or spec.loader is None:
            raise ImportError("no helper")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return str(mod.safe_fs_stem(name))
    except Exception:
        return str(name or "device").replace("/", "_").replace("\\", "_")


_ensure_shared_pkg_on_path()

# NOTE(0007): 0002 imported pipeline_generate.strict_rule_engine here but never used it.
# Model dir name may contain "(L)" and is not a valid package — do not dotted-import it.


def _md_already_llm_filled(path: Path) -> bool:
    """True if md exists and no longer has LLM placeholders / markers."""
    if not path.is_file():
        return False
    try:
        t = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return ("TBD_LLM_FILL" not in t) and ("LLM_BEGIN" not in t)


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


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


def _split_semicolon(s: str) -> List[str]:
    s2 = _compact_str(s)
    if not s2:
        return []
    return [x.strip() for x in s2.split(";") if x.strip()]


def _build_checks_compatible(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compatible loader for old/new report schemas.
    - old schema: use report["checks"]
    - new schema: synthesize checks from report["scenarios"] where type == "fault"
    """
    checks = report.get("checks")
    if isinstance(checks, list) and checks:
        return [x for x in checks if isinstance(x, dict)]

    out: List[Dict[str, Any]] = []
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list):
        return out
    for s in scenarios:
        if not isinstance(s, dict):
            continue
        if _compact_str(s.get("type")).lower() != "fault":
            continue
        if bool(s.get("skipped")):
            continue
        rid = _compact_str(s.get("rule_id"))
        if not rid:
            continue
        ts = _compact_str(s.get("window_end")) or _compact_str(s.get("window_start"))
        out.append(
            {
                "timestamp": ts,
                "target_rule": rid,
                "scenario_window_start": _compact_str(s.get("window_start")),
                "scenario_window_end": _compact_str(s.get("window_end")),
                "note_cn": _compact_str(s.get("note_cn")),
                "strict_triggered_rules": [rid],
                "context_windows": {},
            }
        )
    return out


def _build_two_month_spans(report: Dict[str, Any], df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    tr = report.get("time_range") or {}
    s_raw = _compact_str(tr.get("start")) if isinstance(tr, dict) else ""
    e_raw = _compact_str(tr.get("end")) if isinstance(tr, dict) else ""
    if s_raw and e_raw:
        start = pd.Timestamp(s_raw)
        end = pd.Timestamp(e_raw)
    else:
        start = pd.Timestamp(df.index.min())
        end = pd.Timestamp(df.index.max())
    if end < start:
        return []

    spans: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = pd.Timestamp(start).normalize()
    hard_end = pd.Timestamp(end)
    while cur <= hard_end:
        nxt = cur + pd.DateOffset(months=2)
        span_end_day = (nxt - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=23)
        span_end = min(span_end_day, hard_end)
        spans.append((cur, span_end))
        cur = nxt.normalize()
    return spans


def _snap_timestamp_to_csv_index(ts: pd.Timestamp, idx: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
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


# Cap §3.1 HTML series: keep reports LLM-friendly (gold 0002 ~5–12KB).
_MAX_SERIES_POINTS = 48
_MAX_SERIES_LOOKBACK_HOURS = 48


@dataclass
class FaultEpisode:
    """One fault report episode (= one episode_day when available)."""

    discover_ts: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    rule_id: str
    note_cn: str
    input_database_fields: List[str]
    changed_cols: List[str]
    strategy: str
    scenario: Dict[str, Any]
    matched_scenarios: List[Dict[str, Any]]


def _iter_fault_scenarios(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in report.get("scenarios") or []:
        if not isinstance(s, dict):
            continue
        if _compact_str(s.get("type")).lower() != "fault":
            continue
        if bool(s.get("skipped")):
            continue
        if ( _compact_str(s.get("note_cn")) or "" ).startswith("[validation failed]"):
            continue
        if not _compact_str(s.get("rule_id")):
            continue
        out.append(s)
    return out


def _scenario_window_hours(s: Dict[str, Any]) -> float:
    try:
        ws = pd.Timestamp(_compact_str(s.get("window_start")))
        we = pd.Timestamp(_compact_str(s.get("window_end")))
        if pd.isna(ws) or pd.isna(we):
            return float("inf")
        return max(0.0, (we - ws).total_seconds() / 3600.0)
    except Exception:
        return float("inf")


def _pick_primary_scenario(matched: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prefer shortest fault window, then rule_id — keep long 90d windows out of the header."""
    return sorted(
        matched,
        key=lambda s: (_scenario_window_hours(s), _compact_str(s.get("rule_id"))),
    )[0]


def _list_fault_episodes(report: Dict[str, Any], df: pd.DataFrame) -> List[FaultEpisode]:
    idx = pd.DatetimeIndex(df.index).sort_values()
    scenarios = _iter_fault_scenarios(report)
    episodes: List[FaultEpisode] = []

    def _ep_from_scenarios(
        matched: List[Dict[str, Any]],
        *,
        strategy: str,
        discover_ts: Optional[pd.Timestamp] = None,
    ) -> Optional[FaultEpisode]:
        if not matched:
            return None
        primary = _pick_primary_scenario(matched)
        we_raw = _compact_str(primary.get("window_end")) or _compact_str(primary.get("window_start"))
        if not we_raw:
            return None
        we = _snap_timestamp_to_csv_index(pd.Timestamp(we_raw), idx)
        if we is None:
            print(f"[FAULT_EPISODE] skip scenario: cannot map window_end={we_raw!r}")
            return None
        ws_raw = _compact_str(primary.get("window_start")) or we_raw
        ws = _snap_timestamp_to_csv_index(pd.Timestamp(ws_raw), idx) or we
        if ws > we:
            ws = we
        disc = discover_ts if discover_ts is not None else we
        # Union fields across all matched scenarios for section-1 metadata.
        idf: List[str] = []
        chg: List[str] = []
        rule_ids: List[str] = []
        notes: List[str] = []
        for s in matched:
            rid = _compact_str(s.get("rule_id"))
            if rid and rid not in rule_ids:
                rule_ids.append(rid)
            note = _compact_str(s.get("note_cn"))
            if note and note not in notes:
                notes.append(note)
            for x in s.get("input_database_fields") or []:
                cx = _compact_str(x)
                if cx and cx not in idf:
                    idf.append(cx)
            for x in s.get("changed_cols") or []:
                cx = _compact_str(x)
                if cx and cx not in chg:
                    chg.append(cx)
        return FaultEpisode(
            discover_ts=disc,
            window_start=ws,
            window_end=we,
            rule_id=",".join(rule_ids) if rule_ids else _compact_str(primary.get("rule_id")),
            note_cn="; ".join(notes) if notes else _compact_str(primary.get("note_cn")),
            input_database_fields=idf,
            changed_cols=chg,
            strategy=strategy,
            scenario=primary,
            matched_scenarios=list(matched),
        )

    ep_days = report.get("episode_days")
    if isinstance(ep_days, list) and ep_days:
        strategy = "episode_days+one_report_per_day"
        print(f"[FAULT_EPISODE] strategy={strategy} (episode_days n={len(ep_days)})")
        by_day: Dict[str, List[Dict[str, Any]]] = {}
        for s in scenarios:
            we_raw = _compact_str(s.get("window_end")) or _compact_str(s.get("window_start"))
            if not we_raw:
                continue
            we = _snap_timestamp_to_csv_index(pd.Timestamp(we_raw), idx)
            if we is None:
                continue
            by_day.setdefault(we.normalize().strftime("%Y-%m-%d"), []).append(s)
        for item in ep_days:
            day_s = _compact_str(item)
            if not day_s:
                continue
            day = pd.Timestamp(day_s)
            if pd.isna(day):
                continue
            key = day.normalize().strftime("%Y-%m-%d")
            matched = by_day.get(key) or []
            if not matched:
                print(f"[FAULT_EPISODE] episode_day={key} has no matching fault scenario; skip")
                continue
            discover = _snap_timestamp_to_csv_index(day, idx)
            ep = _ep_from_scenarios(matched, strategy=strategy, discover_ts=discover)
            if ep is not None:
                episodes.append(ep)
                print(
                    f"[FAULT_EPISODE] day={key} matched_rules="
                    f"{[ _compact_str(s.get('rule_id')) for s in matched ]} "
                    f"primary={_compact_str(ep.scenario.get('rule_id'))}"
                )
    else:
        strategy = "scenarios.window_end"
        print(
            f"[FAULT_EPISODE] strategy={strategy} "
            f"(no episode_days; using {len(scenarios)} non-skipped fault scenarios)"
        )
        for s in scenarios:
            ep = _ep_from_scenarios([s], strategy=strategy)
            if ep is not None:
                episodes.append(ep)

    # Stable order by discover time, then primary rule_id
    episodes.sort(key=lambda e: (e.discover_ts, _compact_str(e.scenario.get("rule_id"))))
    print(f"[FAULT_EPISODE] built n={len(episodes)}")
    return episodes


def _load_metric_maps(folder: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    items = _read_json(folder / "metric_catalog.json")
    cn2var: Dict[str, str] = {}
    cn2db: Dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cn = _compact_str(it.get("metric_name"))
        var = _compact_str(it.get("variable_name"))
        db = _compact_str(it.get("database_field"))
        if cn and var:
            cn2var[cn] = var
        if cn and db:
            cn2db[cn] = db
    return cn2var, cn2db


def _load_db2cn(folder: Path) -> Dict[str, str]:
    items = _read_json(folder / "metric_catalog.json")
    out: Dict[str, str] = {}
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                cn = _compact_str(it.get("metric_name"))
                db = _compact_str(it.get("database_field"))
                if cn and db:
                    out[db] = cn
    return out


def _load_db2unit(folder: Path) -> Dict[str, str]:
    items = _read_json(folder / "metric_catalog.json")
    out: Dict[str, str] = {}
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                db = _compact_str(it.get("database_field"))
                unit = _compact_str(it.get("unit"))
                if db and unit:
                    out[db] = unit
    return out


def _downsample_points(pts: List[Dict[str, Any]], max_points: int) -> List[Dict[str, Any]]:
    n = len(pts)
    if n <= max_points or max_points <= 2:
        return pts if n <= max_points else [pts[0], pts[-1]]
    # inclusive indices covering first..last
    idxs = sorted({int(round(i * (n - 1) / (max_points - 1))) for i in range(max_points)})
    return [pts[i] for i in idxs]


def _csv_window_series(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_points: int = _MAX_SERIES_POINTS,
    max_lookback_hours: int = _MAX_SERIES_LOOKBACK_HOURS,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if len(df.index) == 0:
        return out
    end_ts = pd.Timestamp(end)
    start_ts = pd.Timestamp(start)
    # Prefer the last N hours ending at window_end (gold-style: short local context).
    capped_start = max(start_ts, end_ts - pd.Timedelta(hours=int(max_lookback_hours)))
    if capped_start > end_ts:
        capped_start = end_ts
    sub = df.loc[capped_start:end_ts]
    for col in cols:
        c = _compact_str(col)
        if not c or c not in df.columns:
            continue
        pts: List[Dict[str, Any]] = []
        for ts, row in sub.iterrows():
            v = row.get(c)
            if v is None:
                continue
            try:
                if not isinstance(v, str) and pd.isna(v):
                    continue
            except Exception:
                pass
            pts.append({"timestamp": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"), "value": v})
        if pts:
            out[c] = _downsample_points(pts, int(max_points))
    return out


def _safe_md(s: str) -> str:
    return s.replace("|", "\\|")


def _md_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [len(x) for x in rows[0]]
    for r in rows[1:]:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    out = []
    header = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(rows[0])) + " |"
    sep = "| " + " | ".join("---".ljust(widths[i]) for i in range(len(widths))) + " |"
    out.append(header)
    out.append(sep)
    for r in rows[1:]:
        out.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(widths))) + " |")
    return "\n".join(out) + "\n"


@dataclass
class OpenAICompatClient:
    api_key: str
    base_url: str
    model: str
    timeout_s: int = 1800

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_new_tokens: Optional[int] = None) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
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


def _scenario_fields(
    s: Dict[str, Any],
    *,
    rules: Dict[str, Any],
    cn2db: Dict[str, str],
) -> Tuple[pd.Timestamp, pd.Timestamp, str, List[str], List[str], str]:
    rid = _compact_str(s.get("rule_id"))
    note = _compact_str(s.get("note_cn"))
    we = pd.Timestamp(_compact_str(s.get("window_end")) or _compact_str(s.get("window_start")))
    ws = pd.Timestamp(_compact_str(s.get("window_start")) or we)
    if pd.isna(ws):
        ws = we
    if pd.isna(we):
        we = ws
    if ws > we:
        ws = we
    field_cols: List[str] = []
    for x in s.get("input_database_fields") or []:
        cx = _compact_str(x)
        if cx and cx not in field_cols:
            field_cols.append(cx)
    if not field_cols:
        r0 = rules.get(rid) or {}
        for cn in r0.get("input_metrics") or []:
            db = cn2db.get(_compact_str(cn))
            if db and db not in field_cols:
                field_cols.append(db)
    chg: List[str] = []
    for x in s.get("changed_cols") or []:
        cx = _compact_str(x)
        if cx and cx not in chg:
            chg.append(cx)
        if cx and cx not in field_cols:
            field_cols.append(cx)
    return ws, we, rid, field_cols, chg, note


def _build_facts(
    folder: Path,
    report_path: Path,
    csv_path: Path,
    stage1_path: Path,
    stage2_path: Path,
    *,
    episode: FaultEpisode,
) -> Dict[str, Any]:
    report = _read_json(report_path)
    stage1 = _read_json(stage1_path)
    stage2 = _read_json(stage2_path)
    cn2var, cn2db = _load_metric_maps(folder)
    db2cn = _load_db2cn(folder)
    db2unit = _load_db2unit(folder)

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV missing timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    rules = {r.get("rule_id"): r for r in (stage2.get("anomaly_judgment_rules") or []) if isinstance(r, dict)}
    discover_ts = episode.discover_ts
    matched = list(episode.matched_scenarios) if episode.matched_scenarios else [episode.scenario]

    checks_out: List[Dict[str, Any]] = []
    for s in matched:
        ws, we, rid, field_cols, _chg, note = _scenario_fields(s, rules=rules, cn2db=cn2db)
        # Snap to CSV if possible
        we_snap = _snap_timestamp_to_csv_index(we, pd.DatetimeIndex(df.index)) or we
        ws_snap = _snap_timestamp_to_csv_index(ws, pd.DatetimeIndex(df.index)) or ws
        ts = we_snap
        window_series = _csv_window_series(df, cols=field_cols, start=ws_snap, end=we_snap)

        enrich_rules = []
        r = rules.get(rid, {})
        input_cn = [str(x) for x in (r.get("input_metrics") or [])]
        snap = []
        use_dbs = list(field_cols)
        if not use_dbs:
            for cn in input_cn:
                db = cn2db.get(cn)
                if db:
                    use_dbs.append(db)
        for db in use_dbs:
            if db not in df.columns:
                continue
            try:
                val = df.at[ts, db]
            except Exception:
                val = None
            cn = db2cn.get(db) or db
            snap.append(
                {
                    "metric_name": cn,
                    "database_field": db,
                    "unit": db2unit.get(db, ""),
                    "value": val,
                }
            )

        params = []
        for p in (r.get("threshold_params") or []):
            if isinstance(p, dict):
                params.append(
                    {
                        "param_name": _compact_str(p.get("param_name")),
                        "meaning": _compact_str(p.get("meaning")),
                        "param_value": _compact_str(p.get("param_value")),
                    }
                )

        enrich_rules.append(
            {
                "rule_id": rid,
                "object_name": _compact_str(r.get("object_name")),
                "judgment_description": _compact_str(r.get("judgment_description")),
                "rule_expression": _compact_str(r.get("rule_expression")),
                "input_metrics": input_cn,
                "threshold_params": params,
                "input_metric_observations_at_ts": snap,
            }
        )

        ctx: Dict[str, Any] = {}
        if rid and window_series:
            ctx[rid] = {"episode_window": {"series": window_series}}

        checks_out.append(
            {
                "timestamp": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
                "target_rule": rid,
                "scenario_window_start": pd.Timestamp(ws_snap).strftime("%Y-%m-%d %H:%M:%S"),
                "scenario_window_end": pd.Timestamp(we_snap).strftime("%Y-%m-%d %H:%M:%S"),
                "note_cn": note,
                "strict_triggered_rules": [rid] if rid else [],
                "triggered_rule_details": enrich_rules,
                "context_windows": ctx,
            }
        )

    # Sort checks by timestamp then rule for stable §3
    checks_out.sort(key=lambda c: (_compact_str(c.get("timestamp")), _compact_str(c.get("target_rule"))))
    discover_time = discover_ts.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "device_name": _compact_str(report.get("device_name")),
        "device_fields_override": report.get("device_fields_override") or {},
        "data_sources": {
            "csv": str(csv_path),
            "device_report_json": str(report_path),
            "rule_metric_stage1_json": str(stage1_path),
            "rule_metric_stage2_json": str(stage2_path),
        },
        "episode": {
            "strategy": episode.strategy,
            "rule_id": episode.rule_id,
            "window_start": episode.window_start.strftime("%Y-%m-%d %H:%M:%S"),
            "window_end": episode.window_end.strftime("%Y-%m-%d %H:%M:%S"),
            "note_cn": episode.note_cn,
            "input_database_fields": list(episode.input_database_fields),
            "changed_cols": list(episode.changed_cols),
        },
        "applicable_rules": report.get("eligible_rules_with_injector")
        or report.get("applicable_rules")
        or [],
        "chosen_rules_for_injection": report.get("chosen_rules_for_injection") or [],
        "primary_scenario": episode.scenario,
        "checks": checks_out,
        "stage1_instance_fields": stage1.get("instance_locating_fields") or [],
        "stage1_aux_fields": stage1.get("auxiliary_classification_fields") or [],
        "discover_time": discover_time,
        "db2cn": db2cn,
        "db2unit": db2unit,
    }


def _render_report_markdown(facts: Dict[str, Any]) -> str:
    device_name = _compact_str(facts.get("device_name"))
    checks = facts.get("checks") or []
    db2cn = facts.get("db2cn") or {}
    db2unit = facts.get("db2unit") or {}
    episode = facts.get("episode") if isinstance(facts.get("episode"), dict) else {}

    # Discover time: already computed from records' last timestamp in _build_facts
    discover_ts_s = _compact_str(facts.get("discover_time"))
    rule_id = _compact_str(episode.get("rule_id"))
    win_s = _compact_str(episode.get("window_start"))
    win_e = _compact_str(episode.get("window_end"))
    note_cn = _compact_str(episode.get("note_cn"))
    id_fields = episode.get("input_database_fields") if isinstance(episode.get("input_database_fields"), list) else []
    chg_fields = episode.get("changed_cols") if isinstance(episode.get("changed_cols"), list) else []

    md: List[str] = []
    md.append("# Power Equipment Fault Record Report\n\n")

    md.append("## 1. Basic information\n\n")
    # Fixed by program; LLM must not change
    md.append(f"- **Device**: {device_name}\n")
    md.append(f"- **Discovery time**: {discover_ts_s if discover_ts_s else '(no record)'}\n")
    if rule_id:
        md.append(f"- **Related rule**: {rule_id}\n")
    if win_s or win_e:
        md.append(f"- **Fault window**: {win_s or '—'} ~ {win_e or '—'}\n")
    if id_fields:
        md.append(
            "- **Input metric fields**: "
            + ", ".join(f"{db2cn.get(c, c)}({c})" for c in id_fields)
            + "\n"
        )
    if chg_fields:
        md.append(
            "- **Injected field changes**: "
            + ", ".join(f"{db2cn.get(c, c)}({c})" for c in chg_fields)
            + "\n"
        )
    if note_cn:
        md.append(f"- **Scenario note**: {note_cn}\n")
    md.append(f"- **Discovery-time strategy**: {_compact_str(episode.get('strategy')) or '—'}\n")
    # Remaining fields filled by LLM
    md.append("<!-- LLM_BEGIN:section_1 -->\n")
    md.append(f"- **Report ID**: TBD_LLM_FILL\n")
    md.append(f"- **Station/line**: TBD_LLM_FILL\n")
    md.append(f"- **Device tag**: {device_name}\n")
    md.append(f"- **Discoverer**: TBD_LLM_FILL\n")
    md.append(f"- **Crew**: TBD_LLM_FILL\n")
    md.append(f"- **Weather**: TBD_LLM_FILL\n")
    md.append(f"- **Load level**: TBD_LLM_FILL\n")
    md.append("<!-- LLM_END:section_1 -->\n\n")

    md.append("## 2. Fault event overview\n\n")
    md.append("<!-- LLM_BEGIN:section_2 -->\n")
    md.append("- **Event trigger scenario**: TBD_LLM_FILL (do not change section 3 table timestamps/values; no fault modes outside this episode).\n")
    md.append("- **Impact scope**:\n")
    md.append("  - **Outage scope**: TBD_LLM_FILL\n")
    md.append("  - **Affected user types**: TBD_LLM_FILL\n")
    md.append("  - **System risk**: TBD_LLM_FILL\n")
    md.append("<!-- LLM_END:section_2 -->\n\n")

    md.append("## 3. Alarm and protection action log\n\n")
    md.append("| timestamp              | Signal/alarm content | source_system       | action_result | metric | value |\n")
    md.append("| :------------------ | :------------ | :------------- | :------- | :--- | :--- |\n")
    # Convert each strict triggered rule at its check timestamp into rows with measured value + threshold if available.
    for c in checks:
        ts = _compact_str(c.get("timestamp"))
        for rd in (c.get("triggered_rule_details") or []):
            obj = _compact_str(rd.get("object_name"))
            params = rd.get("threshold_params") or []
            p_hint = ""
            if params and isinstance(params, list) and isinstance(params[0], dict):
                pv = _compact_str(params[0].get("param_value"))
                ph = _compact_str(params[0].get("meaning"))
                # unit from first observation if available
                unit_hint = ""
                obs0 = (rd.get("input_metric_observations_at_ts") or [None])[0]
                if isinstance(obs0, dict):
                    unit_hint = _compact_str(obs0.get("unit")) or _compact_str(
                        db2unit.get(_compact_str(obs0.get("database_field")), "")
                    )
                if pv and ph:
                    p_hint = f" ({ph}={pv}{unit_hint})"
                elif pv:
                    p_hint = f" (threshold ref {pv}{unit_hint})"
            obs = rd.get("input_metric_observations_at_ts") or []
            obs_cn = ""
            obs_val = ""
            if obs and isinstance(obs, list) and isinstance(obs[0], dict):
                obs_cn = _compact_str(obs[0].get("metric_name"))
                obs_val = _compact_str(obs[0].get("value"))
            content = f"{device_name}{obj} anomaly alarm{p_hint}"
            md.append(f"| {ts} | {content} | SCADA/online monitoring | alarm | {obs_cn} | {obs_val} |\n")
    if not checks:
        md.append("| TBD | TBD | TBD | TBD | TBD | TBD |\n")
    md.append("\n")

    # Extra detail derived from JSON/CSV (deterministic, no LLM): rule expression/params/observations
    md.append("### 3.1 Rule evaluation details\n\n")
    if checks:
        for c in checks:
            ts = _compact_str(c.get("timestamp"))
            md.append(f"#### Evaluation time: {ts}\n\n")
            for rd in (c.get("triggered_rule_details") or []):
                ctx_windows = c.get("context_windows") or {}
                rid_here = _compact_str(rd.get("rule_id"))
                md.append(f"- **object_name**: {_compact_str(rd.get('object_name'))}\n")
                md.append(f"- **judgment_description**: {_compact_str(rd.get('judgment_description'))}\n\n")

                p_rows = [["meaning", "threshold_param_value"]]
                for p in (rd.get("threshold_params") or []):
                    if isinstance(p, dict):
                        p_rows.append(
                            [
                                _safe_md(_compact_str(p.get("meaning"))),
                                _safe_md(_compact_str(p.get("param_value"))),
                            ]
                        )
                if len(p_rows) > 1:
                    md.append(_md_table(p_rows))
                    md.append("\n")

                # ---- merge point + window series into ONE HTML table (rowspan/colspan friendly) ----
                rule_ctx = {}
                if rid_here and isinstance(ctx_windows, dict):
                    rule_ctx = ctx_windows.get(rid_here) or {}
                # Build rows.
                # - If rule has window series: ONLY render window series (no extra single-point rows)
                # - Else: render single-point observations
                rows_out: List[Tuple[str, str, str]] = []  # (metric_cn, timestamp, value)

                def _db_to_cn(name: str) -> str:
                    n = _compact_str(name)
                    if isinstance(db2cn, dict) and n in db2cn:
                        return _compact_str(db2cn.get(n))
                    return n

                has_window_series = False
                if isinstance(rule_ctx, dict) and rule_ctx:
                    # window series: windows are >=1h only by generation contract
                    for win, wobj in rule_ctx.items():
                        if not isinstance(wobj, dict):
                            continue
                        series = wobj.get("series") or {}
                        if not isinstance(series, dict) or not series:
                            continue
                        for col, pts in series.items():
                            if not isinstance(pts, list) or not pts:
                                continue
                            has_window_series = True
                            for p in pts:
                                if not isinstance(p, dict):
                                    continue
                                rows_out.append(
                                    (
                                        _db_to_cn(_compact_str(col)),
                                        _compact_str(p.get("timestamp")),
                                        _compact_str(p.get("value")),
                                    )
                                )

                if not has_window_series:
                    for s in (rd.get("input_metric_observations_at_ts") or []):
                        if isinstance(s, dict):
                            rows_out.append(
                                (
                                    _compact_str(s.get("metric_name")),
                                    _compact_str(c.get("timestamp")),
                                    _compact_str(s.get("value")),
                                )
                            )

                # render HTML table (supports rowspan/colspan)
                if rows_out:
                    md.append("\n")
                    md.append("<table>\n")
                    md.append("<thead><tr><th>metric</th><th>timestamp</th><th>value</th></tr></thead>\n")
                    md.append("<tbody>\n")
                    # group by metric
                    from collections import defaultdict
                    groups2: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
                    for met, t, v in rows_out:
                        groups2[met].append((t, v))
                    for met in sorted(groups2.keys()):
                        pts_sorted = sorted(groups2[met], key=lambda x: x[0])
                        rowspan = len(pts_sorted)
                        for i2, (t, v) in enumerate(pts_sorted):
                            md.append("<tr>")
                            if i2 == 0:
                                md.append(f"<td rowspan=\"{rowspan}\">{_safe_md(met)}</td>")
                            md.append(f"<td>{_safe_md(t)}</td>")
                            md.append(f"<td>{_safe_md(v)}</td>")
                            md.append("</tr>\n")
                    md.append("</tbody></table>\n")
            md.append("\n")
    else:
        md.append("None (checks not found).\n\n")

    md.append("## 4. Field inspection record\n\n")
    md.append("| check_item | check_method | Field observation/measurement | judgment | recorder |\n")
    md.append("| :----- | :------- | :-------------- | :--- | :----- |\n")
    md.append("<!-- LLM_BEGIN:section_4_table_rows -->\n")
    md.append("| (not covered by generated data) | TBD_LLM_FILL | TBD_LLM_FILL | TBD_LLM_FILL | TBD_LLM_FILL |\n")
    md.append("<!-- LLM_END:section_4_table_rows -->\n\n")

    md.append("## 5. Fault remediation process\n\n")
    md.append("| step_no | remediation_action | execution_time | risk_controls | execution_result |\n")
    md.append("| :------- | :------- | :------- | :----------- | :------- |\n")
    md.append("<!-- LLM_BEGIN:section_5_table_rows -->\n")
    md.append("| 1 | TBD_LLM_FILL | TBD_LLM_FILL | TBD_LLM_FILL | TBD_LLM_FILL |\n")
    md.append("| 2 | TBD_LLM_FILL | TBD_LLM_FILL | TBD_LLM_FILL | TBD_LLM_FILL |\n")
    md.append("<!-- LLM_END:section_5_table_rows -->\n\n")

    md.append("## 6. Root cause analysis\n\n")
    md.append("<!-- LLM_BEGIN:section_6 -->\n")
    md.append("- **Direct cause**: TBD_LLM_FILL (cite section 3 facts only; no invented values/times; no rule ids like K1/K13).\n")
    md.append("  - **Evidence chain**: TBD_LLM_FILL (cite section 3 alarm rows; use object/metric/phenomenon names, not rule ids).\n")
    md.append("- **Indirect cause**: TBD_LLM_FILL\n")
    md.append("- **Management factors**: TBD_LLM_FILL\n")
    md.append("<!-- LLM_END:section_6 -->\n\n")
    return "".join(md)


def _llm_polish(md_facts: str, facts: Dict[str, Any], client: OpenAICompatClient, temperature: float, max_new_tokens: Optional[int]) -> str:
    fact_json = json.dumps(facts, ensure_ascii=False, indent=2)
    prompt = (
        "You are a power O&M fault-report editor. Produce a human-like report from [Fact JSON] and [Fact Markdown].\n"
        "Hard constraints:\n"
        "1) Never change numbers, timestamps, rule ids, thresholds, device_name, or field names; wording/structure only.\n"
        "2) Every conclusion must be supported by the fact JSON.\n"
        "3) Output must be Markdown.\n"
        "4) Preserve the appendix fact-JSON code block verbatim.\n\n"
        "[Fact JSON]\n"
        f"{fact_json}\n\n"
        "[Fact Markdown]\n"
        f"{md_facts}\n"
    )
    return client.chat([{"role": "user", "content": prompt}], temperature=temperature, max_new_tokens=max_new_tokens)


def _extract_block(md: str, block_name: str) -> Tuple[str, str, str]:
    begin = f"<!-- LLM_BEGIN:{block_name} -->"
    end = f"<!-- LLM_END:{block_name} -->"
    s = md
    i = s.find(begin)
    j = s.find(end)
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"missing LLM block: {block_name}")
    before = s[: i + len(begin)]
    inner = s[i + len(begin) : j]
    after = s[j:]
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


def _llm_fill_block(
    *,
    client: OpenAICompatClient,
    facts: Dict[str, Any],
    block_name: str,
    block_inner_current: str,
    temperature: float,
    max_new_tokens: Optional[int],
) -> str:
    fact_json = json.dumps(facts, ensure_ascii=False, indent=2)
    sec6_extra = ""
    if block_name == "section_6":
        sec6_extra = (
            "5) Root-cause block: no rule ids; use object_name/judgment_description/metric names.\n"
        )
    prompt = (
        "Fill one placeholder block of a power-equipment fault report.\n"
        "Hard constraints:\n"
        "1) Output block content only; no BEGIN/END markers or extra headings.\n"
        "2) Do not invent values, timestamps, rule ids, thresholds, device or field names.\n"
        "3) Copy numbers/times verbatim from fact JSON when cited.\n"
        "4) Output English Markdown; table blocks must be valid pipe rows.\n"
        f"{sec6_extra}\n"
        f"[Block name] {block_name}\n\n"
        "[Current block content]\n"
        f"{block_inner_current}\n\n"
        "[Fact JSON]\n"
        f"{fact_json}\n"
    )
    return client.chat([{"role": "user", "content": prompt}], temperature=temperature, max_new_tokens=max_new_tokens)


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
        "Fill placeholder blocks for a power-equipment fault report.\n"
        "Output one JSON object: keys are block names, values are Markdown content (no BEGIN/END markers).\n"
        "\n"
        "Hard constraints:\n"
        "1) JSON only; no extra text.\n"
        "2) Only generate placeholder blocks; other sections are fixed.\n"
        "3) Do not change device_name; use fact JSON verbatim.\n"
        "4) Table row blocks must be valid Markdown rows with matching column count.\n"
        "5) You may add site, crew, weather, load, impact, remediation, root cause aligned with episode facts.\n"
        "6) Do not contradict timestamps/thresholds/observations from section 3/3.1.\n"
        "7) section_6 must not mention rule ids; use object/metric/phenomenon names.\n"
        "8) Narrative must align with fact JSON episode fields.\n"
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


def main() -> int:
    from powerbench.paths import demo_models_dir, DEFAULT_DEMO_TYPE
    # Use model root as the working folder for per_device and rule JSONs
    folder = demo_models_dir() / DEFAULT_DEMO_TYPE
    ap = argparse.ArgumentParser(description="Generate fault report markdown from per-device dataset + rules.")
    ap.add_argument(
        "--model_dir",
        type=str,
        default=str(folder),
        help="model directory under data/demo/models (contains per_device/, pipeline_generate/)",
    )
    ap.add_argument(
        "--device_index",
        type=int,
        default=None,
        help="per_device index (e.g., 0 -> 0000_*.csv/json). If omitted, auto-pick the smallest index found.",
    )
    ap.add_argument(
        "--report_json",
        type=str,
        default="",
        help="optional explicit per_device *.json path; if provided, overrides device_index auto-pick",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="",
        help="optional explicit per_device *.csv path; if provided, overrides device_index auto-pick",
    )
    ap.add_argument("--out", type=str, default="", help="output md path; default: per_device/<idx>_<device>_fault_report.md")
    ap.add_argument(
        "--use_llm",
        action="store_true",
        help="fill LLM placeholder blocks using facts JSON in the SAME run; non-placeholder parts remain unchanged",
    )
    ap.add_argument(
        "--save_placeholders_copy",
        action="store_true",
        help="also save a *_placeholders.md copy before LLM filling (audit/debug).",
    )
    ap.add_argument("--base_url",type=str, default="https://api.your-provider.com/v1",help="LLM base URL (passed to run_pipeline base_url)")
    ap.add_argument("--model", type=str, default="deepseek-v3.2",)
    ap.add_argument("--api-key",type=str,default="",help="or OPENAI_API_KEY / LLM_API_KEY",)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--jobs", "-j", type=int, default=1, help="parallel device workers in full-batch mode")
    args = ap.parse_args()
    folder = Path(args.model_dir).resolve()
    _ensure_shared_pkg_on_path(folder)

    per = folder / "per_device"
    # Batch mode: when device_index/report_json/csv are all omitted, generate all devices.
    if args.device_index is None and not _compact_str(args.report_json) and not _compact_str(args.csv):
        if _compact_str(args.out):
            raise SystemExit("When generating all devices, please do not pass --out (single output path).")
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
            if args.use_llm:
                cmd.append("--use_llm")
            if args.save_placeholders_copy:
                cmd.append("--save_placeholders_copy")
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

            print(f"[PARALLEL] fault devices={len(cmds)} jobs={jobs}", flush=True)

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

    # Resolve inputs: explicit paths > device_index > auto-pick smallest.
    report_path: Optional[Path] = Path(args.report_json) if _compact_str(args.report_json) else None
    csv_path: Optional[Path] = Path(args.csv) if _compact_str(args.csv) else None
    if report_path and not report_path.exists():
        raise SystemExit(f"--report_json not found: {report_path}")
    if csv_path and not csv_path.exists():
        raise SystemExit(f"--csv not found: {csv_path}")

    if report_path is None or csv_path is None:
        if args.device_index is not None:
            idx = int(args.device_index)
            rep_files = sorted(per.glob(f"{idx:04d}_*.json"))
            csv_files = sorted(per.glob(f"{idx:04d}_*.csv"))
        else:
            rep_files = sorted(per.glob("*.json"))
            csv_files = sorted(per.glob("*.csv"))
            # pick smallest index by filename prefix 0000_
            rep_files = sorted(rep_files, key=lambda p: p.name)
            csv_files = sorted([p for p in csv_files if not p.name.endswith("_placeholders.md")], key=lambda p: p.name)

        if report_path is None:
            if not rep_files:
                raise SystemExit(f"no per_device *.json found under: {per}")
            report_path = rep_files[0]
        if csv_path is None:
            if not csv_files:
                raise SystemExit(f"no per_device *.csv found under: {per}")
            # try to match same prefix as report (0000_)
            prefix = report_path.name.split("_", 1)[0] + "_"
            matched = [p for p in csv_files if p.name.startswith(prefix)]
            csv_path = matched[0] if matched else csv_files[0]

    stage1_path = folder / "rule_metric_stage1.json"
    stage2_path = folder / "rule_metric_stage2.json"
    report_obj = _read_json(report_path)
    df_for_span = pd.read_csv(csv_path)
    if "timestamp" not in df_for_span.columns:
        raise SystemExit(f"CSV missing timestamp column: {csv_path}")
    df_for_span["timestamp"] = pd.to_datetime(df_for_span["timestamp"])
    df_for_span = df_for_span.set_index("timestamp").sort_index()
    episodes = _list_fault_episodes(report_obj, df_for_span)
    if not episodes:
        raise SystemExit(
            "No valid fault episodes: need episode_days matching fault scenarios, "
            "or non-skipped scenarios[type=fault].window_end."
        )

    client: Optional[OpenAICompatClient] = None
    max_tokens = int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else None
    if args.use_llm:
        api_key = (
            _compact_str(getattr(args, "api_key", ""))
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("LLM_API_KEY", "")
        )
        if not api_key:
            raise SystemExit("--use_llm requires --api-key (or OPENAI_API_KEY / LLM_API_KEY)")
        client = OpenAICompatClient(api_key=api_key, base_url=args.base_url, model=args.model)

    base_out: Optional[Path] = Path(args.out) if _compact_str(args.out) else None
    ok_count = 0
    for i, episode in enumerate(episodes, start=1):
        facts = _build_facts(
            folder,
            report_path,
            csv_path,
            stage1_path,
            stage2_path,
            episode=episode,
        )
        md_facts = _render_report_markdown(facts)
        prefix = report_path.name.split("_", 1)[0]
        tag = f"E{i:02d}_{episode.discover_ts.strftime('%Y%m%d%H%M%S')}"
        if base_out is not None:
            if len(episodes) == 1:
                out_path = base_out
            else:
                out_path = base_out.with_name(f"{base_out.stem}_{tag}{base_out.suffix}")
        else:
            out_path = folder / "fault_report" / f"{prefix}_{_safe_device_fs(facts['device_name'])}_{tag}_fault_report.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # --use_llm: never overwrite an already-filled report; skip LLM for it.
        if client is not None and _md_already_llm_filled(out_path):
            print(f"[SKIP] already LLM-filled: {out_path.name}")
            ok_count += 1
            continue

        if args.save_placeholders_copy:
            placeholders_path = out_path.with_name(out_path.stem + "_placeholders.md")
            placeholders_path.write_text(md_facts, encoding="utf-8")
        else:
            # Incomplete/missing only: write template before LLM (do not touch filled files).
            out_path.write_text(md_facts, encoding="utf-8")

        if client is not None:
            md2 = md_facts
            block_names = ["section_1", "section_2", "section_4_table_rows", "section_5_table_rows", "section_6"]
            blocks_current: Dict[str, str] = {}
            for b in block_names:
                _before, inner, _after = _extract_block(md2, b)
                blocks_current[b] = inner.strip()
            try:
                filled_map = _llm_fill_blocks_json(
                    client=client,
                    facts=facts,
                    blocks=blocks_current,
                    temperature=float(args.temperature),
                    max_new_tokens=max_tokens,
                )
                for b in block_names:
                    md2 = _replace_block(md2, b, filled_map[b])
                md2 = _strip_llm_markers(md2)
                out_path.write_text(md2, encoding="utf-8")
            except Exception as e:
                # Keep placeholders already on disk; continue remaining episodes.
                print(f"[WARN] LLM fill failed for {out_path.name}: {e}")
        elif args.save_placeholders_copy:
            out_path.write_text(md_facts, encoding="utf-8")

        ok_count += 1
        print(
            f"[OK] wrote ({i}/{len(episodes)}): {out_path} "
            f"[rule={episode.rule_id} discover={episode.discover_ts.strftime('%Y-%m-%d %H:%M:%S')} "
            f"strategy={episode.strategy}]"
        )

    print(f"[DONE] generated {ok_count} fault reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

