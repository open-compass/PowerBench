from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from powerbench.generation.telemetry.strict_rule_engine import eval_rule_strict, load_metric_catalog


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _ensure_col(df: pd.DataFrame, col: str, dtype: str = "float") -> None:
    if col not in df.columns:
        df[col] = np.nan
    if dtype == "object":
        if df[col].dtype != "object":
            df[col] = df[col].astype("object")


def _set_last_hours(df: pd.DataFrame, ts: pd.Timestamp, col: str, hours: int, value: Any, dtype: str = "float") -> None:
    _ensure_col(df, col, dtype=dtype)
    idx_full = pd.date_range(end=ts, periods=hours, freq="h")
    # Clamp to existing df.index to avoid KeyError when idx contains timestamps not in dataset range
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        idx = idx_full[(idx_full >= df.index.min()) & (idx_full <= df.index.max())]
    else:
        idx = idx_full
    if len(idx) == 0:
        return
    df.loc[idx, col] = value


def _set_trend(df: pd.DataFrame, ts: pd.Timestamp, col: str, days: int, slope_per_day: float, base: float = 1.0) -> None:
    _ensure_col(df, col, dtype="float")
    n_full = int(days) * 24
    idx_full = pd.date_range(end=ts, periods=n_full, freq="h")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        idx = idx_full[(idx_full >= df.index.min()) & (idx_full <= df.index.max())]
    else:
        idx = idx_full
    if len(idx) <= 1:
        return
    # effective days based on available span
    eff_days = max(1.0 / 24.0, (len(idx) - 1) / 24.0)
    ramp = np.linspace(0.0, float(slope_per_day) * eff_days, len(idx))
    df.loc[idx, col] = float(base) + ramp


def _set_high_std(df: pd.DataFrame, ts: pd.Timestamp, col: str, hours: int, std_target: float, base: float = 0.0) -> None:
    _ensure_col(df, col, dtype="float")
    idx_full = pd.date_range(end=ts, periods=hours, freq="h")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        idx = idx_full[(idx_full >= df.index.min()) & (idx_full <= df.index.max())]
    else:
        idx = idx_full
    if len(idx) <= 1:
        return
    # alternating +/- to raise std
    amp = std_target * 2.0
    vals = np.array([float(base) + (amp if i % 2 == 0 else -amp) for i in range(len(idx))], dtype=float)
    df.loc[idx, col] = vals


@dataclass
class InjectReport:
    rule_id: str
    injected: bool
    notes: str


def inject_to_trigger(
    folder: Path,
    df: pd.DataFrame,
    ts: pd.Timestamp,
    rule: Dict[str, Any],
    *,
    prefer_random_or: bool = False,
    seed: Optional[int] = None,
) -> InjectReport:
    """
    Strict approach:
    - Pattern-match common primitives in rule expression and apply minimal edits.
    - After each edit, re-evaluate with strict engine; stop when triggered.
    - If prefer_random_or=True, when multiple OR-able patterns are available, try them in randomized order.
    """
    rid = _compact_str(rule.get("rule_id"))
    expr = _compact_str(rule.get("rule_expression"))
    cn2var = load_metric_catalog(folder)
    params = {str(p.get("param_name")): _to_float(p.get("param_value")) for p in (rule.get("threshold_params") or []) if isinstance(p, dict)}
    rng = random.Random(seed)

    # Helper to map a metric mentioned in expression to a dataframe column
    def col_for_metric(metric_cn: str) -> str:
        if metric_cn in cn2var:
            return cn2var[metric_cn]
        return metric_cn

    # 0) quick check
    ok, _ = eval_rule_strict(folder, df, ts, rule)
    if ok:
        return InjectReport(rid, True, "already triggered")

    # 1) consec_3h(X > T) / consec_3h(X < T)
    m: List[Tuple[str, str, str]] = re.findall(
        r"consec_3h\(\s*([A-Za-z_][A-Za-z0-9_]*|[^<>=()]+?)\s*(>|<)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        expr,
    )
    if prefer_random_or and len(m) > 1:
        rng.shuffle(m)
    for lhs, op, thr_name in m:
        thr = params.get(thr_name)
        if thr is None or math.isnan(thr):
            continue
        lhs = lhs.strip()
        col = col_for_metric(lhs) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs) else lhs
        if op == ">":
            _set_last_hours(df, ts, col, 3, thr + 1.0, dtype="float")
        else:
            _set_last_hours(df, ts, col, 3, thr - 1.0, dtype="float")
        ok, _ = eval_rule_strict(folder, df, ts, rule)
        if ok:
            return InjectReport(rid, True, f"inject consec_3h {lhs} {op} {thr_name}")

    # 2) duration_1h(col in {...}) >= 3600
    if "duration_1h" in expr:
        mm: List[Tuple[str, str, str]] = re.findall(
            r"duration_1h\(\s*([A-Za-z_][A-Za-z0-9_]*|[^()]+?)\s+in\s+\{([^}]+)\}\s*\)\s*>=\s*([0-9]+)",
            expr,
        )
        if prefer_random_or and len(mm) > 1:
            rng.shuffle(mm)
        for lhs, set_body, _sec in mm:
            lhs = lhs.strip()
            col = col_for_metric(lhs) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs) else lhs
            choices = [x.strip().strip("'\"") for x in set_body.split(",") if x.strip()]
            state = rng.choice(choices) if prefer_random_or and choices else (choices[0] if choices else "fault")
            _set_last_hours(df, ts, col, 1, state, dtype="object")
            ok, _ = eval_rule_strict(folder, df, ts, rule)
            if ok:
                return InjectReport(rid, True, "inject duration_1h state")

    # 3) slope_*d(X) < S or > S
    sm: List[Tuple[str, str, str, str]] = re.findall(
        r"slope_(\d+)d\(\s*([A-Za-z_][A-Za-z0-9_]*|[^()]+?)\s*\)\s*(<|>)\s*([A-Za-z_][A-Za-z0-9_]*)",
        expr,
    )
    if prefer_random_or and len(sm) > 1:
        rng.shuffle(sm)
    for days_s, lhs, op, thr_name in sm:
        days = int(days_s)
        thr = params.get(thr_name)
        if thr is None or math.isnan(thr):
            continue
        lhs = lhs.strip()
        col = col_for_metric(lhs) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs) else lhs
        slope = thr - abs(thr) * 0.2 - 0.001 if op == "<" else thr + abs(thr) * 0.2 + 0.001
        _set_trend(df, ts, col, days=days, slope_per_day=slope, base=1.0)
        ok, _ = eval_rule_strict(folder, df, ts, rule)
        if ok:
            return InjectReport(rid, True, f"inject slope_{days}d {op} {thr_name}")

    # 4) std_24h/std_90d(X) > T
    stm: List[Tuple[str, str, str, str]] = re.findall(
        r"std_(\d+)(h|d)\(\s*([A-Za-z_][A-Za-z0-9_]*|[^()]+?)\s*\)\s*>\s*([A-Za-z_][A-Za-z0-9_]*)",
        expr,
    )
    if prefer_random_or and len(stm) > 1:
        rng.shuffle(stm)
    for num_s, unit, lhs, thr_name in stm:
        thr = params.get(thr_name)
        if thr is None or math.isnan(thr):
            continue
        lhs = lhs.strip()
        col = col_for_metric(lhs) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs) else lhs
        hours = int(num_s) if unit == "h" else int(num_s) * 24
        _set_high_std(df, ts, col, hours=hours, std_target=thr * 1.5, base=0.0)
        ok, _ = eval_rule_strict(folder, df, ts, rule)
        if ok:
            return InjectReport(rid, True, f"inject std_{num_s}{unit} > {thr_name}")

    # 5) simple point comparisons: X > T or X < T
    pm: List[Tuple[str, str, str]] = re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*|[^<>=()]+?)\s*(>|<)\s*([A-Za-z_][A-Za-z0-9_]*)",
        expr,
    )
    if prefer_random_or and len(pm) > 1:
        rng.shuffle(pm)
    for lhs, op, thr_name in pm:
        if thr_name not in params:
            continue
        thr = params[thr_name]
        lhs = lhs.strip()
        if lhs.startswith("consec_") or lhs.startswith("slope_") or lhs.startswith("std_") or lhs.startswith("avg_") or lhs.startswith("count_"):
            continue
        col = col_for_metric(lhs) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs) else lhs
        _ensure_col(df, col, dtype="float")
        df.loc[ts, col] = thr + 1.0 if op == ">" else thr - 1.0
        ok, _ = eval_rule_strict(folder, df, ts, rule)
        if ok:
            return InjectReport(rid, True, f"inject point {lhs} {op} {thr_name}")

    return InjectReport(rid, False, "no known pattern matched; needs custom injector or extended patterns")

