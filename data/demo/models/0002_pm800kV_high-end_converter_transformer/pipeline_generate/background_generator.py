from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_ALARM_VALUE_HINTS = (
    "fault",
    "alarm",
    "trip",
    "offline",
    "lockout",
    "interrupt",
    "overlimit",
    "severe",
    "warning",
)
_PREFERRED_NORMAL_HINTS = (
    "running",
    "stopped",
    "ready",
    "online",
    "good",
    "closed",
    "engaged",
)


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _to_float(x: Any) -> Optional[float]:
    s = _compact_str(x)
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metric_catalog(model_dir: Path) -> List[Dict[str, Any]]:
    data = _load_json(model_dir / "metric_catalog.json")
    return data if isinstance(data, list) else []


def _load_stage2_rules(model_dir: Path) -> List[Dict[str, Any]]:
    data = _load_json(model_dir / "rule_metric_stage2.json")
    if not isinstance(data, dict):
        return []
    rules = data.get("anomaly_judgment_rules")
    return rules if isinstance(rules, list) else []


def _should_export_field(field: str) -> bool:
    f = _compact_str(field)
    if not f:
        return False
    # Explicit catalog exports that collide with "intermediate name" heuristics.
    _ALWAYS_EXPORT = {"rel_change_tan", "pd_level"}
    if f in _ALWAYS_EXPORT:
        return True
    bad_prefix = (
        "rel_change_",
        "change_rate",
        "pd_change_rate",
        "duration_",
        "avg_",
        "std_",
        "slope_",
        "count_",
    )
    if any(f.startswith(p) for p in bad_prefix):
        return False
    if any(tok in f for tok in ("_1h", "_3h", "_24h", "_7d", "_30d", "_90d", "_180d", "_365d")):
        return False
    return True


def _is_placeholder_range(nr: Dict[str, Any]) -> bool:
    """Detect annotate_normal_ranges placeholder / malformed numeric range."""
    if not isinstance(nr, dict) or not nr:
        return True
    note = _compact_str(nr.get("note"))
    if "not_set" in note or "placeholder" in note.lower():
        return True
    mean = _to_float(nr.get("mean"))
    std = _to_float(nr.get("std"))
    mn = _to_float(nr.get("min"))
    mx = _to_float(nr.get("max"))
    if mean is None or mx is None:
        return True
    if std is not None and std < 0:
        return True
    if mn is not None and mx is not None and mn > mx:
        return True
    if abs(mean) < 1e-12 and abs(float(mx) - 1.0) < 1e-9 and (mn is None or abs(float(mn)) < 1e-12):
        return True
    return False


def _sanitize_catalog_bounds(
    mean: Optional[float],
    std: Optional[float],
    cmin: Optional[float],
    cmax: Optional[float],
) -> Tuple[Optional[float], Optional[float], List[str]]:
    """Drop absurd catalog floors/ceilings; prefer mean±3σ when min is pathological."""
    notes: List[str] = []
    lo, hi = cmin, cmax
    if mean is None:
        return lo, hi, notes
    sigma = std if (std is not None and std > 0) else (abs(mean) * 0.05 + 1e-6)
    soft_lo = mean - 3.0 * sigma
    soft_hi = mean + 3.0 * sigma
    # Pathological absolute floors (e.g. absolute zero on a warm-device metric)
    if lo is not None and mean - lo > 20.0 * sigma and lo < soft_lo:
        notes.append(f"ignore absurd catalog.min={lo:g}; use mean-3sigma")
        lo = soft_lo
    if hi is not None and hi - mean > 20.0 * sigma and hi > soft_hi:
        notes.append(f"ignore wide catalog.max={hi:g}; use mean+3sigma")
        hi = soft_hi
    if lo is not None and hi is not None and hi <= lo:
        notes.append("invalid catalog.min/max; use mean +/- 3sigma")
        lo, hi = soft_lo, soft_hi
    # Non-negative physical quantities: never background-fill below 0 when mean>0
    if mean is not None and mean > 0 and lo is not None and lo < 0:
        notes.append(f"catalog.min={lo:g}<0 and mean>0; clamp to 0")
        lo = 0.0
    return lo, hi, notes


def _field_aliases(item: Dict[str, Any]) -> List[str]:
    """Include metric_name aliases so stage2 expressions bind to the field."""
    out: List[str] = []
    for key in ("database_field", "variable_name", "metric_name"):
        v = _compact_str(item.get(key))
        if v and v not in out:
            out.append(v)
    return out


def _threshold_map(rule: Dict[str, Any]) -> Dict[str, float]:
    params = rule.get("threshold_params")
    out: Dict[str, float] = {}
    if not isinstance(params, list):
        return out
    for p in params:
        if not isinstance(p, dict):
            continue
        name = _compact_str(p.get("param_name"))
        val = _to_float(p.get("param_value"))
        if name and val is not None:
            out[name] = val
    return out


def _subst_thresholds(expr: str, thr: Dict[str, float]) -> str:
    # Longer names first to avoid partial replacements.
    out = expr
    for name in sorted(thr.keys(), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(name)}\b", str(thr[name]), out)
    return out


def _norm_rel_pct(pct: float) -> float:
    """Some stage2 files store 10.0 meaning 10% rather than ratio 0.10."""
    if pct > 1.0:
        return pct / 100.0
    return pct


def _rule_mentions_field(
    rule: Dict[str, Any],
    *,
    aliases: Sequence[str],
    chinese_name: str,
) -> bool:
    inputs = rule.get("input_metrics")
    if chinese_name and isinstance(inputs, list) and chinese_name in inputs:
        return True
    expr = _compact_str(rule.get("rule_expression"))
    if not expr:
        return False
    expr_l = expr.lower()
    for a in aliases:
        if not a:
            continue
        if a.lower() in expr_l or a in expr:
            return True
    return False


def _looks_alarm_token(v: str) -> bool:
    return any(h in v for h in _ALARM_VALUE_HINTS)


def _extract_alarm_enums(expr: str, aliases: Sequence[str]) -> Set[str]:

    alarms: Set[str] = set()
    for alias in aliases:
        pat = re.compile(
            rf"{re.escape(alias)}\s+in\s*\{{\s*([^}}]+)\s*\}}"
        )
        for m in pat.finditer(expr):
            body = m.group(1)
            for tok in re.findall(r"['\"]([^'\"]+)['\"]", body):
                t = tok.strip()
                if t:
                    alarms.add(t)
        eq = re.compile(rf"{re.escape(alias)}\s*=\s*['\"]([^'\"]+)['\"]")
        for m in eq.finditer(expr):
            t = m.group(1).strip()
            if t and _looks_alarm_token(t):
                alarms.add(t)
    return alarms


def _is_divisor_context(expr: str, start: int) -> bool:
    """True if alias at `start` is immediately after `/` or `*` (ratio denom/product)."""
    j = start - 1
    while j >= 0 and expr[j].isspace():
        j -= 1
    return j >= 0 and expr[j] in {"/", "*"}


def _extract_numeric_bounds(
    expr: str, aliases: Sequence[str]
) -> Tuple[List[float], List[float], List[float]]:
    lowers: List[float] = []
    uppers: List[float] = []
    rel_spans: List[float] = []
    for alias in aliases:
        a = re.escape(alias)
        wr = re.compile(
            rf"\({{0,2}}\s*max_\w+\(\s*{a}\s*\)\s*-\s*min_\w+\(\s*{a}\s*\)\s*\)\s*/\s*min_\w+\(\s*{a}\s*\)\s*\)?\s*>\s*([-+0-9.eE]+)"
        )
        for m in wr.finditer(expr):
            pct = _norm_rel_pct(float(m.group(1)))
            if pct > 0:
                rel_spans.append(pct)
        rel = re.compile(
            rf"abs\s*\(\s*\(\s*{a}\s*-\s*([-+0-9.eE]+)\s*\)\s*/\s*\1\s*\)\s*>\s*([-+0-9.eE]+)"
        )
        for m in rel.finditer(expr):
            ref = float(m.group(1))
            pct = _norm_rel_pct(float(m.group(2)))
            if ref != 0 and pct >= 0:
                lowers.append(ref * (1.0 - pct))
                uppers.append(ref * (1.0 + pct))
        rel2 = re.compile(
            rf"abs\s*\(\s*\(\s*{a}\s*-\s*([-+0-9.eE]+)\s*\)\s*/\s*([-+0-9.eE]+)\s*\)\s*>\s*([-+0-9.eE]+)"
        )
        for m in rel2.finditer(expr):
            ref = float(m.group(1))
            den = float(m.group(2))
            pct = _norm_rel_pct(float(m.group(3)))
            if den != 0 and pct >= 0:
                span = abs(den) * pct
                lowers.append(ref - span)
                uppers.append(ref + span)

        for m in re.finditer(rf"{a}\s*<=\s*([-+0-9.eE]+)", expr):
            if not _is_divisor_context(expr, m.start()):
                lowers.append(float(m.group(1)))
        for m in re.finditer(rf"{a}\s*<(?![=])\s*([-+0-9.eE]+)", expr):
            if not _is_divisor_context(expr, m.start()):
                lowers.append(float(m.group(1)))
        for m in re.finditer(rf"{a}\s*>=\s*([-+0-9.eE]+)", expr):
            if not _is_divisor_context(expr, m.start()):
                uppers.append(float(m.group(1)))
        for m in re.finditer(rf"{a}\s*>(?![=])\s*([-+0-9.eE]+)", expr):
            if not _is_divisor_context(expr, m.start()):
                uppers.append(float(m.group(1)))
    return lowers, uppers, rel_spans


def _collect_field_constraints(
    rules: Sequence[Dict[str, Any]],
    *,
    aliases: Sequence[str],
    chinese_name: str,
) -> Tuple[Optional[float], Optional[float], Set[str], Optional[float]]:
    """Aggregate stage2 safe bounds + alarm enums + max relative range for one field."""
    all_lowers: List[float] = []
    all_uppers: List[float] = []
    all_rel: List[float] = []
    alarms: Set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if not _rule_mentions_field(rule, aliases=aliases, chinese_name=chinese_name):
            continue
        thr = _threshold_map(rule)
        expr = _subst_thresholds(_compact_str(rule.get("rule_expression")), thr)
        if not expr:
            continue
        # Also try with Chinese metric name substituted by nothing — enums use CN aliases.
        alarms |= _extract_alarm_enums(expr, aliases)
        lo, hi, rel = _extract_numeric_bounds(expr, aliases)
        all_lowers.extend(lo)
        all_uppers.extend(hi)
        all_rel.extend(rel)
    lo_b = max(all_lowers) if all_lowers else None
    hi_b = min(all_uppers) if all_uppers else None
    rel_b = min(all_rel) if all_rel else None
    return lo_b, hi_b, alarms, rel_b


def _safe_inner_band(
    lo: Optional[float],
    hi: Optional[float],
    *,
    margin_frac: float = 0.10,
) -> Tuple[Optional[float], Optional[float]]:
    if lo is None or hi is None:
        return lo, hi
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return lo, hi
    span = hi - lo
    m = span * margin_frac
    if 2 * m >= span:
        m = span * 0.05
    return lo + m, hi - m


def _resolve_numeric_plan(
    item: Dict[str, Any],
    rules: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    aliases = _field_aliases(item)
    chinese = _compact_str(item.get("metric_name"))
    nr = item.get("normal_range") if isinstance(item.get("normal_range"), dict) else {}
    st_lo, st_hi, _alarms, rel_span = _collect_field_constraints(
        rules, aliases=aliases, chinese_name=chinese
    )

    cat_ok = isinstance(nr, dict) and not _is_placeholder_range(nr)
    mean = _to_float(nr.get("mean")) if cat_ok else None
    std = _to_float(nr.get("std")) if cat_ok else None
    cmin = _to_float(nr.get("min")) if cat_ok else None
    cmax = _to_float(nr.get("max")) if cat_ok else None
    cmin, cmax, san_notes = _sanitize_catalog_bounds(mean, std, cmin, cmax)

    lo: Optional[float] = cmin
    hi: Optional[float] = cmax
    strategy_parts: List[str] = []
    strategy_parts.extend(san_notes)

    if cat_ok and mean is not None:
        strategy_parts.append("catalog normal_range")
    else:
        strategy_parts.append("catalog range placeholder/missing -> stage2 fallback")

    # Stage2 x>=A AND x<=B preconditions may be parsed as inverted bounds (lo>hi); drop to avoid flat background.
    if st_lo is not None and st_hi is not None and st_lo >= st_hi:
        strategy_parts.append(f"ignore inverted stage2 bounds [{st_lo:g},{st_hi:g}]")
        st_lo, st_hi = None, None

    if st_lo is not None or st_hi is not None:
        cand_lo = lo
        cand_hi = hi
        if st_lo is not None:
            cand_lo = st_lo if cand_lo is None else max(cand_lo, st_lo)
            strategy_parts.append(f"stage2 lower={st_lo:g}")
        if st_hi is not None:
            cand_hi = st_hi if cand_hi is None else min(cand_hi, st_hi)
            strategy_parts.append(f"stage2 upper={st_hi:g}")
        if (
            cand_lo is not None
            and cand_hi is not None
            and cand_hi <= cand_lo
            and st_lo is not None
            and st_hi is not None
            and st_hi > st_lo
        ):
            lo, hi = st_lo, st_hi
            strategy_parts.append("catalog intersect stage2 empty -> use stage2 safe band")
        else:
            lo, hi = cand_lo, cand_hi

    if rel_span is not None and rel_span > 0 and mean is not None and np.isfinite(mean) and mean != 0:
        safe_p = float(rel_span) * 0.50
        mid = float(mean)
        if lo is not None and hi is not None and hi > lo:
            mid = float(np.clip(mid, lo, hi))
        half = abs(mid) * safe_p / 2.0
        span_lo, span_hi = mid - half, mid + half
        if lo is None:
            lo = span_lo
        else:
            lo = max(lo, span_lo)
        if hi is None:
            hi = span_hi
        else:
            hi = min(hi, span_hi)
        strategy_parts.append(f"stage2 relative span<={rel_span:g} -> tighten band +/-{safe_p/2:g}")

    if (not cat_ok or mean is None) and st_lo is not None and st_hi is not None and st_hi > st_lo:
        mean = 0.5 * (st_lo + st_hi)
        std = (st_hi - st_lo) / 8.0
        lo, hi = st_lo, st_hi
    elif (not cat_ok or mean is None) and st_lo is not None and st_hi is None:
        mean = st_lo * 1.1 if st_lo != 0 else 1.0
        std = abs(mean) * 0.05 + 1e-6
        lo = st_lo
    elif (not cat_ok or mean is None) and st_hi is not None and st_lo is None:
        mean = st_hi * 0.7 if st_hi != 0 else 0.0
        std = abs(st_hi) * 0.05 + 1e-6
        hi = st_hi
    elif mean is None:
        mean = 1.0
        std = 0.05
        strategy_parts.append("final fallback base=1")

    if std is None or std <= 0:
        std = abs(mean) * 0.05 + 1e-6

    if lo is not None and hi is not None and hi > lo:
        if mean < lo or mean > hi:
            mean = 0.5 * (lo + hi)
            strategy_parts.append("mean out of band -> pull to safe-band midpoint")
            std = min(std, (hi - lo) / 8.0)

    inner_lo, inner_hi = _safe_inner_band(lo, hi, margin_frac=0.10)
    if inner_lo is not None and inner_hi is not None and inner_hi > inner_lo:
        max_sigma = (inner_hi - inner_lo) / 6.0
        if max_sigma > 0:
            std = min(std, max_sigma)
        mean = float(np.clip(mean, inner_lo, inner_hi))

    return {
        "base": float(mean),
        "sigma": float(max(std, 1e-12)),
        "lo": float(inner_lo) if inner_lo is not None else None,
        "hi": float(inner_hi) if inner_hi is not None else None,
        "strategy": "; ".join(strategy_parts),
    }


def _pick_safe_discrete(
    values: Sequence[Any],
    alarm_enums: Set[str],
) -> str:
    opts = [_compact_str(v) for v in values if _compact_str(v)]
    if not opts:
        return "running"

    def is_alarm_like(v: str) -> bool:
        if v in alarm_enums:
            return True
        return any(h in v for h in _ALARM_VALUE_HINTS)

    safe = [v for v in opts if not is_alarm_like(v)]
    pool = safe if safe else opts

    for hint in _PREFERRED_NORMAL_HINTS:
        for v in pool:
            if hint in v:
                return v
    return pool[0]


def _resolve_discrete_plan(
    item: Dict[str, Any],
    rules: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    aliases = _field_aliases(item)
    chinese = _compact_str(item.get("metric_name"))
    _lo, _hi, alarms, _rel = _collect_field_constraints(
        rules, aliases=aliases, chinese_name=chinese
    )
    raw = item.get("discrete_values")
    values: List[Any] = list(raw) if isinstance(raw, list) else []
    choice = _pick_safe_discrete(values, alarms)
    return {
        "value": choice,
        "strategy": (
            f"discrete_values non-alarm pick [{choice}]"
            + (f" (avoid stage2 enum {sorted(alarms)})" if alarms else " (no stage2 enum; prefer normal semantics)")
        ),
    }


def _baseline_numeric(
    n: int,
    *,
    base: float,
    sigma: float,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> np.ndarray:
    x = np.random.normal(loc=base, scale=sigma, size=n).astype(float)
    if lo is not None:
        x = np.maximum(x, lo)
    if hi is not None:
        x = np.minimum(x, hi)
    return x


def _is_discrete_item(item: Dict[str, Any]) -> bool:
    dtype = _compact_str(item.get("data_type")).lower()
    mtype = _compact_str(item.get("metric_type"))
    disc = item.get("discrete_values")
    if dtype in {"string", "str", "bool", "boolean"}:
        return True
    if mtype in {"status"}:
        return True
    if isinstance(disc, list) and len(disc) > 0 and dtype not in {"float", "int", "double", "number"}:
        return True
    return False


def generate_background(df: pd.DataFrame, *, model_dir: Path) -> None:
    model_dir = Path(model_dir)
    idx = df.index
    n = len(idx)
    catalog = _load_metric_catalog(model_dir)
    rules = _load_stage2_rules(model_dir)

    strategies: Dict[str, str] = {}

    for item in catalog:
        if not isinstance(item, dict):
            continue
        col = _compact_str(item.get("database_field"))
        if not col or not _should_export_field(col):
            if col and not _should_export_field(col):
                strategies[col] = "skip: non-persisted intermediate/window naming"
            continue

        if _is_discrete_item(item):
            plan = _resolve_discrete_plan(item, rules)
            df[col] = plan["value"]
            strategies[col] = plan["strategy"]
            continue

        plan = _resolve_numeric_plan(item, rules)
        df[col] = _baseline_numeric(
            n,
            base=plan["base"],
            sigma=plan["sigma"],
            lo=plan["lo"],
            hi=plan["hi"],
        )
        strategies[col] = (
            f"{plan['strategy']}; base={plan['base']:g}, sigma={plan['sigma']:g}, "
            f"clip=[{plan['lo']},{plan['hi']}]"
        )

    # Keep DGA total hydrocarbon consistent with C1–C2 components (K4 diff limb).
    gas_parts = ("oil_dga_ch4", "oil_dga_c2h6", "oil_dga_c2h4", "oil_dga_c2h2")
    if all(c in df.columns for c in gas_parts) and "oil_dga_total_hydrocarbon" in df.columns:
        total = None
        for c in gas_parts:
            s = pd.to_numeric(df[c], errors="coerce")
            total = s if total is None else (total + s)
        df["oil_dga_total_hydrocarbon"] = total
        strategies["oil_dga_total_hydrocarbon"] = (
            strategies.get("oil_dga_total_hydrocarbon", "")
            + "; post: sync=CH4+C2H6+C2H4+C2H2"
        ).strip("; ")

    # Couple body/conservator pressure so |Δ| stays inside K26/K27 quiet band.
    if "transformer_body_pressure" in df.columns and "conservator_pressure" in df.columns:
        body = pd.to_numeric(df["transformer_body_pressure"], errors="coerce")
        noise = np.random.normal(loc=0.0, scale=0.8, size=n)
        noise = np.clip(noise, -2.5, 2.5)
        df["conservator_pressure"] = body + noise
        strategies["conservator_pressure"] = (
            strategies.get("conservator_pressure", "")
            + "; post: couple_to_body±2.5"
        ).strip("; ")

    # Couple winding hotspot to cooling water — keep rise away from K21/K22 alarms.
    if "winding_hotspot_temp" in df.columns and "cooling_water_temp" in df.columns:
        cw = pd.to_numeric(df["cooling_water_temp"], errors="coerce")
        # quiet rise ~15–45°C (alarm 65 / trip 75 / high 70)
        rise = np.random.normal(loc=28.0, scale=6.0, size=n)
        rise = np.clip(rise, 12.0, 50.0)
        df["winding_hotspot_temp"] = cw.to_numpy() + rise
        strategies["winding_hotspot_temp"] = (
            strategies.get("winding_hotspot_temp", "")
            + "; post: couple_to_cooling_water rise∈[12,50]"
        ).strip("; ")

    # Keep CO2/CO ratio away from K28/K29 trip edges on pure background.
    if "oil_dga_co2" in df.columns and "oil_dga_co" in df.columns:
        co2 = pd.to_numeric(df["oil_dga_co2"], errors="coerce")
        co = pd.to_numeric(df["oil_dga_co"], errors="coerce").clip(lower=1e-3)
        ratio = co2 / co
        # stage2: T_ratio_high≈7, T_ratio_low≈3 — keep bg in ~[3.5, 6.2]
        hi, lo = 6.2, 3.5
        too_hi = ratio > hi
        too_lo = ratio < lo
        if too_hi.any():
            df.loc[too_hi, "oil_dga_co"] = (co2[too_hi] / hi).to_numpy()
        if too_lo.any():
            df.loc[too_lo, "oil_dga_co"] = (co2[too_lo] / lo).to_numpy()
        strategies["oil_dga_co"] = (
            strategies.get("oil_dga_co", "") + "; post: clamp_co2_co_ratio[3.5,6.2]"
        ).strip("; ")

    try:
        df.attrs["background_strategies"] = strategies
    except Exception:
        pass


__all__ = ["generate_background"]
