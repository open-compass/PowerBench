from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from powerbench.generation.telemetry.metric_functions import (
    abs_,
    avg,
    consec_nh,
    count_true,
    duration_seconds,
    last_n_hours,
    rolling_window,
    slope_per_day,
    std,
    safe_div,
)


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")



import importlib.util
import sys
import types
from functools import lru_cache
from typing import Set, Tuple

def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _model_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _pipeline_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_derived_metrics_impl():
    try:
        from . import derived_metrics_impl as dm  # type: ignore
        return dm
    except Exception:
        pass
    path = _pipeline_dir() / "derived_metrics_impl.py"
    mod_name = "m0002_pipeline_derived_metrics_impl"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load derived_metrics_impl from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    pkg = "m0002_pipeline"
    if pkg not in sys.modules:
        pkg_mod = types.ModuleType(pkg)
        pkg_mod.__path__ = [str(_pipeline_dir())]  # type: ignore[attr-defined]
        sys.modules[pkg] = pkg_mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_stage2_rules(folder: Path) -> Dict[str, Dict[str, Any]]:
    stage2 = json.loads((folder / "rule_metric_stage2.json").read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for r in stage2.get("anomaly_judgment_rules", []) or []:
        if isinstance(r, dict) and r.get("rule_id"):
            out[str(r["rule_id"]).strip()] = r
    return out


def load_params_by_rule(folder: Path) -> Dict[str, Dict[str, float]]:
    rules = load_stage2_rules(folder)
    out: Dict[str, Dict[str, float]] = {}
    for rid, r in rules.items():
        params: Dict[str, float] = {}
        for p in r.get("threshold_params", []) or []:
            if isinstance(p, dict) and p.get("param_name") is not None:
                params[str(p["param_name"]).strip()] = _to_float(p.get("param_value"))
        out[rid] = params
    return out


@dataclass
class RuleResult:
    rule_id: str
    triggered: bool
    details: Dict[str, Any]


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        # create missing as NaN
        df[col] = float("nan")
    return df[col]


def _win_series(df: pd.DataFrame, ts: pd.Timestamp, window: str, col: str) -> pd.Series:
    return rolling_window(pd.DataFrame({"x": _series(df, col)}), ts, window)["x"]


def _last_n(df: pd.DataFrame, ts: pd.Timestamp, n: int, col: str) -> pd.Series:
    return last_n_hours(pd.DataFrame({"x": _series(df, col)}), ts, n)["x"]


def _ensure_derived_col(df: pd.DataFrame, col: str) -> pd.Series:
    name = _compact_str(col)
    if not name:
        raise ValueError("derived column name is empty")
    dm = _load_derived_metrics_impl()
    existing = df[name] if name in df.columns else None
    try:
        s = dm.compute_one(df, derived_key=name, folder=_model_dir())
        if s is not None and pd.to_numeric(s, errors="coerce").notna().any():
            df[name] = s
            return df[name]
    except Exception:
        pass
    if existing is not None:
        return existing
    df[name] = float("nan")
    return df[name]


def ensure_all_derived(df: pd.DataFrame) -> Dict[str, Any]:
    """Materialise all derived keys; keep prior finite values if recompute is empty."""
    dm = _load_derived_metrics_impl()
    prior = {c: df[c].copy() for c in list(df.columns)}
    out = dm.ensure_derived_columns(df, folder=_model_dir(), write_both_names=False)
    for key, s in list(out.items()):
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().any():
            continue
        old = prior.get(key)
        if old is not None and pd.to_numeric(old, errors="coerce").notna().any():
            df[key] = old
            out[key] = old
    return out


def _last_n_derived(df: pd.DataFrame, ts: pd.Timestamp, n: int, derived_col: str) -> pd.Series:
    s = _ensure_derived_col(df, derived_col)
    return last_n_hours(pd.DataFrame({"x": s}), ts, n)["x"]


def _win_derived(df: pd.DataFrame, ts: pd.Timestamp, window: str, derived_col: str) -> pd.Series:
    s = _ensure_derived_col(df, derived_col)
    return rolling_window(pd.DataFrame({"x": s}), ts, window)["x"]


def _point_derived(df: pd.DataFrame, ts: pd.Timestamp, derived_col: str) -> float:
    s = _ensure_derived_col(df, derived_col)
    try:
        return float(s.loc[ts])
    except Exception:
        return float("nan")


def _rel_change_vs_window_initial(
    df: pd.DataFrame, ts: pd.Timestamp, col: str, *, window: str = "365d"
) -> float:
    if col not in df.columns:
        return float("nan")
    sub = pd.DataFrame({"_v": df[col]})
    w = rolling_window(sub, ts, window)
    if w.empty:
        try:
            w = sub.loc[sub.index.min() : pd.Timestamp(ts)]
        except Exception:
            return float("nan")
    if w.empty:
        return float("nan")
    s = pd.to_numeric(w["_v"], errors="coerce")
    first_i = s.first_valid_index()
    if first_i is None:
        return float("nan")
    initial = float(s.loc[first_i])
    try:
        current = float(s.loc[ts])
    except KeyError:
        return float("nan")
    if not math.isfinite(initial) or not math.isfinite(current):
        return float("nan")
    return safe_div(current - initial, initial)


def _window_nan_frac(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 1.0
    return float(pd.isna(s).sum() / len(s))


def _too_much_missing(s: pd.Series, max_frac: float = 0.3) -> bool:
    return _window_nan_frac(s) > max_frac


def _grid_voltage(df: pd.DataFrame, ts: pd.Timestamp) -> str:
    raw: Optional[str] = None
    if "grid_side_voltage" in df.columns:
        try:
            v = df["grid_side_voltage"].loc[ts]
            if v is not None and not pd.isna(v):
                raw = str(v).strip()
        except KeyError:
            pass
    if raw is None:
        v2 = df.attrs.get("grid_side_voltage")
        if v2 is not None:
            raw = str(v2).strip()
    if raw is None:
        return "500kV"
    if "750" in raw:
        return "750kV"
    if "500" in raw:
        return "500kV"
    return raw


def _series_delta_T(df: pd.DataFrame) -> pd.Series:
    if "delta_T" in df.columns:
        return pd.to_numeric(df["delta_T"], errors="coerce")
    if "T_out" in df.columns and "T_in" in df.columns:
        return pd.to_numeric(df["T_out"], errors="coerce") - pd.to_numeric(df["T_in"], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


def _series_delta_P(df: pd.DataFrame) -> pd.Series:
    if "delta_P" in df.columns:
        return pd.to_numeric(df["delta_P"], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


def _total_hydrocarbon_rates_30d(df: pd.DataFrame, ts: pd.Timestamp) -> tuple[float, float]:
    """(absolute gas rate uL/L/day, relative gas rate %/day), 30d window; returns nan if missing>30%."""
    sub = rolling_window(pd.DataFrame({"c": _series(df, "oil_dga_total_hydrocarbon")}), ts, "30d")
    s = pd.to_numeric(sub["c"], errors="coerce")
    if _too_much_missing(s):
        return float("nan"), float("nan")
    s2 = s.dropna()
    if len(s2) < 2:
        return float("nan"), float("nan")
    t0, t1 = s2.index[0], s2.index[-1]
    c0, c1 = float(s2.iloc[0]), float(s2.iloc[-1])
    dt_days = (t1 - t0).total_seconds() / 86400.0
    if dt_days <= 0:
        return float("nan"), float("nan")
    abs_rate = (c1 - c0) / dt_days
    rel_rate = safe_div((c1 - c0) * 100.0, c0 * dt_days)
    return abs_rate, rel_rate


def _slope_in_window(df: pd.DataFrame, ts: pd.Timestamp, window: str, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    w = rolling_window(pd.DataFrame({"x": df[col]}), ts, window)["x"]
    if _too_much_missing(w):
        return float("nan")
    return slope_per_day(w)


def _std_in_window_series(s: pd.Series, max_miss: float = 0.3) -> float:
    if s is None or len(s) == 0 or _too_much_missing(s, max_miss):
        return float("nan")
    return std(s)


def _oltc_total_ops_30d(df: pd.DataFrame, ts: pd.Timestamp) -> float:
    if "total_operations_30d" in df.columns:
        v = _to_float(df["total_operations_30d"].loc[ts])
        if math.isfinite(v):
            return v
    w = rolling_window(pd.DataFrame({"x": _series(df, "oltc_operation_count")}), ts, "30d")
    s = pd.to_numeric(w["x"], errors="coerce")
    if _too_much_missing(s) or len(s.dropna()) < 2:
        return float("nan")
    s2 = s.dropna()
    return float(s2.iloc[-1] - s2.iloc[0])


def _oltc_daily_ops_stats_30d(df: pd.DataFrame, ts: pd.Timestamp) -> tuple[float, float]:
    w = rolling_window(pd.DataFrame({"x": _series(df, "oltc_operation_count")}), ts, "30d")
    if w.empty or _too_much_missing(w["x"]):
        return float("nan"), float("nan")
    daily = w["x"].resample("1D").last()
    dops = daily.diff().dropna()
    dops = pd.to_numeric(dops, errors="coerce").clip(lower=0).dropna()
    if len(dops) == 0:
        return float("nan"), float("nan")
    mx = float(dops.max())
    st = float(dops.std(ddof=0)) if len(dops) > 1 else float("nan")
    return mx, st


def _co_co2_ratio_series(df: pd.DataFrame) -> pd.Series:
    co = pd.to_numeric(_series(df, "oil_dga_co"), errors="coerce")
    co2 = pd.to_numeric(_series(df, "oil_dga_co2"), errors="coerce")
    out = co / co2.replace(0, float("nan"))
    return out.replace([float("inf"), float("-inf")], float("nan"))


def _oltc_total_hydrocarbon_abs_rate_30d(df: pd.DataFrame, ts: pd.Timestamp) -> float:
    for col in ("oil_dga_total_hydrocarbon_oltc", "oltc_total_hydrocarbon"):
        if col not in df.columns:
            continue
        sub = rolling_window(pd.DataFrame({"c": df[col]}), ts, "30d")
        s = pd.to_numeric(sub["c"], errors="coerce")
        if _too_much_missing(s):
            continue
        s2 = s.dropna()
        if len(s2) < 2:
            continue
        t0, t1 = s2.index[0], s2.index[-1]
        c0, c1 = float(s2.iloc[0]), float(s2.iloc[-1])
        dt_days = (t1 - t0).total_seconds() / 86400.0
        if dt_days <= 0:
            continue
        return (c1 - c0) / dt_days
    return float("nan")


def rule_k1(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    x = _series(df, "top_oil_temp")
    cond = x > p["T_alarm"]
    trig = consec_nh(_last_n(df, ts, 3, "top_oil_temp") > p["T_alarm"], 3) or (_series(df, "top_oil_temp").loc[ts] > p["T_trip"])
    return RuleResult("K1", bool(trig), {"T_alarm": p["T_alarm"], "T_trip": p["T_trip"]})


def rule_k2(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    trig = consec_nh(_last_n(df, ts, 3, "winding_hotspot_temp") > p["T_alarm"], 3) or (_series(df, "winding_hotspot_temp").loc[ts] > p["T_trip"])
    return RuleResult("K2", bool(trig), {})


def rule_k3(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    s = _last_n(df, ts, 3, "oil_level")
    trig = consec_nh(s < p["L_min"], 3) or consec_nh(s > p["L_max"], 3)
    return RuleResult("K3", bool(trig), {})


def rule_k4(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    gases = {
        "oil_dga_h2": p["T_H2"],
        "oil_dga_ch4": p["T_CH4"],
        "oil_dga_c2h6": p["T_C2H6"],
        "oil_dga_c2h4": p["T_C2H4"],
        "oil_dga_c2h2": p["T_C2H2"],
        "oil_dga_co": p["T_CO"],
        "oil_dga_co2": p["T_CO2"],
    }
    point = False
    for col, thr in gases.items():
        if _to_float(_series(df, col).loc[ts]) > thr:
            point = True
            break
    computed = (
        _to_float(_series(df, "oil_dga_ch4").loc[ts])
        + _to_float(_series(df, "oil_dga_c2h6").loc[ts])
        + _to_float(_series(df, "oil_dga_c2h4").loc[ts])
        + _to_float(_series(df, "oil_dga_c2h2").loc[ts])
    )
    diff = abs_(_to_float(_series(df, "oil_dga_total_hydrocarbon").loc[ts]) - computed)
    abs_rate, rel_rate = _total_hydrocarbon_rates_30d(df, ts)
    rate_trig = (math.isfinite(abs_rate) and abs_rate > p["T_abs"]) or (math.isfinite(rel_rate) and rel_rate > p["T_rel"])
    trig = point or rate_trig or (diff > p["T_total_hydrocarbon_diff"])
    return RuleResult(
        "K4",
        bool(trig),
        {"total_hydrocarbon_diff": diff, "abs_rate_30d": abs_rate, "rel_rate_30d": rel_rate},
    )


def rule_k5(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    # HARD=0 stage2: rel_change_tan > T_tanδ  (column stores tanδ)
    tan = _to_float(_series(df, "rel_change_tan").loc[ts])
    thr = p.get("T_tanδ", float("nan"))
    trig = math.isfinite(tan) and math.isfinite(thr) and tan > thr
    return RuleResult("K5", bool(trig), {"rel_change_tan": tan, "T_tanδ": thr})


def rule_k6(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    trig = consec_nh(_last_n(df, ts, 3, "core_ground_current") > p["T_alarm"], 3)
    return RuleResult("K6", bool(trig), {})


def rule_k7(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    s = _win_series(df, ts, "1h", "cooling_system_status")
    if len(s) == 0 or _too_much_missing(s):
        return RuleResult("K7", False, {})
    cond = s.isin(["fault", "alarm"])
    trig = duration_seconds(cond) >= p["T_duration"]
    return RuleResult("K7", bool(trig), {})


def rule_k8(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    s = _win_series(df, ts, "1h", "oltc_overall_status")
    if len(s) == 0 or _too_much_missing(s):
        return RuleResult("K8", False, {})
    cond = s.isin(["fault", "alarm", "tap_overlimit"])
    trig = duration_seconds(cond) >= p["T_duration"]
    return RuleResult("K8", bool(trig), {})


def rule_k9(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    cols = ["transformer_body_pressure", "conservator_pressure", "bushing_pressure"]
    trig = False
    for c in cols:
        s = _last_n(df, ts, 3, c)
        trig = trig or consec_nh(s < p["P_min"], 3) or consec_nh(s > p["P_max"], 3)
    return RuleResult("K9", bool(trig), {})


def rule_k10(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    # HARD=0: consec_3h(pd_level > T_attention) OR avg_24h(pd_level) > T_background * K
    s3 = _last_n(df, ts, 3, "pd_level")
    branch1 = consec_nh(pd.to_numeric(s3, errors="coerce") > p["T_attention"], 3)
    w24 = rolling_window(pd.DataFrame({"pd": _series(df, "pd_level")}), ts, "24h")["pd"]
    w24f = pd.to_numeric(w24, errors="coerce")
    branch2 = False
    if not _too_much_missing(w24f):
        m24 = avg(w24f)
        branch2 = math.isfinite(m24) and m24 > p["T_background"] * p["K"]
    trig = branch1 or branch2
    return RuleResult("K10", bool(trig), {"branch24h_avg": branch2})


def rule_k11(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    if not _applies_when(df, ts, "cooling_method", frozenset({"forced_oil_forced_air"})):
        return RuleResult("K11", False, {})
    trig = False
    if "T_in" in df.columns:
        trig = trig or consec_nh(_last_n(df, ts, 3, "T_in") > p["T_in_max"], 3)
    t_out_col = "T_out" if "T_out" in df.columns else "cooling_water_temp"
    trig = trig or consec_nh(_last_n(df, ts, 3, t_out_col) > p["T_out_max"], 3)
    p3 = pd.to_numeric(_last_n(df, ts, 3, "cooling_water_pressure"), errors="coerce")
    p_bad = (p3 < p["P_min"]) | (p3 > p["P_max"])
    trig = trig or consec_nh(p_bad, 3)
    trig = trig or consec_nh(_last_n(df, ts, 3, "cooling_water_flow") < p["flow_min"], 3)
    dT = _series_delta_T(df)
    w24_dt = rolling_window(pd.DataFrame({"x": dT}), ts, "24h")["x"]
    if not _too_much_missing(pd.to_numeric(w24_dt, errors="coerce")):
        trig = trig or (std(w24_dt) > p["T_std_deltaT"])
    dP = _series_delta_P(df)
    w24_dp = rolling_window(pd.DataFrame({"x": dP}), ts, "24h")["x"]
    if not _too_much_missing(pd.to_numeric(w24_dp, errors="coerce")):
        trig = trig or (std(w24_dp) > p["T_std_deltaP"])
    return RuleResult("K11", bool(trig), {})


def rule_k12(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    if not _applies_when(df, ts, "bushing_type", frozenset({"SF6_gas_insulated_bushing"})):
        return RuleResult("K12", False, {})
    s3 = _last_n(df, ts, 3, "sf6_gas_pressure_density")
    trig = consec_nh(s3 < p["P_alarm"], 3) or (slope_per_day(_win_series(df, ts, "30d", "sf6_gas_pressure_density")) < p["S_min"])
    return RuleResult("K12", bool(trig), {})


def rule_k13(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    if not _applies_when(df, ts, "tap_changing_method", frozenset({"on_load_tap_changing"})):
        return RuleResult("K13", False, {})
    b1 = _to_float(_series(df, "c2h2_oltc").loc[ts]) > p["T_C2H2_oltc"]
    rate_30d = _oltc_total_hydrocarbon_abs_rate_30d(df, ts)
    if math.isfinite(rate_30d):
        b2 = rate_30d > p["T_rate_oltc"]
    else:
        # Without OLTC total hydrocarbon, fall back to gas_rate_oltc (matches stage2 expr; may not be 30d-recomputed)
        b2 = _to_float(_series(df, "gas_rate_oltc").loc[ts]) > p["T_rate_oltc"]
    trig = b1 or b2
    return RuleResult("K13", bool(trig), {})


def rule_k14(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    if not _applies_when(df, ts, "cooling_method", frozenset({"forced_oil_forced_air"})):
        return RuleResult("K14", False, {})
    c_moist = (
        _slope_in_window(df, ts, "90d", "oil_moisture") > p["S_moisture"]
        if "oil_moisture" in df.columns
        else False
    )
    dT = _series_delta_T(df)
    w90_dt = rolling_window(pd.DataFrame({"x": dT}), ts, "90d")["x"]
    c_std_dt = (
        _std_in_window_series(pd.to_numeric(w90_dt, errors="coerce")) > p["T_std_deltaT_long"]
        if len(w90_dt)
        else False
    )
    s_th = _slope_in_window(df, ts, "90d", "oil_dga_total_hydrocarbon")
    s_h2 = _slope_in_window(df, ts, "90d", "oil_dga_h2")
    c_slope_gas = (math.isfinite(s_th) and s_th > p["S_total_hydrocarbon"]) or (
        math.isfinite(s_h2) and s_h2 > p["S_hydrogen"]
    )
    th = _to_float(_series(df, "oil_dga_total_hydrocarbon").loc[ts])
    sm = (
        _to_float(_series(df, "oil_dga_ch4").loc[ts])
        + _to_float(_series(df, "oil_dga_c2h6").loc[ts])
        + _to_float(_series(df, "oil_dga_c2h4").loc[ts])
        + _to_float(_series(df, "oil_dga_c2h2").loc[ts])
    )
    frac = safe_div(sm, th)
    ratio_series = _co_co2_ratio_series(df)
    w90_co = rolling_window(pd.DataFrame({"x": ratio_series}), ts, "90d")["x"]
    s_co2 = slope_per_day(w90_co) if len(w90_co) and not _too_much_missing(w90_co) else float("nan")
    c_ratio = (math.isfinite(frac) and frac > p["R_hydrocarbon_ratio"]) or (
        math.isfinite(s_co2) and s_co2 > p["S_CO_CO2_ratio"]
    )
    st = abs_(_slope_in_window(df, ts, "90d", "cooling_water_temp"))
    sp = abs_(_slope_in_window(df, ts, "90d", "cooling_water_pressure"))
    sf = abs_(_slope_in_window(df, ts, "90d", "cooling_water_flow"))
    c_slopes = (st > p["S_temp_slope"]) or (sp > p["S_pressure_slope"]) or (sf > p["S_flow_slope"])
    idx_perf = st * sp * sf
    c_idx = math.isfinite(idx_perf) and idx_perf > p["I_cooling_performance"]
    trig = bool(c_moist and c_std_dt and c_slope_gas and c_ratio and c_slopes and c_idx)
    return RuleResult("K14", trig, {})


def _device_field(df: pd.DataFrame, ts: pd.Timestamp, key: str) -> Optional[str]:
    if key in df.columns:
        try:
            v = df[key].loc[ts]
            if v is None or pd.isna(v):
                return None
            return str(v).strip()
        except KeyError:
            pass
    v2 = df.attrs.get(key)
    if v2 is None:
        return None
    return str(v2).strip()


def _applies_when(df: pd.DataFrame, ts: pd.Timestamp, field: str, allowed: frozenset[str]) -> bool:
    """If detail provides field, it must be in allowed; otherwise still participate (legacy CSV without metadata)."""
    v = _device_field(df, ts, field)
    if v is None:
        return True
    return v in allowed


def rule_k15(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    if not _applies_when(df, ts, "bushing_type", frozenset({"SF6_gas_insulated_bushing"})):
        return RuleResult("K15", False, {})
    w = rolling_window(pd.DataFrame({"p": _series(df, "sf6_gas_pressure_density")}), ts, "365d")["p"]
    wf = pd.to_numeric(w, errors="coerce")
    if _too_much_missing(wf) or len(wf) < 3:
        return RuleResult("K15", False, {})
    slope = slope_per_day(wf)
    # count_365d(P_SF6 < P_alarm): sample count satisfying condition in window (matches JSON literal)
    n_below = int((wf < p["P_alarm"]).sum())
    trig = math.isfinite(slope) and slope < p["S_leak"] and n_below >= int(p["N_alarm_year"])
    return RuleResult("K15", bool(trig), {"slope_365d": slope, "count_below_alarm_365d": n_below})


def rule_k16(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    if not _applies_when(df, ts, "tap_changing_method", frozenset({"on_load_tap_changing"})):
        return RuleResult("K16", False, {})
    total = _oltc_total_ops_30d(df, ts)
    b1 = math.isfinite(total) and total > p["T_ops_warn"]
    dmax, dst = _oltc_daily_ops_stats_30d(df, ts)
    b2 = (
        math.isfinite(dmax)
        and math.isfinite(dst)
        and dmax > p["T_daily_ops_max"]
        and dst > p["T_daily_ops_std"]
    )
    c2 = _to_float(_series(df, "c2h2_oltc").loc[ts])
    gr = _to_float(_series(df, "gas_rate_oltc").loc[ts])
    b3 = (c2 > p["T_C2H2_oltc_warn"]) and (gr > p["T_rate_oltc_warn"])
    trig = b1 or b2 or b3
    return RuleResult("K16", bool(trig), {})


def rule_k17(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    if not _applies_when(df, ts, "primary_standby_role", frozenset({"standby"})):
        return RuleResult("K17", False, {})
    s_h2 = _slope_in_window(df, ts, "180d", "oil_dga_h2")
    s_oil = _slope_in_window(df, ts, "180d", "oil_level")
    b1 = math.isfinite(s_h2) and s_h2 > p["S_H2_standby"]
    b2 = math.isfinite(s_oil) and s_oil < p["S_oil_standby"]
    trig = b1 or b2
    return RuleResult("K17", bool(trig), {})


def rule_k18(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    if not _applies_when(df, ts, "grid_side_voltage", frozenset({"500kV", "750kV"})):
        return RuleResult("K18", False, {})
    gv = _grid_voltage(df, ts)
    wpd = rolling_window(pd.DataFrame({"x": _series(df, "pd_level")}), ts, "7d")["x"]
    wpd_f = pd.to_numeric(wpd, errors="coerce")
    av = avg(wpd_f) if len(wpd_f) and not _too_much_missing(wpd_f) else float("nan")
    mx = float(wpd_f.max()) if len(wpd_f) and not _too_much_missing(wpd_f) else float("nan")
    trig_pd = False
    if gv == "500kV":
        trig_pd = (math.isfinite(av) and av > p["T_PD_500kV"]) or (math.isfinite(mx) and mx > p["T_PD_500kV_peak"])
    elif gv == "750kV":
        trig_pd = (math.isfinite(av) and av > p["T_PD_750kV"]) or (math.isfinite(mx) and mx > p["T_PD_750kV_peak"])
    wtan = rolling_window(pd.DataFrame({"x": _series(df, "rel_change_tan")}), ts, "7d")["x"]
    wtan_f = pd.to_numeric(wtan, errors="coerce")
    av_tan = avg(wtan_f) if len(wtan_f) and not _too_much_missing(wtan_f) else float("nan")
    trig_tan = False
    if gv == "500kV":
        trig_tan = math.isfinite(av_tan) and av_tan > p["T_tanδ_500kV"]
    elif gv == "750kV":
        trig_tan = math.isfinite(av_tan) and av_tan > p["T_tanδ_750kV"]
    trig = trig_pd or trig_tan
    return RuleResult("K18", bool(trig), {"grid_side_voltage": gv})


def rule_k19(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    # oil_dga_c2h2 / oil_dga_c2h4 > R_discharge
    ratio = _point_derived(df, ts, "dga_ratio_c2h2_c2h4")
    trig = ratio > p["R_discharge"]
    return RuleResult("K19", bool(trig), {"ratio": ratio})


def rule_k20(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    ratio = _point_derived(df, ts, "dga_ratio_c2h2_c2h4")
    total = _to_float(_series(df, "oil_dga_total_hydrocarbon").loc[ts])
    trig = (ratio < p["R_low"]) and (total > p["T_total_hydrocarbon"])
    return RuleResult("K20", bool(trig), {"ratio": ratio, "total_hydrocarbon": total})


def rule_k21(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    s3 = _last_n_derived(df, ts, 3, "winding_hotspot_temperature_rise")
    trig = consec_nh(s3 > p["T_rise_alarm"], 3) or (_point_derived(df, ts, "winding_hotspot_temperature_rise") > p["T_rise_trip"])
    return RuleResult("K21", bool(trig), {})


def rule_k22(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    rise = _point_derived(df, ts, "winding_hotspot_temperature_rise")
    water = _to_float(_series(df, "cooling_water_temp").loc[ts])
    trig = (rise > p["T_rise_high"]) and (water >= p["T_water_min"]) and (water <= p["T_water_max"])
    return RuleResult("K22", bool(trig), {"rise": rise, "cooling_water_temp": water})


def rule_k23(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    s3 = _last_n_derived(df, ts, 3, "oil_to_ambient_temperature_diff")
    trig = consec_nh(s3 > p["T_diff_high"], 3)
    return RuleResult("K23", bool(trig), {})


def rule_k24(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    s3 = _last_n_derived(df, ts, 3, "oil_to_ambient_temperature_diff")
    trig = consec_nh(s3 < p["T_diff_low"], 3)
    return RuleResult("K24", bool(trig), {})


def rule_k25(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    w = _win_derived(df, ts, "24h", "oil_to_ambient_temperature_diff")
    trig = std(w) > p["T_diff_std"]
    return RuleResult("K25", bool(trig), {"std_24h": std(w)})


def rule_k26(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    # consec_3h(abs(body-cons) > P_diff_max)
    body = _series(df, "transformer_body_pressure").astype(float)
    cons = _series(df, "conservator_pressure").astype(float)
    tmp_col = "abs_pressure_diff_body_conservator"
    if tmp_col not in df.columns:
        df[tmp_col] = (body - cons).abs()
    s3 = _last_n(df, ts, 3, tmp_col)
    trig = consec_nh(s3 > p["P_diff_max"], 3)
    return RuleResult("K26", bool(trig), {})


def rule_k27(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    diff = _ensure_derived_col(df, "pressure_diff_body_conservator")
    s3 = last_n_hours(pd.DataFrame({"x": diff}), ts, 3)["x"]
    trig = consec_nh(s3 < -p["P_diff_max"], 3)
    return RuleResult("K27", bool(trig), {})


def rule_k28(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    ratio = _point_derived(df, ts, "dga_ratio_co2_co")
    trig = ratio > p["T_ratio_high"]
    return RuleResult("K28", bool(trig), {"ratio": ratio})


def rule_k29(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    ratio = _point_derived(df, ts, "dga_ratio_co2_co")
    trig = ratio < p["T_ratio_low"]
    return RuleResult("K29", bool(trig), {"ratio": ratio})


def rule_k30(df: pd.DataFrame, ts: pd.Timestamp, p: Dict[str, float]) -> RuleResult:
    w = _win_derived(df, ts, "30d", "dga_ratio_co2_co")
    trig = slope_per_day(w) > p["S_ratio_slope"]
    return RuleResult("K30", bool(trig), {"slope_30d": slope_per_day(w)})


def _adapt_rule_call(fn: Callable[[pd.DataFrame, pd.Timestamp, Dict[str, float]], RuleResult]) -> Callable[..., RuleResult]:
    _params_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _rid_from_fn_name() -> str:
        n = getattr(fn, "__name__", "")
        if n.startswith("rule_k"):
            suf = n.split("rule_k", 1)[1]
            if suf.isdigit():
                return f"K{int(suf)}"
        return ""

    def _load_params_for_rule(model_dir: Any) -> Dict[str, float]:
        try:
            folder = Path(model_dir)
        except Exception:
            return {}
        key = str(folder.resolve())
        if key not in _params_cache:
            _params_cache[key] = load_params_by_rule(folder)
        rid = _rid_from_fn_name()
        if not rid:
            return {}
        return dict(_params_cache[key].get(rid, {}) or {})

    def _to_legacy_return(res: Any) -> Any:
        # Legacy callers in some recheck scripts expect tuple unpack: (triggered, details)
        if isinstance(res, RuleResult):
            return bool(res.triggered), dict(res.details or {})
        return res

    def _wrapped(*args: Any, **kwargs: Any) -> RuleResult:
        if kwargs:
            # Keep native keyword-based calls unchanged.
            return fn(*args, **kwargs)

        if len(args) == 3:
            a0, a1, a2 = args
            # legacy form: (model_dir, df, ts)
            if isinstance(a0, (str, Path)) and isinstance(a1, pd.DataFrame):
                params = _load_params_for_rule(a0)
                return _to_legacy_return(fn(a1, a2, params))
            # native form: (df, ts, params)
            return fn(a0, a1, a2)

        if len(args) == 4:
            a0, a1, a2, a3 = args
            # legacy form: (model_dir, df, ts, params)
            if isinstance(a0, (str, Path)) and isinstance(a1, pd.DataFrame):
                params = a3 if isinstance(a3, dict) and a3 else _load_params_for_rule(a0)
                return _to_legacy_return(fn(a1, a2, params))

        return fn(*args)

    return _wrapped


RULE_REGISTRY: Dict[str, Callable[..., RuleResult]] = {
    "K1": rule_k1,
    "K2": rule_k2,
    "K3": rule_k3,
    "K4": rule_k4,
    "K5": rule_k5,
    "K6": rule_k6,
    "K7": rule_k7,
    "K8": rule_k8,
    "K9": rule_k9,
    "K10": rule_k10,
    "K11": rule_k11,
    "K12": rule_k12,
    "K13": rule_k13,
    "K14": rule_k14,
    "K15": rule_k15,
    "K16": rule_k16,
    "K17": rule_k17,
    "K18": rule_k18,
    "K19": rule_k19,
    "K20": rule_k20,
    "K21": rule_k21,
    "K22": rule_k22,
    "K23": rule_k23,
    "K24": rule_k24,
    "K25": rule_k25,
    "K26": rule_k26,
    "K27": rule_k27,
    "K28": rule_k28,
    "K29": rule_k29,
    "K30": rule_k30,
}

# Expose compatibility-wrapped rule functions for external dynamic callers.
for _name, _obj in list(globals().items()):
    if _name.startswith("rule_k") and callable(_obj):
        globals()[_name] = _adapt_rule_call(_obj)

# Keep registry aligned with wrapped rule callables.
for _k, _fn in list(RULE_REGISTRY.items()):
    RULE_REGISTRY[_k] = _adapt_rule_call(_fn)



@lru_cache(maxsize=8)
def _alias_to_db(folder_str: str) -> Dict[str, str]:
    folder = Path(folder_str)
    mapping: Dict[str, str] = {}
    items = json.loads((folder / "metric_catalog.json").read_text(encoding="utf-8"))
    if not isinstance(items, list):
        return mapping
    for it in items:
        if not isinstance(it, dict):
            continue
        db = _compact_str(it.get("database_field"))
        if not db:
            continue
        for key in ("database_field", "variable_name", "metric_name"):
            a = _compact_str(it.get(key))
            if a:
                mapping[a] = db
                mapping[a.lower()] = db
    dp = folder / "derived_metrics.json"
    if dp.exists():
        data = json.loads(dp.read_text(encoding="utf-8"))
        for it in data.get("derived_metrics") or []:
            if not isinstance(it, dict):
                continue
            key = _compact_str(it.get("derived_key"))
            name = _compact_str(it.get("derived_name"))
            if key:
                mapping[key] = key
                mapping[key.lower()] = key
            if name and key:
                mapping[name] = key
    return mapping


def describe_rule_dependencies(folder: Optional[Path] = None) -> List[Dict[str, Any]]:
    folder2 = Path(folder) if folder is not None else _model_dir()
    rules = load_stage2_rules(folder2)
    alias = _alias_to_db(str(folder2.resolve()))
    rows: List[Dict[str, Any]] = []
    for rid in sorted(rules.keys(), key=lambda x: (len(x), x)):
        r = rules[rid]
        inputs = r.get("input_metrics") or []
        deps: List[str] = []
        for cn in inputs:
            db = alias.get(_compact_str(cn)) or alias.get(_compact_str(cn).lower()) or ""
            if db and db not in deps:
                deps.append(db)
        rows.append(
            {
                "rule_id": rid,
                "function": f"rule_{rid.lower()}",
                "object_name": _compact_str(r.get("object_name")),
                "rule_expression": _compact_str(r.get("rule_expression")),
                "primary_dependency_columns": deps,
                "in_registry": rid in RULE_REGISTRY,
            }
        )
    return rows
