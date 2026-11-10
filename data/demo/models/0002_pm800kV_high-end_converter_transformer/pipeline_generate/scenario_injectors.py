from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def _hours_idx(df: pd.DataFrame, ts: pd.Timestamp, hours: int) -> pd.DatetimeIndex:
    idx_full = pd.date_range(end=ts, periods=int(hours), freq="h")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        return idx_full[(idx_full >= df.index.min()) & (idx_full <= df.index.max())]
    return idx_full


def inject_consec_above(df: pd.DataFrame, ts: pd.Timestamp, col: str, threshold: float, hours: int = 3, margin: float = 1.0) -> None:
    ensure_columns(df, [col])
    idx = _hours_idx(df, ts, hours)
    if len(idx) == 0:
        return
    jitter = np.random.uniform(0.0, max(0.1, margin), size=len(idx))
    df.loc[idx, col] = threshold + margin + jitter


def inject_consec_below(df: pd.DataFrame, ts: pd.Timestamp, col: str, threshold: float, hours: int = 3, margin: float = 1.0) -> None:
    ensure_columns(df, [col])
    idx = _hours_idx(df, ts, hours)
    if len(idx) == 0:
        return
    jitter = np.random.uniform(0.0, max(0.1, margin), size=len(idx))
    df.loc[idx, col] = threshold - margin - jitter


def inject_slope_decreasing(df: pd.DataFrame, ts: pd.Timestamp, col: str, window_days: int, slope_per_day_target: float) -> None:
    ensure_columns(df, [col])
    n_full = int(window_days) * 24
    idx_full = pd.date_range(end=ts, periods=n_full, freq="h")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        idx = idx_full[(idx_full >= df.index.min()) & (idx_full <= df.index.max())]
    else:
        idx = idx_full
    if len(idx) <= 1:
        return
    span_hours = max(1, int(len(idx) - 1))
    eff_days = span_hours / 24.0
    try:
        v0 = df.loc[idx[0], col]
        start = float(v0) if pd.notna(v0) else 1.0
    except Exception:
        start = 1.0
    ramp = np.linspace(0.0, float(slope_per_day_target) * eff_days, len(idx))
    df.loc[idx, col] = start + ramp


def inject_slope_increasing(df: pd.DataFrame, ts: pd.Timestamp, col: str, window_days: int, slope_per_day: float, base: float = 0.0) -> None:
    """Positive slope (units/day) over window_days ending at ts."""
    ensure_columns(df, [col])
    n_full = int(window_days) * 24
    idx_full = pd.date_range(end=ts, periods=n_full, freq="h")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        idx = idx_full[(idx_full >= df.index.min()) & (idx_full <= df.index.max())]
    else:
        idx = idx_full
    if len(idx) <= 1:
        return
    eff_days = max(1.0 / 24.0, (len(idx) - 1) / 24.0)
    ramp = np.linspace(0.0, float(slope_per_day) * eff_days, len(idx))
    df.loc[idx, col] = float(base) + ramp


def inject_state_duration(df: pd.DataFrame, ts: pd.Timestamp, col: str, state: str, hours: int = 1) -> None:
    ensure_columns(df, [col])
    if df[col].dtype != "object":
        df[col] = df[col].astype("object")
    idx = _hours_idx(df, ts, hours)
    if len(idx) == 0:
        return
    df.loc[idx, col] = state


def _sync_cooling_pressure_aliases(df: pd.DataFrame) -> None:
    """rule_k11 reads cooling_water_pressure; background may use P_water."""
    if "cooling_water_pressure" in df.columns and "P_water" not in df.columns:
        df["P_water"] = df["cooling_water_pressure"]
    elif "P_water" in df.columns and "cooling_water_pressure" not in df.columns:
        df["cooling_water_pressure"] = df["P_water"]


# ---- Rule-specific injectors for 0002 ----


@dataclass
class InjectResult:
    rule_id: str
    injected: bool
    notes: str


def inject_k1(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float], col: str = "top_oil_temp") -> InjectResult:
    """rule_k1: consec_3h > T_alarm OR point > T_trip (soft margins limit K25 bleed)."""
    if np.random.rand() < 0.55 and "T_alarm" in params:
        hours = 3
        margin = float(np.random.uniform(0.3, 1.0))
        inject_consec_above(df, ts, col, params["T_alarm"], hours=hours, margin=margin)
        return InjectResult("K1", True, f"branch alarm: last {hours}h > T_alarm (soft)")
    if "T_trip" in params:
        ensure_columns(df, [col])
        margin = float(np.random.uniform(0.3, 1.2))
        df.loc[ts, col] = params["T_trip"] + margin
        return InjectResult("K1", True, "branch trip: single point > T_trip (soft)")
    inject_consec_above(df, ts, col, params["T_alarm"], hours=3, margin=0.5)
    return InjectResult("K1", True, "fallback consec > T_alarm")


def inject_k2(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float], col: str = "winding_hotspot_temp") -> InjectResult:
    if np.random.rand() < 0.55 and "T_alarm" in params:
        hours = 3
        margin = float(np.random.uniform(0.3, 1.2))
        inject_consec_above(df, ts, col, params["T_alarm"], hours=hours, margin=margin)
        return InjectResult("K2", True, f"branch alarm: last {hours}h > T_alarm (soft)")
    if "T_trip" in params:
        ensure_columns(df, [col])
        margin = float(np.random.uniform(0.3, 1.5))
        df.loc[ts, col] = params["T_trip"] + margin
        return InjectResult("K2", True, "branch trip: single point > T_trip")
    inject_consec_above(df, ts, col, params["T_alarm"], hours=3, margin=1.0)
    return InjectResult("K2", True, "fallback consec > T_alarm")


def inject_k3(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float], col: str = "oil_level") -> InjectResult:
    """rule_k3: consec below L_min OR consec above L_max."""
    if np.random.rand() < 0.5 and "L_min" in params:
        hours = int(np.random.choice([3, 4, 6]))
        margin = float(np.random.uniform(0.5, 5.0))
        inject_consec_below(df, ts, col, params["L_min"], hours=hours, margin=margin)
        return InjectResult("K3", True, f"branch low: {hours}h < L_min")
    if "L_max" in params:
        hours = int(np.random.choice([3, 4, 6]))
        margin = float(np.random.uniform(0.5, 5.0))
        inject_consec_above(df, ts, col, params["L_max"], hours=hours, margin=margin)
        return InjectResult("K3", True, f"branch high: {hours}h > L_max")
    inject_consec_below(df, ts, col, params["L_min"], hours=3, margin=1.0)
    return InjectResult("K3", True, "fallback low oil")


def inject_k6(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float], col: str = "core_ground_current") -> InjectResult:
    hours = int(np.random.choice([3, 4, 5]))
    margin = float(np.random.uniform(1.0, 30.0))
    inject_consec_above(df, ts, col, params["T_alarm"], hours=hours, margin=margin)
    return InjectResult("K6", True, f"set last {hours}h above T_alarm by margin~{margin:.2g}")


def inject_k7(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float], col: str = "cooling_system_status") -> InjectResult:
    """rule_k7: duration in 1h window >= T_duration (seconds). Hourly: need ceil(T_duration/3600) trailing hours bad if window extended; impl uses 1h slice — 1 bad hour suffices when T_duration<=3600."""
    state = str(np.random.choice(["fault", "alarm"]))
    dur_s = float(params.get("T_duration", 3600.0))
    nh = max(1, min(24, int(np.ceil(dur_s / 3600.0))))
    inject_state_duration(df, ts, col, state, hours=nh)
    df["cooling_status"] = df[col]
    return InjectResult("K7", True, f"set last {nh}h to {state} (T_duration={dur_s:g}s)")


def inject_k8(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float], col: str = "oltc_overall_status") -> InjectResult:
    state = str(np.random.choice(["fault", "alarm", "tap_overlimit"]))
    dur_s = float(params.get("T_duration", 3600.0))
    nh = max(1, min(24, int(np.ceil(dur_s / 3600.0))))
    inject_state_duration(df, ts, col, state, hours=nh)
    df["oltc_status"] = df[col]
    return InjectResult("K8", True, f"set last {nh}h OLTC to {state} (T_duration={dur_s:g}s)")


def inject_k12(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float], col: str = "sf6_gas_pressure_density") -> InjectResult:
    hours = int(np.random.choice([3, 4, 6]))
    margin = float(np.random.uniform(0.001, 0.08))
    inject_consec_below(df, ts, col, params["P_alarm"], hours=hours, margin=margin)
    if bool(np.random.rand() < 0.7):
        inject_slope_decreasing(df, ts, col, window_days=30, slope_per_day_target=params["S_min"] * float(np.random.uniform(1.1, 2.0)))
    df["P_SF6"] = df[col]
    return InjectResult("K12", True, f"below P_alarm {hours}h + optional slope (P_SF6 mirrored)")


def inject_k4(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k4: gas point OR 30d abs/rel rate OR total vs components diff."""
    mode = str(np.random.choice(["gas_point", "abs_rate", "diff_total"]))
    if mode == "gas_point":
        gas_param_to_col = {
            "T_H2": "oil_dga_h2",
            "T_CH4": "oil_dga_ch4",
            "T_C2H6": "oil_dga_c2h6",
            "T_C2H4": "oil_dga_c2h4",
            "T_C2H2": "oil_dga_c2h2",
            "T_CO": "oil_dga_co",
            "T_CO2": "oil_dga_co2",
        }
        choices = [(p, c) for p, c in gas_param_to_col.items() if p in params]
        if not choices:
            ensure_columns(df, ["oil_dga_c2h2"])
            margin = float(np.random.uniform(0.1, 5.0))
            df.loc[ts, "oil_dga_c2h2"] = params.get("T_C2H2", 1.0) + margin
            return InjectResult("K4", True, f"gas_point fallback c2h2 +{margin:.2g}")
        p_name, gcol = choices[int(np.random.randint(0, len(choices)))]
        ensure_columns(df, [gcol])
        thr = float(params[p_name])
        margin = float(np.random.uniform(0.1, 5.0))
        df.loc[ts, gcol] = thr + margin
        return InjectResult("K4", True, f"gas_point {gcol} > {p_name}")
    if mode == "abs_rate" and "T_abs" in params:
        ensure_columns(df, ["oil_dga_total_hydrocarbon"])
        inject_slope_increasing(df, ts, "oil_dga_total_hydrocarbon", 30, slope_per_day=params["T_abs"] * 1.3, base=20.0)
        return InjectResult("K4", True, "branch 30d total_hydrocarbon abs rate")
    # diff branch: measured total far from sum of components at ts
    ensure_columns(df, ["oil_dga_ch4", "oil_dga_c2h6", "oil_dga_c2h4", "oil_dga_c2h2", "oil_dga_total_hydrocarbon"])
    sm = 30.0
    df.loc[ts, "oil_dga_ch4"] = sm
    df.loc[ts, "oil_dga_c2h6"] = sm
    df.loc[ts, "oil_dga_c2h4"] = sm
    df.loc[ts, "oil_dga_c2h2"] = sm
    diff_thr = float(params.get("T_total_hydrocarbon_diff", 10.0))
    df.loc[ts, "oil_dga_total_hydrocarbon"] = 4 * sm + diff_thr + 15.0
    return InjectResult("K4", True, "branch total_hydrocarbon vs components diff")


def inject_k5(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """HARD=0: rel_change_tan > T_tanδ (tiny margin limits K18 7d-avg bleed)."""
    ensure_columns(df, ["rel_change_tan"])
    thr = float(params["T_tanδ"])
    margin = float(np.random.uniform(0.0002, 0.0008))
    df.loc[ts, "rel_change_tan"] = thr + margin
    return InjectResult("K5", True, "rel_change_tan > T_tanδ (soft point)")


def inject_k9(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k9: any of body / conservator / bushing pressure low or high consec 3h."""
    cols = ["transformer_body_pressure", "conservator_pressure", "bushing_pressure"]
    col = str(np.random.choice(cols))
    hours = int(np.random.choice([3, 4, 5]))
    margin = float(np.random.uniform(0.5, 5.0))
    if np.random.rand() < 0.5:
        inject_consec_below(df, ts, col, params["P_min"], hours=hours, margin=margin)
        return InjectResult("K9", True, f"{col} low {hours}h")
    inject_consec_above(df, ts, col, params["P_max"], hours=hours, margin=margin)
    return InjectResult("K9", True, f"{col} high {hours}h")


def inject_k10(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """HARD=0: consec_3h(pd_level > T_attention) OR avg_24h > T_background*K."""
    col = "pd_level"
    ensure_columns(df, [col])
    mode = np.random.choice(["attention", "avg24"])
    if mode == "attention":
        inject_consec_above(df, ts, col, params["T_attention"], hours=3, margin=20.0)
        return InjectResult("K10", True, "consec_3h pd_level > T_attention")
    # raise ~24h mean above background*K
    idx = pd.date_range(end=ts, periods=24, freq="h")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        idx = idx[(idx >= df.index.min()) & (idx <= df.index.max())]
    thr = float(params["T_background"]) * float(params["K"])
    tgt = thr + float(np.random.uniform(20.0, 80.0))
    df.loc[idx, col] = tgt
    return InjectResult("K10", True, "avg_24h pd_level > T_background*K")


def inject_k11(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k11: T_in/T_out/pressure(compound)/flow consec OR std_24h(delta_T/delta_P). Uses cooling_water_pressure + P_water alias."""
    _sync_cooling_pressure_aliases(df)
    branches: List[str] = []
    if "T_in_max" in params:
        branches.append("T_in_high")
    if "T_out_max" in params:
        branches.append("T_out_high")
    if "P_min" in params:
        branches.append("P_bad")
    if "flow_min" in params:
        branches.append("flow_low")
    branches.extend(["std_dt", "std_dp"])
    if not branches:
        branches = ["std_dt", "flow_low"]
    b = str(np.random.choice(branches))
    hours = int(np.random.choice([3, 4, 5]))
    margin_t = float(np.random.uniform(0.5, 4.0))
    margin_p = float(np.random.uniform(0.02, 0.08))

    if b == "T_in_high":
        ensure_columns(df, ["T_in"])
        inject_consec_above(df, ts, "T_in", params["T_in_max"], hours=hours, margin=margin_t)
        return InjectResult("K11", True, "T_in high consec")
    if b == "T_out_high":
        t_out_col = "T_out" if "T_out" in df.columns else "cooling_water_temp"
        ensure_columns(df, [t_out_col])
        inject_consec_above(df, ts, t_out_col, params["T_out_max"], hours=hours, margin=margin_t)
        return InjectResult("K11", True, f"{t_out_col} high consec")
    if b == "P_bad":
        ensure_columns(df, ["cooling_water_pressure"])
        if np.random.rand() < 0.5:
            inject_consec_below(df, ts, "cooling_water_pressure", params["P_min"], hours=hours, margin=margin_p)
        else:
            inject_consec_above(df, ts, "cooling_water_pressure", params["P_max"], hours=hours, margin=margin_p)
        df["P_water"] = df["cooling_water_pressure"]
        return InjectResult("K11", True, "cooling_water_pressure OR min/max consec (+P_water)")
    if b == "flow_low":
        inject_consec_below(df, ts, "cooling_water_flow", params["flow_min"], hours=hours, margin=float(np.random.uniform(5, 30)))
        return InjectResult("K11", True, "flow low consec")
    idx24 = _hours_idx(df, ts, 24)
    if len(idx24) < 8:
        inject_consec_below(df, ts, "cooling_water_flow", params.get("flow_min", 100.0), hours=3, margin=10.0)
        return InjectResult("K11", True, "fallback flow (short window)")
    if b == "std_dt":
        ensure_columns(df, ["T_in", "T_out"])
        amp = max(3.0, float(params.get("T_std_deltaT", 2.0)) * 1.5)
        t = np.arange(len(idx24), dtype=float)
        df.loc[idx24, "T_in"] = 20.0
        df.loc[idx24, "T_out"] = 28.0 + amp * np.sin(2 * np.pi * t / 3.0)
        return InjectResult("K11", True, "std_24h(delta_T) via T_in/T_out oscillation")
    ensure_columns(df, ["delta_P"])
    amp_p = max(0.06, float(params.get("T_std_deltaP", 0.05)) * 2.0)
    t = np.arange(len(idx24), dtype=float)
    df.loc[idx24, "delta_P"] = amp_p * np.sin(2 * np.pi * t / 2.0)
    return InjectResult("K11", True, "std_24h(delta_P) via delta_P oscillation")


def inject_k13(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k13: durable point triggers only (no 30d slope — avoids lookback bleed)."""
    ensure_columns(df, ["c2h2_oltc", "gas_rate_oltc"])
    margin = float(np.random.uniform(0.5, 5.0))
    df.loc[ts, "c2h2_oltc"] = float(params["T_C2H2_oltc"]) + margin
    df.loc[ts, "gas_rate_oltc"] = float(params["T_rate_oltc"]) + float(np.random.uniform(0.1, 0.5))
    return InjectResult("K13", True, "c2h2+gas_rate durable point")



def inject_k14(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k14: full AND — inject 90d coherent trends (see rule_k14)."""
    hours = 90 * 24
    idx_full = pd.date_range(end=ts, periods=hours, freq="h")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        idx = idx_full[(idx_full >= df.index.min()) & (idx_full <= df.index.max())]
    else:
        idx = idx_full
    if len(idx) < 72:
        return InjectResult("K14", False, "insufficient history for 90d AND injection")

    n = len(idx)
    t = np.arange(n, dtype=float)
    ensure_columns(
        df,
        [
            "oil_moisture",
            "T_in",
            "T_out",
            "oil_dga_total_hydrocarbon",
            "oil_dga_h2",
            "oil_dga_ch4",
            "oil_dga_c2h6",
            "oil_dga_c2h4",
            "oil_dga_c2h2",
            "oil_dga_co",
            "oil_dga_co2",
            "cooling_water_temp",
            "cooling_water_pressure",
            "cooling_water_flow",
        ],
    )

    smois = float(params.get("S_moisture", 0.05)) * 1.6
    df.loc[idx, "oil_moisture"] = 8.0 + smois * (t / 24.0)

    base_in = 22.0
    amp = max(2.8, float(params.get("T_std_deltaT_long", 1.5)) * 1.25)
    df.loc[idx, "T_in"] = base_in
    df.loc[idx, "T_out"] = base_in + 7.0 + amp * np.sin(2 * np.pi * t / 11.0)

    sthc = float(params.get("S_total_hydrocarbon", 0.1)) * 1.6
    sh2 = float(params.get("S_hydrogen", 0.05)) * 1.6
    df.loc[idx, "oil_dga_total_hydrocarbon"] = 75.0 + sthc * (t / 24.0)
    df.loc[idx, "oil_dga_h2"] = 35.0 + sh2 * (t / 24.0)

    th_s = df.loc[idx, "oil_dga_total_hydrocarbon"].astype(float)
    rr = float(params.get("R_hydrocarbon_ratio", 0.7)) + 0.12
    each = (th_s * rr / 4.0).values
    df.loc[idx, "oil_dga_ch4"] = each
    df.loc[idx, "oil_dga_c2h6"] = each * 0.98
    df.loc[idx, "oil_dga_c2h4"] = each * 1.02
    df.loc[idx, "oil_dga_c2h2"] = each * 0.95

    co2v = 1800.0
    sco = float(params.get("S_CO_CO2_ratio", 0.01)) * 8.0
    df.loc[idx, "oil_dga_co2"] = co2v
    df.loc[idx, "oil_dga_co"] = 40.0 + sco * co2v * (t / max(1.0, n - 1))

    stemp = float(params.get("S_temp_slope", 0.02)) * 1.5
    spres = float(params.get("S_pressure_slope", 0.001)) * 1.5
    sflow = float(params.get("S_flow_slope", 0.1)) * 1.5
    df.loc[idx, "cooling_water_temp"] = 24.0 + stemp * (t / 24.0)
    df.loc[idx, "cooling_water_pressure"] = 0.32 + spres * (t / 24.0)
    df.loc[idx, "cooling_water_flow"] = 110.0 + sflow * (t / 24.0)

    return InjectResult("K14", True, "90d multi-signal AND aligned with rule_k14")


def inject_k15(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k15: slope_365d < S_leak AND count(wf < P_alarm) >= N_alarm_year."""
    col = "sf6_gas_pressure_density"
    factor = float(np.random.uniform(1.1, 2.0))
    inject_slope_decreasing(df, ts, col, window_days=365, slope_per_day_target=params["S_leak"] * factor)
    idx = _hours_idx(df, ts, min(365 * 24, max(168, len(df.index) if len(df.index) else 365 * 24)))
    if len(idx) == 0:
        return InjectResult("K15", False, "no idx for below-alarm sprinkle")
    ensure_columns(df, [col])
    n_need = int(params.get("N_alarm_year", 2)) + 3
    rng = np.random.default_rng()
    pick = rng.choice(len(idx), size=min(n_need, len(idx)), replace=False)
    palarm = float(params["P_alarm"])
    for j in pick:
        df.loc[idx[j], col] = palarm - float(np.random.uniform(0.02, 0.15))
    df["P_SF6"] = df[col]
    return InjectResult("K15", True, "365d slope + sprinkled below P_alarm samples")


def inject_k16(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    ensure_columns(df, ["oltc_operation_count", "c2h2_oltc", "gas_rate_oltc"])
    df.loc[ts, "c2h2_oltc"] = float(params["T_C2H2_oltc_warn"]) + 1.0
    df.loc[ts, "gas_rate_oltc"] = float(params["T_rate_oltc_warn"]) + 0.2

    br = str(np.random.choice(["total30", "daily", "gas_pair"]))
    if br == "daily":
        idx = _hours_idx(df, ts, 30 * 24)
        n = len(idx)
        if n >= 72:
            series = np.zeros(n, dtype=float)
            pos = 10000.0
            for di, ds in enumerate(range(0, n, 24)):
                de = min(ds + 24, n)
                step = float(np.random.uniform(2.0, 8.0))
                if di % 2 == 0:
                    step = float(params["T_daily_ops_max"]) + float(np.random.uniform(5.0, 25.0))
                pos += step
                series[ds:de] = pos + np.linspace(0, 0.5, de - ds)
            df.loc[idx, "oltc_operation_count"] = series
            return InjectResult("K16", True, "daily ops + durable gas_pair")
    # total30 (default) / short-window fallback
    idx = _hours_idx(df, ts, 30 * 24)
    if len(idx) < 2:
        df.loc[ts, "oltc_operation_count"] = float(params["T_ops_warn"]) + 50.0
        return InjectResult("K16", True, "fallback ops + durable gas_pair")
    start_v = 1000.0
    end_v = start_v + float(params["T_ops_warn"]) + float(np.random.uniform(20, 120))
    df.loc[idx, "oltc_operation_count"] = np.linspace(start_v, end_v, len(idx))
    return InjectResult("K16", True, "30d ops rise + durable gas_pair")


def inject_k17(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k17: 180d H2 slope OR 180d oil_level slope down."""
    if np.random.rand() < 0.5:
        ensure_columns(df, ["oil_dga_h2"])
        inject_slope_increasing(df, ts, "oil_dga_h2", 180, slope_per_day=params["S_H2_standby"] * 1.5, base=40.0)
        return InjectResult("K17", True, "180d H2 upward slope")
    inject_slope_decreasing(df, ts, "oil_level", window_days=180, slope_per_day_target=float(params.get("S_oil_standby", -0.05)) * 1.3)
    return InjectResult("K17", True, "180d oil_level downward slope")


def inject_k18(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k18: 7d PD or 7d tanδ vs grid voltage (500/750)."""
    gv = "500kV"
    if "grid_side_voltage" in df.columns:
        try:
            raw = str(df["grid_side_voltage"].loc[ts]).strip()
            if "750" in raw:
                gv = "750kV"
        except Exception:
            pass
    elif df.attrs.get("grid_side_voltage"):
        raw = str(df.attrs.get("grid_side_voltage")).strip()
        if "750" in raw:
            gv = "750kV"

    if np.random.rand() < 0.55:
        ensure_columns(df, ["pd_level"])
        idx = _hours_idx(df, ts, 7 * 24)
        if len(idx) == 0:
            return InjectResult("K18", False, "no 7d window for PD")
        if gv == "750kV":
            tgt = float(params["T_PD_750kV"]) + float(np.random.uniform(5, 40))
        else:
            tgt = float(params["T_PD_500kV"]) + float(np.random.uniform(5, 40))
        df.loc[idx, "pd_level"] = tgt
        return InjectResult("K18", True, f"7d PD high ({gv})")

    ensure_columns(df, ["rel_change_tan"])
    idx = _hours_idx(df, ts, 7 * 24)
    if len(idx) == 0:
        return InjectResult("K18", False, "no 7d window for tan")
    if gv == "750kV":
        tgt = float(params["T_tanδ_750kV"]) + 0.001
    else:
        tgt = float(params["T_tanδ_500kV"]) + 0.001
    df.loc[idx, "rel_change_tan"] = tgt
    return InjectResult("K18", True, f"7d tanδ high ({gv})")


def inject_k19(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    ensure_columns(df, ["oil_dga_c2h2", "oil_dga_c2h4"])
    r = float(params.get("R_discharge", 0.1))
    c4 = max(1.0, float(np.random.uniform(15.0, 40.0)))
    df.loc[ts, "oil_dga_c2h4"] = c4
    df.loc[ts, "oil_dga_c2h2"] = c4 * (r + 0.05) + float(np.random.uniform(0.5, 3.0))
    return InjectResult("K19", True, "c2h2/c2h4 > R_discharge")


def inject_k20(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    ensure_columns(df, ["oil_dga_c2h2", "oil_dga_c2h4", "oil_dga_total_hydrocarbon"])
    rlow = float(params.get("R_low", 0.01))
    c4 = 80.0
    df.loc[ts, "oil_dga_c2h4"] = c4
    df.loc[ts, "oil_dga_c2h2"] = c4 * (rlow * 0.5)
    df.loc[ts, "oil_dga_total_hydrocarbon"] = float(params.get("T_total_hydrocarbon", 150.0)) + float(np.random.uniform(5, 40))
    return InjectResult("K20", True, "low ratio + high total_hydrocarbon")


def _set_winding_rise_components(df: pd.DataFrame, ts: pd.Timestamp, rise: float) -> None:
    """winding_hotspot_temp - cooling_water_temp == rise (also write derived)."""
    ensure_columns(
        df,
        ["winding_hotspot_temp", "cooling_water_temp", "winding_hotspot_temperature_rise"],
    )
    cw = 28.0
    df.loc[ts, "cooling_water_temp"] = cw
    df.loc[ts, "winding_hotspot_temp"] = cw + rise
    df.loc[ts, "winding_hotspot_temperature_rise"] = float(rise)


def _sync_oil_ambient(df: pd.DataFrame, idx) -> None:
    ensure_columns(df, ["oil_to_ambient_temperature_diff", "top_oil_temp", "cooling_water_temp"])
    top = pd.to_numeric(df.loc[idx, "top_oil_temp"], errors="coerce")
    cw = pd.to_numeric(df.loc[idx, "cooling_water_temp"], errors="coerce")
    df.loc[idx, "oil_to_ambient_temperature_diff"] = top - cw


def _sync_winding_rise(df: pd.DataFrame, idx) -> None:
    ensure_columns(df, ["winding_hotspot_temperature_rise", "winding_hotspot_temp", "cooling_water_temp"])
    wh = pd.to_numeric(df.loc[idx, "winding_hotspot_temp"], errors="coerce")
    cw = pd.to_numeric(df.loc[idx, "cooling_water_temp"], errors="coerce")
    df.loc[idx, "winding_hotspot_temperature_rise"] = wh - cw


def inject_k21(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    if np.random.rand() < 0.6:
        hours = int(np.random.choice([3, 4, 5]))
        idx = _hours_idx(df, ts, hours)
        rise = float(params["T_rise_alarm"]) + float(np.random.uniform(1.0, 6.0))
        for txi in idx:
            _set_winding_rise_components(df, txi, rise)
        return InjectResult("K21", True, f"consec {hours}h rise > T_rise_alarm")
    rise = float(params["T_rise_trip"]) + float(np.random.uniform(1.0, 5.0))
    _set_winding_rise_components(df, ts, rise)
    return InjectResult("K21", True, "single point rise > T_rise_trip")


def inject_k22(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    wmin, wmax = float(params["T_water_min"]), float(params["T_water_max"])
    wm = float(np.random.uniform(wmin + 1.0, wmax - 1.0)) if wmax > wmin + 2 else (wmin + wmax) / 2.0
    ensure_columns(df, ["cooling_water_temp", "winding_hotspot_temp"])
    df.loc[ts, "cooling_water_temp"] = wm
    df.loc[ts, "winding_hotspot_temp"] = wm + float(params["T_rise_high"]) + float(np.random.uniform(1.0, 8.0))
    _sync_winding_rise(df, [ts])
    return InjectResult("K22", True, "rise high with water in band")


def inject_k23(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    hours = 3
    idx = _hours_idx(df, ts, hours)
    ensure_columns(df, ["top_oil_temp", "cooling_water_temp"])
    cw = 22.0
    df.loc[idx, "cooling_water_temp"] = cw
    need = float(params["T_diff_high"]) + float(np.random.uniform(0.3, 1.2))
    df.loc[idx, "top_oil_temp"] = cw + need
    _sync_oil_ambient(df, idx)
    return InjectResult("K23", True, f"consec {len(idx)}h high oil-ambient diff (soft)")


def inject_k24(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """Low oil-ambient consec; soft margins to limit K25 lookback bleed."""
    hours = 3
    idx = _hours_idx(df, ts, hours)
    ensure_columns(df, ["top_oil_temp", "cooling_water_temp"])
    cw = 35.0
    df.loc[idx, "cooling_water_temp"] = cw
    low = float(params["T_diff_low"]) - float(np.random.uniform(0.2, 0.8))
    df.loc[idx, "top_oil_temp"] = cw + low
    _sync_oil_ambient(df, idx)
    return InjectResult("K24", True, f"consec {len(idx)}h low oil-ambient diff (soft)")


def inject_k25(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    idx = _hours_idx(df, ts, 24)
    if len(idx) < 8:
        return InjectResult("K25", False, "need 24h for std")
    ensure_columns(df, ["top_oil_temp", "cooling_water_temp"])
    df.loc[idx, "cooling_water_temp"] = 20.0
    amp = max(12.0, float(params.get("T_diff_std", 8.0)) * 1.8)
    t = np.arange(len(idx), dtype=float)
    df.loc[idx, "top_oil_temp"] = 45.0 + amp * np.sin(2 * np.pi * t / 4.0)
    _sync_oil_ambient(df, idx)
    return InjectResult("K25", True, "24h oscillating diff → high std")


def inject_k26(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    hours = int(np.random.choice([3, 4, 5]))
    idx = _hours_idx(df, ts, hours)
    ensure_columns(df, ["transformer_body_pressure", "conservator_pressure"])
    pmax = float(params["P_diff_max"])
    gap = float(np.random.uniform(0.5, 4.0))
    df.loc[idx, "conservator_pressure"] = 50.0
    df.loc[idx, "transformer_body_pressure"] = 50.0 + pmax + gap
    return InjectResult("K26", True, f"abs(body-cons) > P_diff_max, {hours}h")


def inject_k27(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    hours = int(np.random.choice([3, 4, 5]))
    idx = _hours_idx(df, ts, hours)
    ensure_columns(df, ["transformer_body_pressure", "conservator_pressure"])
    pmax = float(params["P_diff_max"])
    gap = float(np.random.uniform(0.5, 4.0))
    df.loc[idx, "transformer_body_pressure"] = 45.0
    df.loc[idx, "conservator_pressure"] = 45.0 + pmax + gap
    return InjectResult("K27", True, f"body-cons < -P_diff_max, {hours}h")


def inject_k28(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    ensure_columns(df, ["oil_dga_co2", "oil_dga_co"])
    ratio_tgt = float(params["T_ratio_high"]) + float(np.random.uniform(0.2, 2.0))
    co = max(1.0, float(np.random.uniform(5.0, 25.0)))
    df.loc[ts, "oil_dga_co"] = co
    df.loc[ts, "oil_dga_co2"] = co * ratio_tgt
    return InjectResult("K28", True, "CO2/CO > T_ratio_high")


def inject_k29(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    ensure_columns(df, ["oil_dga_co2", "oil_dga_co"])
    rlo = float(params["T_ratio_low"])
    co2 = 2000.0
    df.loc[ts, "oil_dga_co2"] = co2
    df.loc[ts, "oil_dga_co"] = co2 / max(0.01, rlo * 0.5)
    return InjectResult("K29", True, "CO2/CO < T_ratio_low")


def inject_k30(df: pd.DataFrame, ts: pd.Timestamp, params: Dict[str, float]) -> InjectResult:
    """rule_k30: slope_per_day(dga_ratio_co2_co) > S_ratio_slope over 30d (ratio = CO2/CO)."""
    hours = 30 * 24
    idx_full = pd.date_range(end=ts, periods=hours, freq="h")
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        idx = idx_full[(idx_full >= df.index.min()) & (idx_full <= df.index.max())]
    else:
        idx = idx_full
    if len(idx) < 24:
        return InjectResult("K30", False, "insufficient 30d history")
    ensure_columns(df, ["oil_dga_co2", "oil_dga_co"])
    tt = np.linspace(0.0, 1.0, len(idx))
    # Rising CO2 with slowly falling CO → ratio trend up strongly
    df.loc[idx, "oil_dga_co2"] = 2000.0 + 500.0 * tt
    df.loc[idx, "oil_dga_co"] = np.maximum(15.0, 120.0 * (1.0 - 0.88 * tt))
    return InjectResult("K30", True, "30d rising CO2/CO ratio (co2↑, co↓)")


def _consec_3h(cond: pd.Series) -> bool:
    s = cond.fillna(False).astype(bool)
    return bool(len(s) >= 3 and s.iloc[-3:].all())


def or_branch_warnings_0002(df: pd.DataFrame, ts_end: pd.Timestamp, rid: str, params: Dict[str, float]) -> List[str]:
    rid = str(rid).upper()
    warns: List[str] = []
    t = pd.Timestamp(ts_end)
    idx3 = df.loc[t - pd.Timedelta(hours=2) : t].index

    if rid == "K1" and {"T_alarm", "T_trip"} <= set(params.keys()) and "top_oil_temp" in df.columns:
        s = pd.to_numeric(df["top_oil_temp"], errors="coerce")
        if _consec_3h(s.loc[idx3] > float(params["T_alarm"])) and bool(s.loc[t] > float(params["T_trip"])):
            warns.append("OR branch co-trigger: K1 alarm and trip branches both satisfied")
    elif rid == "K2" and {"T_alarm", "T_trip"} <= set(params.keys()) and "winding_hotspot_temp" in df.columns:
        s = pd.to_numeric(df["winding_hotspot_temp"], errors="coerce")
        if _consec_3h(s.loc[idx3] > float(params["T_alarm"])) and bool(s.loc[t] > float(params["T_trip"])):
            warns.append("OR branch co-trigger: K2 alarm and trip branches both satisfied")
    elif rid == "K3" and {"L_min", "L_max"} <= set(params.keys()) and "oil_level" in df.columns:
        s = pd.to_numeric(df["oil_level"], errors="coerce")
        low = _consec_3h(s.loc[idx3] < float(params["L_min"]))
        high = _consec_3h(s.loc[idx3] > float(params["L_max"]))
        if low and high:
            warns.append("OR branch co-trigger: K3 low-level and high-level branches both satisfied")
    elif rid == "K9" and {"P_min", "P_max"} <= set(params.keys()):
        cols_chk = [c for c in ("transformer_body_pressure", "conservator_pressure", "bushing_pressure") if c in df.columns]
        if cols_chk:
            both = 0
            for c in cols_chk:
                s = pd.to_numeric(df[c], errors="coerce")
                if _consec_3h(s.loc[idx3] < float(params["P_min"])) and _consec_3h(s.loc[idx3] > float(params["P_max"])):
                    both += 1
            if both > 0:
                warns.append("OR branch may co-trigger: K9 pressure channels satisfy low/high branches")
    elif rid == "K10" and {"T_attention", "T_background", "K"} <= set(params.keys()) and "pd_level" in df.columns:
        s = pd.to_numeric(df["pd_level"], errors="coerce")
        b1 = _consec_3h(s.loc[idx3] > float(params["T_attention"]))
        w24 = pd.to_numeric(df.loc[t - pd.Timedelta(hours=23) : t, "pd_level"], errors="coerce")
        if len(w24):
            m24 = float(w24.mean())
            b2 = np.isfinite(m24) and m24 > float(params["T_background"]) * float(params["K"])
            cur = float(s.loc[t]) if t in s.index else float("nan")
            b3 = np.isfinite(m24) and np.isfinite(cur) and m24 != 0 and abs((cur - m24) / m24) > float(params["T_rate"]) / 100.0
            if sum([bool(b1), bool(b2), bool(b3)]) >= 2:
                warns.append("OR branch co-trigger: K10 multiple branches (consec/24h mean/rate) satisfied")

    return warns


# Skip long-lookback rules whose injects routinely bleed into audit background samples
# (7d PD/tan, 24h std, 30d CO2/CO slope). Functions remain for manual re-enable.
_SKIP_INJECT_RULES = frozenset({"K18", "K23", "K24", "K25", "K30"})

# Explicit registry (generator also discovers inject_k* via dir(); this documents parity with RULE_REGISTRY K1–K30)
INJECT_REGISTRY: Dict[str, Any] = {
    f"K{i}": globals()[f"inject_k{i}"]
    for i in range(1, 31)
    if f"inject_k{i}" in globals() and f"K{i}" not in _SKIP_INJECT_RULES
}
