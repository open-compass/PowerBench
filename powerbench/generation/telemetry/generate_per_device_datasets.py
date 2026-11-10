from __future__ import annotations
import argparse
import importlib.util
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from powerbench.generation.telemetry.strict_injector import inject_to_trigger
from powerbench.generation.telemetry.strict_rule_engine import compile_rule, eval_compiled_rule, eval_rule_strict, load_stage2
from powerbench.generation.telemetry.xlsx_columns import row_get


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _split_semicolon(s: str) -> List[str]:
    s2 = _compact_str(s)
    if not s2:
        return []
    parts = [p.strip() for p in s2.split(";")]
    return [p for p in parts if p]


def _safe_fs_stem(name: str, *, max_len: int = 160) -> str:
    s = _compact_str(name)
    if not s:
        return "device"
    for ch in '\\/:*?"<>|':
        s = s.replace(ch, "_")
    s = re.sub(r"\s+", "_", s).strip("._")
    if len(s) > max_len:
        s = s[:max_len].rstrip("._")
    return s or "device"


def _pick_device_allowed_rules(row: pd.Series, *, rules_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Decide which rules are applicable for this device **from xlsx only**.

    Priority:
    1) xlsx column: applicable_rule_ids_all (semicolon-separated Kxx list)
    2) fallback: all rules from json (for legacy xlsx that doesn't have the column)
    """
    applicable_rules = _split_semicolon(_compact_str(row_get(row, "applicable_rule_ids_all")))
    if applicable_rules:
        return applicable_rules
    return sorted(rules_by_id.keys())


def _pick_device_allowed_db_fields(
    row: pd.Series,
    *,
    cn2db: Dict[str, str],
) -> List[str]:
    """
    Decide which db fields this device "owns" **from xlsx only**.

    Priority:
    1) xlsx column: maintenance_metric_db_fields_all (semicolon-separated db_field list)
    2) xlsx column: maintenance_metrics_all (semicolon-separated metric names -> map via cn2db)
    3) fallback: [] (unknown)
    """
    dbs = _split_semicolon(_compact_str(row_get(row, "maintenance_metric_db_fields_all")))
    if dbs:
        return list(dict.fromkeys([x for x in dbs if x]))

    want_cn = _split_semicolon(_compact_str(row_get(row, "maintenance_metrics_all")))
    want_db: List[str] = []
    for cn in want_cn:
        db = cn2db.get(cn)
        if db:
            want_db.append(db)
    return list(dict.fromkeys([x for x in want_db if x]))


def _mask_df_to_allowed_fields(df: pd.DataFrame, *, allowed_db_fields: List[str]) -> pd.DataFrame:
    """
    Mask df columns not in allowed_db_fields by setting them to NaN.
    Keep timestamp index unchanged.

    Note: we do NOT drop columns to avoid breaking rule/injector code that expects columns exist;
    instead we mask values to "not available".
    """
    if not allowed_db_fields:
        return df
    allow = set(allowed_db_fields)
    out = df.copy()
    for c in list(out.columns):
        if c not in allow:
            # Keep object/string columns as None, numeric as NaN
            try:
                if pd.api.types.is_numeric_dtype(out[c].dtype):
                    out[c] = np.nan
                else:
                    out[c] = None
            except Exception:
                out[c] = np.nan
    return out


def _hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq="h")


def _extract_required_windows(expr: str) -> List[str]:
    s = str(expr or "")
    wins: set[str] = set()
    for pat, tag in [
        ("duration_48h", "48h"),
        ("duration_24h", "24h"),
        ("duration_12h", "12h"),
        ("duration_6h", "6h"),
        ("duration_1h", "1h"),
        ("consec_6h", "6h"),
        ("consec_3h", "3h"),
        ("count_24h", "24h"),
        ("count_12h", "12h"),
        ("count_6h", "6h"),
        ("count_1h", "1h"),
        ("avg_24h", "24h"),
        ("avg_48h", "48h"),
        ("std_24h", "24h"),
        ("std_12h", "12h"),
        ("avg_6h", "6h"),
        ("avg_12h", "12h"),
        ("max_6h", "6h"),
        ("max_12h", "12h"),
        ("std_6h", "6h"),
        ("std_1h", "1h"),
        ("avg_1h", "1h"),
        ("max_1h", "1h"),
        ("avg_7d", "7d"),
        ("max_7d", "7d"),
        ("std_30d", "30d"),
        ("slope_30d", "30d"),
        ("slope_90d", "90d"),
        ("std_90d", "90d"),
        ("slope_180d", "180d"),
        ("slope_365d", "365d"),
        ("count_365d", "365d"),
    ]:
        if pat in s:
            wins.add(tag)

    def _key(w: str) -> int:
        if w.endswith("h"):
            return int(w[:-1])
        if w.endswith("d"):
            return int(w[:-1]) * 24
        return 0

    return sorted(wins, key=_key)


def _win_to_hours(win: str) -> int:
    w = _compact_str(win)
    if not w:
        return 0
    if w.endswith("h"):
        try:
            return int(w[:-1])
        except Exception:
            return 0
    if w.endswith("d"):
        try:
            return int(w[:-1]) * 24
        except Exception:
            return 0
    return 0


def _max_required_hours(rule_obj: Dict[str, Any]) -> int:
    wins = _extract_required_windows(_compact_str(rule_obj.get("rule_expression")))
    return max([_win_to_hours(w) for w in wins], default=0)


def _has_enough_history(idx: pd.DatetimeIndex, ts_end: pd.Timestamp, required_hours: int) -> bool:
    if required_hours <= 0:
        return True
    if len(idx) < required_hours:
        return False
    start_need = ts_end - pd.Timedelta(hours=required_hours - 1)
    return (start_need >= idx.min()) and (ts_end <= idx.max())


def _window_index(ts: pd.Timestamp, win: str) -> pd.DatetimeIndex:
    if win == "0h":
        return pd.DatetimeIndex([ts])
    if win.endswith("h"):
        n = int(win[:-1])
        return pd.date_range(end=ts, periods=n, freq="h")
    if win.endswith("d"):
        d = int(win[:-1])
        return pd.date_range(end=ts, periods=d * 24, freq="h")
    return pd.DatetimeIndex([ts])


@dataclass
class Scenario:
    rule_id: str
    window_start: str
    window_end: str
    note_cn: str


def _build_fault_mask(idx: pd.DatetimeIndex, scenarios: List[Scenario]) -> pd.Series:
    mask = pd.Series(False, index=idx)
    for sc in scenarios:
        mask.loc[pd.Timestamp(sc.window_start) : pd.Timestamp(sc.window_end)] = True
    return mask


def _pick_non_overlapping_window(
    *,
    idx: pd.DatetimeIndex,
    occupied_mask: pd.Series,
    hours: int,
    rng: random.Random,
    avoid_edges_hours: int = 48,
) -> Tuple[pd.Timestamp, pd.Timestamp] | None:
    if hours <= 0:
        return None
    if len(idx) < hours + 2 * avoid_edges_hours:
        avoid_edges_hours = 0
    candidates = idx[avoid_edges_hours : max(avoid_edges_hours, len(idx) - avoid_edges_hours)]
    if len(candidates) < hours:
        return None
    for _ in range(2000):
        end = pd.Timestamp(rng.choice(list(candidates)))
        win = pd.date_range(end=end, periods=hours, freq="h")
        start = pd.Timestamp(win[0])
        end2 = pd.Timestamp(win[-1])
        if bool(occupied_mask.loc[start:end2].any()):
            continue
        return start, end2
    return None


def _is_numeric_series(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(s.dtype)
    except Exception:
        return False


def _load_json_list(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    # register in sys.modules to support dataclasses/type annotations inside module
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _default_xlsx_path(model_dir: Path) -> Optional[Path]:
    # Prefer *with_metrics.xlsx if present, else any .xlsx at root.
    cands = sorted(model_dir.glob("*_with_metrics.xlsx"))
    if cands:
        return cands[0]
    cands2 = sorted(model_dir.glob("*.xlsx"))
    return cands2[0] if cands2 else None


def generate_background(df: pd.DataFrame, *, model_dir: Path) -> None:
    """
    Shared *fallback* background generator.

    Important: this function must NOT assume any device-specific column names.
    It only uses `metric_catalog.json` (database_field/data_type) to fill columns.

    For realistic, device-specific backgrounds, provide a model-side implementation:
      <model_dir>/pipeline_generate/background_generator.py
    defining:
      def generate_background(df: pd.DataFrame, *, model_dir: Path) -> None
    """
    idx = df.index
    metric_catalog = _load_json_list(model_dir / "metric_catalog.json")

    def _should_export_field(field: str) -> bool:
        f = _compact_str(field)
        if not f:
            return False
        bad_prefix = ("rel_change_", "change_rate", "pd_change_rate", "duration_", "avg_", "std_", "slope_", "count_")
        if any(f.startswith(p) for p in bad_prefix):
            return False
        if any(tok in f for tok in ["_1h", "_3h", "_24h", "_7d", "_30d", "_90d", "_180d", "_365d"]):
            return False
        return True

    def _is_all_missing(col: str) -> bool:
        if col not in df.columns:
            return True
        s = df[col]
        try:
            if s.dtype == "object":
                ss = s.astype("object")
                return bool(ss.isna().all() or (ss.astype(str).str.strip() == "").all())
        except Exception:
            pass
        try:
            return bool(pd.isna(s).all())
        except Exception:
            return True

    def _fill_numeric(col: str, base: float, sigma: float, lo: float | None = None, hi: float | None = None) -> None:
        if col in df.columns and not _is_all_missing(col):
            return
        x = np.random.normal(loc=base, scale=sigma, size=len(idx)).astype(float)
        if lo is not None:
            x = np.maximum(x, lo)
        if hi is not None:
            x = np.minimum(x, hi)
        df[col] = x

    def _fill_discrete(col: str, default: str = "running") -> None:
        if col in df.columns and not _is_all_missing(col):
            return
        df[col] = default

    for it in metric_catalog:
        if not isinstance(it, dict):
            continue
        col = _compact_str(it.get("database_field"))
        dtype = _compact_str(it.get("data_type"))
        if not col or not _should_export_field(col):
            continue
        if dtype == "string":
            _fill_discrete(col, default="running")
            continue
        # Heuristic fill by name patterns (generic, not device-specific)
        if "pressure" in col:
            _fill_numeric(col, base=0.65, sigma=0.01, lo=0.2, hi=1.2)
        elif "temp" in col:
            _fill_numeric(col, base=60.0, sigma=2.0, lo=-30.0, hi=130.0)
        elif "oil_dga_" in col:
            _fill_numeric(col, base=10.0, sigma=2.0, lo=0.0, hi=5000.0)
        else:
            _fill_numeric(col, base=1.0, sigma=0.05)


def _load_background_generator(model_dir: Path):
    """
    Try to load model-specific background generator:
      <model_dir>/pipeline_generate/background_generator.py
    """
    bg_py = model_dir / "pipeline_generate" / "background_generator.py"
    if not bg_py.exists():
        return None
    mod = _load_module_from_path(f"background_generator_{abs(hash(str(bg_py)))}", bg_py)
    fn = getattr(mod, "generate_background", None)
    return fn if callable(fn) else None


def _fields_override_from_device_row(row: pd.Series) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    if "grid_side_voltage" in row.index:
        v = _compact_str(row.get("grid_side_voltage"))
        if v:
            fields["grid_side_voltage"] = v
    return fields


def _inject_near_miss_generic_strict(
    *,
    folder: Path,
    df: pd.DataFrame,
    ts_end: pd.Timestamp,
    rule_obj: Dict[str, Any],
    compiled: Any,
    fields_override: Dict[str, Any],
    seed: int,
    backoff_steps: int = 14,
) -> Dict[str, Any]:
    rid = _compact_str(rule_obj.get("rule_id"))
    rec: Dict[str, Any] = {"rule_id": rid, "window_end": ts_end.strftime("%Y-%m-%d %H:%M:%S")}
    required_hours = _max_required_hours(rule_obj)
    if required_hours > 0 and not _has_enough_history(df.index, ts_end, required_hours):
        rec.update(
            {
                "skipped": True,
                "required_hours": int(required_hours),
                "note_cn": "skip near-miss: required time window longer than available dataset/history at this timestamp",
            }
        )
        return rec

    try:
        if eval_compiled_rule(folder, df, ts_end, rule_obj, compiled, fields_override=fields_override):
            rec.update({"skipped": True, "note_cn": "background already triggers; skip near-miss"})
            return rec
    except Exception:
        pass

    df0 = df.copy()
    df1 = df.copy()
    rep = inject_to_trigger(folder, df1, ts_end, rule_obj, prefer_random_or=True, seed=seed)
    if not getattr(rep, "injected", False):
        rec.update({"skipped": True, "note_cn": f"strict_injector failed: {getattr(rep, 'notes', '')}"})
        return rec

    idx = df.index
    start = max(idx.min(), ts_end - pd.Timedelta(hours=24 * 30))
    win = idx[(idx >= start) & (idx <= ts_end)]

    changed_cols: List[str] = []
    for c in df.columns:
        if c not in df1.columns:
            continue
        try:
            s0 = df0.loc[win, c]
            s1 = df1.loc[win, c]
        except Exception:
            continue
        if not _is_numeric_series(s0) or not _is_numeric_series(s1):
            continue
        if (s0.fillna(np.nan) != s1.fillna(np.nan)).any():
            changed_cols.append(c)

    if not changed_cols:
        rec.update({"skipped": True, "note_cn": "only discrete/non-numeric changes; near-miss skipped"})
        return rec

    lo = 0.0
    hi = 1.0
    best_alpha = None
    for _ in range(backoff_steps):
        mid = (lo + hi) / 2.0
        df_mid = df0.copy()
        for c in changed_cols:
            a = df0.loc[win, c].astype(float)
            b = df1.loc[win, c].astype(float)
            df_mid.loc[win, c] = a + (b - a) * mid
        try:
            trig = eval_compiled_rule(folder, df_mid, ts_end, rule_obj, compiled, fields_override=fields_override)
        except Exception:
            trig = True
        if trig:
            hi = mid
        else:
            best_alpha = mid
            lo = mid

    if best_alpha is None:
        rec.update({"skipped": True, "note_cn": "could not find non-triggering backoff"})
        return rec

    for c in changed_cols:
        a = df0.loc[win, c].astype(float)
        b = df1.loc[win, c].astype(float)
        df.loc[win, c] = a + (b - a) * best_alpha

    rec.update(
        {
            "note_cn": f"generic near-miss via strict_injector trigger+backoff (alpha~{best_alpha:.3f})",
            "changed_cols": changed_cols[:30],
        }
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="(Shared) Generate per-device datasets for a model folder.")
    ap.add_argument("--model_dir", type=str, required=True, help="Path to a model folder under models/")
    ap.add_argument("--xlsx_path", type=str, default="", help="Override device list xlsx path")
    ap.add_argument("--sheet", type=str, default="device_list")
    ap.add_argument("--device_col", type=str, default="device_name")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_scenarios", type=int, default=5)
    ap.add_argument("--near_miss_count", type=int, default=8)
    ap.add_argument("--near_miss_cluster_hours", type=int, default=24)
    ap.add_argument("--near_miss_seed", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--start_time", type=str, default="2025-01-01 00:00:00")
    ap.add_argument("--end_time", type=str, default="2025-03-31 23:00:00")
    args, _unknown = ap.parse_known_args()

    model_dir = Path(_compact_str(args.model_dir)).resolve()
    if not model_dir.exists():
        raise SystemExit(f"model_dir not found: {model_dir}")

    # load model-specific modules from file paths (robust even with Chinese/± folder names)
    pg_dir = model_dir / "pipeline_generate"
    scen_py = pg_dir / "scenario_injectors.py"
    if not scen_py.exists():
        raise SystemExit(f"missing scenario_injectors.py: {scen_py}")
    scenario_injectors = _load_module_from_path(f"scenario_injectors_{abs(hash(str(scen_py)))}", scen_py)

    rule_py = pg_dir / "rule_functions.py"
    rule_functions = None
    if rule_py.exists():
        rule_functions = _load_module_from_path(f"rule_functions_{abs(hash(str(rule_py)))}", rule_py)

    bg_gen = _load_background_generator(model_dir)

    # params
    params_by_rule: Dict[str, Dict[str, float]] = {}
    if rule_functions is not None and hasattr(rule_functions, "load_params_by_rule"):
        params_by_rule = rule_functions.load_params_by_rule(model_dir)  # type: ignore[assignment]

    xlsx_path = Path(_compact_str(args.xlsx_path)).resolve() if _compact_str(args.xlsx_path) else (_default_xlsx_path(model_dir) or None)
    if not xlsx_path or not xlsx_path.exists():
        raise SystemExit(f"missing xlsx (use --xlsx_path): {xlsx_path}")

    df_dev = pd.read_excel(xlsx_path, sheet_name=str(args.sheet))
    if str(args.device_col) not in df_dev.columns:
        raise SystemExit(f"xlsx missing device column={args.device_col!r} in sheet={args.sheet!r}")

    stage2 = load_stage2(model_dir)
    rules_by_id: Dict[str, Dict[str, Any]] = {}
    for r in stage2.get("anomaly_judgment_rules", []) or []:
        if isinstance(r, dict):
            rid = _compact_str(r.get("rule_id"))
            if rid:
                rules_by_id[rid] = r
    compiled_by_id: Dict[str, Any] = {rid: compile_rule(model_dir, rule) for rid, rule in rules_by_id.items()}

    metric_catalog = _load_json_list(model_dir / "metric_catalog.json")
    cn2db: Dict[str, str] = {}
    for m in metric_catalog:
        cn = _compact_str(m.get("metric_name")) if isinstance(m, dict) else ""
        db = _compact_str(m.get("database_field")) if isinstance(m, dict) else ""
        if cn and db:
            cn2db[cn] = db

    idx = _hourly_index(str(args.start_time), str(args.end_time))
    out_dir = model_dir / "per_device"
    out_dir.mkdir(exist_ok=True)

    start_i = max(0, int(args.start))
    end_i = len(df_dev) if int(args.limit) <= 0 else min(len(df_dev), start_i + int(args.limit))
    max_scenarios = max(0, int(args.max_scenarios))
    base_seed = int(args.seed)
    near_miss_count = max(0, int(args.near_miss_count))
    near_miss_seed = int(args.near_miss_seed)
    near_miss_cluster_hours = max(0, int(args.near_miss_cluster_hours))

    # injector supported: discover inject_* functions in scenario_injectors
    injector_supported = {name.replace("inject_", "").upper() for name in dir(scenario_injectors) if name.startswith("inject_")}
    injector_supported = {rid for rid in injector_supported if rid.startswith("K")}

    # default scenario windows (can be extended later)
    scenario_defaults: Dict[str, Tuple[str, str, str]] = {
        "K1": ("2025-03-30 21:00:00", "2025-03-30 23:00:00", "rule triggered"),
        "K2": ("2025-03-28 10:00:00", "2025-03-28 12:00:00", "rule triggered"),
        "K3": ("2025-03-20 08:00:00", "2025-03-20 10:00:00", "rule triggered"),
        "K4": ("2025-03-15 12:00:00", "2025-03-15 12:00:00", "rule triggered"),
        "K6": ("2025-03-10 06:00:00", "2025-03-10 08:00:00", "rule triggered"),
        "K7": ("2025-02-10 08:00:00", "2025-02-10 08:00:00", "rule triggered"),
        "K8": ("2025-02-12 09:00:00", "2025-02-12 09:00:00", "rule triggered"),
        "K12": ("2025-03-31 21:00:00", "2025-03-31 23:00:00", "rule triggered"),
    }

    def _inject_one(df: pd.DataFrame, sc: Scenario) -> None:
        fn_name = f"inject_{sc.rule_id.lower()}"
        if hasattr(scenario_injectors, fn_name):
            fn = getattr(scenario_injectors, fn_name)
            ts = pd.Timestamp(sc.window_end)
            p = params_by_rule.get(sc.rule_id, {})
            fn(df, ts, p)

    for i in range(start_i, end_i):
        row = df_dev.iloc[i]
        device_name = _compact_str(row.get(str(args.device_col)))
        if not device_name:
            continue

        # ---- constraints from xlsx (rules + owned metrics) ----
        applicable_rules = _pick_device_allowed_rules(row, rules_by_id=rules_by_id)
        applicable_set = set(applicable_rules)
        allowed_db_fields = _pick_device_allowed_db_fields(row, cn2db=cn2db)

        eligible = [rid for rid in applicable_rules if rid in injector_supported]
        rng = random.Random(base_seed + i * 977)
        rng.shuffle(eligible)
        chosen = eligible[: max_scenarios] if max_scenarios > 0 else []

        scenarios: List[Scenario] = []
        for rid in chosen:
            ws, we, note = scenario_defaults.get(rid, (str(idx[-1]), str(idx[-1]), f"{rid} rule triggered"))
            scenarios.append(Scenario(rid, ws, we, note))

        fault_mask = _build_fault_mask(idx, scenarios)
        fields_override = _fields_override_from_device_row(row)

        rules_list = [rules_by_id[rid] for rid in applicable_rules if rid in rules_by_id]
        compiled_list = [compiled_by_id[rid] for rid in applicable_rules if rid in compiled_by_id]

        # background search
        max_attempts = 30
        df_bg: Optional[pd.DataFrame] = None
        for attempt in range(1, max_attempts + 1):
            np.random.seed(1000 + i * 97 + attempt)
            random.seed(1000 + i * 97 + attempt)
            df_try = pd.DataFrame(index=idx)
            if bg_gen is not None:
                # model-specific background (recommended)
                bg_gen(df_try, model_dir=model_dir)
            else:
                # shared fallback
                generate_background(df_try, model_dir=model_dir)

            # Apply per-device metric availability mask (xlsx-driven)
            df_try = _mask_df_to_allowed_fields(df_try, allowed_db_fields=allowed_db_fields)

            non_fault_idx = idx[~fault_mask.values]
            sample_n = min(220, len(non_fault_idx))
            sample_ts = list(np.random.choice(non_fault_idx.values, size=sample_n, replace=False))
            sample_ts = [pd.Timestamp(x) for x in sample_ts]
            sample_ts += [pd.Timestamp(x) for x in non_fault_idx[-48:].values]
            any_hit = False
            for ts in sample_ts:
                for rule_obj, comp in zip(rules_list, compiled_list):
                    if eval_compiled_rule(model_dir, df_try, ts, rule_obj, comp, fields_override=fields_override):
                        any_hit = True
                        break
                if any_hit:
                    break
            if not any_hit:
                # full strict
                for ts in non_fault_idx:
                    for rule_obj, comp in zip(rules_list, compiled_list):
                        if eval_compiled_rule(model_dir, df_try, ts, rule_obj, comp, fields_override=fields_override):
                            any_hit = True
                            break
                    if any_hit:
                        break
            if not any_hit:
                df_bg = df_try
                break
        if df_bg is None:
            raise SystemExit(f"failed background for device={device_name}")

        df = df_bg

        # near-miss
        near_miss: List[Dict[str, Any]] = []
        near_miss_cluster: Dict[str, Any] = {}
        if near_miss_count > 0:
            nm_rng = random.Random(near_miss_seed + i * 9973)
            occupied = fault_mask.copy()
            nm_rules = [rid for rid in applicable_rules if rid in rules_by_id and rid in compiled_by_id]
            nm_rng.shuffle(nm_rules)
            nm_seq: List[str] = []
            while nm_rules and len(nm_seq) < near_miss_count:
                nm_rng.shuffle(nm_rules)
                nm_seq.extend(nm_rules)
            nm_seq = nm_seq[:near_miss_count]

            cluster_start: Optional[pd.Timestamp] = None
            cluster_end: Optional[pd.Timestamp] = None
            if near_miss_cluster_hours > 0:
                win = _pick_non_overlapping_window(idx=idx, occupied_mask=occupied, hours=near_miss_cluster_hours, rng=nm_rng, avoid_edges_hours=72)
                if win is not None:
                    cluster_start, cluster_end = win
                    near_miss_cluster = {
                        "start": cluster_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "end": cluster_end.strftime("%Y-%m-%d %H:%M:%S"),
                        "hours": int(near_miss_cluster_hours),
                    }

            for rid in nm_seq:
                hours = int(nm_rng.choice([1, 3, 4, 5]))
                if cluster_start is not None and cluster_end is not None:
                    sub_idx = idx[(idx >= cluster_start) & (idx <= cluster_end)]
                    win = _pick_non_overlapping_window(idx=sub_idx, occupied_mask=occupied, hours=hours, rng=nm_rng, avoid_edges_hours=0)
                else:
                    win = _pick_non_overlapping_window(idx=idx, occupied_mask=occupied, hours=hours, rng=nm_rng)
                if win is None:
                    near_miss.append({"rule_id": rid, "skipped": True, "note_cn": "near-miss window not found"})
                    continue
                ws, we = win
                rule_obj = rules_by_id.get(rid) or {}
                comp = compiled_by_id.get(rid)
                rec = _inject_near_miss_generic_strict(
                    folder=model_dir,
                    df=df,
                    ts_end=we,
                    rule_obj=rule_obj,
                    compiled=comp,
                    fields_override=fields_override,
                    seed=near_miss_seed + i * 9973 + int(we.value % 1000003),
                )
                rec["window_start"] = ws.strftime("%Y-%m-%d %H:%M:%S")
                rec["window_end"] = we.strftime("%Y-%m-%d %H:%M:%S")
                near_miss.append(rec)

            # validate
            non_fault_idx = idx[~fault_mask.values]
            any_hit = False
            for ts in non_fault_idx:
                for rule_obj, comp in zip(rules_list, compiled_list):
                    if eval_compiled_rule(model_dir, df, ts, rule_obj, comp, fields_override=fields_override):
                        any_hit = True
                        break
                if any_hit:
                    break
            if any_hit:
                df = df_bg.copy()
                near_miss = [{"skipped": True, "note_cn": "near-miss removed due to strict validation failure"}]
                near_miss_cluster = {}
        else:
            nm_seq = []

        # inject faults
        for sc in scenarios:
            if sc.rule_id not in applicable_set:
                continue
            rule_obj = rules_by_id.get(sc.rule_id) or {}
            req_h = _max_required_hours(rule_obj)
            if req_h > 0 and not _has_enough_history(idx, pd.Timestamp(sc.window_end), req_h):
                sc.note_cn = f"{sc.note_cn} (skipped: required window>{req_h}h exceeds constructed data range/history)"
                continue
            _inject_one(df, sc)

        # report checks (same structure as before)
        checks = []
        for sc in scenarios:
            ts = pd.Timestamp(sc.window_end)
            strict_hits = []
            context_windows: Dict[str, Any] = {}
            for r in rules_list:
                rid = _compact_str(r.get("rule_id"))
                ok, details = eval_rule_strict(model_dir, df, ts, r, fields_override=fields_override)
                if ok:
                    strict_hits.append(rid)
                if rid == sc.rule_id:
                    req_wins = _extract_required_windows(r.get("rule_expression", ""))
                    cols: List[str] = []
                    for cn in (r.get("input_metrics") or []):
                        dbf = cn2db.get(_compact_str(cn))
                        if dbf:
                            cols.append(dbf)
                    rule_ctx: Dict[str, Any] = {}
                    for win in req_wins:
                        if win == "0h":
                            continue
                        idx_win = _window_index(ts, win)
                        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                            if (idx_win[0] < df.index.min()) or (idx_win[-1] > df.index.max()):
                                rule_ctx[win] = {
                                    "window": win,
                                    "start": idx_win[0].strftime("%Y-%m-%d %H:%M:%S"),
                                    "end": idx_win[-1].strftime("%Y-%m-%d %H:%M:%S"),
                                    "series": {},
                                    "skipped": True,
                                    "note_cn": "skip window series: required window exceeds dataset range (insufficient history)",
                                }
                                continue
                        win_ctx: Dict[str, Any] = {
                            "window": win,
                            "start": idx_win[0].strftime("%Y-%m-%d %H:%M:%S"),
                            "end": idx_win[-1].strftime("%Y-%m-%d %H:%M:%S"),
                            "series": {},
                        }
                        for col in cols:
                            if col not in df.columns:
                                continue
                            s = df.loc[idx_win, col]

                            def _to_json_val(v: Any) -> Any:
                                if pd.isna(v):
                                    return None
                                if isinstance(v, (int, float, np.integer, np.floating)):
                                    try:
                                        return float(v)
                                    except Exception:
                                        return None
                                return str(v)

                            win_ctx["series"][col] = [
                                {"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"), "value": _to_json_val(v)}
                                for t, v in zip(s.index, s.values)
                            ]
                        rule_ctx[win] = win_ctx
                    context_windows[rid] = rule_ctx

            checks.append(
                {
                    "device_name": device_name,
                    "timestamp": sc.window_end,
                    "target_rule": sc.rule_id,
                    "scenario_window_start": sc.window_start,
                    "scenario_window_end": sc.window_end,
                    "note_cn": sc.note_cn,
                    "strict_triggered_rules": strict_hits,
                    "context_windows": context_windows,
                }
            )

        safe_name = _safe_fs_stem(device_name)
        rep_path = out_dir / f"{i:04d}_{safe_name}_report.json"
        rep_path.write_text(
            json.dumps(
                {
                    "device_name": device_name,
                    "applicable_rules": applicable_rules,
                    "eligible_rules_with_injector": eligible,
                    "chosen_rules_for_injection": chosen,
                    "scenarios": [sc.__dict__ for sc in scenarios],
                    "near_miss_cluster": near_miss_cluster,
                    "near_miss": near_miss,
                    "near_miss_attempted_rules": nm_seq if near_miss_count > 0 else [],
                    "checks": checks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # export per-device CSV: by maintenance_metrics_all if present, else export all df columns
        want_db = sorted(set([c for c in allowed_db_fields if c in df.columns])) if allowed_db_fields else []

        out_csv = out_dir / f"{i:04d}_{safe_name}.csv"
        export_df = df.copy()
        export_df.insert(0, "timestamp", export_df.index.astype("datetime64[ns]"))
        cols = ["timestamp"] + ([c for c in want_db if c in export_df.columns] if want_db else [c for c in export_df.columns if c != "timestamp"])
        export_df = export_df[cols]
        export_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] ({i+1}/{len(df_dev)}) device={device_name} rules={len(applicable_rules)} scenarios={len(scenarios)} -> {out_csv.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

