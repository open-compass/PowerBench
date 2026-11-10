from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from powerbench.generation.telemetry.generate_per_device_datasets import _safe_fs_stem
from powerbench.generation.telemetry.generate_per_device_datasets import generate_background as _shared_generate_background
from powerbench.generation.telemetry.inspection_schedule import add_inspection_schedule_arguments, build_inspection_days
from powerbench.generation.telemetry.xlsx_columns import read_device_sheet, resolve_column_name, row_get
from powerbench.paths import repo_root


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    return repo_root(Path(__file__))


def _ensure_repo_root_on_syspath() -> None:
    rr = _repo_root()
    if str(rr) not in sys.path:
        sys.path.insert(0, str(rr))


def _load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required pipeline module missing: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _load_pipeline_modules(pipeline_dir: Path):
    """Load model-local bg / rules / injectors from <model_dir>/pipeline_generate."""
    required = ("background_generator.py", "rule_functions.py", "scenario_injectors.py")
    missing = [n for n in required if not (pipeline_dir / n).exists()]
    if missing:
        raise SystemExit(
            f"pipeline_generate incomplete under {pipeline_dir}: missing {missing}. "
            "Implement model-specific rules/injectors (orchestrator is shared)."
        )
    # Prefer built modules over stale stubs
    for n, vmin in (("rule_functions.py", 1500), ("scenario_injectors.py", 1500)):
        path = pipeline_dir / n
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "NotImplementedError" in text and len(text) < 2500:
            raise SystemExit(
                f"{path} looks like a placeholder (NotImplementedError). "
                "Shared orchestrator needs a real model kit first."
            )
        has_registry = bool(re.search(r"^RULE_REGISTRY\s*=", text, flags=re.M))
        has_inject = bool(re.search(r"^def\s+inject_k\d+\b", text, flags=re.M)) or bool(
            re.search(r"^INJECT_REGISTRY\s*=", text, flags=re.M)
        )
        if path.stat().st_size < vmin and not (has_registry or has_inject):
            raise SystemExit(
                f"{path} looks like a stub ({path.stat().st_size} bytes). "
                "Implement model-specific rules/injectors first."
            )
    pkg = f"pipeline_{abs(hash(str(pipeline_dir.resolve()))) % (10**8)}"
    if pkg not in sys.modules:
        pkg_mod = types.ModuleType(pkg)
        pkg_mod.__path__ = [str(pipeline_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg] = pkg_mod
    bg = _load_module(f"{pkg}.background_generator", pipeline_dir / "background_generator.py")
    rf = _load_module(f"{pkg}.rule_functions", pipeline_dir / "rule_functions.py")
    inj = _load_module(f"{pkg}.scenario_injectors", pipeline_dir / "scenario_injectors.py")
    return bg, rf, inj


# --------------------------------------------------------------------------- #
# records / helpers
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioRecord:
    rule_id: str
    window_start: str
    window_end: str
    type: str
    skipped: bool = False
    reason: str = ""
    note_cn: str = ""
    alpha: float | None = None
    changed_cols: List[str] | None = None
    details: Dict[str, Any] | None = None


@dataclass
class Interval:
    start: pd.Timestamp
    end: pd.Timestamp


def _dt(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _fmt(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _salted_rng(*parts: Any) -> random.Random:
    """
    Stable per-(seed, model, device, purpose) RNG.

    Avoids cross-model identical episode_days calendars when CLI --seed is reused
    (anti-homogenization: MODEL_DIR + device_index salt).
    """
    key = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _split_semicolon(s: str) -> List[str]:
    s2 = _compact_str(s)
    if not s2:
        return []
    return [p.strip() for p in s2.replace("；", ";").split(";") if p.strip()]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _dimension_column_names(model_dir: Path, devices_df: pd.DataFrame) -> List[str]:
    """
    Dynamic device dimensions:
      stage1 auxiliary_classification_fields union stage2 applicability_conditions.field_name intersect (xlsx columns union known dims)
    Never hardcodes converter-specific dimensions.
    """
    names: List[str] = []
    stage1 = _read_json(model_dir / "rule_metric_stage1.json")
    if isinstance(stage1, dict):
        for x in stage1.get("auxiliary_classification_fields") or []:
            s = _compact_str(x)
            if s and s not in names:
                names.append(s)
    stage2 = _read_json(model_dir / "rule_metric_stage2.json")
    if isinstance(stage2, dict):
        for rule in stage2.get("anomaly_judgment_rules") or []:
            if not isinstance(rule, dict):
                continue
            for cond in rule.get("applicability_conditions") or []:
                if isinstance(cond, dict):
                    s = _compact_str(cond.get("field_name"))
                    if s and s not in names:
                        names.append(s)
    dims_doc = _read_json(model_dir / "dimensions_summary.json")
    if isinstance(dims_doc, dict):
        for d in dims_doc.get("dimensions") or []:
            if isinstance(d, dict):
                s = _compact_str(d.get("column"))
                if s and s not in names:
                    names.append(s)
    # Keep only columns that exist on the device sheet OR are stage1 auxiliary_classification_fields
    # (injectors may write the latter as attrs even if missing from a sparse sheet).
    present = set(map(str, devices_df.columns))
    stage1_set = set()
    if isinstance(stage1, dict):
        stage1_set = {_compact_str(x) for x in (stage1.get("auxiliary_classification_fields") or [])}
    out = [n for n in names if n in present or n in stage1_set]
    return out


def _fields_override_from_device_row(row: pd.Series, dim_cols: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in dim_cols:
        if col not in row.index:
            continue
        v = _compact_str(row.get(col))
        if v:
            out[col] = v
    return out


def _mask_df_to_allowed_fields(df: pd.DataFrame, *, allowed_db_fields: List[str]) -> None:
    if not allowed_db_fields:
        return
    allow = set(allowed_db_fields)
    for c in list(df.columns):
        if c in allow:
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                df[c] = np.nan
            else:
                df[c] = None
        except Exception:
            df[c] = np.nan


def _details_has_nan(x: Any) -> bool:
    """
    True only for actual float/str NaN in metric/diagnostic values.
    None is intentional for one-sided thresholds (lo/hi) and optional branches —
    do NOT treat None as missing data (that wrongly drops K2/K5/... eligibility).
    """
    try:
        if isinstance(x, float) and np.isnan(x):
            return True
    except Exception:
        pass
    try:
        if isinstance(x, (np.floating,)) and np.isnan(float(x)):
            return True
    except Exception:
        pass
    if isinstance(x, str) and x.strip().lower() == "nan":
        return True
    if isinstance(x, dict):
        if x.get("skipped") is True:
            return False
        return any(_details_has_nan(v) for k, v in x.items() if k not in {"reason", "expr", "note", "source", "col"})
    if isinstance(x, (list, tuple)):
        return any(_details_has_nan(v) for v in x)
    return False


def _rule_window_hours(rule: Dict[str, Any], default: int = 3) -> int:
    """
    Resolve evaluation-window length in hours.

    Prefer the largest signal among stage2 metadata AND the rule expression
    (consec_Nh / duration_Nh / slope_Nh / slope_Nd). Metadata alone can be
    wrong (e.g. window_hours_max=1 on a consec_6h rule) and would shrink mutex
    so later short injections stomp long-window faults.
    """
    import re as _re

    cands: list[int] = []

    feats = ((rule or {}).get("complexity_detail") or {}).get("features") or {}
    wh = feats.get("window_hours_max")
    try:
        if wh is not None and int(wh) > 0:
            cands.append(int(wh))
    except Exception:
        pass

    tr = (rule or {}).get("temporal_requirements") or {}
    wl = _compact_str(tr.get("window_length"))
    if wl:
        wl_l = wl.lower()
        m = _re.fullmatch(r"(\d+)\s*([hdwmy]|days|hours)", wl_l)
        if m:
            n = int(m.group(1))
            u = m.group(2)
            mapped = {"h": n, "hours": n, "d": n * 24, "days": n * 24, "w": n * 168, "m": n * 720, "y": n * 8760}.get(u)
            if mapped:
                cands.append(int(mapped))
        else:
            # e.g. "around 24h before/after alarm" — fullmatch may miss prefixed labels
            m2 = _re.search(r"(\d+)\s*(h|hours|d|days)", wl_l)
            if m2:
                n = int(m2.group(1))
                u = m2.group(2)
                hours = n if u in ("h", "hours") else n * 24
                if "around" in wl or "before" in wl or "after" in wl:
                    hours *= 2  # ±Nh ⇒ occupy 2N hours
                cands.append(int(hours))

    pr = (rule or {}).get("pointwise_criteria") or {}
    try:
        n = int(pr.get("consecutive_points") or 0)
        if n > 0:
            cands.append(n)
    except Exception:
        pass

    expr = _compact_str((rule or {}).get("rule_expression"))
    if expr:
        for n in _re.findall(r"consec_(\d+)\s*h", expr, flags=_re.I):
            cands.append(int(n))
        for n in _re.findall(r"duration_(\d+)\s*h", expr, flags=_re.I):
            cands.append(int(n))
        for n in _re.findall(r"slope_(\d+)\s*h", expr, flags=_re.I):
            cands.append(int(n))
        for n in _re.findall(r"slope_(\d+)\s*d", expr, flags=_re.I):
            cands.append(int(n) * 24)
        for n in _re.findall(r"last_(\d+)\s*h", expr, flags=_re.I):
            cands.append(int(n))
        # prev_7d(...) / prev_24h(...) lookbacks (e.g. coating K18 ratio decline)
        for n in _re.findall(r"prev_(\d+)\s*d", expr, flags=_re.I):
            cands.append(int(n) * 24)
        for n in _re.findall(r"prev_(\d+)\s*h", expr, flags=_re.I):
            cands.append(int(n))
        # Rolling aggregates used by many model exprs (avg_24h / max_24h / count_24h …)
        for n in _re.findall(r"(?:avg|max|min|std|count)_(\d+)\s*h", expr, flags=_re.I):
            cands.append(int(n))
        # Association windows: within_24h_around ⇒ ±24h (48h mutex); within_1h_after ⇒ 2h span
        for n in _re.findall(r"within_(\d+)\s*h_around", expr, flags=_re.I):
            cands.append(int(n) * 2)
        for n in _re.findall(r"within_(\d+)\s*h_after", expr, flags=_re.I):
            cands.append(int(n) + 1)

    if cands:
        return max(cands)
    return default


def _mutex_hours(rule: Dict[str, Any], episode_span_h: int) -> int:
    """
    Occupancy for non-overlap:
      - short/consec rules: their own window
      - long lookback (corr/slope): occupy the full evaluation window.

    Previously long windows were capped at 7d, which still let 30d slope rules
    (e.g. bushing K9/K11) overlap and silently destroy each other's ramps.
    """
    wh = _rule_window_hours(rule, 3)
    if wh <= 72:
        return max(1, wh)
    return max(int(episode_span_h), int(wh))


def _is_around_window_rule(rule: Dict[str, Any]) -> bool:
    """True when evaluation / inject footprint is centered (±Nh), not lookback-only."""
    import re as _re

    wl = _compact_str(((rule or {}).get("temporal_requirements") or {}).get("window_length"))
    if wl and ("around" in wl or "before" in wl or "after" in wl):
        return True
    expr = _compact_str((rule or {}).get("rule_expression"))
    if expr and _re.search(r"within_\d+\s*h_around", expr, flags=_re.I):
        return True
    return False


def _mutex_interval(te: pd.Timestamp, rule: Dict[str, Any], episode_span_h: int) -> Interval:
    """
    Occupancy interval ending around te.

    Lookback rules occupy [te-(wh-1), te].
    Around rules (before/after Nh / within_Nh_around) occupy [te-Nh, te+Nh] so forward
    side-effects (e.g. K12 clearing visual ±24h) cannot land on later short faults.
    """
    wh = max(1, int(_mutex_hours(rule, episode_span_h)))
    te = pd.Timestamp(te)
    if _is_around_window_rule(rule):
        half = max(1, wh // 2)
        return Interval(te - pd.Timedelta(hours=half), te + pd.Timedelta(hours=half))
    return Interval(te - pd.Timedelta(hours=wh - 1), te)


def _lookback_start_pos(idx: pd.DatetimeIndex, hours: int) -> int:
    if len(idx) == 0:
        return 0
    return min(max(0, int(hours) - 1), len(idx) - 1)


def _overlaps(a0: pd.Timestamp, a1: pd.Timestamp, occupied: Sequence[Interval]) -> bool:
    for it in occupied:
        if a0 <= it.end and a1 >= it.start:
            return True
    return False


def _inject_registry(inj_mod) -> Dict[str, Any]:
    if hasattr(inj_mod, "INJECT_REGISTRY") and isinstance(getattr(inj_mod, "INJECT_REGISTRY"), dict):
        return dict(inj_mod.INJECT_REGISTRY)
    reg: Dict[str, Any] = {}
    for name in dir(inj_mod):
        if name.startswith("inject_k"):
            fn = getattr(inj_mod, name)
            if callable(fn):
                suf = name.split("inject_k", 1)[1]
                if suf.isdigit():
                    reg[f"K{int(suf)}"] = fn
    return reg


def _copy_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)
    out.attrs = dict(getattr(df, "attrs", {}) or {})
    if "fields_override" in out.attrs and isinstance(out.attrs["fields_override"], dict):
        out.attrs["fields_override"] = dict(out.attrs["fields_override"])
    return out


def _load_derived_column_names(model_dir: Path) -> Set[str]:
    """derived_key + derived_name from derived_metrics.json (EN/CN)."""
    out: Set[str] = set()
    dmj = Path(model_dir) / "derived_metrics.json"
    if not dmj.exists():
        return out
    try:
        data = _read_json(dmj) or {}
    except Exception:
        return out
    for sp in data.get("derived_metrics") or []:
        if not isinstance(sp, dict):
            continue
        for k in ("derived_key", "derived_name"):
            v = _compact_str(sp.get(k))
            if v:
                out.add(v)
    return out


def _strip_derived_columns(df: pd.DataFrame, derived_keys: Optional[Set[str]]) -> None:
    """Remove materialised derived cols so rules recompute from atoms (near-miss safe)."""
    if not derived_keys:
        return
    for c in list(derived_keys):
        if c in df.columns:
            del df[c]


def _merge_df(dst: pd.DataFrame, src: pd.DataFrame) -> None:
    """Copy columns from src into dst (injectors may add temp columns)."""
    for c in src.columns:
        dst[c] = src[c]
    # preserve attrs updates (dimensions written by injectors)
    for k, v in (getattr(src, "attrs", {}) or {}).items():
        if k == "fields_override" and isinstance(v, dict):
            base = dict(dst.attrs.get("fields_override") or {})
            base.update(v)
            dst.attrs["fields_override"] = base
        else:
            dst.attrs[k] = v


def _changed_cols(before: pd.DataFrame, after: pd.DataFrame) -> List[str]:
    changed: List[str] = []
    cols = [c for c in after.columns if c in before.columns]
    # bool is numeric_dtype in pandas but does not support ndarray `-` (numpy≥1.25).
    num_cols = [
        c
        for c in cols
        if pd.api.types.is_numeric_dtype(before[c]) and not pd.api.types.is_bool_dtype(before[c])
    ]
    for c in num_cols:
        a = pd.to_numeric(before[c], errors="coerce")
        b = pd.to_numeric(after[c], errors="coerce")
        if ((a - b).abs() > 1e-9).fillna(False).any():
            changed.append(c)
    for c in cols:
        if c in num_cols:
            continue
        try:
            if not before[c].astype("object").equals(after[c].astype("object")):
                changed.append(c)
        except Exception:
            changed.append(c)
    for c in after.columns:
        if c not in before.columns:
            changed.append(c)
    return changed


def _eval_rule(
    rule_fn,
    df: pd.DataFrame,
    ts_end: pd.Timestamp,
    params: Dict[str, float],
    *,
    derived_keys: Optional[Set[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    # Drop materialised derived cols so ratios/products are recomputed from atoms.
    # Prevents near-miss alpha blend of stale derived columns from masking a true trigger.
    _strip_derived_columns(df, derived_keys)
    fo = (getattr(df, "attrs", {}) or {}).get("fields_override") or {}
    try:
        res = rule_fn(df, ts_end, params, fo if fo else None)
    except TypeError:
        try:
            res = rule_fn(df, ts_end, params)
        except Exception as e:
            return False, {"error": str(e)}
    except Exception as e:
        return False, {"error": str(e)}
    if isinstance(res, tuple) and len(res) >= 2:
        return bool(res[0]), res[1] if isinstance(res[1], dict) else {}
    hit = bool(getattr(res, "triggered", False))
    det = getattr(res, "details", {})
    return hit, det if isinstance(det, dict) else {}


def _rule_matches_device(rule: Dict[str, Any], fields: Dict[str, Any]) -> bool:
    conds = rule.get("applicability_conditions") or []
    if not conds:
        return True
    for c in conds:
        if not isinstance(c, dict):
            continue
        col = _compact_str(c.get("field_name"))
        want = [_compact_str(x) for x in (c.get("field_value") or [])]
        got = _compact_str(fields.get(col))
        if not got:
            return False
        if want and got not in want:
            return False
    return True


def _rule_data_available(
    rule_fn,
    df: pd.DataFrame,
    ts_probe: pd.Timestamp,
    params: Dict[str, float],
    *,
    derived_keys: Optional[Set[str]] = None,
) -> bool:
    for v in params.values():
        try:
            if not np.isfinite(float(v)):
                return False
        except Exception:
            return False
    hit, det = _eval_rule(rule_fn, df, ts_probe, params, derived_keys=derived_keys)
    _ = hit
    if det.get("skipped") is True:
        # skipped for missing external inputs (PRPD / T_ambient) — still injectable if injector can create them
        reason = _compact_str(det.get("reason"))
        if "PRPD" in reason:
            return False  # no injector for K16
        # T_ambient: injector creates the column → keep eligible
        if "ambient temperature" in reason or "T_ambient" in reason:
            return True
        # applicability failure should have been filtered already
        if "applicability_conditions" in reason:
            return False
        return True
    return not _details_has_nan(det)


def _binary_backoff_injection(
    *,
    df: pd.DataFrame,
    ts_end: pd.Timestamp,
    inject_fn,
    rule_fn,
    params: Dict[str, float],
    derived_keys: Optional[Set[str]] = None,
    max_iter: int = 18,
) -> Tuple[bool, Dict[str, Any]]:
    dkeys = set(derived_keys or ())
    base = _copy_df(df)
    _strip_derived_columns(base, dkeys)
    base_hit, base_det = _eval_rule(rule_fn, base, ts_end, params, derived_keys=dkeys)
    if base_hit:
        return False, {"reason": f"base already triggers at ts_end; det={base_det}"}

    work = _copy_df(df)
    _strip_derived_columns(work, dkeys)
    inject_fn(work, ts_end, params)
    _strip_derived_columns(work, dkeys)
    hit, det = _eval_rule(rule_fn, work, ts_end, params, derived_keys=dkeys)
    if not hit:
        return False, {"reason": "fault injection did not trigger rule at ts_end", "details": det}

    inj = _copy_df(work)
    _strip_derived_columns(inj, dkeys)
    shared = [c for c in base.columns if c in inj.columns and c not in dkeys]
    num_cols = [
        c
        for c in shared
        if pd.api.types.is_numeric_dtype(base[c]) and not pd.api.types.is_bool_dtype(base[c])
    ]
    obj_cols = [c for c in shared if c not in num_cols]

    grid = sorted(set([0.0] + [2.0 ** (-k) for k in range(1, 11)] + [i / 32.0 for i in range(1, 33)]))
    trig_map: Dict[float, bool] = {}
    det_map: Dict[float, Dict[str, Any]] = {}
    for a in grid:
        trial = _copy_df(base)
        _strip_derived_columns(trial, dkeys)
        if num_cols:
            trial[num_cols] = base[num_cols] * (1.0 - a) + inj[num_cols] * a
        for c in obj_cols:
            if a > 0.5:
                trial[c] = inj[c]
        # keep injector-written attrs (dimensions / ambient flags) when a>0.5
        if a > 0.5:
            for k, v in (inj.attrs or {}).items():
                if k == "fields_override" and isinstance(v, dict):
                    fo = dict(trial.attrs.get("fields_override") or {})
                    fo.update(v)
                    trial.attrs["fields_override"] = fo
                else:
                    trial.attrs[k] = v
        hit2, det2 = _eval_rule(rule_fn, trial, ts_end, params, derived_keys=dkeys)
        trig_map[float(a)] = bool(hit2)
        det_map[float(a)] = det2 if isinstance(det2, dict) else {}

    feasible = [a for a, t in trig_map.items() if (a > 0.0 and not t)]
    if not feasible:
        return False, {"reason": "could not backoff to a non-triggering near-miss", "details": det_map.get(0.0, {})}
    best_alpha = float(max(feasible))
    best_det: Dict[str, Any] = det_map.get(best_alpha, {})

    higher_trigger = [a for a, t in trig_map.items() if (a > best_alpha and t)]
    if higher_trigger:
        lo, hi = best_alpha, float(min(higher_trigger))
        for _ in range(max_iter):
            a = (lo + hi) / 2.0
            trial = _copy_df(base)
            _strip_derived_columns(trial, dkeys)
            if num_cols:
                trial[num_cols] = base[num_cols] * (1.0 - a) + inj[num_cols] * a
            for c in obj_cols:
                if a > 0.5:
                    trial[c] = inj[c]
            if a > 0.5:
                for k, v in (inj.attrs or {}).items():
                    if k == "fields_override" and isinstance(v, dict):
                        fo = dict(trial.attrs.get("fields_override") or {})
                        fo.update(v)
                        trial.attrs["fields_override"] = fo
            hit2, det2 = _eval_rule(rule_fn, trial, ts_end, params, derived_keys=dkeys)
            if hit2:
                hi = a
            else:
                best_alpha = a
                best_det = det2 if isinstance(det2, dict) else {}
                lo = a
            if (hi - lo) < 1e-3:
                break

    final = _copy_df(base)
    _strip_derived_columns(final, dkeys)
    if num_cols:
        final[num_cols] = base[num_cols] * (1.0 - best_alpha) + inj[num_cols] * best_alpha
    for c in obj_cols:
        if best_alpha > 0.5:
            final[c] = inj[c]
    if best_alpha > 0.5:
        for k, v in (inj.attrs or {}).items():
            if k == "fields_override" and isinstance(v, dict):
                fo = dict(final.attrs.get("fields_override") or {})
                fo.update(v)
                final.attrs["fields_override"] = fo
    _strip_derived_columns(final, dkeys)
    _merge_df(df, final)
    _strip_derived_columns(df, dkeys)
    return True, {
        "alpha": float(best_alpha),
        "details": best_det,
        "changed_cols": _changed_cols(base, final),
    }


def _bg_self_check(
    *,
    df: pd.DataFrame,
    rules: Dict[str, Any],
    rule_objs: Dict[str, Dict[str, Any]],
    params_by_rule: Dict[str, Dict[str, float]],
    eligible: Sequence[str],
    derived_keys: Optional[Set[str]] = None,
    n_probes: int = 6,
) -> List[Dict[str, Any]]:
    """Sample timestamps; report any unexpected background triggers."""
    if len(df.index) == 0 or not eligible:
        return []
    probes = []
    positions = np.linspace(0.15, 0.95, num=min(n_probes, len(df.index)))
    for f in positions:
        probes.append(pd.Timestamp(df.index[int(f * (len(df.index) - 1))]))
    hits: List[Dict[str, Any]] = []
    for rid in eligible:
        fn = rules.get(rid)
        if fn is None:
            continue
        # skip rules that will always skip on background (missing external cols)
        p = params_by_rule.get(rid, {})
        for ts in probes:
            hit, det = _eval_rule(fn, df, ts, p, derived_keys=derived_keys)
            if hit:
                hits.append({"rule_id": rid, "ts": _fmt(ts), "details": det})
                break
    return hits


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None, *, default_model_dir: str | Path | None = None) -> int:
    _ensure_repo_root_on_syspath()
    if default_model_dir is not None:
        main._default_model_dir = Path(default_model_dir)  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description="Generate per-device datasets (shared orchestrator).")
    ap.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="path to models/<MODEL_DIR> (required unless default_model_dir is passed by a thin wrapper)",
    )
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--start", type=str, default="2024-01-01 00:00:00")
    ap.add_argument("--end", type=str, default="2025-12-31 23:00:00")
    ap.add_argument("--limit", type=int, default=0, help="0=all devices; >0 = first N rows")
    ap.add_argument("--device_index", type=int, default=-1, help="single 0-based device row; -1=all/limit")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--max_scenarios",
        type=int,
        default=0,
        help="optional hard cap on successful fault scenarios (0=no cap; density from episodes)",
    )
    ap.add_argument("--bg_retries", type=int, default=3, help="background regenerate retries if self-check fires")
    add_inspection_schedule_arguments(ap)
    args = ap.parse_args(argv)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    default_md = getattr(main, "_default_model_dir", None)
    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
    elif default_md:
        model_dir = Path(default_md).resolve()
    else:
        raise SystemExit("Missing --model_dir (example: --model_dir 'data/demo/models/0002_pm800kV_high-end_converter_transformer')")
    if not model_dir.is_dir():
        raise SystemExit(f"model_dir not found: {model_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (model_dir / "per_device")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_dir = model_dir / "pipeline_generate"
    bg, rf, inj_mod = _load_pipeline_modules(pipeline_dir)
    RULES: Dict[str, Any] = dict(getattr(rf, "RULE_REGISTRY", None) or {})
    if not RULES:
        raise SystemExit(
            f"{pipeline_dir / 'rule_functions.py'} has no RULE_REGISTRY "
            "(model kit not implemented yet; orchestrator is shared)."
        )
    if not hasattr(rf, "load_params_by_rule") or not hasattr(rf, "load_stage2_rules"):
        raise SystemExit(
            f"{pipeline_dir / 'rule_functions.py'} missing load_params_by_rule / load_stage2_rules."
        )
    params_by_rule: Dict[str, Dict[str, float]] = rf.load_params_by_rule(model_dir)  # type: ignore[attr-defined]
    rule_objs: Dict[str, Dict[str, Any]] = rf.load_stage2_rules(model_dir)  # type: ignore[attr-defined]
    INJECT = _inject_registry(inj_mod)
    if not INJECT:
        raise SystemExit(
            f"{pipeline_dir / 'scenario_injectors.py'} has no inject_k* / INJECT_REGISTRY "
            "(model kit not implemented yet)."
        )
    # Derived cols must not be alpha-blended / trusted as materialised caches in near-miss.
    derived_keys = _load_derived_column_names(model_dir)

    catalog = _read_json(model_dir / "metric_catalog.json") or []
    cols = sorted(
        {
            _compact_str(it.get("database_field"))
            for it in catalog
            if isinstance(it, dict) and _compact_str(it.get("database_field"))
        }
    )
    cn2db: Dict[str, str] = {}
    for it in catalog if isinstance(catalog, list) else []:
        if not isinstance(it, dict):
            continue
        cn, db = _compact_str(it.get("metric_name")), _compact_str(it.get("database_field"))
        if cn and db:
            cn2db[cn] = db

    xlsx_cands = sorted(model_dir.glob("*_with_metrics.xlsx")) + sorted(model_dir.glob("*.xlsx"))
    xlsx_cands = [p for p in xlsx_cands if not p.name.startswith("~$") and ".bak_" not in p.name]
    if not xlsx_cands:
        raise SystemExit(f"Device list xlsx not found under model_dir: {model_dir}")
    xlsx = xlsx_cands[0]
    try:
        _, devices_df = read_device_sheet(xlsx)
    except Exception:
        devices_df = pd.read_excel(xlsx)

    dim_cols = _dimension_column_names(model_dir, devices_df)
    device_col = resolve_column_name(devices_df.columns, "device_name") or (devices_df.columns[0] if len(devices_df.columns) else "device_name")

    rows = [(j, devices_df.iloc[j]) for j in range(len(devices_df))]
    if int(args.device_index) >= 0:
        j = int(args.device_index)
        if j >= len(rows):
            raise SystemExit(f"--device_index {j} out of range (n={len(rows)})")
        rows = [rows[j]]
    elif int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    t0, t1 = _dt(args.start), _dt(args.end)
    idx = pd.date_range(start=t0, end=t1, freq="h")
    episode_span_h = max(1, int(args.episode_span_hours))

    for j, row in rows:
        device_name = _compact_str(row.get(device_col)) or f"device-{j}"
        row_fields = _fields_override_from_device_row(row, dim_cols)

        # --- background with optional self-check retry ---
        bg_hits: List[Dict[str, Any]] = []
        df = pd.DataFrame(index=idx, columns=cols)
        for attempt in range(max(1, int(args.bg_retries))):
            df = pd.DataFrame(index=idx, columns=cols)
            np.random.seed(int(args.seed) + 1000 * attempt + j)
            # Set device attrs before background so model generators can scale /
            # specialize by rated_current, primary_standby_role, etc. (e.g. 0012 K13 backup heartbeat).
            df.attrs["model_dir"] = str(model_dir)
            df.attrs["fields_override"] = dict(row_fields)
            df.attrs["device_fields_override"] = dict(row_fields)
            _shared_generate_background(df, model_dir=model_dir)
            bg.generate_background(df, model_dir=model_dir)  # type: ignore[attr-defined]

            allowed_db_fields = _split_semicolon(_compact_str(row_get(row, "maintenance_metric_db_fields_all")))
            if not allowed_db_fields:
                want_cn = _split_semicolon(_compact_str(row_get(row, "maintenance_metrics_all")))
                allowed_db_fields = [cn2db[c] for c in want_cn if c in cn2db]
            _mask_df_to_allowed_fields(df, allowed_db_fields=allowed_db_fields)

            # eligible = device applicable ∩ RULE ∩ INJECT ∩ data-available
            applicable_rules = _split_semicolon(_compact_str(row_get(row, "applicable_rule_ids_all")))
            if not applicable_rules:
                applicable_rules = sorted(RULES.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)

            eligible: List[str] = []
            ts_probe = pd.Timestamp(df.index[len(df.index) // 2]) if len(df.index) else t0
            for rid in applicable_rules:
                if rid not in RULES or rid not in INJECT:
                    continue
                robj = rule_objs.get(rid) or {}
                if not _rule_matches_device(robj, row_fields):
                    continue
                if _rule_data_available(
                    RULES[rid],
                    df,
                    ts_probe,
                    params_by_rule.get(rid, {}),
                    derived_keys=derived_keys,
                ):
                    eligible.append(rid)

            bg_hits = _bg_self_check(
                df=df,
                rules=RULES,
                rule_objs=rule_objs,
                params_by_rule=params_by_rule,
                eligible=eligible,
                derived_keys=derived_keys,
            )
            if not bg_hits:
                break

        # Per-(seed, model, device) inject RNG so episode calendars / fault picks
        # do not clone across MODEL_DIRs that reuse the same CLI --seed.
        inject_rng = _salted_rng(int(args.seed), model_dir.name, j, "inject")
        random.setstate(inject_rng.getstate())

        random.shuffle(eligible)
        eligible_rules_with_injector = list(eligible)
        chosen_rules_for_injection: List[str] = []
        scenarios: List[ScenarioRecord] = []
        occupied: List[Interval] = []

        inspection_days = build_inspection_days(
            idx,
            count=int(args.inspection_count),
            interval_months=int(args.inspection_interval_months),
            hour=int(args.inspection_hour),
        )
        day_candidates: List[pd.Timestamp] = []
        if len(idx):
            uniq_days = pd.to_datetime(pd.Series(idx.normalize()).drop_duplicates())
            for d in uniq_days:
                anchor = pd.Timestamp(d) + pd.Timedelta(hours=max(0, min(23, int(args.inspection_hour))))
                pos = idx.get_indexer([anchor], method="nearest")[0]
                if pos >= 0:
                    day_candidates.append(pd.Timestamp(idx[pos]))
        dedup_days: List[pd.Timestamp] = []
        seen_days: set = set()
        for d in day_candidates:
            k_day = str(pd.Timestamp(d))
            if k_day in seen_days:
                continue
            seen_days.add(k_day)
            dedup_days.append(pd.Timestamp(d))

        episode_n = min(len(dedup_days), max(0, int(args.alert_episodes)))
        ep_rng = _salted_rng(int(args.seed), model_dir.name, j, "episode_days")
        episode_days = ep_rng.sample(dedup_days, k=episode_n) if episode_n > 0 else []
        episode_span = pd.Timedelta(hours=episode_span_h)
        alerts_min = max(1, int(args.alerts_per_episode_min))
        alerts_max = max(alerts_min, int(args.alerts_per_episode_max))
        max_faults = int(args.max_scenarios) if int(args.max_scenarios) > 0 else 10**9

        def _n_faults_ok() -> int:
            return len([s for s in scenarios if s.type == "fault" and not s.skipped])

        for epi_day in episode_days:
            if _n_faults_ok() >= max_faults:
                break
            n_alerts = random.randint(alerts_min, alerts_max)
            # prefer rotating through eligible rules
            pool = list(eligible) if eligible else []
            random.shuffle(pool)
            tried = 0
            while n_alerts > 0 and pool and tried < len(pool) * 3 and _n_faults_ok() < max_faults:
                tried += 1
                rid = pool[tried % len(pool)]
                robj = rule_objs.get(rid) or {}
                lo = pd.Timestamp(epi_day) - (episode_span / 2)
                hi = pd.Timestamp(epi_day) + (episode_span / 2)
                cand_idx = df.loc[(df.index >= lo) & (df.index <= hi)].index
                min_pos = _lookback_start_pos(df.index, _rule_window_hours(robj))
                if len(df.index):
                    cand_idx = cand_idx[cand_idx >= df.index[min_pos]]
                if len(cand_idx) == 0:
                    cand_idx = df.index[df.index >= df.index[min_pos]] if len(df.index) else df.index
                if len(cand_idx) == 0:
                    continue

                # pick a non-overlapping candidate
                cand_pool = list(cand_idx)
                random.shuffle(cand_pool)
                ts_end = None
                for ts_try in cand_pool[:64]:
                    te = pd.Timestamp(ts_try)
                    mit = _mutex_interval(te, robj, episode_span_h)
                    if not _overlaps(mit.start, mit.end, occupied):
                        ts_end = te
                        break
                if ts_end is None:
                    continue

                rec_h = _rule_window_hours(robj)
                win_start = ts_end - pd.Timedelta(hours=max(1, int(rec_h)) - 1)
                p = params_by_rule.get(rid, {})
                try:
                    df_work = _copy_df(df)
                    inj_out = INJECT[rid](df_work, ts_end, p)
                    injected_ok = True
                    inj_notes = ""
                    if inj_out is not None and hasattr(inj_out, "injected"):
                        injected_ok = bool(inj_out.injected)
                        inj_notes = str(getattr(inj_out, "notes", "") or "")
                    if not injected_ok:
                        scenarios.append(
                            ScenarioRecord(
                                rule_id=rid,
                                window_start=_fmt(win_start),
                                window_end=_fmt(ts_end),
                                type="fault",
                                skipped=True,
                                reason=f"injector declined: {inj_notes}",
                            )
                        )
                        continue
                    hit, det = _eval_rule(RULES[rid], df_work, ts_end, p, derived_keys=derived_keys)
                    if not hit:
                        scenarios.append(
                            ScenarioRecord(
                                rule_id=rid,
                                window_start=_fmt(win_start),
                                window_end=_fmt(ts_end),
                                type="fault",
                                skipped=True,
                                reason=f"injection did not trigger; det={det}",
                                details=det if isinstance(det, dict) else None,
                            )
                        )
                        continue
                    if det.get("skipped") is True:
                        scenarios.append(
                            ScenarioRecord(
                                rule_id=rid,
                                window_start=_fmt(win_start),
                                window_end=_fmt(ts_end),
                                type="fault",
                                skipped=True,
                                reason=f"rule skipped after inject: {det.get('reason')}",
                                details=det,
                            )
                        )
                        continue
                    # hit=True already: do not reject for optional NaN diagnostics in details
                    changed = _changed_cols(df, df_work)
                    _merge_df(df, df_work)
                    occupied.append(_mutex_interval(ts_end, robj, episode_span_h))
                    chosen_rules_for_injection.append(rid)
                    scenarios.append(
                        ScenarioRecord(
                            rule_id=rid,
                            window_start=_fmt(win_start),
                            window_end=_fmt(ts_end),
                            type="fault",
                            note_cn="fault injection succeeded (alert cluster)",
                            changed_cols=changed,
                            details=det,
                        )
                    )
                    n_alerts -= 1
                except Exception as e:
                    scenarios.append(
                        ScenarioRecord(
                            rule_id=rid,
                            window_start=_fmt(win_start),
                            window_end=_fmt(ts_end),
                            type="fault",
                            skipped=True,
                            reason=str(e),
                        )
                    )

        # density: floor ≥12–16; soft target ~30–40 via episode*max + top-up
        floor_target = max(12, int(args.alert_episodes) * int(args.alerts_per_episode_min))
        soft_target = min(
            40,
            max(
                floor_target,
                int(args.alert_episodes) * int(args.alerts_per_episode_max),
                min(len(eligible) * 3, 36) if eligible else floor_target,
            ),
        )
        extra_guard = 0
        while _n_faults_ok() < min(soft_target, max_faults) and eligible and extra_guard < 160:
            extra_guard += 1
            rid = random.choice(eligible)
            robj = rule_objs.get(rid) or {}
            min_pos = _lookback_start_pos(df.index, _rule_window_hours(robj))
            cand = list(df.index[min_pos:]) if len(df.index) else []
            if not cand:
                break
            random.shuffle(cand)
            ts_end = None
            for ts_try in cand[:80]:
                te = pd.Timestamp(ts_try)
                mit = _mutex_interval(te, robj, episode_span_h)
                if not _overlaps(mit.start, mit.end, occupied):
                    ts_end = te
                    break
            if ts_end is None:
                continue
            rec_h = _rule_window_hours(robj)
            win_start = ts_end - pd.Timedelta(hours=max(1, int(rec_h)) - 1)
            p = params_by_rule.get(rid, {})
            try:
                df_work = _copy_df(df)
                inj_out = INJECT[rid](df_work, ts_end, p)
                if inj_out is not None and hasattr(inj_out, "injected") and not inj_out.injected:
                    continue
                hit, det = _eval_rule(RULES[rid], df_work, ts_end, p, derived_keys=derived_keys)
                if not hit or det.get("skipped"):
                    continue
                changed = _changed_cols(df, df_work)
                _merge_df(df, df_work)
                occupied.append(_mutex_interval(ts_end, robj, episode_span_h))
                chosen_rules_for_injection.append(rid)
                scenarios.append(
                    ScenarioRecord(
                        rule_id=rid,
                        window_start=_fmt(win_start),
                        window_end=_fmt(ts_end),
                        type="fault",
                        note_cn="fault injection succeeded (density fill)",
                        changed_cols=changed,
                        details=det,
                    )
                )
            except Exception:
                continue

        # --- near-miss ---
        near_miss: List[ScenarioRecord] = []
        nm_rule_ids = list(eligible_rules_with_injector)
        random.shuffle(nm_rule_ids)
        near_miss_attempted_rules: List[str] = []
        nm_days_n = min(len(inspection_days), max(0, int(args.near_miss_days)))
        nm_rng = _salted_rng(int(args.seed), model_dir.name, j, "near_miss_days")
        nm_days = nm_rng.sample(inspection_days, k=nm_days_n) if nm_days_n > 0 else []
        nm_per_day = max(1, int(args.near_miss_per_day))
        nm_span = pd.Timedelta(hours=episode_span_h)
        nm_occupied: List[Interval] = list(occupied)  # also avoid fault windows

        for day in nm_days:
            day_nm_count = random.randint(1, nm_per_day) if nm_per_day > 1 else 1
            for _ in range(day_nm_count):
                if not nm_rule_ids:
                    break
                rid = random.choice(nm_rule_ids)
                robj = rule_objs.get(rid) or {}
                lo = pd.Timestamp(day) - (nm_span / 2)
                hi = pd.Timestamp(day) + (nm_span / 2)
                cand_idx = df.loc[(df.index >= lo) & (df.index <= hi)].index
                min_pos = _lookback_start_pos(df.index, _rule_window_hours(robj))
                if len(df.index):
                    cand_idx = cand_idx[cand_idx >= df.index[min_pos]]
                if len(cand_idx) == 0:
                    continue
                cand_pool = list(cand_idx)
                random.shuffle(cand_pool)
                base_clean: List[pd.Timestamp] = []
                p = params_by_rule.get(rid, {})
                for ts_try in cand_pool[:48]:
                    te = pd.Timestamp(ts_try)
                    mit = _mutex_interval(te, robj, episode_span_h)
                    if _overlaps(mit.start, mit.end, nm_occupied):
                        continue
                    base_hit, _ = _eval_rule(RULES[rid], df, te, p, derived_keys=derived_keys)
                    if not base_hit:
                        base_clean.append(te)
                if not base_clean:
                    continue
                near_miss_attempted_rules.append(rid)
                success = False
                last_reason = "near-miss failed after candidate retries"
                last_details: Optional[Dict[str, Any]] = None
                ts_end = base_clean[0]
                for ts_try in base_clean[:16]:
                    ts_end = pd.Timestamp(ts_try)
                    rec_h = _rule_window_hours(robj)
                    win_start = ts_end - pd.Timedelta(hours=max(1, int(rec_h)) - 1)
                    try:
                        df_work = _copy_df(df)
                        ok, info = _binary_backoff_injection(
                            df=df_work,
                            ts_end=ts_end,
                            inject_fn=INJECT[rid],
                            rule_fn=RULES[rid],
                            params=p,
                            derived_keys=derived_keys,
                        )
                        if not ok:
                            last_reason = str(info.get("reason", "near-miss failed"))
                            last_details = info.get("details") if isinstance(info.get("details"), dict) else None
                            continue
                        # confirm still not triggered
                        hit_nm, det_nm = _eval_rule(RULES[rid], df_work, ts_end, p, derived_keys=derived_keys)
                        if hit_nm:
                            last_reason = "near-miss still triggers after backoff"
                            last_details = det_nm
                            continue
                        # Reject if this NM compounds an earlier NM into a true trigger
                        # (inject footprint can exceed mutex when injector writes > rule window).
                        prior_ruined = False
                        for prev in near_miss:
                            if prev.skipped:
                                continue
                            prev_rid = str(prev.rule_id)
                            if prev_rid not in RULES:
                                continue
                            te_prev = pd.Timestamp(prev.window_end)
                            hit_prev, _ = _eval_rule(
                                RULES[prev_rid],
                                df_work,
                                te_prev,
                                params_by_rule.get(prev_rid, {}),
                                derived_keys=derived_keys,
                            )
                            if hit_prev:
                                prior_ruined = True
                                break
                        if prior_ruined:
                            last_reason = "near-miss would re-trigger a prior near_miss"
                            last_details = det_nm if isinstance(det_nm, dict) else None
                            continue
                        det2 = info.get("details")
                        _merge_df(df, df_work)
                        nm_occupied.append(_mutex_interval(ts_end, robj, episode_span_h))
                        alpha = info.get("alpha")
                        changed_cols = info.get("changed_cols")
                        note = (
                            f"near-miss injection succeeded: trigger+backoff (alpha~{float(alpha):.3f})"
                            if alpha is not None
                            else "near-miss injection succeeded"
                        )
                        near_miss.append(
                            ScenarioRecord(
                                rule_id=rid,
                                window_start=_fmt(win_start),
                                window_end=_fmt(ts_end),
                                type="near_miss",
                                note_cn=note,
                                alpha=float(alpha) if alpha is not None else None,
                                changed_cols=[str(x) for x in changed_cols] if isinstance(changed_cols, list) else None,
                                details=det2 if isinstance(det2, dict) else None,
                            )
                        )
                        success = True
                        break
                    except Exception as e:
                        last_reason = str(e)
                        last_details = None
                        continue
                if not success:
                    near_miss.append(
                        ScenarioRecord(
                            rule_id=rid,
                            window_start=_fmt(ts_end - pd.Timedelta(hours=max(1, int(_rule_window_hours(robj))) - 1)),
                            window_end=_fmt(ts_end),
                            type="near_miss",
                            skipped=True,
                            reason=last_reason,
                            note_cn="near-miss failed/skipped",
                            details=last_details,
                        )
                    )

        # --- post checks + repair destroyed long-window faults (multi-pass) ---
        # Short windows first each round; long corr/slope last so they survive.
        # Repeat so a later long-window repair that clobbers an earlier one can
        # be re-fixed on the next pass (fixed-point, max 4 rounds).
        for _repair_round in range(4):
            fault_ok = [s for s in scenarios if s.type == "fault" and not s.skipped]
            fault_ok_ordered = sorted(
                fault_ok,
                key=lambda s: _rule_window_hours(rule_objs.get(s.rule_id) or {}, 3),
            )
            any_repair = False
            for s in fault_ok_ordered:
                te = _dt(s.window_end)
                hit, det = _eval_rule(
                    RULES[s.rule_id], df, te, params_by_rule.get(s.rule_id, {}), derived_keys=derived_keys
                )
                if hit and not det.get("skipped"):
                    continue
                try:
                    df_work = _copy_df(df)
                    p = params_by_rule.get(s.rule_id, {})
                    inj_out = INJECT[s.rule_id](df_work, te, p)
                    if inj_out is not None and hasattr(inj_out, "injected") and not inj_out.injected:
                        s.skipped = True
                        s.reason = f"post-check failed and injector declined: {getattr(inj_out, 'notes', '')}"
                        continue
                    hit2, det2 = _eval_rule(RULES[s.rule_id], df_work, te, p, derived_keys=derived_keys)
                    if hit2 and not det2.get("skipped"):
                        changed = _changed_cols(df, df_work)
                        _merge_df(df, df_work)
                        s.changed_cols = changed
                        s.details = det2
                        s.note_cn = (s.note_cn or "") + f"; post-check re-inject repair (r{_repair_round+1})"
                        any_repair = True
                    else:
                        s.skipped = True
                        s.reason = f"post-check failed after repair; det={det2}"
                except Exception as e:
                    s.skipped = True
                    s.reason = f"post-check repair error: {e}"
            if not any_repair:
                break

        # Residual same-metric long-window conflicts: keep the longer/earlier lookback,
        # skip shorter-or-later rivals, then re-inject the kept scenario.
        fault_ok = [s for s in scenarios if s.type == "fault" and not s.skipped]
        pending_fail: List[ScenarioRecord] = []
        for s in fault_ok:
            te = _dt(s.window_end)
            hit, det = _eval_rule(
                RULES[s.rule_id], df, te, params_by_rule.get(s.rule_id, {}), derived_keys=derived_keys
            )
            if not hit or det.get("skipped"):
                pending_fail.append(s)
        for s in list(pending_fail):
            if s.skipped:
                continue
            s_rule = rule_objs.get(s.rule_id) or {}
            s_wh = _mutex_hours(s_rule, episode_span_h)
            s_cols = set(s.changed_cols or [])
            s_te = _dt(s.window_end)
            s_it = _mutex_interval(s_te, s_rule, episode_span_h)
            rivals: List[Tuple[ScenarioRecord, int]] = []
            for o in [x for x in scenarios if x.type == "fault" and not x.skipped and x is not s]:
                o_cols = set(o.changed_cols or [])
                if not (s_cols & o_cols):
                    continue
                o_rule = rule_objs.get(o.rule_id) or {}
                o_wh = _mutex_hours(o_rule, episode_span_h)
                o_te = _dt(o.window_end)
                o_it = _mutex_interval(o_te, o_rule, episode_span_h)
                if _overlaps(s_it.start, s_it.end, [o_it]):
                    rivals.append((o, o_wh))
            if not rivals:
                continue

            def _keep_first(a: ScenarioRecord, a_wh: int, b: ScenarioRecord, b_wh: int) -> bool:
                """True if a should be kept over b."""
                if a_wh != b_wh:
                    return a_wh > b_wh
                return _dt(a.window_end) <= _dt(b.window_end)

            if any(not _keep_first(s, s_wh, o, o_wh) for o, o_wh in rivals):
                # This failer loses: skip it.
                s.skipped = True
                s.reason = (
                    f"skipped residual long-window conflict vs "
                    f"{[o.rule_id for o, _ in rivals]}; kept longer/earlier lookback"
                )
                continue
            # This failer wins: skip rival losers then re-inject.
            for o, o_wh in rivals:
                if not _keep_first(s, s_wh, o, o_wh):
                    o.skipped = True
                    o.reason = (
                        f"skipped residual long-window conflict vs {s.rule_id}; "
                        f"kept longer/earlier lookback"
                    )
            try:
                df_work = _copy_df(df)
                p = params_by_rule.get(s.rule_id, {})
                inj_out = INJECT[s.rule_id](df_work, s_te, p)
                hit2, det2 = _eval_rule(RULES[s.rule_id], df_work, s_te, p, derived_keys=derived_keys)
                if hit2 and not det2.get("skipped"):
                    changed = _changed_cols(df, df_work)
                    _merge_df(df, df_work)
                    s.changed_cols = changed
                    s.details = det2
                    s.note_cn = (s.note_cn or "") + "; conflict-resolve re-inject"
                else:
                    s.skipped = True
                    s.reason = f"conflict-resolve repair failed; det={det2}"
            except Exception as e:
                s.skipped = True
                s.reason = f"conflict-resolve error: {e}"

        fault_ok = [s for s in scenarios if s.type == "fault" and not s.skipped]
        nm_ok = [s for s in near_miss if s.type == "near_miss" and not s.skipped]
        fault_verify: List[Dict[str, Any]] = []
        for s in fault_ok:
            te = _dt(s.window_end)
            hit, det = _eval_rule(
                RULES[s.rule_id], df, te, params_by_rule.get(s.rule_id, {}), derived_keys=derived_keys
            )
            fault_verify.append({"rule_id": s.rule_id, "window_end": s.window_end, "triggered": bool(hit)})
        nm_verify: List[Dict[str, Any]] = []
        for s in nm_ok:
            te = _dt(s.window_end)
            hit, det = _eval_rule(
                RULES[s.rule_id], df, te, params_by_rule.get(s.rule_id, {}), derived_keys=derived_keys
            )
            nm_verify.append({"rule_id": s.rule_id, "window_end": s.window_end, "triggered": bool(hit)})

        n_eligible = len(eligible_rules_with_injector)
        # soft density targets scale with injectable rule count
        fault_target_hi = 40
        fault_target_lo = 12 if n_eligible < 8 else (30 if n_eligible >= 12 else 16)
        nm_target_lo, nm_target_hi = 6, 12
        density_note = ""
        if n_eligible < 12:
            density_note = (
                f"injectable rules {n_eligible} (no K16; after dimension/data filter), "
                f"fault floor >={fault_target_lo}, soft target {soft_target}"
            )
        density = {
            "fault_ok": len(fault_ok),
            "near_miss_ok": len(nm_ok),
            "fault_target_range": [fault_target_lo, fault_target_hi],
            "near_miss_target_range": [nm_target_lo, nm_target_hi],
            "soft_target_used": soft_target,
            "eligible_injectable_rules": n_eligible,
            "fault_meets_floor": len(fault_ok) >= fault_target_lo,
            "fault_meets_soft": len(fault_ok) >= min(soft_target, fault_target_hi),
            "near_miss_meets_floor": len(nm_ok) >= nm_target_lo,
            "note": density_note,
        }

        checks = {
            "background_sample_triggers": bg_hits,
            "background_clean": len(bg_hits) == 0,
            "fault_verified": fault_verify,
            "fault_all_trigger": all(x.get("triggered") for x in fault_verify) if fault_verify else True,
            "near_miss_verified": nm_verify,
            "near_miss_none_trigger": (not any(x.get("triggered") for x in nm_verify)) if nm_verify else True,
            "density": density,
        }

        # --- export ---
        safe_name = _safe_fs_stem(device_name)
        file_stem = f"{j:04d}_{safe_name}"
        csv_path = out_dir / f"{file_stem}.csv"
        df_out = df.copy()
        # Recompute derived metrics so inject-window atoms match exported derived cols.
        if hasattr(rf, "ensure_all_derived"):
            try:
                rf.ensure_all_derived(df_out)  # type: ignore[attr-defined]
            except Exception:
                pass
        # export allow-list ∩ catalog, plus inject-created helpers needed to replay checks
        # (e.g. T_ambient for K18 — not in metric_catalog but required by stage2 expr)
        if allowed_db_fields:
            keep_metric = [c for c in cols if c in df_out.columns and c in set(allowed_db_fields)]
        else:
            keep_metric = [c for c in cols if c in df_out.columns]
        # Derived keys from derived_metrics.json (may already be in allow-list).
        # Prefer per-device xlsx whitelist derived_metric_keys_all when present.
        # Empty whitelist (NaN / "") means export NO derived cols for this device
        # (e.g. vacuum breaker must not land sf6_moisture_density).
        # Missing column → legacy: keep all derived keys that materialised.
        derived_raw = row_get(row, "derived_metric_keys_all") if hasattr(row, "get") else None
        has_derived_whitelist_col = False
        try:
            has_derived_whitelist_col = resolve_column_name(getattr(row, "index", []), "derived_metric_keys_all") is not None
        except Exception:
            has_derived_whitelist_col = derived_raw is not None
        derived_allow = set(_split_semicolon(_compact_str(derived_raw)))
        try:
            dmj = model_dir / "derived_metrics.json"
            if dmj.exists():
                dm_data = json.loads(dmj.read_text(encoding="utf-8"))
                for sp in dm_data.get("derived_metrics") or []:
                    if not isinstance(sp, dict):
                        continue
                    key = _compact_str(sp.get("derived_key"))
                    name = _compact_str(sp.get("derived_name"))
                    if has_derived_whitelist_col and key and key not in derived_allow:
                        # drop unwanted materialised derived cols for this device
                        if key in df_out.columns:
                            del df_out[key]
                        if name and name in df_out.columns:
                            del df_out[name]
                        continue
                    # Product CSV: export English derived_key only (CN derived_name is soft noise).
                    if name and name != key and name in df_out.columns:
                        del df_out[name]
                    if key and key in df_out.columns and key not in keep_metric:
                        keep_metric.append(key)
        except Exception:
            pass
        helper_cols: List[str] = []
        touched: set = set()
        for s in scenarios + near_miss:
            for c in s.changed_cols or []:
                touched.add(str(c))
            if s.skipped:
                continue
            # Injector-created helpers required to replay stage2 (may be missing from changed_cols after repair)
            if s.rule_id == "K18":
                touched.add("T_ambient")
            if s.rule_id == "K14":
                touched.update({"V_side1", "V_side2", "dV_operate", "dVdt_operate"})
            if s.rule_id == "K10":
                touched.update({"operating_current_peak", "operating_voltage_min"})
            if s.rule_id == "K15":
                touched.add("operating_time_interval")
            if s.rule_id == "K21":
                # Peer helpers may look "unchanged" after repair re-inject (already at inject
                # targets) and drop out of changed_cols — still required for stage2 replay.
                touched.update(
                    {
                        "peer_cpu_utilization",
                        "peer_memory_utilization",
                        "peer_device_temperature",
                        "peer_board_status",
                        "peer_port_status",
                        "peer_tunnel_pw_status",
                    }
                )
        dim_like = set(dim_cols) | {
            "operating_mechanism",
            "breaker_type",
            "structure_type",
            "primary_standby_role",
            "bay_number",
            "rated_voltage",
            "insulation_medium",
        }
        # Internal inject proxies — never ship as product columns.
        exclude_helpers = {
            "pd_straight_ref",
            "t_straight_ref",
            "PD_straight_ref",
            "T_straight_ref",
        }
        for c in df_out.columns:
            if c in keep_metric or c in dim_like or c in exclude_helpers:
                continue
            if c in touched and c in df_out.columns:
                helper_cols.append(c)
        keep_metric = keep_metric + [c for c in helper_cols if c not in keep_metric]
        df_out = df_out[[c for c in keep_metric if c not in exclude_helpers]]
        df_out.insert(0, "timestamp", df_out.index.strftime("%Y-%m-%d %H:%M:%S"))
        df_out.insert(0, "device_name", device_name)
        df_out.to_csv(csv_path, index=False, encoding="utf-8")

        rep = {
            "model_name": model_dir.name,
            "device_name": device_name,
            "device_index": int(j),
            "device_fields_override": row_fields,
            "dimension_columns": dim_cols,
            "time_range": {"start": _fmt(t0), "end": _fmt(t1), "freq": "1h", "n_rows": int(len(idx))},
            "eligible_rules_with_injector": eligible_rules_with_injector,
            "chosen_rules_for_injection": chosen_rules_for_injection,
            "inspection_days": [_fmt(x) for x in inspection_days],
            "episode_days": [_fmt(x) for x in episode_days],
            "near_miss_attempted_rules": near_miss_attempted_rules,
            "scenarios": [asdict(s) for s in scenarios],
            "near_miss": [asdict(s) for s in near_miss],
            "checks": checks,
        }
        rep_path = out_dir / f"{file_stem}.json"
        rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[OK] {device_name}: csv={csv_path.name} report={rep_path.name} "
            f"insp={len(inspection_days)} ep={len(episode_days)} "
            f"faults={len(fault_ok)} near_miss={len(nm_ok)} "
            f"bg_clean={checks['background_clean']} density_floor={density['fault_meets_floor']}"
        )
        if not density["fault_meets_floor"] or not density["near_miss_meets_floor"]:
            print(f"  [WARN] density below floor: {density}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
