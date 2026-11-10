from __future__ import annotations
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def safe_div(a: float, b: float, default: float = float("nan")) -> float:
    try:
        bb = float(b)
    except Exception:
        return default
    if bb == 0 or (isinstance(bb, float) and not math.isfinite(bb)):
        return default
    try:
        aa = float(a)
    except Exception:
        return default
    if not math.isfinite(aa):
        return default
    try:
        return float(aa / bb)
    except Exception:
        return default


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _default_folder() -> Path:
    # .../models/<MODEL_DIR>/pipeline_generate/this_file.py → MODEL_DIR
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_catalog_alias_maps(folder: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    p = folder / "metric_catalog.json"
    if not p.exists():
        return {}, {}
    items = _read_json(p)
    db2var: Dict[str, str] = {}
    var2db: Dict[str, str] = {}
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                continue
            db = _compact_str(it.get("database_field"))
            var = _compact_str(it.get("variable_name"))
            if db and var:
                db2var[db] = var
                var2db[var] = db
                # case-insensitive convenience (same metric only)
                db2var.setdefault(db.lower(), var)
                var2db.setdefault(var.lower(), db)
    return db2var, var2db


@dataclass(frozen=True)
class DerivedSpec:
    derived_key: str
    derived_name: str
    formula: str
    atomic_db_fields: Tuple[str, ...]


def _load_derived_specs(folder: Path) -> List[DerivedSpec]:
    p = folder / "derived_metrics.json"
    if not p.exists():
        return []
    data = _read_json(p)
    items = data.get("derived_metrics") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: List[DerivedSpec] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        key = _compact_str(it.get("derived_key"))
        name = _compact_str(it.get("derived_name"))
        formula = _compact_str(it.get("formula"))
        if not key or not formula:
            continue
        atomics: List[str] = []
        raw_atoms = it.get("atomic_metrics")
        if isinstance(raw_atoms, list):
            for a in raw_atoms:
                if isinstance(a, dict):
                    db = _compact_str(a.get("db_field"))
                    if db:
                        atomics.append(db)
        out.append(
            DerivedSpec(
                derived_key=key,
                derived_name=name or key,
                formula=formula,
                atomic_db_fields=tuple(atomics),
            )
        )
    return out


def _nan_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.nan, index=df.index, dtype=float)


def _resolve_column(
    df: pd.DataFrame,
    *,
    db_field: str,
    db2var: Dict[str, str],
) -> Optional[str]:
    db = _compact_str(db_field)
    if not db:
        return None
    var = _compact_str(db2var.get(db) or db2var.get(db.lower()))
    candidates = [db]
    if var:
        candidates.append(var)
    # exact then case-insensitive against df columns
    colset = {str(c): str(c) for c in df.columns}
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in colset:
            return colset[cand]
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _series_for_db_field(
    df: pd.DataFrame,
    *,
    db_field: str,
    db2var: Dict[str, str],
) -> pd.Series:
    col = _resolve_column(df, db_field=db_field, db2var=db2var)
    if col is None:
        return _nan_series(df)
    return pd.to_numeric(df[col], errors="coerce")


def _bind_formula_env(
    df: pd.DataFrame,
    spec: DerivedSpec,
    db2var: Dict[str, str],
) -> Dict[str, pd.Series]:
    env: Dict[str, pd.Series] = {}
    for db in spec.atomic_db_fields:
        s = _series_for_db_field(df, db_field=db, db2var=db2var)
        env[db] = s
        env[db.lower()] = s
        var = _compact_str(db2var.get(db) or db2var.get(db.lower()))
        if var:
            env[var] = s
            env[var.lower()] = s
    return env


def _safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
    a2, b2 = a.align(b, join="outer")
    aa = pd.to_numeric(a2, errors="coerce").astype(float)
    bb = pd.to_numeric(b2, errors="coerce").astype(float)
    out = aa / bb
    out = out.mask(~np.isfinite(bb) | (bb == 0.0), np.nan)
    out = out.mask(~np.isfinite(aa), np.nan)
    return out.astype(float)


def _eval_formula(formula: str, env: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
    tree = ast.parse(formula, mode="eval")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"unsupported constant: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise KeyError(f"unbound identifier in formula: {node.id}")
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.UAdd):
                return v
            if isinstance(node.op, ast.USub):
                if isinstance(v, pd.Series):
                    return (-pd.to_numeric(v, errors="coerce")).astype(float)
                return -float(v)
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            left = ev(node.left)
            right = ev(node.right)
            if isinstance(node.op, ast.Div):
                if not isinstance(left, pd.Series):
                    left = pd.Series(float(left), index=index, dtype=float)
                if not isinstance(right, pd.Series):
                    right = pd.Series(float(right), index=index, dtype=float)
                return _safe_div_series(left, right)
            # Promote scalars to series when mixed
            def as_series(x: Any) -> pd.Series:
                if isinstance(x, pd.Series):
                    return pd.to_numeric(x, errors="coerce").astype(float)
                return pd.Series(float(x), index=index, dtype=float)

            if isinstance(node.op, ast.Add):
                return (as_series(left) + as_series(right)).astype(float)
            if isinstance(node.op, ast.Sub):
                return (as_series(left) - as_series(right)).astype(float)
            if isinstance(node.op, ast.Mult):
                return (as_series(left) * as_series(right)).astype(float)
            if isinstance(node.op, ast.Pow):
                return (as_series(left) ** as_series(right)).astype(float)
            if isinstance(node.op, ast.Mod):
                return (as_series(left) % as_series(right)).astype(float)
            raise ValueError(f"unsupported binop: {type(node.op).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only simple function calls allowed")
            fname = node.func.id
            if fname != "abs" or len(node.args) != 1 or node.keywords:
                raise ValueError(f"unsupported call: {fname}")
            v = ev(node.args[0])
            if isinstance(v, pd.Series):
                return pd.to_numeric(v, errors="coerce").abs().astype(float)
            return abs(float(v))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    result = ev(tree)
    if isinstance(result, pd.Series):
        return result.reindex(index).astype(float)
    return pd.Series(float(result), index=index, dtype=float)


def _compute_spec(
    df: pd.DataFrame,
    spec: DerivedSpec,
    db2var: Dict[str, str],
) -> pd.Series:
    # If any declared atomic column is missing, return all-NaN (do not guess other cols).
    missing = [
        db
        for db in spec.atomic_db_fields
        if _resolve_column(df, db_field=db, db2var=db2var) is None
    ]
    if missing:
        return _nan_series(df)
    env = _bind_formula_env(df, spec, db2var)
    try:
        return _eval_formula(spec.formula, env, df.index)
    except Exception:
        return _nan_series(df)


def _find_spec(specs: Sequence[DerivedSpec], derived_key: str) -> DerivedSpec:
    key = _compact_str(derived_key)
    for sp in specs:
        if sp.derived_key == key or sp.derived_name == key:
            return sp
        if sp.derived_key.lower() == key.lower():
            return sp
    raise KeyError(f"unknown derived_key: {derived_key}")


def list_derived_keys(folder: Optional[Path] = None) -> List[str]:
    folder2 = Path(folder) if folder is not None else _default_folder()
    return [sp.derived_key for sp in _load_derived_specs(folder2)]


def compute_one(df: pd.DataFrame, *, derived_key: str, folder: Optional[Path] = None) -> pd.Series:
    """Compute a single derived metric by derived_key (or derived_name)."""
    folder2 = Path(folder) if folder is not None else _default_folder()
    specs = _load_derived_specs(folder2)
    spec = _find_spec(specs, derived_key)
    db2var, _var2db = _load_catalog_alias_maps(folder2)
    return _compute_spec(df, spec, db2var)


def compute_all(
    df: pd.DataFrame,
    *,
    folder: Optional[Path] = None,
    return_both_keys: bool = True,
) -> Dict[str, pd.Series]:
    folder2 = Path(folder) if folder is not None else _default_folder()
    specs = _load_derived_specs(folder2)
    db2var, _var2db = _load_catalog_alias_maps(folder2)
    out: Dict[str, pd.Series] = {}
    for sp in specs:
        s = _compute_spec(df, sp, db2var)
        out[sp.derived_key] = s
        if return_both_keys and sp.derived_name and sp.derived_name not in out:
            out[sp.derived_name] = s
    return out


def compute_all_from_json(df: pd.DataFrame, *, folder: Optional[Path] = None) -> Dict[str, pd.Series]:
    """Alias: always driven by derived_metrics.json."""
    return compute_all(df, folder=folder, return_both_keys=True)


def ensure_derived_columns(
    df: pd.DataFrame,
    *,
    folder: Optional[Path] = None,
    keys: Optional[Sequence[str]] = None,
    write_both_names: bool = False,
) -> Dict[str, pd.Series]:
    folder2 = Path(folder) if folder is not None else _default_folder()
    specs = _load_derived_specs(folder2)
    if keys is not None:
        wanted: Set[str] = {_compact_str(k) for k in keys if _compact_str(k)}
        specs = [
            sp
            for sp in specs
            if sp.derived_key in wanted
            or sp.derived_name in wanted
            or sp.derived_key.lower() in {w.lower() for w in wanted}
        ]
    computed = compute_all(df, folder=folder2, return_both_keys=False)
    out: Dict[str, pd.Series] = {}
    for sp in specs:
        s = computed.get(sp.derived_key)
        if s is None:
            s = _nan_series(df)
        df[sp.derived_key] = s
        out[sp.derived_key] = s
        if write_both_names and sp.derived_name:
            df[sp.derived_name] = s
            out[sp.derived_name] = s
    return out


__all__ = [
    "compute_one",
    "compute_all",
    "compute_all_from_json",
    "ensure_derived_columns",
    "list_derived_keys",
    "safe_div",
]
