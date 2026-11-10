from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from powerbench.generation.telemetry.metric_functions import (
    avg,
    consec_nh,
    count_true,
    duration_seconds,
    last_n_hours,
    rolling_window,
    slope_per_day,
    slope_per_hour,
    std,
)


Number = Union[int, float]


def _compact_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def load_metric_catalog(folder: Path) -> Dict[str, str]:
    """metric_name -> variable_name"""
    p = folder / "metric_catalog.json"
    if not p.exists():
        return {}
    items = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                cn = _compact_str(it.get("metric_name"))
                var = _compact_str(it.get("variable_name"))
                if cn and var:
                    out[cn] = var
    return out


def load_stage2(folder: Path) -> Dict[str, Any]:
    return json.loads((folder / "rule_metric_stage2.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CompiledRule:
    rule_id: str
    expr_py: str
    tree: ast.Expression
    name_env: Dict[str, str]
    params: Dict[str, float]
    cn2var: Dict[str, str]


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _normalize_expr(expr: str) -> str:
    """
    Make stage2 expression compatible with AST parsing.
    - AND/OR to python
    - "= true/false" to "== True/False"
    - ABS -> abs
    - tanδ -> tan_delta (identifier-safe)
    """
    s = _compact_str(expr)
    s = s.replace("AND", "and").replace("OR", "or")
    s = s.replace("= true", "== True").replace("= false", "== False")
    # Replace bare '=' used as equality in some expressions (e.g., grid_side_voltage='500kV')
    s = re.sub(r"(?<![<>=!])=(?!=)", "==", s)
    s = s.replace("ABS(", "abs(")
    s = s.replace("tanδ", "tan_delta")
    s = s.replace("rel_change_tanδ", "rel_change_tan_delta")
    return s


def _safe_metric_token(name: str) -> str:
    """
    Convert a metric (cn or var) into a python identifier token used in expressions.
    We keep existing snake_case; for Chinese names we hash to m_xxxxxxxx.
    """
    s = _compact_str(name)
    if not s:
        return "m_unknown"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
        return s
    # stable hash (FNV-1a)
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return f"m_{h:08x}"


@dataclass
class RuleEvalContext:
    df: pd.DataFrame  # indexed by timestamp (hourly)
    ts: pd.Timestamp
    cn2var: Dict[str, str]
    params: Dict[str, float]
    derived_series: Dict[str, pd.Series]
    fields: Dict[str, Any]

    def series(self, key: str) -> pd.Series:
        """
        key can be:
        - a metric Chinese name (will map via cn2var)
        - a variable name column
        - a derived metric name (if computed)
        """
        k = _compact_str(key)
        if not k:
            return pd.Series(dtype=float)
        if k in self.derived_series:
            return self.derived_series[k]
        if k in self.cn2var:
            col = self.cn2var[k]
        else:
            col = k
        if col not in self.df.columns:
            # create missing as NaN float
            self.df[col] = np.nan
        return self.df[col]

    def point(self, key: str) -> Any:
        s = self.series(key)
        try:
            return s.loc[self.ts]
        except KeyError:
            # if asked timestamp outside df index, return NaN (treat as missing)
            return float("nan")


class SafeExprEvaluator(ast.NodeVisitor):
    """
    Evaluate an expression AST with restricted nodes.
    Supports:
    - arithmetic, comparisons, boolean
    - membership: x in {'a','b'}
    - functions: consec_2h, consec_3h, consec_6h, duration_1h, duration_6h, duration_12h, duration_24h, duration_48h, avg_24h, std_24h, std_90d, slope_6h, slope_30d, slope_90d, slope_180d, slope_365d,
                 avg_7d, max_7d, min_7d, min_1p/max_1p (point), abs, avg_6h, avg_12h, max_6h, std_6h, count_12h
    - identifiers: metric vars / derived metric tokens / params
    """

    def __init__(self, ctx: RuleEvalContext, name_env: Dict[str, str]):
        self.ctx = ctx
        self.name_env = name_env  # token -> original metric key (cn or var or derived)

    # ---- literals ----
    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Set(self, node: ast.Set) -> Any:
        return {self.visit(elt) for elt in node.elts}

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)

    # ---- names ----
    def visit_Name(self, node: ast.Name) -> Any:
        ident = node.id
        if ident in {"True", "False"}:
            return ident == "True"
        if ident in self.ctx.params:
            return self.ctx.params[ident]
        if ident in self.ctx.fields:
            return self.ctx.fields[ident]
        if ident in self.name_env:
            # metric/derived token: return point value at ts
            return self.ctx.point(self.name_env[ident])
        # Fallback: allow direct dataframe column identifiers not captured in name_env.
        if ident in self.ctx.df.columns:
            return self.ctx.point(ident)
        # Fallback for Unicode identifiers produced by expression rewriting.
        if any(ord(ch) > 127 for ch in ident):
            # Common 0041 alias corruption pattern for ground-current to load-current ratio.
            if ("i_ground" in ident) and ("i_load" in ident) and ("ground_current_load_ratio" in self.ctx.df.columns):
                return self.ctx.point("ground_current_load_ratio")
            return self.ctx.point(ident)
        raise ValueError(f"unknown identifier: {ident}")

    # ---- ops ----
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.Not):
            return not bool(v)
        raise ValueError("unsupported unary op")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        a = self.visit(node.left)
        b = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if isinstance(node.op, ast.Mod):
            return a % b
        raise ValueError("unsupported bin op")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not bool(self.visit(v)):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if bool(self.visit(v)):
                    return True
            return False
        raise ValueError("unsupported bool op")

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
            ok = None
            if isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:
                raise ValueError("unsupported compare op")
            if not bool(ok):
                return False
            left = right
        return True

    # ---- calls ----
    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("only simple function calls allowed")
        fn = node.func.id
        args = [self.visit(a) for a in node.args]

        # Windowed computations operate on metric series, passed as identifier (Name).
        # To support that, we also allow raw metric token names as ast.Name in args by
        # looking up node.args in visit_Name (point). For window functions we need series,
        # so we special-case: if arg is a Name and in name_env, use series instead of point.
        def _series_from_node(n: ast.AST) -> pd.Series:
            # Name -> metric/derived series
            if isinstance(n, ast.Name):
                key = self.name_env.get(n.id, n.id)
                return self.ctx.series(key)
            # Constant -> scalar broadcast
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return pd.Series(float(n.value), index=self.ctx.df.index, dtype=float)
            # Unary +/- on series
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                s = _series_from_node(n.operand)
                return -s if isinstance(n.op, ast.USub) else +s
            # BinOp arithmetic on series
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                a = _series_from_node(n.left)
                b = _series_from_node(n.right)
                if isinstance(n.op, ast.Add):
                    return a + b
                if isinstance(n.op, ast.Sub):
                    return a - b
                if isinstance(n.op, ast.Mult):
                    return a * b
                if isinstance(n.op, ast.Div):
                    return a / b
            raise ValueError(f"cannot build series from node: {type(n).__name__}")

        ts = self.ctx.ts
        df = self.ctx.df

        if fn == "abs":
            return float(abs(args[0]))
        if fn == "consec_3h":
            cond_series = self._recompute_bool_series(node.args[0], window="3h")
            w = last_n_hours(pd.DataFrame({"c": cond_series}), ts, 3)["c"]
            return consec_nh(w, 3)
        if fn == "consec_2h":
            cond_series = self._recompute_bool_series(node.args[0], window="2h")
            w = last_n_hours(pd.DataFrame({"c": cond_series}), ts, 2)["c"]
            return consec_nh(w, 2)
        if fn == "consec_12h":
            cond_series = self._recompute_bool_series(node.args[0], window="12h")
            w = last_n_hours(pd.DataFrame({"c": cond_series}), ts, 12)["c"]
            return consec_nh(w, 12)
        if fn == "consec_24h":
            cond_series = self._recompute_bool_series(node.args[0], window="24h")
            w = last_n_hours(pd.DataFrame({"c": cond_series}), ts, 24)["c"]
            return consec_nh(w, 24)
        if fn == "consec_6h":
            cond_series = self._recompute_bool_series(node.args[0], window="6h")
            w = last_n_hours(pd.DataFrame({"c": cond_series}), ts, 6)["c"]
            return consec_nh(w, 6)
        if fn == "duration_1h":
            cond_series = self._recompute_bool_series(node.args[0], window="1h")
            w = rolling_window(pd.DataFrame({"c": cond_series}), ts, "1h")["c"]
            return duration_seconds(w)
        if fn == "duration_6h":
            cond_series = self._recompute_bool_series(node.args[0], window="6h")
            w = rolling_window(pd.DataFrame({"c": cond_series}), ts, "6h")["c"]
            return duration_seconds(w)
        if fn == "duration_24h":
            cond_series = self._recompute_bool_series(node.args[0], window="24h")
            w = rolling_window(pd.DataFrame({"c": cond_series}), ts, "24h")["c"]
            return duration_seconds(w)
        if fn == "duration_12h":
            cond_series = self._recompute_bool_series(node.args[0], window="12h")
            w = rolling_window(pd.DataFrame({"c": cond_series}), ts, "12h")["c"]
            return duration_seconds(w)
        if fn == "duration_48h":
            cond_series = self._recompute_bool_series(node.args[0], window="48h")
            w = rolling_window(pd.DataFrame({"c": cond_series}), ts, "48h")["c"]
            return duration_seconds(w)
        m_dur = re.fullmatch(r"duration_(\d+)([hd])", fn)
        if m_dur:
            win = f"{int(m_dur.group(1))}{m_dur.group(2)}"
            cond_series = self._recompute_bool_series(node.args[0], window=win)
            w = rolling_window(pd.DataFrame({"c": cond_series}), ts, win)["c"]
            return duration_seconds(w)
        if fn == "unchanged":
            s = _series_from_node(node.args[0])
            return s.eq(s.shift(1)).fillna(False)
        if fn == "avg_24h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "24h")["x"]
            return avg(w)
        if fn == "avg_72h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "72h")["x"]
            return avg(w)
        if fn == "std_72h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "72h")["x"]
            return std(w)
        if fn == "avg_48h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "48h")["x"]
            return avg(w)
        if fn == "std_48h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "48h")["x"]
            return std(w)
        if fn == "std_24h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "24h")["x"]
            return std(w)
        if fn == "std_90d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "90d")["x"]
            return std(w)
        if fn == "slope_30d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "30d")["x"]
            return slope_per_day(w)
        if fn == "slope_90d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "90d")["x"]
            return slope_per_day(w)
        if fn == "slope_180d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "180d")["x"]
            return slope_per_day(w)
        if fn == "slope_365d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "365d")["x"]
            return slope_per_day(w)
        if fn == "count_365d":
            cond_series = self._recompute_bool_series(node.args[0], window="365d")
            w = rolling_window(pd.DataFrame({"c": cond_series}), ts, "365d")["c"]
            return count_true(w)
        if fn == "avg_7d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "7d")["x"]
            return avg(w)
        if fn == "slope_7d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "7d")["x"]
            return slope_per_day(w)
        if fn == "std_12h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "12h")["x"]
            return std(w)
        if fn == "slope_24h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "24h")["x"]
            return slope_per_hour(w)
        if fn == "slope_6h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "6h")["x"]
            return slope_per_hour(w)
        if fn == "slope_1h":
            s = _series_from_node(node.args[0])
            w = last_n_hours(pd.DataFrame({"x": s}), ts, 2)["x"]
            if len(w) < 2:
                return float("nan")
            return float(float(w.iloc[-1]) - float(w.iloc[0]))
        if fn == "corr_6h":
            a = _series_from_node(node.args[0])
            b = _series_from_node(node.args[1])
            wa = rolling_window(pd.DataFrame({"x": a}), ts, "6h")["x"]
            wb = rolling_window(pd.DataFrame({"x": b}), ts, "6h")["x"]
            if len(wa) < 2 or len(wb) < 2:
                return float("nan")
            pair = pd.DataFrame({"a": pd.to_numeric(wa, errors="coerce"), "b": pd.to_numeric(wb, errors="coerce")}).dropna()
            if len(pair) < 2:
                return float("nan")
            return float(pair["a"].corr(pair["b"]))
        if fn == "prev_1h":
            s = _series_from_node(node.args[0])
            pos = s.index.get_indexer([pd.Timestamp(ts)], method="nearest")[0]
            if pos <= 0:
                return float("nan")
            return float(pd.to_numeric(s.iloc[pos - 1], errors="coerce"))
        if fn == "max_1h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "1h")["x"]
            return float(w.max()) if len(w) else float("nan")
        if fn == "max_6h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "6h")["x"]
            return float(w.max()) if len(w) else float("nan")
        if fn == "max_24h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "24h")["x"]
            return float(w.max()) if len(w) else float("nan")
        if fn == "avg_1h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "1h")["x"]
            return avg(w)
        if fn == "avg_6h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "6h")["x"]
            return avg(w)
        if fn == "avg_12h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "12h")["x"]
            return avg(w)
        if fn == "std_6h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "6h")["x"]
            return std(w)
        if fn == "std_1h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "1h")["x"]
            return std(w)
        if fn == "count_1h":
            cond_series = self._recompute_bool_series(node.args[0], window="1h")
            return int(cond_series.fillna(False).sum())
        if fn == "count_6h":
            cond_series = self._recompute_bool_series(node.args[0], window="6h")
            return int(cond_series.fillna(False).sum())
        if fn == "count_12h":
            cond_series = self._recompute_bool_series(node.args[0], window="12h")
            return int(cond_series.fillna(False).sum())
        if fn == "count_24h":
            cond_series = self._recompute_bool_series(node.args[0], window="24h")
            return int(cond_series.fillna(False).sum())
        if fn == "count_48h":
            cond_series = self._recompute_bool_series(node.args[0], window="48h")
            return int(cond_series.fillna(False).sum())
        if fn == "max_duration_24h":
            cond_series = self._recompute_bool_series(node.args[0], window="24h")
            vals = cond_series.fillna(False).astype(bool).tolist()
            best = 0
            cur = 0
            for v in vals:
                if v:
                    cur += 1
                    if cur > best:
                        best = cur
                else:
                    cur = 0
            return float(best * 3600)
        if fn == "count_14d":
            cond_series = self._recompute_bool_series(node.args[0], window="14d")
            return int(cond_series.fillna(False).sum())
        if fn == "max_7d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "7d")["x"]
            return float(w.max())
        if fn == "min_7d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "7d")["x"]
            return float(w.min())
        if fn == "max_30d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "30d")["x"]
            return float(w.max())
        if fn == "std_30d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "30d")["x"]
            return std(w)
        if fn == "max_1p":
            s = _series_from_node(node.args[0])
            return float(s.loc[ts])
        if fn == "min_1p":
            s = _series_from_node(node.args[0])
            return float(s.loc[ts])
        if fn == "slope_12h":
            s = _series_from_node(node.args[0])
            w = last_n_hours(pd.DataFrame({"x": s}), ts, 13)["x"]
            if len(w) < 2:
                return float("nan")
            return float((float(w.iloc[-1]) - float(w.iloc[0])) / 12.0)
        if fn == "rate_12h":
            s = _series_from_node(node.args[0])
            w = last_n_hours(pd.DataFrame({"x": s}), ts, 13)["x"]
            if len(w) < 2:
                return float("nan")
            return float((float(w.iloc[-1]) - float(w.iloc[0])) / 12.0)
        if fn == "max_12h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "12h")["x"]
            return float(w.max()) if len(w) else float("nan")
        if fn == "slope_14d":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "14d")["x"]
            return slope_per_day(w)
        if fn == "max_abs_delta_1h_over_24h":
            s = _series_from_node(node.args[0])
            w = rolling_window(pd.DataFrame({"x": s}), ts, "24h")["x"]
            if len(w) < 2:
                return float("nan")
            vals = w.astype(float).to_numpy()
            diffs = np.abs(np.diff(vals))
            return float(np.max(diffs)) if len(diffs) else float("nan")
        if fn == "max":
            if len(args) < 2:
                raise ValueError("max() needs at least 2 args")
            return float(max(args))

        raise ValueError(f"unsupported function: {fn}")

    def _recompute_bool_series(self, expr_node: ast.AST, window: str) -> pd.Series:
        """
        Strictly recompute a boolean series for a subexpression over a window ending at ctx.ts.
        This is needed for consec_3h(cond) / duration_1h(cond) where cond is an expression.
        """
        ts = self.ctx.ts
        # build window index
        end = pd.Timestamp(ts)
        unit = window[-1].lower()
        num = int(window[:-1])
        if unit == "h":
            idx = pd.date_range(end=end, periods=num, freq="h")
        elif unit == "d":
            idx = pd.date_range(end=end, periods=num * 24, freq="h")
        else:
            raise ValueError(f"bad window: {window}")

        out = []
        for t in idx:
            sub_ctx = RuleEvalContext(
                df=self.ctx.df,
                ts=pd.Timestamp(t),
                cn2var=self.ctx.cn2var,
                params=self.ctx.params,
                derived_series=self.ctx.derived_series,
                fields=self.ctx.fields,
            )
            sub_eval = SafeExprEvaluator(sub_ctx, self.name_env)
            out.append(bool(sub_eval.visit(expr_node)))
        return pd.Series(out, index=idx)


def build_name_env(expr: str, cn2var: Dict[str, str], derived_names: List[str]) -> Tuple[str, Dict[str, str]]:
    """
    Replace Chinese metric names (and derived metric names) with safe python tokens.
    Returns (rewritten_expr, token->original_name)
    """
    s = _normalize_expr(expr)
    # Compat DSL: "metric unchanged" -> "unchanged(metric)"
    # Example: duration_1095d(tap_position unchanged)
    s = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+unchanged\b", r"unchanged(\1)", s)
    name_env: Dict[str, str] = {}

    # replace derived names first (longest-first)
    for dn in sorted(set(derived_names), key=len, reverse=True):
        if not dn:
            continue
        tok = _safe_metric_token(dn)
        name_env[tok] = dn
        s = s.replace(dn, tok)

    # replace Chinese metric names with their var_name token (if available), else hashed token
    for cn in sorted(cn2var.keys(), key=len, reverse=True):
        var = cn2var[cn]
        tok = _safe_metric_token(var)
        name_env[tok] = cn  # point/series should use CN -> var mapping
        s = s.replace(cn, tok)

    # Replace any remaining CJK identifiers (e.g., legacy CN field names) with safe tokens and bind to themselves.
    # IMPORTANT: do NOT touch CJK inside string literals like {'fault','alarm'}.
    _lit_pat = re.compile(r"(?:'[^'\\\\]*(?:\\\\.[^'\\\\]*)*')|(?:\"[^\"\\\\]*(?:\\\\.[^\"\\\\]*)*\")")
    literals: List[str] = []

    def _mask(m: re.Match) -> str:
        literals.append(m.group(0))
        return f"__LIT_{len(literals)-1}__"

    masked = _lit_pat.sub(_mask, s)
    for cn_ident in sorted(set(re.findall(r"[\u4e00-\u9fff]{2,}", masked)), key=len, reverse=True):
        tok = _safe_metric_token(cn_ident)
        if tok not in name_env:
            name_env[tok] = cn_ident
        masked = masked.replace(cn_ident, tok)

    # restore string literals
    for i, lit in enumerate(literals):
        masked = masked.replace(f"__LIT_{i}__", lit)
    s = masked

    # Also allow bare variable identifiers appearing in expressions (e.g., top_oil_temp, tan_delta, PD, P_SF6).
    # We bind them to themselves, so Name -> point df[col].loc[ts].
    for ident in sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", s)), key=len, reverse=True):
        if ident in {"and", "or", "not", "True", "False"}:
            continue
        if ident in name_env:
            continue
        # ignore known function names and obvious param names
        if ident in {
            "consec_3h",
            "consec_2h",
            "consec_6h",
            "consec_12h",
            "consec_24h",
            "duration_1h",
            "duration_6h",
            "duration_12h",
            "duration_24h",
            "duration_48h",
            "avg_24h",
            "avg_48h",
            "avg_72h",
            "std_24h",
            "std_12h",
            "std_48h",
            "std_72h",
            "std_90d",
            "slope_12h",
            "slope_14d",
            "slope_24h",
            "slope_6h",
            "slope_1h",
            "corr_6h",
            "slope_7d",
            "slope_30d",
            "slope_90d",
            "slope_180d",
            "slope_365d",
            "count_365d",
            "count_1h",
            "count_6h",
            "count_12h",
            "count_24h",
            "count_48h",
            "max_duration_24h",
            "count_14d",
            "avg_7d",
            "max_7d",
            "min_7d",
            "max_1h",
            "max_6h",
            "max_24h",
            "max_12h",
            "avg_1h",
            "avg_6h",
            "avg_12h",
            "std_6h",
            "std_1h",
            "max_30d",
            "std_30d",
            "max_1p",
            "min_1p",
            "max_abs_delta_1h_over_24h",
            "rate_12h",
            "unchanged",
            "prev_1h",
            "max",
            "abs",
        }:
            continue
        # bind identifier directly to df column with same name
        name_env[ident] = ident

    return s, name_env


def _slope_14d_vectorized(series: pd.Series) -> pd.Series:
    """
    Per-timestamp slope_per_day over trailing 14d hourly window (336 points max).
    Uses rolling raw apply to avoid O(n²) DataFrame.loc slicing per timestamp.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)

    def _win_slope_per_day(y: np.ndarray) -> float:
        m = np.isfinite(y)
        yy = y[m].astype(float, copy=False)
        if yy.shape[0] < 3:
            return float("nan")
        x = np.arange(yy.shape[0], dtype=float) / 24.0
        try:
            a, _b = np.polyfit(x, yy, 1)
            return float(a)
        except Exception:
            return float("nan")

    return s.rolling(336, min_periods=48).apply(_win_slope_per_day, raw=True)


def _derived_eval_namespace(
    df: pd.DataFrame,
    out: Dict[str, pd.Series],
    params: Dict[str, float],
) -> Dict[str, Any]:
    def _num(s: Any) -> pd.Series:
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors="coerce").reindex(df.index)
        return pd.to_numeric(pd.Series(s, index=df.index), errors="coerce")

    def avg_72h(col: Any) -> pd.Series:
        s = _num(col)
        return s.rolling(72, min_periods=2).mean()

    def std_72h(col: Any) -> pd.Series:
        s = _num(col)
        return s.rolling(72, min_periods=2).std(ddof=0)

    def avg_48h(col: Any) -> pd.Series:
        s = _num(col)
        return s.rolling(48, min_periods=2).mean()

    def std_48h(col: Any) -> pd.Series:
        s = _num(col)
        return s.rolling(48, min_periods=2).std(ddof=0)

    def avg_12h(col: Any) -> pd.Series:
        s = _num(col)
        return s.rolling(12, min_periods=2).mean()

    def std_12h_derived(col: Any) -> pd.Series:
        s = _num(col)
        return s.rolling(12, min_periods=2).std(ddof=0)

    def slope_12h(col: Any) -> pd.Series:
        s = _num(col)
        return (s - s.shift(12)) / 12.0

    def max_12h(col: Any) -> pd.Series:
        s = _num(col)
        return s.rolling(12, min_periods=1).max()

    ns: Dict[str, Any] = {
        "__builtins__": {},
        "df": df,
        "out": out,
        "np": np,
        "abs": np.abs,
        "avg_72h": avg_72h,
        "std_72h": std_72h,
        "avg_48h": avg_48h,
        "std_48h": std_48h,
        "avg_12h": avg_12h,
        "std_12h": std_12h_derived,
        "slope_12h": slope_12h,
        "max_12h": max_12h,
        "slope_14d": _slope_14d_vectorized,
        "np.maximum": np.maximum,
    }
    for k, v in params.items():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            ns[str(k)] = float(v)
    return ns


def compute_derived_series(
    folder: Path,
    df: pd.DataFrame,
    cn2var: Dict[str, str],
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, pd.Series]:
    """
    Compute derived metric series defined in stage2: derived_metric_definitions[*].derived_metric_name with formula.
    Supports window helpers (avg_72h/std_72h, slope_12h, max_12h, slope_14d) and threshold params in formulas.
    """
    stage2 = load_stage2(folder)
    rules = stage2.get("anomaly_judgment_rules") or []
    derived_defs: Dict[str, str] = {}
    if isinstance(rules, list):
        for r in rules:
            if isinstance(r, dict):
                for d in r.get("derived_metric_definitions", []) or []:
                    if isinstance(d, dict):
                        name = _compact_str(d.get("derived_metric_name"))
                        formula = _compact_str(d.get("formula"))
                        if name and formula:
                            derived_defs[name] = formula

    out: Dict[str, pd.Series] = {}
    p = dict(params or {})

    for name, formula in derived_defs.items():
        f = _normalize_expr(formula)
        expr = f
        # stage2 uses max(a,b) for element-wise series; map to np.maximum(a,b)
        expr = re.sub(r"(?<![a-zA-Z0-9_])max\s*\(", "np.maximum(", expr)
        # Replace explicit cn metric names
        for cn in sorted(cn2var.keys(), key=len, reverse=True):
            var = cn2var[cn]
            expr = expr.replace(cn, f"df[{var!r}]")
        # Replace bare identifiers that are columns
        for col in list(df.columns):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(col)):
                expr = re.sub(rf"\\b{re.escape(str(col))}\\b", f"df[{str(col)!r}]", expr)
        # Replace derived names
        for dn in sorted(out.keys(), key=len, reverse=True):
            expr = expr.replace(dn, f"out[{dn!r}]")
        expr = expr.replace(" ", "")
        try:
            ns = _derived_eval_namespace(df, out, p)
            # noqa: S307 (controlled locals)
            series = eval(expr, {"__builtins__": {}}, ns)
            if isinstance(series, pd.Series):
                out[name] = series.reindex(df.index)
        except Exception:
            continue

    return out


def eval_rule_strict(
    folder: Path,
    df: pd.DataFrame,
    ts: pd.Timestamp,
    rule: Dict[str, Any],
    fields_override: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    compiled = compile_rule(folder, rule)
    triggered = eval_compiled_rule(folder, df, ts, rule, compiled, fields_override=fields_override)
    return triggered, {"expr_py": compiled.expr_py, "params": compiled.params}


def compile_rule(folder: Path, rule: Dict[str, Any]) -> CompiledRule:
    cn2var = load_metric_catalog(folder)
    params_raw = {str(p.get("param_name")): _to_float(p.get("param_value")) for p in (rule.get("threshold_params") or []) if isinstance(p, dict)}
    params: Dict[str, float] = {}
    for k, v in params_raw.items():
        params[_normalize_expr(k)] = v

    expr = _compact_str(rule.get("rule_expression"))
    derived_names = []
    for x in (rule.get("derived_metric_definitions") or []):
        if isinstance(x, dict):
            dn = _compact_str(x.get("derived_metric_name") or x.get("name"))
            if dn:
                derived_names.append(dn)
    expr2, name_env = build_name_env(expr, cn2var, derived_names)
    tree = ast.parse(expr2, mode="eval")
    rid = _compact_str(rule.get("rule_id")) or "UNKNOWN"
    return CompiledRule(rule_id=rid, expr_py=expr2, tree=tree, name_env=name_env, params=params, cn2var=cn2var)


def eval_compiled_rule(
    folder: Path,
    df: pd.DataFrame,
    ts: pd.Timestamp,
    rule: Dict[str, Any],
    compiled: CompiledRule,
    fields_override: Optional[Dict[str, Any]] = None,
) -> bool:
    derived = compute_derived_series(folder, df, compiled.cn2var, params=compiled.params)
    fields = {"grid_side_voltage": "500kV"}
    if fields_override:
        fields.update(fields_override)
    ctx = RuleEvalContext(
        df=df,
        ts=pd.Timestamp(ts),
        cn2var=compiled.cn2var,
        params=compiled.params,
        derived_series=derived,
        fields=fields,
    )
    ev = SafeExprEvaluator(ctx, compiled.name_env)
    return bool(ev.visit(compiled.tree.body))

