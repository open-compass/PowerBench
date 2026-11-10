from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


def last_n_hours(df: pd.DataFrame, ts: pd.Timestamp, n_hours: int) -> pd.DataFrame:
    """Return window ending at ts inclusive with length n_hours (hourly frequency expected)."""
    end = pd.Timestamp(ts)
    start = end - pd.Timedelta(hours=n_hours - 1)
    return df.loc[start:end]


def rolling_window(df: pd.DataFrame, ts: pd.Timestamp, window: str) -> pd.DataFrame:
    """window like '24h', '30d', '90d', '365d', '7d', '180d'."""
    end = pd.Timestamp(ts)
    unit = window[-1].lower()
    num = int(window[:-1])
    if unit == "h":
        start = end - pd.Timedelta(hours=num - 1)
    elif unit == "d":
        start = end - pd.Timedelta(days=num) + pd.Timedelta(hours=1)
    else:
        raise ValueError(f"unsupported window: {window}")
    # If we don't have full history in df, treat as insufficient window.
    # This prevents early-period false triggers for long-window rules.
    if not df.index.empty and start < df.index.min():
        return df.iloc[0:0]
    return df.loc[start:end]


def consec_nh(cond: pd.Series, n: int) -> bool:
    """Whether last n points are all True."""
    if cond is None or len(cond) < n:
        return False
    tail = cond.tail(n)
    return bool(tail.fillna(False).all())


def duration_seconds(cond: pd.Series) -> float:
    """
    Given hourly boolean series (indexed by timestamp), return duration of trailing True-run in seconds.
    """
    if cond is None or cond.empty:
        return 0.0
    tail = cond.fillna(False)
    # count trailing True values
    cnt = 0
    for v in reversed(tail.tolist()):
        if v is True:
            cnt += 1
        else:
            break
    return float(cnt * 3600)


def avg(series: pd.Series) -> float:
    return float(series.mean()) if series is not None and len(series) else float("nan")


def std(series: pd.Series) -> float:
    return float(series.std(ddof=0)) if series is not None and len(series) else float("nan")


def slope_per_day(series: pd.Series) -> float:
    """
    Linear regression slope in units per day.
    Hourly index expected.
    """
    s = series.dropna() if series is not None else None
    if s is None or s.shape[0] < 3:
        return float("nan")
    y = s.astype(float).to_numpy()
    x = np.arange(len(y), dtype=float) / 24.0  # days
    # fit y = a*x + b
    try:
        a, _b = np.polyfit(x, y, 1)
    except Exception:
        # numerical fallback
        a = (y[-1] - y[0]) / max(1e-9, (x[-1] - x[0]))
    return float(a)


def slope_per_hour(series: pd.Series) -> float:
    """
    Linear regression slope in units per 1-hour step (hourly index).
    Used for slope_24h(...) style rules (e.g. dB/h, %/h).
    """
    s = series.dropna() if series is not None else None
    if s is None or s.shape[0] < 2:
        return float("nan")
    y = s.astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    try:
        a, _b = np.polyfit(x, y, 1)
    except Exception:
        a = (y[-1] - y[0]) / max(1.0, float(len(y) - 1))
    return float(a)


def count_true(cond: pd.Series) -> int:
    if cond is None or cond.empty:
        return 0
    return int(cond.fillna(False).sum())


def abs_(x: float) -> float:
    return float(abs(x))


def safe_div(a: float, b: float, default: float = float("nan")) -> float:
    if b == 0 or (isinstance(b, float) and math.isfinite(b) is False):
        return default
    return float(a / b)

