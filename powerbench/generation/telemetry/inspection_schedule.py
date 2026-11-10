from __future__ import annotations

import argparse
from typing import List, Optional

import pandas as pd


def nearest_in_index(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    if len(idx) == 0:
        return None
    pos = idx.get_indexer([pd.Timestamp(ts)], method="nearest")[0]
    if pos < 0:
        return None
    return pd.Timestamp(idx[pos])


def build_inspection_days(
    idx: pd.DatetimeIndex,
    *,
    count: int,
    interval_months: int,
    hour: int,
) -> List[pd.Timestamp]:
    """
    Build evenly spaced inspection anchors (e.g. every 2 months from timeline start),
    then snap each anchor to the nearest timestamp in idx.
    """
    if len(idx) == 0 or count <= 0:
        return []
    start = pd.Timestamp(idx[0]).normalize() + pd.Timedelta(hours=max(0, min(23, int(hour))))
    days: List[pd.Timestamp] = []
    cur = start
    for _ in range(int(count)):
        snap = nearest_in_index(idx, cur)
        if snap is not None:
            days.append(snap)
        cur = cur + pd.DateOffset(months=max(1, int(interval_months)))
    out: List[pd.Timestamp] = []
    seen = set()
    for d in days:
        k = str(pd.Timestamp(d))
        if k in seen:
            continue
        seen.add(k)
        out.append(pd.Timestamp(d))
    # Fallback for short timelines: month-based anchors may collapse to very few unique points.
    # In that case, fill up toward count (not exceeding available calendar days).
    target_n = min(max(1, int(count)), len(pd.to_datetime(pd.Series(idx.normalize()).drop_duplicates())))
    if len(out) < target_n:
        uniq_days = pd.to_datetime(pd.Series(idx.normalize()).drop_duplicates())
        aligned_days: List[pd.Timestamp] = []
        for d in uniq_days:
            anchor = pd.Timestamp(d) + pd.Timedelta(hours=max(0, min(23, int(hour))))
            snap = nearest_in_index(idx, anchor)
            if snap is not None:
                aligned_days.append(pd.Timestamp(snap))
        cand: List[pd.Timestamp] = []
        seen_cand = set()
        for d in aligned_days:
            k = str(pd.Timestamp(d))
            if k in seen_cand:
                continue
            seen_cand.add(k)
            cand.append(pd.Timestamp(d))
        if cand:
            need = target_n - len(out)
            remaining = [d for d in cand if str(pd.Timestamp(d)) not in seen]
            if need > 0 and remaining:
                take_n = min(need, len(remaining))
                if take_n == len(remaining):
                    picks = remaining
                else:
                    step = (len(remaining) - 1) / float(take_n - 1) if take_n > 1 else 0.0
                    idxs = sorted({int(round(i * step)) for i in range(take_n)})
                    while len(idxs) < take_n:
                        idxs.append(min(len(remaining) - 1, idxs[-1] + 1 if idxs else 0))
                    picks = [remaining[i] for i in idxs[:take_n]]
                for p in picks:
                    k = str(pd.Timestamp(p))
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(pd.Timestamp(p))
    out = sorted(out)
    return out


def add_inspection_schedule_arguments(ap: argparse.ArgumentParser) -> None:
    """Register CLI args shared by power pipeline generators (0002–0006 style)."""
    ap.add_argument("--inspection_count", type=int, default=12, help="number of inspection days (default 12)")
    ap.add_argument("--inspection_interval_months", type=int, default=2, help="inspection interval in months (default 2)")
    ap.add_argument("--inspection_hour", type=int, default=12, help="inspection anchor hour (0-23, default 12)")
    ap.add_argument("--alert_episodes", type=int, default=8, help="alert episode count (each spans multiple anomalies in a window)")
    ap.add_argument("--alerts_per_episode_min", type=int, default=2, help="minimum anomalies per alert episode")
    ap.add_argument("--alerts_per_episode_max", type=int, default=4, help="maximum anomalies per alert episode")
    ap.add_argument("--episode_span_hours", type=int, default=24, help="time window hours per alert episode / near-miss day")
    ap.add_argument("--near_miss_days", type=int, default=6, help="inspection days with near-miss events")
    ap.add_argument("--near_miss_per_day", type=int, default=3, help="max near-miss injections per inspection day (random 1..N)")
