#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-anchor cycle estimator (v6)
- Computes period from GREEN-start spacing and RED-start spacing per group.
- Reconciles them per group, then aggregates to a single GLOBAL period.
- Prints ONLY the final global period (integer seconds) to stdout.
- Saves rich per-group stats for plotting.

IO fixed to /home/tzhang174/EVData_XGame/data
Inputs expected:
- wait.csv (red intervals), with columns: time_stamp, end_time, road_id, date, seg_id, direction (micro, e.g. A1-1), direction_group (A1/A2)
- cross_over_time_A.csv (green intervals for A group), with columns: enter_time, exit_time, road_id, date, seg_id, direction (micro), direction_group (A1/A2)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import os, json, sys

# ---------------- Fixed configuration ----------------
BASE_DIR = "/home/tzhang174/EVData_XGame/data"
WAIT_FILE = os.path.join(BASE_DIR, "wait.csv")                       # red
# Use A-group green intervals exclusively
CROSS_FILE_A = os.path.join(BASE_DIR, "cross_over_time_A.csv")       # green (A group only)
# Real (A1) green intervals (file name kept consistent with generator)
REAL_CROSS_FILE = os.path.join(BASE_DIR, "corss_timeA_real.csv")
# Fallback alternate name if needed
REAL_CROSS_FILE_FALLBACK = os.path.join(BASE_DIR, "cross_over_timeA_real.csv")
# With-outer (union including outer-adjacent fallback intervals)
CROSS_FILE_WITH_OUTER = os.path.join(BASE_DIR, "cross_over_time_with_outer.csv")
# All groups (A/B) unified intervals
CROSS_FILE_ALL = os.path.join(BASE_DIR, "cross_over_time.csv")
# Try file (ONLY outer-adjacent intervals)
CROSS_FILE_TRY = os.path.join(BASE_DIR, "cross_over_time_try.csv")

OUT_PER_GROUP_CSV = os.path.join(BASE_DIR, "period_stats_per_group.csv")
OUT_PER_GROUP_JSON = os.path.join(BASE_DIR, "period_stats_per_group.json")
OUT_GLOBAL_JSON = os.path.join(BASE_DIR, "period_stats_global.json")

TIME_FMT = "%H:%M:%S"

# Controls
TOL_ABS_S = 6.0            # green/red period agreement tolerance in seconds
MIN_PAIRS = 3              # minimal pairs to trust spacing median
IQR_CLAMP = 1.5            # clamp factor for robust median
# -----------------------------------------------------

@dataclass
class Interval:
    start: float
    end: float

def parse_time_to_sec(t: str) -> float:
    dt = datetime.strptime(t, TIME_FMT)
    return dt.hour*3600 + dt.minute*60 + dt.second

def parse_time_to_sec_any(t: str) -> Optional[float]:
    """
    Parse time-of-day from:
      - 'HH:MM' or 'HH:MM:SS'
      - 'YYYY-MM-DD HH:MM:SS[.ffffff]'
    Returns seconds since midnight or None if unparsable.
    """
    if t is None:
        return None
    s = str(t).strip()
    if not s or s.lower() == "nat":
        return None
    # Quick parse for HH:MM or HH:MM:SS
    try:
        parts = s.split(":")
        if len(parts) == 2:
            h, m = int(parts[0]), int(parts[1])
            return float(h * 3600 + m * 60)
        if len(parts) >= 3 and " " not in parts[0]:
            h, m = int(parts[0]), int(parts[1])
            sec_part = parts[2]
            if "." in sec_part:
                sec_part = sec_part.split(".")[0]
            sec = int(sec_part)
            return float(h * 3600 + m * 60 + sec)
    except Exception:
        pass
    # Fallback: full timestamp parsing
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return float(dt.hour*3600 + dt.minute*60 + dt.second)
    except Exception:
        return None

def load_wait(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.rename(columns={'time_stamp':'start_time', 'end_time':'end_time'}, inplace=True)
    for col in ["road_id","direction","seg_id","date"]:
        if col not in df.columns:
            df[col] = "NA"
    # Ensure direction_group exists (A1/A2) for downstream grouping if desired
    if "direction_group" not in df.columns:
        up = df["direction"].astype(str).str.upper()
        df["direction_group"] = up.apply(lambda s: "A1" if s.startswith("A1") else ("A2" if s.startswith("A2") else "NA"))
    return df

def load_cross(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.rename(columns={'enter_time':'start_time', 'exit_time':'end_time'}, inplace=True)
    for col in ["road_id","direction","seg_id","date"]:
        if col not in df.columns:
            df[col] = "NA"
    if "direction_group" not in df.columns:
        up = df["direction"].astype(str).str.upper()
        df["direction_group"] = up.apply(lambda s: "A1" if s.startswith("A1") else ("A2" if s.startswith("A2") else "NA"))
    return df

def make_group_key(row: pd.Series) -> str:
    # Prefer grouping by A1/A2 if available; fallback to micro direction
    dir_val = row.get('direction_group', None)
    if dir_val is None or str(dir_val).strip() == "":
        dir_val = row.get('direction', 'NA')
    return f"{row.get('road_id','NA')}|{dir_val}|{row.get('seg_id','NA')}|{row.get('date','NA')}"

def build_intervals(df: pd.DataFrame) -> List[Interval]:
    out: List[Interval] = []
    for _, r in df.iterrows():
        try:
            s_val = parse_time_to_sec_any(r.get('start_time'))
            e_val = parse_time_to_sec_any(r.get('end_time'))
            if s_val is None or e_val is None:
                continue
            if e_val > s_val:
                out.append(Interval(float(s_val), float(e_val)))
        except Exception:
            continue
    if not out:
        return out
    out.sort(key=lambda x: x.start)
    # merge within color
    merged = [out[0]]
    for it in out[1:]:
        last = merged[-1]
        if it.start <= last.end + 1e-6:
            last.end = max(last.end, it.end)
        else:
            merged.append(Interval(it.start, it.end))
    return merged

def robust_median(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float('nan')
    q25 = np.percentile(arr, 25); q75 = np.percentile(arr, 75)
    iqr = q75 - q25
    if iqr <= 0:
        return float(np.median(arr))
    lo, hi = q25 - IQR_CLAMP*iqr, q75 + IQR_CLAMP*iqr
    f = arr[(arr>=lo) & (arr<=hi)]
    if f.size == 0:
        f = arr
    return float(np.median(f))

def spacing_median(starts: List[float]) -> Tuple[float, int]:
    if len(starts) < 2:
        return float('nan'), 0
    s = np.diff(np.array(sorted(starts), dtype=float))
    return robust_median(s), int(len(s))

def durations_median(intervals: List[Interval]) -> float:
    if not intervals:
        return float('nan')
    d = np.array([iv.end - iv.start for iv in intervals], dtype=float)
    return robust_median(d)

def process_group(wait_g: pd.DataFrame, cross_g: pd.DataFrame, key: str):
    # Parse direction group from key for output tagging
    dir_group = "NA"
    try:
        parts = str(key).split("|")
        if len(parts) >= 2:
            dir_group = parts[1]
    except Exception:
        pass
    # Intervals
    reds = build_intervals(wait_g)
    greens = build_intervals(cross_g)

    red_starts = [iv.start for iv in reds]
    green_starts = [iv.start for iv in greens]

    green_period, n_g = spacing_median(green_starts)
    red_period, n_r = spacing_median(red_starts)

    green_med = durations_median(greens)
    red_med = durations_median(reds)

    # reconcile
    reconciled = float('nan'); method = "none"; agree = False
    if n_g >= MIN_PAIRS and np.isfinite(green_period) and n_r >= MIN_PAIRS and np.isfinite(red_period):
        if abs(green_period - red_period) <= TOL_ABS_S:
            reconciled = 0.5*(green_period + red_period)
            method = "fuse_green_red"
            agree = True
        else:
            if n_g > n_r:
                reconciled = green_period; method = "green_only_disagree"
            elif n_r > n_g:
                reconciled = red_period; method = "red_only_disagree"
            else:
                # tie: pick the one closer to green_med+red_med as weak prior
                prior = green_med + red_med if np.isfinite(green_med) and np.isfinite(red_med) else np.nan
                if np.isfinite(prior):
                    if abs(green_period - prior) <= abs(red_period - prior):
                        reconciled = green_period; method = "green_only_tie_prior"
                    else:
                        reconciled = red_period; method = "red_only_tie_prior"
                else:
                    reconciled = green_period; method = "green_only_tie"
    elif n_g >= MIN_PAIRS and np.isfinite(green_period):
        reconciled = green_period; method = "green_only"
    elif n_r >= MIN_PAIRS and np.isfinite(red_period):
        reconciled = red_period; method = "red_only"
    else:
        method = "insufficient_pairs"

    # weights for global aggregation (prefer groups with more pairs on both sides)
    weight = min(n_g, n_r) if (n_g>0 and n_r>0) else max(n_g, n_r)

    row = {
        "group_key": key,
        "direction_group": dir_group,
        "green_period_s": green_period,
        "red_period_s": red_period,
        "reconciled_period_s": reconciled,
        "pairs_green": n_g,
        "pairs_red": n_r,
        "weight": weight,
        "agree_within_tol": agree,
        "method": method,
        "green_median_dur_s": green_med,
        "red_median_dur_s": red_med
    }
    return row

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float('nan')
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    cum = np.cumsum(w) / np.sum(w) if np.sum(w)>0 else np.cumsum(np.ones_like(w))/len(w)
    idx = np.searchsorted(cum, 0.5)
    idx = min(idx, len(v)-1)
    return float(v[idx])

def compute_real_period() -> Optional[float]:
    """
    Compute a global green period using A1 real crossover intervals only.
    Strategy:
      - Load REAL_CROSS_FILE (fallback to REAL_CROSS_FILE_FALLBACK)
      - Group by group_key (road|A1|seg|date) and compute per-group spacing median on starts
      - Aggregate with weighted median by pair counts; fallback to aggregated starts if needed
    Returns period in seconds (float) or None if not computable.
    """
    path = None
    if os.path.exists(REAL_CROSS_FILE):
        path = REAL_CROSS_FILE
    elif os.path.exists(REAL_CROSS_FILE_FALLBACK):
        path = REAL_CROSS_FILE_FALLBACK
    if path is None:
        return None
    try:
        df = load_cross(path)
    except Exception:
        return None
    # Force A1-only if direction present
    try:
        up = df["direction"].astype(str).str.upper()
        df = df[up.str.startswith("A1")]
    except Exception:
        pass
    if df.empty:
        return None
    df["group_key"] = df.apply(make_group_key, axis=1)
    per_periods: List[float] = []
    per_weights: List[int] = []
    # compute per-group green spacing median and pair counts
    for key, g in df.groupby("group_key"):
        greens = build_intervals(g)
        starts = [iv.start for iv in greens]
        p, n_pairs = spacing_median(starts)
        if np.isfinite(p) and n_pairs >= MIN_PAIRS:
            per_periods.append(float(p))
            per_weights.append(int(n_pairs))
    if len(per_periods) > 0:
        return weighted_median(np.array(per_periods, dtype=float), np.array(per_weights, dtype=float))
    # Fallback: aggregate all starts
    greens = build_intervals(df)
    starts = sorted([iv.start for iv in greens])
    p, _ = spacing_median(starts)
    return float(p) if np.isfinite(p) else None

def compute_period_for_road_from_cross(cross_path: str, road_id: str, restrict_A_group: bool = True) -> Optional[float]:
    """
    Compute a global period for a specific road_id using the same reconciliation logic,
    combining red intervals from wait.csv and green intervals from cross_path.
    If restrict_A_group is True, only use A-group greens (direction starts with 'A').
    Returns seconds (float) or None if not computable.
    """
    if not os.path.exists(cross_path) or not os.path.exists(WAIT_FILE):
        return None
    try:
        wait_df = load_wait(WAIT_FILE)
        wait_df = wait_df[wait_df.get("road_id", "NA") == road_id].copy()
        cross_df = load_cross(cross_path)
        cross_df = cross_df[cross_df.get("road_id", "NA") == road_id].copy()
        if restrict_A_group:
            try:
                up = cross_df["direction"].astype(str).str.upper()
                cross_df = cross_df[up.str.startswith("A")]
            except Exception:
                pass
        if wait_df.empty or cross_df.empty:
            return None
        wait_df["group_key"] = wait_df.apply(make_group_key, axis=1)
        cross_df["group_key"] = cross_df.apply(make_group_key, axis=1)
        keys = sorted(set(wait_df["group_key"]).union(set(cross_df["group_key"])))
        if not keys:
            return None
        rows = []
        for key in keys:
            w = wait_df[wait_df["group_key"] == key]
            c = cross_df[cross_df["group_key"] == key]
            if w.empty and c.empty:
                continue
            rows.append(process_group(w, c, key))
        if not rows:
            return None
        df = pd.DataFrame(rows)
        vals = pd.to_numeric(df["reconciled_period_s"], errors="coerce").to_numpy()
        wts = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).to_numpy()
        vals = vals[np.isfinite(vals)]
        wts = wts[:len(vals)] if len(wts) != len(vals) else wts
        period = weighted_median(vals, wts) if vals.size > 0 else float('nan')
        return float(period) if np.isfinite(period) else None
    except Exception:
        return None

def main():
    # Load wait (required)
    try:
        wait_df = load_wait(WAIT_FILE)
    except Exception:
        print("NaN")
        return
    # Load cross with fallback
    cross_df = None
    try:
        if os.path.exists(CROSS_FILE_A):
            cross_df = load_cross(CROSS_FILE_A)
        else:
            print("NaN")
            return
    except Exception:
        print("NaN")
        return

    wait_df["group_key"] = wait_df.apply(make_group_key, axis=1)
    cross_df["group_key"] = cross_df.apply(make_group_key, axis=1)

    keys = sorted(set(wait_df["group_key"]).union(set(cross_df["group_key"])))

    rows = []
    for key in keys:
        w = wait_df[wait_df["group_key"] == key]
        c = cross_df[cross_df["group_key"] == key]
        rows.append(process_group(w, c, key))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PER_GROUP_CSV, index=False, float_format="%.3f")
    with open(OUT_PER_GROUP_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Global period: weighted median over reconciled per-group period
    vals = pd.to_numeric(df["reconciled_period_s"], errors="coerce").to_numpy()
    wts = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).to_numpy()
    vals = vals[np.isfinite(vals)]; wts = wts[:len(vals)] if len(wts)!=len(vals) else wts

    global_period = weighted_median(vals, wts) if vals.size>0 else float('nan')
    out = {
        "global_period_s": global_period,
        "aggregation": "weighted_median over reconciled per-group periods",
        "weights": "min(pairs_green, pairs_red) else max(n_g, n_r)",
        "groups_used": int(len(vals))
    }
    with open(OUT_GLOBAL_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # PRINT ONLY integer seconds to stdout (round to nearest)
    if np.isfinite(global_period):
        print(int(round(global_period)))
    else:
        print("NaN")
    # Also compute and print REAL A1 period (second line)
    try:
        real_period = compute_real_period()
        if real_period is None or not np.isfinite(real_period):
            print("NaN")
        else:
            print(int(round(real_period)))
    except Exception:
        print("NaN")

    # Third line: compute period using cross_over_time_with_outer.csv (A-group filtered)
    try:
        if os.path.exists(CROSS_FILE_WITH_OUTER):
            cross_outer_df = load_cross(CROSS_FILE_WITH_OUTER)
            # Filter to A-group if direction present (A1/A2)
            try:
                up = cross_outer_df["direction"].astype(str).str.upper()
                cross_outer_df = cross_outer_df[up.str.startswith("A")]
            except Exception:
                pass
            cross_outer_df["group_key"] = cross_outer_df.apply(make_group_key, axis=1)
            keys2 = sorted(set(wait_df["group_key"]).union(set(cross_outer_df["group_key"])))
            rows2 = []
            for key in keys2:
                w = wait_df[wait_df["group_key"] == key]
                c = cross_outer_df[cross_outer_df["group_key"] == key]
                rows2.append(process_group(w, c, key))
            df2 = pd.DataFrame(rows2)
            vals2 = pd.to_numeric(df2["reconciled_period_s"], errors="coerce").to_numpy()
            wts2 = pd.to_numeric(df2["weight"], errors="coerce").fillna(1.0).to_numpy()
            vals2 = vals2[np.isfinite(vals2)]; wts2 = wts2[:len(vals2)] if len(wts2)!=len(vals2) else wts2
            gp2 = weighted_median(vals2, wts2) if vals2.size>0 else float('nan')
            if np.isfinite(gp2):
                print(int(round(gp2)))
            else:
                print("NaN")
        else:
            print("NaN")
    except Exception:
        print("NaN")

    # Fourth line: compute period using cross_over_time_try.csv (A-group filtered)
    try:
        if os.path.exists(CROSS_FILE_TRY):
            cross_try_df = load_cross(CROSS_FILE_TRY)
            try:
                up = cross_try_df["direction"].astype(str).str.upper()
                cross_try_df = cross_try_df[up.str.startswith("A")]
            except Exception:
                pass
            cross_try_df["group_key"] = cross_try_df.apply(make_group_key, axis=1)
            keys3 = sorted(set(wait_df["group_key"]).union(set(cross_try_df["group_key"])))
            rows3 = []
            for key in keys3:
                w = wait_df[wait_df["group_key"] == key]
                c = cross_try_df[cross_try_df["group_key"] == key]
                rows3.append(process_group(w, c, key))
            df3 = pd.DataFrame(rows3)
            vals3 = pd.to_numeric(df3["reconciled_period_s"], errors="coerce").to_numpy()
            wts3 = pd.to_numeric(df3["weight"], errors="coerce").fillna(1.0).to_numpy()
            vals3 = vals3[np.isfinite(vals3)]; wts3 = wts3[:len(vals3)] if len(wts3)!=len(vals3) else wts3
            gp3 = weighted_median(vals3, wts3) if vals3.size>0 else float('nan')
            if np.isfinite(gp3):
                print(int(round(gp3)))
            else:
                print("NaN")
        else:
            print("NaN")
    except Exception:
        print("NaN")

    # Fifth line: A0008 period using cross_over_time.csv (A-group filtered)
    try:
        p_a0008 = compute_period_for_road_from_cross(CROSS_FILE_ALL, "A0008", restrict_A_group=True)
        if p_a0008 is None or not np.isfinite(p_a0008):
            print("NaN")
        else:
            print(int(round(p_a0008)))
    except Exception:
        print("NaN")

if __name__ == "__main__":
    main()
