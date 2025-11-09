#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-second flow from data/cross_over_time_try.csv (A0003 only) and
then fold into a single signal cycle by summing corresponding seconds.

Outputs:
  - data/per_second_flow.csv
  - data/per_cycle_flow.csv
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd


DIR4: Tuple[str, str, str, str] = ("A1", "A2", "B1", "B2")

# Fixed configuration
TRY_CSV = os.path.join("data", "cross_over_time_try.csv")
OUT_PER_SECOND_FLOW = os.path.join("data", "per_second_flow.csv")
OUT_PER_CYCLE_FLOW = os.path.join("data", "per_cycle_flow.csv")
ROAD_ID = "A0003"
START_TIME_STR = "16:34:24"
END_TIME_STR = "17:28:33"
CYCLE_SECONDS = 130


def _to_dir4(direction: object) -> Optional[str]:
    try:
        d = str(direction)
    except Exception:
        return None
    for k in DIR4:
        if d.startswith(k):
            return k
    return None


def _read_try_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    need = {"road_id", "date", "direction", "enter_time", "exit_time"}
    if not need.issubset(set(df.columns)):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def build_per_second_flow_from_try(trdf: pd.DataFrame) -> pd.DataFrame:
    """
    Per-second occupancy counts for A1/A2/B1/B2.
    Interval is [enter_time, exit_time), i.e., end exclusive.
    Returns columns: road_id, date, time, A1, A2, B1, B2
    """
    if trdf is None or trdf.empty:
        return pd.DataFrame(columns=["road_id", "date", "time", *DIR4])
    work = trdf.copy()
    # Filter to one road
    work = work[work["road_id"] == ROAD_ID].copy()
    if work.empty:
        return pd.DataFrame(columns=["road_id", "date", "time", *DIR4])
    # Map to 4-direction groups
    work["dir4"] = work["direction"].map(_to_dir4)
    work = work.dropna(subset=["dir4"]).astype({"dir4": str})
    # Build absolute timestamps
    work["start"] = pd.to_datetime(work["date"].astype(str) + " " + work["enter_time"].astype(str), errors="coerce")
    work["end"] = pd.to_datetime(work["date"].astype(str) + " " + work["exit_time"].astype(str), errors="coerce")
    # Filter valid, non-zero intervals
    work = work[(~work["start"].isna()) & (~work["end"].isna()) & (work["start"] < work["end"])].copy()
    if work.empty:
        return pd.DataFrame(columns=["road_id", "date", "time", *DIR4])

    out_frames: List[pd.DataFrame] = []
    for (road_id, date_val), g in work.groupby(["road_id", "date"]):
        # Collect event deltas per direction
        dir_events: Dict[str, Dict[pd.Timestamp, int]] = {d: {} for d in DIR4}
        for _, r in g.iterrows():
            d = r["dir4"]
            s = r["start"]
            e = r["end"]
            dir_events[d][s] = dir_events[d].get(s, 0) + 1
            dir_events[d][e] = dir_events[d].get(e, 0) - 1
        # Determine group time span
        all_keys = []
        for d in DIR4:
            all_keys.extend(list(dir_events[d].keys()))
        if not all_keys:
            continue
        t0 = min(all_keys).floor("s")
        t1 = max(all_keys).ceil("s")
        if t0 >= t1:
            continue
        # Build 1-second index; end exclusive
        idx = pd.date_range(t0, t1 - pd.Timedelta(seconds=1), freq="s")
        data = {}
        for d in DIR4:
            ev = dir_events[d]
            if not ev:
                series = pd.Series(0, index=idx, dtype="int64")
            else:
                evs = pd.Series(ev).sort_index()
                series = evs.cumsum()
                # Align to idx: forward fill current occupancy and fill gaps with 0
                series = series.reindex(idx, method="ffill").fillna(0).astype("int64")
            data[d] = series
        df_group = pd.DataFrame(data, index=idx).reset_index().rename(columns={"index": "dt"})
        df_group["road_id"] = road_id
        df_group["date"] = date_val
        df_group["time"] = df_group["dt"].dt.strftime("%H:%M:%S")
        out_frames.append(df_group[["road_id", "date", "time", *DIR4]])
    if not out_frames:
        return pd.DataFrame(columns=["road_id", "date", "time", *DIR4])
    out = pd.concat(out_frames, ignore_index=True)
    # Ensure integer dtype
    for d in DIR4:
        out[d] = out[d].astype("int64")
    return out.sort_values(["road_id", "date", "time"]).reset_index(drop=True)


def _fold_per_second_to_cycle(per_second_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fold A0003 per-second flow into one 130s cycle across all dates
    within [START_TIME_STR, END_TIME_STR].
    """
    if per_second_df is None or per_second_df.empty:
        return pd.DataFrame(columns=["offset_s", "time", *DIR4])
    work = per_second_df[per_second_df["road_id"] == ROAD_ID].copy()
    if work.empty:
        return pd.DataFrame(columns=["offset_s", "time", *DIR4])
    # Time-of-day as dummy dt for range operations
    work["dt"] = pd.to_datetime("1970-01-01 " + work["time"].astype(str), errors="coerce")
    start_dt = pd.to_datetime("1970-01-01 " + START_TIME_STR)
    end_dt = pd.to_datetime("1970-01-01 " + END_TIME_STR)
    # Keep only window
    work = work[(work["dt"] >= start_dt) & (work["dt"] <= end_dt)].copy()
    if work.empty:
        return pd.DataFrame(columns=["offset_s", "time", *DIR4])
    # Sum across all dates for each dt second
    summed = work.groupby("dt", as_index=False)[list(DIR4)].sum()
    summed = summed.set_index("dt")
    # Ensure complete 1s coverage
    full_idx = pd.date_range(start=start_dt, end=end_dt, freq="s")
    summed = summed.reindex(full_idx).fillna(0.0)
    # Compute cycle offset
    offsets = ((summed.index - start_dt) / pd.Timedelta(seconds=1)).astype("int64") % CYCLE_SECONDS
    summed = summed.copy()
    summed["offset_s"] = offsets
    folded = summed.groupby("offset_s", as_index=False)[list(DIR4)].sum()
    folded["time"] = (start_dt + pd.to_timedelta(folded["offset_s"], unit="s")).dt.strftime("%H:%M:%S")
    folded = folded.sort_values("offset_s")[["offset_s", "time", *DIR4]].reset_index(drop=True)
    for d in DIR4:
        folded[d] = folded[d].round().astype("int64")
    return folded


def main() -> int:
    trdf = _read_try_csv(TRY_CSV)
    per_second_flow = build_per_second_flow_from_try(trdf)
    if per_second_flow is None or per_second_flow.empty:
        # Still write an empty CSV with expected columns
        pd.DataFrame(columns=["road_id", "date", "time", *DIR4]).to_csv(OUT_PER_SECOND_FLOW, index=False)
        print(f"No rows; wrote empty {OUT_PER_SECOND_FLOW}")
        # And an empty per-cycle CSV
        pd.DataFrame(columns=["offset_s", "time", *DIR4]).to_csv(OUT_PER_CYCLE_FLOW, index=False)
        print(f"No rows; wrote empty {OUT_PER_CYCLE_FLOW}")
        return 0
    per_second_flow.to_csv(OUT_PER_SECOND_FLOW, index=False)
    print(f"Wrote {len(per_second_flow)} per-second rows to {OUT_PER_SECOND_FLOW}")
    per_cycle_flow = _fold_per_second_to_cycle(per_second_flow)
    per_cycle_flow.to_csv(OUT_PER_CYCLE_FLOW, index=False)
    print(f"Wrote {len(per_cycle_flow)} per-cycle rows to {OUT_PER_CYCLE_FLOW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


