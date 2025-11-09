#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fold per-second flow/wait series into a single signal cycle by summing
corresponding seconds across cycles, and generate four per-cycle plots.

Fixed settings:
  - Input files:
      - data/per_second_flow.csv
      - data/per_second_wait.csv
  - Output files:
      - data/per_cycle_flow.csv
      - data/per_cycle_wait.csv
  - Road filter: A0003 only
  - Cycle start time (HH:MM:SS): 16:34:24
  - Cycle end time (HH:MM:SS):   17:28:33
  - Cycle length: 130 seconds
  - Number of cycles in window: 25

For each second offset k in [0, 129], this script sums all rows whose
time-of-day equals (start_time + k seconds + n * 130 seconds) for n=0..24,
aggregated across all dates for road A0003.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DIR4: Tuple[str, str, str, str] = ("A1", "A2", "B1", "B2")

# Fixed configuration
FLOW_CSV = os.path.join("data", "per_second_flow.csv")
WAIT_CSV = os.path.join("data", "per_second_wait.csv")
OUT_FLOW = os.path.join("data", "per_cycle_flow.csv")
OUT_WAIT = os.path.join("data", "per_cycle_wait.csv")
PLOT_DIR = "plots"
ROAD_ID = "A0003"
START_TIME_STR = "16:34:24"
END_TIME_STR = "17:28:33"
CYCLE_SECONDS = 130
NUM_CYCLES = 25


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    needed = {"road_id", "date", "time", *DIR4}
    if not needed.issubset(set(df.columns)):
        missing = sorted(list(needed - set(df.columns)))
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def _fold_to_cycle(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["offset_s", "time", *DIR4])
    # Keep A0003 only
    work = df.loc[df["road_id"] == ROAD_ID].copy()
    if work.empty:
        return pd.DataFrame(columns=["offset_s", "time", *DIR4])
    # Parse time-of-day to a dummy datetime for range operations
    work["dt"] = pd.to_datetime("1970-01-01 " + work["time"].astype(str), errors="coerce")
    start_dt = pd.to_datetime("1970-01-01 " + START_TIME_STR)
    end_dt = pd.to_datetime("1970-01-01 " + END_TIME_STR)
    # Filter to window (inclusive)
    work = work[(work["dt"] >= start_dt) & (work["dt"] <= end_dt)].copy()
    if work.empty:
        return pd.DataFrame(columns=["offset_s", "time", *DIR4])
    # Sum across all dates for each second-of-day
    summed = work.groupby("dt", as_index=False)[list(DIR4)].sum()
    summed = summed.set_index("dt")
    # Ensure a complete 1-second index across the window; fill gaps with 0
    full_idx = pd.date_range(start=start_dt, end=end_dt, freq="s")
    summed = summed.reindex(full_idx).fillna(0.0)
    # Compute cycle offset for each second
    offsets = ((summed.index - start_dt) / pd.Timedelta(seconds=1)).astype("int64") % CYCLE_SECONDS
    summed = summed.copy()
    summed["offset_s"] = offsets
    # Fold: sum by offset
    folded = summed.groupby("offset_s", as_index=False)[list(DIR4)].sum()
    # Add canonical time label for each offset (as the first-cycle time)
    folded["time"] = (start_dt + pd.to_timedelta(folded["offset_s"], unit="s")).dt.strftime("%H:%M:%S")
    # Order columns and sort by offset
    folded = folded.sort_values("offset_s")[["offset_s", "time", *DIR4]].reset_index(drop=True)
    # Cast to integers where appropriate
    for d in DIR4:
        # values can be floats due to fillna; they are counts so cast safely
        folded[d] = folded[d].round().astype("int64")
    return folded


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_cycle_direction(direction: str,
                          flow_cycle: pd.DataFrame,
                          wait_cycle: pd.DataFrame,
                          out_path: str) -> None:
    # Align by offset_s and plot
    f = flow_cycle[["offset_s", direction]].rename(columns={direction: "flow"}).copy()
    w = wait_cycle[["offset_s", direction]].rename(columns={direction: "wait"}).copy()
    f["flow"] = pd.to_numeric(f["flow"], errors="coerce").fillna(0.0)
    w["wait"] = pd.to_numeric(w["wait"], errors="coerce").fillna(0.0)
    # Z-score normalization per type for THIS direction
    flow_mean = float(f["flow"].mean())
    flow_std = float(f["flow"].std(ddof=0))
    wait_mean = float(w["wait"].mean())
    wait_std = float(w["wait"].std(ddof=0))
    f["flow_z"] = (f["flow"] - flow_mean) / flow_std if flow_std > 0 else 0.0
    w["wait_z"] = (w["wait"] - wait_mean) / wait_std if wait_std > 0 else 0.0
    # Merge for plotting
    merged = pd.merge(f[["offset_s", "flow_z"]], w[["offset_s", "wait_z"]], on="offset_s", how="outer").fillna(0.0)
    merged = merged.sort_values("offset_s")
    plt.figure(figsize=(12, 5))
    plt.plot(merged["offset_s"], merged["flow_z"], label="Flow (z)", linewidth=1.6)
    plt.plot(merged["offset_s"], merged["wait_z"], label="Wait (z)", linewidth=1.6)
    plt.title(f"{direction} (cycle length {CYCLE_SECONDS}s, N={NUM_CYCLES} cycles)")
    plt.xlabel("Second in cycle (s)")
    plt.ylabel("Z-score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    flow_df = _read_csv(FLOW_CSV)
    wait_df = _read_csv(WAIT_CSV)
    flow_cycle = _fold_to_cycle(flow_df)
    wait_cycle = _fold_to_cycle(wait_df)
    flow_cycle.to_csv(OUT_FLOW, index=False)
    wait_cycle.to_csv(OUT_WAIT, index=False)
    print(f"Wrote {len(flow_cycle)} rows to {OUT_FLOW}")
    print(f"Wrote {len(wait_cycle)} rows to {OUT_WAIT}")
    # Plots
    _ensure_output_dir(PLOT_DIR)
    out_paths: List[str] = []
    for d in DIR4:
        out = os.path.join(PLOT_DIR, f"per_cycle_{d}.png")
        _plot_cycle_direction(d, flow_cycle, wait_cycle, out)
        out_paths.append(out)
    print(f"Wrote {len(out_paths)} figures:")
    for p in out_paths:
        print(f" - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


