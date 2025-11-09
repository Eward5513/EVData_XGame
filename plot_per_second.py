#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate line charts from per_second_flow.csv and per_second_wait.csv (固定配置).

Creates four plots (A1, A2, B1, B2). Each plot overlays two lines:
  - Flow: counts from per_second_flow.csv
  - Wait: counts from per_second_wait.csv
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pandas as pd

# Use non-interactive backend for headless environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402


DIR4: Tuple[str, str, str, str] = ("A1", "A2", "B1", "B2")

# 固定配置
FLOW_CSV = os.path.join("data", "per_second_flow.csv")
WAIT_CSV = os.path.join("data", "per_second_wait.csv")
OUTPUT_DIR = "plots"
ROAD_ID = "A0003"


def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    expected_cols = {"road_id", "date", "time", *DIR4}
    missing = [c for c in ["road_id", "date", "time"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    # Ensure direction columns exist; if missing, create zeros
    for d in DIR4:
        if d not in df.columns:
            df[d] = 0
    return df


def _aggregate_by_time(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", *DIR4])
    # Sum over all (road_id, date) for each time
    agg = df.groupby("time", as_index=False)[list(DIR4)].sum()
    return agg


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Parse HH:MM:SS into datetime for plotting
    # Use a fixed dummy date to leverage matplotlib date formatters
    dt = pd.to_datetime("1970-01-01 " + df["time"].astype(str), errors="coerce")
    df = df.copy()
    df["dt"] = dt
    df = df.dropna(subset=["dt"]).sort_values("dt")
    return df


def _plot_one_direction(direction: str,
                        df_flow: pd.DataFrame,
                        df_wait: pd.DataFrame,
                        title: str,
                        out_path: str) -> None:
    # Align on time
    f = df_flow[["time", direction]].rename(columns={direction: "flow"})
    w = df_wait[["time", direction]].rename(columns={direction: "wait"})
    merged = pd.merge(f, w, on="time", how="outer").fillna(0.0)
    merged = _to_time_index(merged)
    if merged is None or merged.empty:
        # Still produce an empty plot to signal absence
        plt.figure(figsize=(10, 4.5))
        plt.title(title + " (no data)")
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return
    # Convert to numeric
    for c in ["flow", "wait"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
    plt.figure(figsize=(12, 5))
    plt.plot(merged["dt"], merged["flow"], label="Flow", linewidth=1.5)
    plt.plot(merged["dt"], merged["wait"], label="Wait", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    ax = plt.gca()
    # Format x-axis as HH:MM
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_plots() -> List[str]:
    flow = _read_csv_safe(FLOW_CSV)
    wait = _read_csv_safe(WAIT_CSV)
    _ensure_output_dir(OUTPUT_DIR)
    outputs: List[str] = []
    # Only keep A0003, aggregate across dates by time-of-day
    flow_a3 = flow.loc[flow["road_id"] == ROAD_ID].copy()
    wait_a3 = wait.loc[wait["road_id"] == ROAD_ID].copy()
    f = _aggregate_by_time(flow_a3)
    w = _aggregate_by_time(wait_a3)
    for dir4 in DIR4:
        title = f"{dir4} (road {ROAD_ID}, all dates combined)"
        out = os.path.join(OUTPUT_DIR, f"per_second_{ROAD_ID}_{dir4}.png")
        _plot_one_direction(dir4, f, w, title, out)
        outputs.append(out)
    return outputs


def main() -> int:
    outputs = generate_plots()
    print(f"Wrote {len(outputs)} figures:")
    for p in outputs:
        print(f" - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


