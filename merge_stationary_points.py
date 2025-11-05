#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge consecutive low-speed (stationary) trajectory points into representative points
and add an end_time column to indicate the last timestamp of merged clusters.

Inputs:
  - /home/tzhang174/EVData_XGame/A0003_split.csv
  - /home/tzhang174/EVData_XGame/A0008_split.csv

Outputs:
  - /home/tzhang174/EVData_XGame/A0003_merged.csv
  - /home/tzhang174/EVData_XGame/A0008_merged.csv

Rules:
  - Only points with speed < 3 (km/h) are eligible for merging.
  - Merge consecutive points when each adjacent pair is within a distance threshold.
  - Threshold depends on sampling interval (median Δcollectiontime in seconds):
      * If ~1s (0.8s–1.2s), threshold = 3 m
      * Else threshold = 5 m
  - Any point with speed ≥ 3 is never merged.
  - For merged runs (length ≥ 2):
      * Output one row located at the average latitude/longitude of the run
      * time_stamp = first point's time_stamp; end_time = last point's time_stamp
      * collectiontime = first point's collectiontime
      * Other fields: keep from the first row, except numeric speed/acceleratorpedal averaged
  - For non-merged points: copy as-is with end_time empty
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from trajectory_utils import pair_distance_m
from config import BASE_DIR


INPUT_FILES = [
    os.path.join(BASE_DIR, "data", "A0003_split.csv"),
    os.path.join(BASE_DIR, "data", "A0008_split.csv"),
]


def determine_distance_threshold_secs(times_ms: np.ndarray) -> float:
    """
    Determine the spatial threshold (meters) based on the median sampling interval (seconds).
    If the median Δt is ~1s (0.8..1.2), return 3m; otherwise return 5m.
    """
    if times_ms is None or len(times_ms) < 2:
        return 5.0
    diffs = np.diff(times_ms.astype(np.float64)) / 1000.0  # seconds
    # Exclude non-positive diffs to avoid duplicated timestamps
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 5.0
    med = float(np.median(diffs))
    if 0.8 <= med <= 1.2:
        return 3.0
    return 5.0


def merge_group(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    Merge consecutive stationary points within a single (road_id, vehicle_id, date, seg_id) group.
    Returns a new dataframe with an added 'end_time' column.
    """
    if df_group.empty:
        return df_group

    # Ensure proper ordering
    g = df_group.sort_values('collectiontime').reset_index(drop=True)

    # Determine distance threshold based on sampling interval
    times = g['collectiontime'].to_numpy(dtype=np.int64)
    dist_thresh_m = determine_distance_threshold_secs(times)

    # Pull needed arrays
    lats = g['latitude'].to_numpy(dtype=float)
    lons = g['longitude'].to_numpy(dtype=float)
    # Speed in km/h (as-is); NaN treated as large to disable merge
    sp = pd.to_numeric(g['speed'], errors='coerce').to_numpy()
    sp = np.where(np.isnan(sp), np.inf, sp)

    out_rows: List[pd.Series] = []
    n = len(g)
    i = 0
    while i < n:
        # Attempt to start a stationary run at i
        if sp[i] < 3.0:
            j = i + 1
            while j < n:
                if not (sp[j] < 3.0):
                    break
                # Check distance constraint between consecutive points
                d_m = pair_distance_m(lats[j - 1], lons[j - 1], lats[j], lons[j])
                if d_m > dist_thresh_m:
                    break
                j += 1

            run_len = j - i
            if run_len >= 2:
                sub = g.iloc[i:j].copy()
                # Build output row primarily from the first row
                row = sub.iloc[0].copy()
                # Average coordinates for the merged representative point
                row['latitude'] = float(sub['latitude'].mean())
                row['longitude'] = float(sub['longitude'].mean())
                # Keep collectiontime/time_stamp from the first; set end_time to last time_stamp
                if 'time_stamp' in sub.columns:
                    row['end_time'] = str(sub.iloc[-1]['time_stamp'])
                else:
                    row['end_time'] = ''
                # Average numeric metrics where appropriate
                if 'speed' in sub.columns:
                    row['speed'] = float(pd.to_numeric(sub['speed'], errors='coerce').mean())
                if 'acceleratorpedal' in sub.columns:
                    row['acceleratorpedal'] = float(pd.to_numeric(sub['acceleratorpedal'], errors='coerce').mean())
                # Keep brakestatus/gearnum/havebrake/havedriver from the first (if present)
                out_rows.append(row)
                i = j
                continue
        # Not merged: copy as-is with empty end_time
        row = g.iloc[i].copy()
        row['end_time'] = ''
        out_rows.append(row)
        i += 1

    out = pd.DataFrame(out_rows)
    return out


def process_file(input_path: str, output_path: str) -> None:
    print(f"Reading {input_path} ...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"Input is empty; writing empty output: {output_path}")
        df_out = df.copy()
        if 'end_time' not in df_out.columns:
            df_out['end_time'] = ''
        # Ensure output directory exists even in empty-input path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_out.to_csv(output_path, index=False)
        return

    required_cols = ['vehicle_id', 'date', 'seg_id', 'collectiontime', 'latitude', 'longitude', 'speed']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    # Process per (road_id, vehicle_id, date, seg_id) if road_id exists; else per (vehicle_id, date, seg_id)
    group_cols = ['vehicle_id', 'date', 'seg_id']
    if 'road_id' in df.columns:
        group_cols = ['road_id'] + group_cols

    out_parts: List[pd.DataFrame] = []
    for _, sub in df.groupby(group_cols, sort=False):
        out_parts.append(merge_group(sub))
    merged = pd.concat(out_parts, ignore_index=True)

    # Reorder columns: keep original order and append 'end_time' at the end
    cols = list(df.columns)
    if 'end_time' not in cols:
        cols.append('end_time')
    # Retain only columns present in merged (in case some optional columns are absent)
    final_cols = [c for c in cols if c in merged.columns]
    merged = merged[final_cols]

    print(f"Writing {output_path} ...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Wrote {len(merged)} rows to {output_path}")
def main():
    for in_path in INPUT_FILES:
        base = os.path.basename(in_path)
        name_no_ext = base.rsplit('.', 1)[0]
        out_path = os.path.join(BASE_DIR, "data", f"{name_no_ext.replace('_split', '')}_merged.csv")
        process_file(in_path, out_path)

# Module is imported by main.py; no standalone execution


