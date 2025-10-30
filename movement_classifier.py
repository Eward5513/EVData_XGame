
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
movement_classifier.py
----------------------
Classify each vehicle trajectory at one intersection into movement types:
  - straight / left / right / uturn
and report entry/exit directions (8-compass sectors).

Input CSV must contain columns (case-sensitive):
  vehicle_id, seg_id, collectiontime, date, time_stamp, road_id,
  longitude, latitude, speed, acceleratorpedal, brakestatus, gearnum,
  havebrake, havedriver

Usage:
  python movement_classifier.py
All configuration values (input/output paths, intersection parameters,
and thresholds) are defined directly in this file.
"""

import math
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# ------------------ configuration ------------------

OUTPUT_CSV_PATH = '/home/tzhang174/EVData_XGame/movement.csv'

INTERSECTIONS = [
    {
        'road_id': 'A0003',
        'csv_path': '/home/tzhang174/EVData_XGame/A0003_split.csv',
        'projection_lon0': 123.152539,
        'projection_lat0': 32.345137,
        'center_lon': 123.152539,
        'center_lat': 32.345137,
        'radius_m': 35.0,
        'pre_seconds': 1.5,
        'post_seconds': 1.5,
        'smooth_window': 5,
        'straight_thresh': 25.0,
        'turn_lo': 45.0,
        'turn_hi': 135.0,
        'uturn_thresh': 150.0,
    },
    {
        'road_id': 'A0008',
        'csv_path': '/home/tzhang174/EVData_XGame/A0008_split.csv',
        'projection_lon0': 123.181261,
        'projection_lat0': 32.327137,
        'center_lon': 123.181261,
        'center_lat': 32.327137,
        'radius_m': 35.0,
        'pre_seconds': 1.5,
        'post_seconds': 1.5,
        'smooth_window': 5,
        'straight_thresh': 25.0,
        'turn_lo': 45.0,
        'turn_hi': 135.0,
        'uturn_thresh': 150.0,
    },
]

# ------------------ math helpers ------------------

R_EARTH = 6378137.0

def lonlat_to_xy(lon, lat, lon0, lat0):
    lat0_rad = np.deg2rad(lat0)
    x = np.deg2rad(lon - lon0) * R_EARTH * np.cos(lat0_rad)
    y = np.deg2rad(lat - lat0) * R_EARTH
    return x, y

def wrap_deg(a):
    """Wrap angle to (-180, 180] degrees."""
    a = (a + 180.0) % 360.0 - 180.0
    return a

def heading_from_dxdy(dx, dy):
    """Return heading in degrees: 0=East, 90=North (atan2)."""
    return (np.rad2deg(np.arctan2(dy, dx)) + 360.0) % 360.0

def moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.copy()
    k = int(k)
    k = max(1, k if k % 2 == 1 else k + 1)  # enforce odd
    pad = k // 2
    if len(arr) <= 1:
        return arr.copy()
    # edge-pad by reflecting
    left = arr[1:pad+1][::-1] if pad < len(arr) else arr[::-1]
    right = arr[-pad-1:-1][::-1] if pad < len(arr) else arr[::-1]
    arr_pad = np.concatenate([left, arr, right])
    kernel = np.ones(k) / k
    sm = np.convolve(arr_pad, kernel, mode='valid')
    return sm[:len(arr)]

def circular_mean_deg(angles_deg: np.ndarray) -> float:
    if len(angles_deg) == 0:
        return float('nan')
    ang = np.deg2rad(angles_deg % 360.0)
    c = np.mean(np.cos(ang))
    s = np.mean(np.sin(ang))
    if c == 0 and s == 0:
        return float('nan')
    return (np.rad2deg(math.atan2(s, c)) + 360.0) % 360.0

def sector8(angle_deg: float) -> str:
    names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    idx = int(((angle_deg + 22.5) % 360) // 45)
    return names[idx]

# ------------------ core logic ------------------

def get_time_ms(df: pd.DataFrame) -> pd.Series:
    if 'collectiontime' in df.columns and df['collectiontime'].notna().any():
        t = pd.to_numeric(df['collectiontime'], errors='coerce')
        return t
    if 'date' in df.columns and 'time_stamp' in df.columns:
        dt = pd.to_datetime(df['date'].astype(str) + ' ' + df['time_stamp'].astype(str), errors='coerce', utc=True)
        return dt.view('int64') // 10**6
    raise ValueError("No usable time column: expected 'collectiontime' (ms) or 'date'+'time_stamp'.")

def classify_turn(turn_deg: float,
                  straight_thresh: float = 25.0,
                  turn_lo: float = 45.0,
                  turn_hi: float = 135.0,
                  uturn_thresh: float = 150.0) -> str:
    a = wrap_deg(turn_deg)
    aa = abs(a)
    if aa < straight_thresh:
        return 'straight'
    if aa >= uturn_thresh:
        return 'uturn'
    if a > 0:
        if turn_lo <= a <= turn_hi:
            return 'left'
        return 'left-like'
    else:
        if -turn_hi <= a <= -turn_lo:
            return 'right'
        return 'right-like'

def process_vehicle(g: pd.DataFrame,
                    center_xy: Tuple[float, float],
                    radius_m: float,
                    pre_s: float,
                    post_s: float,
                    smooth_k: int,
                    straight_thresh: float,
                    turn_lo: float,
                    turn_hi: float,
                    uturn_thresh: float) -> Optional[dict]:
    g = g.sort_values('_t_ms')
    if len(g) < 2:
        return None
    x = g['_x'].values
    y = g['_y'].values
    # smooth positions for heading stability
    xs = moving_average(x, smooth_k)
    ys = moving_average(y, smooth_k)

    # velocity-based heading (finite differences on smoothed pos)
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    heading = heading_from_dxdy(dx, dy)

    t = g['_t_ms'].values.astype('float64')  # ms
    # distance to center
    cx, cy = center_xy
    r = np.hypot(xs - cx, ys - cy)

    outside = r > radius_m
    inside  = ~outside

    enter_idx = None
    exit_idx = None
    for i in range(1, len(r)):
        if outside[i-1] and inside[i]:
            enter_idx = i
            break
    if enter_idx is None:
        return None
    for j in range(enter_idx+1, len(r)):
        if inside[j-1] and outside[j]:
            exit_idx = j
            break
    if exit_idx is None:
        return None

    te = t[enter_idx]
    to = t[exit_idx]
    in_mask = (t >= te - pre_s*1000.0) & (t < te)
    out_mask = (t > to) & (t <= to + post_s*1000.0)

    dir_in = circular_mean_deg(heading[in_mask])
    dir_out = circular_mean_deg(heading[out_mask])

    h_inside = heading[enter_idx:exit_idx+1]
    if len(h_inside) < 2:
        return None
    diffs = np.diff(h_inside)
    diffs = np.array([wrap_deg(d) for d in diffs])
    turn_total = float(np.sum(diffs))

    movement = classify_turn(
        turn_total,
        straight_thresh=straight_thresh,
        turn_lo=turn_lo,
        turn_hi=turn_hi,
        uturn_thresh=uturn_thresh
    )

    base_row = g.iloc[0]

    date_value = None
    if 'date' in g.columns:
        valid_dates = g['date'].dropna()
        if not valid_dates.empty:
            date_value = str(valid_dates.iloc[0])

    seg_value = None
    if 'seg_id' in g.columns:
        valid_seg = g['seg_id'].dropna()
        if not valid_seg.empty:
            try:
                seg_value = int(valid_seg.iloc[0])
            except Exception:
                try:
                    seg_value = int(float(valid_seg.iloc[0]))
                except Exception:
                    seg_value = None

    return {
        'vehicle_id': int(base_row['vehicle_id']),
        'date': date_value,
        'seg_id': seg_value if seg_value is not None else 0,
        't_enter_ms': int(te),
        't_exit_ms': int(to),
        'dir_in_deg': None if np.isnan(dir_in) else float(dir_in),
        'dir_in_sector': None if np.isnan(dir_in) else sector8(dir_in),
        'dir_out_deg': None if np.isnan(dir_out) else float(dir_out),
        'dir_out_sector': None if np.isnan(dir_out) else sector8(dir_out),
        'turn_deg': float(turn_total),
        'movement': movement
    }

def main():
    all_results = []
    intersection_counts = []

    for cfg in INTERSECTIONS:
        road_id = cfg.get('road_id', 'UNKNOWN')
        csv_path = cfg.get('csv_path')

        if not csv_path:
            print(f"WARNING: Missing csv_path for intersection {road_id}; skipping.", file=sys.stderr)
            continue

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"WARNING: CSV file not found for {road_id}: {csv_path}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"ERROR: Failed to load {csv_path} for {road_id}: {exc}", file=sys.stderr)
            continue

        df['_t_ms'] = get_time_ms(df)
        if df['_t_ms'].isna().all():
            print(f"ERROR: Could not parse time column for {road_id}.", file=sys.stderr)
            sys.exit(2)

        lon0 = cfg.get('projection_lon0')
        lat0 = cfg.get('projection_lat0')
        if lon0 is None:
            lon0 = float(np.median(df['longitude']))
        if lat0 is None:
            lat0 = float(np.median(df['latitude']))
        lon0 = float(lon0)
        lat0 = float(lat0)

        x, y = lonlat_to_xy(df['longitude'].values, df['latitude'].values, lon0, lat0)
        df['_x'] = x
        df['_y'] = y

        center_lon = cfg.get('center_lon')
        center_lat = cfg.get('center_lat')
        if center_lon is not None and center_lat is not None:
            cx, cy = lonlat_to_xy(center_lon, center_lat, lon0, lat0)
        else:
            cx, cy = float(np.median(x)), float(np.median(y))

        df = df.sort_values(['vehicle_id', '_t_ms'])

        radius_m = float(cfg.get('radius_m', 35.0))
        pre_seconds = float(cfg.get('pre_seconds', 1.5))
        post_seconds = float(cfg.get('post_seconds', 1.5))
        smooth_window = int(cfg.get('smooth_window', 5))
        straight_thresh = float(cfg.get('straight_thresh', 25.0))
        turn_lo = float(cfg.get('turn_lo', 45.0))
        turn_hi = float(cfg.get('turn_hi', 135.0))
        uturn_thresh = float(cfg.get('uturn_thresh', 150.0))

        intersection_results = []
        for _, g in df.groupby('vehicle_id', sort=False):
            res = process_vehicle(
                g,
                (cx, cy),
                radius_m,
                pre_seconds,
                post_seconds,
                smooth_window,
                straight_thresh,
                turn_lo,
                turn_hi,
                uturn_thresh
            )
            if res is not None:
                res['road_id'] = road_id
                intersection_results.append(res)

        all_results.extend(intersection_results)
        intersection_counts.append((road_id, len(intersection_results)))
        print(f"Processed {road_id}: {len(intersection_results)} classified trajectories.")

    out_df = pd.DataFrame(all_results)

    output_columns = ['vehicle_id', 'date', 'seg_id', 'road_id', 'movement']
    if len(out_df) == 0:
        out_df_final = pd.DataFrame(columns=output_columns)
    else:
        for col in output_columns:
            if col not in out_df.columns:
                if col in ('vehicle_id', 'seg_id'):
                    out_df[col] = 0
                else:
                    out_df[col] = ''

        out_df['vehicle_id'] = out_df['vehicle_id'].astype(int)
        out_df['seg_id'] = out_df['seg_id'].fillna(0).astype(int)
        out_df['movement'] = out_df['movement'].fillna('').astype(str)
        out_df['date'] = out_df['date'].fillna('').astype(str)
        out_df['road_id'] = out_df['road_id'].fillna('').astype(str)

        out_df_final = out_df[output_columns].copy()

    out_df_final.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')

    total = len(out_df_final)
    print(f"Done. Wrote {OUTPUT_CSV_PATH} with {total} classified trajectories.")
    for road_id, count in intersection_counts:
        print(f"  - {road_id}: {count}")
    if total:
        print(out_df_final.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
