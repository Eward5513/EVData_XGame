#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys

import numpy as np
import pandas as pd

# ------------------ configuration ------------------

INTERSECTIONS = [
    {
        'road_id': 'A0003',
        'input_csv': '/home/tzhang174/EVData_XGame/A0003_split.csv',
        'output_csv': '/home/tzhang174/EVData_XGame/A0003_smoother.csv',
        'lon0': 123.152539,
        'lat0': 32.345137,
        'window': 7,
        'method': 'ma',
        'extra_columns': False,
    },
    {
        'road_id': 'A0008',
        'input_csv': '/home/tzhang174/EVData_XGame/A0008_split.csv',
        'output_csv': '/home/tzhang174/EVData_XGame/A0008_smoother.csv',
        'lon0': 123.181261,
        'lat0': 32.327137,
        'window': 7,
        'method': 'ma',
        'extra_columns': False,
    },
]

# ------------------ math helpers ------------------

R_EARTH = 6378137.0

def lonlat_to_xy(lon, lat, lon0, lat0):
    lat0_rad = math.radians(lat0)
    x = np.radians(lon - lon0) * R_EARTH * math.cos(lat0_rad)
    y = np.radians(lat - lat0) * R_EARTH
    return x, y

def xy_to_lonlat(x, y, lon0, lat0):
    lat0_rad = math.radians(lat0)
    lon = lon0 + np.degrees(x / (R_EARTH * math.cos(lat0_rad)))
    lat = lat0 + np.degrees(y / R_EARTH)
    return lon, lat

# ------------------ smoothing helpers ------------------

def get_time_ms(df: pd.DataFrame) -> pd.Series:
    if 'collectiontime' in df.columns and df['collectiontime'].notna().any():
        return pd.to_numeric(df['collectiontime'], errors='coerce')
    if 'date' in df.columns and 'time_stamp' in df.columns:
        dt = pd.to_datetime(df['date'].astype(str) + ' ' + df['time_stamp'].astype(str),
                            errors='coerce', utc=True)
        return dt.view('int64') // 10**6
    return pd.Series(np.arange(len(df)), index=df.index, dtype='int64')

def moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(arr) <= 2:
        return arr.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    if len(arr) <= pad:
        return arr.copy()
    left = arr[1:pad+1][::-1]
    right = arr[-pad-1:-1][::-1]
    arr_pad = np.concatenate([left, arr, right])
    kernel = np.ones(k, dtype=float) / k
    out = np.convolve(arr_pad, kernel, mode='valid')
    return out[:len(arr)]

def savgol_smooth(arr: np.ndarray, k: int, poly: int = 2) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
        if k % 2 == 0:
            k += 1
        if k >= len(arr):
            return moving_average(arr, min(5, len(arr) if len(arr)%2==1 else len(arr)-1))
        return savgol_filter(arr, window_length=k, polyorder=min(poly, k-1), mode='interp')
    except Exception:
        return moving_average(arr, k)

def smooth_series(x: np.ndarray, method: str, window: int) -> np.ndarray:
    if method == 'sg':
        return savgol_smooth(x, window, poly=2)
    return moving_average(x, window)

# ------------------ processing logic ------------------

def smooth_intersection(cfg: dict):
    road_id = cfg.get('road_id', 'UNKNOWN')
    input_csv = cfg.get('input_csv')
    output_csv = cfg.get('output_csv')
    method = cfg.get('method', 'ma')
    window = int(cfg.get('window', 5))
    extra_columns = bool(cfg.get('extra_columns', False))

    if not input_csv or not output_csv:
        print(f"Skipping {road_id}: missing input or output path")
        return

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"WARNING: Input CSV not found for {road_id}: {input_csv}")
        return
    except Exception as exc:
        print(f"ERROR: Failed to read {input_csv} for {road_id}: {exc}")
        return

    essential = ['vehicle_id', 'longitude', 'latitude']
    for col in essential:
        if col not in df.columns:
            print(f"ERROR: Missing column '{col}' in {input_csv}. Skipping {road_id}.", file=sys.stderr)
            return

    df['_t_ms'] = get_time_ms(df)
    df = df.sort_values(['vehicle_id', '_t_ms']).copy()

    lon0 = cfg.get('lon0')
    lat0 = cfg.get('lat0')
    if lon0 is None:
        lon0 = float(np.nanmedian(df['longitude']))
    if lat0 is None:
        lat0 = float(np.nanmedian(df['latitude']))

    lon0 = float(lon0)
    lat0 = float(lat0)

    X, Y = lonlat_to_xy(df['longitude'].values, df['latitude'].values, lon0, lat0)
    df['_x'] = X
    df['_y'] = Y

    x_s_all = np.empty_like(X)
    y_s_all = np.empty_like(Y)

    indexer = df.index
    for _, g in df.groupby('vehicle_id', sort=False):
        idx = g.index.values
        x = g['_x'].values
        y = g['_y'].values
        xs = smooth_series(x, method, window)
        ys = smooth_series(y, method, window)
        positions = indexer.get_indexer(idx)
        x_s_all[positions] = xs
        y_s_all[positions] = ys

    lon_s, lat_s = xy_to_lonlat(x_s_all, y_s_all, lon0, lat0)

    out = df.copy()
    if extra_columns:
        out['smoothed_longitude'] = lon_s
        out['smoothed_latitude'] = lat_s
    else:
        out['longitude'] = lon_s
        out['latitude'] = lat_s

    out = out.drop(columns=['_t_ms', '_x', '_y'])

    try:
        out.to_csv(output_csv, index=False, encoding='utf-8')
    except Exception as exc:
        print(f"ERROR: Failed to write {output_csv} for {road_id}: {exc}")
        return

    print(f"{road_id}: smoothing complete -> {output_csv}")
    print(f"  method={method}, window={window}, origin=({lon0:.6f},{lat0:.6f}), extra_columns={extra_columns}")


def main():
    print("Starting smoothing for configured intersections...")
    for cfg in INTERSECTIONS:
        smooth_intersection(cfg)


    essential = ['vehicle_id', 'longitude', 'latitude']
    for col in essential:
        if col not in df.columns:
            print(f"ERROR: Missing column '{col}' in input.", file=sys.stderr)
            sys.exit(2)

    df['_t_ms'] = get_time_ms(df)
    df = df.sort_values(['vehicle_id', '_t_ms']).copy()

    lon0 = float(args.lon0) if args.lon0 is not None else float(np.nanmedian(df['longitude']))
    lat0 = float(args.lat0) if args.lat0 is not None else float(np.nanmedian(df['latitude']))

    X, Y = lonlat_to_xy(df['longitude'].values, df['latitude'].values, lon0, lat0)
    df['_x'] = X
    df['_y'] = Y

    x_s_all = np.empty_like(X)
    y_s_all = np.empty_like(Y)

    # use index positions for assignment
    pos = np.arange(len(df))
    for _, g in df.groupby('vehicle_id', sort=False):
        idx = g.index.values
        x = g['_x'].values
        y = g['_y'].values
        xs = smooth_series(x, args.method, args.window)
        ys = smooth_series(y, args.method, args.window)
        # place back using integer positions
        x_s_all[df.index.get_indexer(idx)] = xs
        y_s_all[df.index.get_indexer(idx)] = ys

    lon_s, lat_s = xy_to_lonlat(x_s_all, y_s_all, lon0, lat0)

    out = df.copy()
    if args.extra:
        out['smoothed_longitude'] = lon_s
        out['smoothed_latitude'] = lat_s
    else:
        out['longitude'] = lon_s
        out['latitude'] = lat_s

    out = out.drop(columns=['_t_ms', '_x', '_y'])
    out.to_csv(args.out_csv, index=False, encoding='utf-8')
    print(f"Done. Wrote smoothed CSV to {args.out_csv}")
    print(f"Method={args.method}, window={args.window}, origin=({lon0:.6f},{lat0:.6f})")
    if args.extra:
        print("Extra columns added: smoothed_longitude, smoothed_latitude.")

if __name__ == '__main__':
    main()
