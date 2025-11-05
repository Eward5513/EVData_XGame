#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split trajectories in A0003.csv and A0008.csv by adaptive time-gap rule.

Output files:
- A0003_split.csv
- A0008_split.csv

Schema: same as input, but insert seg_id immediately after vehicle_id.
seg_id starts at 0 per (vehicle_id, date) trajectory; if no split occurs, all rows have seg_id=0.
"""

import pandas as pd
import numpy as np
import math
import os
from config import BASE_DIR, CENTERS

# Intersection centers are passed per road; no global default center

# Manual overrides: hardcoded segmentation per (road_id, vehicle_id, date)
# Define segments using 1-based inclusive positions within the group's time-ordered rows.
# Example: ('A0003', 543, '2024-06-20'): {'segments': [(1, 3), (4, None)]}
# means: first 3 points = seg 0, remaining points = seg 1.
MANUAL_OVERRIDES: dict[tuple[str, int, str], dict] = {
    # Add your overrides here:
    ('A0003', 543, '2024-06-20'): {'segments': [(1, 3), (4, None)]},
    ('A0003', 296, '2024-06-19'): {'segments': [(1, 25), (26, None)]},
    ('A0008', 132, '2024-06-17'): {'segments': [(1, 2)]},
    ('A0008', 622, '2024-06-20'): {'segments': [(3, 4)]},
    ('A0008', 348, '2024-06-19'): {'segments': [(1, 3)]},
    ('A0003', 287, '2024-06-19'): {'segments': [(3, 20),(55,66),(67,None)]},
    ('A0003', 1157, '2024-06-21'): {'segments': [(1, 6),(7,None)]},
    ('A0008', 1665, '2024-06-17'): {'segments': [(1, 10),(11,14),(15,None)]},
    ('A0008', 18, '2024-06-19'): {'segments': [(1, 12),(13,23),(24,None)]},
    ('A0008', 1999, '2024-06-18'): {'segments': [(1, 6),(7,None)]},
    ('A0008', 1303, '2024-06-21'): {'segments': [(1, 14),(15,None)]},
    ('A0008', 1775, '2024-06-19'): {'segments': [(1, 3),(3,None)]},
    ('A0003', 1581, '2024-06-17'): {'segments': [(1, 2),(2,None)]},
    ('A0003', 2297, '2024-06-19'): {'segments': [(1, 2),(2,None)]},
    ('A0003', 217, '2024-06-19'): {'segments': [(1, 10),(79,None)]},
    ('A0003', 1614, '2024-06-17'): {'segments': [(20,None)]},
    ('A0003', 812, '2024-06-18'): {'segments': [(11,None)]},
    ('A0003', 1117, '2024-06-21'): {'segments': [(68,None)]},
}

# Manual exclusions: remove entire segments after splitting/overrides.
# Key: (road_id, vehicle_id, date, seg_id)
# Example: {('A0008', 1775, '2024-06-19', -1)}  # seg_id=-1 means remove ALL segments for that (road_id, vehicle_id, date)
MANUAL_EXCLUSIONS: set[tuple[str, int, str, int]] = {
    ('A0003', 27, '2024-06-19', -1),
    ('A0003', 663, '2024-06-20', -1),
    ('A0003', 276, '2024-06-19', -1),
    ('A0003', 1443, '2024-06-17', 1),
    ('A0003', 496, '2024-06-20', 1),

}

# Center-based splitting hysteresis thresholds (meters)
# near: considered "near center" when distance <= NEAR_RADIUS_M
# far: considered "far from center" when distance >= FAR_RADIUS_M
# Using hysteresis avoids flapping near the boundary
NEAR_RADIUS_M = 100.0
FAR_RADIUS_M = 150.0
DR_NOISE_EPS_M = 0.5  # minimal radial change to consider trend change (meters)


# Exclusion radius for preprocessing (meters)
EXCLUDE_OUTSIDE_RADIUS_M = 200.0
# If a segment has no points within the exclusion radius, keep it when the
# angular change of start vs end around the center exceeds this threshold (deg)
ANGULAR_CHANGE_DEG_THRESHOLD = 60.0


def exclude_segments_outside_radius(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    radius_m: float = EXCLUDE_OUTSIDE_RADIUS_M,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Remove segments (vehicle_id, date, seg_id) whose ALL points lie beyond radius_m from the center,
    EXCEPT keep segments whose start vs end polar angle around the center changes by more than
    ANGULAR_CHANGE_DEG_THRESHOLD degrees.

    Returns (kept_df, excluded_df, stats_dict).
    """
    if df.empty:
        return df, df.iloc[0:0], {"excluded_segments": 0, "excluded_rows": 0}

    out = df.copy()
    # Ensure seg_id exists and is integer
    if 'seg_id' not in out.columns:
        raise ValueError("exclude_segments_outside_radius requires seg_id column")

    # Compute per-row distance to center (meters)
    lats = out['latitude'].to_numpy()
    lons = out['longitude'].to_numpy()
    dy = (lats - center_lat) * 111000.0
    dx = (lons - center_lon) * 111000.0 * np.cos(np.radians(lats))
    dist = np.sqrt(dx * dx + dy * dy)
    near = dist <= float(radius_m)

    # For each (vehicle_id, date, seg_id), keep if ANY near==True; else evaluate angular change rule
    grp_keys = ['vehicle_id', 'date', 'seg_id']
    # Make sure seg_id is int for stable grouping
    out['seg_id'] = out['seg_id'].astype('int64')
    any_near = (
        pd.Series(near, index=out.index)
        .groupby([out['vehicle_id'], out['date'], out['seg_id']])
        .transform('any')
    )

    # Start with keep mask based on near; we'll add angle-based keep for groups with no near points
    keep_mask = any_near.copy()

    # Helper: compute normalized polar angle (degrees) of vector center->(lat,lon)
    def compute_angle_deg(lat_val: float, lon_val: float) -> float:
        dy_m = (float(lat_val) - float(center_lat)) * 111000.0
        dx_m = (float(lon_val) - float(center_lon)) * 111000.0 * math.cos(math.radians(float(lat_val)))
        ang_deg = math.degrees(math.atan2(dy_m, dx_m))
        # Normalize to [0,360)
        if ang_deg < 0:
            ang_deg += 360.0
        return ang_deg

    # Helper: minimal absolute angular difference (degrees) accounting for wrap-around
    def ang_diff_deg(a: float, b: float) -> float:
        d = abs(((a - b + 180.0) % 360.0) - 180.0)
        return d

    # Evaluate angle rule per group only for those with no near points
    for (vid, date, sid), idx in out.groupby(grp_keys).groups.items():
        # Check group near status using label-based indexing
        any_near_group = bool(any_near.loc[idx].any()) if len(idx) > 0 else False
        if any_near_group:
            continue  # already kept

        sub = out.loc[idx]
        if sub.empty:
            continue
        # Order chronologically
        if 'collectiontime' in sub.columns:
            sub_ord = sub.sort_values('collectiontime')
        else:
            sub_ord = sub
        # Get start and end
        start_row = sub_ord.iloc[0]
        end_row = sub_ord.iloc[-1]
        try:
            a0 = compute_angle_deg(float(start_row['latitude']), float(start_row['longitude']))
            a1 = compute_angle_deg(float(end_row['latitude']), float(end_row['longitude']))
            delta = ang_diff_deg(a1, a0)
            if delta > float(ANGULAR_CHANGE_DEG_THRESHOLD):
                # Keep entire group by angle rule
                keep_mask.loc[idx] = True
        except Exception:
            # If angle computation fails, do not change keep_mask (remain excluded)
            pass

    kept = out[keep_mask].copy()
    excluded = out[~keep_mask].copy()
    excluded_rows = int(len(excluded))
    excluded_segments = 0
    if excluded_rows > 0:
        excluded['excluded_reason'] = 'outside_200m_all_points'
        # Count unique segments excluded
        excluded_segments = excluded.drop_duplicates(subset=grp_keys).shape[0]

    stats = {"excluded_segments": excluded_segments, "excluded_rows": excluded_rows}
    return kept, excluded, stats




def apply_manual_overrides_inline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply hardcoded manual segmentation after automatic logic.
    MANUAL_OVERRIDES keys: (road_id, vehicle_id, date)
    Value supports either:
      - {'segments': [(start1, end1), (start2, end2), ...]} with 1-based inclusive indices; end=None means to the last point.
      - {'cuts': [k1, k2, ...]} with 1-based positions to split AFTER (equivalent to segments [(1,k1), (k1+1,k2), (k2+1,None)]).

    If provided segments/cuts do NOT cover all points within the group, the uncovered
    points are treated as "not needed" and will be removed from the output. Only the
    specified ranges are kept and re-labeled with seg_id starting from 0 in listed order.
    """
    if not MANUAL_OVERRIDES:
        return df

    out = df.copy()
    # Normalize key column types to ensure mask matches
    if 'road_id' in out.columns:
        out['road_id'] = out['road_id'].astype(str)
    if 'vehicle_id' in out.columns:
        out['vehicle_id'] = out['vehicle_id'].astype('int64')
    if 'date' in out.columns:
        out['date'] = out['date'].astype(str)
    # Ensure required columns
    required_cols = ['road_id', 'vehicle_id', 'date', 'collectiontime', 'seg_id']
    for c in required_cols:
        if c not in out.columns:
            raise ValueError(f"Dataframe missing required column: {c}")

    # Process each override
    for (road_id, vehicle_id, date), spec in MANUAL_OVERRIDES.items():
        mask = (out['road_id'] == str(road_id)) & (out['vehicle_id'] == int(vehicle_id)) & (out['date'] == str(date))
        idx = out[mask].index
        if len(idx) == 0:
            continue
        # Order rows chronologically within the group
        sub = out.loc[idx].sort_values('collectiontime')
        ordered_idx = sub.index.to_list()
        n = len(ordered_idx)
        if n == 0:
            continue

        # Build segments from spec
        segments: list[tuple[int, int]] = []  # 0-based inclusive ranges over ordered positions
        if 'segments' in spec and isinstance(spec['segments'], (list, tuple)):
            for rng in spec['segments']:
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    continue
                s1, e1 = rng[0], rng[1]
                if not isinstance(s1, int):
                    continue
                start0 = max(0, s1 - 1)
                end0 = n - 1 if (e1 is None) else max(0, int(e1) - 1)
                if start0 > end0 or start0 >= n:
                    continue
                end0 = min(end0, n - 1)
                segments.append((start0, end0))
        elif 'cuts' in spec and isinstance(spec['cuts'], (list, tuple)):
            cuts = sorted(int(k) for k in spec['cuts'] if isinstance(k, (int, np.integer)))
            prev = 0
            for cpos in cuts:
                end0 = min(max(0, cpos - 1), n - 1)
                if prev <= end0:
                    segments.append((prev, end0))
                prev = end0 + 1
            if prev <= n - 1:
                segments.append((prev, n - 1))
        else:
            # If spec malformed, skip
            continue

        if not segments:
            continue

        # Build replacement rows allowing overlapping positions to be duplicated across segments
        replacement_rows: list[pd.Series] = []
        cur_seg = 0
        for (a, b) in segments:
            for pos in range(a, b + 1):
                if 0 <= pos < n:
                    src_idx = ordered_idx[pos]
                    row = out.loc[src_idx].copy()
                    row['seg_id'] = cur_seg
                    replacement_rows.append(row)
            cur_seg += 1

        # Replace the group's rows with the reconstructed rows (drop uncovered points and allow duplicates)
        out = out.drop(index=ordered_idx)
        if replacement_rows:
            out = pd.concat([out, pd.DataFrame(replacement_rows)], ignore_index=True)

    return out

def apply_manual_exclusions_inline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows for segments listed in MANUAL_EXCLUSIONS.
    Keys: (road_id, vehicle_id, date, seg_id).
    Occurs after automatic splitting and manual overrides, before radius exclusion.
    """
    if not MANUAL_EXCLUSIONS:
        return df
    out = df.copy()
    required_cols = ['road_id', 'vehicle_id', 'date', 'seg_id']
    for c in required_cols:
        if c not in out.columns:
            raise ValueError(f"Dataframe missing required column for manual exclusions: {c}")
    out['road_id'] = out['road_id'].astype(str)
    out['vehicle_id'] = out['vehicle_id'].astype('int64')
    out['date'] = out['date'].astype(str)
    out['seg_id'] = out['seg_id'].astype('int64')

    drop_idx_all: list[int] = []
    for (road_id, vehicle_id, date, seg_id) in MANUAL_EXCLUSIONS:
        base_mask = (
            (out['road_id'] == str(road_id)) &
            (out['vehicle_id'] == int(vehicle_id)) &
            (out['date'] == str(date))
        )
        if int(seg_id) == -1:
            mask = base_mask  # remove all segments for this (road_id, vehicle_id, date)
        else:
            mask = base_mask & (out['seg_id'] == int(seg_id))
        drop_idx_all.extend(out[mask].index.tolist())
    if drop_idx_all:
        out = out.drop(index=drop_idx_all)
    return out
def adaptive_split_segment_indices(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    max_gap_seconds: float = 180.0,
    enable_center_splitting: bool = True,
    near_radius_m: float = NEAR_RADIUS_M,
    far_radius_m: float = FAR_RADIUS_M,
) -> tuple[pd.Series, int]:
    """
    Compute seg_id per (vehicle_id, date) group using adaptive time-gap splitting.
    - Start seg_id=0; for each point i, if time gap to previous point > 2x median of previous gaps
      within current seg, or > max_gap_seconds, then start a new seg (seg_id += 1) at i.
    - Requires df sorted by collectiontime within each group.
    Returns (seg_id Series, extra_center_splits) where extra_center_splits counts splits caused by center rule.
    """
    # We'll collect confirmed split indices first, then build seg_id at the end
    confirmed_split_local_indices: list[int] = []

    # Initialize time-gap baseline history
    prev_diffs: list[float] = []

    times = df['collectiontime'].to_numpy()  # expected in ms

    # Precompute distance to center for hysteresis-based splitting
    lats = df['latitude'].to_numpy()
    lons = df['longitude'].to_numpy()
    # Use per-point local scale for longitude
    avg_lats_rad = np.radians(lats)
    dy_m = (lats - center_lat) * 111000.0
    dx_m = (lons - center_lon) * 111000.0 * np.cos(avg_lats_rad)
    dist_to_center = np.sqrt(dx_m * dx_m + dy_m * dy_m)

    # Helper: classify near/far using hysteresis windows (for readability)

    # Center state machine with hysteresis
    def classify_near_far(distance_m: float, prev_state: str | None) -> str:
        if prev_state == 'near':
            return 'far' if distance_m >= far_radius_m else 'near'
        if prev_state == 'far':
            return 'near' if distance_m <= near_radius_m else 'far'
        # unknown -> decide based on hysteresis window
        if distance_m <= near_radius_m:
            return 'near'
        if distance_m >= far_radius_m:
            return 'far'
        return 'far'

    # New pre-splitting state
    pre_candidate_idx: int | None = None  # first re-approach index (tentative split start)
    # Track radial trend: 'away' or 'approach'
    prev_trend: str | None = None

    # Count of splits confirmed by center pre-splitting
    extra_center_splits = 0

    n = len(df)
    for i in range(1, n):
        cur_diff = (times[i] - times[i - 1]) / 1000.0  # seconds

        # 1) Time-gap splitting (immediate)
        time_gap_split = False
        if len(prev_diffs) >= 3:
            baseline = float(np.median(prev_diffs))
            if baseline > 0 and float(cur_diff) > 2.0 * baseline:
                time_gap_split = True
        if float(cur_diff) > float(max_gap_seconds):
            time_gap_split = True

        if time_gap_split:
            confirmed_split_local_indices.append(i)
            prev_diffs = []
            # Reset pre-splitting state across time-based boundary
            pre_candidate_idx = None
            prev_trend = None

        # 2) Center pre-splitting (tentative confirmation model)
        if enable_center_splitting:
            r_prev = float(dist_to_center[i - 1])
            r_cur = float(dist_to_center[i])
            dr = r_cur - r_prev
            cur_trend: str | None = None
            if dr > DR_NOISE_EPS_M:
                cur_trend = 'away'
            elif dr < -DR_NOISE_EPS_M:
                cur_trend = 'approach'

            # Start a tentative split when we are far and switch from away -> approach
            if pre_candidate_idx is None:
                if cur_trend == 'approach' and prev_trend == 'away' and r_cur >= float(far_radius_m) and r_prev >= float(far_radius_m):
                    pre_candidate_idx = i  # first point re-approaching the center while far
            else:
                # If we reach near zone, confirm the split at the candidate index
                if r_cur <= float(near_radius_m):
                    confirmed_split_local_indices.append(pre_candidate_idx)
                    extra_center_splits += 1
                    pre_candidate_idx = None
                    # After confirmation, reset trend to avoid immediate re-trigger
                    prev_trend = None
                else:
                    # If another away->approach occurs while still far, move candidate to the new index (invalidate previous)
                    if cur_trend == 'approach' and prev_trend == 'away' and r_cur >= float(far_radius_m) and r_prev >= float(far_radius_m):
                        pre_candidate_idx = i

            # Update previous trend when determinable
            if cur_trend is not None:
                prev_trend = cur_trend

        prev_diffs.append(float(cur_diff))

    # Build seg_id series from confirmed split indices
    confirmed_split_local_indices = sorted(set(confirmed_split_local_indices))
    seg_ids_local = np.zeros(n, dtype=np.int64)
    current_seg = 0
    cursor = 0
    for split_idx in confirmed_split_local_indices:
        # rows [cursor, split_idx) belong to current_seg
        # rows [split_idx, ...) will have new segment
        if split_idx > cursor:
            seg_ids_local[split_idx:] += 1
            current_seg += 1
            cursor = split_idx

    # Map back to original df indices
    seg_ids = pd.Series(seg_ids_local, index=df.index, dtype='int64')

    return seg_ids, extra_center_splits


def process_file(
    input_path: str,
    output_path: str,
    center_lat: float,
    center_lon: float,
    max_gap_seconds: float = 180.0,
) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path)

    # Validate required columns
    required_cols = ['vehicle_id', 'collectiontime', 'date']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Sort by vehicle_id, date, collectiontime to ensure correct ordering
    df = df.sort_values(['vehicle_id', 'date', 'collectiontime']).reset_index(drop=True)

    # Compute seg_id per (vehicle_id, date) and collect center split metrics
    seg_ids_all = []
    affected_trajectories = 0
    center_splits_total = 0
    for (vehicle_id, date), group_idx in df.groupby(['vehicle_id', 'date']).groups.items():
        group_df = df.loc[group_idx]
        seg_ids_group, extra_center_splits = adaptive_split_segment_indices(
            group_df,
            max_gap_seconds=max_gap_seconds,
            enable_center_splitting=True,
            near_radius_m=NEAR_RADIUS_M,
            far_radius_m=FAR_RADIUS_M,
            center_lat=center_lat,
            center_lon=center_lon,
        )
        seg_ids_all.append(seg_ids_group)
        if extra_center_splits > 0:
            affected_trajectories += 1
            center_splits_total += extra_center_splits

    seg_ids_concat = pd.concat(seg_ids_all).sort_index()

    # Insert seg_id after vehicle_id column
    cols = list(df.columns)
    try:
        vid_pos = cols.index('vehicle_id')
    except ValueError:
        vid_pos = 0
    df.insert(vid_pos + 1, 'seg_id', seg_ids_concat.astype('int64').to_numpy())

    # Apply inline hardcoded overrides
    df = apply_manual_overrides_inline(df)

    # Apply manual exclusions inline after splitting/overrides
    df = apply_manual_exclusions_inline(df)

    # Exclude segments whose all points are outside the center radius
    kept_df, excluded_df, ex_stats = exclude_segments_outside_radius(
        df, center_lat=center_lat, center_lon=center_lon, radius_m=EXCLUDE_OUTSIDE_RADIUS_M
    )

    # Write kept (split) file
    print(f"Writing {output_path} ...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    kept_df.to_csv(output_path, index=False)
    print(f"Wrote {len(kept_df)} rows to {output_path}")

    # Write excluded file (parallel to split file name)
    excluded_path = output_path.replace('_split.csv', '_excluded.csv')
    if len(excluded_df) > 0:
        print(f"Writing excluded rows to {excluded_path} ...")
        excluded_df.to_csv(excluded_path, index=False)
        print(
            f"Excluded segments={ex_stats['excluded_segments']}, rows={ex_stats['excluded_rows']} (radius={EXCLUDE_OUTSIDE_RADIUS_M}m)"
        )
    else:
        # If no excluded rows, and an older excluded file exists, we do not touch it
        print("No segments excluded by radius rule.")
    # Report center-based splitting metrics
    print(
        f"Center-based splitting: affected trajectories = {affected_trajectories}, "
        f"extra splits due to center = {center_splits_total}"
    )


def main():
    data_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)
    # Per-road intersection centers (lat, lon)
    A0003_CENTER = CENTERS['A0003']
    A0008_CENTER = CENTERS['A0008']

    process_file(
        f'{data_dir}/A0003.csv',
        f'{data_dir}/A0003_split.csv',
        center_lat=A0003_CENTER[0],
        center_lon=A0003_CENTER[1],
    )
    process_file(
        f'{data_dir}/A0008.csv',
        f'{data_dir}/A0008_split.csv',
        center_lat=A0008_CENTER[0],
        center_lon=A0008_CENTER[1],
    )


## Module is imported by main.py; no standalone execution


