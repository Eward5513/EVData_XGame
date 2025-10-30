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
    ('A0003', 1157, '2024-06-21'): {'segments': [(1, 6),(7,None)]},
    ('A0008', 1665, '2024-06-17'): {'segments': [(1, 10),(11,14),(15,None)]},
}

# Center-based splitting hysteresis thresholds (meters)
# near: considered "near center" when distance <= NEAR_RADIUS_M
# far: considered "far from center" when distance >= FAR_RADIUS_M
# Using hysteresis avoids flapping near the boundary
NEAR_RADIUS_M = 100.0
FAR_RADIUS_M = 150.0
DR_NOISE_EPS_M = 0.5  # minimal radial change to consider trend change (meters)


def apply_manual_forced_splits(
    df: pd.DataFrame,
    rules_csv_path: str,
) -> pd.DataFrame:
    """
    Apply manual forced splits after automatic segmentation.
    
    Rules CSV schema (columns):
    - road_id: str
    - vehicle_id: int
    - date: YYYY-MM-DD
    - seg_id: int  (the existing segment to split)
    - split_at_time: HH:MM:SS  (force a split at the first row with time_stamp >= this)
      OR
      split_at_collectiontime: int (ms since epoch; if provided, use the first row with collectiontime >= this)
    Multiple rules can exist per (vehicle_id,date,seg_id).
    The function re-assigns seg_id within the affected (vehicle_id,date) group by incrementing seg_id
    for rows at and after the split index. If multiple rules affect the same group, they are applied in
    ascending order of the split index.
    """
    if not os.path.exists(rules_csv_path):
        return df

    rules = pd.read_csv(rules_csv_path)
    # Normalize columns
    for col in ['road_id', 'date']:
        if col in rules.columns:
            rules[col] = rules[col].astype(str)

    # Group rules per (road_id, vehicle_id, date)
    key_cols = ['road_id', 'vehicle_id', 'date']
    missing = [k for k in key_cols if k not in rules.columns]
    if missing:
        print(f"manual_splits.csv missing columns: {missing}; skip manual splits")
        return df

    df = df.copy()
    # Ensure needed columns exist in df
    required_df_cols = ['road_id', 'vehicle_id', 'date', 'seg_id', 'time_stamp', 'collectiontime']
    for c in required_df_cols:
        if c not in df.columns:
            raise ValueError(f"Dataframe missing required column: {c}")

    # Process per (road_id, vehicle_id, date)
    for (road_id, vehicle_id, date), df_idx in df.groupby(key_cols).groups.items():
        sub = df.loc[df_idx].copy()
        # Gather rules for this group
        rsub = rules[(rules['road_id'] == str(road_id)) & (rules['vehicle_id'] == int(vehicle_id)) & (rules['date'] == str(date))]
        if rsub.empty:
            continue

        # Prepare split points per seg_id
        split_points_by_seg: dict[int, list[int]] = {}
        for _, r in rsub.iterrows():
            if 'seg_id' not in r or pd.isna(r['seg_id']):
                continue
            seg = int(r['seg_id'])

            # Find split index within this group's rows for this seg
            seg_mask = (sub['seg_id'] == seg)
            seg_rows = sub[seg_mask]
            if seg_rows.empty:
                continue
            # Determine split index by time_stamp or collectiontime
            split_idx_local: int | None = None
            if 'split_at_collectiontime' in r and pd.notna(r['split_at_collectiontime']):
                try:
                    t_ms = int(r['split_at_collectiontime'])
                    match = seg_rows[seg_rows['collectiontime'] >= t_ms]
                    if not match.empty:
                        split_idx_local = int(match.index[0])
                except Exception:
                    pass
            if split_idx_local is None and 'split_at_time' in r and pd.notna(r['split_at_time']):
                try:
                    t_str = str(r['split_at_time']).strip()
                    match = seg_rows[seg_rows['time_stamp'] >= t_str]
                    if not match.empty:
                        split_idx_local = int(match.index[0])
                except Exception:
                    pass
            if split_idx_local is None:
                continue
            split_points_by_seg.setdefault(seg, []).append(split_idx_local)

        if not split_points_by_seg:
            continue

        # For deterministic behavior, apply splits per seg_id in ascending order, and within seg by ascending index
        for seg in sorted(split_points_by_seg.keys()):
            indices = sorted(set(split_points_by_seg[seg]))
            for split_abs_idx in indices:
                # For all rows in (this group) with index >= split_abs_idx and seg_id >= seg,
                # increment seg_id by 1 to create a new segment starting at split
                mask_after = (df.index.isin(df_idx)) & (df.index >= split_abs_idx)
                # Only bump seg_id for the targeted seg and any later segs within the same (vid,date) group
                # so that later splits remain consistent
                df.loc[mask_after & (df['seg_id'] >= seg), 'seg_id'] = df.loc[mask_after & (df['seg_id'] >= seg), 'seg_id'] + 1

    return df


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

        # Assign seg_id per segment starting from 0; drop uncovered points
        seg_id_local = np.full(n, -1, dtype=np.int64)
        cur_seg = 0
        for (a, b) in segments:
            for pos in range(a, b + 1):
                if 0 <= pos < n:
                    seg_id_local[pos] = cur_seg
            cur_seg += 1

        keep_positions = [pos for pos in range(n) if seg_id_local[pos] >= 0]
        drop_positions = [pos for pos in range(n) if seg_id_local[pos] < 0]

        if drop_positions:
            drop_indices = [ordered_idx[pos] for pos in drop_positions]
            out = out.drop(index=drop_indices)

        if keep_positions:
            keep_indices = [ordered_idx[pos] for pos in keep_positions]
            out.loc[keep_indices, 'seg_id'] = np.take(seg_id_local, keep_positions)

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
    manual_rules_path: str | None = None,
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

    # Apply manual forced splits if rules are provided
    if manual_rules_path and os.path.exists(manual_rules_path):
        df = apply_manual_forced_splits(df, manual_rules_path)
    # Apply inline hardcoded overrides
    df = apply_manual_overrides_inline(df)

    print(f"Writing {output_path} ...")
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    # Report center-based splitting metrics
    print(
        f"Center-based splitting: affected trajectories = {affected_trajectories}, "
        f"extra splits due to center = {center_splits_total}"
    )


def main():
    base = '/home/tzhang174/EVData_XGame'
    # Per-road intersection centers (lat, lon) consistent with analyze_direction.py
    A0003_CENTER = (32.345137, 123.152539)
    A0008_CENTER = (32.327137, 123.181261)
    manual_rules = os.path.join(base, 'manual_splits.csv') if os.path.exists(os.path.join(base, 'manual_splits.csv')) else None

    process_file(
        f'{base}/A0003.csv',
        f'{base}/A0003_split.csv',
        center_lat=A0003_CENTER[0],
        center_lon=A0003_CENTER[1],
        manual_rules_path=manual_rules,
    )
    process_file(
        f'{base}/A0008.csv',
        f'{base}/A0008_split.csv',
        center_lat=A0008_CENTER[0],
        center_lon=A0008_CENTER[1],
        manual_rules_path=manual_rules,
    )


if __name__ == '__main__':
    main()


