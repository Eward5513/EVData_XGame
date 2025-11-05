#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refine trajectories after direction classification by trimming off-direction noise
and extracting a clean skeleton per segment. Segments with <5 points are not trimmed.

Inputs:
  - data/A0003_merged.csv, data/A0008_merged.csv
  - data/direction.csv

Outputs:
  - data/A0003_refined.csv, data/A0008_refined.csv
  - data/refined_spans.csv (report kept index ranges per segment)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    CENTERS,
    REFINE_NEAR_RADIUS_M,
    STEP_AXIS_RATIO,
    STEP_MIN_LENGTH_M,
    MAX_OFF_AXIS_CONSEC,
)
from trajectory_utils import (
    step_axis_label,
    axis_from_label,
    center_min_distance_to_segment_m,
    segment_projection_contains_center,
)


def _compute_step_labels(lats, lons, axis_ratio: float, min_step_m: float) -> List[Optional[str]]:
    labels: List[Optional[str]] = []
    n = len(lats)
    for i in range(n - 1):
        labels.append(step_axis_label(lats, lons, i, i + 1, axis_ratio=axis_ratio, min_step_m=min_step_m))
    return labels


def _compute_step_min_dists(lats, lons, center_lat: float, center_lon: float) -> List[float]:
    n = len(lats)
    d: List[float] = []
    for i in range(n - 1):
        d.append(center_min_distance_to_segment_m(i, i + 1, lats, lons, center_lat, center_lon))
    return d


def find_center_anchor(df: pd.DataFrame, center_lat: float, center_lon: float) -> Dict:
    """
    Find nearest step (preferred, with projection inside) or nearest point to the center.
    Returns a dict with keys: {'type': 'step'|'point', 'index': int, 'proj_in': bool, 'd': float}
    """
    g = df.sort_values('collectiontime').reset_index(drop=True)
    lats = g['latitude'].to_numpy()
    lons = g['longitude'].to_numpy()
    n = len(g)

    # Try steps
    best_k, best_d_step, best_proj_in = None, float('inf'), False
    for i in range(n - 1):
        dmin = center_min_distance_to_segment_m(i, i + 1, lats, lons, center_lat, center_lon)
        if dmin < best_d_step:
            best_d_step = dmin
            best_k = i
            best_proj_in = segment_projection_contains_center(i, i + 1, lats, lons, center_lat, center_lon)

    # Try points
    best_p, best_d_point = None, float('inf')
    for i in range(n):
        avg = (float(lats[i]) + float(center_lat)) / 2.0
        dx = (float(lons[i]) - float(center_lon)) * 111000.0 * np.cos(np.radians(avg))
        dy = (float(lats[i]) - float(center_lat)) * 111000.0
        d = float(np.hypot(dx, dy))
        if d < best_d_point:
            best_d_point = d
            best_p = i

    if best_k is not None and best_proj_in:
        return {'type': 'step', 'index': int(best_k), 'proj_in': True, 'd': float(best_d_step)}
    # fallback: nearest point
    return {'type': 'point', 'index': int(best_p if best_p is not None else 0), 'proj_in': False, 'd': float(best_d_point)}


def _first_outside_dir_backward(
    anchor_step_idx: int,
    step_labels: List[Optional[str]],
    step_dmin: List[float],
    near_radius_m: float,
) -> Optional[str]:
    for p in range(anchor_step_idx - 1, -1, -1):
        if step_dmin[p] > near_radius_m:
            lab = step_labels[p]
            if lab is not None:
                return lab
    return None


def _first_outside_dir_forward(
    anchor_step_idx: int,
    step_labels: List[Optional[str]],
    step_dmin: List[float],
    near_radius_m: float,
) -> Optional[str]:
    for q in range(anchor_step_idx, len(step_labels)):
        if step_dmin[q] > near_radius_m:
            lab = step_labels[q]
            if lab is not None:
                return lab
    return None


def expected_dirs_from_label(
    label: str,
    step_labels: List[Optional[str]],
    step_dmin: List[float],
    anchor: Dict,
    near_radius_m: float,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (mode, approach_dir, exit_dir):
      - mode 'straight': approach/exit None
      - mode 'turn'    : both may be inferred from nearest outside steps
    """
    up = str(label).upper()
    if up in ('A1', 'B1'):
        return 'straight', None, None

    # Turn types: prefer label-based expected approach/exit directions when available
    # Mapping convention:
    #   - A2-* turns exit along A-axis (EW). '-1' => E, '-2' => W. Approach along B-axis: N for '-1', S for '-2'.
    #   - B2-* turns exit along B-axis (NS). '-1' => N, '-2' => S. Approach along A-axis: E for '-1', W for '-2'.
    label_turn_map = {
        'A2-1': ('N', 'E'),
        'A2-2': ('S', 'W'),
        'B2-1': ('E', 'N'),
        'B2-2': ('W', 'S'),
        # Right-turn/left-turn variants with "3" use the same approach/exit convention as "2"
        'A3-1': ('N', 'E'),
        'A3-2': ('S', 'W'),
        'B3-1': ('E', 'N'),
        'B3-2': ('W', 'S'),
    }
    if up in label_turn_map:
        app_dir, ext_dir = label_turn_map[up]
        return 'turn', app_dir, ext_dir

    # Fallback: infer directions from data around the center
    if anchor['type'] == 'step':
        k = int(anchor['index'])
    else:
        # nearest point → consider steps (k-1,k) and (k,k+1); forward uses k, backward uses k-1
        k = int(anchor['index'])
    app = _first_outside_dir_backward(k, step_labels, step_dmin, near_radius_m)
    ext = _first_outside_dir_forward(k, step_labels, step_dmin, near_radius_m)
    return 'turn', app, ext


def _scan_crop_forward(
    start_step_idx: int,
    step_labels: List[Optional[str]],
    step_dmin: List[float],
    near_radius_m: float,
    match_axis: Optional[str] = None,
    match_dir: Optional[str] = None,
    max_off_consec: int = 2,
    boundary_trim_min_off: int = 0,
    gate_point_dists: Optional[List[float]] = None,
    sticky_gate: bool = False,
) -> Optional[object]:
    """
    Scan forward over steps; return the first step index at which we should crop (crop from this step's start).
    Only steps with dmin > near_radius_m are checked; others are ignored.
    If match_dir is provided, enforce exact cardinal; else enforce axis via match_axis.
    """
    off_consec = 0
    off_run_start: Optional[int] = None
    entered_far = False
    for s in range(start_step_idx, len(step_labels)):
        # Gating by distance: default uses step_dmin; if gate_point_dists is provided,
        # use the end point (s+1) distance for forward scanning. If sticky_gate is True,
        # once we enter far zone, we keep including subsequent steps without distance check.
        if gate_point_dists is not None:
            if not entered_far:
                gate_idx = s + 1
                if gate_idx >= len(gate_point_dists):
                    gate_idx = len(gate_point_dists) - 1
                if gate_idx < 0 or gate_point_dists[gate_idx] <= near_radius_m:
                    continue
                entered_far = True if sticky_gate else False
        else:
            if not entered_far:
                if step_dmin[s] <= near_radius_m:
                    continue
                entered_far = True if sticky_gate else False
        lab = step_labels[s]
        if lab is None:
            continue
        ok = False
        if match_dir is not None:
            ok = (lab == match_dir)
        elif match_axis is not None:
            ok = (axis_from_label(lab) == match_axis)
        if ok:
            off_consec = 0
            off_run_start = None
        else:
            if off_consec == 0:
                off_run_start = s
            off_consec += 1
            if off_consec > max_off_consec:
                # Threshold-based cut (not boundary tail)
                return (off_run_start, False)
    if boundary_trim_min_off > 0 and off_consec >= boundary_trim_min_off and off_run_start is not None:
        # Boundary tail cut
        return (off_run_start, True)
    return None


def _scan_crop_backward(
    start_step_idx: int,
    step_labels: List[Optional[str]],
    step_dmin: List[float],
    near_radius_m: float,
    match_axis: Optional[str] = None,
    match_dir: Optional[str] = None,
    max_off_consec: int = 2,
    boundary_trim_min_off: int = 0,
    gate_point_dists: Optional[List[float]] = None,
    reverse_labels: bool = False,
    sticky_gate: bool = False,
) -> Optional[object]:
    """
    Scan backward over steps; return the step index after which to start keeping (crop earlier points).
    Only steps with dmin > near_radius_m are checked; others are ignored.
    If match_dir is provided, enforce exact cardinal; else enforce axis via match_axis.
    Return value r means keep from point index (r+1).
    """
    off_consec = 0
    off_run_first: Optional[int] = None
    entered_far = False
    for s in range(start_step_idx, -1, -1):
        # Gating by distance: default uses step_dmin; if gate_point_dists is provided,
        # use the start point (s) distance for backward scanning. If sticky_gate is True,
        # once we enter far zone, include subsequent steps without distance check.
        if gate_point_dists is not None:
            if not entered_far:
                gate_idx = s
                if gate_idx < 0 or gate_idx >= len(gate_point_dists) or gate_point_dists[gate_idx] <= near_radius_m:
                    continue
                entered_far = True if sticky_gate else False
        else:
            if not entered_far:
                if step_dmin[s] <= near_radius_m:
                    continue
                entered_far = True if sticky_gate else False
        lab = step_labels[s]
        if lab is None:
            continue
        if reverse_labels:
            if lab == 'E':
                lab = 'W'
            elif lab == 'W':
                lab = 'E'
            elif lab == 'N':
                lab = 'S'
            elif lab == 'S':
                lab = 'N'
        ok = False
        if match_dir is not None:
            ok = (lab == match_dir)
        elif match_axis is not None:
            ok = (axis_from_label(lab) == match_axis)
        if ok:
            off_consec = 0
            off_run_first = None
        else:
            if off_consec == 0:
                off_run_first = s
            off_consec += 1
            if off_consec > max_off_consec:
                return (off_run_first, False)
    if boundary_trim_min_off > 0 and off_consec >= boundary_trim_min_off and off_run_first is not None:
        return (off_run_first, True)
    return None


def refine_trajectory_by_direction(
    segment_df: pd.DataFrame,
    label: str,
    center_lat: float,
    center_lon: float,
    *,
    near_radius_m: float = REFINE_NEAR_RADIUS_M,
    axis_ratio: float = STEP_AXIS_RATIO,
    min_step_m: float = STEP_MIN_LENGTH_M,
    max_off_axis: int = MAX_OFF_AXIS_CONSEC,
) -> Tuple[pd.DataFrame, int, int, str]:
    """
    Apply direction-consistency trimming. Returns (refined_df, keep_start_idx, keep_end_idx, reason).
    Indices are 0-based over the chronologically ordered points in segment_df.
    """
    if len(segment_df) < 5:
        g = segment_df.sort_values('collectiontime')
        return g, 0, len(g) - 1, 'short'

    g = segment_df.sort_values('collectiontime').reset_index(drop=True)
    lats = g['latitude'].to_numpy()
    lons = g['longitude'].to_numpy()
    n = len(g)

    step_labels = _compute_step_labels(lats, lons, axis_ratio=axis_ratio, min_step_m=min_step_m)
    step_dmin = _compute_step_min_dists(lats, lons, center_lat, center_lon)
    anchor = find_center_anchor(g, center_lat, center_lon)

    # Special rule for 1-second cadence: use min_step_m=0 for labeling and require
    # a stronger off-direction run length (10 consecutive steps) when trimming.
    times = g['collectiontime'].to_numpy() if 'collectiontime' in g.columns else None
    one_sec_all = False
    if times is not None and len(times) >= 2:
        try:
            t = np.asarray(times, dtype=np.int64)
            one_sec_all = bool(np.all(np.diff(t) == 1000))
        except Exception:
            one_sec_all = False
    step_labels_eff = step_labels if not one_sec_all else _compute_step_labels(
        lats, lons, axis_ratio=axis_ratio, min_step_m=0.0
    )

    # Point-wise distance to center for terminal boundary checks (meters)
    point_d2center: List[float] = []
    for i in range(n):
        avg = (float(lats[i]) + float(center_lat)) / 2.0
        dx = (float(lons[i]) - float(center_lon)) * 111000.0 * np.cos(np.radians(avg))
        dy = (float(lats[i]) - float(center_lat)) * 111000.0
        point_d2center.append(float(np.hypot(dx, dy)))

    up = str(label).upper()
    if up in ('C', 'U', 'A1', 'B1'):
        return g, 0, n - 1, 'skip_label'

    keep_start = 0
    keep_end = n - 1

    if up in ('A1-1', 'A1-2', 'B1-1', 'B1-2'):
        # Straight with precise direction
        expected_dir = 'E' if up == 'A1-1' else (
            'W' if up == 'A1-2' else (
                'N' if up == 'B1-1' else 'S'
            )
        )
        # For 1s cadence segments, require longer off-direction run (10 steps)
        max_off_required = (9 if one_sec_all else max_off_axis)
        # Forward scan starting from anchor step
        f_start = anchor['index'] if anchor['type'] == 'step' else anchor['index']
        cut_f = _scan_crop_forward(
            start_step_idx=min(max(0, f_start), len(step_labels) - 1),
            step_labels=step_labels_eff,
            step_dmin=step_dmin,
            near_radius_m=near_radius_m,
            match_axis=None,
            match_dir=expected_dir,
            max_off_consec=max_off_required,
            boundary_trim_min_off=1,
            gate_point_dists=point_d2center,
            sticky_gate=True,
        )
        if cut_f is not None:
            idx, is_boundary = (int(cut_f[0]), bool(cut_f[1])) if isinstance(cut_f, tuple) else (int(cut_f), False)
            if is_boundary:
                # keep up to boundary step's end point
                keep_end = min(keep_end, idx)
            else:
                # threshold exceed: cut off-run entirely
                keep_end = min(keep_end, idx - 1)
        else:
            # Terminal forward fallback: check last point distance and last step direction
            if n >= 2 and point_d2center[-1] > near_radius_m:
                s_last = len(step_labels_eff) - 1
                if s_last >= 0:
                    lab_last = step_labels_eff[s_last]
                    if lab_last is not None and lab_last != expected_dir:
                        keep_end = min(keep_end, int(s_last))

        # Backward scan ending before anchor step
        b_start = (anchor['index'] - 1) if anchor['type'] == 'step' else (anchor['index'] - 1)
        cut_b = _scan_crop_backward(
            start_step_idx=min(max(0, b_start), len(step_labels) - 1),
            step_labels=step_labels_eff,
            step_dmin=step_dmin,
            near_radius_m=near_radius_m,
            match_axis=None,
            match_dir=('E' if expected_dir=='W' else 'W' if expected_dir=='E' else 'N' if expected_dir=='S' else 'S'),
            max_off_consec=max_off_required,
            boundary_trim_min_off=1,
            gate_point_dists=point_d2center,
            reverse_labels=True,
            sticky_gate=True,
        )
        if cut_b is not None:
            idx, is_boundary = (int(cut_b[0]), bool(cut_b[1])) if isinstance(cut_b, tuple) else (int(cut_b), False)
            if is_boundary:
                # For backward boundary trim, keep from point index (idx + 1)
                keep_start = max(keep_start, idx + 1)
            else:
                # Threshold exceed: cut entire off-run (also keep from next point)
                keep_start = max(keep_start, idx + 1)
        else:
            # Terminal backward fallback: check first point distance and first step direction
            if n >= 2 and point_d2center[0] > near_radius_m:
                if len(step_labels_eff) > 0:
                    lab_first = step_labels_eff[0]
                    if lab_first is not None and lab_first != expected_dir:
                        keep_start = max(keep_start, 1)

        reason = 'straight_trim_dir'
    else:
        # Turn types: infer approach/exit directions from data
        mode, app_dir, ext_dir = expected_dirs_from_label(up, step_labels, step_dmin, anchor, near_radius_m)
        if mode != 'turn' or (app_dir is None and ext_dir is None):
            return g, 0, n - 1, 'turn_no_ref'

        f_start = anchor['index'] if anchor['type'] == 'step' else anchor['index']
        if ext_dir is not None:
            cut_f = _scan_crop_forward(
                start_step_idx=min(max(0, f_start), len(step_labels) - 1),
                step_labels=step_labels,
                step_dmin=step_dmin,
                near_radius_m=near_radius_m,
                match_axis=None,
                match_dir=ext_dir,
                max_off_consec=max_off_axis,
                boundary_trim_min_off=1,
                gate_point_dists=point_d2center,
                sticky_gate=True,
            )
            if cut_f is not None:
                idx, is_boundary = (int(cut_f[0]), bool(cut_f[1])) if isinstance(cut_f, tuple) else (int(cut_f), False)
                if is_boundary:
                    keep_end = min(keep_end, idx)
                else:
                    keep_end = min(keep_end, idx - 1)

        b_start = (anchor['index'] - 1) if anchor['type'] == 'step' else (anchor['index'] - 1)
        if app_dir is not None:
            cut_b = _scan_crop_backward(
                start_step_idx=min(max(0, b_start), len(step_labels) - 1),
                step_labels=step_labels,
                step_dmin=step_dmin,
                near_radius_m=near_radius_m,
                match_axis=None,
                match_dir=('E' if app_dir=='W' else 'W' if app_dir=='E' else 'N' if app_dir=='S' else 'S'),
                max_off_consec=max_off_axis,
                boundary_trim_min_off=1,
                gate_point_dists=point_d2center,
                reverse_labels=True,
                sticky_gate=True,
            )
            if cut_b is not None:
                idx, is_boundary = (int(cut_b[0]), bool(cut_b[1])) if isinstance(cut_b, tuple) else (int(cut_b), False)
                if is_boundary:
                    # Drop the boundary point for backward boundary trim (consistent with straight case contract)
                    keep_start = max(keep_start, idx + 1)
                else:
                    keep_start = max(keep_start, idx + 1)

        reason = 'turn_trim'

    # Clamp indices and slice
    keep_start = int(max(0, min(keep_start, n - 1)))
    keep_end = int(max(keep_start, min(keep_end, n - 1)))
    refined = g.iloc[keep_start: keep_end + 1].copy()
    return refined, keep_start, keep_end, reason


def refine_one_source(
    df_merged: pd.DataFrame,
    df_dir: pd.DataFrame,
    road_id: str,
    center_lat: float,
    center_lon: float,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Refine all segments for a given road source, returning refined dataframe and span records."""
    df = df_merged.copy()
    df = df.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime']).reset_index(drop=True)
    spans: List[Dict] = []
    refined_rows: List[pd.DataFrame] = []

    # build mapping from (vehicle_id,date,seg_id) to label
    dmap: Dict[Tuple[int, str, int], str] = {}
    dsub = df_dir[df_dir['road_id'] == road_id]
    for _, r in dsub.iterrows():
        try:
            dmap[(int(r['vehicle_id']), str(r['date']), int(r['seg_id']))] = str(r['direction'])
        except Exception:
            continue

    for (vid, date, sid), sub in df.groupby(['vehicle_id', 'date', 'seg_id'], sort=False):
        sub = sub.sort_values('collectiontime')
        key = (int(vid), str(date), int(sid))
        label = dmap.get(key)
        if label is None:
            refined = sub.copy()
            rsn = 'no_label'
            keep_s, keep_e = 0, len(refined) - 1
        else:
            refined, keep_s, keep_e, rsn = refine_trajectory_by_direction(
                sub, label, center_lat=center_lat, center_lon=center_lon,
                near_radius_m=REFINE_NEAR_RADIUS_M,
                axis_ratio=STEP_AXIS_RATIO,
                min_step_m=STEP_MIN_LENGTH_M,
                max_off_axis=MAX_OFF_AXIS_CONSEC,
            )
        spans.append({
            'road_id': road_id,
            'vehicle_id': int(vid),
            'date': str(date),
            'seg_id': int(sid),
            'direction': label if label is not None else '',
            'keep_start_idx': int(keep_s),
            'keep_end_idx': int(keep_e),
            'points_before': int(len(sub)),
            'points_after': int(len(refined)),
            'reason': rsn,
        })
        refined_rows.append(refined)

    out = pd.concat(refined_rows, ignore_index=True) if refined_rows else df.iloc[0:0].copy()
    # preserve column order as in input
    return out, spans


def main():
    data_dir = os.path.join(BASE_DIR, 'data')
    path_3 = os.path.join(data_dir, 'A0003_merged.csv')
    path_8 = os.path.join(data_dir, 'A0008_merged.csv')
    path_dir = os.path.join(data_dir, 'direction.csv')

    df1 = pd.read_csv(path_3)
    df2 = pd.read_csv(path_8)
    dfd = pd.read_csv(path_dir)

    # Validate columns in direction
    reqd = {'vehicle_id', 'date', 'seg_id', 'road_id', 'direction'}
    miss = [c for c in reqd if c not in dfd.columns]
    if miss:
        raise ValueError(f"direction.csv missing columns: {miss}")

    # Road centers
    c3_lat, c3_lon = CENTERS['A0003']
    c8_lat, c8_lon = CENTERS['A0008']

    out1, spans1 = refine_one_source(df1, dfd, 'A0003', c3_lat, c3_lon)
    out2, spans2 = refine_one_source(df2, dfd, 'A0008', c8_lat, c8_lon)

    # Write refined outputs
    out1_path = os.path.join(data_dir, 'A0003_refined.csv')
    out2_path = os.path.join(data_dir, 'A0008_refined.csv')
    os.makedirs(data_dir, exist_ok=True)
    out1.to_csv(out1_path, index=False)
    out2.to_csv(out2_path, index=False)

    spans = spans1 + spans2
    spans_df = pd.DataFrame(spans)
    spans_path = os.path.join(data_dir, 'refined_spans.csv')
    spans_df.to_csv(spans_path, index=False)

    print(f"Refine done. Wrote {len(out1)} rows to {out1_path}, {len(out2)} rows to {out2_path}")
    print(f"Span report: {spans_path} ({len(spans)} segments)")

# Module is imported by main.py; no standalone execution


