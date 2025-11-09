#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate intersection crossover segments for straight and left-turn directions and write:
  - data/cross_over.csv            (points with same schema/order as <road>_refined.csv)
  - data/cross_over_time.csv       (intervals: road_id,vehicle_id,date,seg_id,direction,enter_time,exit_time,enter_idx,exit_idx,duration_s,num_points)
"""
import json
import math
import os
import sys
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd

# Resolve repository root (this file in repo root)
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

STOPLINE_FILE = os.path.join(DATA_DIR, 'stopline.json')
DIRECTION_FILE = os.path.join(DATA_DIR, 'direction.csv')
INTERSECTION_FILE = os.path.join(DATA_DIR, 'intersection_A.json')


def load_centers() -> Dict[str, Tuple[float, float]]:
    """Load CENTERS from config.py (lat, lon)."""
    centers = {}
    try:
        sys.path.insert(0, REPO_ROOT)
        import config  # type: ignore
        c = getattr(config, 'CENTERS', {}) or {}
        for k, v in c.items():
            try:
                lat, lon = float(v[0]), float(v[1])
                centers[str(k)] = (lat, lon)
            except Exception:
                continue
    except Exception as e:
        print(f"[WARN] Failed to load centers from config.py: {e}")
    finally:
        try:
            sys.path.remove(REPO_ROOT)
        except ValueError:
            pass
    return centers


def parse_directions(directions_str: str | None) -> List[str]:
    if not directions_str:
        return []
    out = []
    for part in str(directions_str).split(','):
        v = part.strip()
        if v:
            out.append(v)
    return out


def lonlat_to_xy(lon: float, lat: float, cos_lat: float) -> Tuple[float, float]:
    return (float(lon) * cos_lat, float(lat))


def unit_normal(ax: float, ay: float, bx: float, by: float) -> Tuple[float, float]:
    vx, vy = (bx - ax), (by - ay)
    nx, ny = (vy, -vx)
    norm = math.hypot(nx, ny)
    if norm <= 0:
        return (0.0, 0.0)
    return (nx / norm, ny / norm)


def between_band(val: float, a: float, b: float) -> bool:
    lo, hi = (a, b) if a <= b else (b, a)
    return (val >= lo) and (val <= hi)


def _load_refined_df(road_id: str, date: Optional[str]) -> pd.DataFrame:
    refined_path = os.path.join(DATA_DIR, f'{road_id}_refined.csv')
    if not os.path.exists(refined_path):
        raise FileNotFoundError(f'Refined CSV not found: {refined_path}')
    df = pd.read_csv(refined_path)
    if 'seg_id' not in df.columns:
        raise ValueError('Required column seg_id not found in refined CSV')
    if date:
        df = df[df['date'] == date]
    return df


def _filter_segments_by_direction(df: pd.DataFrame, road_id: str, directions: List[str], date: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    """Filter trajectory df to only rows (veh,date,seg) listed in direction.csv for the requested directions."""
    if df is None or df.empty:
        return df, pd.DataFrame(), set()
    dir_df = pd.read_csv(DIRECTION_FILE)
    dir_filt = (dir_df['road_id'] == road_id)
    if directions:
        up_dirs = [d.upper() for d in directions]
        dir_filt &= dir_df['direction'].astype(str).str.upper().isin(up_dirs)
    if date:
        dir_filt &= (dir_df['date'] == date)
    dir_df = dir_df[dir_filt].copy()
    if dir_df.empty:
        return df.iloc[0:0], dir_df, set()
    valid_triplets = set((int(r['vehicle_id']), str(r['date']), int(r['seg_id']), str(r['direction'])) for _, r in dir_df.iterrows())
    base_keys = {(v, d, s) for (v, d, s, _) in valid_triplets}
    df2 = df[df.apply(lambda r: (int(r['vehicle_id']), str(r['date']), int(r['seg_id'])) in base_keys, axis=1)]
    return df2, dir_df, valid_triplets


def _get_stopline_segment(stop_all: Dict[str, Any], road_id: str, key: str) -> Optional[List[List[float]]]:
    one = stop_all.get(road_id, {}) or {}
    obj = one.get(key) or {}
    seg = obj.get('stopline_segment')
    if isinstance(seg, list) and len(seg) >= 2:
        return seg
    return None


def _ensure_projection_center(road_id: str,
                              seg1: List[List[float]],
                              seg2: List[List[float]],
                              centers: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float]:
    """Return (lat_center, lon_center, cos_lat)."""
    lat_center, lon_center = centers.get(road_id, (None, None))
    if lat_center is None or lon_center is None:
        # Average the two segments' endpoints as fallback
        lats = [float(seg1[0][1]), float(seg1[1][1]), float(seg2[0][1]), float(seg2[1][1])]
        lons = [float(seg1[0][0]), float(seg1[1][0]), float(seg2[0][0]), float(seg2[1][0])]
        lat_center = sum(lats) / 4.0
        lon_center = sum(lons) / 4.0
    cos_lat = math.cos(float(lat_center) * math.pi / 180.0) or 1.0
    return float(lat_center), float(lon_center), float(cos_lat)


def _signed_distance_to_line(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    nx, ny = unit_normal(ax, ay, bx, by)
    return (px - ax) * nx + (py - ay) * ny


def _load_lane_divider(road_id: str) -> Optional[List[List[float]]]:
    try:
        with open(INTERSECTION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        one = data.get(str(road_id), {}) or {}
        lane_div = one.get('lane_divider')
        if isinstance(lane_div, list) and len(lane_div) >= 2:
            return lane_div
    except Exception as e:
        print(f'[WARN] Failed to load lane_divider for {road_id} from {INTERSECTION_FILE}: {e}')
    return None


def _build_centerline_projector(lane_divider: List[List[float]], cos_lat: float) -> Dict[str, Any]:
    """
    Build centerline projector with precomputed XY (deg-scaled) and cumulative arclength in meters.
    """
    xs: List[float] = []
    ys: List[float] = []
    for lon, lat in lane_divider:
        x, y = lonlat_to_xy(float(lon), float(lat), cos_lat)
        xs.append(x)
        ys.append(y)
    # cumulative arclength in meters (degree metric: deg_y * 111_320; x already scaled by cos_lat)
    M_PER_DEG_LAT = 111_320.0
    s_m: List[float] = [0.0]
    seg_len_m: List[float] = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        seg_len = math.hypot(dx, dy) * M_PER_DEG_LAT
        seg_len_m.append(seg_len)
        s_m.append(s_m[-1] + seg_len)
    return {
        'xs': xs,
        'ys': ys,
        's_m': s_m,
        'seg_len_m': seg_len_m,
        'M_PER_DEG_LAT': M_PER_DEG_LAT
    }


def _project_point_to_centerline_s(lon: float, lat: float, cos_lat: float, proj: Dict[str, Any]) -> Tuple[float, int, float]:
    """
    Project a point to centerline and return (s_m, seg_index, t_on_segment in [0,1]).
    """
    px, py = lonlat_to_xy(float(lon), float(lat), cos_lat)
    xs = proj['xs']
    ys = proj['ys']
    s_m = proj['s_m']
    seg_len_m = proj['seg_len_m']
    best_dd = float('inf')
    best_i = 0
    best_t = 0.0
    for i in range(len(xs) - 1):
        ax, ay = xs[i], ys[i]
        bx, by = xs[i + 1], ys[i + 1]
        vx, vy = (bx - ax), (by - ay)
        denom = vx * vx + vy * vy
        if denom <= 0:
            continue
        t = ((px - ax) * vx + (py - ay) * vy) / denom
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        qx = ax + t * vx
        qy = ay + t * vy
        dd = (px - qx) * (px - qx) + (py - qy) * (py - qy)
        if dd < best_dd:
            best_dd = dd
            best_i = i
            best_t = t
    s_here = s_m[best_i] + best_t * seg_len_m[best_i] if best_i < len(seg_len_m) else s_m[-1]
    return s_here, best_i, best_t


def _find_centerline_crossings_s(ax: float, ay: float, bx: float, by: float, proj: Dict[str, Any]) -> List[float]:
    """
    Find all s_m positions where the centerline crosses the given line (ax,ay)-(bx,by).
    """
    xs = proj['xs']
    ys = proj['ys']
    s_m = proj['s_m']
    seg_len_m = proj['seg_len_m']
    out: List[float] = []
    for i in range(len(xs) - 1):
        d0 = _signed_distance_to_line(ax, ay, bx, by, xs[i], ys[i])
        d1 = _signed_distance_to_line(ax, ay, bx, by, xs[i + 1], ys[i + 1])
        if d0 == 0.0:
            out.append(float(s_m[i]))
        elif d0 * d1 < 0.0:
            # Crossing within segment
            t = d0 / (d0 - d1)  # fraction along i->i+1 where distance crosses zero
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            out.append(float(s_m[i] + t * seg_len_m[i]))
        elif d1 == 0.0:
            out.append(float(s_m[i + 1]))
    # Deduplicate near-equal s values
    out_sorted = sorted(out)
    dedup: List[float] = []
    EPS = 1e-6
    for v in out_sorted:
        if not dedup or abs(dedup[-1] - v) > EPS:
            dedup.append(v)
    return dedup


def _choose_boundary_pair_s(s_w_list: List[float], s_e_list: List[float], proj: Dict[str, Any], lon_center: float, lat_center: float, cos_lat: float) -> Tuple[float, float]:
    """
    Choose one s for west and one s for east crossings, preferring those nearest to the center projection.
    """
    s_center, _, _ = _project_point_to_centerline_s(lon_center, lat_center, cos_lat, proj)
    def pick_nearest(cands: List[float]) -> Optional[float]:
        if not cands:
            return None
        return sorted(cands, key=lambda v: abs(v - s_center))[0]
    s_w = pick_nearest(s_w_list)
    s_e = pick_nearest(s_e_list)
    if s_w is None and s_e is None:
        # Fallback: take min and max centerline s as a degenerate band
        return float(proj['s_m'][0]), float(proj['s_m'][-1])
    if s_w is None:
        # Fallback: choose closest centerline node for west
        s_w = float(proj['s_m'][0])
    if s_e is None:
        s_e = float(proj['s_m'][-1])
    lo, hi = (s_w, s_e) if s_w <= s_e else (s_e, s_w)
    return float(lo), float(hi)


def _centerline_lonlat_at_s(proj: Dict[str, Any], s_target: float, cos_lat: float) -> Tuple[float, float]:
    """
    Return (lon, lat) on the centerline at arclength s_target (meters).
    Clamps to [s_min, s_max] if outside.
    """
    xs = proj['xs']
    ys = proj['ys']
    s_m = proj['s_m']
    seg_len_m = proj['seg_len_m']
    if not xs or len(xs) < 2:
        return (0.0, 0.0)
    if s_target <= s_m[0]:
        x, y = xs[0], ys[0]
        return (x / cos_lat, y)
    if s_target >= s_m[-1]:
        x, y = xs[-1], ys[-1]
        return (x / cos_lat, y)
    # Find segment
    lo = 0
    hi = len(s_m) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if s_m[mid] <= s_target:
            lo = mid
        else:
            hi = mid
    i = max(0, min(lo, len(xs) - 2))
    denom = seg_len_m[i] if i < len(seg_len_m) else 0.0
    if denom <= 0:
        x, y = xs[i], ys[i]
        return (x / cos_lat, y)
    t = (s_target - s_m[i]) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    x = xs[i] + t * (xs[i + 1] - xs[i])
    y = ys[i] + t * (ys[i + 1] - ys[i])
    return (x / cos_lat, y)


def _speed_avg_ms(row_a: pd.Series, row_b: pd.Series) -> Optional[float]:
    """
    Average speed between two points in m/s using km/h speed column if available.
    Fallback: None if speeds invalid.
    """
    try:
        va = float(row_a.get('speed', float('nan')))
    except Exception:
        va = float('nan')
    try:
        vb = float(row_b.get('speed', float('nan')))
    except Exception:
        vb = float('nan')
    if (va != va) or (vb != vb):
        return None
    v_kmh = max(0.0, (va + vb) / 2.0)
    return v_kmh / 3.6


def _interp_time_by_distance(t0: pd.Timestamp, t1: pd.Timestamp, dist0_m: float, dist1_m: float, dist_target_m: float) -> Optional[pd.Timestamp]:
    """
    If speed invalid, fall back to linear interpolation by timestamps between two points based on distance fraction.
    """
    if dist1_m <= dist0_m:
        return None
    frac = (dist_target_m - dist0_m) / (dist1_m - dist0_m)
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    dt = (t1 - t0).total_seconds()
    return t0 + pd.to_timedelta(frac * dt, unit='s')


def _synthesize_row_between(row_out: pd.Series,
                            row_in: pd.Series,
                            t_frac: float,
                            time_real: pd.Timestamp) -> Dict[str, Any]:
    """
    Synthesize a boundary row by interpolating lon/lat and time between row_out and row_in.
    """
    out = dict(row_out.to_dict())
    try:
        lon0 = float(row_out['longitude'])
        lat0 = float(row_out['latitude'])
        lon1 = float(row_in['longitude'])
        lat1 = float(row_in['latitude'])
    except Exception:
        lon0 = float(row_out.get('longitude', 0.0))
        lat0 = float(row_out.get('latitude', 0.0))
        lon1 = float(row_in.get('longitude', 0.0))
        lat1 = float(row_in.get('latitude', 0.0))
    if t_frac < 0.0:
        t_frac = 0.0
    elif t_frac > 1.0:
        t_frac = 1.0
    out['longitude'] = lon0 + (lon1 - lon0) * t_frac
    out['latitude'] = lat0 + (lat1 - lat0) * t_frac
    # time fields
    out['time_stamp'] = str(time_real)
    # collectiontime interpolation if available
    try:
        c0 = float(row_out.get('collectiontime', float('nan')))
        c1 = float(row_in.get('collectiontime', float('nan')))
        if c0 == c0 and c1 == c1:
            out['collectiontime'] = int(round(c0 + (c1 - c0) * t_frac))
    except Exception:
        pass
    # speed interpolation/averaging
    try:
        sp0 = float(row_out.get('speed', float('nan')))
        sp1 = float(row_in.get('speed', float('nan')))
        if sp0 == sp0 and sp1 == sp1:
            out['speed'] = (sp0 + sp1) / 2.0
    except Exception:
        pass
    return out

def _synthesize_row_on_centerline(template_row: pd.Series,
                                  lon: float,
                                  lat: float,
                                  time_real: pd.Timestamp,
                                  t_frac_for_collectiontime: Optional[float] = None) -> Dict[str, Any]:
    """
    Create a new row using template_row for non-geom fields but placing the point at provided centerline lon/lat.
    Optionally interpolate collectiontime using t_frac_for_collectiontime in [0,1] if provided.
    """
    out = dict(template_row.to_dict())
    out['longitude'] = float(lon)
    out['latitude'] = float(lat)
    out['time_stamp'] = str(time_real)
    if t_frac_for_collectiontime is not None:
        try:
            c0 = float(template_row.get('collectiontime', float('nan')))
            # Use same for both endpoints if unavailable; downstream code may overwrite
            if c0 == c0:
                out['collectiontime'] = int(round(c0 + 0.0 * t_frac_for_collectiontime))
        except Exception:
            pass
    return out


def compute_a1_real_outputs(road_id: str,
                            directions: List[str],
                            date: Optional[str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Compute A1 real entry/exit times and synthesized boundary points using centerline lane_divider
    and average speeds between bracketing samples.
    Returns (points_df_for_A1, intervals_real_for_A1).
    """
    df = _load_refined_df(road_id, date)
    if df.empty:
        return df.iloc[0:0], []
    df, dir_df, valid_triplets = _filter_segments_by_direction(df, road_id, directions, date)
    if df.empty or dir_df.empty:
        return df.iloc[0:0], []

    # Stoplines and projection setup
    with open(STOPLINE_FILE, 'r', encoding='utf-8') as f:
        stop_all = json.load(f)
    seg_west = _get_stopline_segment(stop_all, road_id, 'west')
    seg_east = _get_stopline_segment(stop_all, road_id, 'east')
    if not (seg_west and seg_east):
        raise ValueError(f'Stoplines not found for {road_id}: west/east')
    (w1_lon, w1_lat), (w2_lon, w2_lat) = seg_west[0], seg_west[1]
    (e1_lon, e1_lat), (e2_lon, e2_lat) = seg_east[0], seg_east[1]

    centers = load_centers()
    lat_center, lon_center, cos_lat = _ensure_projection_center(road_id, seg_west, seg_east, centers)

    # Build centerline projector
    lane_div = _load_lane_divider(road_id)
    if not lane_div:
        print(f'[WARN] No lane_divider for {road_id}, skip A1 real outputs.')
        return df.iloc[0:0], []
    proj = _build_centerline_projector(lane_div, cos_lat)

    # Compute centerline crossing s for west/east
    wx1, wy1 = lonlat_to_xy(w1_lon, w1_lat, cos_lat)
    wx2, wy2 = lonlat_to_xy(w2_lon, w2_lat, cos_lat)
    ex1, ey1 = lonlat_to_xy(e1_lon, e1_lat, cos_lat)
    ex2, ey2 = lonlat_to_xy(e2_lon, e2_lat, cos_lat)
    s_w_all = _find_centerline_crossings_s(wx1, wy1, wx2, wy2, proj)
    s_e_all = _find_centerline_crossings_s(ex1, ey1, ex2, ey2, proj)
    s_lo, s_hi = _choose_boundary_pair_s(s_w_all, s_e_all, proj, lon_center, lat_center, cos_lat)

    # Helper: inside by centerline s
    def inside_by_s(sv: float) -> bool:
        lo, hi = (s_lo, s_hi) if s_lo <= s_hi else (s_hi, s_lo)
        return (sv >= lo) and (sv <= hi)

    # Prepare mapping for direction text
    key_to_dir = {}
    for _, r in dir_df.iterrows():
        key_to_dir[(int(r['vehicle_id']), str(r['date']), int(r['seg_id']))] = str(r['direction'])

    df = df.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime'])
    points_real_parts: List[pd.DataFrame] = []
    intervals_real: List[Dict[str, Any]] = []

    for (veh, d, s), seg_df in df.groupby(['vehicle_id', 'date', 'seg_id']):
        seg_df = seg_df.reset_index(drop=True)
        if seg_df.empty or len(seg_df) < 2:
            continue
        # Precompute s along centerline for all points
        s_list: List[float] = []
        for _, row in seg_df.iterrows():
            s_val, _, _ = _project_point_to_centerline_s(float(row['longitude']), float(row['latitude']), cos_lat, proj)
            s_list.append(s_val)
        inside_flags = [inside_by_s(v) for v in s_list]

        # Find bands (outside->inside -> ... -> outside)
        bands: List[Tuple[int, int]] = []
        inside = False
        enter_idx: Optional[int] = None
        for i in range(1, len(inside_flags)):
            prev_in = inside_flags[i - 1]
            cur_in = inside_flags[i]
            if (not prev_in) and cur_in:
                enter_idx = i - 1
                inside = True
            elif prev_in and (not cur_in) and inside:
                exit_idx = i
                bands.append((int(enter_idx), int(exit_idx)))
                inside = False
                enter_idx = None

        def add_interval_and_points(a: int, b: int, is_fallback_pair: bool) -> None:
            nonlocal points_real_parts, intervals_real
            a = max(0, int(a))
            b = min(len(seg_df) - 1, int(b))
            if a >= b:
                return
            # Bracketing pairs
            enter_out = a
            enter_in = min(a + 1, b)
            exit_in = max(b - 1, a)
            exit_out = b
            # Determine entry/exit boundary s values that lie between the bracketing s's
            s_out_e = s_list[enter_out]
            s_in_e = s_list[enter_in]
            s_out_x = s_list[exit_out]
            s_in_x = s_list[exit_in]

            def pick_boundary_between(s0: float, s1: float) -> float:
                lo, hi = (s0, s1) if s0 <= s1 else (s1, s0)
                if s_lo >= lo and s_lo <= hi:
                    return s_lo
                if s_hi >= lo and s_hi <= hi:
                    return s_hi
                # Fallback: choose closer boundary to s1
                return s_lo if abs(s1 - s_lo) < abs(s1 - s_hi) else s_hi

            if is_fallback_pair:
                # Both points outside: use (a,b) pair for both entry/exit boundaries
                s_entry = pick_boundary_between(s_list[a], s_list[b])
                s_exit = pick_boundary_between(s_list[a], s_list[b])
            else:
                s_entry = pick_boundary_between(s_out_e, s_in_e)
                s_exit = pick_boundary_between(s_in_x, s_out_x)  # use last-inside to first-outside segment

            # Times and synthesized endpoints
            row_a = seg_df.iloc[enter_out]
            row_a1 = seg_df.iloc[enter_in]
            row_bm1 = seg_df.iloc[exit_in]
            row_b = seg_df.iloc[exit_out]

            # Parse timestamps
            t_a = pd.to_datetime(str(row_a['time_stamp']))
            t_a1 = pd.to_datetime(str(row_a1['time_stamp']))
            t_bm1 = pd.to_datetime(str(row_bm1['time_stamp']))
            t_b = pd.to_datetime(str(row_b['time_stamp']))

            # New rule: if there are inside-band points and the entering outside point has a valid end_time,
            # do NOT project; use end_time as enter_time and exit_out time as exit_time. Points = interior only.
            if not is_fallback_pair:
                et_raw_nr = row_a.get('end_time') if 'end_time' in row_a else None
                if et_raw_nr is not None and not pd.isna(et_raw_nr):
                    et_str_nr = str(et_raw_nr).strip()
                    if et_str_nr and et_str_nr.lower() not in ('nan', 'nat'):
                        et_dt_nr = pd.to_datetime(et_str_nr, errors='coerce')
                        if et_dt_nr is not None and not pd.isna(et_dt_nr):
                            enter_time_real = et_dt_nr
                            exit_time_real = t_b
                            # Build points_df for this band: interior inside points only
                            interior = []
                            if (enter_in <= exit_in) and (exit_in - enter_in >= 0):
                                interior = seg_df.iloc[enter_in:exit_in + 1].copy()
                            if isinstance(interior, pd.DataFrame) and not interior.empty:
                                band_df_real = interior[seg_df.columns.tolist()]
                                points_real_parts.append(band_df_real)
                            duration_s = int(max(0, (exit_time_real - enter_time_real).total_seconds()))
                            intervals_real.append({
                                'road_id': road_id,
                                'vehicle_id': int(veh),
                                'date': str(d),
                                'seg_id': int(s),
                                'direction': str(key_to_dir.get((int(veh), str(d), int(s)), '')),
                                'enter_time': str(enter_time_real),
                                'exit_time': str(exit_time_real),
                                'enter_idx': int(enter_out),
                                'exit_idx': int(exit_out),
                                'duration_s': duration_s,
                                'num_points': int(interior.shape[0]) if isinstance(interior, pd.DataFrame) else 0
                            })
                            return

            # Entry time
            use_stop_as_start = False
            enter_time_real: Optional[pd.Timestamp] = None
            # Robust end_time extraction (ignore NaN/NaT strings)
            try:
                et_raw = row_a.get('end_time') if 'end_time' in row_a else None
            except Exception:
                et_raw = None
            if et_raw is not None and not pd.isna(et_raw):
                et_field = str(et_raw).strip()
                if et_field and et_field.lower() not in ('nan', 'nat'):
                    # Stop-from-rest: use the stop row's end_time as entry time
                    try:
                        enter_time_real = pd.to_datetime(et_field, errors='coerce')
                        if enter_time_real is not None and not pd.isna(enter_time_real):
                            use_stop_as_start = True
                        else:
                            enter_time_real = None
                    except Exception:
                        enter_time_real = None
            if not use_stop_as_start:
                if is_fallback_pair:
                    v_ms_e = _speed_avg_ms(row_a, row_b)
                    s_a = s_list[a]
                    dist_e = abs(s_entry - s_a)
                    if v_ms_e and v_ms_e > 0:
                        dt_e = dist_e / v_ms_e
                        enter_time_real = t_a + pd.to_timedelta(dt_e, unit='s')
                    else:
                        # Timestamp interpolation between a and b
                        dist0, dist1 = (0.0, abs(s_list[b] - s_list[a]))
                        enter_time_real = _interp_time_by_distance(t_a, t_b, dist0, dist1, dist_e)
                else:
                    v_ms_e = _speed_avg_ms(row_a, row_a1)
                    # Distance from outside to boundary along centerline
                    dist_e = abs(s_entry - s_out_e)
                    if v_ms_e and v_ms_e > 0:
                        dt_e = dist_e / v_ms_e
                        enter_time_real = t_a + pd.to_timedelta(dt_e, unit='s')
                    else:
                        # Fallback: interpolate by timestamps
                        dist0, dist1 = (0.0, abs(s_in_e - s_out_e))
                        enter_time_real = _interp_time_by_distance(t_a, t_a1, dist0, dist1, dist_e)

            # Exit time
            if is_fallback_pair:
                v_ms_x = _speed_avg_ms(row_a, row_b)
                s_a = s_list[a]
                dist_x = abs(s_exit - s_a)
                if v_ms_x and v_ms_x > 0:
                    dt_x = dist_x / v_ms_x
                    # If starting from a stop, base from end_time (enter_time_real)
                    base_t = enter_time_real if use_stop_as_start and (enter_time_real is not None) else t_a
                    exit_time_real = base_t + pd.to_timedelta(dt_x, unit='s')
                else:
                    dist0, dist1 = (0.0, abs(s_list[b] - s_list[a]))
                    # Interpolate between t_a and t_b; if stop-start, shift by (enter_time_real - t_a)
                    exit_time_tmp = _interp_time_by_distance(t_a, t_b, dist0, dist1, dist_x)
                    if use_stop_as_start and (enter_time_real is not None) and (exit_time_tmp is not None):
                        shift = (enter_time_real - t_a)
                        exit_time_real = exit_time_tmp + shift
                    else:
                        exit_time_real = exit_time_tmp
            else:
                v_ms_x = _speed_avg_ms(row_bm1, row_b)
                dist_x = abs(s_exit - s_in_x)
                if v_ms_x and v_ms_x > 0:
                    dt_x = dist_x / v_ms_x
                    exit_time_real = t_bm1 + pd.to_timedelta(dt_x, unit='s')
                else:
                    dist0, dist1 = (0.0, abs(s_out_x - s_in_x))
                    exit_time_real = _interp_time_by_distance(t_bm1, t_b, dist0, dist1, dist_x)

            if (enter_time_real is None) or (exit_time_real is None) or pd.isna(enter_time_real) or pd.isna(exit_time_real):
                return

            # Synthesize boundary rows on centerline at exact stopline intersections
            if use_stop_as_start:
                start_row_dict = dict(row_a.to_dict())
                start_row_dict['time_stamp'] = str(enter_time_real)
            else:
                lon_e, lat_e = _centerline_lonlat_at_s(proj, s_entry, cos_lat)
                # Choose template for non-geom fields
                start_row_dict = _synthesize_row_on_centerline(row_a, lon_e, lat_e, enter_time_real)
            lon_x, lat_x = _centerline_lonlat_at_s(proj, s_exit, cos_lat)
            # Use row_bm1 as template for exit-side attributes
            end_row_dict = _synthesize_row_on_centerline(row_bm1, lon_x, lat_x, exit_time_real)

            # Build points_df for this band: start + interior inside points + end
            # Interior points: strictly inside indices (a+1 .. b-1), but ensure order
            interior = []
            if (enter_in <= exit_in) and (exit_in - enter_in >= 1):
                interior = seg_df.iloc[enter_in:exit_in + 1].copy()
                # Adjust first/last if needed: we already synthesized boundaries
                if len(interior) > 0:
                    # Drop the very first and very last original boundary bracketing points
                    if len(interior) >= 1:
                        interior = interior.iloc[0:len(interior)]
            band_rows = [pd.DataFrame([start_row_dict])]
            if isinstance(interior, pd.DataFrame) and not interior.empty:
                band_rows.append(interior)
            band_rows.append(pd.DataFrame([end_row_dict]))
            band_df_real = pd.concat(band_rows, ignore_index=True)
            # Ensure ordering and required columns
            band_df_real = band_df_real[seg_df.columns.tolist()]
            points_real_parts.append(band_df_real)

            duration_s = int(max(0, (exit_time_real - enter_time_real).total_seconds()))
            intervals_real.append({
                'road_id': road_id,
                'vehicle_id': int(veh),
                'date': str(d),
                'seg_id': int(s),
                'direction': str(key_to_dir.get((int(veh), str(d), int(s)), '')),
                'enter_time': str(enter_time_real),
                'exit_time': str(exit_time_real),
                'enter_idx': int(enter_out),
                'exit_idx': int(exit_out),
                'duration_s': duration_s,
                'num_points': int(band_df_real.shape[0])
            })

        if not bands:
            # No inside points: use best segment near the centerline band by distance to [s_lo,s_hi]
            # Select the segment (i,i+1) whose [s_i,s_{i+1}] overlaps [s_lo, s_hi] the most; otherwise nearest.
            best_i = None
            best_overlap = -1.0
            best_near = float('inf')
            for i in range(len(s_list) - 1):
                a0, a1 = s_list[i], s_list[i + 1]
                lo = max(min(a0, a1), min(s_lo, s_hi))
                hi = min(max(a0, a1), max(s_lo, s_hi))
                overlap = max(0.0, hi - lo)
                near = min(abs(a0 - s_lo), abs(a0 - s_hi), abs(a1 - s_lo), abs(a1 - s_hi))
                if overlap > best_overlap or (overlap == best_overlap and near < best_near):
                    best_overlap = overlap
                    best_near = near
                    best_i = i
            if best_i is None:
                continue
            add_interval_and_points(best_i, best_i + 1, True)
        else:
            # Select the band that contains the point closest to center (to be consistent with existing behavior)
            # Find nearest-to-center index
            cx, cy = lonlat_to_xy(lon_center, lat_center, cos_lat)
            best_idx = 0
            best_dd = float('inf')
            for i, row in seg_df.iterrows():
                px, py = lonlat_to_xy(float(row['longitude']), float(row['latitude']), cos_lat)
                dd = (px - cx) * (px - cx) + (py - cy) * (py - cy)
                if dd < best_dd:
                    best_dd = dd
                    best_idx = i
            chosen_band = None
            for a, b in bands:
                if a is not None and b is not None and a <= best_idx <= b:
                    chosen_band = (a, b)
                    break
            selected = [chosen_band] if chosen_band is not None else bands[:1]
            for a, b in selected:
                add_interval_and_points(a, b, False)

    if not intervals_real or not points_real_parts:
        return df.iloc[0:0], []
    out_points = pd.concat(points_real_parts, ignore_index=True)
    out_points = out_points[df.columns.tolist()]
    out_points = out_points.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime'])
    return out_points, intervals_real


def compute_crossover_straight(road_id: str,
                               directions: List[str],
                               date: Optional[str],
                               line_key_lo: str,
                               line_key_hi: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Compute crossover segments for straight movements between two parallel stoplines.
    line_key_lo: base line (e.g., 'west' or 'south'); line_key_hi: opposite line (e.g., 'east' or 'north').
    Returns (points_df, intervals_list).

    Behavior:
      1) Prefer bands (continuous inside-band ranges) and pick the one that contains the point
         closest to the intersection center; if none contains it, fall back to previous ranking.
      2) If no inside-band points exist (sampling jumps across the band), fall back to selecting
         the trajectory segment (i, i+1) whose projection contains the center point (t in [0,1]),
         and use that segment's endpoints as enter/exit times.
    """
    df = _load_refined_df(road_id, date)
    if df.empty:
        return df.iloc[0:0], []

    df, dir_df, valid_triplets = _filter_segments_by_direction(df, road_id, directions, date)
    if df.empty or dir_df.empty:
        return df.iloc[0:0], []

    # Stoplines (two parallels)
    with open(STOPLINE_FILE, 'r', encoding='utf-8') as f:
        stop_all = json.load(f)
    seg_lo = _get_stopline_segment(stop_all, road_id, line_key_lo)
    seg_hi = _get_stopline_segment(stop_all, road_id, line_key_hi)
    if not (seg_lo and seg_hi):
        raise ValueError(f'Stoplines not found for {road_id}: {line_key_lo}/{line_key_hi}')
    (l1_lon, l1_lat), (l2_lon, l2_lat) = seg_lo[0], seg_lo[1]
    (h1_lon, h1_lat), (h2_lon, h2_lat) = seg_hi[0], seg_hi[1]

    # Projection
    centers = load_centers()
    lat_center, lon_center, cos_lat = _ensure_projection_center(road_id, seg_lo, seg_hi, centers)

    ax, ay = lonlat_to_xy(l1_lon, l1_lat, cos_lat)
    bx, by = lonlat_to_xy(l2_lon, l2_lat, cos_lat)
    hx, hy = lonlat_to_xy(h1_lon, h1_lat, cos_lat)

    nx, ny = unit_normal(ax, ay, bx, by)
    if nx == 0.0 and ny == 0.0:
        return df.iloc[0:0], []
    d_hi = (hx - ax) * nx + (hy - ay) * ny
    d_min, d_max = (0.0, d_hi) if d_hi >= 0 else (d_hi, 0.0)

    cx, cy = lonlat_to_xy(lon_center, lat_center, cos_lat)

    def nearest_center_index(seg_df: pd.DataFrame) -> int:
        xs = seg_df['longitude'].astype(float).values
        ys = seg_df['latitude'].astype(float).values
        best_idx = 0
        best = float('inf')
        for i, (lon, lat) in enumerate(zip(xs, ys)):
            px, py = lonlat_to_xy(lon, lat, cos_lat)
            dd = (px - cx) * (px - cx) + (py - cy) * (py - cy)
            if dd < best:
                best = dd
                best_idx = i
        return best_idx

    def best_segment_with_center_projection(seg_df: pd.DataFrame) -> Optional[int]:
        """
        Return index i for which the segment (i, i+1) has the orthogonal projection
        of the center lying inside the segment (t in [0,1]) and minimizes distance
        to the center. If none, return None.
        """
        xs = seg_df['longitude'].astype(float).values
        ys = seg_df['latitude'].astype(float).values
        best_i: Optional[int] = None
        best_dd = float('inf')
        for i in range(len(xs) - 1):
            ax_, ay_ = lonlat_to_xy(xs[i], ys[i], cos_lat)
            bx_, by_ = lonlat_to_xy(xs[i + 1], ys[i + 1], cos_lat)
            vx, vy = (bx_ - ax_), (by_ - ay_)
            denom = vx * vx + vy * vy
            if denom <= 0:
                continue
            t = ((cx - ax_) * vx + (cy - ay_) * vy) / denom
            if t < 0.0 or t > 1.0:
                continue
            px = ax_ + t * vx
            py = ay_ + t * vy
            dd = (px - cx) * (px - cx) + (py - cy) * (py - cy)
            if dd < best_dd:
                best_dd = dd
                best_i = i
        return best_i

    def min_dist2_to_center(points_df: pd.DataFrame) -> float:
        if points_df is None or points_df.empty:
            return float('inf')
        xs = points_df['longitude'].astype(float).values
        ys = points_df['latitude'].astype(float).values
        best = float('inf')
        for lon, lat in zip(xs, ys):
            px, py = lonlat_to_xy(lon, lat, cos_lat)
            dd = (px - cx) * (px - cx) + (py - cy) * (py - cy)
            if dd < best:
                best = dd
        return best

    df = df.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime'])
    collected_rows: List[pd.DataFrame] = []
    intervals: List[Dict[str, Any]] = []
    key_to_dir = {(v, d, s): dd for (v, d, s, dd) in valid_triplets}

    for (veh, d, s), seg_df in df.groupby(['vehicle_id', 'date', 'seg_id']):
        seg_df = seg_df.reset_index(drop=True)
        xs = seg_df['longitude'].astype(float).values
        ys = seg_df['latitude'].astype(float).values
        band_vals = []
        for lon, lat in zip(xs, ys):
            dx, dy = (lon * cos_lat - ax, lat - ay)
            band_vals.append(dx * nx + dy * ny)
        inside_flags = [between_band(v, d_min, d_max) for v in band_vals]

        # Index of point closest to center
        ci = nearest_center_index(seg_df)

        bands = []
        inside = False
        enter_idx = None
        for i in range(1, len(inside_flags)):
            prev_in = inside_flags[i - 1]
            cur_in = inside_flags[i]
            if (not prev_in) and cur_in:
                enter_idx = i - 1
                inside = True
            elif prev_in and (not cur_in) and inside:
                exit_idx = i
                bands.append((enter_idx, exit_idx))
                inside = False
                enter_idx = None

        if not bands:
            # New rule: skip segments with no points inside the band
            continue

        band_infos = []
        for (a, b) in bands:
            a = max(0, a)
            b = min(len(seg_df) - 1, b)
            band_slice = seg_df.iloc[a:b + 1].copy()
            score = min_dist2_to_center(band_slice)
            band_infos.append((score, a, b, band_slice))
        band_infos.sort(key=lambda t: t[0])

        # Prefer the band that contains the nearest-to-center point, if any
        chosen_band = None
        for _, a, b, _ in band_infos:
            if a is not None and b is not None and a <= ci <= b:
                chosen_band = (a, b)
                break

        if chosen_band is not None:
            selected = [(0.0, chosen_band[0], chosen_band[1], seg_df.iloc[chosen_band[0]:chosen_band[1] + 1].copy())]
        else:
            # Fallback to previous top-2 by proximity to center
            selected = band_infos[:2]

        for (_, a, b, band_df) in selected:
            collected_rows.append(band_df)
            # Enter time: use end_time if the entering point is a merged point with non-empty end_time
            enter_row = seg_df.iloc[a]
            enter_time = str(enter_row['time_stamp'])
            if 'end_time' in enter_row and isinstance(enter_row['end_time'], str) and enter_row['end_time'].strip():
                enter_time = str(enter_row['end_time']).strip()
            exit_time = str(seg_df.iloc[b]['time_stamp'])
            try:
                t0 = pd.to_datetime(enter_time)
                t1 = pd.to_datetime(exit_time)
                duration_s = max(0, int((t1 - t0).total_seconds()))
            except Exception:
                duration_s = None
            intervals.append({
                'road_id': road_id,
                'vehicle_id': int(veh),
                'date': str(d),
                'seg_id': int(s),
                'direction': str(key_to_dir.get((veh, d, s), '')),
                'enter_time': enter_time,
                'exit_time': exit_time,
                'enter_idx': int(a),
                'exit_idx': int(b),
                'duration_s': duration_s,
                'num_points': int(b - a + 1)
            })

    if not collected_rows:
        return df.iloc[0:0], []
    out_df = pd.concat(collected_rows, axis=0, ignore_index=True)
    out_df = out_df[df.columns.tolist()]
    out_df = out_df.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime'])
    return out_df, intervals


def compute_crossover_turn(road_id: str,
                           directions: List[str],
                           date: Optional[str],
                           entry_key: str,
                           exit_key: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Compute crossover segments for left-turn movements:
      - Find the point closest to intersection center (center_idx).
      - From center_idx, scan backward to find last point before crossing entry stopline.
      - From center_idx, scan forward to find first point after crossing exit stopline.
      - Collect the slice [enter_idx, exit_idx].
    Returns (points_df, intervals_list).
    """
    df = _load_refined_df(road_id, date)
    if df.empty:
        return df.iloc[0:0], []
    df, dir_df, valid_triplets = _filter_segments_by_direction(df, road_id, directions, date)
    if df.empty or dir_df.empty:
        return df.iloc[0:0], []

    with open(STOPLINE_FILE, 'r', encoding='utf-8') as f:
        stop_all = json.load(f)
    seg_entry = _get_stopline_segment(stop_all, road_id, entry_key)
    seg_exit = _get_stopline_segment(stop_all, road_id, exit_key)
    if not (seg_entry and seg_exit):
        raise ValueError(f'Stoplines not found for {road_id}: entry={entry_key}, exit={exit_key}')

    centers = load_centers()
    lat_center, lon_center, cos_lat = _ensure_projection_center(road_id, seg_entry, seg_exit, centers)
    cx, cy = lonlat_to_xy(lon_center, lat_center, cos_lat)

    # Line params
    e_ax, e_ay = lonlat_to_xy(seg_entry[0][0], seg_entry[0][1], cos_lat)
    e_bx, e_by = lonlat_to_xy(seg_entry[1][0], seg_entry[1][1], cos_lat)
    x_ax, x_ay = lonlat_to_xy(seg_exit[0][0], seg_exit[0][1], cos_lat)
    x_bx, x_by = lonlat_to_xy(seg_exit[1][0], seg_exit[1][1], cos_lat)

    df = df.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime'])
    collected_rows: List[pd.DataFrame] = []
    intervals: List[Dict[str, Any]] = []
    key_to_dir = {(v, d, s): dd for (v, d, s, dd) in valid_triplets}

    def nearest_center_index(seg_df: pd.DataFrame) -> int:
        xs = seg_df['longitude'].astype(float).values
        ys = seg_df['latitude'].astype(float).values
        best_idx = 0
        best = float('inf')
        for i, (lon, lat) in enumerate(zip(xs, ys)):
            px, py = lonlat_to_xy(lon, lat, cos_lat)
            dd = (px - cx) * (px - cx) + (py - cy) * (py - cy)
            if dd < best:
                best = dd
                best_idx = i
        return best_idx

    def signed_distances(seg_df: pd.DataFrame, ax: float, ay: float, bx: float, by: float) -> List[float]:
        xs = seg_df['longitude'].astype(float).values
        ys = seg_df['latitude'].astype(float).values
        dd = []
        for lon, lat in zip(xs, ys):
            px, py = lonlat_to_xy(lon, lat, cos_lat)
            dd.append(_signed_distance_to_line(ax, ay, bx, by, px, py))
        return dd

    EPS = 1e-9
    def find_last_before_crossing(dists: List[float], start_idx: int) -> Optional[int]:
        # scan backward; when sign change between i-1 and i, return i-1
        for i in range(start_idx, 0, -1):
            a, b = dists[i - 1], dists[i]
            if (a == 0.0 and b != 0.0) or (b == 0.0 and a != 0.0) or (a * b < 0):
                return i - 1
            if abs(a) <= EPS and abs(b) <= EPS:
                # both near zero -> keep scanning
                continue
        return None

    def find_first_after_crossing(dists: List[float], start_idx: int) -> Optional[int]:
        # scan forward; when sign change between i and i+1, return i+1
        n = len(dists)
        for i in range(start_idx, n - 1):
            a, b = dists[i], dists[i + 1]
            if (a == 0.0 and b != 0.0) or (b == 0.0 and a != 0.0) or (a * b < 0):
                return i + 1
            if abs(a) <= EPS and abs(b) <= EPS:
                continue
        return None

    for (veh, d, s), seg_df in df.groupby(['vehicle_id', 'date', 'seg_id']):
        seg_df = seg_df.reset_index(drop=True)
        if seg_df.empty or len(seg_df) < 2:
            continue
        ci = nearest_center_index(seg_df)
        entry_d = signed_distances(seg_df, e_ax, e_ay, e_bx, e_by)
        exit_d = signed_distances(seg_df, x_ax, x_ay, x_bx, x_by)
        enter_idx = find_last_before_crossing(entry_d, ci)
        exit_idx = find_first_after_crossing(exit_d, ci)
        if enter_idx is None or exit_idx is None:
            continue
        if not (0 <= enter_idx < exit_idx <= len(seg_df) - 1):
            continue
        # New rule: require at least one point inside the intersection area between entry and exit lines
        has_inside_point = False
        for k in range(enter_idx, exit_idx + 1):
            a = entry_d[k]
            b = exit_d[k]
            if (abs(a) <= EPS) or (abs(b) <= EPS) or (a * b < 0):
                has_inside_point = True
                break
        if not has_inside_point:
            continue
        band_df = seg_df.iloc[enter_idx:exit_idx + 1].copy()
        collected_rows.append(band_df)
        enter_row = seg_df.iloc[enter_idx]
        enter_time = str(enter_row['time_stamp'])
        if 'end_time' in enter_row and isinstance(enter_row['end_time'], str) and enter_row['end_time'].strip():
            enter_time = str(enter_row['end_time']).strip()
        exit_time = str(seg_df.iloc[exit_idx]['time_stamp'])
        try:
            t0 = pd.to_datetime(enter_time)
            t1 = pd.to_datetime(exit_time)
            duration_s = max(0, int((t1 - t0).total_seconds()))
        except Exception:
            duration_s = None
        intervals.append({
            'road_id': road_id,
            'vehicle_id': int(veh),
            'date': str(d),
            'seg_id': int(s),
            'direction': str(key_to_dir.get((veh, d, s), '')),
            'enter_time': enter_time,
            'exit_time': exit_time,
            'enter_idx': int(enter_idx),
            'exit_idx': int(exit_idx),
            'duration_s': duration_s,
            'num_points': int(exit_idx - enter_idx + 1)
        })

    if not collected_rows:
        return df.iloc[0:0], []
    out_df = pd.concat(collected_rows, axis=0, ignore_index=True)
    out_df = out_df[df.columns.tolist()]
    out_df = out_df.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime'])
    return out_df, intervals


def _select_best_pair_for_band(band_vals: List[float], d_min: float, d_max: float) -> Optional[int]:
    """
    For straight movement: select index i (segment i->i+1) whose value range overlaps the band [d_min, d_max]
    the most; fallback to the one nearest to the band if no overlap.
    Returns i or None.
    """
    if not band_vals or len(band_vals) < 2:
        return None
    best_i = None
    best_overlap = -1.0
    best_near = float('inf')
    for i in range(len(band_vals) - 1):
        a0, a1 = band_vals[i], band_vals[i + 1]
        lo = max(min(a0, a1), min(d_min, d_max))
        hi = min(max(a0, a1), max(d_min, d_max))
        overlap = max(0.0, hi - lo)
        near = min(abs(a0 - d_min), abs(a0 - d_max), abs(a1 - d_min), abs(a1 - d_max))
        if overlap > best_overlap or (overlap == best_overlap and near < best_near):
            best_overlap = overlap
            best_near = near
            best_i = i
    return best_i


def compute_crossover_straight_outer(road_id: str,
                                     directions: List[str],
                                     date: Optional[str],
                                     line_key_lo: str,
                                     line_key_hi: str) -> List[Dict[str, Any]]:
    """
    For straight movements, process segments with NO inside-band points.
    Choose the adjacent pair (i,i+1) that best overlaps the band and use:
      - enter_time: use end_time of point i if available, else time_stamp of point i
      - exit_time: time_stamp of point i+1
    Returns list of interval dicts (same schema as cross_over_time.csv).
    """
    df = _load_refined_df(road_id, date)
    if df.empty:
        return []
    df, dir_df, valid_triplets = _filter_segments_by_direction(df, road_id, directions, date)
    if df.empty or dir_df.empty:
        return []

    with open(STOPLINE_FILE, 'r', encoding='utf-8') as f:
        stop_all = json.load(f)
    seg_lo = _get_stopline_segment(stop_all, road_id, line_key_lo)
    seg_hi = _get_stopline_segment(stop_all, road_id, line_key_hi)
    if not (seg_lo and seg_hi):
        raise ValueError(f'Stoplines not found for {road_id}: {line_key_lo}/{line_key_hi}')
    (l1_lon, l1_lat), (l2_lon, l2_lat) = seg_lo[0], seg_lo[1]
    (h1_lon, h1_lat), (h2_lon, h2_lat) = seg_hi[0], seg_hi[1]

    centers = load_centers()
    lat_center, lon_center, cos_lat = _ensure_projection_center(road_id, seg_lo, seg_hi, centers)

    ax, ay = lonlat_to_xy(l1_lon, l1_lat, cos_lat)
    bx, by = lonlat_to_xy(l2_lon, l2_lat, cos_lat)
    hx, hy = lonlat_to_xy(h1_lon, h1_lat, cos_lat)

    nx, ny = unit_normal(ax, ay, bx, by)
    if nx == 0.0 and ny == 0.0:
        return []
    d_hi = (hx - ax) * nx + (hy - ay) * ny
    d_min, d_max = (0.0, d_hi) if d_hi >= 0 else (d_hi, 0.0)

    df = df.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime'])
    key_to_dir = {(int(r['vehicle_id']), str(r['date']), int(r['seg_id'])): str(r['direction']) for _, r in dir_df.iterrows()}
    intervals_outer: List[Dict[str, Any]] = []

    for (veh, d, s), seg_df in df.groupby(['vehicle_id', 'date', 'seg_id']):
        seg_df = seg_df.reset_index(drop=True)
        if seg_df.empty or len(seg_df) < 2:
            continue
        xs = seg_df['longitude'].astype(float).values
        ys = seg_df['latitude'].astype(float).values
        band_vals: List[float] = []
        for lon, lat in zip(xs, ys):
            dx, dy = (lon * cos_lat - ax, lat - ay)
            band_vals.append(dx * nx + dy * ny)
        inside_flags = [between_band(v, d_min, d_max) for v in band_vals]
        # Skip segments that already have inside-band points; this function handles only the "no-inside" case
        if any(inside_flags):
            continue
        best_i = _select_best_pair_for_band(band_vals, d_min, d_max)
        if best_i is None:
            continue
        i = int(best_i)
        j = i + 1
        if not (0 <= i < j <= len(seg_df) - 1):
            continue
        row_i = seg_df.iloc[i]
        row_j = seg_df.iloc[j]
        # Times
        enter_time = str(row_i['time_stamp'])
        try:
            et = row_i.get('end_time') if 'end_time' in row_i else None
            if et is not None and str(et).strip() and str(et).strip().lower() not in ('nan', 'nat'):
                enter_time = str(et).strip()
        except Exception:
            pass
        exit_time = str(row_j['time_stamp'])
        try:
            t0 = pd.to_datetime(enter_time)
            t1 = pd.to_datetime(exit_time)
            duration_s = max(0, int((t1 - t0).total_seconds()))
        except Exception:
            duration_s = None
        intervals_outer.append({
            'road_id': road_id,
            'vehicle_id': int(veh),
            'date': str(d),
            'seg_id': int(s),
            'direction': str(key_to_dir.get((int(veh), str(d), int(s)), '')),
            'enter_time': enter_time,
            'exit_time': exit_time,
            'enter_idx': int(i),
            'exit_idx': int(j),
            'duration_s': duration_s,
            'num_points': 2
        })
    return intervals_outer


def compute_crossover_turn_outer(road_id: str,
                                 directions: List[str],
                                 date: Optional[str],
                                 entry_key: str,
                                 exit_key: str) -> List[Dict[str, Any]]:
    """
    For left-turn movements, process segments with NO inside-area points.
    Choose the adjacent pair (i,i+1) that best indicates crossing from entry to exit area:
      - Prefer pairs whose distance ranges to BOTH lines include zero.
      - Fallback to the pair minimizing total |distance| to both lines.
    Use end_time of earlier point for enter_time if available, else its time_stamp; exit_time from later point.
    """
    df = _load_refined_df(road_id, date)
    if df.empty:
        return []
    df, dir_df, valid_triplets = _filter_segments_by_direction(df, road_id, directions, date)
    if df.empty or dir_df.empty:
        return []

    with open(STOPLINE_FILE, 'r', encoding='utf-8') as f:
        stop_all = json.load(f)
    seg_entry = _get_stopline_segment(stop_all, road_id, entry_key)
    seg_exit = _get_stopline_segment(stop_all, road_id, exit_key)
    if not (seg_entry and seg_exit):
        raise ValueError(f'Stoplines not found for {road_id}: entry={entry_key}, exit={exit_key}')

    centers = load_centers()
    lat_center, lon_center, cos_lat = _ensure_projection_center(road_id, seg_entry, seg_exit, centers)
    cx, cy = lonlat_to_xy(lon_center, lat_center, cos_lat)

    e_ax, e_ay = lonlat_to_xy(seg_entry[0][0], seg_entry[0][1], cos_lat)
    e_bx, e_by = lonlat_to_xy(seg_entry[1][0], seg_entry[1][1], cos_lat)
    x_ax, x_ay = lonlat_to_xy(seg_exit[0][0], seg_exit[0][1], cos_lat)
    x_bx, x_by = lonlat_to_xy(seg_exit[1][0], seg_exit[1][1], cos_lat)

    def signed_distances(seg_df: pd.DataFrame, ax: float, ay: float, bx: float, by: float) -> List[float]:
        xs = seg_df['longitude'].astype(float).values
        ys = seg_df['latitude'].astype(float).values
        dd = []
        for lon, lat in zip(xs, ys):
            px, py = lonlat_to_xy(lon, lat, cos_lat)
            dd.append(_signed_distance_to_line(ax, ay, bx, by, px, py))
        return dd

    def center_score_for_pair(seg_df: pd.DataFrame, i: int) -> float:
        # distance^2 from pair midpoint to center
        try:
            lon0 = float(seg_df.iloc[i]['longitude']); lat0 = float(seg_df.iloc[i]['latitude'])
            lon1 = float(seg_df.iloc[i + 1]['longitude']); lat1 = float(seg_df.iloc[i + 1]['latitude'])
            px, py = lonlat_to_xy((lon0 + lon1) / 2.0, (lat0 + lat1) / 2.0, cos_lat)
            return (px - cx) * (px - cx) + (py - cy) * (py - cy)
        except Exception:
            return float('inf')

    df = df.sort_values(['vehicle_id', 'date', 'seg_id', 'collectiontime'])
    key_to_dir = {(int(r['vehicle_id']), str(r['date']), int(r['seg_id'])): str(r['direction']) for _, r in dir_df.iterrows()}
    intervals_outer: List[Dict[str, Any]] = []

    for (veh, d, s), seg_df in df.groupby(['vehicle_id', 'date', 'seg_id']):
        seg_df = seg_df.reset_index(drop=True)
        if seg_df.empty or len(seg_df) < 2:
            continue
        entry_d = signed_distances(seg_df, e_ax, e_ay, e_bx, e_by)
        exit_d = signed_distances(seg_df, x_ax, x_ay, x_bx, x_by)
        # Detect if there are already inside points (then skip here)
        EPS = 1e-9
        has_inside = False
        for k in range(len(seg_df)):
            a = entry_d[k]; b = exit_d[k]
            if (abs(a) <= EPS) or (abs(b) <= EPS) or (a * b < 0):
                has_inside = True
                break
        if has_inside:
            continue
        # Collect candidate pairs
        good_pairs: List[Tuple[float, int]] = []  # (center_score, i)
        fallback_pairs: List[Tuple[float, float, int]] = []  # (dist_sum, center_score, i)
        for i in range(len(seg_df) - 1):
            e0, e1 = entry_d[i], entry_d[i + 1]
            x0, x1 = exit_d[i], exit_d[i + 1]
            emin, emax = (e0, e1) if e0 <= e1 else (e1, e0)
            xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
            e_in = (emin <= 0.0 <= emax)
            x_in = (xmin <= 0.0 <= xmax)
            cscore = center_score_for_pair(seg_df, i)
            if e_in and x_in:
                good_pairs.append((cscore, i))
            else:
                dist_sum = min(abs(e0), abs(e1)) + min(abs(x0), abs(x1))
                fallback_pairs.append((dist_sum, cscore, i))
        chosen_i: Optional[int] = None
        if good_pairs:
            good_pairs.sort(key=lambda t: t[0])  # nearest to center
            chosen_i = good_pairs[0][1]
        elif fallback_pairs:
            fallback_pairs.sort(key=lambda t: (t[0], t[1]))
            chosen_i = fallback_pairs[0][2]
        else:
            continue
        i = int(chosen_i)
        j = i + 1
        row_i = seg_df.iloc[i]
        row_j = seg_df.iloc[j]
        enter_time = str(row_i['time_stamp'])
        try:
            et = row_i.get('end_time') if 'end_time' in row_i else None
            if et is not None and str(et).strip() and str(et).strip().lower() not in ('nan', 'nat'):
                enter_time = str(et).strip()
        except Exception:
            pass
        exit_time = str(row_j['time_stamp'])
        try:
            t0 = pd.to_datetime(enter_time)
            t1 = pd.to_datetime(exit_time)
            duration_s = max(0, int((t1 - t0).total_seconds()))
        except Exception:
            duration_s = None
        intervals_outer.append({
            'road_id': road_id,
            'vehicle_id': int(veh),
            'date': str(d),
            'seg_id': int(s),
            'direction': str(key_to_dir.get((int(veh), str(d), int(s)), '')),
            'enter_time': enter_time,
            'exit_time': exit_time,
            'enter_idx': int(i),
            'exit_idx': int(j),
            'duration_s': duration_s,
            'num_points': 2
        })
    return intervals_outer


def build_wait_df(road_id: str, directions: List[str], date: str | None, points_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the selected crossover points, output time ranges from time_stamp to end_time
    for all qualifying points within directions, as wait intervals.
    """
    if points_df is None or points_df.empty:
        return points_df.iloc[0:0]
    # Require both time_stamp and end_time to exist and be non-empty
    if 'time_stamp' not in points_df.columns or 'end_time' not in points_df.columns:
        return points_df.iloc[0:0]
    wdf = points_df.copy()
    has_start = wdf['time_stamp'].notna() & wdf['time_stamp'].astype(str).str.strip().ne('')
    has_end = wdf['end_time'].notna() & wdf['end_time'].astype(str).str.strip().ne('')
    wdf = wdf[has_start & has_end].copy()
    if wdf.empty:
        return wdf
    # Attach direction via direction.csv mapping
    try:
        dir_df = pd.read_csv(DIRECTION_FILE)
        mask = (dir_df['road_id'] == road_id)
        if directions:
            up_dirs = [d.upper() for d in directions]
            mask &= dir_df['direction'].astype(str).str.upper().isin(up_dirs)
        if date:
            mask &= (dir_df['date'] == date)
        dir_df = dir_df[mask].copy()
        key_to_dir = {
            (int(r['vehicle_id']), str(r['date']), int(r['seg_id'])): str(r['direction'])
            for _, r in dir_df.iterrows()
        }
        wdf['direction'] = wdf.apply(
            lambda r: key_to_dir.get((int(r['vehicle_id']), str(r['date']), int(r['seg_id'])) , ''),
            axis=1
        )
    except Exception:
        # If direction.csv is unavailable for any reason, still output without direction
        wdf['direction'] = ''
    # Add road_id column and select/reorder desired columns
    wdf['road_id'] = road_id
    desired_cols = ['road_id', 'vehicle_id', 'date', 'seg_id', 'direction', 'time_stamp', 'end_time']
    existing_cols = [c for c in desired_cols if c in wdf.columns]
    return wdf[existing_cols].reset_index(drop=True)


# -------------------- Per-second occupancy (A1/A2/B1/B2) --------------------
DIR4 = ('A1', 'A2', 'B1', 'B2')


def _to_dir4(direction: Any) -> Optional[str]:
    try:
        d = str(direction)
    except Exception:
        return None
    for k in DIR4:
        if d.startswith(k):
            return k
    return None


def build_per_second_occupancy(df: pd.DataFrame,
                               start_col: str,
                               end_col: str,
                               direction_col: str) -> pd.DataFrame:
    """
    Build per-second occupancy counts per (road_id, date) and direction A1/A2/B1/B2.
    Interval is [start, end) i.e., end exclusive.
    Returns columns: road_id, date, time, A1, A2, B1, B2
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['road_id', 'date', 'time', *DIR4])
    work = df.copy()
    # Map to 4-direction groups
    work['dir4'] = work[direction_col].map(_to_dir4)
    work = work.dropna(subset=['dir4']).astype({'dir4': str})
    # Build absolute timestamps
    work['start'] = pd.to_datetime(work['date'].astype(str) + ' ' + work[start_col].astype(str), errors='coerce')
    work['end'] = pd.to_datetime(work['date'].astype(str) + ' ' + work[end_col].astype(str), errors='coerce')
    # Filter valid, non-zero intervals
    work = work[(~work['start'].isna()) & (~work['end'].isna()) & (work['start'] < work['end'])].copy()
    if work.empty:
        return pd.DataFrame(columns=['road_id', 'date', 'time', *DIR4])

    out_frames: List[pd.DataFrame] = []
    for (road_id, date_val), g in work.groupby(['road_id', 'date']):
        # Collect event deltas per direction
        dir_events: Dict[str, Dict[pd.Timestamp, int]] = {d: {} for d in DIR4}
        for _, r in g.iterrows():
            d = r['dir4']
            s = r['start']
            e = r['end']
            dir_events[d][s] = dir_events[d].get(s, 0) + 1
            dir_events[d][e] = dir_events[d].get(e, 0) - 1
        # Determine group time span
        all_keys = []
        for d in DIR4:
            all_keys.extend(list(dir_events[d].keys()))
        if not all_keys:
            continue
        t0 = min(all_keys).floor('s')
        t1 = max(all_keys).ceil('s')
        if t0 >= t1:
            continue
        # Build 1-second index; end exclusive
        idx = pd.date_range(t0, t1 - pd.Timedelta(seconds=1), freq='s')
        data = {}
        for d in DIR4:
            ev = dir_events[d]
            if not ev:
                series = pd.Series(0, index=idx, dtype='int64')
            else:
                evs = pd.Series(ev).sort_index()
                series = evs.cumsum()
                # Align to idx: forward fill current occupancy and fill gaps with 0
                series = series.reindex(idx, method='ffill').fillna(0).astype('int64')
            data[d] = series
        df_group = pd.DataFrame(data, index=idx).reset_index().rename(columns={'index': 'dt'})
        df_group['road_id'] = road_id
        df_group['date'] = date_val
        df_group['time'] = df_group['dt'].dt.strftime('%H:%M:%S')
        out_frames.append(df_group[['road_id', 'date', 'time', *DIR4]])
    if not out_frames:
        return pd.DataFrame(columns=['road_id', 'date', 'time', *DIR4])
    out = pd.concat(out_frames, ignore_index=True)
    # Ensure integer dtype
    for d in DIR4:
        out[d] = out[d].astype('int64')
    return out.sort_values(['road_id', 'date', 'time']).reset_index(drop=True)


def compute_and_write_per_second_counts() -> None:
    """
    Read data/cross_over_time_try.csv and data/wait.csv,
    compute per-second occupancy counts for A1/A2/B1/B2, and write:
      - data/per_second_flow.csv
      - data/per_second_wait.csv
    """
    flow_in = os.path.join(DATA_DIR, 'cross_over_time_try.csv')
    wait_in = os.path.join(DATA_DIR, 'wait.csv')
    flow_out = os.path.join(DATA_DIR, 'per_second_flow.csv')
    wait_out = os.path.join(DATA_DIR, 'per_second_wait.csv')

    # Flow: enter_time -> exit_time
    try:
        if os.path.exists(flow_in):
            fdf = pd.read_csv(flow_in)
            need_cols = {'road_id', 'date', 'direction', 'enter_time', 'exit_time'}
            if need_cols.issubset(set(fdf.columns)):
                flow_ps = build_per_second_occupancy(fdf, 'enter_time', 'exit_time', 'direction')
                if flow_ps is not None and not flow_ps.empty:
                    flow_ps.to_csv(flow_out, index=False)
                    print(f'Wrote {len(flow_ps)} per-second rows to {flow_out}')
                else:
                    pd.DataFrame(columns=['road_id', 'date', 'time', *DIR4]).to_csv(flow_out, index=False)
                    print(f'No per-second flow rows; wrote empty CSV to {flow_out}')
            else:
                print(f'[WARN] Missing columns in {flow_in}; expected {need_cols}')
        else:
            print(f'[WARN] Flow input not found: {flow_in}')
    except Exception as e:
        print(f'[WARN] Failed computing per-second flow: {e}')

    # Wait: time_stamp -> end_time
    try:
        if os.path.exists(wait_in):
            wdf = pd.read_csv(wait_in)
            need_cols = {'road_id', 'date', 'direction', 'time_stamp', 'end_time'}
            if need_cols.issubset(set(wdf.columns)):
                wait_ps = build_per_second_occupancy(wdf, 'time_stamp', 'end_time', 'direction')
                if wait_ps is not None and not wait_ps.empty:
                    wait_ps.to_csv(wait_out, index=False)
                    print(f'Wrote {len(wait_ps)} per-second rows to {wait_out}')
                else:
                    pd.DataFrame(columns=['road_id', 'date', 'time', *DIR4]).to_csv(wait_out, index=False)
                    print(f'No per-second wait rows; wrote empty CSV to {wait_out}')
            else:
                print(f'[WARN] Missing columns in {wait_in}; expected {need_cols}')
        else:
            print(f'[WARN] Wait input not found: {wait_in}')
    except Exception as e:
        print(f'[WARN] Failed computing per-second wait: {e}')


def main():
    # Hardcoded initial configuration (edit here as needed)
    road_ids = ['A0003', 'A0008']
    # Direction groups
    dirs_a1 = ['A1-1', 'A1-2']  # EW straight
    dirs_b1 = ['B1-1', 'B1-2']  # NS straight
    dirs_a2 = ['A2-1', 'A2-2']  # EW left-turn
    dirs_b2 = ['B2-1', 'B2-2']  # NS left-turn
    date = None  # or 'YYYY-MM-DD'
    output_points_unified = os.path.join(DATA_DIR, 'cross_over.csv')
    output_intervals_unified = os.path.join(DATA_DIR, 'cross_over_time.csv')
    output_wait_unified = os.path.join(DATA_DIR, 'wait.csv')
    # A1-specific outputs
    output_points_a1 = os.path.join(DATA_DIR, 'cross_overA.csv')
    output_intervals_a1_real = os.path.join(DATA_DIR, 'corss_timeA_real.csv')

    print(f'Generating crossover for roads {road_ids}, date={date or "ALL"}')
    print(f' - A1 (straight EW) = {dirs_a1}')
    print(f' - B1 (straight NS) = {dirs_b1}')
    print(f' - A2 (left-turn EW) = {dirs_a2}')
    print(f' - B2 (left-turn NS) = {dirs_b2}')

    intervals_all: List[Dict[str, Any]] = []
    points_parts: List[pd.DataFrame] = []
    wait_parts: List[pd.DataFrame] = []
    a1_points_real_parts: List[pd.DataFrame] = []
    a1_intervals_real_parts: List[Dict[str, Any]] = []
    intervals_with_outer_extra: List[Dict[str, Any]] = []

    for road_id in road_ids:
        print(f'Processing road {road_id}...')
        # Compute straight bands
        pts_a1, intervals_a1 = compute_crossover_straight(road_id, dirs_a1, date, 'west', 'east')
        pts_b1, intervals_b1 = compute_crossover_straight(road_id, dirs_b1, date, 'south', 'north')
        # Compute left-turns per mapping
        pts_a2_1, intervals_a2_1 = compute_crossover_turn(road_id, ['A2-1'], date, 'north', 'east')
        pts_a2_2, intervals_a2_2 = compute_crossover_turn(road_id, ['A2-2'], date, 'south', 'west')
        pts_b2_1, intervals_b2_1 = compute_crossover_turn(road_id, ['B2-1'], date, 'west', 'north')
        pts_b2_2, intervals_b2_2 = compute_crossover_turn(road_id, ['B2-2'], date, 'east', 'south')

        # Accumulate intervals
        for part in [intervals_a1, intervals_b1, intervals_a2_1, intervals_a2_2, intervals_b2_1, intervals_b2_2]:
            if part:
                intervals_all.extend(part)

        # Extra: generate outer-fallback intervals for segments with no inside points
        try:
            outer_a1 = compute_crossover_straight_outer(road_id, dirs_a1, date, 'west', 'east')
            outer_b1 = compute_crossover_straight_outer(road_id, dirs_b1, date, 'south', 'north')
            outer_a2_1 = compute_crossover_turn_outer(road_id, ['A2-1'], date, 'north', 'east')
            outer_a2_2 = compute_crossover_turn_outer(road_id, ['A2-2'], date, 'south', 'west')
            outer_b2_1 = compute_crossover_turn_outer(road_id, ['B2-1'], date, 'west', 'north')
            outer_b2_2 = compute_crossover_turn_outer(road_id, ['B2-2'], date, 'east', 'south')
            for part in [outer_a1, outer_b1, outer_a2_1, outer_a2_2, outer_b2_1, outer_b2_2]:
                if part:
                    intervals_with_outer_extra.extend(part)
        except Exception as e:
            print(f'[WARN] Failed to compute outer intervals for {road_id}: {e}')

        # A1 real entry/exit times and synthesized points (lane_divider-based)
        try:
            a1_points_real, a1_intervals_real = compute_a1_real_outputs(road_id, dirs_a1, date)
            if a1_points_real is not None and not a1_points_real.empty:
                if 'road_id' not in a1_points_real.columns:
                    a1_points_real['road_id'] = road_id
                a1_points_real_parts.append(a1_points_real)
            if a1_intervals_real:
                a1_intervals_real_parts.extend(a1_intervals_real)
        except Exception as e:
            print(f'[WARN] Failed to compute A1 real outputs for {road_id}: {e}')

        # Accumulate points per road
        pts_concat_list = [df for df in [pts_a1, pts_b1, pts_a2_1, pts_a2_2, pts_b2_1, pts_b2_2] if df is not None and not df.empty]
        if pts_concat_list:
            pts_all = pd.concat(pts_concat_list, ignore_index=True)
            # Ensure road_id column exists
            if 'road_id' not in pts_all.columns:
                pts_all['road_id'] = road_id
            # Attach direction column by mapping (vehicle_id, date, seg_id) -> direction from direction.csv
            try:
                dir_map = pd.read_csv(DIRECTION_FILE)
                dir_map = dir_map[dir_map['road_id'] == road_id].copy()
                if date:
                    dir_map = dir_map[dir_map['date'] == date]
                dir_map = dir_map[['vehicle_id', 'date', 'seg_id', 'direction']].drop_duplicates()
                pts_all = pts_all.merge(dir_map, on=['vehicle_id', 'date', 'seg_id'], how='left')
            except Exception as e:
                print(f'[WARN] Failed to attach direction to points for {road_id}: {e}')
                if 'direction' not in pts_all.columns:
                    pts_all['direction'] = ''
            points_parts.append(pts_all)

            # Wait intervals for this road (per group)
            for dirs, pts in [
                (dirs_a1, pts_a1),
                (dirs_b1, pts_b1),
                (dirs_a2, pd.concat([p for p in [pts_a2_1, pts_a2_2] if p is not None and not p.empty], ignore_index=True) if any(p is not None and not p.empty for p in [pts_a2_1, pts_a2_2]) else pd.DataFrame()),
                (dirs_b2, pd.concat([p for p in [pts_b2_1, pts_b2_2] if p is not None and not p.empty], ignore_index=True) if any(p is not None and not p.empty for p in [pts_b2_1, pts_b2_2]) else pd.DataFrame()),
            ]:
                if pts is not None and not pts.empty:
                    w = build_wait_df(road_id, dirs, date, pts)
                    if w is not None and not w.empty:
                        # Ensure road_id exists
                        if 'road_id' not in w.columns:
                            w['road_id'] = road_id
                        wait_parts.append(w)

    # Write unified intervals
    if intervals_all:
        df_intervals_all = pd.DataFrame(intervals_all)
        df_intervals_all.to_csv(output_intervals_unified, index=False)
        print(f'Wrote {len(df_intervals_all)} intervals to {output_intervals_unified}')
    else:
        print('No intervals to write to cross_over_time.csv')

    # Write unified points
    if points_parts:
        pts_all = pd.concat(points_parts, ignore_index=True)
        if 'road_id' not in pts_all.columns:
            # Fallback set if somehow missing
            pts_all['road_id'] = road_ids[0]
        pts_all.to_csv(output_points_unified, index=False)
        print(f'Wrote {len(pts_all)} points to {output_points_unified}')
    else:
        print('No points to write to cross_over.csv')

    # Write unified wait.csv
    if wait_parts:
        wait_all = pd.concat(wait_parts, ignore_index=True)
        if 'road_id' not in wait_all.columns:
            wait_all['road_id'] = road_ids[0]
        wait_all.to_csv(output_wait_unified, index=False)
        print(f'Wrote {len(wait_all)} wait intervals to {output_wait_unified}')
    else:
        print('No wait intervals found (no rows with non-empty end_time).')

    # Write cross_over_time_with_outer.csv (union of normal intervals and extra outer ones)
    output_intervals_with_outer = os.path.join(DATA_DIR, 'cross_over_time_with_outer.csv')
    intervals_all_with_outer = list(intervals_all)
    if intervals_with_outer_extra:
        intervals_all_with_outer.extend(intervals_with_outer_extra)
    if intervals_all_with_outer:
        df_with_outer = pd.DataFrame(intervals_all_with_outer)
        df_with_outer.to_csv(output_intervals_with_outer, index=False)
        print(f'Wrote {len(df_with_outer)} intervals to {output_intervals_with_outer}')
    else:
        print('No intervals to write to cross_over_time_with_outer.csv')

    # Write cross_over_time_try.csv (ONLY outer-adjacent intervals)
    output_intervals_try = os.path.join(DATA_DIR, 'cross_over_time_try.csv')
    if intervals_with_outer_extra:
        df_try = pd.DataFrame(intervals_with_outer_extra)
        df_try.to_csv(output_intervals_try, index=False)
        print(f'Wrote {len(df_try)} intervals to {output_intervals_try}')
    else:
        print('No intervals to write to cross_over_time_try.csv')

    # Write A1-specific outputs
    if a1_intervals_real_parts:
        df_a1_real = pd.DataFrame(a1_intervals_real_parts)
        df_a1_real.to_csv(output_intervals_a1_real, index=False)
        print(f'Wrote {len(df_a1_real)} intervals to {output_intervals_a1_real}')
    else:
        print('No A1 real intervals to write.')
    if a1_points_real_parts:
        df_a1_pts = pd.concat(a1_points_real_parts, ignore_index=True)
        if 'road_id' not in df_a1_pts.columns:
            df_a1_pts['road_id'] = road_ids[0]
        df_a1_pts.to_csv(output_points_a1, index=False)
        print(f'Wrote {len(df_a1_pts)} points to {output_points_a1}')
    else:
        print('No A1 real points to write.')
    # Compute per-second flow/wait series based on cross_over_time_try.csv and wait.csv
    compute_and_write_per_second_counts()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


