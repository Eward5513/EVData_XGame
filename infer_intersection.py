#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from config import BASE_DIR, CENTERS
rng = np.random.default_rng(42)

# ---------- Configuration ----------
# Input CSVs to process (produced by split_trajectories)
INPUT_FILES = [
    os.path.join(BASE_DIR, 'data', 'A0003_split.csv'),
    os.path.join(BASE_DIR, 'data', 'A0008_split.csv'),
]
K = 4                        # Number of approach direction clusters
V_STOP = 0.5                 # Stationary speed threshold (m/s)
V_GO = 1.5                   # Start-moving speed threshold (m/s)
LOOKAHEAD_S = 5.0            # Lookahead window for detecting start (s)
RANSAC_THRESH = 1.5          # RANSAC inlier distance threshold (m)
LATERAL_BAND = 2.0           # Lateral band width for stop-line estimation (m)
STOP_QUANTILE = 0.15         # Quantile for conservative stop-line estimate
WRITE_GEOJSON = False        # If True, also write a GeoJSON per CSV

DATA_DIR = os.path.join(BASE_DIR, 'data')
DIRECTION_FILE = os.path.join(DATA_DIR, 'direction.csv')
AXIS_POLY_BIN_M = 50.0       # Bin size (meters) to build piecewise axis/lane polyline
AXIS_POLY_MIN_POINTS = 5     # Min points per bin to keep a vertex

# Direction-grouped labels per inbound side (W,E,S,N)
LANE_GROUPS = {
    'W': {
        'straight': ['A1-1', 'A1-2'],
        'left': ['B2-1', 'A3-2'],
        'right': ['A2-2', 'B3-2'],
        'stop_labels': ['A1-1', 'B2-1', 'B3-2']
    },
    'E': {
        'straight': ['A1-1', 'A1-2'],
        'left': ['B2-2', 'A3-1'],
        'right': ['A2-1', 'B3-1'],
        'stop_labels': ['A1-2', 'B2-2', 'B3-1']
    },
    'S': {
        'straight': ['B1-1', 'B1-2'],
        'left': ['A2-2', 'B3-2'],
        'right': ['B2-2', 'A3-1'],
        'stop_labels': ['B1-1', 'A2-2', 'A3-1']
    },
    'N': {
        'straight': ['B1-1', 'B1-2'],
        'left': ['A2-1', 'B3-1'],
        'right': ['B2-1', 'A3-2'],
        'stop_labels': ['B1-2', 'A2-1', 'A3-2']
    }
}

# ---------- Utilities ----------

def lonlat_to_xy(lon, lat, lon0, lat0):
    """
    Approximate local tangent-plane projection (meters) centered at (lon0, lat0).
    """
    # Earth radius (approximate)
    R = 6378137.0
    lat0_rad = np.deg2rad(lat0)
    mx = (np.deg2rad(lon - lon0) * R * np.cos(lat0_rad))
    my = (np.deg2rad(lat - lat0) * R)
    return mx, my

def xy_to_lonlat(x, y, lon0, lat0):
    R = 6378137.0
    lat0_rad = np.deg2rad(lat0)
    lon = lon0 + np.rad2deg(x / (R * np.cos(lat0_rad)))
    lat = lat0 + np.rad2deg(y / R)
    return lon, lat

def angle_wrap_deg(a):
    """Wrap angle into [-180, 180)."""
    a = (a + 180.0) % 360.0 - 180.0
    return a

def heading_from_xy(dx, dy):
    """Heading with east=0°, counter-clockwise positive: atan2(dy, dx)."""
    return (np.rad2deg(np.arctan2(dy, dx)) + 360.0) % 360.0

def robust_quantile(values, q, max_iters=2):
    """Quantile after IQR-based outlier clipping (robust to extremes)."""
    v = np.asarray(values).copy()
    if len(v) == 0:
        return np.nan
    for _ in range(max_iters):
        q1, q3 = np.quantile(v, [0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        v = v[(v >= lo) & (v <= hi)]
        if len(v) == 0:
            break
    if len(v) == 0:
        return np.nan
    return float(np.quantile(v, q))

# ---------- Line representation and fitting ----------

@dataclass
class Line:
    n: np.ndarray   # 法向单位向量 shape (2,)
    rho: float      # 截距, n^T x = rho
    t: np.ndarray   # 切向单位向量 (与 n 垂直)
    angle_deg: float  # 切向角度（以东为0逆时针）

def pca_tls_direction(points: np.ndarray, weights: np.ndarray=None) -> np.ndarray:
    """
    TLS principal direction (tangent) from 2D points (N,2).
    Returns a unit vector t.
    """
    if weights is None:
        w = np.ones(len(points))
    else:
        w = weights
    w = w / (w.sum() + 1e-12)
    mu = (points * w[:, None]).sum(axis=0)
    X = points - mu
    C = (X * w[:, None]).T @ X
    # 主特征向量对应最大特征值
    vals, vecs = np.linalg.eigh(C)
    t = vecs[:, np.argmax(vals)]
    t = t / (np.linalg.norm(t) + 1e-12)
    return t

def line_from_points_tls(points: np.ndarray, weights: np.ndarray=None) -> Line:
    """
    TLS-fit line in normal form: n^T x = rho.
    """
    t = pca_tls_direction(points, weights)
    n = np.array([-t[1], t[0]])
    # Use weighted median of normal projections for rho (robust)
    if weights is None:
        weights = np.ones(len(points))
    proj = points @ n
    rho = robust_quantile(np.repeat(proj, repeats=np.maximum(1, weights.astype(int))), 0.5)
    if np.isnan(rho):
        rho = float(np.median(proj))
    angle_deg = (np.rad2deg(np.arctan2(t[1], t[0])) + 360.0) % 360.0
    return Line(n=n, rho=rho, t=t, angle_deg=angle_deg)

def ransac_line(points: np.ndarray, thresh=1.5, iters=300, min_inliers=6) -> Tuple[Line, np.ndarray]:
    """
    Fit a line with RANSAC, then refine with TLS.
    Returns a Line and a boolean inlier mask.
    """
    N = len(points)
    if N < 2:
        raise ValueError("Not enough points for RANSAC")
    best_inliers = None
    best_count = -1
    idx = np.arange(N)
    for _ in range(iters):
        i, j = rng.choice(idx, size=2, replace=False)
        p1, p2 = points[i], points[j]
        v = p2 - p1
        if np.linalg.norm(v) < 1e-3:
            continue
        t = v / np.linalg.norm(v)
        n = np.array([-t[1], t[0]])
        rho = float(n @ p1)
        d = np.abs(points @ n - rho)
        inliers = d <= thresh
        count = inliers.sum()
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < max(min_inliers, 2):
        # Degenerate case: use TLS on all points
        line = line_from_points_tls(points)
        inliers = np.ones(N, dtype=bool)
        return line, inliers
    # TLS refinement
    line = line_from_points_tls(points[best_inliers])
    return line, best_inliers

def lsq_intersection_center(lines: List[Line], weights: List[float]=None) -> np.ndarray:
    """
    Least-squares intersection point of lines n_j^T x = rho_j.
    """
    if weights is None:
        weights = [1.0] * len(lines)
    A = np.zeros((2,2))
    b = np.zeros(2)
    for L, w in zip(lines, weights):
        n = L.n.reshape(2,1)
        A += w * (n @ n.T)
        b += w * (L.rho * L.n)
    c = np.linalg.solve(A, b)
    return c  # shape (2,)

# ---------- Main pipeline ----------

def compute_heading_per_track(df: pd.DataFrame, lon0: float, lat0: float) -> pd.DataFrame:
    """
    For each vehicle_id, sort by time and compute heading between consecutive points (degrees).
    If 'collectiontime' (milliseconds) exists, prefer it; otherwise parse from 'date' + 'time_stamp'.
    """
    if 'collectiontime' in df.columns and df['collectiontime'].notna().any():
        df['_t'] = pd.to_numeric(df['collectiontime'], errors='coerce')
    else:
        # Attempt to parse from 'date' and 'time_stamp'
        df['_t'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time_stamp'].astype(str), errors='coerce').astype('int64')//10**6
    # Sort by vehicle and time
    df = df.sort_values(['vehicle_id','_t']).copy()
    # Compute heading
    x, y = lonlat_to_xy(df['longitude'].values, df['latitude'].values, lon0, lat0)
    df['_x'] = x
    df['_y'] = y
    df['_x_prev'] = df.groupby('vehicle_id')['_x'].shift(1)
    df['_y_prev'] = df.groupby('vehicle_id')['_y'].shift(1)
    dx = df['_x'] - df['_x_prev']
    dy = df['_y'] - df['_y_prev']
    hd = heading_from_xy(dx.fillna(0).values, dy.fillna(0).values)
    df['heading_est'] = hd
    return df

def extract_last_stop_points(df: pd.DataFrame,
                             v_stop=0.5, v_go=1.5, lookahead_s=5.0) -> pd.DataFrame:
    """
    Extract the last stationary point before starting:
    speed < v_stop and, within lookahead_s, a future speed > v_go.
    """
    # If speed is likely in km/h, attempt conversion based on distribution
    sp = pd.to_numeric(df['speed'], errors='coerce').fillna(0).values
    # Heuristic: if 95th percentile > 80, assume km/h and convert to m/s
    if np.nanpercentile(sp, 95) > 80:
        sp = sp / 3.6
    df['_speed_ms'] = sp

    # Use millisecond timestamps
    tms = pd.to_numeric(df['_t'], errors='coerce').fillna(0).values
    df['_t_ms'] = tms

    # Scan each vehicle independently
    out_rows = []
    for vid, g in df.groupby('vehicle_id', sort=False):
        g = g.sort_values('_t_ms')
        arr_t = g['_t_ms'].values
        arr_v = g['_speed_ms'].values
        arr_x = g['_x'].values
        arr_y = g['_y'].values
        arr_hd = g['heading_est'].values

        N = len(g)
        i = 0
        while i < N:
            if arr_v[i] < v_stop:
                # Look ahead within lookahead_s to see if speed exceeds v_go
                j = i
                moved = False
                while j < N and (arr_t[j] - arr_t[i]) <= lookahead_s * 1000.0:
                    if arr_v[j] > v_go:
                        moved = True
                        break
                    j += 1
                if moved:
                    # Last stationary point before start: within [i, j),
                    # choose the slowest and nearest to j
                    k = np.argmax((arr_v[i:j] < v_stop).astype(int) * np.arange(j - i)) + i
                    out_rows.append({
                        'vehicle_id': vid,
                        '_t_ms': arr_t[k],
                        '_x': arr_x[k], '_y': arr_y[k],
                        'heading': arr_hd[k]
                    })
                    # Jump to after j and continue
                    i = j + 1
                    continue
            i += 1
    return pd.DataFrame(out_rows)

def cluster_directions(points_df: pd.DataFrame, K=4) -> np.ndarray:
    """
    Cluster headings into K approach directions.
    If headings are missing, you could instead use polar angles to a provisional center.
    """
    hd = points_df['heading'].values % 360.0
    # Embed onto the unit circle
    X = np.c_[np.cos(np.deg2rad(hd)), np.sin(np.deg2rad(hd))]
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels  # 0..K-1

def load_direction_df(path: str = DIRECTION_FILE) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[Error] Failed to read direction file '{path}': {exc}")
        return pd.DataFrame()

def _triplet_key(vid, date, seg) -> Tuple[int, str, int]:
    """Normalize (vehicle_id, date, seg_id) to int, str, int tuple."""
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return -1
    return (_to_int(vid), str(date), _to_int(seg))

def build_triplet_set_for_labels(dir_df: pd.DataFrame, road_id: str, labels: List[str]) -> set:
    """Return set of (vehicle_id, date, seg_id) triplets for given road_id and label list."""
    if dir_df is None or len(dir_df) == 0:
        return set()
    sub = dir_df[(dir_df['road_id'] == road_id) & (dir_df['direction'].astype(str).isin(labels))]
    out = set()
    for _, r in sub.iterrows():
        out.add(_triplet_key(r['vehicle_id'], r['date'], r['seg_id']))
    return out

def filter_df_by_triplets(df: pd.DataFrame, triplets: set) -> pd.DataFrame:
    if not triplets:
        return df.iloc[0:0].copy()
    def _keep(row):
        return _triplet_key(row['vehicle_id'], row['date'], row['seg_id']) in triplets
    return df[df.apply(_keep, axis=1)].copy()

def collect_points_for_labels(df_raw: pd.DataFrame, dir_df: pd.DataFrame, road_id: str, labels: List[str]) -> np.ndarray:
    trip = build_triplet_set_for_labels(dir_df, road_id, labels)
    dff = filter_df_by_triplets(df_raw, trip)
    return dff[['_x', '_y']].to_numpy()

def collect_stop_points_for_labels(df_raw: pd.DataFrame, dir_df: pd.DataFrame, road_id: str, labels: List[str]) -> pd.DataFrame:
    trip = build_triplet_set_for_labels(dir_df, road_id, labels)
    dff = filter_df_by_triplets(df_raw, trip)
    return extract_last_stop_points(dff, v_stop=V_STOP, v_go=V_GO, lookahead_s=LOOKAHEAD_S)

def fit_two_lanes_from_points(P: np.ndarray, base_line: Line) -> List[Line]:
    """Split by lateral offset into two clusters (K=2) and fit TLS line per cluster.
    Returns list of 1 or 2 Line objects depending on viability.
    """
    if P is None or len(P) < 2:
        return []
    # Lateral offsets to base line
    d = (P @ base_line.n - base_line.rho).reshape(-1, 1)
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        lab = km.fit_predict(d)
    except Exception:
        # Degenerate: return a single lane from all points
        return [line_from_points_tls(P)]
    lanes: List[Line] = []
    for k in (0, 1):
        Q = P[lab == k]
        if len(Q) >= 2:
            lanes.append(line_from_points_tls(Q))
    if len(lanes) == 0:
        lanes = [line_from_points_tls(P)]
    return lanes

def polyline_along_s(P: np.ndarray, center_xy: np.ndarray, t_dir: np.ndarray,
                     lon0: float, lat0: float,
                     bin_m: float = AXIS_POLY_BIN_M, min_pts: int = AXIS_POLY_MIN_POINTS) -> dict | None:
    """Build a piecewise polyline by binning points along s = (p-center)·t_dir."""
    if P is None or len(P) == 0:
        return None
    s_vals = (P - center_xy.reshape(1, 2)) @ t_dir
    if not (np.isfinite(s_vals).all()):
        return None
    s_min = float(np.min(s_vals))
    s_max = float(np.max(s_vals))
    if not (np.isfinite(s_min) and np.isfinite(s_max)) or s_max <= s_min + 1.0:
        return None
    num_bins = max(1, int(np.ceil((s_max - s_min) / bin_m)))
    xs = np.linspace(s_min, s_max, num_bins + 1)
    coords = []
    for i in range(num_bins):
        a = xs[i]
        b = xs[i + 1]
        mask = (s_vals >= a) & (s_vals <= b)
        if np.count_nonzero(mask) >= 1:
            p_mean = P[mask].mean(axis=0)
            lon_i, lat_i = xy_to_lonlat(p_mean[0], p_mean[1], lon0, lat0)
            coords.append([float(lon_i), float(lat_i)])
    if len(coords) >= 2:
        return {'coordinates': coords}
    return None

# Curves to build from specified label pairs
CURVE_PAIRS = [
    (("A2-1", "B3-1"), "E_right"),
    (("B3-2", "A2-2"), "W_right"),
    (("A3-2", "B2-1"), "W_left"),
    (("B2-2", "A3-1"), "E_left"),
]

def build_curve_poly(df_raw: pd.DataFrame, dir_df: pd.DataFrame, road_id: str,
                     label_pair: Tuple[str, str], center_xy: np.ndarray,
                     lon0: float, lat0: float) -> dict | None:
    P = collect_points_for_labels(df_raw, dir_df, road_id, list(label_pair))
    if P is None or len(P) < 2:
        return None
    t_dir = pca_tls_direction(P)
    return polyline_along_s(P, center_xy, t_dir, lon0, lat0)

def fit_axis_per_cluster(points_df: pd.DataFrame, labels: np.ndarray,
                         ransac_thresh=1.5) -> Dict[int, Dict]:
    """
    For each cluster, fit a line with RANSAC + TLS.
    Returns: {label: {'line': Line, 'inliers_mask': mask, 'points': (N,2)}}.
    """
    result = {}
    for lab in np.unique(labels):
        g = points_df[labels == lab]
        P = g[['_x','_y']].values
        if len(P) < 4:
            continue
        line, inliers = ransac_line(P, thresh=ransac_thresh, iters=400, min_inliers= max(6, len(P)//3))
        result[int(lab)] = {'line': line, 'inliers_mask': inliers, 'points': P}
    return result

def stopline_along_axis(points_df: pd.DataFrame, line: Line, center_xy: np.ndarray,
                        lateral_band=2.0, use_quantile=0.15) -> float:
    """
    Compute the stop-line longitudinal coordinate s_stop (meters) along the given axis.
    """
    P = points_df[['_x','_y']].values
    # Keep only points within a lateral band around the axis
    lat_dev = np.abs(P @ line.n - line.rho)
    keep = lat_dev <= lateral_band
    if not np.any(keep):
        return float('nan')
    T = line.t
    s = (P[keep] - center_xy.reshape(1,2)) @ T
    if np.isnan(use_quantile):
        s_stop = float(np.median(s))
    else:
        s_stop = robust_quantile(s, use_quantile)
    return s_stop

def main():
    for csv_path in INPUT_FILES:
        base = os.path.basename(csv_path)
        name_no_ext = base.rsplit('.', 1)[0]
        road_id = name_no_ext.split('_')[0] if '_' in name_no_ext else name_no_ext
        df = pd.read_csv(csv_path)
        # Coarse origin for local projection; prefer configured centers, otherwise use medians
        c_hint = CENTERS.get(road_id)
        center_lat_hint = float(c_hint[0]) if c_hint is not None else None
        center_lon_hint = float(c_hint[1]) if c_hint is not None else None
        lat0 = center_lat_hint if center_lat_hint is not None else float(np.median(df['latitude']))
        lon0 = center_lon_hint if center_lon_hint is not None else float(np.median(df['longitude']))

        # 1) Compute headings and local coordinates
        df = compute_heading_per_track(df, lon0, lat0)

        # 2) Load direction labels for this road
        dir_df_all = load_direction_df(DIRECTION_FILE)
        if dir_df_all.empty:
            raise RuntimeError("direction.csv not available for direction-based grouping")
        dir_df = dir_df_all[dir_df_all['road_id'] == road_id].copy()

        # 3) Fit global EW and NS axes from combined straight labels to estimate center
        # Updated: NS from A1, EW from B1
        P_NS = collect_points_for_labels(df, dir_df, road_id, ['A1-1', 'A1-2'])
        P_EW = collect_points_for_labels(df, dir_df, road_id, ['B1-1', 'B1-2'])
        if len(P_NS) < 2 or len(P_EW) < 2:
            # Fallback to configured center if insufficient to estimate center
            center_lat = center_lat_hint if center_lat_hint is not None else float(np.median(df['latitude']))
            center_lon = center_lon_hint if center_lon_hint is not None else float(np.median(df['longitude']))
            cx, cy = lonlat_to_xy(np.array([center_lon]), np.array([center_lat]), lon0, lat0)
            center_xy = np.array([float(cx[0]), float(cy[0])])
            # Still compute EW/NS if possible (may be missing)
            line_NS, inliers_NS = (line_from_points_tls(P_NS), np.ones(len(P_NS), dtype=bool)) if len(P_NS) >= 2 else (None, None)
            line_EW, inliers_EW = (line_from_points_tls(P_EW), np.ones(len(P_EW), dtype=bool)) if len(P_EW) >= 2 else (None, None)
        else:
            line_NS, inliers_NS = ransac_line(P_NS, thresh=RANSAC_THRESH, iters=400, min_inliers=max(6, len(P_NS)//3))
            line_EW, inliers_EW = ransac_line(P_EW, thresh=RANSAC_THRESH, iters=400, min_inliers=max(6, len(P_EW)//3))
            center_xy = lsq_intersection_center([line_NS, line_EW], weights=[float(inliers_NS.sum()), float(inliers_EW.sum())])
            center_lon, center_lat = xy_to_lonlat(center_xy[0], center_xy[1], lon0, lat0)

        # 4) Per-side straight base axes and two-lane split
        side_results = {}
        # Determine side assignment by sign of s = (p-center)·t along axis orientation
        # For EW, use t.x sign to map s>=0 to East if t.x>=0, else to West
        # For NS, use t.y sign to map s>=0 to North if t.y>=0, else to South
        # Build helper to clone a Line object
        def _clone_line(L: Line) -> Line:
            return Line(n=L.n.copy(), rho=float(L.rho), t=L.t.copy(), angle_deg=float(L.angle_deg))

        # Precompute signs and s-values for combined point sets
        if P_EW is not None and len(P_EW) >= 2 and line_EW is not None:
            s_EW = (P_EW - center_xy.reshape(1, 2)) @ line_EW.t
            signE = 1.0 if line_EW.t[0] >= 0.0 else -1.0
        else:
            s_EW = None
            signE = 1.0
        if P_NS is not None and len(P_NS) >= 2 and line_NS is not None:
            s_NS = (P_NS - center_xy.reshape(1, 2)) @ line_NS.t
            signN = 1.0 if line_NS.t[1] >= 0.0 else -1.0
        else:
            s_NS = None
            signN = 1.0

        for side in ['W', 'E']:
            if line_EW is None or s_EW is None:
                side_results[side] = {
                    'base_axis': None,
                    'inliers_mask': None,
                    'points': np.zeros((0, 2)),
                    'lanes': []
                }
                continue
            if side == 'E':
                mask = (signE * s_EW) >= 0.0
            else:  # 'W'
                mask = (signE * s_EW) < 0.0
            P_side = P_EW[mask]
            base_axis = _clone_line(line_EW)
            lanes = fit_two_lanes_from_points(P_side, base_axis) if len(P_side) >= 2 else []
            side_results[side] = {
                'base_axis': base_axis,
                'inliers_mask': np.ones(len(P_side), dtype=bool),
                'points': P_side,
                'lanes': lanes
            }

        for side in ['S', 'N']:
            if line_NS is None or s_NS is None:
                side_results[side] = {
                    'base_axis': None,
                    'inliers_mask': None,
                    'points': np.zeros((0, 2)),
                    'lanes': []
                }
                continue
            if side == 'N':
                mask = (signN * s_NS) >= 0.0
            else:  # 'S'
                mask = (signN * s_NS) < 0.0
            P_side = P_NS[mask]
            base_axis = _clone_line(line_NS)
            lanes = fit_two_lanes_from_points(P_side, base_axis) if len(P_side) >= 2 else []
            side_results[side] = {
                'base_axis': base_axis,
                'inliers_mask': np.ones(len(P_side), dtype=bool),
                'points': P_side,
                'lanes': lanes
            }

        # 5) Orient all axis tangents outward per side (if available)
        for side, res in side_results.items():
            if res['base_axis'] is None or len(res['points']) == 0:
                continue
            line = res['base_axis']
            mean_vec = res['points'].mean(axis=0) - center_xy
            if (mean_vec @ line.t) < 0:
                line.t = -line.t
                line.n = -line.n
                line.rho = -line.rho
                line.angle_deg = (line.angle_deg + 180.0) % 360.0
            # Orient lanes similarly
            for ln in res['lanes']:
                if (mean_vec @ ln.t) < 0:
                    ln.t = -ln.t
                    ln.n = -ln.n
                    ln.rho = -ln.rho
                    ln.angle_deg = (ln.angle_deg + 180.0) % 360.0

        # 6) Stopline per side (use union of stop labels)
        stoplines = {}
        for side in ['W', 'E', 'S', 'N']:
            base_axis = side_results[side]['base_axis']
            if base_axis is None:
                stoplines[side] = {'s_stop_m_from_center': None, 'method': f'quantile_{STOP_QUANTILE}', 'labels': LANE_GROUPS[side]['stop_labels']}
                continue
            pts_stop = collect_stop_points_for_labels(df, dir_df, road_id, LANE_GROUPS[side]['stop_labels'])
            s_stop = stopline_along_axis(pts_stop, base_axis, center_xy, lateral_band=LATERAL_BAND, use_quantile=STOP_QUANTILE)
            stoplines[side] = {
                's_stop_m_from_center': float(s_stop) if s_stop is not None and np.isfinite(s_stop) else None,
                'method': f'quantile_{STOP_QUANTILE}' if not np.isnan(STOP_QUANTILE) else 'median',
                'labels': LANE_GROUPS[side]['stop_labels']
            }

        # 7) (Removed) Side-based turn polylines; we only keep four specified curves below
        turn_lanes = {s: {'left': None, 'right': None} for s in ['W', 'E', 'S', 'N']}

        # Build specified curves forming the road network
        curves = []
        for (labels_pair, name) in CURVE_PAIRS:
            poly = build_curve_poly(df, dir_df, road_id, labels_pair, center_xy, lon0, lat0)
            curves.append({'name': name, 'labels': list(labels_pair), 'polyline': poly})

        # 8) Compose output JSON
        out = {
            'center_point': {
                'x_m': float(center_xy[0]),
                'y_m': float(center_xy[1]),
                'lon': center_lon,
                'lat': center_lat
            },
            'lanes_straight': {},
            'lanes_turn': {},
            'stoplines': stoplines,
            'approaches': []
        }

        # Road network summary: axes and curves
        out['road_network'] = {
            'axes': {
                'NS': {
                    'angle_deg': float(line_NS.angle_deg) if 'line_NS' in locals() and line_NS is not None else None,
                    'n': [float(line_NS.n[0]), float(line_NS.n[1])] if 'line_NS' in locals() and line_NS is not None else None,
                    't': [float(line_NS.t[0]), float(line_NS.t[1])] if 'line_NS' in locals() and line_NS is not None else None,
                    'rho_m': float(line_NS.rho) if 'line_NS' in locals() and line_NS is not None else None,
                },
                'EW': {
                    'angle_deg': float(line_EW.angle_deg) if 'line_EW' in locals() and line_EW is not None else None,
                    'n': [float(line_EW.n[0]), float(line_EW.n[1])] if 'line_EW' in locals() and line_EW is not None else None,
                    't': [float(line_EW.t[0]), float(line_EW.t[1])] if 'line_EW' in locals() and line_EW is not None else None,
                    'rho_m': float(line_EW.rho) if 'line_EW' in locals() and line_EW is not None else None,
                },
            },
            'curves': [c for c in curves if c['polyline'] is not None]
        }

        for side, res in side_results.items():
            # Straight lanes (suppressed: only keep base axes in output)
            lanes_serialized = []
            base_axis = res['base_axis']
            base_ser = None
            if base_axis is not None:
                base_ser = {
                    'angle_deg': float(base_axis.angle_deg),
                    'n': [float(base_axis.n[0]), float(base_axis.n[1])],
                    't': [float(base_axis.t[0]), float(base_axis.t[1])],
                    'rho_m': float(base_axis.rho)
                }
            out['lanes_straight'][side] = {
                'base_axis': base_ser,
                'lanes': lanes_serialized,
                'samples': int(len(res['points'])),
            }
            # Turn lanes
            out['lanes_turn'][side] = turn_lanes[side]

        # Also export four base approaches for backward compatibility
        for side, res in side_results.items():
            base_axis = res['base_axis']
            if base_axis is None:
                continue
            out['approaches'].append({
                'cluster_label': side,
                'label_hint': side,
                'axis': {
                    'angle_deg': float(base_axis.angle_deg),
                    'n': [float(base_axis.n[0]), float(base_axis.n[1])],
                    't': [float(base_axis.t[0]), float(base_axis.t[1])],
                    'rho_m': float(base_axis.rho)
                },
                'stopline': stoplines[side],
                'axis_polyline': None,
                'samples': int(len(res['points'])),
                'inliers': int(res['inliers_mask'].sum()) if res['inliers_mask'] is not None else int(len(res['points']))
            })

        # Determine intersection id from file name
        inter_id = road_id

        # Write results JSON under BASE_DIR/data named by intersection id
        out_json_path = os.path.join(DATA_DIR, f"{inter_id}_intersection.json")
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # Optionally write GeoJSON named by intersection id
        if WRITE_GEOJSON:
            out_geojson_path = os.path.join(DATA_DIR, f"{inter_id}_intersection.geojson")
            features = []
            # Center point
            features.append({
                "type": "Feature",
                "properties": {"type": "center"},
                "geometry": {"type": "Point", "coordinates": [center_lon, center_lat]}
            })
            # Visualize axes (only two global axes) and stoplines
            L_vis = 120.0
            # Draw NS axis
            if 'line_NS' in locals() and line_NS is not None:
                t = line_NS.t
                n = line_NS.n
                rho = line_NS.rho
                alpha = rho - n @ center_xy
                p0 = center_xy + alpha * n
                p1 = p0 - t * (L_vis/2)
                p2 = p0 + t * (L_vis/2)
                lon1, lat1 = xy_to_lonlat(p1[0], p1[1], lon0, lat0)
                lon2, lat2 = xy_to_lonlat(p2[0], p2[1], lon0, lat0)
                features.append({
                    "type": "Feature",
                    "properties": {"type": "axis", "name": "NS", "angle_deg": float(line_NS.angle_deg)},
                    "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]}
                })
            # Draw EW axis
            if 'line_EW' in locals() and line_EW is not None:
                t = line_EW.t
                n = line_EW.n
                rho = line_EW.rho
                alpha = rho - n @ center_xy
                p0 = center_xy + alpha * n
                p1 = p0 - t * (L_vis/2)
                p2 = p0 + t * (L_vis/2)
                lon1, lat1 = xy_to_lonlat(p1[0], p1[1], lon0, lat0)
                lon2, lat2 = xy_to_lonlat(p2[0], p2[1], lon0, lat0)
                features.append({
                    "type": "Feature",
                    "properties": {"type": "axis", "name": "EW", "angle_deg": float(line_EW.angle_deg)},
                    "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]}
                })
            # Stoplines per side (unchanged)
            for side, res in side_results.items():
                base_axis = res['base_axis']
                if base_axis is None:
                    continue
                t = base_axis.t
                n = base_axis.n
                s_stop = stoplines[side]['s_stop_m_from_center']
                if s_stop is not None:
                    p_stop_center = center_xy + t * s_stop
                    Ls = 20.0
                    q1 = p_stop_center - n * (Ls/2)
                    q2 = p_stop_center + n * (Ls/2)
                    lon3, lat3 = xy_to_lonlat(q1[0], q1[1], lon0, lat0)
                    lon4, lat4 = xy_to_lonlat(q2[0], q2[1], lon0, lat0)
                    features.append({
                        "type": "Feature",
                        "properties": {"type": "stopline", "side": side},
                        "geometry": {"type": "LineString", "coordinates": [[lon3, lat3], [lon4, lat4]]}
                    })
            # (Removed) turn_lanes visualizations; only draw specified curves below
            # Specified curves forming the road network
            for c in curves:
                if c['polyline'] is not None:
                    features.append({
                        "type": "Feature",
                        "properties": {"type": "curve", "name": c['name'], "labels": c['labels']},
                        "geometry": {"type": "LineString", "coordinates": c['polyline']['coordinates']}
                    })
            with open(out_geojson_path, 'w', encoding='utf-8') as f:
                json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

        # Console summary
        print(f"\n=== Intersection Inference Results ({inter_id}) ===")
        print(f"Center (local m): x={center_xy[0]:.2f}, y={center_xy[1]:.2f}")
        print(f"Center (lon,lat): ({center_lon:.6f}, {center_lat:.6f})")
        for side, res in side_results.items():
            base_axis = res['base_axis']
            if base_axis is None:
                print(f"\n[{side}]  no straight base axis")
                continue
            print(f"\n[{side}]  straight_samples={len(res['points'])}  lanes={len(res['lanes'])}")
            print(f"  Base axis angle = {base_axis.angle_deg:.2f} deg (0=E,90=N)")
            print(f"  Base axis eqn   : n^T x = rho,  n={[float(base_axis.n[0]), float(base_axis.n[1])]}, rho={base_axis.rho:.2f} m")
            sstop = stoplines[side]['s_stop_m_from_center']
            print(f"  Stop-line s(from center) = {None if sstop is None else f'{sstop:.2f} m'}  ({stoplines[side]['method']})")

        if WRITE_GEOJSON:
            print(f"\nWrote {out_json_path} and {out_geojson_path}.")
        else:
            print(f"\nWrote {out_json_path}.")

# Module is imported and executed via main.py
