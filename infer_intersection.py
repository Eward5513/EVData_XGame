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
rng = np.random.default_rng(42)

# ---------- Configuration (hardcoded values) ----------
CSV_PATHS = ['A0003_split.csv', 'A0008_split.csv']  # Input CSVs to process
# Optional global defaults (fallback if road-specific values below are missing)
LAT0 = None
LON0 = None
# Intersection origin hints: seeded from analyze_direction.py (lat, lon)
INTERSECTION_ORIGINS = {
    'A0003': {'lat': 32.345137, 'lon': 123.152539},
    'A0008': {'lat': 32.327137, 'lon': 123.181261},
}
K = 4                        # Number of approach direction clusters
V_STOP = 0.5                 # Stationary speed threshold (m/s)
V_GO = 1.5                   # Start-moving speed threshold (m/s)
LOOKAHEAD_S = 5.0            # Lookahead window for detecting start (s)
RANSAC_THRESH = 1.5          # RANSAC inlier distance threshold (m)
LATERAL_BAND = 2.0           # Lateral band width for stop-line estimation (m)
STOP_QUANTILE = 0.15         # Quantile for conservative stop-line estimate
WRITE_GEOJSON = False        # If True, also write a GeoJSON per CSV

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
    for csv_path in CSV_PATHS:
        base = os.path.basename(csv_path)
        name_no_ext = base.rsplit('.', 1)[0]
        road_id = name_no_ext.split('_')[0] if '_' in name_no_ext else name_no_ext
        df = pd.read_csv(csv_path)
        # Coarse origin for local projection; prefer predefined origins, otherwise use medians
        origin_hint = INTERSECTION_ORIGINS.get(road_id, {})
        lat_pref = origin_hint.get('lat', LAT0)
        lon_pref = origin_hint.get('lon', LON0)
        lat0 = lat_pref if lat_pref is not None else float(np.median(df['latitude']))
        lon0 = lon_pref if lon_pref is not None else float(np.median(df['longitude']))

        # 1) Compute headings and local coordinates
        df = compute_heading_per_track(df, lon0, lat0)

        # 2) Extract last-stop points before moving
        pts = extract_last_stop_points(df,
                                       v_stop=V_STOP,
                                       v_go=V_GO,
                                       lookahead_s=LOOKAHEAD_S)
        if len(pts) < 8:
            print(f"[Warn] Only {len(pts)} candidate stop points found for {csv_path}; results may be unstable. "
                  f"Consider relaxing thresholds or using a longer time window.")

        # 3) Cluster approach directions (default K=4)
        labels = cluster_directions(pts, K=K)

        # 4) Fit an axis for each direction with RANSAC+TLS
        axes = fit_axis_per_cluster(pts.assign(label=labels), labels, ransac_thresh=RANSAC_THRESH)

        if len(axes) < 2:
            raise RuntimeError(f"Fewer than 2 valid directions for {csv_path}; cannot compute center point. Check data or thresholds.")

        # 5) Compute intersection center as the least-squares point
        lines = [v['line'] for v in axes.values()]
        weights = [float(v['inliers_mask'].sum()) for v in axes.values()]
        center_xy = lsq_intersection_center(lines, weights=weights)
        center_lon, center_lat = xy_to_lonlat(center_xy[0], center_xy[1], lon0, lat0)

        # 6) Estimate stop-line position along each approach axis
        out = {
            'center_point': {
                'x_m': float(center_xy[0]),
                'y_m': float(center_xy[1]),
                'lon': center_lon,
                'lat': center_lat
            },
            'approaches': []
        }

        # For friendlier labels, map axis angles to compass sectors (coarse)
        lab_to_dir = {}
        for lab, v in axes.items():
            ang = v['line'].angle_deg  # Axis angle: 0=East, 90=North
            # Normalize to 0..360
            a = ang % 360.0
            # Coarse mapping
            dirs = ['E','NE','N','NW','W','SW','S','SE']
            idx = int(((a + 22.5) % 360) // 45)
            lab_to_dir[lab] = dirs[idx]

        for lab, v in axes.items():
            line = v['line']
            # Orient tangent outward: if mean(point - center) · t is negative, flip
            P = v['points'][v['inliers_mask']]
            mean_vec = P.mean(axis=0) - center_xy
            if (mean_vec @ line.t) < 0:
                line.t = -line.t
                line.n = -line.n
                line.rho = -line.rho
                line.angle_deg = (line.angle_deg + 180.0) % 360.0

            # Stop line estimate
            s_stop = stopline_along_axis(pts.loc[labels==lab], line, center_xy,
                                         lateral_band=LATERAL_BAND,
                                         use_quantile=STOP_QUANTILE)

            out['approaches'].append({
                'cluster_label': int(lab),
                'label_hint': lab_to_dir.get(lab, f'cluster_{lab}'),
                'axis': {
                    'angle_deg': float(line.angle_deg),         # 0=E,90=N
                    'n': [float(line.n[0]), float(line.n[1])],
                    't': [float(line.t[0]), float(line.t[1])],
                    'rho_m': float(line.rho)                   # n^T x = rho
                },
                'stopline': {
                    's_stop_m_from_center': float(s_stop) if not np.isnan(s_stop) else None,
                    'method': f'quantile_{STOP_QUANTILE}' if not np.isnan(STOP_QUANTILE) else 'median'
                },
                'samples': int(len(P)),
                'inliers': int(v['inliers_mask'].sum())
            })

        # Determine intersection id from file name
        inter_id = road_id

        # Write results JSON named by intersection id
        out_json_path = f"{inter_id}.json"
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # Optionally write GeoJSON named by intersection id
        if WRITE_GEOJSON:
            features = []
            # Center point
            features.append({
                "type": "Feature",
                "properties": {"type": "center"},
                "geometry": {"type": "Point", "coordinates": [center_lon, center_lat]}
            })
            # Axes and stop lines (draw 120 m length for visualization)
            L_vis = 120.0
            for app in out['approaches']:
                t = np.array(app['axis']['t'])
                n = np.array(app['axis']['n'])
                rho = app['axis']['rho_m']
                # Find a point p0 near the center lying on the axis:
                # n^T p = rho and along the line direction ± L_vis/2.
                # First, compute the point p0 on the line closest to the center:
                # n^T p0 = rho, p0 = center + alpha * n; solve for alpha
                alpha = rho - n @ center_xy
                p0 = center_xy + alpha * n
                p1 = p0 - t * (L_vis/2)
                p2 = p0 + t * (L_vis/2)
                lon1, lat1 = xy_to_lonlat(p1[0], p1[1], lon0, lat0)
                lon2, lat2 = xy_to_lonlat(p2[0], p2[1], lon0, lat0)
                features.append({
                    "type": "Feature",
                    "properties": {"type": "axis", "label": app['label_hint'], "angle_deg": app['axis']['angle_deg']},
                    "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]}
                })
                # Stop line: perpendicular to the axis, passing through s_stop
                s_stop = app['stopline']['s_stop_m_from_center']
                if s_stop is not None:
                    p_stop_center = center_xy + t * s_stop
                    # Render stop line with length 20 m
                    Ls = 20.0
                    q1 = p_stop_center - n * (Ls/2)
                    q2 = p_stop_center + n * (Ls/2)
                    lon1, lat1 = xy_to_lonlat(q1[0], q1[1], lon0, lat0)
                    lon2, lat2 = xy_to_lonlat(q2[0], q2[1], lon0, lat0)
                    features.append({
                        "type": "Feature",
                        "properties": {"type": "stopline", "label": app['label_hint']},
                        "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]}
                    })
            out_geojson_path = f"{inter_id}.geojson"
            with open(out_geojson_path, 'w', encoding='utf-8') as f:
                json.dump({"type":"FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)

        # Console summary
        print(f"\n=== Intersection Inference Results ({inter_id}) ===")
        print(f"Center (local m): x={center_xy[0]:.2f}, y={center_xy[1]:.2f}")
        print(f"Center (lon,lat): ({center_lon:.6f}, {center_lat:.6f})")
        for app in out['approaches']:
            print(f"\n[{app['label_hint']}]  samples={app['samples']}  inliers={app['inliers']}")
            print(f"  Axis angle = {app['axis']['angle_deg']:.2f} deg (0=E,90=N)")
            print(f"  Axis eqn   : n^T x = rho,  n={app['axis']['n']}, rho={app['axis']['rho_m']:.2f} m")
            sstop = app['stopline']['s_stop_m_from_center']
            print(f"  Stop-line s(from center) = {None if sstop is None else f'{sstop:.2f} m'}  ({app['stopline']['method']})")

        if WRITE_GEOJSON:
            print(f"\nWrote {out_json_path} and {out_geojson_path}.")
        else:
            print(f"\nWrote {out_json_path}.")

if __name__ == "__main__":
    main()
