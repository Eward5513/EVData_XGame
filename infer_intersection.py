#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString, mapping
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

# -----------------------------
# Small geodesy helpers
# -----------------------------
def lonlat_to_local_xy(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lon/lat (deg) to local x,y (meters) using equirectangular approximation
    around the mean latitude. Good enough for small intersection neighborhoods.
    """
    lat0 = np.deg2rad(np.nanmean(lat))
    R = 6371000.0
    x = np.deg2rad(lon) * R * math.cos(lat0)
    y = np.deg2rad(lat) * R
    return x, y

def local_xy_to_lonlat(x: np.ndarray, y: np.ndarray, lat0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    lat0 = math.radians(lat0_deg)
    R = 6371000.0
    lon = np.rad2deg(x / (R * math.cos(lat0)))
    lat = np.rad2deg(y / R)
    return lon, lat

def angle_of(vecx: float, vecy: float) -> float:
    """Return angle in radians within [0, 2π)."""
    a = math.atan2(vecy, vecx)
    if a < 0:
        a += 2 * math.pi
    return a

def circmean(angles: np.ndarray) -> float:
    """Circular mean of angles (radians)."""
    return math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) % (2*math.pi)

# -----------------------------
# Core data containers
# -----------------------------
@dataclass
class Arm:
    id: int
    angle: float  # direction angle (radians) pointing outward from center
    tip_xy: Tuple[float, float]  # a point along the arm (meters, local frame)
    node_xy: Tuple[float, float]  # node location on inner ring (meters, local frame)

@dataclass
class MovementEdge:
    u: int
    v: int
    weight: int
    geometry: LineString  # simple curve between nodes

# -----------------------------
# Preprocessing & feature derivation
# -----------------------------
def parse_time(df: pd.DataFrame) -> pd.Series:
    # Prefer epoch in 'time_stamp' if valid
    ts = df.get("time_stamp", pd.Series([np.nan]*len(df)))
    if np.issubdtype(ts.dtype, np.number):
        # Heuristics: seconds vs milliseconds
        ts_vals = ts.to_numpy(np.float64)
        # sanitize
        bad = ~np.isfinite(ts_vals)
        ts_vals[bad] = np.nan
        # if looks like ms (e.g., > 10^10), convert
        if np.nanmedian(ts_vals) > 1e10:
            ts_vals = ts_vals / 1000.0
        return pd.to_datetime(ts_vals, unit="s", errors="coerce")
    # Fallback: collectiontime or date
    for col in ["collectiontime", "date"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    # Last resort: construct from separate 'date' + 'time' if available
    if "time" in df.columns and "date" in df.columns:
        return pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    # otherwise NaT
    return pd.to_datetime(pd.Series([np.nan]*len(df)))

def compute_heading_and_curvature(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward heading and a simple curvature proxy (heading change magnitude).
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    headings = np.arctan2(dy, dx)  # [-pi, pi]
    # unwrap angles to avoid jumps, then rewrap to [-pi, pi]
    headings_unwrapped = np.unwrap(headings)
    dhead = np.gradient(headings_unwrapped)
    curv = np.abs(dhead)
    # rewrap headings to [0, 2π)
    headings = (headings + 2*np.pi) % (2*np.pi)
    return headings, curv

# -----------------------------
# Intersection center detection
# -----------------------------
def detect_intersection_center(x: np.ndarray, y: np.ndarray, speed: np.ndarray,
                               headings: np.ndarray, curvature: np.ndarray,
                               low_speed_kmh: float = 10.0,
                               high_curv_thresh: float = 0.15) -> Tuple[float, float]:
    """
    Heuristic: select candidate points where speed low AND curvature high.
    Cluster by DBSCAN; choose the densest cluster centroid as center.
    Returns (cx, cy) in local meters.
    """
    # to m/s from whatever speed may be (assume km/h if > 40 typically)
    sp = speed.copy().astype(float)
    sp[~np.isfinite(sp)] = np.nan
    if np.nanmedian(sp) > 40.0:  # km/h
        sp = sp / 3.6
    # boolean mask
    mask = (sp < low_speed_kmh/3.6) | (curvature > high_curv_thresh)
    cand_x = x[mask & np.isfinite(x) & np.isfinite(y)]
    cand_y = y[mask & np.isfinite(x) & np.isfinite(y)]
    if len(cand_x) < 30:
        # fallback: use overall KDE peak on all points
        pts = np.vstack([x, y]).T
    else:
        pts = np.vstack([cand_x, cand_y]).T
    if len(pts) < 10:
        # degenerate fallback: center of mass
        return float(np.nanmean(x)), float(np.nanmean(y))
    clustering = DBSCAN(eps=8.0, min_samples=15).fit(pts)
    labels = clustering.labels_
    best_center = (float(np.nanmean(x)), float(np.nanmean(y)))
    best_count = -1
    for lab in set(labels):
        if lab == -1:
            continue
        cluster_pts = pts[labels == lab]
        if len(cluster_pts) > best_count:
            best_count = len(cluster_pts)
            best_center = (float(cluster_pts[:,0].mean()), float(cluster_pts[:,1].mean()))
    return best_center

# -----------------------------
# Arm detection on an annulus
# -----------------------------
def detect_arms(x: np.ndarray, y: np.ndarray, cx: float, cy: float,
                inner_r: float = 20.0, outer_r: float = 120.0,
                angle_bin_deg: float = 18.0, min_bin_pts: int = 100) -> List[Arm]:
    """
    Use an annulus around the center; bin points by polar angle; keep bins that are strong peaks.
    For each retained bin, estimate an arm direction (circular mean) and a tip point.
    Returns a list of Arm objects with node on inner circle and tip near outer boundary.
    """
    dx = x - cx
    dy = y - cy
    r = np.hypot(dx, dy)
    ang = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)

    mask = (r >= inner_r) & (r <= outer_r)
    ang_m = ang[mask]
    r_m = r[mask]
    dx_m = dx[mask]
    dy_m = dy[mask]

    if len(ang_m) < 50:
        return []

    # angle histogram
    bin_rad = math.radians(angle_bin_deg)
    bins = int(math.ceil(2*np.pi / bin_rad))
    hist, edges = np.histogram(ang_m, bins=bins, range=(0, 2*np.pi))
    # smooth histogram (simple moving average)
    smooth = np.convolve(hist, np.ones(3)/3.0, mode='same')

    # find peaks: bins whose count > neighbors and above threshold
    peaks = []
    thr = max(min_bin_pts, int(0.02 * len(ang_m)))
    for i in range(bins):
        prev_i = (i-1) % bins
        next_i = (i+1) % bins
        if smooth[i] > smooth[prev_i] and smooth[i] > smooth[next_i] and smooth[i] >= thr:
            peaks.append(i)

    arms: List[Arm] = []
    for k, bi in enumerate(peaks):
        a0 = edges[bi]
        a1 = edges[(bi+1) % len(edges)]
        if a1 <= a0:
            a1 += 2*np.pi
        # select points within this bin (with wrap handling)
        sel = (ang_m >= a0) & (ang_m < a1)
        if not np.any(sel):
            continue
        angs = ang_m[sel]
        # circular mean
        mu = circmean(angs)
        # pick a tip point near outer radius along this sector
        sector_dx = dx_m[sel]
        sector_dy = dy_m[sel]
        sector_r = r_m[sel]
        if len(sector_r) == 0:
            continue
        j = np.argmax(sector_r)
        tip = (cx + float(sector_dx[j]), cy + float(sector_dy[j]))
        node = (cx + inner_r * math.cos(mu), cy + inner_r * math.sin(mu))
        arms.append(Arm(id=len(arms), angle=mu, tip_xy=tip, node_xy=node))
    # Reassign sequential IDs
    for i, a in enumerate(arms):
        a.id = i
    return arms

# -----------------------------
# Movement extraction (enter->exit)
# -----------------------------
def label_arm(angle: float, arms: List[Arm]) -> Optional[int]:
    if not arms:
        return None
    diffs = [min((abs(angle - a.angle), 2*np.pi - abs(angle - a.angle)))[0] for a in arms]
    idx = int(np.argmin(diffs))
    # reject if too far (e.g., > 30 deg)? keep permissive first
    return arms[idx].id

def extract_movements_per_traj(traj_xy_t: np.ndarray,
                               center: Tuple[float,float],
                               inner_r: float,
                               arms: List[Arm]) -> Optional[Tuple[int,int]]:
    """
    From a single trajectory (x,y,t) sorted by t, detect first inward crossing into inner_r
    and subsequent outward crossing to identify (enter_arm -> exit_arm).
    """
    cx, cy = center
    x = traj_xy_t[:,0]; y = traj_xy_t[:,1]
    r = np.hypot(x - cx, y - cy)
    if len(r) < 3:
        return None
    # Find indices where it first goes inside inner_r and then exits
    inside = r < inner_r
    if not np.any(inside):
        return None
    first_in = np.argmax(inside)  # first True index
    # search forward for first out->in transition end (i.e., leave the disk)
    after = inside[first_in:]
    # if always inside, skip
    if np.all(after):
        return None
    # index of first False after first_in
    rel = np.argmax(~after)
    first_out = first_in + rel
    # determine angles near entry and exit
    # entry angle: use a small window before first_in (5 points)
    s0 = max(0, first_in-5)
    e0 = first_in
    if e0 <= s0:
        return None
    vx0 = np.gradient(x[s0:e0+1])
    vy0 = np.gradient(y[s0:e0+1])
    ang0 = angle_of(np.nanmean(vx0), np.nanmean(vy0))
    arm_in = label_arm(ang0, arms)
    # exit angle: small window after first_out
    s1 = first_out
    e1 = min(len(x)-1, first_out+5)
    vx1 = np.gradient(x[s1:e1+1])
    vy1 = np.gradient(y[s1:e1+1])
    ang1 = angle_of(np.nanmean(vx1), np.nanmean(vy1))
    arm_out = label_arm(ang1, arms)
    if arm_in is None or arm_out is None or arm_in == arm_out:
        return None
    return (arm_in, arm_out)

def build_edges(arms: List[Arm],
                movements: List[Tuple[int,int]],
                min_support: int = 5) -> List[MovementEdge]:
    counts = Counter(movements)
    edges: List[MovementEdge] = []
    for (u, v), w in counts.items():
        if w < min_support:
            continue
        # simple curved geometry: quadratic Bezier approximated by 20 samples
        p0 = np.array(arms[u].node_xy)
        p2 = np.array(arms[v].node_xy)
        # control point: rotate p0 direction halfway toward p2 and offset outward a bit
        ang_mid = math.atan2((p2 - p0)[1], (p2 - p0)[0])
        # use center as mild attractor to produce a bend
        center = np.array([0.0, 0.0])  # since node_xy are absolute, we need center; will override later in main
        # Placeholder: will be replaced in main with actual center if available
        c = 0.5*(p0 + p2)
        ts = np.linspace(0, 1, 20)
        pts = [(1-t)**2 * p0 + 2*(1-t)*t * c + t**2 * p2 for t in ts]
        line = LineString(pts)
        edges.append(MovementEdge(u=u, v=v, weight=w, geometry=line))
    return edges

# -----------------------------
# GeoJSON export
# -----------------------------
def export_geojson(arms: List[Arm], edges: List[MovementEdge],
                   cx: float, cy: float, lat0_deg: float, outfile: str):
    # Collect all geometries and convert back to lon/lat
    features = []

    # center point
    clon, clat = local_xy_to_lonlat(np.array([cx]), np.array([cy]), lat0_deg)
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(clon[0]), float(clat[0]) ]},
        "properties": {"kind": "center"}
    })

    # nodes
    if arms:
        nx = np.array([a.node_xy[0] for a in arms])
        ny = np.array([a.node_xy[1] for a in arms])
        lon, lat = local_xy_to_lonlat(nx, ny, lat0_deg)
        for i, a in enumerate(arms):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [float(lon[i]), float(lat[i]) ]},
                "properties": {"kind": "arm_node", "id": a.id, "angle_rad": a.angle}
            })

    # edges
    for e in edges:
        xs = np.array([p[0] for p in np.asarray(e.geometry.coords)])
        ys = np.array([p[1] for p in np.asarray(e.geometry.coords)])
        lon, lat = local_xy_to_lonlat(xs, ys, lat0_deg)
        coords = [[float(lon[i]), float(lat[i])] for i in range(len(lon))]
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"kind": "movement", "u": e.u, "v": e.v, "weight": e.weight}
        })

    gj = {"type": "FeatureCollection", "features": features}
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)

# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(df: pd.DataFrame,
                 inner_r: float = 25.0,
                 outer_r: float = 120.0,
                 min_support: int = 5,
                 ann_bin_deg: float = 18.0) -> Dict:
    # Basic cleaning
    need_cols = ["vehicle_id","longitude","latitude"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    # parse time and sort
    df = df.copy()
    df["t"] = parse_time(df)
    if df["t"].isna().all():
        # if no time, use monotonic index per vehicle
        df["t"] = df.groupby("vehicle_id").cumcount()
    df = df.sort_values(["vehicle_id","t"]).reset_index(drop=True)

    # project to local meters
    lon = df["longitude"].to_numpy(np.float64)
    lat = df["latitude"].to_numpy(np.float64)
    lat0 = np.nanmean(lat) if np.isfinite(lat).any() else 0.0
    X, Y = lonlat_to_local_xy(lon, lat)

    # compute headings & curvature per whole stream (approximation)
    headings, curv = compute_heading_and_curvature(X, Y)
    speed = df.get("speed", pd.Series([np.nan]*len(df))).to_numpy(np.float64)

    # detect center
    cx, cy = detect_intersection_center(X, Y, speed, headings, curv)

    # detect arms
    arms = detect_arms(X, Y, cx, cy, inner_r=inner_r, outer_r=outer_r,
                       angle_bin_deg=ann_bin_deg, min_bin_pts=100)

    # collect movements per trajectory
    movements = []
    for vid, g in df.groupby("vehicle_id", sort=False):
        xg, yg, tg = X[g.index], Y[g.index], df.loc[g.index, "t"].to_numpy()
        traj = np.vstack([xg, yg, tg]).T
        mv = extract_movements_per_traj(traj, (cx,cy), inner_r, arms)
        if mv is not None:
            movements.append(mv)

    # build edges
    edges = build_edges(arms, movements, min_support=min_support)

    # Return everything
    return {
        "center_xy": (cx, cy),
        "lat0_deg": float(lat0),
        "arms": arms,
        "edges": edges,
        "movements": movements
    }

def main():
    ap = argparse.ArgumentParser(description="Infer a single-intersection road topology from trajectories.")
    ap.add_argument("--input", required=True, help="Input CSV path.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--inner_r", type=float, default=25.0, help="Inner radius (m) for intersection disk.")
    ap.add_argument("--outer_r", type=float, default=120.0, help="Outer radius (m) for arm detection annulus.")
    ap.add_argument("--min_support", type=int, default=5, help="Minimum trajectories to keep a movement edge.")
    ap.add_argument("--ann_bin_deg", type=float, default=18.0, help="Angle bin size for arm histogram (degrees).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.input)

    res = run_pipeline(df, inner_r=args.inner_r, outer_r=args.outer_r,
                       min_support=args.min_support, ann_bin_deg=args.ann_bin_deg)

    # export geojson
    gj_path = os.path.join(args.out, "intersection_topology.geojson")
    export_geojson(res["arms"], res["edges"], res["center_xy"][0], res["center_xy"][1],
                   res["lat0_deg"], gj_path)

    # also export summary json
    summary = {
        "arms": [{"id": a.id, "angle_deg": math.degrees(a.angle)} for a in res["arms"]],
        "edge_supports": [{"u": e.u, "v": e.v, "weight": e.weight} for e in res["edges"]],
        "movements_total": len(res["movements"])
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved: {gj_path}")
    print(f"Saved: {os.path.join(args.out, 'summary.json')}")

if __name__ == "__main__":
    main()
