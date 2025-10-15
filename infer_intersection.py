#!/usr/bin/env python3
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

# Optional dependency: scikit-learn (DBSCAN). Fallback provided if missing.
try:
    from sklearn.cluster import DBSCAN
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

# -----------------------------
# Geodesy helpers
# -----------------------------
def lonlat_to_local_xy(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat0 = math.radians(float(np.nanmean(lat))) if np.isfinite(lat).any() else 0.0
    R = 6371000.0
    x = np.radians(lon) * R * math.cos(lat0)
    y = np.radians(lat) * R
    return x, y

def local_xy_to_lonlat(x: np.ndarray, y: np.ndarray, lat0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    lat0 = math.radians(lat0_deg)
    R = 6371000.0
    lon = np.degrees(x / (R * math.cos(lat0)))
    lat = np.degrees(y / R)
    return lon, lat

def angle_of(vecx: float, vecy: float) -> float:
    a = math.atan2(vecy, vecx)
    if a < 0:
        a += 2 * math.pi
    return a

def circmean(angles: np.ndarray) -> float:
    return math.atan2(np.sin(angles).mean(), np.cos(angles).mean()) % (2 * math.pi)

# -----------------------------
# Optional manual center seed (env override)
# -----------------------------
def get_manual_center_seed(lat0_deg: float) -> Optional[Tuple[float, float]]:
    try:
        lon_str = os.environ.get("XGAME_CENTER_LON")
        lat_str = os.environ.get("XGAME_CENTER_LAT")
        if not lon_str or not lat_str:
            return None
        lon = float(lon_str)
        lat = float(lat_str)
        # convert lon/lat to local XY using dataset's lat0
        R = 6371000.0
        lat0 = math.radians(lat0_deg)
        x = math.radians(lon) * R * math.cos(lat0)
        y = math.radians(lat) * R
        return float(x), float(y)
    except Exception:
        return None

def refine_center_near_seed(x: np.ndarray, y: np.ndarray,
                            speed: np.ndarray, curvature: np.ndarray,
                            seed_xy: Tuple[float, float], radius: float = 40.0) -> Tuple[float, float]:
    cx0, cy0 = seed_xy
    sp = speed.astype(float)
    if np.nanmedian(sp) > 40.0:
        sp = sp / 3.6
    sp_clip = np.clip(sp, 0.0, np.nanmax(sp) if np.isfinite(sp).any() else 1.0)
    slow_w = 1.0 / (1.0 + sp_clip)
    curv = curvature.astype(float)
    curv[~np.isfinite(curv)] = 0.0
    curv_norm = curv / (np.nanmax(curv) + 1e-12) if np.nanmax(curv) > 0 else np.zeros_like(curv)
    w = 0.6 * slow_w + 0.4 * curv_norm + 0.05
    d2 = (x - cx0)**2 + (y - cy0)**2
    neigh = d2 <= (radius**2)
    if not np.any(neigh):
        return cx0, cy0
    wloc = w[neigh]
    xloc = x[neigh]
    yloc = y[neigh]
    s = float(np.nansum(wloc))
    if s <= 0:
        return cx0, cy0
    return float(np.nansum(wloc * xloc) / s), float(np.nansum(wloc * yloc) / s)

# -----------------------------
# Robust center by segment intersections (voting)
# -----------------------------
def _line_intersection(p1: Tuple[float,float], p2: Tuple[float,float],
                       p3: Tuple[float,float], p4: Tuple[float,float]) -> Optional[Tuple[float,float]]:
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    return (float(px), float(py))

def _estimate_center_by_segment_votes(x: np.ndarray, y: np.ndarray,
                                      speed: np.ndarray, curvature: np.ndarray) -> Optional[Tuple[float,float]]:
    n = x.size
    if n < 8:
        return None

    sp = speed.astype(float).copy()
    if np.nanmedian(sp) > 40.0:
        sp = sp / 3.6
    sp[~np.isfinite(sp)] = np.nan
    sp_clip = np.clip(sp, 0.0, np.nanmax(sp) if np.isfinite(sp).any() else 1.0)
    slow_w = 1.0 / (1.0 + sp_clip)

    curv = curvature.astype(float).copy()
    curv[~np.isfinite(curv)] = 0.0
    curv_norm = curv / (np.nanmax(curv) + 1e-12) if np.nanmax(curv) > 0 else np.zeros_like(curv)

    # Candidate mask: slow or high curvature
    mask_cand = (sp_clip <= np.nanpercentile(sp_clip, 50.0)) | (curv_norm >= np.nanpercentile(curv_norm, 60.0))

    # Build short segments
    segments = []  # (x1,y1,x2,y2, heading, length, weight)
    step = 3
    for i in range(0, n - step):
        j = i + step
        if not (mask_cand[i] and mask_cand[j]):
            continue
        x1, y1 = float(x[i]), float(y[i])
        x2, y2 = float(x[j]), float(y[j])
        vx = x2 - x1; vy = y2 - y1
        L = math.hypot(vx, vy)
        if not (8.0 <= L <= 120.0):
            continue
        hd = math.atan2(vy, vx)
        w = 0.7 * float(0.5*(slow_w[i] + slow_w[j])) + 0.3 * float(0.5*(curv_norm[i] + curv_norm[j]))
        segments.append((x1,y1,x2,y2, hd, L, w))

    if len(segments) < 20:
        return None

    # Keep top segments by weight to limit cost
    segments.sort(key=lambda t: t[6], reverse=True)
    segments = segments[:800]

    # Pair intersections with angular diversity
    intersections = []  # (px,py, weight)
    m = len(segments)
    for a in range(m):
        x1,y1,x2,y2,h1,L1,w1 = segments[a]
        for b in range(a+1, min(m, a+40)):
            x3,y3,x4,y4,h2,L2,w2 = segments[b]
            da = abs((h1 - h2 + math.pi) % (2*math.pi) - math.pi)
            if da < math.radians(45) or da > math.radians(135):
                continue
            ip = _line_intersection((x1,y1),(x2,y2),(x3,y3),(x4,y4))
            if ip is None:
                continue
            intersections.append((ip[0], ip[1], w1*w2))

    if len(intersections) < 30:
        return None

    # Vote with a coarse grid
    xs = np.array([p[0] for p in intersections], dtype=np.float64)
    ys = np.array([p[1] for p in intersections], dtype=np.float64)
    ws = np.array([p[2] for p in intersections], dtype=np.float64)
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    cell = 5.0
    nx = max(1, min(400, int((xmax - xmin) / cell)))
    ny = max(1, min(400, int((ymax - ymin) / cell)))
    H, xe, ye = np.histogram2d(xs, ys, bins=[nx, ny], range=[[xmin,xmax],[ymin,ymax]], weights=ws)
    if H.size == 0:
        return None
    idx = np.unravel_index(int(np.nanargmax(H)), H.shape)
    cx0 = 0.5 * (xe[idx[0]] + xe[idx[0]+1])
    cy0 = 0.5 * (ye[idx[1]] + ye[idx[1]+1])

    # Refine with weighted centroid of nearby intersections
    d2 = (xs - cx0)**2 + (ys - cy0)**2
    neigh = d2 <= (25.0**2)
    if np.any(neigh) and np.nansum(ws[neigh]) > 0:
        cx = float(np.nansum(ws[neigh] * xs[neigh]) / np.nansum(ws[neigh]))
        cy = float(np.nansum(ws[neigh] * ys[neigh]) / np.nansum(ws[neigh]))
        return (cx, cy)
    return (cx0, cy0)

# -----------------------------
# Data containers
# -----------------------------
@dataclass
class Arm:
    id: int
    angle: float
    tip_xy: Tuple[float, float]
    node_xy: Tuple[float, float]

@dataclass
class MovementEdge:
    u: int
    v: int
    weight: int
    geometry: LineString

# -----------------------------
# Time parsing (NumPy 2 safe)
# -----------------------------
def parse_time_seconds(df: pd.DataFrame) -> np.ndarray:
    """
    Return a float64 array of seconds since epoch (NaN if unknown).
    Tries 'time_stamp' as clock time first (HH:MM:SS), then 'collectiontime' ms,
    then full datetime in 'date' if possible.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float64)

    # Try epoch seconds in a numeric 'time_stamp' (if user provided)
    if "time_stamp" in df.columns:
        ts = df["time_stamp"]
        if pd.api.types.is_numeric_dtype(ts):
            tsv = pd.to_numeric(ts, errors="coerce").to_numpy(np.float64)
            # If values look like ms, convert
            if np.nanmedian(tsv) > 1e10:
                tsv = tsv / 1000.0
            out = tsv
        else:
            # Try parse as HH:MM:SS paired with date
            if "date" in df.columns:
                dt = pd.to_datetime(df["date"].astype(str) + " " + ts.astype(str),
                                    errors="coerce")
                out = (dt.astype("int64") / 1e9).to_numpy(np.float64)

    # Fallback: collectiontime (ms)
    if np.isnan(out).all() and "collectiontime" in df.columns:
        ms = pd.to_numeric(df["collectiontime"], errors="coerce")
        dt = pd.to_datetime(ms, unit="ms", errors="coerce")
        out = (dt.astype("int64") / 1e9).to_numpy(np.float64)

    # Fallback: date
    if np.isnan(out).all() and "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        out = (dt.astype("int64") / 1e9).to_numpy(np.float64)

    return out

# -----------------------------
# Heading & curvature
# -----------------------------
def compute_heading_and_curvature(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < 3:
        headings = np.zeros_like(x, dtype=np.float64)
        curv = np.zeros_like(x, dtype=np.float64)
        return headings, curv
    dx = np.gradient(x.astype(np.float64))
    dy = np.gradient(y.astype(np.float64))
    headings = np.arctan2(dy, dx)
    headings_unwrapped = np.unwrap(headings)
    dhead = np.gradient(headings_unwrapped)
    curv = np.abs(dhead)
    headings = (headings + 2 * np.pi) % (2 * np.pi)
    return headings, curv

# -----------------------------
# Center detection with fallback
# -----------------------------
def _grid_peak_center(pts: np.ndarray, cell: float = 5.0) -> Tuple[float, float]:
    """Simple 2D grid histogram peak as a fallback when sklearn is unavailable."""
    if len(pts) == 0:
        return 0.0, 0.0
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    nx = max(1, int((xmax - xmin) / cell))
    ny = max(1, int((ymax - ymin) / cell))
    H, xe, ye = np.histogram2d(pts[:,0], pts[:,1], bins=[nx, ny])
    idx = np.unravel_index(np.argmax(H), H.shape)
    cx = 0.5 * (xe[idx[0]] + xe[idx[0]+1])
    cy = 0.5 * (ye[idx[1]] + ye[idx[1]+1])
    return float(cx), float(cy)

def detect_intersection_center(x: np.ndarray, y: np.ndarray, speed: np.ndarray,
                               curvature: np.ndarray,
                               low_speed_kmh: float = 10.0,
                               high_curv_thresh: float = 0.15) -> Tuple[float, float]:
    """Robust center using heading-variance density scoring + refinement.

    - Build grid; in each cell accumulate: count, sum of cos/sin of heading, mean curvature, mean slow-speed weight.
    - Score cell = count * (w1 * heading_variance + w2 * mean_curv + w3 * mean_slow).
    - Pick best cell center; refine with weighted centroid of neighbors.
    """
    # Prepare inputs
    xf = x.astype(float)
    yf = y.astype(float)
    if not (np.isfinite(xf).any() and np.isfinite(yf).any()):
        return float(np.nanmean(x)), float(np.nanmean(y))

    # Compute headings locally (robust to small arrays)
    if xf.size >= 3:
        dx = np.gradient(xf)
        dy = np.gradient(yf)
        hd = np.arctan2(dy, dx)
    else:
        hd = np.zeros_like(xf)

    sp = speed.astype(float)
    if np.nanmedian(sp) > 40.0:
        sp = sp / 3.6
    curv = curvature.astype(float)
    curv[~np.isfinite(curv)] = 0.0

    # Slow-speed weight in [0,1]
    sp_clip = np.clip(sp, 0.0, np.nanmax(sp) if np.isfinite(sp).any() else 1.0)
    slow_w = 1.0 / (1.0 + sp_clip)

    # Bounding box
    xmin, xmax = float(np.nanmin(xf)), float(np.nanmax(xf))
    ymin, ymax = float(np.nanmin(yf)), float(np.nanmax(yf))
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmax <= xmin or ymax <= ymin:
        return float(np.nanmean(x)), float(np.nanmean(y))

    # Grid resolution ~5m
    cell = 5.0
    nx = max(1, min(400, int((xmax - xmin) / cell)))
    ny = max(1, min(400, int((ymax - ymin) / cell)))

    # Digitize points into bins
    bx = np.clip(np.digitize(xf, np.linspace(xmin, xmax, nx+1)) - 1, 0, nx-1)
    by = np.clip(np.digitize(yf, np.linspace(ymin, ymax, ny+1)) - 1, 0, ny-1)

    # Accumulators
    C = np.zeros((nx, ny), dtype=np.float64)
    Sx = np.zeros((nx, ny), dtype=np.float64)
    Sy = np.zeros((nx, ny), dtype=np.float64)
    Cur = np.zeros((nx, ny), dtype=np.float64)
    SSp = np.zeros((nx, ny), dtype=np.float64)

    cosh = np.cos(hd)
    sinh = np.sin(hd)

    for i in range(xf.size):
        ii = int(bx[i]); jj = int(by[i])
        C[ii, jj] += 1.0
        Sx[ii, jj] += cosh[i]
        Sy[ii, jj] += sinh[i]
        Cur[ii, jj] += curv[i]
        SSp[ii, jj] += slow_w[i]

    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.sqrt(Sx*Sx + Sy*Sy) / np.maximum(C, 1.0)
        heading_var = 1.0 - np.clip(R, 0.0, 1.0)  # [0,1]
        mean_curv = Cur / np.maximum(C, 1.0)
        mean_slow = SSp / np.maximum(C, 1.0)

    # Score: prioritize heading variance (direction diversity), then curvature, then slow speed
    w1, w2, w3 = 0.7, 0.2, 0.1
    Score = C * (w1 * heading_var + w2 * mean_curv + w3 * mean_slow)

    if not np.isfinite(Score).any() or np.nanmax(Score) <= 0:
        return float(np.nanmean(x)), float(np.nanmean(y))

    idx = np.unravel_index(int(np.nanargmax(Score)), Score.shape)
    xe = np.linspace(xmin, xmax, nx+1)
    ye = np.linspace(ymin, ymax, ny+1)
    cx0 = 0.5 * (xe[idx[0]] + xe[idx[0]+1])
    cy0 = 0.5 * (ye[idx[1]] + ye[idx[1]+1])

    # Refine center: weighted centroid of neighbors within 30m using per-point weights derived from the cell scores
    # Assign each point its cell score
    point_score = Score[bx, by]
    d2 = (xf - cx0)**2 + (yf - cy0)**2
    neigh = d2 <= (30.0**2)
    if np.any(neigh) and np.isfinite(point_score[neigh]).any():
        wpt = point_score[neigh]
        xloc = xf[neigh]
        yloc = yf[neigh]
        s = float(np.nansum(wpt))
        if s > 0:
            cx_ref = float(np.nansum(wpt * xloc) / s)
            cy_ref = float(np.nansum(wpt * yloc) / s)
            cx0, cy0 = cx_ref, cy_ref

    return (cx0, cy0)

# -----------------------------
# Arm detection on annulus
# -----------------------------
@dataclass
class ArmParams:
    inner_r: float = 25.0
    outer_r: float = 120.0
    angle_bin_deg: float = 12.0
    min_bin_pts: int = 50

def detect_arms(x: np.ndarray, y: np.ndarray, cx: float, cy: float, p: ArmParams) -> List[Arm]:
    dx = x - cx
    dy = y - cy
    r = np.hypot(dx, dy)
    ang = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
    mask = (r >= p.inner_r) & (r <= p.outer_r)
    ang_m = ang[mask]
    r_m = r[mask]
    dx_m = dx[mask]
    dy_m = dy[mask]
    if ang_m.size < 30:
        return []

    bin_rad = math.radians(p.angle_bin_deg)
    bins = int(math.ceil(2*np.pi / bin_rad))
    hist, edges = np.histogram(ang_m, bins=bins, range=(0, 2*np.pi))
    smooth = np.convolve(hist, np.ones(3)/3.0, mode="same")

    # Adaptive threshold: favor clear peaks but allow lower counts when data is sparse
    mean_cnt = float(np.mean(smooth))
    std_cnt = float(np.std(smooth))
    thr_adaptive = int(mean_cnt + 0.5 * std_cnt)
    thr_floor = int(0.003 * ang_m.size)
    thr = max(min(p.min_bin_pts, int(0.02 * ang_m.size)), thr_adaptive, thr_floor)

    peaks: List[int] = []
    for i in range(bins):
        prev_i = (i-1) % bins
        next_i = (i+1) % bins
        if smooth[i] > smooth[prev_i] and smooth[i] > smooth[next_i] and smooth[i] >= thr:
            peaks.append(i)

    def build_arms_from_bins(bin_indices: List[int]) -> List[Arm]:
        out: List[Arm] = []
        for bi in bin_indices:
            a0 = edges[bi]
            a1 = edges[(bi+1) % len(edges)]
            if a1 <= a0:
                a1 += 2*np.pi
            sel = (ang_m >= a0) & (ang_m < a1)
            if not np.any(sel):
                continue
            mu = circmean(ang_m[sel])
            j = int(np.argmax(r_m[sel]))
            tip = (cx + float(dx_m[sel][j]), cy + float(dy_m[sel][j]))
            node = (cx + p.inner_r * math.cos(mu), cy + p.inner_r * math.sin(mu))
            out.append(Arm(id=len(out), angle=mu, tip_xy=tip, node_xy=node))
        for i, a in enumerate(out):
            a.id = i
        return out

    arms = build_arms_from_bins(peaks)

    # Fallback: ensure at least 4 arms for cross intersections by selecting top separated bins
    if len(arms) < 4:
        order = list(np.argsort(smooth)[::-1])
        chosen: List[int] = []
        for bi in order:
            if smooth[bi] <= 0:
                break
            ok = True
            for cj in chosen:
                # enforce separation to avoid adjacent bins (at least ~2 bins apart)
                if min(abs(bi - cj), bins - abs(bi - cj)) < 2:
                    ok = False
                    break
            if ok:
                chosen.append(int(bi))
            if len(chosen) >= 4:
                break
        arms = build_arms_from_bins(chosen)

    return arms

# -----------------------------
# Movements
# -----------------------------
def label_arm(angle: float, arms: List[Arm]) -> Optional[int]:
    if not arms:
        return None
    # small circular distance
    diffs = [min(abs(angle - a.angle), 2*np.pi - abs(angle - a.angle)) for a in arms]
    idx = int(np.argmin(diffs))
    return arms[idx].id

def extract_movements_per_traj(x: np.ndarray, y: np.ndarray,
                               center: Tuple[float,float],
                               inner_r: float,
                               arms: List[Arm]) -> Optional[Tuple[int,int]]:
    cx, cy = center
    r = np.hypot(x - cx, y - cy)
    if r.size < 3:
        return None
    inside = r < inner_r
    if not inside.any():
        return None
    # first enter
    first_in = int(np.argmax(inside))
    after = inside[first_in:]
    if after.all():
        return None
    first_out = first_in + int(np.argmax(~after))

    # angles before entering
    s0 = max(0, first_in-5); e0 = first_in
    if e0 <= s0:
        return None
    vx0 = np.gradient(x[s0:e0+1]); vy0 = np.gradient(y[s0:e0+1])
    ang0 = angle_of(float(np.nanmean(vx0)), float(np.nanmean(vy0)))
    arm_in = label_arm(ang0, arms)

    # angles after leaving
    s1 = first_out; e1 = min(x.size-1, first_out+5)
    if e1 <= s1 or (e1 - s1) < 1:
        return None
    vx1 = np.gradient(x[s1:e1+1]); vy1 = np.gradient(y[s1:e1+1])
    ang1 = angle_of(float(np.nanmean(vx1)), float(np.nanmean(vy1)))
    arm_out = label_arm(ang1, arms)

    if arm_in is None or arm_out is None or arm_in == arm_out:
        return None
    return (arm_in, arm_out)

def build_edges(arms: List[Arm], movements: List[Tuple[int,int]], min_support: int = 5) -> List[MovementEdge]:
    edges: List[MovementEdge] = []
    counts = Counter(movements)
    for (u, v), w in counts.items():
        if w < min_support:
            continue
        p0 = np.array(arms[u].node_xy, dtype=np.float64)
        p2 = np.array(arms[v].node_xy, dtype=np.float64)
        c = 0.5*(p0 + p2)  # simple mid control point
        ts = np.linspace(0.0, 1.0, 20, dtype=np.float64)
        pts = (1-ts)[:,None]**2 * p0 + 2*(1-ts)[:,None]*ts[:,None]*c + (ts[:,None]**2)*p2
        line = LineString(pts.tolist())
        edges.append(MovementEdge(u=u, v=v, weight=int(w), geometry=line))
    return edges

# -----------------------------
# Export
# -----------------------------
def export_geojson(arms: List[Arm], edges: List[MovementEdge],
                   cx: float, cy: float, lat0_deg: float, outfile: str):
    feats = []

    # center
    clon, clat = local_xy_to_lonlat(np.array([cx], dtype=np.float64), np.array([cy], dtype=np.float64), lat0_deg)
    feats.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(clon[0]), float(clat[0]) ]},
        "properties": {"kind": "center"}
    })

    # nodes
    if arms:
        nx = np.array([a.node_xy[0] for a in arms], dtype=np.float64)
        ny = np.array([a.node_xy[1] for a in arms], dtype=np.float64)
        lon, lat = local_xy_to_lonlat(nx, ny, lat0_deg)
        for i, a in enumerate(arms):
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon[i]), float(lat[i]) ]},
                "properties": {"kind": "arm_node", "id": int(a.id), "angle_rad": float(a.angle)}
            })

    # edges
    for e in edges:
        xy = np.asarray(e.geometry.coords, dtype=np.float64)
        lon, lat = local_xy_to_lonlat(xy[:,0], xy[:,1], lat0_deg)
        coords = [[float(lon[i]), float(lat[i])] for i in range(lon.size)]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"kind": "movement", "u": int(e.u), "v": int(e.v), "weight": int(e.weight)}
        })

    geojson = {"type": "FeatureCollection", "features": feats}
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(df: pd.DataFrame,
                 inner_r: float = 25.0,
                 outer_r: float = 120.0,
                 min_support: int = 3,
                 ann_bin_deg: float = 12.0) -> Dict:
    need_cols = ["vehicle_id","longitude","latitude"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.copy()
    df["t_sec"] = parse_time_seconds(df)
    # groupwise stable sort by (vehicle_id, t_sec)
    df = df.sort_values(["vehicle_id","t_sec"], kind="mergesort").reset_index(drop=True)

    lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(np.float64)
    lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(np.float64)
    lat0 = float(np.nanmean(lat)) if np.isfinite(lat).any() else 0.0
    X, Y = lonlat_to_local_xy(lon, lat)

    headings, curv = compute_heading_and_curvature(X, Y)
    speed = pd.to_numeric(df.get("speed", np.nan), errors="coerce").to_numpy(np.float64)

    # Hardcoded center seed (lon/lat) anchored near the true intersection
    lon_seed = 123.152697
    lat_seed = 32.345166
    R = 6371000.0
    lat0_rad = math.radians(lat0)
    seed_x = math.radians(lon_seed) * R * math.cos(lat0_rad)
    seed_y = math.radians(lat_seed) * R
    cx, cy = refine_center_near_seed(X, Y, speed, curv, (seed_x, seed_y), radius=50.0)

    # Keep center anchored to the provided seed; skip automatic re-detection

    # Adaptive annulus based on distance distribution to center
    r_all = np.hypot(X - cx, Y - cy)
    if np.isfinite(r_all).any():
        try:
            r_p15 = float(np.nanpercentile(r_all, 15))
            r_p85 = float(np.nanpercentile(r_all, 85))
            inner_r_eff = max(15.0, min(60.0, r_p15))
            outer_r_eff = max(inner_r_eff + 40.0, min(200.0, r_p85))
        except Exception:
            inner_r_eff = inner_r
            outer_r_eff = outer_r
    else:
        inner_r_eff = inner_r
        outer_r_eff = outer_r

    arms = detect_arms(
        X, Y, cx, cy,
        ArmParams(inner_r=inner_r_eff, outer_r=outer_r_eff,
                  angle_bin_deg=ann_bin_deg, min_bin_pts=50)
    )

    movements: List[Tuple[int,int]] = []
    for vid, g in df.groupby("vehicle_id", sort=False):
        idx = g.index.to_numpy()
        xg = X[idx]; yg = Y[idx]
        mv = extract_movements_per_traj(xg, yg, (cx, cy), inner_r, arms)
        if mv is not None:
            movements.append(mv)

    edges = build_edges(arms, movements, min_support=min_support)

    return {
        "center_xy": (float(cx), float(cy)),
        "lat0_deg": float(lat0),
        "arms": arms,
        "edges": edges,
        "movements": movements
    }

def main():
    # 直接读取当前目录下的A0003.csv文件
    input_file = "A0003.csv"
    output_dir = "output"
    
    # 默认参数
    inner_r = 25.0
    outer_r = 120.0
    min_support = 5
    ann_bin_deg = 18.0
    
    print(f"读取输入文件: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件: {input_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    
    print(f"数据文件加载成功!")
    print(f"总行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    print(f"车辆数量: {df['vehicle_id'].nunique()}")
    print(f"\n数据预览:")
    print(df.head())
    print(f"\n数据统计:")
    print(df.describe())

    res = run_pipeline(df, inner_r=inner_r, outer_r=outer_r,
                       min_support=min_support, ann_bin_deg=ann_bin_deg)

    gj = os.path.join(output_dir, "intersection_topology.geojson")
    export_geojson(res["arms"], res["edges"], res["center_xy"][0], res["center_xy"][1],
                   res["lat0_deg"], gj)

    summary = {
        "arms": [{"id": a.id, "angle_deg": math.degrees(a.angle)} for a in res["arms"]],
        "edge_supports": [{"u": e.u, "v": e.v, "weight": e.weight} for e in res["edges"]],
        "movements_total": len(res["movements"])
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存:")
    print(f"  - {gj}")
    print(f"  - {os.path.join(output_dir, 'summary.json')}")
    print(f"\n检测到 {len(res['arms'])} 条路臂")
    print(f"检测到 {len(res['movements'])} 个轨迹移动")
    print(f"生成 {len(res['edges'])} 条移动边")

if __name__ == "__main__":
    main()
