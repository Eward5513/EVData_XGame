
# -*- coding: utf-8 -*-
"""
Fixed 3-vs-2 straight lanes with robust geographic-north labeling.

- Data dir (fixed): /home/tzhang174/EVData_XGame/data
- Files: direction.csv, A0003_refined.csv, A0008_refined.csv
- Filter: only A1-1 / A1-2
- Divider: PCA-1D -> along-path binning (bin_size_m=17) -> per-bin two-peak on t -> midpoint -> interpolate + smooth
- Lanes: north side = 3 centers at +{1.75, 5.25, 8.75} m, south side = 2 centers at -{1.75, 5.25} m
- Robust "north" test: build a +half (1.75 m) offset of divider, convert to lon/lat, check median(lat_offset - lat_divider);
  if negative, invert the sign so that "north" is the side with higher latitude.
- Output JSON: /home/tzhang174/EVData_XGame/data/intersection.json
"""

import os, csv, json, math
from typing import Any, Dict, List, Tuple
import numpy as np

# ---------------- Configuration ----------------
DATA_DIR      = "/home/tzhang174/EVData_XGame/data"
INPUT_FILES   = [("A0003_refined.csv", "A0003"), ("A0008_refined.csv", "A0008")]
OUTPUT_JSON   = os.path.join(DATA_DIR, "intersection.json")

BIN_SIZE_M        = 17.0
MIN_PTS_PER_BIN   = 25
NBINS_T           = 64
MIN_LANE_SEP_M    = 3.2
SIGMA_BINS        = 1.5
SMOOTH_WIN        = 7
LANE_WIDTH        = 3.5
HALF              = LANE_WIDTH / 2.0  # 1.75 m

# ---------------- Geometry helpers ----------------
def lonlat_to_local_xy(lon, lat, lon0=None, lat0=None):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    if lon0 is None: lon0 = float(np.nanmean(lon))
    if lat0 is None: lat0 = float(np.nanmean(lat))
    R = 6371000.0
    x = R*np.cos(np.deg2rad(lat0))*np.deg2rad(lon - lon0)
    y = R*np.deg2rad(lat - lat0)
    return x, y, lon0, lat0

def local_xy_to_lonlat(x, y, lon0, lat0):
    R = 6371000.0
    lon = lon0 + np.rad2deg(x/(R*np.cos(np.deg2rad(lat0))))
    lat = lat0 + np.rad2deg(y/R)
    return lon, lat

def pca_first_axis(X: np.ndarray):
    mu = X.mean(axis=0)
    Y  = X - mu
    C  = (Y.T @ Y) / max(len(X)-1, 1)
    w, V = np.linalg.eigh(C)
    u = V[:, np.argmax(w)]; u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    return u, v, mu

# ------------- Per-bin two-peak detection -------------
def _gauss_smooth_1d(y: np.ndarray, sigma_bins=1.5, radius=4):
    if len(y) == 0: return y
    r = int(max(1, np.ceil(radius*sigma_bins)))
    xs = np.arange(-r, r+1)
    ker = np.exp(-0.5*(xs/sigma_bins)**2); ker /= ker.sum()
    pad = (len(ker)-1)//2
    yp  = np.pad(y, (pad,pad), mode='edge')
    return np.convolve(yp, ker, mode='valid')

def _two_peaks_in_bin(tvals: np.ndarray, nbins=64, min_sep=3.2, sigma_bins=1.5):
    """
    Return (t_low, t_up); if only one peak found, returns (t_single, None);
    else returns None.
    """
    if len(tvals) < 5:
        return None
    tmin, tmax = np.quantile(tvals, 0.02), np.quantile(tvals, 0.98)
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        return None
    if tmax <= tmin: tmax = tmin + 1e-3
    H, edges = np.histogram(tvals, bins=nbins, range=(tmin, tmax))
    Hs = _gauss_smooth_1d(H, sigma_bins=sigma_bins)

    peaks = []
    for i in range(1, len(Hs)-1):
        if Hs[i] >= Hs[i-1] and Hs[i] >= Hs[i+1] and Hs[i] > 0:
            t_center = 0.5*(edges[i] + edges[i+1])
            peaks.append((Hs[i], t_center))
    if not peaks:
        return None
    peaks.sort(reverse=True, key=lambda z: z[0])

    top = []
    for h, tc in peaks:
        if not top or all(abs(tc - tc2) >= min_sep for _, tc2 in top):
            top.append((h, tc))
        if len(top) == 2:
            break

    if len(top) == 1:
        return (top[0][1], None)
    t1 = min(top[0][1], top[1][1]); t2 = max(top[0][1], top[1][1])
    return (t1, t2)

# ------------- Divider from (s,t) -------------
def _lane_divider_xy_from_st(s: np.ndarray, t: np.ndarray,
                             u: np.ndarray, v: np.ndarray, mu: np.ndarray,
                             bin_size_m=BIN_SIZE_M, min_pts_per_bin=MIN_PTS_PER_BIN, nbins_t=NBINS_T,
                             min_lane_sep_m=MIN_LANE_SEP_M, sigma_bins=SIGMA_BINS, smooth_win=SMOOTH_WIN):
    if s.size == 0: return np.array([]), np.array([])
    smin, smax = float(np.min(s)), float(np.max(s))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax - smin < 1e-9:
        return np.array([]), np.array([])
    edges = np.arange(smin, smax + bin_size_m, bin_size_m)
    idx   = np.digitize(s, edges) - 1

    T_div_bins = np.full(len(edges)-1, np.nan)

    for b in range(len(edges)-1):
        m = (idx == b)
        if np.sum(m) < min_pts_per_bin:
            continue
        tp = _two_peaks_in_bin(t[m], nbins=nbins_t, min_sep=min_lane_sep_m, sigma_bins=sigma_bins)
        if tp is None:
            continue
        t1, t2 = tp
        if t2 is None:
            t_div = t1
        else:
            t_div = 0.5*(t1 + t2)
        T_div_bins[b] = float(t_div)

    if np.all(np.isnan(T_div_bins)):
        return np.array([]), np.array([])

    idx_bins = np.arange(len(T_div_bins))
    good = ~np.isnan(T_div_bins)
    if np.sum(good) >= 2:
        T_div_bins = np.interp(idx_bins, idx_bins[good], T_div_bins[good])
    elif np.sum(good) == 1:
        T_div_bins[:] = T_div_bins[good][0]

    S_centers = 0.5*(edges[:-1] + edges[1:])
    T_arr = T_div_bins.copy()
    if smooth_win and smooth_win > 1:
        if smooth_win % 2 == 0: smooth_win += 1
        pad = smooth_win//2
        ker = np.ones(smooth_win)/smooth_win
        Tp  = np.pad(T_arr, (pad,pad), mode='edge')
        T_arr = np.convolve(Tp, ker, mode='valid')

    C = mu + S_centers[:,None]*u + T_arr[:,None]*v
    return C[:,0], C[:,1]

# ------------- Normals and offsets -------------
def _compute_normals(cx: np.ndarray, cy: np.ndarray):
    n = len(cx)
    if n < 2:
        return np.array([]), np.array([])
    tx = np.zeros(n); ty = np.zeros(n)
    dx = np.diff(cx); dy = np.diff(cy)
    tx[0], ty[0] = dx[0], dy[0]
    tx[-1], ty[-1] = dx[-1], dy[-1]
    if n > 2:
        tx[1:-1] = (cx[2:] - cx[:-2]) * 0.5
        ty[1:-1] = (cy[2:] - cy[:-2]) * 0.5
    L = np.hypot(tx, ty); L[L==0]=1.0
    tx/=L; ty/=L
    nx, ny = -ty, tx
    NL = np.hypot(nx, ny); NL[NL==0]=1.0
    nx/=NL; ny/=NL
    return nx, ny

def _offset_polyline(cx: np.ndarray, cy: np.ndarray, d: float):
    nx, ny = _compute_normals(cx, cy)
    if nx.size == 0:
        return np.array([]), np.array([])
    return cx + d*nx, cy + d*ny

# ------------- IO helpers -------------
def load_directions(path: str) -> set:
    allowed = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("direction","").strip()
            if d in {"A1-1","A1-2"}:
                key = (
                    row.get("vehicle_id","").strip(),
                    row.get("date","").strip(),
                    row.get("seg_id","").strip(),
                    row.get("road_id","").strip(),
                )
                allowed.add(key)
    return allowed

def load_points(path: str, allowed: set, expect_road: str):
    lons, lats = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("road_id","").strip() != expect_road:
                continue
            key = (
                row.get("vehicle_id","").strip(),
                row.get("date","").strip(),
                row.get("seg_id","").strip(),
                row.get("road_id","").strip(),
            )
            if key not in allowed:
                continue
            try:
                lon = float(row.get("longitude",""))
                lat = float(row.get("latitude",""))
            except ValueError:
                continue
            if math.isfinite(lon) and math.isfinite(lat):
                lons.append(lon); lats.append(lat)
    return np.array(lons, float), np.array(lats, float)

# ------------- Main per-road pipeline -------------
def process_one(points_csv: str, road_id: str, allowed: set):
    lons, lats = load_points(points_csv, allowed, road_id)
    if lons.size < 10:
        return {
            "lane_divider": [],
            "north": {"lanes": {"north_lane1": [], "north_lane2": [], "north_lane3": []}},
            "south": {"lanes": {"south_lane1": [], "south_lane2": []}}
        }

    x, y, lon0, lat0 = lonlat_to_local_xy(lons, lats)
    X  = np.column_stack([x, y])
    u, v, mu = pca_first_axis(X)
    Y  = X - mu
    s  = Y @ u
    t  = Y @ v

    # Divider in local XY
    cx, cy = _lane_divider_xy_from_st(s, t, u, v, mu)
    if cx.size < 2:
        return {
            "lane_divider": [],
            "north": {"lanes": {"north_lane1": [], "north_lane2": [], "north_lane3": []}},
            "south": {"lanes": {"south_lane1": [], "south_lane2": []}}
        }

    # --- Robust geographic-north test using latitudes ---
    # Build a +half offset (mathematical + side), compare latitude vs divider
    ox_test, oy_test = _offset_polyline(cx, cy, HALF)
    test_lon, test_lat = local_xy_to_lonlat(ox_test, oy_test, lon0, lat0)
    div_lon,  div_lat  = local_xy_to_lonlat(cx, cy, lon0, lat0)
    # If median(test_lat - div_lat) < 0, then mathematical '+' points to geographic south;
    # we should invert signs so that '+' corresponds to true north.
    import numpy as _np
    sign = +1.0
    if _np.median(test_lat - div_lat) < 0.0:
        sign = -1.0

    # Offsets: fixed counts (north=3, south=2) with corrected sign so that 'north' is geographic north
    north_offsets = [ sign*(HALF + k*LANE_WIDTH) for k in range(3) ]   # +1.75,+5.25,+8.75 toward geographic north
    south_offsets = [-sign*(HALF + k*LANE_WIDTH) for k in range(2) ]   # -1.75,-5.25 toward geographic south

    # Divider to lon/lat (already computed as div_lon/div_lat)
    out = {
        "lane_divider": [[float(a), float(b)] for a,b in zip(div_lon.tolist(), div_lat.tolist())],
        "north": {"lanes": {}},
        "south": {"lanes": {}}
    }

    # Build offset lanes
    for i, d in enumerate(north_offsets, start=1):
        ox, oy = _offset_polyline(cx, cy, d)
        lon, lat = local_xy_to_lonlat(ox, oy, lon0, lat0)
        out["north"]["lanes"][f"north_lane{i}"] = [[float(a), float(b)] for a,b in zip(lon.tolist(), lat.tolist())]
    for i, d in enumerate(south_offsets, start=1):
        ox, oy = _offset_polyline(cx, cy, d)
        lon, lat = local_xy_to_lonlat(ox, oy, lon0, lat0)
        out["south"]["lanes"][f"south_lane{i}"] = [[float(a), float(b)] for a,b in zip(lon.tolist(), lat.tolist())]

    return out

def main():
    direction_csv = os.path.join(DATA_DIR, "direction.csv")
    allowed = load_directions(direction_csv)

    result = {}
    for fname, road in INPUT_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            result[road] = {
                "lane_divider": [],
                "north": {"lanes": {"north_lane1": [], "north_lane2": [], "north_lane3": []}},
                "south": {"lanes": {"south_lane1": [], "south_lane2": []}}
            }
            continue
        result[road] = process_one(path, road, allowed)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
