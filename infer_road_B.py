
# -*- coding: utf-8 -*-
"""
B 向（南北向）车道提取与分析双输出：
- 仅使用方向 B1-1 / B1-2 的轨迹
- 计算分界线（PCA-1D + 沿程分箱 17m + t 轴双峰）
- 侧别按东西（EAST/WEST），用 +1.75m 偏移的经度差确定 “+” 是东还是西
- 生成两文件：
  1) intersection_B.json：分界线 + 东/西各两条直行车道中轴（共 4 条），偏移为 1.75/5.25m
  2) analyse_B.json：统计信息（|t| 直方图、峰位置、推断车道数、参数）, 便于后续画图

数据与输出目录（固定）：/home/tzhang174/EVData_XGame/data/
"""

import os, csv, json, math
from typing import Any, Dict, List, Tuple
import numpy as np

# ---------------- Config ----------------
DATA_DIR      = "/home/tzhang174/EVData_XGame/data"
INPUT_FILES   = [("A0003_refined.csv", "A0003"), ("A0008_refined.csv", "A0008")]
OUT_INTER     = os.path.join(DATA_DIR, "intersection_B.json")
OUT_ANALYSE   = os.path.join(DATA_DIR, "analyse_B.json")

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

# ---------------- Peak helpers ----------------
def _gauss_smooth_1d(y: np.ndarray, sigma_bins=1.5, radius=4):
    if len(y) == 0: return y
    r = int(max(1, np.ceil(radius*sigma_bins)))
    xs = np.arange(-r, r+1)
    ker = np.exp(-0.5*(xs/sigma_bins)**2); ker /= ker.sum()
    pad = (len(ker)-1)//2
    yp  = np.pad(y, (pad,pad), mode='edge')
    return np.convolve(yp, ker, mode='valid')

def _two_peaks_in_bin(tvals: np.ndarray, nbins=64, min_sep=3.2, sigma_bins=1.5):
    if len(tvals) < 5: return None
    tmin, tmax = np.quantile(tvals, 0.02), np.quantile(tvals, 0.98)
    if not np.isfinite(tmin) or not np.isfinite(tmax): return None
    if tmax <= tmin: tmax = tmin + 1e-3
    H, edges = np.histogram(tvals, bins=nbins, range=(tmin, tmax))
    Hs = _gauss_smooth_1d(H, sigma_bins=sigma_bins)
    peaks = []
    for i in range(1, len(Hs)-1):
        if Hs[i] >= Hs[i-1] and Hs[i] >= Hs[i+1] and Hs[i] > 0:
            t_center = 0.5*(edges[i] + edges[i+1])
            peaks.append((Hs[i], t_center))
    if not peaks: return None
    peaks.sort(reverse=True, key=lambda z: z[0])
    top = []
    for h, tc in peaks:
        if not top or all(abs(tc - tc2) >= min_sep for _, tc2 in top):
            top.append((h, tc))
        if len(top) == 2: break
    if len(top) == 1: return (top[0][1], None)
    t1 = min(top[0][1], top[1][1]); t2 = max(top[0][1], top[1][1])
    return (t1, t2)

def _k_peaks_1d(values: np.ndarray, nbins=64, min_sep=3.2, sigma_bins=1.5, kmax=4, min_rel_height=0.08):
    if values.size < 5: return []
    vmin, vmax = np.quantile(values, 0.02), np.quantile(values, 0.98)
    if not np.isfinite(vmin) or not np.isfinite(vmax): return []
    if vmax <= vmin: vmax = vmin + 1e-3
    H, edges = np.histogram(values, bins=nbins, range=(vmin, vmax))
    Hs = _gauss_smooth_1d(H, sigma_bins=sigma_bins)
    Hmax = float(np.max(Hs)) if Hs.size>0 else 0.0
    if Hmax <= 0: return []
    peaks = []
    for i in range(1, len(Hs)-1):
        if Hs[i] >= Hs[i-1] and Hs[i] >= Hs[i+1] and Hs[i] >= min_rel_height*Hmax:
            t_center = 0.5*(edges[i] + edges[i+1])
            peaks.append((Hs[i], t_center))
    if not peaks: return []
    peaks.sort(reverse=True, key=lambda z: z[0])
    selected = []
    for h, tc in peaks:
        if all(abs(tc - s) >= min_sep for s in selected):
            selected.append(tc)
        if len(selected) >= kmax: break
    selected.sort()
    return selected

# ---------------- Divider ----------------
def _lane_divider_st_and_xy(s: np.ndarray, t: np.ndarray, u: np.ndarray, v: np.ndarray, mu: np.ndarray):
    bin_size_m=BIN_SIZE_M; min_pts_per_bin=MIN_PTS_PER_BIN; nbins_t=NBINS_T
    min_lane_sep_m=MIN_LANE_SEP_M; sigma_bins=SIGMA_BINS; smooth_win=SMOOTH_WIN
    if s.size == 0: return np.array([]), np.array([]), np.array([]), np.array([])
    smin, smax = float(np.min(s)), float(np.max(s))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax - smin < 1e-9:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.arange(smin, smax + bin_size_m, bin_size_m)
    idx   = np.digitize(s, edges) - 1
    T_div_bins = np.full(len(edges)-1, np.nan)
    for b in range(len(edges)-1):
        m = (idx == b)
        if np.sum(m) < min_pts_per_bin: continue
        tp = _two_peaks_in_bin(t[m], nbins=nbins_t, min_sep=min_lane_sep_m, sigma_bins=sigma_bins)
        if tp is None: continue
        t1, t2 = tp
        t_div = t1 if t2 is None else 0.5*(t1 + t2)
        T_div_bins[b] = float(t_div)
    if np.all(np.isnan(T_div_bins)): return np.array([]), np.array([]), np.array([]), np.array([])
    idx_bins = np.arange(len(T_div_bins)); good = ~np.isnan(T_div_bins)
    if np.sum(good) >= 2: T_div_bins = np.interp(idx_bins, idx_bins[good], T_div_bins[good])
    elif np.sum(good) == 1: T_div_bins[:] = T_div_bins[good][0]
    S_centers = 0.5*(edges[:-1] + edges[1:])
    T_arr = T_div_bins.copy()
    if smooth_win and smooth_win > 1:
        if smooth_win % 2 == 0: smooth_win += 1
        pad = smooth_win//2; ker = np.ones(smooth_win)/smooth_win
        Tp  = np.pad(T_arr, (pad,pad), mode='edge')
        T_arr = np.convolve(Tp, ker, mode='valid')
    C = mu + S_centers[:,None]*u + T_arr[:,None]*v
    cx, cy = C[:,0], C[:,1]
    return S_centers, T_arr, cx, cy

# ---------------- IO ----------------
def load_directions(path: str) -> set:
    allowed = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("direction","").strip()
            if d in {"B1-1","B1-2"}:
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
            if row.get("road_id","").strip() != expect_road: continue
            key = (
                row.get("vehicle_id","").strip(),
                row.get("date","").strip(),
                row.get("seg_id","").strip(),
                row.get("road_id","").strip(),
            )
            if key not in allowed: continue
            try:
                lon = float(row.get("longitude","")); lat = float(row.get("latitude",""))
            except ValueError:
                continue
            if math.isfinite(lon) and math.isfinite(lat):
                lons.append(lon); lats.append(lat)
    return np.array(lons, float), np.array(lats, float)

# ---------------- Normals & Offsets ----------------
def _compute_normals(cx: np.ndarray, cy: np.ndarray):
    n = len(cx)
    if n < 2: return np.array([]), np.array([])
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
    if nx.size == 0: return np.array([]), np.array([])
    return cx + d*nx, cy + d*ny

# ---------------- Main per-road ----------------
def process_one(points_csv: str, road_id: str, allowed: set):
    lons, lats = load_points(points_csv, allowed, road_id)
    if lons.size < 10:
        return (
            {
                "lane_divider": [],
                "east": {"lanes": {"east_lane1": [], "east_lane2": []}},
                "west": {"lanes": {"west_lane1": [], "west_lane2": []}}
            },
            {
                "all":   {"abs_t_hist_edges_m": [], "abs_t_hist_counts": []},
                "east":  {"abs_t_hist_edges_m": [], "abs_t_hist_counts": [], "abs_t_peaks_m": [], "likely_lane_count": 0},
                "west":  {"abs_t_hist_edges_m": [], "abs_t_hist_counts": [], "abs_t_peaks_m": [], "likely_lane_count": 0},
                "params": params_dict()
            }
        )

    x, y, lon0, lat0 = lonlat_to_local_xy(lons, lats)
    X  = np.column_stack([x, y])
    u, v, mu = pca_first_axis(X)
    Y  = X - mu
    s  = Y @ u
    t  = Y @ v

    # Divider in s,t and XY
    S_centers, T_div_s, cx, cy = _lane_divider_st_and_xy(s, t, u, v, mu)
    if cx.size < 2:
        return (
            {
                "lane_divider": [],
                "east": {"lanes": {"east_lane1": [], "east_lane2": []}},
                "west": {"lanes": {"west_lane1": [], "west_lane2": []}}
            },
            {
                "all":   {"abs_t_hist_edges_m": [], "abs_t_hist_counts": []},
                "east":  {"abs_t_hist_edges_m": [], "abs_t_hist_counts": [], "abs_t_peaks_m": [], "likely_lane_count": 0},
                "west":  {"abs_t_hist_edges_m": [], "abs_t_hist_counts": [], "abs_t_peaks_m": [], "likely_lane_count": 0},
                "params": params_dict()
            }
        )

    # EAST/WEST mapping by longitude using +1.75 m test offset
    ox_test, oy_test = _offset_polyline(cx, cy, HALF)
    test_lon, _ = local_xy_to_lonlat(ox_test, oy_test, lon0, lat0)
    div_lon,  _ = local_xy_to_lonlat(cx, cy, lon0, lat0)
    import numpy as _np
    east_sign = +1.0 if _np.median(test_lon - div_lon) > 0.0 else -1.0

    # Interpolate divider t(s) for each point, compute lateral offset
    T_interp = np.interp(s, S_centers, T_div_s)
    t_rel = t - T_interp

    # EAST if east_sign * t_rel >= 0 ; WEST otherwise
    east_mask = (east_sign * t_rel) >= 0.0
    west_mask = ~east_mask

    abs_t_all  = np.abs(t_rel)
    abs_t_east = np.abs(t_rel[east_mask])
    abs_t_west = np.abs(t_rel[west_mask])

    def _hist_pack(vals: np.ndarray, nbins=60):
        if vals.size == 0: return [], []
        vmax = float(np.quantile(vals, 0.995))
        if vmax <= 0: vmax = float(np.max(vals)) if vals.size>0 else 1.0
        if vmax <= 0: vmax = 1.0
        H, edges = np.histogram(vals, bins=nbins, range=(0.0, vmax))
        return edges.tolist(), H.tolist()

    all_edges,  all_counts   = _hist_pack(abs_t_all)
    east_edges, east_counts  = _hist_pack(abs_t_east)
    west_edges, west_counts  = _hist_pack(abs_t_west)

    east_peaks = _k_peaks_1d(abs_t_east, nbins=64, min_sep=LANE_WIDTH*0.9, sigma_bins=1.5, kmax=4, min_rel_height=0.08)
    west_peaks = _k_peaks_1d(abs_t_west, nbins=64, min_sep=LANE_WIDTH*0.9, sigma_bins=1.5, kmax=4, min_rel_height=0.08)

    # --- Build intersection_B.json content ---
    divider_lon, divider_lat = local_xy_to_lonlat(cx, cy, lon0, lat0)
    inter = {
        "lane_divider": [[float(a), float(b)] for a,b in zip(divider_lon.tolist(), divider_lat.tolist())],
        "east": {"lanes": {}},
        "west": {"lanes": {}}
    }
    east_offsets = [ east_sign*(HALF), east_sign*(HALF + LANE_WIDTH) ]      # +1.75, +5.25 toward EAST
    west_offsets = [ -east_sign*(HALF), -east_sign*(HALF + LANE_WIDTH) ]    # -1.75, -5.25 toward WEST

    for i, d in enumerate(east_offsets, start=1):
        ox, oy = _offset_polyline(cx, cy, d)
        lon, lat = local_xy_to_lonlat(ox, oy, lon0, lat0)
        inter["east"]["lanes"][f"east_lane{i}"] = [[float(a), float(b)] for a,b in zip(lon.tolist(), lat.tolist())]
    for i, d in enumerate(west_offsets, start=1):
        ox, oy = _offset_polyline(cx, cy, d)
        lon, lat = local_xy_to_lonlat(ox, oy, lon0, lat0)
        inter["west"]["lanes"][f"west_lane{i}"] = [[float(a), float(b)] for a,b in zip(lon.tolist(), lat.tolist())]

    # --- Build analyse_B.json content ---
    analyse = {
        "all":   {"abs_t_hist_edges_m": all_edges,  "abs_t_hist_counts": all_counts},
        "east":  {"abs_t_hist_edges_m": east_edges, "abs_t_hist_counts": east_counts,
                  "abs_t_peaks_m": [float(x) for x in east_peaks], "likely_lane_count": int(len(east_peaks))},
        "west":  {"abs_t_hist_edges_m": west_edges, "abs_t_hist_counts": west_counts,
                  "abs_t_peaks_m": [float(x) for x in west_peaks], "likely_lane_count": int(len(west_peaks))},
        "params": params_dict(),
        "notes": [
            "B1-1/B1-2 方向：南北向，左右按东西侧划分",
            "分界线：PCA-1D + s分箱(17m) + t双峰中点 + 插值平滑",
            "侧别确定：用 +1.75m 偏移与分界线的经度比较，median(lon_offset - lon_divider)>0 则 '+' 为 EAST",
            "直行数估计：在各侧 |t| 分布上做多峰检测，最小峰距≈3.5m；峰数即直行条数的强信号"
        ]
    }
    return inter, analyse

def params_dict():
    return {
        "bin_size_m": BIN_SIZE_M,
        "min_pts_per_bin": MIN_PTS_PER_BIN,
        "nbins_t": NBINS_T,
        "min_lane_sep_m": MIN_LANE_SEP_M,
        "sigma_bins": SIGMA_BINS,
        "smooth_win": SMOOTH_WIN,
        "lane_width": LANE_WIDTH,
        "half_lane": HALF
    }

def main():
    direction_csv = os.path.join(DATA_DIR, "direction.csv")
    allowed = load_directions(direction_csv)

    out_inter = {}
    out_analyse = {}

    for fname, road in INPUT_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            out_inter[road] = {
                "lane_divider": [],
                "east": {"lanes": {"east_lane1": [], "east_lane2": []}},
                "west": {"lanes": {"west_lane1": [], "west_lane2": []}}
            }
            out_analyse[road] = {
                "all": {"abs_t_hist_edges_m": [], "abs_t_hist_counts": []},
                "east": {"abs_t_hist_edges_m": [], "abs_t_hist_counts": [], "abs_t_peaks_m": [], "likely_lane_count": 0},
                "west": {"abs_t_hist_edges_m": [], "abs_t_hist_counts": [], "abs_t_peaks_m": [], "likely_lane_count": 0},
                "params": params_dict(),
                "notes": ["file not found"]
            }
            continue

        inter, analyse = process_one(path, road, allowed)
        out_inter[road] = inter
        out_analyse[road] = analyse

    with open(OUT_INTER, "w", encoding="utf-8") as f:
        json.dump(out_inter, f, ensure_ascii=False, indent=2)
    with open(OUT_ANALYSE, "w", encoding="utf-8") as f:
        json.dump(out_analyse, f, ensure_ascii=False, indent=2)
    print("Wrote:", OUT_INTER)
    print("Wrote:", OUT_ANALYSE)

if __name__ == "__main__":
    main()
