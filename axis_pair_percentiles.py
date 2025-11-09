#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

# =============== 硬编码路径 ===============
DATA_DIR = "/home/tzhang174/EVData_XGame"
REFINED = {
    "A0003": f"{DATA_DIR}/data/A0003_refined.csv",
    "A0008": f"{DATA_DIR}/data/A0008_refined.csv",
}
DIRECTION_PATH = f"{DATA_DIR}/data/direction.csv"
INTERSECTION_JSON_PATH_A = f"{DATA_DIR}/data/intersection_A.json"  # 东西向中心线
INTERSECTION_JSON_PATH_B = f"{DATA_DIR}/data/intersection_B.json"  # 南北向中心线
OUT_PATH = f"{DATA_DIR}/data/axis_ecdf.json"

# =============== 规则参数（与之前一致） ===============
SPEED_THRESH = 5.0 / 3.6      # m/s
MAX_UPSTREAM_M = 120.0        # 仅取中心上游 120m
ECDF_STEP = 0.5               # 输出等距百分位步长（单位：百分比）
STOP_Q = 0.95                 # ≤5% 越线（上取）

SIDE_TO_DIRS = {
    "west":  {"A1-1", "B2-1", "B3-2"},
    "east":  {"A1-2", "B2-2", "B3-1"},
    "south": {"B1-1", "A2-2", "A3-1"},
    "north": {"B1-2", "A2-1", "A3-2"},
}
SIDE_TO_FILE = {"west": "A", "east": "A", "south": "B", "north": "B"}

# 路口中心点（你给的是 (lat, lon)；内部按 (lon, lat) 用）
CENTER_OVERRIDES_LATLON = {
    "A0003": (32.34513595, 123.15253329),
    "A0008": (32.32708227, 123.18126882),
}

# =============== 工具函数 ===============
def read_json(path):
    txt = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except Exception:
        return json.loads(txt.replace("\n", "").replace("\r", ""))

def best_local_crs(lon, lat):
    zone = int(math.floor((lon + 180) / 6) + 1)
    return CRS.from_epsg(32600 + zone if lat >= 0 else 32700 + zone)

def unit(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def get_center_ll(road_id: str):
    lat, lon = CENTER_OVERRIDES_LATLON[road_id]
    return np.array([lon, lat], dtype=float)

def load_lane_divider(road_id: str, side: str):
    primary = INTERSECTION_JSON_PATH_A if SIDE_TO_FILE[side] == "A" else INTERSECTION_JSON_PATH_B
    backup  = INTERSECTION_JSON_PATH_B if SIDE_TO_FILE[side] == "A" else INTERSECTION_JSON_PATH_A
    for p in (primary, backup):
        data = read_json(p)
        node = data.get(road_id)
        if node and "lane_divider" in node:
            arr = np.array(node["lane_divider"], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 2:
                return arr
    raise ValueError(f"{road_id}/{side}: 未找到 lane_divider（检查 intersection_A/B.json）")

def is_stop(row) -> bool:
    et = str(row.get("end_time", "")).strip()
    if et and et.lower() != "nan":
        return True
    spd = float(row.get("speed", 0.0))
    return spd <= SPEED_THRESH

def compute_s_series(road_id: str, side: str, refined: pd.DataFrame, direction_all: pd.DataFrame):
    keys = ["vehicle_id", "date", "seg_id", "road_id"]
    dirs = SIDE_TO_DIRS[side]
    sub_dir = direction_all[(direction_all["road_id"] == road_id) & (direction_all["direction"].isin(dirs))]
    df = refined.merge(sub_dir[keys + ["direction"]], on=keys, how="inner")
    if df.empty:
        return np.array([], dtype=float)

    df = df[df.apply(is_stop, axis=1)].copy()
    if df.empty:
        return np.array([], dtype=float)

    lane_div_ll = load_lane_divider(road_id, side)   # [lon,lat]
    center_ll   = get_center_ll(road_id)             # [lon,lat]

    crs_geo = CRS.from_epsg(4326)
    crs_loc = best_local_crs(center_ll[0], center_ll[1])
    to_xy = Transformer.from_crs(crs_geo, crs_loc, always_xy=True)

    def ll2xy(a):
        x, y = to_xy.transform(a[:,0], a[:,1])
        return np.column_stack([x, y])
    def pt_ll2xy(lon, lat):
        x, y = to_xy.transform(lon, lat)
        return np.array([x, y])

    lane_xy = ll2xy(lane_div_ll)
    center_xy = pt_ll2xy(center_ll[0], center_ll[1])

    v = lane_xy[-1] - lane_xy[-2]
    v_hat = unit(v)
    if np.dot(v_hat, center_xy - lane_xy[-1]) < 0:
        v_hat = -v_hat

    pts_ll = df[["longitude","latitude"]].to_numpy(dtype=float)
    pts_xy = ll2xy(pts_ll)
    s_raw = np.einsum("ij,j->i", pts_xy - center_xy, v_hat)

    if float(np.mean(s_raw > 0)) > 0.5:
        v_hat = -v_hat
        s_raw = np.einsum("ij,j->i", pts_xy - center_xy, v_hat)

    s = s_raw[(s_raw <= 0.0) & (s_raw >= -MAX_UPSTREAM_M)]
    return s

def exact_percentiles_from_samples(samples_abs: np.ndarray, step_pct: float):
    if samples_abs.size == 0:
        return {"percent": [], "distance_m": []}
    perc = np.arange(0.0, 100.0 + 1e-9, step_pct, dtype=float)
    dist = np.quantile(samples_abs, perc/100.0, method="linear")
    return {"percent": [float(x) for x in perc], "distance_m": [float(y) for y in dist]}

def pair_threshold_and_beyond(samples_abs: np.ndarray, stop_q: float):
    if samples_abs.size == 0:
        return {"d_m": None, "beyond_fraction": None, "p_star_pct": None}
    # 上取分位，保证 P(|s| < d) ≤ 1 - stop_q
    # numpy 没有 'higher'，用排序后索引实现
    sorted_vals = np.sort(samples_abs)
    n = sorted_vals.size
    # 目标 p = 1 - stop_q（例如 0.05）
    p = 1.0 - stop_q
    # 上取：first index k s.t. (k/n) >= p
    k = int(np.ceil(p * n))
    if k <= 0: k = 1
    if k > n: k = n
    d = float(sorted_vals[k-1])
    # 严格不等号
    beyond = float(np.mean(samples_abs < d))
    p_star = beyond * 100.0
    return {"d_m": d, "beyond_fraction": beyond, "p_star_pct": p_star}

# =============== 主流程：输出精确 ECDF（反函数）和成对阈值 ===============
def main():
    Path(Path(OUT_PATH).parent).mkdir(parents=True, exist_ok=True)
    direction_all = pd.read_csv(DIRECTION_PATH)

    out = {}
    for rid, refined_path in REFINED.items():
        refined = pd.read_csv(refined_path)

        s_w = compute_s_series(rid, "west",  refined, direction_all)
        s_e = compute_s_series(rid, "east",  refined, direction_all)
        s_s = compute_s_series(rid, "south", refined, direction_all)
        s_n = compute_s_series(rid, "north", refined, direction_all)

        dist_EW = np.concatenate([np.abs(s_e), np.abs(s_w)], axis=0)
        dist_NS = np.concatenate([np.abs(s_n), np.abs(s_s)], axis=0)

        ew_q = exact_percentiles_from_samples(dist_EW, ECDF_STEP)
        ns_q = exact_percentiles_from_samples(dist_NS, ECDF_STEP)

        ew_pair = pair_threshold_and_beyond(dist_EW, STOP_Q)
        ns_pair = pair_threshold_and_beyond(dist_NS, STOP_Q)

        out[rid] = {
            "EW": {
                "n": int(dist_EW.size),
                "ecdf_percent": ew_q["percent"],       # x: 百分位（0 → 100）
                "ecdf_distance_m": ew_q["distance_m"], # y: 对应的|s|距离
                "pair_threshold_m": ew_pair["d_m"],
                "combined_beyond_fraction": ew_pair["beyond_fraction"],
                "p_star_pct": ew_pair["p_star_pct"],
                "by_side_counts": {"east": int(s_e.size), "west": int(s_w.size)}
            },
            "NS": {
                "n": int(dist_NS.size),
                "ecdf_percent": ns_q["percent"],
                "ecdf_distance_m": ns_q["distance_m"],
                "pair_threshold_m": ns_pair["d_m"],
                "combined_beyond_fraction": ns_pair["beyond_fraction"],
                "p_star_pct": ns_pair["p_star_pct"],
                "by_side_counts": {"north": int(s_n.size), "south": int(s_s.size)}
            }
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] 写出：{OUT_PATH}")

if __name__ == "__main__":
    main()
