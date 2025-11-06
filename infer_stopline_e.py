#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from pyproj import CRS, Transformer

# =========================
# 硬编码路径 & 参数
# =========================
DATA_DIR = "/home/tzhang174/EVData_XGame"
REFINED = {
    "A0003": f"{DATA_DIR}/data/A0003_refined.csv",
    "A0008": f"{DATA_DIR}/data/A0008_refined.csv",
}
DIRECTION_PATH = f"{DATA_DIR}/data/direction.csv"
INTERSECTION_JSON_PATH = f"{DATA_DIR}/data/intersection_A.json"  # JSON，包含 lane_divider
OUT_DIR = f"{DATA_DIR}/data"

# 东侧进口相关方向
DIRECTIONS_USED = {"A1-2", "B2-2", "B3-1"}

# 驻停判定与采样窗口
SPEED_THRESH = 5.0 / 3.6   # m/s，对应 5 km/h
MAX_UPSTREAM_M = 120.0     # 仅使用中心点上游这段距离
STOP_Q = 0.95              # 90% 分位 → 约 10% 越过；想更少越过可调更大(如0.95)

# 覆盖中心点（你提供的是 (lat, lon)，脚本会转成 (lon, lat)）
CENTER_OVERRIDES_LATLON = {
    "A0003": (32.34513595, 123.15253329),
    "A0008": (32.32708227, 123.18126882),
}

# =========================
# 工具函数
# =========================
def best_local_crs(lon, lat):
    utm_zone = int(math.floor((lon + 180) / 6) + 1)
    return CRS.from_epsg(32600 + utm_zone if lat >= 0 else 32700 + utm_zone)

def unit(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def load_intersection(json_path: str, road_id: str):
    text = Path(json_path).read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except Exception:
        fixed = text.replace("\n", "").replace("\r", "")
        data = json.loads(fixed)

    if road_id not in data:
        raise ValueError(f"{json_path} 中没有 {road_id} 条目")
    node = data[road_id]
    if "lane_divider" not in node:
        raise ValueError(f"{json_path} 的 {road_id} 缺少 lane_divider")

    lane_div = np.array(node["lane_divider"], dtype=float)  # (N,2) -> [lon,lat]
    if lane_div.ndim != 2 or lane_div.shape[1] != 2 or lane_div.shape[0] < 2:
        raise ValueError(f"{road_id} 的 lane_divider 至少需要两个 [lon,lat] 点")

    # 中心点：优先使用覆盖（lat,lon -> lon,lat）
    if road_id in CENTER_OVERRIDES_LATLON:
        lat, lon = CENTER_OVERRIDES_LATLON[road_id]
        center = np.array([lon, lat], dtype=float)
    else:
        center = None
        for k in ("center", "center_point", "centerPoint"):
            if k in node and isinstance(node[k], (list, tuple)) and len(node[k]) == 2:
                center = np.array(node[k], dtype=float)
                break
        if center is None:
            center = lane_div[-1].copy()  # 兜底

    return lane_div, center

def describe_series(x: np.ndarray):
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None
    return {
        "count": int(x.size),
        "min": float(np.min(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
    }

# =========================
# 核心流程（每个路口）
# =========================
def process_road(road_id: str):
    refined_path = REFINED[road_id]
    refined = pd.read_csv(refined_path)
    direction = pd.read_csv(DIRECTION_PATH)

    # 方向/路口筛选并关联
    keys = ["vehicle_id", "date", "seg_id", "road_id"]
    direction = direction[(direction["road_id"] == road_id) & (direction["direction"].isin(DIRECTIONS_USED))]
    df = refined.merge(direction[keys + ["direction"]], on=keys, how="inner")

    # 驻停判定：end_time 优先；无则速度兜底
    def is_stop(row) -> bool:
        et = str(row.get("end_time", "")).strip()
        if et and et.lower() != "nan":
            return True
        try:
            spd = float(row.get("speed", 0.0))
        except Exception:
            spd = 0.0
        return spd <= SPEED_THRESH
    df = df[df.apply(is_stop, axis=1)].copy()
    if len(df) == 0:
        return {
            "stopline_segment": None,
            "analysis": {"n_points": 0, "note": "no stop points after filter"}
        }

    # 中心线与中心点
    lane_div_ll, center_ll = load_intersection(INTERSECTION_JSON_PATH, road_id)

    # 局部投影
    crs_geo = CRS.from_epsg(4326)
    crs_loc = best_local_crs(center_ll[0], center_ll[1])  # center_ll = (lon,lat)
    to_xy = Transformer.from_crs(crs_geo, crs_loc, always_xy=True)
    to_ll = Transformer.from_crs(crs_loc, crs_geo, always_xy=True)

    def ll_to_xy(arr_ll):
        x, y = to_xy.transform(arr_ll[:, 0], arr_ll[:, 1])  # lon, lat
        return np.column_stack([x, y])

    def one_ll_to_xy(lon, lat):
        x, y = to_xy.transform(lon, lat)
        return np.array([x, y])

    lane_div_xy = ll_to_xy(lane_div_ll)
    center_xy = one_ll_to_xy(center_ll[0], center_ll[1])

    # ---- 方向确定（自动自检翻转，确保上游为 s<=0）----
    v = lane_div_xy[-1] - lane_div_xy[-2]
    v_hat = unit(v)
    to_center_vec = center_xy - lane_div_xy[-1]
    if np.dot(v_hat, to_center_vec) < 0:
        v_hat = -v_hat
    n_hat = np.array([-v_hat[1], v_hat[0]])

    # s_raw：相对中心的沿路坐标（朝向中心增大）
    pts_ll = df[["longitude", "latitude"]].to_numpy(dtype=float)
    pts_xy = ll_to_xy(pts_ll)
    s_raw = np.einsum("ij,j->i", pts_xy - center_xy, v_hat)

    # 多数应在上游（s<=0）；若多数s_raw>0，翻转
    share_pos = float(np.nanmean(s_raw > 0))
    if share_pos > 0.5:
        v_hat = -v_hat
        n_hat = np.array([-v_hat[1], v_hat[0]])
        s_raw = np.einsum("ij,j->i", pts_xy - center_xy, v_hat)

    # 仅用上游窗口
    df["s_m"] = s_raw
    df = df[(df["s_m"] <= 0.0) & (df["s_m"] >= -MAX_UPSTREAM_M)].copy()
    if len(df) == 0:
        return {
            "stopline_segment": None,
            "analysis": {"n_points": 0, "note": "no upstream stop points after orientation check"}
        }

    # ====== 严格控制越线比例 ≤ (1-STOP_Q) ======
    s_vals = np.sort(df["s_m"].to_numpy(float))  # 更负→更上游
    N = s_vals.size
    m_allow = int(math.floor((1.0 - STOP_Q) * N))  # 允许越线的最大数量
    idx = max(0, N - m_allow - 1)                  # 选阈值，使严格 > 阈值 的个数 ≤ m_allow
    stop_s = float(s_vals[idx])
    stop_s = min(0.0, max(stop_s, -MAX_UPSTREAM_M))  # 约束在窗口内

    df["beyond_line"] = df["s_m"] > stop_s
    frac_beyond = float(df["beyond_line"].mean())

    # 停止线段（20m 垂线）
    stop_pt_xy = center_xy + stop_s * v_hat
    half_len = 10.0
    p_left_xy = stop_pt_xy - half_len * n_hat
    p_right_xy = stop_pt_xy + half_len * n_hat

    def xy_to_ll(xy):
        lon, lat = to_ll.transform(xy[0], xy[1])
        return [float(lon), float(lat)]

    stopline_segment = [xy_to_ll(p_left_xy), xy_to_ll(p_right_xy)]

    # 分析
    analyse = {
        "params": {
            "directions_used": sorted(list(DIRECTIONS_USED)),
            "speed_thresh_mps": SPEED_THRESH,
            "max_upstream_m": MAX_UPSTREAM_M,
            "quantile_for_stopline": STOP_Q,
        },
        "orientation_check": {
            "share_s_raw_pos_before_flip": share_pos,
            "final_share_s_pos": float(np.nanmean(df["s_m"] > 0)),
            "expect_upstream_nonpos": True
        },
        "stopline": {
            "stop_s_m": stop_s,                           # ≤ 0
            "distance_to_center_m": abs(stop_s),
            "beyond_fraction": frac_beyond,               # 严格 ≤ (1-STOP_Q)
            "constraint_ok": frac_beyond <= (1.0 - STOP_Q + 1e-12)
        },
        "overall": {
            "n_points": int(len(df)),
            "s_m_summary": describe_series(df["s_m"].to_numpy(float)),
            "unique_vehicles": int(df["vehicle_id"].nunique()) if "vehicle_id" in df.columns else None,
            "unique_segments": int(df["seg_id"].nunique()) if "seg_id" in df.columns else None,
        }
    }

    return {
        "stopline_segment": stopline_segment,
        "analysis": analyse
    }

# =========================
# 主入口：只写两个 JSON 文件（EAST）
# =========================
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    stoplines = {}
    analyses = {}

    for rid in ("A0003", "A0008"):
        result = process_road(rid)
        stoplines[rid] = {"stopline_segment": result["stopline_segment"]}
        analyses[rid] = result["analysis"]

    with open(f"{OUT_DIR}/stopline_east.json", "w", encoding="utf-8") as f:
        json.dump(stoplines, f, ensure_ascii=False, indent=2)

    with open(f"{OUT_DIR}/analyse_stopline_east.json", "w", encoding="utf-8") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=2)

    print("[OK] 输出完成：")
    print(f"  - {OUT_DIR}/stopline_east.json")
    print(f"  - {OUT_DIR}/analyse_stopline_east.json")

if __name__ == "__main__":
    main()
