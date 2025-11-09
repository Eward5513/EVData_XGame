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

# 车道中心线：假定东西向在 A，南北向在 B（若两个都有，优先 A）
INTERSECTION_JSON_PATH_A = f"{DATA_DIR}/data/intersection_A.json"
INTERSECTION_JSON_PATH_B = f"{DATA_DIR}/data/intersection_B.json"

OUT_DIR = f"{DATA_DIR}/data"

# 各方向使用的“方向码”
SIDE_TO_DIRS = {
    "west":  {"A1-1", "B2-1", "B3-2"},
    "east":  {"A1-2", "B2-2", "B3-1"},
    "south": {"B1-1", "A2-2", "A3-1"},
    "north": {"B1-2", "A2-1", "A3-2"},
}

# 东西向中心线用 A，南北向中心线用 B
SIDE_TO_INTERSECTION_FILE = {
    "west":  "A",
    "east":  "A",
    "south": "B",
    "north": "B",
}

# 驻停判定与采样窗口
SPEED_THRESH = 5.0 / 3.6   # m/s，对应 5 km/h
MAX_UPSTREAM_M = 120.0     # 仅使用中心点上游这段距离
STOP_Q = 0.95              # p95 → 约5% 越过（用秩选择保证 ≤ 5%）

# 覆盖中心点（你给的是 (lat, lon)，脚本转为 (lon, lat)）
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

def load_intersection_for_side(road_id: str, side: str):
    """按方向选择中心线文件：东西->A，南北->B；若主文件缺失则尝试另一个。"""
    files = []
    if SIDE_TO_INTERSECTION_FILE.get(side, "A") == "A":
        files = [INTERSECTION_JSON_PATH_A, INTERSECTION_JSON_PATH_B]
    else:
        files = [INTERSECTION_JSON_PATH_B, INTERSECTION_JSON_PATH_A]

    data = None
    for p in files:
        if not Path(p).exists():
            continue
        text = Path(p).read_text(encoding="utf-8").strip()
        try:
            node = json.loads(text).get(road_id)
        except Exception:
            node = json.loads(text.replace("\n", "").replace("\r", "")).get(road_id)
        if node and "lane_divider" in node:
            lane_div = np.array(node["lane_divider"], dtype=float)  # (N,2) [lon,lat]
            if lane_div.ndim == 2 and lane_div.shape[1] == 2 and lane_div.shape[0] >= 2:
                data = lane_div
                break
    if data is None:
        raise ValueError(f"{road_id}/{side}: 找不到 lane_divider，请检查 intersection_A/B.json")
    return data

def load_center_ll(road_id: str):
    if road_id in CENTER_OVERRIDES_LATLON:
        lat, lon = CENTER_OVERRIDES_LATLON[road_id]
        return np.array([lon, lat], dtype=float)
    # 如未覆盖，可选择从 A/B 中读取可选字段 center；此处直接抛错更安全
    raise ValueError(f"{road_id}: 未提供 center 覆盖坐标。")

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
# 核心计算（单路口、单方向）
# =========================
def estimate_stopline_one_side(road_id: str, side: str, refined: pd.DataFrame, direction_all: pd.DataFrame):
    dirs = SIDE_TO_DIRS[side]
    # 筛方向
    keys = ["vehicle_id", "date", "seg_id", "road_id"]
    direction = direction_all[(direction_all["road_id"] == road_id) & (direction_all["direction"].isin(dirs))]
    df = refined.merge(direction[keys + ["direction"]], on=keys, how="inner")
    if len(df) == 0:
        return {"stopline_segment": None}, {
            "params": {"directions_used": sorted(dirs), "speed_thresh_mps": SPEED_THRESH, "max_upstream_m": MAX_UPSTREAM_M, "quantile_for_stopline": STOP_Q},
            "note": "no points after join",
            "overall": {"n_points": 0},
        }

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
        return {"stopline_segment": None}, {
            "params": {"directions_used": sorted(dirs), "speed_thresh_mps": SPEED_THRESH, "max_upstream_m": MAX_UPSTREAM_M, "quantile_for_stopline": STOP_Q},
            "note": "no stop points after filter",
            "overall": {"n_points": 0},
        }

    # 中心线与中心点
    lane_div_ll = load_intersection_for_side(road_id, side)  # [lon,lat]
    center_ll = load_center_ll(road_id)                      # [lon,lat]

    # 局部投影
    crs_geo = CRS.from_epsg(4326)
    crs_loc = best_local_crs(center_ll[0], center_ll[1])
    to_xy = Transformer.from_crs(crs_geo, crs_loc, always_xy=True)
    to_ll = Transformer.from_crs(crs_loc, crs_geo, always_xy=True)

    def ll_to_xy(arr_ll):
        x, y = to_xy.transform(arr_ll[:, 0], arr_ll[:, 1])
        return np.column_stack([x, y])

    def one_ll_to_xy(lon, lat):
        x, y = to_xy.transform(lon, lat)
        return np.array([x, y])

    lane_div_xy = ll_to_xy(lane_div_ll)
    center_xy = one_ll_to_xy(center_ll[0], center_ll[1])

    # 轴向方向：取 lane_divider 最后一段，并指向中心
    v = lane_div_xy[-1] - lane_div_xy[-2]
    v_hat = unit(v)
    to_center_vec = center_xy - lane_div_xy[-1]
    if np.dot(v_hat, to_center_vec) < 0:
        v_hat = -v_hat
    n_hat = np.array([-v_hat[1], v_hat[0]])

    # 计算 s_raw（相对中心的沿路坐标，朝向中心增大）
    pts_ll = df[["longitude", "latitude"]].to_numpy(dtype=float)
    pts_xy = ll_to_xy(pts_ll)
    s_raw = np.einsum("ij,j->i", pts_xy - center_xy, v_hat)

    # 多数应在上游（s<=0）；若多数 s_raw>0 则翻转
    share_pos = float(np.nanmean(s_raw > 0))
    if share_pos > 0.5:
        v_hat = -v_hat
        n_hat = np.array([-v_hat[1], v_hat[0]])
        s_raw = np.einsum("ij,j->i", pts_xy - center_xy, v_hat)

    # 仅用上游窗口
    df["s_m"] = s_raw
    df = df[(df["s_m"] <= 0.0) & (df["s_m"] >= -MAX_UPSTREAM_M)].copy()
    if len(df) == 0:
        analyse = {
            "params": {"directions_used": sorted(dirs), "speed_thresh_mps": SPEED_THRESH, "max_upstream_m": MAX_UPSTREAM_M, "quantile_for_stopline": STOP_Q},
            "orientation_check": {"share_s_raw_pos_before_flip": share_pos, "final_share_s_pos": None, "expect_upstream_nonpos": True},
            "note": "no upstream points after orientation check",
            "overall": {"n_points": 0},
        }
        return {"stopline_segment": None}, analyse

    # ====== 用秩阈值严格控制越线 ≤ (1-STOP_Q) ======
    s_vals = np.sort(df["s_m"].to_numpy(float))  # 升序（更负→更上游）
    N = s_vals.size
    m_allow = int(math.floor((1.0 - STOP_Q) * N))  # 允许严格>阈值 的最大数量
    idx = max(0, N - m_allow - 1)
    stop_s = float(s_vals[idx])
    stop_s = min(0.0, max(stop_s, -MAX_UPSTREAM_M))  # 夹在窗口中

    df["beyond_line"] = df["s_m"] > stop_s
    frac_beyond = float(df["beyond_line"].mean())

    # 构造 20m 垂线段
    stop_pt_xy = center_xy + stop_s * v_hat
    half_len = 10.0
    p_left_xy = stop_pt_xy - half_len * n_hat
    p_right_xy = stop_pt_xy + half_len * n_hat

    def xy_to_ll(xy):
        lon, lat = to_ll.transform(xy[0], xy[1])
        return [float(lon), float(lat)]

    stopline_segment = [xy_to_ll(p_left_xy), xy_to_ll(p_right_xy)]

    analyse = {
        "params": {
            "directions_used": sorted(list(dirs)),
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
            "beyond_fraction": frac_beyond,               # 严格 ≤ 1-STOP_Q
            "constraint_ok": frac_beyond <= (1.0 - STOP_Q + 1e-12)
        },
        "overall": {
            "n_points": int(len(df)),
            "s_m_summary": describe_series(df["s_m"].to_numpy(float)),
            "unique_vehicles": int(df["vehicle_id"].nunique()) if "vehicle_id" in df.columns else None,
            "unique_segments": int(df["seg_id"].nunique()) if "seg_id" in df.columns else None,
        }
    }

    return {"stopline_segment": stopline_segment}, analyse

# =========================
# 主入口（两个路口 × 四个方向）
# =========================
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 读 direction.csv 一次
    direction_all = pd.read_csv(DIRECTION_PATH)

    stoplines = {rid: {"west": {}, "east": {}, "south": {}, "north": {}} for rid in REFINED.keys()}
    analyses  = {rid: {"west": {}, "east": {}, "south": {}, "north": {}} for rid in REFINED.keys()}

    for rid, refined_path in REFINED.items():
        refined = pd.read_csv(refined_path)

        for side in ("west", "east", "south", "north"):
            res, ana = estimate_stopline_one_side(rid, side, refined, direction_all)
            stoplines[rid][side] = {"stopline_segment": res["stopline_segment"]}
            analyses[rid][side] = ana

    # 写结果文件
    with open(f"{OUT_DIR}/stopline.json", "w", encoding="utf-8") as f:
        json.dump(stoplines, f, ensure_ascii=False, indent=2)

    with open(f"{OUT_DIR}/stopline_analyse.json", "w", encoding="utf-8") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=2)

    print("[OK] 输出完成：")
    print(f"  - {OUT_DIR}/stopline.json")
    print(f"  - {OUT_DIR}/stopline_analyse.json")

if __name__ == "__main__":
    main()
