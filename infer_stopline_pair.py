#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
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
INTERSECTION_JSON_PATH_A = f"{DATA_DIR}/data/intersection_A.json"
INTERSECTION_JSON_PATH_B = f"{DATA_DIR}/data/intersection_B.json"
OUT_DIR = f"{DATA_DIR}/data"

# 方向与车道中心线来源（E/W 用 A，N/S 用 B；若缺则互为兜底）
SIDE_TO_DIRS = {
    "west":  {"A1-1", "B2-1", "B3-2"},
    "east":  {"A1-2", "B2-2", "B3-1"},
    "south": {"B1-1", "A2-2", "A3-1"},
    "north": {"B1-2", "A2-1", "A3-2"},
}
SIDE_TO_FILE = {
    "west": "A", "east": "A",
    "south": "B", "north": "B",
}

# 驻停与窗口
SPEED_THRESH = 5.0 / 3.6   # m/s (5 km/h)
MAX_UPSTREAM_M = 120.0     # 仅取中心点上游这段
STOP_Q = 0.95              # p95 → 目标越线 ≤ 5%（用秩阈值严格控制）

# 路口中心（你给的是 (lat, lon)，脚本里转成 (lon, lat) ）
CENTER_OVERRIDES_LATLON = {
    "A0003": (32.34513595, 123.15253329),
    "A0008": (32.32708227, 123.18126882),
}

# =========================
# 小工具
# =========================
def best_local_crs(lon, lat):
    utm_zone = int(math.floor((lon + 180) / 6) + 1)
    return CRS.from_epsg(32600 + utm_zone if lat >= 0 else 32700 + utm_zone)

def unit(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def read_json(path):
    if not Path(path).exists():
        return {}
    txt = Path(path).read_text(encoding="utf-8").strip()
    try:
        return json.loads(txt)
    except Exception:
        return json.loads(txt.replace("\n", "").replace("\r", ""))

def load_lane_divider(road_id: str, side: str):
    """E/W 优先 A.json，N/S 优先 B.json；若缺尝试另一份。返回 Nx2 [lon,lat]."""
    which = SIDE_TO_FILE.get(side, "A")
    primary = INTERSECTION_JSON_PATH_A if which == "A" else INTERSECTION_JSON_PATH_B
    backup  = INTERSECTION_JSON_PATH_B if which == "A" else INTERSECTION_JSON_PATH_A
    for p in (primary, backup):
        data = read_json(p)
        node = data.get(road_id)
        if node and "lane_divider" in node:
            arr = np.array(node["lane_divider"], dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 2:
                return arr
    raise ValueError(f"{road_id}/{side}: 未找到 lane_divider（检查 intersection_A/B.json）")

def get_center_ll(road_id: str):
    if road_id not in CENTER_OVERRIDES_LATLON:
        raise ValueError(f"{road_id}: 缺少中心点覆盖坐标")
    lat, lon = CENTER_OVERRIDES_LATLON[road_id]
    return np.array([lon, lat], dtype=float)

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
# 基础投影与 s 计算（单个方向）
# =========================
def compute_s_series(road_id: str, side: str, refined: pd.DataFrame, direction_all: pd.DataFrame):
    """返回：dict(
        ok, s_m (np.ndarray, 上游为负且截到 [-MAX,0]),
        summary, orient_check, v_hat, n_hat, center_xy, to_ll
    )"""
    # 方向关联
    keys = ["vehicle_id", "date", "seg_id", "road_id"]
    dirs = SIDE_TO_DIRS[side]
    direction = direction_all[(direction_all["road_id"] == road_id) & (direction_all["direction"].isin(dirs))]
    df = refined.merge(direction[keys + ["direction"]], on=keys, how="inner")
    if df.empty:
        return {"ok": False, "note": "join_no_points"}

    # 驻停判定
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
    if df.empty:
        return {"ok": False, "note": "no_stop_points"}

    # 中心线、中心点、投影
    lane_div_ll = load_lane_divider(road_id, side)      # [lon,lat]
    center_ll = get_center_ll(road_id)                  # [lon,lat]
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

    # 轴向方向：以 lane_divider 最后一段，指向中心；法向 n_hat
    v = lane_div_xy[-1] - lane_div_xy[-2]
    v_hat = unit(v)
    if np.dot(v_hat, center_xy - lane_div_xy[-1]) < 0:
        v_hat = -v_hat
    n_hat = np.array([-v_hat[1], v_hat[0]])

    # s_raw：相对中心沿路坐标（朝向中心增大），确保上游为负（多数应 <= 0）
    pts_ll = df[["longitude", "latitude"]].to_numpy(dtype=float)
    pts_xy = ll_to_xy(pts_ll)
    s_raw = np.einsum("ij,j->i", pts_xy - center_xy, v_hat)
    share_pos = float(np.nanmean(s_raw > 0))
    if share_pos > 0.5:
        v_hat = -v_hat
        n_hat = np.array([-v_hat[1], v_hat[0]])
        s_raw = np.einsum("ij,j->i", pts_xy - center_xy, v_hat)

    # 仅上游窗口
    s = s_raw[(s_raw <= 0.0) & (s_raw >= -MAX_UPSTREAM_M)]
    if s.size == 0:
        return {"ok": False, "note": "no_upstream_points",
                "orientation_check": {"share_s_raw_pos_before_flip": share_pos}}

    return {
        "ok": True,
        "s_m": s,                 # 负值，距中心的“沿线坐标”
        "summary": describe_series(s),
        "orientation_check": {
            "share_s_raw_pos_before_flip": share_pos,
            "final_share_s_pos": float(np.nanmean(s > 0.0)),
            "expect_upstream_nonpos": True
        },
        "v_hat": v_hat,
        "n_hat": n_hat,
        "center_xy": center_xy,
        "to_ll": to_ll,
    }

# ===== 成对联合阈值（E-W / N-S） =====
def pairwise_distance_threshold(distA: np.ndarray, distB: np.ndarray, stop_q: float):
    """
    输入：两侧距离样本（均为正数 = abs(s)），输出联合的秩阈值 d，使得
    (count(distA<d*) + count(distB<d*)) / (NA+NB) ≈ 1 - stop_q
    采用秩选择严格保证 “> d 的比例 ≤ 1 - stop_q”：
      - 合并 distances 升序 = dists
      - m_allow = floor((1-stop_q)*(N_total))
      - idx = N_total - m_allow - 1
      - d = dists[idx]
    """
    dists = np.concatenate([distA, distB], axis=0)
    if dists.size == 0:
        return None
    # 目标：P(|s| < d) ≤ 1 - stop_q   （例如 stop_q=0.95 → 5%）
    p = 1.0 - stop_q
    # 用分位数实现严格上界（如需更保守，可再乘 0.99）
    d = float(np.quantile(dists, p, method="higher"))
    return d

def build_stopline_segment(center_xy, v_hat, n_hat, to_ll, d_abs):
    """给定统一距离 d_abs（正数，距中心），生成 20m 垂线段的经纬度端点。"""
    stop_s = -float(d_abs)  # 上游为负
    stop_pt_xy = center_xy + stop_s * v_hat
    half = 10.0
    left_xy  = stop_pt_xy - half * n_hat
    right_xy = stop_pt_xy + half * n_hat
    def xy2ll(xy):
        lon, lat = to_ll.transform(xy[0], xy[1])
        return [float(lon), float(lat)]
    return [xy2ll(left_xy), xy2ll(right_xy)], stop_s

# =========================
# 主流程
# =========================
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    direction_all = pd.read_csv(DIRECTION_PATH)
    result_lines = {rid: {"west": {}, "east": {}, "south": {}, "north": {}} for rid in REFINED}
    analyses     = {rid: {"west": {}, "east": {}, "south": {}, "north": {},
                          "_pairs": {}} for rid in REFINED}

    for rid, refined_path in REFINED.items():
        refined = pd.read_csv(refined_path)

        # 分别计算四个方向的 s 数组与几何要素
        computed = {}
        for side in ("west", "east", "south", "north"):
            computed[side] = compute_s_series(rid, side, refined, direction_all)

        # ---- 成对：E-W ----
        ew_ok = computed["east"]["ok"] or computed["west"]["ok"]
        if ew_ok:
            # 取已有侧的几何（若两侧都有，用各自）
            east = computed["east"] if computed["east"]["ok"] else None
            west = computed["west"] if computed["west"]["ok"] else None

            # 准备距离样本
            dist_e = np.abs(east["s_m"]) if east else np.array([], dtype=float)
            dist_w = np.abs(west["s_m"]) if west else np.array([], dtype=float)

            # 合并秩阈值
            d_ew = pairwise_distance_threshold(dist_e, dist_w, STOP_Q)
            pair_note = None
            if d_ew is None:
                pair_note = "no_samples_both_sides"
                d_ew = 0.0

            # 生成两侧停止线段（用各自的几何）
            if west:
                seg_w, stop_sw = build_stopline_segment(
                    west["center_xy"], west["v_hat"], west["n_hat"], west["to_ll"], d_ew
                )
                result_lines[rid]["west"] = {"stopline_segment": seg_w}
                # 统计越线
                beyond_w = float(np.mean(west["s_m"] > -d_ew))
                analyses[rid]["west"] = {
                    "params": {"directions_used": sorted(SIDE_TO_DIRS["west"]),
                               "speed_thresh_mps": SPEED_THRESH, "max_upstream_m": MAX_UPSTREAM_M,
                               "quantile_for_stopline": STOP_Q, "paired_axis": "EW"},
                    "orientation_check": west.get("orientation_check"),
                    "stopline": {"stop_s_m": stop_sw, "distance_to_center_m": d_ew,
                                 "beyond_fraction": beyond_w, "constraint_ok": None},
                    "overall": {"n_points": int(west["summary"]["count"]) if west["summary"] else 0,
                                "s_m_summary": west["summary"]}
                }
            else:
                result_lines[rid]["west"] = {"stopline_segment": None}
                analyses[rid]["west"] = {"note": "no_west_data"}

            if east:
                seg_e, stop_se = build_stopline_segment(
                    east["center_xy"], east["v_hat"], east["n_hat"], east["to_ll"], d_ew
                )
                result_lines[rid]["east"] = {"stopline_segment": seg_e}
                beyond_e = float(np.mean(east["s_m"] > -d_ew))
                analyses[rid]["east"] = {
                    "params": {"directions_used": sorted(SIDE_TO_DIRS["east"]),
                               "speed_thresh_mps": SPEED_THRESH, "max_upstream_m": MAX_UPSTREAM_M,
                               "quantile_for_stopline": STOP_Q, "paired_axis": "EW"},
                    "orientation_check": east.get("orientation_check"),
                    "stopline": {"stop_s_m": stop_se, "distance_to_center_m": d_ew,
                                 "beyond_fraction": beyond_e, "constraint_ok": None},
                    "overall": {"n_points": int(east["summary"]["count"]) if east["summary"] else 0,
                                "s_m_summary": east["summary"]}
                }
            else:
                result_lines[rid]["east"] = {"stopline_segment": None}
                analyses[rid]["east"] = {"note": "no_east_data"}

            # 合并越线比例与约束
            n_e = int(east["summary"]["count"]) if east and east["summary"] else 0
            n_w = int(west["summary"]["count"]) if west and west["summary"] else 0
            b_e = float(np.sum(east["s_m"] > -d_ew)) if east else 0.0
            b_w = float(np.sum(west["s_m"] > -d_ew)) if west else 0.0
            n_tot = n_e + n_w
            frac_pair = (b_e + b_w) / n_tot if n_tot > 0 else 0.0
            ok_pair = frac_pair <= (1.0 - STOP_Q + 1e-12)

            analyses[rid]["_pairs"]["EW"] = {
                "distance_to_center_m": d_ew,
                "n_east": n_e, "n_west": n_w, "n_total": n_tot,
                "beyond_count_east": int(b_e), "beyond_count_west": int(b_w),
                "beyond_fraction_combined": frac_pair,
                "constraint_ok": ok_pair,
                "note": pair_note
            }
            # 回填单侧 constraint_ok（给你查看单侧越线率，也可不填）
            if "stopline" in analyses[rid]["east"]:
                analyses[rid]["east"]["stopline"]["constraint_ok"] = ok_pair
            if "stopline" in analyses[rid]["west"]:
                analyses[rid]["west"]["stopline"]["constraint_ok"] = ok_pair
        else:
            analyses[rid]["_pairs"]["EW"] = {"note": "no_data_both_sides"}

        # ---- 成对：N-S ----
        ns_ok = computed["north"]["ok"] or computed["south"]["ok"]
        if ns_ok:
            north = computed["north"] if computed["north"]["ok"] else None
            south = computed["south"] if computed["south"]["ok"] else None

            dist_n = np.abs(north["s_m"]) if north else np.array([], dtype=float)
            dist_s = np.abs(south["s_m"]) if south else np.array([], dtype=float)

            d_ns = pairwise_distance_threshold(dist_n, dist_s, STOP_Q)
            pair_note = None
            if d_ns is None:
                pair_note = "no_samples_both_sides"
                d_ns = 0.0

            if north:
                seg_n, stop_sn = build_stopline_segment(
                    north["center_xy"], north["v_hat"], north["n_hat"], north["to_ll"], d_ns
                )
                result_lines[rid]["north"] = {"stopline_segment": seg_n}
                beyond_n = float(np.mean(north["s_m"] > -d_ns))
                analyses[rid]["north"] = {
                    "params": {"directions_used": sorted(SIDE_TO_DIRS["north"]),
                               "speed_thresh_mps": SPEED_THRESH, "max_upstream_m": MAX_UPSTREAM_M,
                               "quantile_for_stopline": STOP_Q, "paired_axis": "NS"},
                    "orientation_check": north.get("orientation_check"),
                    "stopline": {"stop_s_m": stop_sn, "distance_to_center_m": d_ns,
                                 "beyond_fraction": beyond_n, "constraint_ok": None},
                    "overall": {"n_points": int(north["summary"]["count"]) if north["summary"] else 0,
                                "s_m_summary": north["summary"]}
                }
            else:
                result_lines[rid]["north"] = {"stopline_segment": None}
                analyses[rid]["north"] = {"note": "no_north_data"}

            if south:
                seg_s, stop_ss = build_stopline_segment(
                    south["center_xy"], south["v_hat"], south["n_hat"], south["to_ll"], d_ns
                )
                result_lines[rid]["south"] = {"stopline_segment": seg_s}
                beyond_s = float(np.mean(south["s_m"] > -d_ns))
                analyses[rid]["south"] = {
                    "params": {"directions_used": sorted(SIDE_TO_DIRS["south"]),
                               "speed_thresh_mps": SPEED_THRESH, "max_upstream_m": MAX_UPSTREAM_M,
                               "quantile_for_stopline": STOP_Q, "paired_axis": "NS"},
                    "orientation_check": south.get("orientation_check"),
                    "stopline": {"stop_s_m": stop_ss, "distance_to_center_m": d_ns,
                                 "beyond_fraction": beyond_s, "constraint_ok": None},
                    "overall": {"n_points": int(south["summary"]["count"]) if south["summary"] else 0,
                                "s_m_summary": south["summary"]}
                }
            else:
                result_lines[rid]["south"] = {"stopline_segment": None}
                analyses[rid]["south"] = {"note": "no_south_data"}

            n_n = int(north["summary"]["count"]) if north and north["summary"] else 0
            n_s = int(south["summary"]["count"]) if south and south["summary"] else 0
            b_n = float(np.sum(north["s_m"] > -d_ns)) if north else 0.0
            b_s = float(np.sum(south["s_m"] > -d_ns)) if south else 0.0
            n_tot = n_n + n_s
            frac_pair = (b_n + b_s) / n_tot if n_tot > 0 else 0.0
            ok_pair = frac_pair <= (1.0 - STOP_Q + 1e-12)

            analyses[rid]["_pairs"]["NS"] = {
                "distance_to_center_m": d_ns,
                "n_north": n_n, "n_south": n_s, "n_total": n_tot,
                "beyond_count_north": int(b_n), "beyond_count_south": int(b_s),
                "beyond_fraction_combined": frac_pair,
                "constraint_ok": ok_pair,
                "note": pair_note
            }
            if "stopline" in analyses[rid]["north"]:
                analyses[rid]["north"]["stopline"]["constraint_ok"] = ok_pair
            if "stopline" in analyses[rid]["south"]:
                analyses[rid]["south"]["stopline"]["constraint_ok"] = ok_pair
        else:
            analyses[rid]["_pairs"]["NS"] = {"note": "no_data_both_sides"}

    # 写文件
    with open(f"{OUT_DIR}/stopline.json", "w", encoding="utf-8") as f:
        json.dump(result_lines, f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/stopline_analyse.json", "w", encoding="utf-8") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=2)

    print("[OK] 输出完成：")
    print(f"  - {OUT_DIR}/stopline.json")
    print(f"  - {OUT_DIR}/stopline_analyse.json")

if __name__ == "__main__":
    main()
