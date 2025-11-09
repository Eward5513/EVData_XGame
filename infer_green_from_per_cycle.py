#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer most likely green intervals per direction (A1/A2/B1/B2)
assuming: crossing (flow) occurs during green; wait is ignored.

Inputs (fixed paths):
  - data/per_cycle_flow.csv

Output:
  - data/inferred_green.csv with columns:
      direction,start_offset_s,end_offset_s,start_time,end_time,score

Notes:
  - Cycle is treated as circular; an interval may wrap past 129 back to 0.
  - Likelihood uses only per-direction flow (smoothed then z-scored).
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DIRS: Tuple[str, str, str, str] = ("A1", "A2", "B1", "B2")
FLOW_CSV = os.path.join("data", "per_cycle_flow.csv")
OUT_CSV = os.path.join("data", "inferred_green.csv")
CYCLE_SECONDS = 130


def _read_per_cycle(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    need = {"offset_s", "time", *DIRS}
    if not need.issubset(set(df.columns)):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"Missing columns in {path}: {missing}")
    # Ensure dense 0..CYCLE_SECONDS-1 offsets
    df = df.sort_values("offset_s").reset_index(drop=True)
    return df


def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    m = float(np.mean(x))
    s = float(np.std(x))
    if s <= 0:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def _derivative_circular(x: np.ndarray) -> np.ndarray:
    # Central difference with wrap
    return (np.roll(x, -1) - np.roll(x, 1)) / 2.0


def _smooth_ma(x: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return x.astype(float)
    w = int(window)
    pad = w // 2
    # circular pad
    x_pad = np.r_[x[-pad:], x, x[:pad]]
    ker = np.ones(w, dtype=float) / w
    y = np.convolve(x_pad, ker, mode="valid")
    return y.astype(float)


def _compute_likelihood(flow: np.ndarray) -> np.ndarray:
    # Smooth flow and z-score; higher means more likely green
    f_s = _smooth_ma(flow, 5)
    zf = _zscore(f_s)
    # Shift to non-negative
    zf = zf - min(0.0, float(zf.min()))
    return zf


def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) inclusive indices for True runs in a circular mask.
    A segment may wrap (start > end) indicating wraparound.
    """
    n = len(mask)
    if n == 0:
        return []
    # Trivial cases
    if np.all(mask):
        return [(0, n - 1)]
    if not np.any(mask):
        return []
    segs: List[Tuple[int, int]] = []
    inside = False
    start = -1
    for i in range(n):
        if mask[i] and not inside:
            inside = True
            start = i
        elif not mask[i] and inside:
            segs.append((start, i - 1))
            inside = False
            start = -1
    if inside:
        # closes at end; may wrap with beginning
        if mask[0]:
            # Merge with head run
            # Find head run length
            head_end = 0
            while head_end < n and mask[head_end]:
                head_end += 1
            segs.append((start, (head_end - 1)))  # wrap
        else:
            segs.append((start, n - 1))
    # Merge first and last if both don't wrap and they touch around boundary
    if len(segs) >= 2 and segs[0][0] == 0 and segs[-1][1] == n - 1:
        s0, e0 = segs[0]
        sl, el = segs[-1]
        segs = [(sl, e0)] + segs[1:-1]
    return segs


def _score_segment(L: np.ndarray, seg: Tuple[int, int]) -> float:
    n = len(L)
    s, e = seg
    if s <= e:
        return float(L[s : e + 1].sum())
    # wrap
    return float(L[s:].sum() + L[: e + 1].sum())


def _infer_one_direction(flow: np.ndarray, times: List[str]) -> Tuple[int, int, float, str, str]:
    """
    Return (start_idx, end_idx, score, start_time, end_time)
    """
    n = len(flow)
    L = _compute_likelihood(flow)
    # Adaptive threshold
    thr = float(np.quantile(L, 0.70))
    mask = L >= thr
    segs = _segments_from_mask(mask)
    if not segs:
        # fallback: take top k seconds around the best second
        best = int(np.argmax(L))
        k = 8  # ~16s window
        s = (best - k) % n
        e = (best + k) % n
        score = float(L[best])
        return s, e, score, times[s], times[e]
    # Score segments by integral
    seg_scores = [(_score_segment(L, seg), seg) for seg in segs]
    seg_scores.sort(key=lambda t: t[0], reverse=True)
    best_score, best_seg = seg_scores[0]
    s, e = best_seg
    return s, e, float(best_score), times[s], times[e]


def infer_all() -> pd.DataFrame:
    fdf = _read_per_cycle(FLOW_CSV)
    times = fdf["time"].astype(str).tolist()
    rows: List[Dict[str, object]] = []
    for d in DIRS:
        flow = fdf[d].astype(float).to_numpy()
        s, e, score, ts, te = _infer_one_direction(flow, times)
        rows.append({
            "direction": d,
            "start_offset_s": int(s),
            "end_offset_s": int(e),
            "start_time": ts,
            "end_time": te,
            "score": float(round(score, 6)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    out = infer_all()
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote inferred intervals to {OUT_CSV}")
    for _, r in out.iterrows():
        print(f"- {r['direction']}: [{r['start_offset_s']}, {r['end_offset_s']}]  ({r['start_time']} ~ {r['end_time']})  score={r['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


