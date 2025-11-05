#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified trajectory processing pipeline orchestrator.

Steps:
  1) split_trajectories.main() -> outputs A0003_split.csv, A0008_split.csv
  2) merge_stationary_points.main() -> outputs A0003_merged.csv, A0008_merged.csv
  3) analyze_direction.main() -> outputs direction.csv
  4) refine_skeleton.main() -> outputs A0003_refined.csv, A0008_refined.csv, refined_spans.csv
  5) infer_intersection.main() -> outputs A0003_intersection.json, A0008_intersection.json

This file only orchestrates calls; internal logic of the modules is unchanged.
"""

import os
import sys
import time
import traceback

from config import BASE_DIR

# Import the processing modules
import split_trajectories
import merge_stationary_points
import analyze_direction
import refine_skeleton
import infer_intersection


def _require_files(paths: list[str]) -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Expected output not found: {missing}")


def run_pipeline() -> None:
    start_all = time.perf_counter()

    # Ensure data directory exists
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Split trajectories
    print("[1/5] Splitting trajectories ...")
    t0 = time.perf_counter()
    try:
        # Clean previous outputs for this step (including excluded files)
        for p in [
            os.path.join(BASE_DIR, "data", "A0003_split.csv"),
            os.path.join(BASE_DIR, "data", "A0008_split.csv"),
            os.path.join(BASE_DIR, "data", "A0003_excluded.csv"),
            os.path.join(BASE_DIR, "data", "A0008_excluded.csv"),
        ]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        split_trajectories.main()
        _require_files([
            os.path.join(BASE_DIR, "data", "A0003_split.csv"),
            os.path.join(BASE_DIR, "data", "A0008_split.csv"),
        ])
    except Exception:
        print("Split step failed:")
        traceback.print_exc()
        sys.exit(1)
    print(f"[1/5] Done in {time.perf_counter() - t0:.2f}s\n")

    # Step 2: Merge stationary points
    print("[2/5] Merging stationary points ...")
    t0 = time.perf_counter()
    try:
        # Clean previous outputs for this step
        for p in [
            os.path.join(BASE_DIR, "data", "A0003_merged.csv"),
            os.path.join(BASE_DIR, "data", "A0008_merged.csv"),
        ]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        merge_stationary_points.main()
        _require_files([
            os.path.join(BASE_DIR, "data", "A0003_merged.csv"),
            os.path.join(BASE_DIR, "data", "A0008_merged.csv"),
        ])
    except Exception:
        print("Merge step failed:")
        traceback.print_exc()
        sys.exit(1)
    print(f"[2/5] Done in {time.perf_counter() - t0:.2f}s\n")

    # Step 3: Analyze direction
    print("[3/5] Analyzing directions ...")
    t0 = time.perf_counter()
    try:
        # Clean previous outputs for this step
        p = os.path.join(BASE_DIR, "data", "direction.csv")
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        analyze_direction.main()
        _require_files([
            os.path.join(BASE_DIR, "data", "direction.csv"),
        ])
    except Exception:
        print("Direction analysis step failed:")
        traceback.print_exc()
        sys.exit(1)
    print(f"[3/5] Done in {time.perf_counter() - t0:.2f}s\n")

    # Step 4: Refine trajectories
    print("[4/5] Refining trajectories ...")
    t0 = time.perf_counter()
    try:
        # Clean previous outputs for this step
        for p in [
            os.path.join(BASE_DIR, "data", "A0003_refined.csv"),
            os.path.join(BASE_DIR, "data", "A0008_refined.csv"),
            os.path.join(BASE_DIR, "data", "refined_spans.csv"),
        ]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        refine_skeleton.main()
        _require_files([
            os.path.join(BASE_DIR, "data", "A0003_refined.csv"),
            os.path.join(BASE_DIR, "data", "A0008_refined.csv"),
            os.path.join(BASE_DIR, "data", "refined_spans.csv"),
        ])
    except Exception:
        print("Refine step failed:")
        traceback.print_exc()
        sys.exit(1)
    print(f"[4/5] Done in {time.perf_counter() - t0:.2f}s\n")

    # Step 5: Infer road network at intersections
    print("[5/5] Inferring road network at intersections ...")
    t0 = time.perf_counter()
    try:
        # Clean previous outputs for this step
        for p in [
            os.path.join(BASE_DIR, "data", "A0003_intersection.json"),
            os.path.join(BASE_DIR, "data", "A0008_intersection.json"),
            os.path.join(BASE_DIR, "data", "A0003_intersection.geojson"),
            os.path.join(BASE_DIR, "data", "A0008_intersection.geojson"),
        ]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        infer_intersection.main()
        _require_files([
            os.path.join(BASE_DIR, "data", "A0003_intersection.json"),
            os.path.join(BASE_DIR, "data", "A0008_intersection.json"),
        ])
    except Exception:
        print("Intersection inference step failed:")
        traceback.print_exc()
        sys.exit(1)
    print(f"[5/5] Done in {time.perf_counter() - t0:.2f}s\n")

    print(f"All steps finished in {time.perf_counter() - start_all:.2f}s")


if __name__ == "__main__":
    run_pipeline()


