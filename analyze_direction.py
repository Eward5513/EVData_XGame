#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze driving direction for each segment in A0003_split.csv and A0008_split.csv
Classify directions based on trajectory relative to intersection center:
A1: North-South straight
A2: Left turn toward North/South (E->S or W->N)
B1: East-West straight  
B2: Left turn toward East/West (N->E or S->W)
C: Other directions (including right turns, U-turns, etc.)

Intersection centers (per road):
- A0003: lon=123.152480, lat=32.345120
- A0008: lon=123.181261, lat=32.327137
"""

import pandas as pd
import numpy as np
import math

# Default intersection center (fallback; functions now take explicit center parameters)
INTERSECTION_CENTER_LON = 123.152480
INTERSECTION_CENTER_LAT = 32.345120

# Radius for using near-center steps to classify turns (A2/B2) and as near-center anchor threshold
TURN_CLASSIFY_RADIUS_M = 200.0

# Ignore extremely short steps in direction judgement (meters)
MIN_STEP_DISTANCE_M = 3.0

# Two-point special handling thresholds
TWO_POINT_MIN_ANGLE_DELTA_DEG = 45.0       # minimum polar-angle change to consider a turn
TWO_POINT_STRAIGHT_MAX_DELTA_DEG = 20.0    # maximum angle change to be considered straight
TWO_POINT_MAX_RADIUS_M = 150.0             # both points should be reasonably near the center
#############################################
# Manual direction overrides (hardcoded)
# Key: (road_id, vehicle_id, date, seg_id) -> value in {'A1','A2','B1','B2','C'}
# Example:
# MANUAL_DIRECTION_OVERRIDES = {
#     ('A0003', 543, '2024-06-20', 0): 'B1',
# }
MANUAL_DIRECTION_OVERRIDES: dict[tuple[str, int, str, int], str] = {}


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing between two points (degrees)
    Return range: 0-360 degrees, 0 degrees is North
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)
    
    y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    bearing_deg = (bearing_deg + 360) % 360
    
    return bearing_deg

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points (meters)
    Using simplified planar coordinate calculation (suitable for small areas)
    """
    # Convert lat/lon differences to meters (rough calculation)
    lat_diff = (lat2 - lat1) * 111000  # 1 degree latitude ≈ 111km
    lon_diff = (lon2 - lon1) * 111000 * math.cos(math.radians((lat1 + lat2) / 2))
    
    return math.sqrt(lat_diff**2 + lon_diff**2)

def _segment_plane_coords(i1: int, i2: int, lats, lons, center_lat: float, center_lon: float) -> tuple[float, float, float, float]:
    """Return local-plane coordinates (meters) of segment endpoints relative to center: (x1,y1,x2,y2)."""
    avg_lat1 = (lats[i1] + center_lat) / 2.0
    avg_lat2 = (lats[i2] + center_lat) / 2.0
    x1 = (lons[i1] - center_lon) * 111000.0 * math.cos(math.radians(avg_lat1))
    y1 = (lats[i1] - center_lat) * 111000.0
    x2 = (lons[i2] - center_lon) * 111000.0 * math.cos(math.radians(avg_lat2))
    y2 = (lats[i2] - center_lat) * 111000.0
    return x1, y1, x2, y2

def center_min_distance_to_segment_m(i1: int, i2: int, lats, lons, center_lat: float, center_lon: float) -> float:
    """Minimum distance from center to segment (i1->i2) in meters."""
    x1, y1, x2, y2 = _segment_plane_coords(i1, i2, lats, lons, center_lat, center_lon)
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return math.hypot(x1, y1)
    t = max(0.0, min(1.0, (-(x1) * vx + (-(y1)) * vy) / denom))
    px, py = x1 + t * vx, y1 + t * vy
    return math.hypot(px, py)

def segment_projection_contains_center(i1: int, i2: int, lats, lons, center_lat: float, center_lon: float) -> bool:
    """Whether perpendicular projection of center onto segment (i1->i2) falls within the open segment (0<t<1)."""
    x1, y1, x2, y2 = _segment_plane_coords(i1, i2, lats, lons, center_lat, center_lon)
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return False
    t = (-(x1) * vx + (-(y1)) * vy) / denom
    return 0.0 < t < 1.0

def polar_angle_deg_from_center(lat: float, lon: float, center_lat: float, center_lon: float) -> float:
    """
    Polar angle (degrees in [0,360)) of point relative to intersection center,
    with 0°=East, 90°=North, 180°=West, 270°=South.
    """
    avg_lat = (lat + center_lat) / 2.0
    dx_m = (lon - center_lon) * 111000.0 * math.cos(math.radians(avg_lat))
    dy_m = (lat - center_lat) * 111000.0
    theta = math.degrees(math.atan2(dy_m, dx_m))
    return (theta + 360.0) % 360.0

def minimal_angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (degrees)."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return abs(d)

def quadrant_from_polar(angle_deg: float) -> str:
    """
    Map polar angle to quadrant with hard splits:
    E: [-45°,45°) U [315°,360°)
    N: [45°,135°)
    W: [135°,225°)
    S: [225°,315°)
    """
    a = angle_deg % 360.0
    if a >= 315.0 or a < 45.0:
        return 'E'
    if 45.0 <= a < 135.0:
        return 'N'
    if 135.0 <= a < 225.0:
        return 'W'
    return 'S'

def angle_to_cardinal(angle_deg: float, tol: float = 45.0) -> str | None:
    """
    Map angle to nearest cardinal among {E(0), N(90), W(180), S(270)} within tolerance.
    Returns one of 'E','N','W','S' or None if outside tolerance windows.
    """
    # Normalize
    a = angle_deg % 360.0
    targets = [(0.0, 'E'), (90.0, 'N'), (180.0, 'W'), (270.0, 'S')]
    best = None
    best_diff = 1e9
    for t, label in targets:
        diff = minimal_angle_diff_deg(a, t)
        if diff < best_diff:
            best_diff = diff
            best = label
    if best_diff <= tol:
        return best  # type: ignore[return-value]
    return None

def classify_two_point_by_polar_angle(trajectory_df, center_lat: float, center_lon: float) -> str:
    """
    Two-point classification using quadrant-based polar angles:
    - Straight (no threshold): opposite sides along same axis → A1 (N/S) or B1 (E/W)
    - Left turns (require sufficient angle change):
        * W→N or E→S → A2
        * N→E or S→W → B2
    - Otherwise C
    """
    start = trajectory_df.iloc[0]
    end = trajectory_df.iloc[1]

    a1 = polar_angle_deg_from_center(start['latitude'], start['longitude'], center_lat, center_lon)
    a2 = polar_angle_deg_from_center(end['latitude'], end['longitude'], center_lat, center_lon)
    delta = minimal_angle_diff_deg(a1, a2)

    q1 = quadrant_from_polar(a1)
    q2 = quadrant_from_polar(a2)

    # Straight across center: opposite sides on same axis
    if q1 != q2:
        if {q1, q2} == {'E', 'W'}:
            return 'B1'
        if {q1, q2} == {'N', 'S'}:
            return 'A1'

    # Left turn: require sufficient angle change
    if delta >= TWO_POINT_MIN_ANGLE_DELTA_DEG:
        if (q1 == 'W' and q2 == 'N') or (q1 == 'E' and q2 == 'S'):
            return 'A2'
        if (q1 == 'N' and q2 == 'E') or (q1 == 'S' and q2 == 'W'):
            return 'B2'

    return 'C'

def _step_axis_label(lats, lons, i: int, j: int, axis_ratio: float = 1.4) -> str | None:
    """
    Label a step by dominant axis using dx/dy in meters.
    Returns one of {'N','S','E','W'} or None if no dominant axis or too short.
    """
    if i < 0 or j >= len(lats) or j <= i:
        return None
    avg = (lats[i] + lats[j]) / 2.0
    dy = (lats[j] - lats[i]) * 111000.0
    dx = (lons[j] - lons[i]) * 111000.0 * math.cos(math.radians(avg))
    if math.hypot(dx, dy) < MIN_STEP_DISTANCE_M:
        return None
    abs_x = abs(dx)
    abs_y = abs(dy)
    if abs_y >= axis_ratio * abs_x:
        return 'N' if dy > 0 else 'S'
    if abs_x >= axis_ratio * abs_y:
        return 'E' if dx > 0 else 'W'
    return None

def _axis_from_label(lab: str | None) -> str | None:
    if lab is None:
        return None
    return 'NS' if lab in ('N', 'S') else ('EW' if lab in ('E', 'W') else None)

def _is_left_turn(from_dir: str, to_dir: str) -> bool:
    """
    Check if turning from from_dir to to_dir is a left turn (counterclockwise).
    Left turns (A2 for NS-axis destination, B2 for EW-axis destination):
    - E→S, W→N (A2: turn to NS axis)
    - N→E, S→W (B2: turn to EW axis)
    """
    left_turns = {('E', 'S'), ('W', 'N'), ('N', 'E'), ('S', 'W')}
    return (from_dir, to_dir) in left_turns

def _classify_turn(prev_lab: str | None, next_lab: str | None) -> str | None:
    """
    Classify turn based on entrance and exit directions.
    Returns A1 (NS straight), A2 (left to NS), B1 (EW straight), B2 (left to EW), or None.
    """
    if not prev_lab or not next_lab:
        return None
    
    prev_ax = _axis_from_label(prev_lab)
    next_ax = _axis_from_label(next_lab)
    
    # Straight through: same axis AND (same direction OR opposite direction)
    if prev_ax == next_ax:
        # Check if directions are valid for straight
        straight_pairs = {
            ('N', 'N'), ('N', 'S'), ('S', 'N'), ('S', 'S'),  # NS axis
            ('E', 'E'), ('E', 'W'), ('W', 'E'), ('W', 'W')   # EW axis
        }
        if (prev_lab, next_lab) in straight_pairs:
            return 'A1' if prev_ax == 'NS' else 'B1'
        # Same axis but invalid combination (shouldn't happen with current logic)
        return None
    
    # Turn: check if it's a left turn
    if _is_left_turn(prev_lab, next_lab):
        # Determine category by destination axis
        return 'A2' if next_ax == 'NS' else 'B2'
    
    # Right turn or U-turn → C
    return None

def classify_near_center_multistep(trajectory_df, center_lat: float, center_lon: float,
                                   near_radius_m: float = 50.0,
                                   axis_ratio: float = 1.4) -> str | None:
    """
    Near-center multi-step classification per spec:
    1) Find the nearest step (segment). If center projection falls within that segment, use the
       previous step, the center step, and the next step to determine dominant axes:
       - If all three have the same dominant axis → straight: A1 (NS) or B1 (EW)
       - If the dominant axes of previous and next differ → turn: A2 (EW→NS) or B2 (NS→EW)
       - If prev/next label missing, keep looking one more step outward.
    2) Else (nearest is a point): pick the point nearest to center, consider two steps before
       and two steps after. Determine prev/center/next axes similarly and apply the same rules.
    Returns a direction label or None if undecidable.
    """
    if len(trajectory_df) < 2:
        return None
    df = trajectory_df.sort_values('collectiontime').reset_index(drop=True)
    lats = df['latitude'].to_numpy()
    lons = df['longitude'].to_numpy()
    n = len(lats)

    # Find nearest step and whether projection of center falls within it
    best_k, best_d_step, best_proj_in = None, float('inf'), False
    for i in range(n - 1):
        dmin = center_min_distance_to_segment_m(i, i + 1, lats, lons, center_lat, center_lon)
        if dmin < best_d_step:
            best_d_step = dmin
            best_k = i
            best_proj_in = segment_projection_contains_center(i, i + 1, lats, lons, center_lat, center_lon)

    # Find nearest point
    best_p, best_d_point = None, float('inf')
    for i in range(n):
        # reuse polar plane distance
        avg_lat = (lats[i] + center_lat) / 2.0
        dx = (lons[i] - center_lon) * 111000.0 * math.cos(math.radians(avg_lat))
        dy = (lats[i] - center_lat) * 111000.0
        d = math.hypot(dx, dy)
        if d < best_d_point:
            best_d_point = d
            best_p = i

    def find_prev_label_from(idx: int) -> str | None:
        for p in range(idx - 1, -1, -1):
            lab = _step_axis_label(lats, lons, p, p + 1, axis_ratio)
            if lab:
                return lab
        return None

    def find_next_label_from(idx: int) -> str | None:
        for q in range(idx, n - 1):
            lab = _step_axis_label(lats, lons, q, q + 1, axis_ratio)
            if lab:
                return lab
        return None

    # Case 1: step is valid and nearest within near radius
    if best_k is not None and best_proj_in and best_d_step <= near_radius_m:
        prev_lab = find_prev_label_from(best_k)
        next_lab = find_next_label_from(best_k + 1)
        result = _classify_turn(prev_lab, next_lab)
        if result:
            return result

    # Case 2: nearest is a point (or step projection not inside)
    if best_p is not None and best_d_point <= near_radius_m:
        k = best_p
        # prev side: prefer (k-1,k) then (k-2,k-1)
        prev_lab = None
        if k - 1 >= 0:
            prev_lab = _step_axis_label(lats, lons, k - 1, k, axis_ratio)
        if prev_lab is None and k - 2 >= 0:
            prev_lab = _step_axis_label(lats, lons, k - 2, k - 1, axis_ratio)
        if prev_lab is None:
            prev_lab = find_prev_label_from(k)

        # next side: prefer (k,k+1) then (k+1,k+2)
        next_lab = None
        if k + 1 < n:
            next_lab = _step_axis_label(lats, lons, k, k + 1, axis_ratio)
        if next_lab is None and k + 2 < n:
            next_lab = _step_axis_label(lats, lons, k + 1, k + 2, axis_ratio)
        if next_lab is None:
            next_lab = find_next_label_from(k + 1)

        result = _classify_turn(prev_lab, next_lab)
        if result:
            return result

    return None

def analyze_trajectory_direction(trajectory_df, center_lat: float, center_lon: float):
    """
    Analyze driving direction of a single vehicle trajectory
    Based on start and end points relative to intersection center
    """
    # Filter out invalid data
    trajectory_df = trajectory_df.copy()
    trajectory_df = trajectory_df.sort_values('collectiontime')
    
    if len(trajectory_df) < 2:
        return 'C'  # Too few data points
    if len(trajectory_df) == 2:
        return classify_two_point_by_polar_angle(trajectory_df, center_lat, center_lon)
    if len(trajectory_df) == 3:
        # Use start and end points as a two-point segment for classification
        return classify_two_point_by_polar_angle(trajectory_df.iloc[[0, -1]], center_lat, center_lon)
    
    # Get start and end points
    start_point = trajectory_df.iloc[0]
    end_point = trajectory_df.iloc[-1]
    
    # Calculate total displacement
    total_distance = calculate_distance(
        start_point['latitude'], start_point['longitude'],
        end_point['latitude'], end_point['longitude']
    )
    
    if total_distance < 10:  # Total displacement < 10m, consider as stationary or irregular
        return 'C'

    # Use near-center multi-step classification with 50m threshold only
    result = classify_near_center_multistep(trajectory_df, center_lat, center_lon,
                                            near_radius_m=50.0, axis_ratio=1.4)
    # Return result if valid, otherwise default to 'C'
    return result if result else 'C'

def process_dataframe(df: pd.DataFrame, source_label: str, center_lat: float, center_lon: float) -> list[dict]:
    print(f"Grouping {source_label} by vehicle_id, date, seg_id ...")
    records: list[dict] = []
    for (vehicle_id, date, seg_id), segment in df.groupby(['vehicle_id', 'date', 'seg_id']):
        # Manual override first
        road_id = segment['road_id'].iloc[0] if 'road_id' in segment.columns else None
        manual_key = (str(road_id) if road_id is not None else '', int(vehicle_id), str(date), int(seg_id))
        if manual_key in MANUAL_DIRECTION_OVERRIDES:
            direction = MANUAL_DIRECTION_OVERRIDES[manual_key]
        else:
            direction = analyze_trajectory_direction(segment, center_lat, center_lon)
        records.append({
            'vehicle_id': vehicle_id,
            'date': date,
            'seg_id': int(seg_id),
            'road_id': road_id,
            'direction': direction,
        })
    return records

def main():
    print("Starting to read split trajectory data files...")
    
    # Read pre-split data files
    print("Reading A0003_split.csv...")
    df1 = pd.read_csv('/home/tzhang174/EVData_XGame/A0003_split.csv')
    print(f"A0003_split.csv: {len(df1)} records")
    
    print("Reading A0008_split.csv...")
    df2 = pd.read_csv('/home/tzhang174/EVData_XGame/A0008_split.csv')
    print(f"A0008_split.csv: {len(df2)} records")
    
    # Validate columns per source (process separately; do not combine)
    required_cols = ['vehicle_id', 'date', 'seg_id', 'collectiontime', 'latitude', 'longitude']
    for name, dframe in (('A0003', df1), ('A0008', df2)):
        missing = [c for c in required_cols if c not in dframe.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing required columns in {name} split data: {missing}")
    
    # Intersection centers per road
    centers: dict[str, tuple[float, float]] = {
        'A0003': (32.345120, 123.152480),  # (lat, lon)
        'A0008': (32.327137, 123.181261),  # (lat, lon)
    }

    # Analyze each source independently with its corresponding center, then merge results
    records = []
    c3_lat, c3_lon = centers['A0003']
    c8_lat, c8_lon = centers['A0008']
    records.extend(process_dataframe(df1, 'A0003', c3_lat, c3_lon))
    records.extend(process_dataframe(df2, 'A0008', c8_lat, c8_lon))

    df_output = pd.DataFrame.from_records(records)
    if len(df_output) == 0:
        print("No segments found in input files.")
        return

    # Sort output (include road_id to keep intersections distinct)
    sort_cols = ['vehicle_id', 'date', 'road_id', 'seg_id']
    df_output = df_output.sort_values(sort_cols).reset_index(drop=True)

    # Save results
    output_file = '/home/tzhang174/EVData_XGame/direction.csv'
    df_output.to_csv(output_file, index=False)
    print(f"\n=== Results Saved ===")
    print(f"Output file: {output_file}")
    print(f"Total trajectory segments: {len(df_output)}")

    # Direction counts
    print("\n=== Direction Classification Statistics (by segment) ===")
    for direction, count in df_output['direction'].value_counts().items():
        print(f"{direction}: {count} segments")

    # Show sample results
    print("\n=== Sample Results (First 10 rows) ===")
    sample_df = df_output.head(10)
    print(sample_df.to_string(index=False))

if __name__ == "__main__":
    main()
