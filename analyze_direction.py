#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze driving direction for each vehicle trajectory in A0003.csv and A0008.csv
Classify directions based on trajectory relative to intersection center:
A1: North-South straight
A2: Left turn toward North/South (E->S or W->N)
B1: East-West straight  
B2: Left turn toward East/West (N->E or S->W)
C: Other directions (including right turns, U-turns, etc.)

Intersection center: longitude=123.152578, latitude=32.345267
"""

import pandas as pd
import numpy as np
import math

# Intersection center coordinates
INTERSECTION_CENTER_LON = 123.152578
INTERSECTION_CENTER_LAT = 32.345267

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

def get_direction_from_center(lat, lon):
    """
    Determine the direction of a point relative to intersection center
    Returns: 'N', 'E', 'S', 'W' or None
    """
    bearing = calculate_bearing(INTERSECTION_CENTER_LAT, INTERSECTION_CENTER_LON, lat, lon)
    
    # North: 315-45 degrees, East: 45-135 degrees, South: 135-225 degrees, West: 225-315 degrees
    if (bearing >= 315 or bearing < 45):
        return 'N'
    elif (bearing >= 45 and bearing < 135):
        return 'E'
    elif (bearing >= 135 and bearing < 225):
        return 'S'
    elif (bearing >= 225 and bearing < 315):
        return 'W'
    return None

def split_trajectory_by_time_gap(trajectory_df, max_gap_seconds=180):
    """
    Split trajectory into segments based on time gaps
    If the time interval between two consecutive points exceeds max_gap_seconds (default 180s = 3 minutes),
    split the trajectory into separate segments
    
    Args:
        trajectory_df: DataFrame with trajectory data sorted by collectiontime
        max_gap_seconds: Maximum time gap in seconds before splitting (default 180)
    
    Returns:
        List of DataFrames, each representing a trajectory segment
    """
    if len(trajectory_df) < 2:
        return [trajectory_df]
    
    trajectory_df = trajectory_df.copy()
    # Keep original indices so we can map segments back to the original dataframe
    trajectory_df = trajectory_df.sort_values('collectiontime')
    
    # Calculate time differences between consecutive points (in seconds)
    time_diffs = trajectory_df['collectiontime'].diff() / 1000.0  # Convert ms to seconds

    # Adaptive splitting: split when current gap > 2x median of previous gaps within current segment
    segments = []
    start_idx = 0
    prev_diffs = []  # seconds, within current segment only

    for i in range(1, len(trajectory_df)):
        cur_diff = time_diffs.iloc[i]
        if pd.isna(cur_diff):
            continue

        should_split = False

        if len(prev_diffs) >= 3:
            baseline = float(np.median(prev_diffs))
            if baseline > 0 and float(cur_diff) > 2.0 * baseline:
                should_split = True

        # Also honor absolute gap threshold if provided
        if float(cur_diff) > float(max_gap_seconds):
            should_split = True

        if should_split:
            segment = trajectory_df.iloc[start_idx:i].copy()
            if len(segment) >= 2:  # Only keep segments with at least 2 points
                segments.append(segment)
            start_idx = i
            prev_diffs = []  # reset baseline for new segment
        else:
            prev_diffs.append(float(cur_diff))

    # Add the last segment
    last_segment = trajectory_df.iloc[start_idx:].copy()
    if len(last_segment) >= 2:
        segments.append(last_segment)

    # If no valid segments, return original trajectory
    if len(segments) == 0:
        return [trajectory_df]

    return segments

def analyze_trajectory_direction(trajectory_df):
    """
    Analyze driving direction of a single vehicle trajectory
    Based on start and end points relative to intersection center
    """
    # Filter out invalid data
    trajectory_df = trajectory_df.copy()
    trajectory_df = trajectory_df.sort_values('collectiontime')
    
    if len(trajectory_df) < 2:
        return 'C'  # Too few data points
    
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

    # Step-based classification using consecutive displacements (no center dependency)
    lats = trajectory_df['latitude'].to_numpy()
    lons = trajectory_df['longitude'].to_numpy()
    n_points = len(lats)

    dx_steps: list[float] = []
    dy_steps: list[float] = []
    for i in range(1, n_points):
        avg_lat_pair = (lats[i - 1] + lats[i]) / 2.0
        dy_i = (lats[i] - lats[i - 1]) * 111000.0
        dx_i = (lons[i] - lons[i - 1]) * 111000.0 * math.cos(math.radians(avg_lat_pair))
        if abs(dx_i) < 0.5 and abs(dy_i) < 0.5:  # filter jitter
            continue
        dx_steps.append(dx_i)
        dy_steps.append(dy_i)

    def label_step(dx: float, dy: float, axis_ratio_step: float = 1.4):
        abs_x = abs(dx)
        abs_y = abs(dy)
        if abs_y >= axis_ratio_step * abs_x:
            return 'N' if dy > 0 else 'S'
        if abs_x >= axis_ratio_step * abs_y:
            return 'E' if dx > 0 else 'W'
        return None

    step_labels = [label_step(dx, dy) for dx, dy in zip(dx_steps, dy_steps)]
    step_labels = [lab for lab in step_labels if lab is not None]

    if len(step_labels) > 0:
        # Compress consecutive duplicates
        compressed: list[str] = []
        for lab in step_labels:
            if len(compressed) == 0 or lab != compressed[-1]:
                compressed.append(lab)

        if len(compressed) == 1:
            # Single-axis movement → straight
            if compressed[0] in ('N', 'S'):
                return 'A1'
            else:
                return 'B1'

        # Two-phase axis change → potential left turn by mapping
        first_lab = compressed[0]
        last_lab = compressed[-1]

        # A2: left turn toward North/South (E->N or W->S)
        if (first_lab in ('E', 'W') and last_lab in ('N', 'S')):
            if (first_lab == 'E' and last_lab == 'N') or (first_lab == 'W' and last_lab == 'S'):
                return 'A2'

        # B2: left turn toward East/West (N->W or S->E)
        if (first_lab in ('N', 'S') and last_lab in ('E', 'W')):
            if (first_lab == 'N' and last_lab == 'W') or (first_lab == 'S' and last_lab == 'E'):
                return 'B2'

    # Net displacement fallback (no center dependency)
    if len(dx_steps) > 0:
        dx_total = float(sum(dx_steps))
        dy_total = float(sum(dy_steps))
    else:
        avg_lat = (start_point['latitude'] + end_point['latitude']) / 2.0
        dy_total = (end_point['latitude'] - start_point['latitude']) * 111000.0
        dx_total = (end_point['longitude'] - start_point['longitude']) * 111000.0 * math.cos(math.radians(avg_lat))

    axis_ratio_total = 1.5
    if abs(dy_total) >= axis_ratio_total * abs(dx_total):
        return 'A1'
    if abs(dx_total) >= axis_ratio_total * abs(dy_total):
        return 'B1'

    return 'C'

def main():
    print("Starting to read trajectory data files...")
    
    # Read both data files
    print("Reading A0003.csv...")
    df1 = pd.read_csv('/home/tzhang174/EVData_XGame/A0003.csv')
    print(f"A0003.csv: {len(df1)} records")
    
    print("Reading A0008.csv...")
    df2 = pd.read_csv('/home/tzhang174/EVData_XGame/A0008.csv')
    print(f"A0008.csv: {len(df2)} records")
    
    # Combine the dataframes
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined total: {len(df)} trajectory records")
    
    # Add direction column (unassigned initially)
    df['direction'] = None
    
    # Group by vehicle_id and date to create unique trajectories
    df['trajectory_id'] = df['vehicle_id'].astype(str) + '_' + df['date']
    trajectory_ids = df['trajectory_id'].unique()
    print(f"Found {len(trajectory_ids)} unique trajectories (vehicle_id + date)")
    
    # Add segment_id column for tracking split segments (unassigned = -1)
    df['segment_id'] = -1
    
    # Statistics
    total_segments = 0
    trajectories_split = 0
    
    # Analyze each trajectory
    for i, trajectory_id in enumerate(trajectory_ids):
        if i % 100 == 0:
            print(f"Processing trajectory {i+1}/{len(trajectory_ids)} (ID: {trajectory_id})")
        
        # Get trajectory data for this vehicle on this date
        trajectory_data = df[df['trajectory_id'] == trajectory_id].copy()
        
        # Split trajectory by time gaps (3 minutes = 180 seconds)
        segments = split_trajectory_by_time_gap(trajectory_data, max_gap_seconds=180)
        
        # Track if this trajectory was split
        if len(segments) > 1:
            trajectories_split += 1
        
        total_segments += len(segments)
        
        # Analyze direction for each segment
        for seg_idx, segment in enumerate(segments):
            direction = analyze_trajectory_direction(segment)
            
            # Update dataframe with segment_id and direction
            segment_indices = segment.index
            df.loc[segment_indices, 'segment_id'] = seg_idx
            df.loc[segment_indices, 'direction'] = direction
    
    # Print trajectory splitting statistics
    print(f"\n=== Trajectory Splitting Statistics ===")
    print(f"Total original trajectories: {len(trajectory_ids)}")
    print(f"Trajectories split due to time gaps (>3 min): {trajectories_split}")
    print(f"Total segments after splitting: {total_segments}")
    print(f"Average segments per trajectory: {total_segments / len(trajectory_ids):.2f}")
    
    # Statistics by segment (only for valid segments with segment_id >= 0)
    valid_segment_rows = df[df['segment_id'] >= 0]
    segment_direction_counts = valid_segment_rows.groupby(['trajectory_id', 'segment_id'])['direction'].first().value_counts()
    print("\n=== Direction Classification Statistics (by segment) ===")
    for direction, count in segment_direction_counts.items():
        print(f"{direction}: {count} segments")
    
    # Create output dataframe with segments (one row per vehicle_id + date + segment_id)
    df_output = valid_segment_rows.groupby(['trajectory_id', 'segment_id']).agg({
        'vehicle_id': 'first',
        'date': 'first', 
        'road_id': 'first',
        'direction': 'first'
    }).reset_index()
    
    # Rename segment_id to be more descriptive in output
    df_output = df_output.rename(columns={'segment_id': 'trajectory_segment'})
    df_output = df_output[['vehicle_id', 'date', 'trajectory_segment', 'road_id', 'direction']]
    
    # Sort by vehicle_id and date
    df_output = df_output.sort_values(['vehicle_id', 'date']).reset_index(drop=True)
    
    # Save results
    output_file = '/home/tzhang174/EVData_XGame/direction.csv'
    df_output.to_csv(output_file, index=False)
    print(f"\n=== Results Saved ===")
    print(f"Output file: {output_file}")
    print(f"Total trajectory segments: {len(df_output)}")
    print(f"Note: Trajectories with time gaps >3 minutes are split into separate segments")
    
    # Show sample results
    print("\n=== Sample Results (First 10 rows) ===")
    sample_df = df_output.head(10)
    print(sample_df.to_string(index=False))

if __name__ == "__main__":
    main()
