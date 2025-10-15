#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze driving direction for each vehicle trajectory in A0003.csv and A0008.csv
Classify directions based on trajectory relative to intersection center:
A1: North-South straight
A2: North-South left turn
A3: East-West straight  
A4: East-West left turn
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
    
    # Determine the direction of start and end points relative to intersection center
    start_direction = get_direction_from_center(start_point['latitude'], start_point['longitude'])
    end_direction = get_direction_from_center(end_point['latitude'], end_point['longitude'])
    
    if start_direction is None or end_direction is None:
        return 'C'
    
    # Classify based on entry and exit directions
    # A1: North-South straight (N->S or S->N)
    if (start_direction == 'N' and end_direction == 'S') or (start_direction == 'S' and end_direction == 'N'):
        return 'A1'
    
    # A2: North-South left turn (N->E or S->W)
    elif (start_direction == 'N' and end_direction == 'E') or (start_direction == 'S' and end_direction == 'W'):
        return 'A2'
    
    # A3: East-West straight (E->W or W->E)
    elif (start_direction == 'E' and end_direction == 'W') or (start_direction == 'W' and end_direction == 'E'):
        return 'A3'
    
    # A4: East-West left turn (E->N or W->S)
    elif (start_direction == 'E' and end_direction == 'N') or (start_direction == 'W' and end_direction == 'S'):
        return 'A4'
    
    # All other cases (right turns, U-turns, same direction, etc.)
    else:
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
    
    # Add direction column
    df['direction'] = 'C'  # Default value
    
    # Group by vehicle_id and date to create unique trajectories
    df['trajectory_id'] = df['vehicle_id'].astype(str) + '_' + df['date']
    trajectory_ids = df['trajectory_id'].unique()
    print(f"Found {len(trajectory_ids)} unique trajectories (vehicle_id + date)")
    
    # Analyze each trajectory
    for i, trajectory_id in enumerate(trajectory_ids):
        if i % 100 == 0:
            print(f"Processing trajectory {i+1}/{len(trajectory_ids)} (ID: {trajectory_id})")
        
        # Get trajectory data for this vehicle on this date
        trajectory_data = df[df['trajectory_id'] == trajectory_id].copy()
        
        # Analyze direction
        direction = analyze_trajectory_direction(trajectory_data)
        
        # Update dataframe
        df.loc[df['trajectory_id'] == trajectory_id, 'direction'] = direction
    
    # Statistics
    direction_counts = df.groupby('trajectory_id')['direction'].first().value_counts()
    print("\nDirection classification statistics:")
    for direction, count in direction_counts.items():
        print(f"{direction}: {count} trajectories")
    
    # Create output dataframe with unique trajectories only (one row per vehicle_id + date)
    df_output = df.groupby('trajectory_id').agg({
        'vehicle_id': 'first',
        'date': 'first', 
        'road_id': 'first',
        'direction': 'first'
    }).reset_index(drop=True)
    
    # Sort by vehicle_id and date
    df_output = df_output.sort_values(['vehicle_id', 'date']).reset_index(drop=True)
    
    # Save results
    output_file = '/home/tzhang174/EVData_XGame/direction.csv'
    df_output.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Total unique trajectories: {len(df_output)}")
    
    # Show sample results
    print("\nFirst 10 rows sample:")
    sample_df = df_output.head(10)
    print(sample_df.to_string(index=False))

if __name__ == "__main__":
    main()
