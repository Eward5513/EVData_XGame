#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze driving direction for each vehicle trajectory in A0003.csv and A0008.csv
Classify directions based on longitude/latitude changes:
A1: North-South straight
A2: North-South left turn
A3: East-West straight  
A4: East-West left turn
C: Other directions (including right turns, U-turns, etc.)
"""

import pandas as pd
import numpy as np
import math

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

def analyze_trajectory_direction(trajectory_df):
    """
    Analyze driving direction of a single vehicle trajectory
    """
    # Filter out invalid data and stationary states based on movement distance
    trajectory_df = trajectory_df.copy()
    trajectory_df = trajectory_df.sort_values('collectiontime')
    
    # Filter out stationary points based on distance between consecutive points
    if len(trajectory_df) < 2:
        return 'C'  # Too few data points
    
    # Calculate distances between consecutive points and keep only moving segments
    valid_indices = [0]  # Always keep the first point
    
    for i in range(1, len(trajectory_df)):
        prev_point = trajectory_df.iloc[valid_indices[-1]]
        curr_point = trajectory_df.iloc[i]
        
        distance = calculate_distance(
            prev_point['latitude'], prev_point['longitude'],
            curr_point['latitude'], curr_point['longitude']
        )
        
        # If distance > 5 meters, consider it as movement
        if distance > 5:
            valid_indices.append(i)
    
    # Filter trajectory to only include moving points
    trajectory_df = trajectory_df.iloc[valid_indices]
    
    if len(trajectory_df) < 3:
        return 'C'  # Too few movement points
    
    # Sort by time
    trajectory_df = trajectory_df.sort_values('collectiontime')
    
    # Get start and end points
    start_point = trajectory_df.iloc[0]
    end_point = trajectory_df.iloc[-1]
    
    # Calculate total displacement
    total_distance = calculate_distance(
        start_point['latitude'], start_point['longitude'],
        end_point['latitude'], end_point['longitude']
    )
    
    if total_distance < 10:  # Total displacement < 10m, consider as stationary
        return 'C'
    
    # Calculate overall direction
    overall_bearing = calculate_bearing(
        start_point['latitude'], start_point['longitude'],
        end_point['latitude'], end_point['longitude']
    )
    
    # Analyze trajectory curvature
    # Calculate actual path length
    actual_path_length = 0
    for i in range(1, len(trajectory_df)):
        prev_point = trajectory_df.iloc[i-1]
        curr_point = trajectory_df.iloc[i]
        segment_distance = calculate_distance(
            prev_point['latitude'], prev_point['longitude'],
            curr_point['latitude'], curr_point['longitude']
        )
        actual_path_length += segment_distance
    
    # Calculate curvature ratio (actual path length / straight line distance)
    if total_distance > 0:
        curvature_ratio = actual_path_length / total_distance
    else:
        curvature_ratio = 1.0
    
    # Analyze direction changes
    bearings = []
    for i in range(1, len(trajectory_df)):
        prev_point = trajectory_df.iloc[i-1]
        curr_point = trajectory_df.iloc[i]
        bearing = calculate_bearing(
            prev_point['latitude'], prev_point['longitude'],
            curr_point['latitude'], curr_point['longitude']
        )
        bearings.append(bearing)
    
    if len(bearings) == 0:
        return 'C'
    
    # Calculate standard deviation of directions
    bearings = np.array(bearings)
    # Handle circular nature of angles (0 and 360 degrees are close)
    bearing_std = np.std(np.unwrap(np.radians(bearings))) * 180 / np.pi
    
    # Determine main driving direction
    avg_bearing = np.mean(bearings)
    
    # Define direction categories
    # North: 315-45 degrees, East: 45-135 degrees, South: 135-225 degrees, West: 225-315 degrees
    if (avg_bearing >= 315 or avg_bearing < 45):
        main_direction = 'N'  # North
    elif (avg_bearing >= 45 and avg_bearing < 135):
        main_direction = 'E'  # East
    elif (avg_bearing >= 135 and avg_bearing < 225):
        main_direction = 'S'  # South
    else:
        main_direction = 'W'  # West
    
    # Determine if it's straight or turning, and if turning, left or right
    is_turn = (curvature_ratio > 1.3) or (bearing_std > 30)
    
    # For turns, determine if it's left or right turn
    turn_direction = None
    if is_turn and len(bearings) > 2:
        # Calculate the overall direction change
        start_bearing = bearings[0]
        end_bearing = bearings[-1]
        
        # Calculate the signed angle difference (accounting for circular nature)
        angle_diff = (end_bearing - start_bearing + 180) % 360 - 180
        
        # Positive angle_diff means right turn, negative means left turn
        if abs(angle_diff) > 30:  # Significant turn
            if angle_diff > 0:
                turn_direction = 'right'
            else:
                turn_direction = 'left'
    
    # Classify based on direction and turning status
    if main_direction in ['N', 'S']:  # North-South direction
        if is_turn:
            if turn_direction == 'left':
                return 'A2'  # North-South left turn
            elif turn_direction == 'right':
                return 'C'  # Right turn belongs to C category
            else:
                return 'C'  # Uncertain turn direction, classify as C
        else:
            return 'A1'  # North-South straight
    elif main_direction in ['E', 'W']:  # East-West direction
        if is_turn:
            if turn_direction == 'left':
                return 'A4'  # East-West left turn
            elif turn_direction == 'right':
                return 'C'  # Right turn belongs to C category
            else:
                return 'C'  # Uncertain turn direction, classify as C
        else:
            return 'A3'  # East-West straight
    else:
        return 'C'  # Other cases

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
