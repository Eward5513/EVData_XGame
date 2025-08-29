#!/usr/bin/env python3
"""
Script: Calculate the duration of each traffic light cycle at each intersection
Input: traffic_signal.csv - contains the start and end times of red lights for each phase of each cycle
Output: cycle_duration.csv - contains the duration of each cycle at each intersection
"""

import pandas as pd
import sys
import os

def calculate_cycle_duration(input_file, output_file=None):
    """
    Calculate the duration of each traffic light cycle at each intersection
    
    Parameters:
    - input_file: Path to input CSV file
    - output_file: Path to output CSV file (optional, defaults to cycle_duration.csv)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    try:
        # Read CSV file
        print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_columns = ['road_id', 'phase_id', 'cycle_num', 'start_time', 'end_time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return
        
        # Convert time columns to datetime
        print("Converting time format...")
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Calculate earliest start time and latest end time for each intersection and cycle
        print("Calculating cycle durations...")
        cycle_stats = df.groupby(['road_id', 'cycle_num']).agg({
            'start_time': 'min',  # Earliest start time
            'end_time': 'max'     # Latest end time
        }).reset_index()
        
        # Calculate cycle duration in seconds
        cycle_stats['duration'] = (
            cycle_stats['end_time'] - cycle_stats['start_time']
        ).dt.total_seconds()
        
        # Sort results
        cycle_stats = cycle_stats.sort_values(['road_id', 'cycle_num'])
        
        # Select and rename columns for output
        output_df = cycle_stats[['road_id', 'cycle_num', 'start_time', 'end_time', 'duration']].copy()
        
        # Set output file name
        if output_file is None:
            output_file = 'cycle_duration.csv'
        
        # Save results
        print(f"Saving results to: {output_file}")
        output_df.to_csv(output_file, index=False)
        
        # Display statistics
        print("\n=== Calculation Results Summary ===")
        print(f"Total intersections: {output_df['road_id'].nunique()}")
        print(f"Total cycles: {len(output_df)}")
        print(f"Average cycle duration: {output_df['duration'].mean():.2f} seconds")
        print(f"Minimum cycle duration: {output_df['duration'].min():.2f} seconds")
        print(f"Maximum cycle duration: {output_df['duration'].max():.2f} seconds")
        
        # Statistics by intersection
        print("\n=== Statistics by Intersection ===")
        road_stats = output_df.groupby('road_id').agg({
            'cycle_num': 'count',
            'duration': ['mean', 'min', 'max', 'std']
        }).round(2)
        road_stats.columns = ['Cycle Count', 'Avg Duration(s)', 'Min Duration(s)', 'Max Duration(s)', 'Std Duration(s)']
        print(road_stats)
        
        # Display first few results
        print(f"\n=== First 10 Results Preview ===")
        print(output_df.head(10).to_string(index=False))
        
        return output_df
        
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        return None

def main():
    """Main function"""
    
    # Fixed file paths
    input_file = 'traffic_signal.csv'
    output_file = 'cycle_duration.csv'
    
    print("=== Traffic Light Cycle Duration Calculator ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Execute calculation
    result = calculate_cycle_duration(input_file, output_file)
    
    if result is not None:
        print("\nCalculation completed successfully!")
    else:
        print("\nCalculation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
