#!/usr/bin/env python3
"""
Script: Sort traffic signal data by road_id and cycle_num
Input: traffic_signal.csv
Output: signal_cycle.csv
"""

import pandas as pd
import os

def sort_traffic_signals():
    """
    Sort traffic signal data by road_id and cycle_num
    """
    
    input_file = 'traffic_signal.csv'
    output_file = 'signal_cycle.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    try:
        print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print("Original data preview:")
        print(df.head())
        
        # Sort by road_id and cycle_num
        print("\nSorting data by road_id and cycle_num...")
        sorted_df = df.sort_values(['road_id', 'cycle_num'])
        
        # Save sorted data
        print(f"Saving sorted data to: {output_file}")
        sorted_df.to_csv(output_file, index=False)
        
        print(f"\nSorted data shape: {sorted_df.shape}")
        print("Sorted data preview:")
        print(sorted_df.head(10))
        
        print(f"\nSorting completed successfully!")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    sort_traffic_signals()
