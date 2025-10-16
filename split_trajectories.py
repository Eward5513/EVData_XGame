#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split trajectories in A0003.csv and A0008.csv by adaptive time-gap rule.

Output files:
- A0003_split.csv
- A0008_split.csv

Schema: same as input, but insert seg_id immediately after vehicle_id.
seg_id starts at 0 per (vehicle_id, date) trajectory; if no split occurs, all rows have seg_id=0.
"""

import pandas as pd
import numpy as np


def adaptive_split_segment_indices(df: pd.DataFrame, max_gap_seconds: float = 180.0) -> pd.Series:
    """
    Compute seg_id per (vehicle_id, date) group using adaptive time-gap splitting.
    - Start seg_id=0; for each point i, if time gap to previous point > 2x median of previous gaps
      within current seg, or > max_gap_seconds, then start a new seg (seg_id += 1) at i.
    - Requires df sorted by collectiontime within each group.
    Returns a Series with seg_id aligned to df index.
    """
    seg_ids = pd.Series(index=df.index, dtype=int)

    # Initialize
    current_seg_id = 0
    prev_diffs: list[float] = []
    seg_ids.iloc[0] = current_seg_id

    times = df['collectiontime'].to_numpy()  # expected in ms

    for i in range(1, len(df)):
        cur_diff = (times[i] - times[i - 1]) / 1000.0  # seconds
        should_split = False

        if len(prev_diffs) >= 3:
            baseline = float(np.median(prev_diffs))
            if baseline > 0 and float(cur_diff) > 2.0 * baseline:
                should_split = True

        if float(cur_diff) > float(max_gap_seconds):
            should_split = True

        if should_split:
            current_seg_id += 1
            prev_diffs = []

        seg_ids.iloc[i] = current_seg_id
        prev_diffs.append(float(cur_diff))

    return seg_ids


def process_file(input_path: str, output_path: str, max_gap_seconds: float = 180.0) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path)

    # Validate required columns
    required_cols = ['vehicle_id', 'collectiontime', 'date']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Sort by vehicle_id, date, collectiontime to ensure correct ordering
    df = df.sort_values(['vehicle_id', 'date', 'collectiontime']).reset_index(drop=True)

    # Compute seg_id per (vehicle_id, date)
    seg_ids_all = []
    for (vehicle_id, date), group_idx in df.groupby(['vehicle_id', 'date']).groups.items():
        group_df = df.loc[group_idx]
        seg_ids = adaptive_split_segment_indices(group_df, max_gap_seconds=max_gap_seconds)
        seg_ids_all.append(seg_ids)

    seg_ids_concat = pd.concat(seg_ids_all).sort_index()

    # Insert seg_id after vehicle_id column
    cols = list(df.columns)
    try:
        vid_pos = cols.index('vehicle_id')
    except ValueError:
        vid_pos = 0
    df.insert(vid_pos + 1, 'seg_id', seg_ids_concat.values)

    print(f"Writing {output_path} ...")
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


def main():
    base = '/home/tzhang174/EVData_XGame'
    process_file(f'{base}/A0003.csv', f'{base}/A0003_split.csv')
    process_file(f'{base}/A0008.csv', f'{base}/A0008_split.csv')


if __name__ == '__main__':
    main()


