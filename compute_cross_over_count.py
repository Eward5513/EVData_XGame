#!/usr/bin/env python3
"""
Compute per-second overlap counts from a CSV of time intervals.

Input CSV is expected to contain at least the columns:
  - date (YYYY-MM-DD)
  - enter_time (HH:MM:SS)
  - exit_time (HH:MM:SS)

Each row defines an interval [enter_time, exit_time) on the given date.
For every whole second t where enter_time <= t < exit_time, the counter is incremented.

Usage:
  python compute_cross_over_count.py \
    --input data/cross_over_time_A.csv \
    --output cross_over_count.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_timestamp(date_str: str, time_str: str) -> datetime:
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")


def read_intervals(
    csv_path: Path,
    road_id: str | None = None,
    direction: str | None = None,
    only_date: str | None = None,
) -> List[Tuple[datetime, datetime]]:
    intervals: List[Tuple[datetime, datetime]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"date", "enter_time", "exit_time"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            if road_id and row.get("road_id") != road_id:
                continue
            if direction and row.get("direction") != direction:
                continue
            if only_date and row.get("date") != only_date:
                continue

            start = parse_timestamp(row["date"], row["enter_time"])
            end = parse_timestamp(row["date"], row["exit_time"])
            # Treat as [start, end). If end <= start (e.g., crossed midnight), roll end to next day.
            if end <= start:
                end = end + timedelta(days=1)
            intervals.append((start, end))
    return intervals


def compute_per_second_counts(intervals: Iterable[Tuple[datetime, datetime]]) -> Dict[datetime, int]:
    counts: Dict[datetime, int] = defaultdict(int)
    one_second = timedelta(seconds=1)
    for start, end in intervals:
        current = start
        while current < end:
            counts[current] += 1
            current += one_second
    return counts


def write_counts_csv(counts: Dict[datetime, int], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "count"])
        for ts in sorted(counts):
            writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), counts[ts]])
            num_rows += 1
    return num_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-second overlap counts from interval CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="data/cross_over_time_A.csv",
        help="Input CSV path (default: data/cross_over_time_A.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="cross_over_count.csv",
        help="Output CSV path (default: cross_over_count.csv)",
    )
    parser.add_argument(
        "--road-id",
        help="Optional filter: only include this road_id",
    )
    parser.add_argument(
        "--direction",
        help="Optional filter: only include this direction",
    )
    parser.add_argument(
        "--date",
        help="Optional filter: only include the specific date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    intervals = read_intervals(
        csv_path=input_path,
        road_id=args.road_id,
        direction=args.direction,
        only_date=args.date,
    )
    counts = compute_per_second_counts(intervals)
    num_rows = write_counts_csv(counts, output_path)
    print(f"Wrote {num_rows} rows to {output_path}")


if __name__ == "__main__":
    main()


