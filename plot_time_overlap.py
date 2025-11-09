#!/usr/bin/env python3
"""
Plot overlapping time intervals from a CSV as a time-axis strip.

- X-axis: time of day
- Each record's interval contributes to the shading
- Darker color => more vehicles overlapping at that time

How to run:
  python plot_time_overlap.py
Configure parameters in-file below (no CLI args).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Use safe Latin fonts to avoid garbled axes; switch labels to English
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
	"DejaVu Sans",
	"Arial",
	"Liberation Sans",
	"Nimbus Sans L",
]
rcParams["axes.unicode_minus"] = True

# =========================
# Configuration (edit here)
# =========================
CSV_PATH = "/home/tzhang174/EVData_XGame/data/cross_over_time_A.csv"
OUTPUT_PATH = "/home/tzhang174/EVData_XGame/plots/cross_over_time_A_overlap_3.png"
# Real (A1) intervals and output
REAL_CSV_PATH = "/home/tzhang174/EVData_XGame/data/corss_timeA_real.csv"
REAL_OUTPUT_PATH = "/home/tzhang174/EVData_XGame/plots/cross_over_time_A_real_overlap.png"
WAIT_CSV_PATH = "/home/tzhang174/EVData_XGame/data/wait.csv"
WAIT_OUTPUT_PATH = "/home/tzhang174/EVData_XGame/plots/wait_overlap.png"
CSV_PATH_ALL = "/home/tzhang174/EVData_XGame/data/cross_over_time.csv"
GROUPED_OUTPUT_PATH = "/home/tzhang174/EVData_XGame/plots/cross_over_time_grouped_overlay.png"
GROUPED_TITLE = "Overlap by direction groups (A1/B1/A2/B2)"
GROUP_DIR_COLORS = {
	"A1": "#1f77b4",
	"B1": "#ff7f0e",
	"A2": "#2ca02c",
	"B2": "#d62728",
}
# Single-group output config
SINGLE_OUTPUT_DIR = "/home/tzhang174/EVData_XGame/plots"
SINGLE_TITLE_PREFIX = "Overlap - "
# Separate outputs for single-group figures
GROUP_SINGLE_OUTPUTS = {
	"A1": "/home/tzhang174/EVData_XGame/plots/cross_over_time_A1_overlay.png",
	"B1": "/home/tzhang174/EVData_XGame/plots/cross_over_time_B1_overlay.png",
	"A2": "/home/tzhang174/EVData_XGame/plots/cross_over_time_A2_overlay.png",
	"B2": "/home/tzhang174/EVData_XGame/plots/cross_over_time_B2_overlay.png",
}
COUNT_OUTPUT_PATH = "/home/tzhang174/EVData_XGame/plots/cross_over_time_A_overlap_counts.png"
WAIT_COUNT_OUTPUT_PATH = "/home/tzhang174/EVData_XGame/plots/wait_overlap_counts.png"
REFINED_CSV_PATH = "/home/tzhang174/EVData_XGame/data/A0003_refined.csv"
DIRECTION_CSV_PATH = "/home/tzhang174/EVData_XGame/data/direction.csv"

# Optional filters; set to None to include all
ROAD_ID = "A0003"  # e.g., "A0003" or None
TARGET_DATE = None  # e.g., "2024-06-20" or None
SEGMENT = None      # e.g., "A1-1" or None
DIRECTION = None    # e.g., "A1-1" or "A1-2" or None

# Plot mode: "alpha" => translucent bars overlaid; "heat" => binned heat strip
MODE = "alpha"
HEAT_BIN_SECONDS = 5  # used only when MODE == "heat"
TITLE = "Interval overlap"
SAVE_DPI = 220
WAIT_TITLE = "Wait time overlap"
COUNT_TITLE = "Overlap count per second"
WAIT_COUNT_TITLE = "Wait count per second"
WAIT_COLOR = "#7f0000"
COMBINED_OUTPUT_PATH = "/home/tzhang174/EVData_XGame/plots/overlap_combined.png"
COMBINED_TITLE = "Combined overlap (cross_over + wait)"
CROSS_COLOR = "#00441b"
CYCLE_CSV_PATH = "/home/tzhang174/EVData_XGame/data/cycle_preview.csv"
DRAW_CYCLE_MARKERS = True
CYCLE_START_COLOR = "#9467bd"  # purple
GREEN_START_COLOR = "#ffbf00"  # gold
CYCLE_START_LS = ":"
GREEN_START_LS = "-"
CYCLE_MARK_LW = 1.8
CYCLE_ZORDER = 12

# End-time markers from merged points (A1-1 & A1-2)
DRAW_END_TIME_LINES = True
ENDLINE_DIRECTIONS = ("A1-1", "A1-2")
ENDLINE_COLORS = {
	"A1-1": "#d62728",  # red
	"A1-2": "#2ca02c",  # green
}
ENDLINE_ALPHA = 0.9
ENDLINE_LW = 1.8
ENDLINE_LS = "--"
ENDLINE_ZORDER = 10


def parse_time_to_seconds(time_str: str) -> int:
	"""Convert 'HH:MM:SS' to seconds since 00:00:00."""
	h, m, s = time_str.split(":")
	return int(h) * 3600 + int(m) * 60 + int(s)


def parse_time_to_seconds_any(time_str: str) -> Optional[int]:
	"""
	Parse time-of-day from:
	- 'HH:MM' or 'HH:MM:SS'
	- 'YYYY-MM-DD HH:MM:SS[.ffffff]'
	Returns seconds since midnight or None if unparsable.
	"""
	if time_str is None:
		return None
	t = str(time_str).strip()
	if not t or t.lower() == "nat":
		return None
	# Fast path: HH:MM or HH:MM:SS
	try:
		parts = t.split(":")
		if len(parts) == 2:
			h, m = int(parts[0]), int(parts[1])
			return h * 3600 + m * 60
		if len(parts) >= 3 and " " not in parts[0]:
			h, m = int(parts[0]), int(parts[1])
			sec_part = parts[2]
			# Trim fractional seconds if present
			if "." in sec_part:
				sec_part = sec_part.split(".")[0]
			s = int(sec_part)
			return h * 3600 + m * 60 + s
	except Exception:
		pass
	# Fallback: use pandas to parse full timestamps
	try:
		ts = pd.to_datetime(t, errors="coerce")
		if ts is pd.NaT:
			return None
		return int(ts.hour) * 3600 + int(ts.minute) * 60 + int(ts.second)
	except Exception:
		return None


def format_seconds_as_hhmm(seconds: float) -> str:
	"""Format seconds since midnight as 'HH:MM'."""
	seconds = int(round(seconds))
	h = seconds // 3600
	m = (seconds % 3600) // 60
	return f"{h:02d}:{m:02d}"


@dataclass
class Interval:
	start_s: int
	end_s: int

	@property
	def width(self) -> int:
		return max(0, self.end_s - self.start_s)


def load_intervals(
	csv_path: str,
	road_id: Optional[str] = None,
	target_date: Optional[str] = None,
	segment: Optional[str] = None,
	direction: Optional[str] = None,
) -> List[Interval]:
	"""
	Read CSV and return a list of intervals [enter_time, exit_time) in seconds of day.
	Expected columns: road_id, date, enter_time, exit_time, seg_id, direction
	"""
	df = pd.read_csv(csv_path, dtype=str)

	# Basic filters if provided
	if road_id:
		df = df[df["road_id"] == road_id]
	if target_date:
		df = df[df["date"] == target_date]
	if segment:
		df = df[df["seg_id"] == segment]
	if direction:
		df = df[df["direction"] == direction]

	intervals: List[Interval] = []
	for _, row in df.iterrows():
		try:
			start = parse_time_to_seconds_any(row.get("enter_time"))
			end = parse_time_to_seconds_any(row.get("exit_time"))
			if start is None or end is None:
				continue
			# Guard against malformed rows
			if end < start:
				# If exit before enter (e.g., cross-midnight not expected here), skip
				continue
			if end == start:
				continue
			intervals.append(Interval(start, end))
		except Exception:
			# Skip malformed time strings
			continue
	return intervals


def load_wait_intervals(
	csv_path: str,
	road_id: Optional[str] = None,
	target_date: Optional[str] = None,
	segment: Optional[str] = None,
	direction: Optional[str] = None,
) -> List[Interval]:
	"""
	Read wait.csv and return a list of wait intervals [time_stamp, end_time) in seconds of day.
	Expected columns: road_id, date, time_stamp, end_time, seg_id, direction
	"""
	df = pd.read_csv(csv_path, dtype=str)

	# Basic filters if provided
	if road_id:
		df = df[df["road_id"] == road_id]
	if target_date:
		df = df[df["date"] == target_date]
	if segment:
		df = df[df["seg_id"] == segment]
	if direction:
		df = df[df["direction"] == direction]

	# Drop rows with missing times
	df = df.dropna(subset=["time_stamp", "end_time"])

	intervals: List[Interval] = []
	for _, row in df.iterrows():
		try:
			start = parse_time_to_seconds(str(row["time_stamp"]))
			end = parse_time_to_seconds(str(row["end_time"]))
			if end <= start:
				continue
			intervals.append(Interval(start, end))
		except Exception:
			continue
	return intervals


def compute_bounds(intervals: Sequence[Interval], start: Optional[str], end: Optional[str]) -> Tuple[int, int]:
	"""Compute plotting [min_s, max_s] bounds in seconds."""
	if not intervals:
		# Default to a daytime hour if empty
		return parse_time_to_seconds("00:00:00"), parse_time_to_seconds("23:59:59")
	min_s = min(iv.start_s for iv in intervals)
	max_s = max(iv.end_s for iv in intervals)
	if start:
		min_s = max(min_s, parse_time_to_seconds(start))
	if end:
		max_s = min(max_s, parse_time_to_seconds(end))
	return min_s, max_s


def plot_alpha_overlay(
	intervals: Sequence[Interval],
	min_s: int,
	max_s: int,
	title: Optional[str],
	output_path: Optional[str],
	end_markers_by_dir: Optional[dict] = None,
	color: str = "#1f77b4",
	alpha_scale: float = 1.0,
	min_alpha: Optional[float] = None,
) -> None:
	"""
	Draw one translucent bar per interval. Overlaps naturally look darker.
	"""
	num = max(1, len(intervals))
	# Dynamic alpha: fewer intervals => more visible; clamp to a reasonable range
	base_alpha = float(np.clip(4.0 / num, 0.015, 0.08))
	alpha = float(np.clip(base_alpha * max(0.1, float(alpha_scale)), 0.015, 0.30))
	if min_alpha is not None:
		try:
			alpha = max(alpha, float(min_alpha))
		except Exception:
			pass

	fig, ax = plt.subplots(figsize=(14, 2.2), constrained_layout=True)
	for iv in intervals:
		# Clip to bounds
		start = max(min_s, iv.start_s)
		end = min(max_s, iv.end_s)
		if end <= start:
			continue
		ax.add_patch(Rectangle((start, 0), end - start, 1, color=color, alpha=alpha, linewidth=0))

	ax.set_ylim(0, 1)
	ax.set_xlim(min_s, max_s)
	ax.set_yticks([])
	ax.set_xlabel("Time (HH:MM)")
	if title:
		ax.set_title(title)

	# Draw end_time markers as vertical dashed lines (per direction)
	if end_markers_by_dir:
		legend_handles = []
		for d, times in end_markers_by_dir.items():
			col = ENDLINE_COLORS.get(d, "#444444")
			for t in times:
				if min_s <= t <= max_s:
					ax.axvline(t, 0, 1, color=col, linestyle=ENDLINE_LS, linewidth=ENDLINE_LW, alpha=ENDLINE_ALPHA, zorder=ENDLINE_ZORDER)
			# Add legend proxy if this direction had any markers
			if times:
				legend_handles.append(Line2D([0], [0], color=col, linestyle=ENDLINE_LS, linewidth=ENDLINE_LW, label=d))
		if legend_handles:
			ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9, ncol=len(legend_handles))

	ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_seconds_as_hhmm(x)))
	ax.grid(axis="x", linestyle="--", alpha=0.25)

	ensure_parent_dir(output_path)
	if output_path:
		plt.savefig(output_path, dpi=SAVE_DPI)
	else:
		plt.show()
	plt.close(fig)


def compute_overlap_counts(
	intervals: Sequence[Interval],
	min_s: int,
	max_s: int,
	step_s: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Compute overlap counts per time step in [min_s, max_s) using a difference array.
	Returns:
		xs_seconds: np.ndarray of second marks (seconds since midnight)
		counts: np.ndarray of counts per second
	"""
	if max_s <= min_s:
		return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
	step_s = max(1, int(step_s))
	# For step_s > 1, we still compute at 1-second resolution and then aggregate
	n_seconds = max(0, int(max_s - min_s))
	if n_seconds == 0:
		return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
	diff = np.zeros(n_seconds + 1, dtype=np.int32)
	for iv in intervals:
		start = max(min_s, iv.start_s)
		end = min(max_s, iv.end_s)  # half-open [start, end)
		if end <= start:
			continue
		i0 = int(start - min_s)
		i1 = int(end - min_s)
		diff[i0] += 1
		diff[i1] -= 1
	per_sec = np.cumsum(diff[:-1])
	if step_s == 1:
		xs = np.arange(min_s, max_s, dtype=np.int64)
		return xs, per_sec
	# Aggregate into bins of size step_s
	n_bins = int(np.ceil(n_seconds / step_s))
	agg = np.zeros(n_bins, dtype=np.int32)
	for i in range(n_bins):
		start_idx = i * step_s
		end_idx = min(n_seconds, (i + 1) * step_s)
		if end_idx <= start_idx:
			continue
		agg[i] = int(per_sec[start_idx:end_idx].mean())
	xs = np.arange(min_s, min_s + n_bins * step_s, step_s, dtype=np.int64)
	return xs, agg


def plot_overlap_count_line(
	intervals: Sequence[Interval],
	min_s: int,
	max_s: int,
	title: Optional[str],
	output_path: Optional[str],
	color: str = "#1f77b4",
	step_s: int = 1,
	linewidth: float = 1.8,
) -> None:
	"""
	Draw a line chart where Y is the number of intervals overlapping each second.
	"""
	xs, counts = compute_overlap_counts(intervals, min_s, max_s, step_s=step_s)
	if xs.size == 0:
		ensure_parent_dir(output_path)
		if output_path:
			# Save an empty figure to keep pipeline consistent
			fig, ax = plt.subplots(figsize=(14, 2.2), constrained_layout=True)
			ax.set_yticks([])
			ax.set_xticks([])
			ax.set_title(title or "Empty")
			plt.savefig(output_path, dpi=SAVE_DPI)
			plt.close(fig)
		return
	fig, ax = plt.subplots(figsize=(14, 2.2), constrained_layout=True)
	ax.plot(xs, counts, color=color, linewidth=linewidth)
	ax.set_xlim(min_s, max_s)
	ax.set_xlabel("Time (HH:MM)")
	ax.set_ylabel("Count")
	if title:
		ax.set_title(title)
	ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_seconds_as_hhmm(x)))
	ax.grid(axis="both", linestyle="--", alpha=0.25)
	ensure_parent_dir(output_path)
	if output_path:
		plt.savefig(output_path, dpi=SAVE_DPI)
	else:
		plt.show()
	plt.close(fig)

def _parse_group_key(g: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
	try:
		parts = str(g).split("|")
		if len(parts) >= 4:
			return parts[0], parts[1], parts[2], parts[3]
		return None, None, None, None
	except Exception:
		return None, None, None, None

def load_cycle_markers(
	cycle_csv_path: str,
	road_id: Optional[str],
	allowed_dirs: Sequence[str],
	min_s: Optional[int] = None,
	max_s: Optional[int] = None,
) -> dict:
	"""
	Load cycle and green start times from cycle_preview.csv and return:
	{
	  'cycle_start': {dir: [seconds...]},
	  'green_start': {dir: [seconds...]}
	}
	"""
	df = pd.read_csv(cycle_csv_path, dtype=str)
	need_cols = {"group_key", "cycle_start_time", "green_start_time"}
	missing = need_cols - set(df.columns)
	if missing:
		raise ValueError(f"cycle_preview.csv missing columns: {missing}")

	# Parse group_key to columns: road_id, direction, seg_id, date
	meta = df["group_key"].apply(_parse_group_key)
	df["road_id_parsed"] = [t[0] for t in meta]
	df["direction_parsed"] = [t[1] for t in meta]
	df["seg_id_parsed"] = [t[2] for t in meta]
	df["date_parsed"] = [t[3] for t in meta]

	if road_id:
		df = df[df["road_id_parsed"] == str(road_id)]
	df = df[df["direction_parsed"].isin(list(allowed_dirs))]

	def times_to_seconds(series: pd.Series) -> list[int]:
		out: list[int] = []
		for v in series.fillna("").astype(str):
			vt = v.strip()
			if not vt:
				continue
			try:
				sec = parse_time_to_seconds(vt)
				out.append(sec)
			except Exception:
				continue
		# Dedup and keep sorted
		out = sorted(set(out))
		# Filter view if requested
		if min_s is not None and max_s is not None:
			out = [t for t in out if min_s <= t <= max_s]
		return out

	result = {
		"cycle_start": {},
		"green_start": {},
	}
	for d in allowed_dirs:
		sub = df[df["direction_parsed"] == d]
		cycle_list = times_to_seconds(sub["cycle_start_time"])
		green_list = times_to_seconds(sub["green_start_time"])
		result["cycle_start"][d] = cycle_list
		result["green_start"][d] = green_list
	return result


def plot_combined_alpha_overlay(
	intervals1: Sequence[Interval],
	intervals2: Sequence[Interval],
	min_s: int,
	max_s: int,
	title: Optional[str],
	output_path: Optional[str],
	color1: str,
	color2: str,
	cycle_markers: Optional[dict] = None,
	alpha_scale1: float = 1.0,
	alpha_scale2: float = 1.6,
	min_alpha1: Optional[float] = None,
	min_alpha2: Optional[float] = None,
) -> None:
	"""
	Draw two translucent overlays (cross_over and wait) on the same time axis.
	Overlaps within each set and between sets will appear darker.
	"""
	num1 = max(1, len(intervals1))
	num2 = max(1, len(intervals2))
	base_alpha1 = float(np.clip(4.0 / num1, 0.015, 0.08))
	base_alpha2 = float(np.clip(4.0 / num2, 0.015, 0.08))
	alpha1 = float(np.clip(base_alpha1 * max(0.1, float(alpha_scale1)), 0.015, 0.30))
	alpha2 = float(np.clip(base_alpha2 * max(0.1, float(alpha_scale2)), 0.015, 0.30))
	if min_alpha1 is not None:
		try:
			alpha1 = max(alpha1, float(min_alpha1))
		except Exception:
			pass
	if min_alpha2 is not None:
		try:
			alpha2 = max(alpha2, float(min_alpha2))
		except Exception:
			pass

	fig, ax = plt.subplots(figsize=(14, 2.2), constrained_layout=True)

	# First layer
	for iv in intervals1:
		start = max(min_s, iv.start_s)
		end = min(max_s, iv.end_s)
		if end <= start:
			continue
		ax.add_patch(Rectangle((start, 0), end - start, 1, color=color1, alpha=alpha1, linewidth=0))

	# Second layer
	for iv in intervals2:
		start = max(min_s, iv.start_s)
		end = min(max_s, iv.end_s)
		if end <= start:
			continue
		ax.add_patch(Rectangle((start, 0), end - start, 1, color=color2, alpha=alpha2, linewidth=0))

	ax.set_ylim(0, 1)
	ax.set_xlim(min_s, max_s)
	ax.set_yticks([])
	ax.set_xlabel("Time (HH:MM)")
	if title:
		ax.set_title(title)

	ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_seconds_as_hhmm(x)))
	ax.grid(axis="x", linestyle="--", alpha=0.25)

	# Legend for colors
	try:
		legend_handles = [
			Line2D([0], [0], color=color1, linewidth=6, label="cross_over"),
			Line2D([0], [0], color=color2, linewidth=6, label="wait"),
		]
		# Add cycle legend markers if provided
		if cycle_markers:
			legend_handles.extend([
				Line2D([0], [0], color=CYCLE_START_COLOR, linestyle=CYCLE_START_LS, linewidth=2.5, label="cycle start"),
				Line2D([0], [0], color=GREEN_START_COLOR, linestyle=GREEN_START_LS, linewidth=2.5, label="green start"),
			])
		ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9, ncol=2 if not cycle_markers else 3)
	except Exception:
		pass

	# Draw cycle markers as short vertical ticks near top/bottom to avoid clutter
	if cycle_markers:
		def dir_band(direction: str) -> tuple[float, float]:
			# Reserve top band for A1-1, bottom band for A1-2
			if direction == "A1-1":
				return 0.80, 0.98
			if direction == "A1-2":
				return 0.02, 0.20
			return 0.10, 0.30
		for kind, by_dir in cycle_markers.items():
			col = CYCLE_START_COLOR if kind == "cycle_start" else GREEN_START_COLOR
			ls = CYCLE_START_LS if kind == "cycle_start" else GREEN_START_LS
			for d, times in by_dir.items():
				y0, y1 = dir_band(d)
				for t in times:
					if min_s <= t <= max_s:
						ax.vlines(
							t, y0, y1,
							colors=col,
							linestyles=ls,
							linewidth=CYCLE_MARK_LW,
							zorder=CYCLE_ZORDER,
							clip_on=False,
						)

	ensure_parent_dir(output_path)
	if output_path:
		plt.savefig(output_path, dpi=SAVE_DPI)
	else:
		plt.show()
	plt.close(fig)

def plot_heat_overlap(
	intervals: Sequence[Interval],
	min_s: int,
	max_s: int,
	title: Optional[str],
	output_path: Optional[str],
	bin_s: int = 5,
	cmap: str = "Blues",
	end_markers_by_dir: Optional[dict] = None,
) -> None:
	"""
	Aggregate overlap counts per bin and render as a 1D heat strip with a colorbar.
	Darker color => higher count (more vehicles simultaneously inside).
	"""
	bin_s = max(1, int(bin_s))
	if max_s <= min_s:
		max_s = min_s + 1
	n_bins = int(np.ceil((max_s - min_s) / bin_s))
	counts = np.zeros(n_bins, dtype=np.int32)

	for iv in intervals:
		start = max(min_s, iv.start_s)
		end = min(max_s, iv.end_s)
		if end <= start:
			continue
		i0 = int((start - min_s) // bin_s)
		# Subtract a tiny epsilon so an interval ending exactly at a bin edge does not spill into next bin
		i1 = int((max(min_s, end) - min_s - 1e-9) // bin_s)
		i0 = max(0, min(i0, n_bins - 1))
		i1 = max(0, min(i1, n_bins - 1))
		if i1 >= i0:
			counts[i0 : i1 + 1] += 1

	# Prepare the image (1 x n_bins)
	img = counts[np.newaxis, :]

	fig, ax = plt.subplots(figsize=(14, 1.8), constrained_layout=True)
	extent = [min_s, min_s + n_bins * bin_s, 0, 1]
	im = ax.imshow(img, aspect="auto", cmap=cmap, extent=extent, origin="lower", interpolation="nearest")
	ax.set_ylim(0, 1)
	ax.set_xlim(min_s, max_s)
	ax.set_yticks([])
	ax.set_xlabel("Time (HH:MM)")
	if title:
		ax.set_title(title)
	cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.12)
	cb.set_label("Concurrent vehicles")

	# Draw end_time markers as vertical dashed lines (per direction)
	if end_markers_by_dir:
		legend_handles = []
		for d, times in end_markers_by_dir.items():
			col = ENDLINE_COLORS.get(d, "#444444")
			for t in times:
				if min_s <= t <= max_s:
					ax.axvline(t, 0, 1, color=col, linestyle=ENDLINE_LS, linewidth=ENDLINE_LW, alpha=ENDLINE_ALPHA, zorder=ENDLINE_ZORDER)
			# Add legend proxy if this direction had any markers
			if times:
				legend_handles.append(Line2D([0], [0], color=col, linestyle=ENDLINE_LS, linewidth=ENDLINE_LW, label=d))
		if legend_handles:
			ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9, ncol=len(legend_handles))

	ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_seconds_as_hhmm(x)))
	ax.grid(axis="x", linestyle="--", alpha=0.25)

	ensure_parent_dir(output_path)
	if output_path:
		plt.savefig(output_path, dpi=SAVE_DPI)
	else:
		plt.show()
	plt.close(fig)


def load_intervals_grouped_by_dir(
	csv_path: str,
	road_id: Optional[str],
	target_date: Optional[str],
	segment: Optional[str],
) -> dict:
	"""
	Load intervals from cross_over_time.csv and group into:
	  A1: {A1-1, A1-2}
	  B1: {B1-1, B1-2}
	  A2: {A2-1, A2-2}
	  B2: {B2-1, B2-2}
	"""
	df = pd.read_csv(csv_path, dtype=str)
	if road_id:
		df = df[df["road_id"] == str(road_id)]
	if target_date:
		df = df[df["date"] == str(target_date)]
	if segment:
		df = df[df["seg_id"] == str(segment)]
	df = df.dropna(subset=["enter_time", "exit_time", "direction"])

	dir_map = {
		"A1-1": "A1", "A1-2": "A1",
		"B1-1": "B1", "B1-2": "B1",
		"A2-1": "A2", "A2-2": "A2",
		"B2-1": "B2", "B2-2": "B2",
	}
	grouped: dict[str, list[Interval]] = {"A1": [], "B1": [], "A2": [], "B2": []}
	for _, row in df.iterrows():
		d = str(row["direction"]).strip()
		g = dir_map.get(d)
		if not g:
			continue
		try:
			start = parse_time_to_seconds(str(row["enter_time"]))
			end = parse_time_to_seconds(str(row["exit_time"]))
			if end <= start:
				continue
			grouped[g].append(Interval(start, end))
		except Exception:
			continue
	return grouped


def plot_multi_alpha_overlay(
	layers: Sequence[tuple[str, Sequence[Interval], str]],
	min_s: int,
	max_s: int,
	title: Optional[str],
	output_path: Optional[str],
	alpha_scale: float = 2.2,
	min_alpha: Optional[float] = 0.10,
) -> None:
	"""
	Draw multiple translucent overlays on the same axis.
	layers: list of (label, intervals, color)
	"""
	fig, ax = plt.subplots(figsize=(14, 2.4), constrained_layout=True)
	legend_handles: list[Line2D] = []

	for label, intervals, color in layers:
		num = max(1, len(intervals))
		base_alpha = float(np.clip(4.0 / num, 0.015, 0.08))
		alpha = float(np.clip(base_alpha * max(0.1, float(alpha_scale)), 0.015, 0.30))
		if min_alpha is not None:
			try:
				alpha = max(alpha, float(min_alpha))
			except Exception:
				pass
		for iv in intervals:
			start = max(min_s, iv.start_s)
			end = min(max_s, iv.end_s)
			if end <= start:
				continue
			ax.add_patch(Rectangle((start, 0), end - start, 1, color=color, alpha=alpha, linewidth=0))
		legend_handles.append(Line2D([0], [0], color=color, linewidth=6, label=label))

	ax.set_ylim(0, 1)
	ax.set_xlim(min_s, max_s)
	ax.set_yticks([])
	ax.set_xlabel("Time (HH:MM)")
	if title:
		ax.set_title(title)
	ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format_seconds_as_hhmm(x)))
	ax.grid(axis="x", linestyle="--", alpha=0.25)
	if legend_handles:
		ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9, ncol=4)

	ensure_parent_dir(output_path)
	if output_path:
		plt.savefig(output_path, dpi=SAVE_DPI)
	else:
		plt.show()
	plt.close(fig)


def plot_single_group_overlays(
	grouped: dict,
	title_prefix: str,
	output_dir: str,
	use_common_bounds: bool = True,
	alpha_scale: float = 2.2,
	min_alpha: float = 0.10,
) -> None:
	"""
	Save four standalone images for A1, B1, A2, B2.
	If use_common_bounds=True, time axis is aligned using all groups' min/max.
	"""
	keys = ("A1", "B1", "A2", "B2")
	# Compute bounds (common or per-group)
	if use_common_bounds:
		all_intervals = [iv for k in keys for iv in grouped.get(k, [])]
		if len(all_intervals) == 0:
			print("[single] No intervals in any group; skip single overlays.")
			return
		min_s = min(iv.start_s for iv in all_intervals)
		max_s = max(iv.end_s for iv in all_intervals)
	else:
		min_s = None
		max_s = None

	for k in keys:
		ivals = grouped.get(k, [])
		if len(ivals) == 0:
			print(f"[single] {k}: empty; skip.")
			continue
		out_path = os.path.join(output_dir, f"cross_over_time_{k}_overlay.png")
		color = GROUP_DIR_COLORS.get(k, "#999999")
		if not use_common_bounds:
			# Per-group bounds
			min_s_k = min(iv.start_s for iv in ivals)
			max_s_k = max(iv.end_s for iv in ivals)
			ms, mx = min_s_k, max_s_k
		else:
			ms, mx = min_s, max_s
		print(f"[single] {k}: n={len(ivals)} range={format_seconds_as_hhmm(ms)}-{format_seconds_as_hhmm(mx)} -> {out_path}")
		plot_alpha_overlay(
			intervals=ivals,
			min_s=ms,
			max_s=mx,
			title=f"{title_prefix}{k}",
			output_path=out_path,
			end_markers_by_dir=None,
			color=color,
			alpha_scale=alpha_scale,
			min_alpha=min_alpha,
		)


def ensure_parent_dir(path: Optional[str]) -> None:
	if not path:
		return
	parent = os.path.dirname(os.path.abspath(path))
	if parent and not os.path.exists(parent):
		os.makedirs(parent, exist_ok=True)

def load_end_time_markers(
	refined_csv_path: str,
	direction_csv_path: str,
	road_id: Optional[str],
	allowed_dirs: Sequence[str],
) -> dict:
	"""
	Load merged points' end_time for specified directions and return mapping:
	{direction: [seconds_of_day, ...]}
	"""
	try:
		refined = pd.read_csv(refined_csv_path, dtype=str, usecols=[
			"vehicle_id","date","seg_id","road_id","end_time"
		])
	except Exception:
		# Fallback: read full if selective read fails (schema differences)
		refined = pd.read_csv(refined_csv_path, dtype=str)
		colset = set(refined.columns)
		need = {"vehicle_id","date","seg_id","road_id","end_time"}
		missing = need - colset
		if missing:
			raise ValueError(f"Refined CSV missing columns: {missing}")
		refined = refined[list(need)]

	directions = pd.read_csv(direction_csv_path, dtype=str)
	directions = directions[["vehicle_id","date","seg_id","road_id","direction"]]

	if road_id:
		refined = refined[refined["road_id"] == str(road_id)]
		directions = directions[directions["road_id"] == str(road_id)]

	directions = directions[directions["direction"].isin(list(allowed_dirs))]

	if len(directions) == 0:
		return {d: [] for d in allowed_dirs}

	keys = ["vehicle_id","date","seg_id","road_id"]
	df = refined.merge(directions, on=keys, how="inner")
	# Filter rows with non-empty end_time
	df = df[df["end_time"].astype(str).str.strip().astype(bool)]

	end_by_dir = {}
	for d in allowed_dirs:
		sub = df[df["direction"] == d]
		# Convert HH:MM:SS to seconds-of-day, drop malformed values
		seconds_list: List[int] = []
		for t in sub["end_time"].tolist():
			try:
				seconds_list.append(parse_time_to_seconds(str(t)))
			except Exception:
				continue
		# Deduplicate and sort
		end_by_dir[d] = sorted(set(seconds_list))
	return end_by_dir


def main() -> None:
	intervals = load_intervals(
		csv_path=CSV_PATH,
		road_id=ROAD_ID,
		target_date=TARGET_DATE,
		segment=SEGMENT,
		direction=DIRECTION,
	)

	if not intervals:
		print("No intervals after filtering. Skip base plots but continue with grouped overlay.")

	min_s, max_s = compute_bounds(intervals, start=None, end=None)
	print(f"[plot] intervals={len(intervals)}  range={format_seconds_as_hhmm(min_s)}-{format_seconds_as_hhmm(max_s)}  mode={MODE}")

	end_markers_by_dir = None
	if DRAW_END_TIME_LINES:
		try:
			end_markers_by_dir = load_end_time_markers(
				refined_csv_path=REFINED_CSV_PATH,
				direction_csv_path=DIRECTION_CSV_PATH,
				road_id=ROAD_ID,
				allowed_dirs=list(ENDLINE_DIRECTIONS),
			)
		except Exception as e:
			print(f"Failed to load end_time markers: {e}")
			end_markers_by_dir = None
	
	# Print marker statistics
	if end_markers_by_dir is None:
		print("[markers] none loaded (disabled or load failed)")
	else:
		for d in ENDLINE_DIRECTIONS:
			times = end_markers_by_dir.get(d, []) if isinstance(end_markers_by_dir, dict) else []
			total_cnt = len(times)
			in_view = [t for t in times if min_s <= t <= max_s]
			in_cnt = len(in_view)
			print(f"[markers] {d}: total={total_cnt}, in_view={in_cnt}")
			if in_cnt > 0:
				sample = ", ".join(format_seconds_as_hhmm(t) for t in (in_view[:10]))
				if in_cnt > 10:
					print(f"          samples: {sample} ...")
				else:
					print(f"          times: {sample}")

	if MODE == "alpha":
		print(f"[save] output -> {OUTPUT_PATH}")
		plot_alpha_overlay(
			intervals,
			min_s,
			max_s,
			TITLE,
			OUTPUT_PATH,
			end_markers_by_dir=end_markers_by_dir,
			color=CROSS_COLOR,
			alpha_scale=2.2,
			min_alpha=0.12,
		)
	else:
		print(f"[save] output -> {OUTPUT_PATH}")
		plot_heat_overlap(intervals, min_s, max_s, TITLE, OUTPUT_PATH, bin_s=HEAT_BIN_SECONDS, end_markers_by_dir=end_markers_by_dir)

	# Real (A1) overlay with the same configuration
	try:
		real_intervals = load_intervals(
			csv_path=REAL_CSV_PATH,
			road_id=ROAD_ID,
			target_date=TARGET_DATE,
			segment=SEGMENT,
			direction=DIRECTION,
		)
		if not real_intervals:
			print("[real] No intervals after filtering. Skip real overlay.")
		else:
			rmin_s, rmax_s = compute_bounds(real_intervals, start=None, end=None)
			print(f"[real-plot] intervals={len(real_intervals)}  range={format_seconds_as_hhmm(rmin_s)}-{format_seconds_as_hhmm(rmax_s)}")
			print(f"[save] output -> {REAL_OUTPUT_PATH}")
			plot_alpha_overlay(
				intervals=real_intervals,
				min_s=rmin_s,
				max_s=rmax_s,
				title=f"{TITLE} (real)",
				output_path=REAL_OUTPUT_PATH,
				end_markers_by_dir=None,
				color=CROSS_COLOR,
				alpha_scale=2.2,
				min_alpha=0.12,
			)
	except Exception as e:
		print(f"[real] Failed to plot real overlay: {e}")

	# Line chart for overlap counts (cross_over)
	try:
		print(f"[save] output -> {COUNT_OUTPUT_PATH}")
		plot_overlap_count_line(
			intervals=intervals,
			min_s=min_s,
			max_s=max_s,
			title=COUNT_TITLE,
			output_path=COUNT_OUTPUT_PATH,
			color=CROSS_COLOR,
			step_s=1,
			linewidth=2.0,
		)
	except Exception as e:
		print(f"[count] Failed to plot overlap count line: {e}")

	# Additional figure: wait intervals from wait.csv (darker shading, overlap => darker)
	try:
		# Load cycle markers for combined plot
		cycle_markers = None
		if DRAW_CYCLE_MARKERS:
			try:
				cycle_markers = load_cycle_markers(
					cycle_csv_path=CYCLE_CSV_PATH,
					road_id=ROAD_ID,
					allowed_dirs=list(ENDLINE_DIRECTIONS),
					min_s=min_s,
					max_s=max_s,
				)
				# Print stats
				for kind in ("cycle_start", "green_start"):
					by_dir = cycle_markers.get(kind, {}) if isinstance(cycle_markers, dict) else {}
					for d in ENDLINE_DIRECTIONS:
						times = by_dir.get(d, [])
						print(f"[cycles] {kind} {d}: count={len(times)}")
			except Exception as e:
				print(f"[cycles] Failed to load cycle markers: {e}")
				cycle_markers = None

		wait_intervals = load_wait_intervals(
			csv_path=WAIT_CSV_PATH,
			road_id=ROAD_ID,
			target_date=TARGET_DATE,
			segment=SEGMENT,
			direction=DIRECTION,
		)
		if not wait_intervals:
			print("[wait] No wait intervals after filtering. Skip wait plot.")
		else:
			wmin_s, wmax_s = compute_bounds(wait_intervals, start=None, end=None)
			print(f"[plot-wait] intervals={len(wait_intervals)}  range={format_seconds_as_hhmm(wmin_s)}-{format_seconds_as_hhmm(wmax_s)}")
			print(f"[save] output -> {WAIT_OUTPUT_PATH}")
			plot_alpha_overlay(
				wait_intervals,
				wmin_s,
				wmax_s,
				WAIT_TITLE,
				WAIT_OUTPUT_PATH,
				end_markers_by_dir=None,
				color=WAIT_COLOR,
				alpha_scale=2.6,
				min_alpha=0.14,
			)
			# Combined figure (cross_over + wait)
			cmin_s = min(min_s, wmin_s)
			cmax_s = max(max_s, wmax_s)
			print(f"[plot-combined] cross={len(intervals)}, wait={len(wait_intervals)}  range={format_seconds_as_hhmm(cmin_s)}-{format_seconds_as_hhmm(cmax_s)}")
			print(f"[save] output -> {COMBINED_OUTPUT_PATH}")
			plot_combined_alpha_overlay(
				intervals1=intervals,
				intervals2=wait_intervals,
				min_s=cmin_s,
				max_s=cmax_s,
				title=COMBINED_TITLE,
				output_path=COMBINED_OUTPUT_PATH,
				color1=CROSS_COLOR,
				color2=WAIT_COLOR,
				cycle_markers=cycle_markers,
				alpha_scale1=2.2,
				alpha_scale2=2.6,
				min_alpha1=0.12,
				min_alpha2=0.14,
			)
			# Wait count-per-second line chart
			try:
				print(f"[save] output -> {WAIT_COUNT_OUTPUT_PATH}")
				plot_overlap_count_line(
					intervals=wait_intervals,
					min_s=wmin_s,
					max_s=wmax_s,
					title=WAIT_COUNT_TITLE,
					output_path=WAIT_COUNT_OUTPUT_PATH,
					color=WAIT_COLOR,
					step_s=1,
					linewidth=2.0,
				)
			except Exception as e:
				print(f"[wait-count] Failed to plot wait count line: {e}")
	except Exception as e:
		print(f"[wait] Failed to plot wait intervals: {e}")

	# Grouped overlay (A1/B1/A2/B2) from cross_over_time.csv
	try:
		grouped = load_intervals_grouped_by_dir(
			csv_path=CSV_PATH_ALL,
			road_id=ROAD_ID,
			target_date=TARGET_DATE,
			segment=SEGMENT,
		)
		total = sum(len(v) for v in grouped.values())
		if total == 0:
			print("[grouped] No intervals found in cross_over_time.csv after filters. Skip grouped overlay.")
		else:
			gmin = min((iv.start_s for vals in grouped.values() for iv in vals), default=None)
			gmax = max((iv.end_s for vals in grouped.values() for iv in vals), default=None)
			if gmin is None or gmax is None or gmax <= gmin:
				print("[grouped] Invalid bounds. Skip.")
			else:
				print(f"[grouped] counts: A1={len(grouped['A1'])}, B1={len(grouped['B1'])}, A2={len(grouped['A2'])}, B2={len(grouped['B2'])}")
				layers: list[tuple[str],] = []  # placeholder for typing context
				layers = []
				for key in ("A1", "B1", "A2", "B2"):
					ivals = grouped.get(key, [])
					if len(ivals) == 0:
						continue
					color = GROUP_DIR_COLORS.get(key, "#999999")
					layers.append((key, ivals, color))
				if layers:
					print(f"[save] output -> {GROUPED_OUTPUT_PATH}")
					plot_multi_alpha_overlay(
						layers=layers,
						min_s=gmin,
						max_s=gmax,
						title=GROUPED_TITLE,
						output_path=GROUPED_OUTPUT_PATH,
						alpha_scale=2.2,
						min_alpha=0.10,
					)
				else:
					print("[grouped] No non-empty groups to draw.")
	except Exception as e:
		print(f"[grouped] Failed to plot grouped overlay: {e}")

	# Separate single-group overlays for A1, B1, A2, B2
	try:
		if 'grouped' not in locals():
			grouped = load_intervals_grouped_by_dir(
				csv_path=CSV_PATH_ALL,
				road_id=ROAD_ID,
				target_date=TARGET_DATE,
				segment=SEGMENT,
			)
		for key in ("A1", "B1", "A2", "B2"):
			ivals = grouped.get(key, [])
			if not ivals:
				print(f"[single] {key}: no intervals, skip.")
				continue
			smin, smax = compute_bounds(ivals, start=None, end=None)
			out_path = GROUP_SINGLE_OUTPUTS.get(key)
			color = GROUP_DIR_COLORS.get(key, "#999999")
			title = f"{key} overlap"
			print(f"[single] {key}: {len(ivals)} intervals  range={format_seconds_as_hhmm(smin)}-{format_seconds_as_hhmm(smax)}")
			print(f"[save] output -> {out_path}")
			plot_alpha_overlay(
				intervals=ivals,
				min_s=smin,
				max_s=smax,
				title=title,
				output_path=out_path,
				end_markers_by_dir=None,
				color=color,
				alpha_scale=2.2,
				min_alpha=0.10,
			)
	except Exception as e:
		print(f"[single] Failed to plot single-group overlays: {e}")

if __name__ == "__main__":
	main()


