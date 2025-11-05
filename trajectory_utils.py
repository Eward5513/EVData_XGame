#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared geometry and direction utilities for trajectory processing.

These helpers consolidate logic used across modules (classification, merging, refine).
"""

from __future__ import annotations

import math
from typing import Optional


def pair_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate planar distance in meters between two lon/lat points."""
    lat_avg = (float(lat1) + float(lat2)) / 2.0
    dy = (float(lat2) - float(lat1)) * 111000.0
    dx = (float(lon2) - float(lon1)) * 111000.0 * math.cos(math.radians(lat_avg))
    return math.hypot(dx, dy)


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Simplified planar distance between two points (meters)."""
    lat_diff = (float(lat2) - float(lat1)) * 111000.0
    lon_diff = (float(lon2) - float(lon1)) * 111000.0 * math.cos(
        math.radians((float(lat1) + float(lat2)) / 2.0)
    )
    return math.hypot(lon_diff, lat_diff)


def _segment_plane_coords(
    i1: int, i2: int, lats, lons, center_lat: float, center_lon: float
) -> tuple[float, float, float, float]:
    """Local-plane coordinates (meters) of segment endpoints relative to center."""
    avg_lat1 = (float(lats[i1]) + float(center_lat)) / 2.0
    avg_lat2 = (float(lats[i2]) + float(center_lat)) / 2.0
    x1 = (float(lons[i1]) - float(center_lon)) * 111000.0 * math.cos(math.radians(avg_lat1))
    y1 = (float(lats[i1]) - float(center_lat)) * 111000.0
    x2 = (float(lons[i2]) - float(center_lon)) * 111000.0 * math.cos(math.radians(avg_lat2))
    y2 = (float(lats[i2]) - float(center_lat)) * 111000.0
    return x1, y1, x2, y2


def center_min_distance_to_segment_m(
    i1: int, i2: int, lats, lons, center_lat: float, center_lon: float
) -> float:
    """Minimum distance from center to segment (i1->i2) in meters."""
    x1, y1, x2, y2 = _segment_plane_coords(i1, i2, lats, lons, center_lat, center_lon)
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return math.hypot(x1, y1)
    t = max(0.0, min(1.0, (-(x1) * vx + (-(y1)) * vy) / denom))
    px, py = x1 + t * vx, y1 + t * vy
    return math.hypot(px, py)


def segment_projection_contains_center(
    i1: int, i2: int, lats, lons, center_lat: float, center_lon: float
) -> bool:
    """
    Whether perpendicular projection of center onto segment (i1->i2)
    falls within the open segment (0<t<1).
    """
    x1, y1, x2, y2 = _segment_plane_coords(i1, i2, lats, lons, center_lat, center_lon)
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return False
    t = (-(x1) * vx + (-(y1)) * vy) / denom
    return 0.0 < t < 1.0


def polar_angle_deg_from_center(
    lat: float, lon: float, center_lat: float, center_lon: float
) -> float:
    """
    Polar angle in degrees [0,360): 0°=East, 90°=North, 180°=West, 270°=South
    of point relative to intersection center.
    """
    avg_lat = (float(lat) + float(center_lat)) / 2.0
    dx_m = (float(lon) - float(center_lon)) * 111000.0 * math.cos(math.radians(avg_lat))
    dy_m = (float(lat) - float(center_lat)) * 111000.0
    theta = math.degrees(math.atan2(dy_m, dx_m))
    return (theta + 360.0) % 360.0


def minimal_angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (degrees)."""
    d = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    return abs(d)


def quadrant_from_polar(angle_deg: float) -> str:
    """
    Map polar angle to quadrant with hard splits:
    E: [-45°,45°) U [315°,360°)
    N: [45°,135°)
    W: [135°,225°)
    S: [225°,315°)
    """
    a = float(angle_deg) % 360.0
    if a >= 315.0 or a < 45.0:
        return 'E'
    if 45.0 <= a < 135.0:
        return 'N'
    if 135.0 <= a < 225.0:
        return 'W'
    return 'S'


def axis_from_label(cardinal: Optional[str]) -> Optional[str]:
    if cardinal is None:
        return None
    return 'NS' if cardinal in ('N', 'S') else ('EW' if cardinal in ('E', 'W') else None)


def step_axis_label(
    lats,
    lons,
    i: int,
    j: int,
    axis_ratio: float = 1.4,
    min_step_m: float = 3.0,
) -> Optional[str]:
    """
    Label a step by dominant axis using dx/dy in meters.
    Returns one of {'N','S','E','W'} or None if no dominant axis or too short.
    """
    if i < 0 or j >= len(lats) or j <= i:
        return None
    avg = (float(lats[i]) + float(lats[j])) / 2.0
    dy = (float(lats[j]) - float(lats[i])) * 111000.0
    dx = (float(lons[j]) - float(lons[i])) * 111000.0 * math.cos(math.radians(avg))
    if math.hypot(dx, dy) < float(min_step_m):
        return None
    abs_x = abs(dx)
    abs_y = abs(dy)
    if abs_y >= float(axis_ratio) * abs_x:
        return 'N' if dy > 0 else 'S'
    if abs_x >= float(axis_ratio) * abs_y:
        return 'E' if dx > 0 else 'W'
    return None


