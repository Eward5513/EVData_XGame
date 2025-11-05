#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global configuration shared across the trajectory processing pipeline.

Only `main.py` should be used as the entrypoint; other modules import from here.
"""

from __future__ import annotations

# Absolute workspace base
BASE_DIR: str = "/home/tzhang174/EVData_XGame"

# Per-road intersection centers as (lat, lon)
CENTERS: dict[str, tuple[float, float]] = {
    'A0003': (32.345137, 123.152539),
    'A0008': (32.327137, 123.181261),
}

# Refine step parameters
REFINE_NEAR_RADIUS_M: float = 115.0
STEP_AXIS_RATIO: float = 1.5
STEP_MIN_LENGTH_M: float = 6.0
MAX_OFF_AXIS_CONSEC: int = 2
VOTE_WINDOW: int = 3


