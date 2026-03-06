"""
visualizer.py — Advanced Overlay Rendering
============================================
Provides higher-level drawing primitives for the collision-avoidance UI:

- **Safe Corridor**: A semi-transparent trapezoid in the bottom half of the
  frame, showing the navigable path within the critical zone.
- **Proximity Heat Bars**: Vertical colour-coded bars beside each bounding
  box that give an instant visual cue of distance (green / yellow / red).

Usage
-----
    from src.utils.visualizer import draw_safe_corridor, draw_proximity_bars

    draw_safe_corridor(frame, zone_left, zone_right, is_stopped=False)
    draw_proximity_bars(frame, detections, frame_count=0)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


# ── Colour constants ─────────────────────────────────────────────────
_GREEN: Tuple[int, int, int] = (0, 200, 0)
_YELLOW: Tuple[int, int, int] = (0, 220, 255)
_RED: Tuple[int, int, int] = (0, 0, 255)
_DARK: Tuple[int, int, int] = (30, 30, 30)


# =====================================================================
#  Safe Corridor
# =====================================================================

def draw_safe_corridor(
    frame: np.ndarray,
    zone_left: int,
    zone_right: int,
    is_stopped: bool = False,
    alpha: float = 0.20,
    taper: float = 0.35,
) -> None:
    """Draw a semi-transparent corridor trapezoid on the bottom half.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to draw on (modified in-place).
    zone_left, zone_right : int
        Critical zone x-boundaries (pixels).
    is_stopped : bool
        If ``True`` the corridor is red; otherwise green.
    alpha : float
        Opacity of the corridor overlay (0 = invisible, 1 = solid).
    taper : float
        How much the top edge narrows relative to the bottom edge.
        ``0.35`` means the top edge is 35 % narrower on each side.
    """
    h, w = frame.shape[:2]
    mid_y = h // 2

    # Bottom edge = full zone width.
    bl = (zone_left, h)
    br = (zone_right, h)

    # Top edge = tapered inward.
    zone_w = zone_right - zone_left
    inset = int(zone_w * taper)
    tl = (zone_left + inset, mid_y)
    tr = (zone_right - inset, mid_y)

    pts = np.array([bl, br, tr, tl], dtype=np.int32)

    colour = _RED if is_stopped else _GREEN

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw the corridor border for clarity.
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2,
                  lineType=cv2.LINE_AA)


# =====================================================================
#  Proximity Heat Bar
# =====================================================================

def _bar_colour(
    distance: float,
    stop_dist: float = 1.5,
    avoid_dist: float = 3.0,
    frame_count: int = 0,
) -> Tuple[int, int, int]:
    """Return the heat-bar colour for a given distance.

    - > ``avoid_dist``            → green
    - ``stop_dist``–``avoid_dist`` → yellow
    - < ``stop_dist``             → flashing red (toggles every ~8 frames)
    """
    if distance > avoid_dist:
        return _GREEN
    if distance > stop_dist:
        return _YELLOW
    # Flashing red: alternate between bright red and dark every 8 frames.
    return _RED if (frame_count // 8) % 2 == 0 else _DARK


def _bar_fill(
    distance: float,
    max_dist: float = 5.0,
) -> float:
    """Return fill ratio (0–1) for the bar.  Closer = more filled."""
    return max(0.0, min(1.0, 1.0 - distance / max_dist))


def draw_proximity_bars(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    frame_count: int = 0,
    bar_width: int = 8,
    bar_margin: int = 4,
    stop_dist: float = 1.5,
    avoid_dist: float = 3.0,
) -> None:
    """Draw a vertical heat bar to the right of each bounding box.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (modified in-place).
    detections : list[dict]
        Enriched detections (must have ``"distance"`` and ``"bbox"``).
    frame_count : int
        Current frame index — used to drive the red flash animation.
    bar_width : int
        Width of each bar in pixels.
    bar_margin : int
        Gap between the bounding box and the bar.
    stop_dist, avoid_dist : float
        Thresholds for colour transitions.
    """
    for det in detections:
        dist = det.get("distance")
        if dist is None:
            continue

        x1, y1, x2, y2 = det["bbox"]
        bar_h = y2 - y1
        if bar_h <= 0:
            continue

        # Bar position: right edge of bbox + margin.
        bx = x2 + bar_margin
        by = y1

        # Clamp to frame width.
        if bx + bar_width >= frame.shape[1]:
            bx = x1 - bar_margin - bar_width  # fallback: left side

        # Background (dark).
        cv2.rectangle(
            frame, (bx, by), (bx + bar_width, by + bar_h), _DARK, cv2.FILLED
        )

        # Filled portion.
        fill = _bar_fill(dist)
        fill_h = int(bar_h * fill)
        colour = _bar_colour(dist, stop_dist, avoid_dist, frame_count)
        cv2.rectangle(
            frame,
            (bx, by + bar_h - fill_h),
            (bx + bar_width, by + bar_h),
            colour,
            cv2.FILLED,
        )

        # Thin border.
        cv2.rectangle(
            frame, (bx, by), (bx + bar_width, by + bar_h),
            (180, 180, 180), 1,
        )
