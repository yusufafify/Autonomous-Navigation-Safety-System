"""
decision.py — Safety Intervention Decision Engine
===================================================
Rule-based engine that maps detected objects (with estimated distances) to
real-time navigation control flags: ``STOP``, ``AVOID_LEFT``, ``AVOID_RIGHT``,
or ``GO``.

Key features
------------
- **Critical zone**: A configurable vertical strip in the centre of the frame.
  Only objects whose bounding-box centre falls inside this strip are considered
  direct threats.
- **Weighted AVOID direction**: When the decision is ``AVOID``, a weighted
  threat score is computed for the left and right halves of the frame.  The
  weight for each detection is ``1 / distance²`` (closer = heavier).  The side
  with the *lower* total threat weight is recommended as the escape direction.
- **Temporal smoothing**: A rolling window (deque) of the last *N* raw flags
  is maintained.  The output flag is the **mode** (most frequent value) of
  that window, which eliminates single-frame flicker.

Usage
-----
    from src.core.decision import SafetyIntervention

    intervention = SafetyIntervention(frame_width=640)
    result = intervention.process(detections)  # detections must have "distance"
    print(result["flag"], result["color"])
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# ── Flag constants ───────────────────────────────────────────────────
FLAG_STOP = "STOP"
FLAG_OBSTRUCTION = "STOP: OBSTRUCTION"
FLAG_AVOID_LEFT = "AVOID LEFT"
FLAG_AVOID_RIGHT = "AVOID RIGHT"
FLAG_GO = "GO"

# BGR colours for each flag.
COLOR_STOP: Tuple[int, int, int] = (0, 0, 255)       # Red
COLOR_AVOID: Tuple[int, int, int] = (0, 200, 255)     # Yellow-orange
COLOR_GO: Tuple[int, int, int] = (0, 220, 0)          # Green


class SafetyIntervention:
    """Rule-based collision-avoidance decision engine.

    Parameters
    ----------
    frame_width : int
        Width of the video frame in pixels (used to compute zone boundaries).
    critical_zone_pct : float
        Fraction of the frame width occupied by the critical zone,
        centred horizontally.  ``0.40`` means the middle 40 % of the image.
    stop_distance : float
        Objects closer than this (metres) inside the critical zone trigger
        ``STOP``.
    avoid_distance : float
        Objects between ``stop_distance`` and this value (metres) inside
        the critical zone trigger ``AVOID``.
    smoothing_window : int
        Number of past frames to consider for temporal smoothing.
        Higher values add more stability but also more latency.
    """

    def __init__(
        self,
        frame_width: int = 640,
        critical_zone_pct: float = 0.40,
        stop_distance: float = 1.5,
        avoid_distance: float = 3.0,
        smoothing_window: int = 5,
        obstruction_threshold: int = 220,
        obstruction_ratio: float = 0.30,
    ) -> None:
        self._frame_width = frame_width
        self._stop_dist = stop_distance
        self._avoid_dist = avoid_distance
        self._smoothing_window = smoothing_window
        self._obstruction_thresh = obstruction_threshold
        self._obstruction_ratio = obstruction_ratio

        # Critical zone boundaries (pixel x-coordinates).
        margin = (1.0 - critical_zone_pct) / 2.0
        self._zone_left = int(frame_width * margin)
        self._zone_right = int(frame_width * (1.0 - margin))

        # Rolling history for temporal smoothing.
        self._history: Deque[str] = deque(maxlen=smoothing_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        detections: List[Dict[str, Any]],
        depth_map: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluate all detections and return a smoothed control flag.

        Each detection dict **must** contain:
        - ``"bbox"``      : ``[x1, y1, x2, y2]``
        - ``"distance"``  : ``float | None``

        Parameters
        ----------
        detections : list[dict]
            Enriched detections (output of
            :meth:`DistanceEstimator.enrich_detections`).
        depth_map : np.ndarray | None
            Normalised depth map (H × W, uint8).  If provided, the engine
            checks for obstructions in the critical zone even when YOLO
            sees nothing.

        Returns
        -------
        dict
            - ``flag``    : ``str``  — ``"STOP"`` / ``"STOP: OBSTRUCTION"`` /
                            ``"AVOID LEFT"`` / ``"AVOID RIGHT"`` / ``"GO"``.
            - ``color``   : ``tuple[int,int,int]`` — BGR colour for display.
            - ``closest`` : ``dict | None`` — the detection that triggered
                            the flag (closest in-zone threat), or ``None``.
        """
        # ── Check for depth-map obstruction first ─────────────────────
        if depth_map is not None and self._check_obstruction(depth_map):
            raw_flag = FLAG_OBSTRUCTION
            closest = None
        else:
            raw_flag, closest = self._evaluate_raw(detections)

        # Push the raw flag into the smoothing window.
        self._history.append(raw_flag)

        # Smoothed flag = most common flag in the window.
        smoothed_flag = Counter(self._history).most_common(1)[0][0]

        color = self._flag_color(smoothed_flag)

        return {
            "flag": smoothed_flag,
            "color": color,
            "closest": closest,
        }

    # ------------------------------------------------------------------
    # Zone helpers (public — useful for drawing overlays)
    # ------------------------------------------------------------------

    @property
    def zone_left(self) -> int:
        """Left boundary (x-pixel) of the critical zone."""
        return self._zone_left

    @property
    def zone_right(self) -> int:
        """Right boundary (x-pixel) of the critical zone."""
        return self._zone_right

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_raw(
        self, detections: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Compute the *unsmoothed* flag for the current frame.

        Returns
        -------
        tuple[str, dict | None]
            ``(raw_flag, closest_threat_detection)``
        """
        in_zone_threats: List[Dict[str, Any]] = []

        for det in detections:
            dist = det.get("distance")
            if dist is None:
                continue

            # Bounding-box centre x.
            cx = (det["bbox"][0] + det["bbox"][2]) / 2.0

            # Is the object in the critical zone?
            if self._zone_left <= cx <= self._zone_right:
                in_zone_threats.append(det)

        if not in_zone_threats:
            return FLAG_GO, None

        # Sort by distance (closest first).
        in_zone_threats.sort(key=lambda d: d["distance"])
        closest = in_zone_threats[0]
        closest_dist = closest["distance"]

        # ── STOP: closest object dangerously near ─────────────────────
        if closest_dist < self._stop_dist:
            return FLAG_STOP, closest

        # ── AVOID: object in warning range ────────────────────────────
        if closest_dist < self._avoid_dist:
            direction = self._weighted_avoid_direction(detections)
            flag = FLAG_AVOID_LEFT if direction == "LEFT" else FLAG_AVOID_RIGHT
            return flag, closest

        # ── GO: everything is far enough ──────────────────────────────
        return FLAG_GO, None

    def _weighted_avoid_direction(
        self, detections: List[Dict[str, Any]]
    ) -> str:
        """Decide whether to avoid LEFT or RIGHT.

        For every detection with a known distance, compute a threat weight
        of ``1 / distance²``.  Sum the weights for the left and right halves
        of the frame.  The side with the **lower** total threat is the
        recommended escape direction (it is "clearer").

        Returns
        -------
        str
            ``"LEFT"`` or ``"RIGHT"``.
        """
        mid_x = self._frame_width / 2.0
        left_weight = 0.0
        right_weight = 0.0

        for det in detections:
            dist = det.get("distance")
            if dist is None or dist <= 0:
                continue

            cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
            weight = 1.0 / (dist * dist)

            if cx < mid_x:
                left_weight += weight
            else:
                right_weight += weight

        # Recommend the side with LESS threat.
        return "LEFT" if left_weight <= right_weight else "RIGHT"

    @staticmethod
    def _flag_color(flag: str) -> Tuple[int, int, int]:
        """Map a flag string to a BGR colour."""
        if flag in (FLAG_STOP, FLAG_OBSTRUCTION):
            return COLOR_STOP
        if flag in (FLAG_AVOID_LEFT, FLAG_AVOID_RIGHT):
            return COLOR_AVOID
        return COLOR_GO

    def _check_obstruction(self, depth_map: np.ndarray) -> bool:
        """Check for a close obstruction in the critical zone via depth map.

        If more than ``obstruction_ratio`` of pixels in the critical zone
        strip exceed ``obstruction_threshold`` (meaning they are very
        close), return ``True``.
        """
        zone_strip = depth_map[:, self._zone_left:self._zone_right]
        if zone_strip.size == 0:
            return False
        bright_count = np.count_nonzero(zone_strip >= self._obstruction_thresh)
        ratio = bright_count / zone_strip.size
        return ratio >= self._obstruction_ratio
