"""
distance.py — Hybrid Distance Estimation
==========================================
Estimates the metric distance (in metres) from the camera to a detected object
using one of two strategies:

1. **Geometry (primary)** — pinhole camera model ``D = (H_real × f) / h_pixels``
   for classes with a known real-world height.
2. **MiDaS depth-map fallback** — for unknown classes, samples the normalised
   depth map and converts to metres via a linear scale factor.

Usage
-----
    from src.core.distance import DistanceEstimator

    estimator = DistanceEstimator(focal_length=500)
    # Geometry only:
    distance = estimator.estimate(bbox=[100, 50, 300, 400], class_name="person")
    # Hybrid (with depth map):
    estimator.enrich_detections(detections, depth_map=depth_map)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.core.depth import DepthEstimator


# ── Default Reference Heights (metres) ───────────────────────────────
# Approximate standing/visible heights for common COCO object classes.
# These are deliberately conservative estimates suitable for a prototype.
DEFAULT_REFERENCE_HEIGHTS: Dict[str, float] = {
    "person": 1.70,
    "car": 1.50,
    "truck": 3.00,
    "bus": 3.20,
    "bicycle": 1.10,
    "motorcycle": 1.10,
    "chair": 0.80,
    "dog": 0.50,
    "cat": 0.30,
    "backpack": 0.50,
    "suitcase": 0.70,
    "bottle": 0.25,
    "cup": 0.15,
    "laptop": 0.30,
    "cell phone": 0.15,
    "tv": 0.50,
    "couch": 0.90,
    "bed": 0.60,
    "dining table": 0.75,
    "refrigerator": 1.80,
}


class DistanceEstimator:
    """Estimate object distance using pinhole-camera geometry.

    Parameters
    ----------
    focal_length : float
        Effective focal length of the camera **in pixels**.  A reasonable
        default for most laptop webcams at 640 × 480 is around 500–700.
        For better accuracy, calibrate with a known-size object at a
        measured distance.
    reference_heights : dict[str, float] | None
        Mapping of COCO class names → real-world heights (metres).
        ``None`` uses the built-in defaults.
    """

    def __init__(
        self,
        focal_length: float = 500.0,
        reference_heights: Optional[Dict[str, float]] = None,
        depth_scale: float = 5.0,
    ) -> None:
        self._focal_length = focal_length
        self._ref_heights = reference_heights or dict(DEFAULT_REFERENCE_HEIGHTS)
        self._depth_scale = depth_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        bbox: List[int],
        class_name: str,
    ) -> Optional[float]:
        """Estimate the distance to an object from its bounding box.

        Parameters
        ----------
        bbox : list[int]
            ``[x1, y1, x2, y2]`` bounding box in pixel coordinates.
        class_name : str
            COCO class label (must exist in the reference-height table).

        Returns
        -------
        float | None
            Estimated distance in **metres**, or ``None`` if the class is
            not in the reference table or the bounding box height is zero.
        """
        h_real = self._ref_heights.get(class_name)
        if h_real is None:
            return None

        h_pixels = abs(bbox[3] - bbox[1])
        if h_pixels == 0:
            return None

        distance = (h_real * self._focal_length) / h_pixels
        return round(distance, 2)

    def enrich_detections(
        self,
        detections: List[Dict[str, Any]],
        depth_map: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Add ``"distance"`` and ``"dist_source"`` keys to each detection.

        Strategy:
        - If the class is in the height dictionary → **geometry**.
        - Else if a ``depth_map`` is provided → **MiDaS depth fallback**.
        - Otherwise → ``None``.

        Parameters
        ----------
        detections : list[dict]
            Output of :class:`ObjectDetector.detect`.
        depth_map : np.ndarray | None
            Normalised depth map (H × W, uint8, higher = closer) from
            :class:`DepthEstimator`.  Pass ``None`` to use geometry only.

        Returns
        -------
        list[dict]
            Each dict gains ``"distance"`` (float | None) and
            ``"dist_source"`` (``"geometry"`` | ``"depth"`` | None).
        """
        for det in detections:
            geo_dist = self.estimate(det["bbox"], det["class_name"])

            if geo_dist is not None:
                det["distance"] = geo_dist
                det["dist_source"] = "geometry"
            elif depth_map is not None:
                det["distance"] = self._depth_to_metres(
                    depth_map, det["bbox"]
                )
                det["dist_source"] = "depth"
            else:
                det["distance"] = None
                det["dist_source"] = None

        return detections

    # ------------------------------------------------------------------
    # MiDaS depth → metric conversion
    # ------------------------------------------------------------------

    def _depth_to_metres(
        self,
        depth_map: np.ndarray,
        bbox: List[int],
    ) -> Optional[float]:
        """Convert a MiDaS depth sample to a rough metric distance.

        Uses a simple linear mapping::

            D ≈ depth_scale × (1 − median_value / 255)

        Where ``median_value`` is sampled from the centre of the bbox.

        Returns
        -------
        float | None
            Distance in metres, or ``None`` if sampling fails.
        """
        value = DepthEstimator.sample_depth(depth_map, bbox)
        if value <= 0:
            return None
        distance = self._depth_scale * (1.0 - value / 255.0)
        return round(max(distance, 0.1), 2)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def focal_length(self) -> float:
        """Current focal length (pixels)."""
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value: float) -> None:
        self._focal_length = value

    @property
    def reference_heights(self) -> Dict[str, float]:
        """Copy of the reference-height lookup table."""
        return dict(self._ref_heights)
