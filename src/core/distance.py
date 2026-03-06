"""
distance.py — Geometry-Based Distance Estimation
==================================================
Estimates the metric distance (in metres) from the camera to a detected object
using the **pinhole camera model**.

The core formula is::

    D = (H_real × f) / h_pixels

where:

- ``D``         — estimated distance to the object (metres).
- ``H_real``    — known real-world height of the object class (metres).
- ``f``         — effective focal length of the camera (pixels).
- ``h_pixels``  — height of the bounding box in the image (pixels).

This is a fast, zero-overhead approach that works well when the camera's
focal length is roughly calibrated and the detected objects have predictable
real-world sizes.

Usage
-----
    from src.core.distance import DistanceEstimator

    estimator = DistanceEstimator(focal_length=500)
    distance = estimator.estimate(bbox=[100, 50, 300, 400], class_name="person")
    # distance ≈ 2.43 m
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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
    ) -> None:
        self._focal_length = focal_length
        self._ref_heights = reference_heights or dict(DEFAULT_REFERENCE_HEIGHTS)

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
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add a ``"distance"`` key to each detection dict.

        Parameters
        ----------
        detections : list[dict]
            Output of :class:`ObjectDetector.detect`.

        Returns
        -------
        list[dict]
            The same list, with each dict now containing a ``"distance"``
            key (``float | None``).
        """
        for det in detections:
            det["distance"] = self.estimate(det["bbox"], det["class_name"])
        return detections

    # ------------------------------------------------------------------
    # Placeholder — MiDaS / Depth Anything
    # ------------------------------------------------------------------

    def estimate_depth_model(self, frame, bbox: List[int]) -> Optional[float]:
        """Placeholder for neural-network-based depth estimation.

        A future implementation would:
        1. Run MiDaS or Depth Anything on the full frame to get a depth map.
        2. Crop the depth map to the bounding-box region.
        3. Return the median depth value as the object's distance.

        Returns
        -------
        None
            Always ``None`` in the current prototype.
        """
        # TODO: Integrate MiDaS / Depth Anything when GPU headroom allows.
        return None

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
