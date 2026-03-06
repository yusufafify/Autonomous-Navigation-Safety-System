"""
detector.py — YOLOv8 Object Detection Wrapper
===============================================
Thin abstraction over the Ultralytics YOLOv8 inference API.  Accepts a raw
BGR frame and returns a clean list of detection dictionaries that downstream
modules (depth estimation, decision engine) can consume without coupling to
the Ultralytics internals.

Usage
-----
    from src.core.detector import ObjectDetector

    detector = ObjectDetector(model_path="yolov8n.pt", confidence=0.5)
    detections = detector.detect(frame)

    for det in detections:
        print(det["class_name"], det["confidence"], det["bbox"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """Wrapper for YOLOv8 real-time object detection.

    Parameters
    ----------
    model_path : str
        Path to a YOLOv8 weights file (e.g. ``yolov8n.pt``).
        If the file does not exist locally, Ultralytics will download it
        automatically on first use.
    confidence : float
        Minimum confidence threshold for keeping a detection.
    device : str | None
        Inference device — ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
        ``None`` lets Ultralytics pick the best available device.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._device = device

        # Cache the class-name mapping from the loaded model.
        self._class_names: Dict[int, str] = self._model.names  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLOv8 inference on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3) as returned by OpenCV.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``bbox``        : ``[x1, y1, x2, y2]`` in pixel coordinates.
            - ``class_id``    : ``int`` COCO class index.
            - ``class_name``  : ``str`` human-readable label.
            - ``confidence``  : ``float`` detection confidence ∈ (0, 1].
        """
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_id": cls_id,
                        "class_name": self._class_names.get(cls_id, "unknown"),
                        "confidence": round(conf, 3),
                    }
                )

        return detections

    # ------------------------------------------------------------------
    # Drawing Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def draw(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """Draw bounding boxes and labels onto a frame.

        Parameters
        ----------
        frame : np.ndarray
            The image to annotate (modified **in-place** and also returned).
        detections : list[dict]
            Output of :meth:`detect`.
        color : tuple[int, int, int]
            BGR color for boxes and text.
        thickness : int
            Line thickness in pixels.
        font_scale : float
            OpenCV font scale factor.

        Returns
        -------
        np.ndarray
            The annotated frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"

            # Bounding box.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label background for readability.
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 10), (x1 + tw, y1), color, cv2.FILLED
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        return frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> Dict[int, str]:
        """Mapping of class IDs → human-readable names from the loaded model."""
        return dict(self._class_names)
