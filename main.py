"""
main.py — Real-Time Collision Avoidance Demo
=============================================
Entry point for the Autonomous Navigation Safety System prototype.

Pipeline (per frame):
  1. Capture a frame from the threaded camera stream.
  2. Run YOLOv8 object detection.
  3. Run MiDaS depth estimation (with frame-skip cooldown).
  4. Estimate distance (geometry for known classes, depth fallback).
  5. Run safety-intervention decision engine → STOP / AVOID / GO.
  6. Render overlays: bounding boxes, distance labels, critical-zone
     markers, control flag banner, and depth preview inset.

Press **q** to quit.

Usage
-----
    python main.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.core.decision import SafetyIntervention
from src.core.depth import DepthEstimator
from src.core.detector import ObjectDetector
from src.core.distance import DistanceEstimator
from src.utils.camera import CameraStream


# ── Configuration ────────────────────────────────────────────────────
CAMERA_SOURCE: int | str = 0            # 0 = default webcam
YOLO_MODEL: str = "yolov8n.pt"          # Nano model — fast, lightweight
CONFIDENCE_THRESHOLD: float = 0.45      # Minimum detection confidence
FOCAL_LENGTH: float = 500.0             # Approx. focal length (pixels)
FRAME_WIDTH: int = 640                  # Expected frame width
CRITICAL_ZONE_PCT: float = 0.40         # Middle 40 % of the frame
STOP_DISTANCE: float = 1.5              # metres
AVOID_DISTANCE: float = 3.0             # metres
SMOOTHING_WINDOW: int = 5               # frames for temporal smoothing
DEPTH_SKIP_FRAMES: int = 3              # run MiDaS every N-th frame
DEPTH_PREVIEW_SIZE: tuple = (200, 150)  # (width, height) of inset
WINDOW_NAME: str = "Collision Avoidance — Live Feed"


# ── Drawing Helpers ──────────────────────────────────────────────────

def draw_critical_zone(
    frame: np.ndarray,
    zone_left: int,
    zone_right: int,
    color: tuple[int, int, int] = (255, 255, 0),
) -> None:
    """Draw two vertical dashed lines marking the critical zone."""
    h = frame.shape[0]
    dash_len = 15
    gap = 10
    for line_x in (zone_left, zone_right):
        y = 0
        while y < h:
            cv2.line(frame, (line_x, y), (line_x, min(y + dash_len, h)),
                     color, 2, cv2.LINE_AA)
            y += dash_len + gap


def draw_distance_labels(
    frame: np.ndarray,
    detections: list[dict],
) -> None:
    """Draw the estimated distance above each bounding box."""
    for det in detections:
        dist = det.get("distance")
        if dist is None:
            continue
        x1, y1 = det["bbox"][0], det["bbox"][1]

        # Show source indicator for depth-estimated distances.
        source = det.get("dist_source", "")
        suffix = " [D]" if source == "depth" else ""
        label = f"{dist:.1f}m{suffix}"

        # Background rectangle for readability.
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - th - 26),
            (x1 + tw + 6, y1 - 12),
            (40, 40, 40),
            cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1 + 3, y1 - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )


def draw_control_banner(
    frame: np.ndarray,
    flag: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a large control-flag banner at the top-centre of the frame."""
    w = frame.shape[1]
    banner_text = f"[ {flag} ]"

    (tw, th), _ = cv2.getTextSize(
        banner_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3
    )
    # Centre horizontally, position near the top.
    tx = (w - tw) // 2
    ty = 55

    # Semi-transparent background.
    overlay = frame.copy()
    pad = 12
    cv2.rectangle(
        overlay,
        (tx - pad, ty - th - pad),
        (tx + tw + pad, ty + pad),
        (30, 30, 30),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Banner text.
    cv2.putText(
        frame, banner_text, (tx, ty),
        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3, cv2.LINE_AA,
    )


def draw_depth_preview(
    frame: np.ndarray,
    depth_map: np.ndarray,
    preview_size: tuple[int, int] = (200, 150),
) -> None:
    """Overlay a colourised depth preview inset in the bottom-right corner."""
    pw, ph = preview_size
    fh, fw = frame.shape[:2]

    # Colourise and resize the depth map.
    colour_depth = DepthEstimator.colorise(depth_map)
    inset = cv2.resize(colour_depth, (pw, ph), interpolation=cv2.INTER_AREA)

    # Add "AI DEPTH VIEW" label.
    cv2.putText(
        inset, "AI DEPTH VIEW", (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )

    # Draw a border around the inset.
    cv2.rectangle(inset, (0, 0), (pw - 1, ph - 1), (200, 200, 200), 2)

    # Place in the bottom-right corner with a small margin.
    margin = 10
    y_start = fh - ph - margin
    x_start = fw - pw - margin
    frame[y_start:y_start + ph, x_start:x_start + pw] = inset


# ── Main Loop ────────────────────────────────────────────────────────

def main() -> None:
    """Run the real-time collision-avoidance loop."""

    # ── Initialise components ────────────────────────────────────────
    print("[INFO] Starting camera stream …")
    camera = CameraStream(src=CAMERA_SOURCE)
    camera.start()
    actual_width = camera.frame_size[0]
    print(f"[INFO] Camera resolution: {camera.frame_size}  |  FPS: {camera.fps}")

    print(f"[INFO] Loading YOLO model: {YOLO_MODEL}")
    detector = ObjectDetector(
        model_path=YOLO_MODEL,
        confidence=CONFIDENCE_THRESHOLD,
    )

    print("[INFO] Loading MiDaS depth model …")
    depth_model = DepthEstimator(skip_frames=DEPTH_SKIP_FRAMES)

    estimator = DistanceEstimator(focal_length=FOCAL_LENGTH)

    intervention = SafetyIntervention(
        frame_width=actual_width,
        critical_zone_pct=CRITICAL_ZONE_PCT,
        stop_distance=STOP_DISTANCE,
        avoid_distance=AVOID_DISTANCE,
        smoothing_window=SMOOTHING_WINDOW,
    )
    print("[INFO] All modules loaded. Press 'q' to quit.\n")

    # ── Main loop ────────────────────────────────────────────────────
    fps_start = time.perf_counter()
    frame_count = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[WARN] Empty frame — stream may have ended.")
                break

            # 1. Detect objects.
            detections = detector.detect(frame)

            # 2. Run depth estimation (internally frame-skipped).
            depth_map = depth_model.infer(frame)

            # 3. Estimate distances (hybrid: geometry + depth fallback).
            estimator.enrich_detections(detections, depth_map=depth_map)

            # 4. Decision engine (with obstruction check).
            result = intervention.process(detections, depth_map=depth_map)

            # 5. Draw overlays.
            # — Bounding boxes.
            detector.draw(frame, detections)

            # — Critical zone lines.
            draw_critical_zone(
                frame, intervention.zone_left, intervention.zone_right
            )

            # — Distance labels above each box.
            draw_distance_labels(frame, detections)

            # — Control flag banner.
            draw_control_banner(frame, result["flag"], result["color"])

            # — Depth preview inset (bottom-right).
            draw_depth_preview(frame, depth_map, DEPTH_PREVIEW_SIZE)

            # — FPS & object count.
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  |  Objects: {len(detections)}",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # 6. Display.
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        print("[INFO] Releasing resources …")
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
