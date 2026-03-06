"""
main.py — Real-Time Collision Avoidance Demo
=============================================
Entry point for the Autonomous Navigation Safety System prototype.

This script:
  1. Initialises a threaded camera stream.
  2. Runs YOLOv8 object detection on every frame.
  3. Draws annotated bounding boxes onto the live feed.
  4. Displays the result in an OpenCV window.

Press **q** to quit.

Usage
-----
    python main.py
"""

from __future__ import annotations

import sys
import time

import cv2

from src.core.detector import ObjectDetector
from src.utils.camera import CameraStream


# ── Configuration ────────────────────────────────────────────────────
CAMERA_SOURCE: int | str = 0          # 0 = default webcam
YOLO_MODEL: str = "yolov8n.pt"        # Nano model — fast, lightweight
CONFIDENCE_THRESHOLD: float = 0.45    # Minimum detection confidence
WINDOW_NAME: str = "Collision Avoidance — Live Feed"


def main() -> None:
    """Run the real-time object-detection loop."""

    # ── Initialise components ────────────────────────────────────────
    print("[INFO] Starting camera stream …")
    camera = CameraStream(src=CAMERA_SOURCE)
    camera.start()
    print(f"[INFO] Camera resolution: {camera.frame_size}  |  FPS: {camera.fps}")

    print(f"[INFO] Loading YOLO model: {YOLO_MODEL}")
    detector = ObjectDetector(
        model_path=YOLO_MODEL,
        confidence=CONFIDENCE_THRESHOLD,
    )
    print("[INFO] Model loaded. Press 'q' to quit.\n")

    # ── Main loop ────────────────────────────────────────────────────
    fps_start = time.perf_counter()
    frame_count = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[WARN] Empty frame — stream may have ended.")
                break

            # Run detection.
            detections = detector.detect(frame)

            # Annotate frame with bounding boxes.
            annotated = detector.draw(frame, detections)

            # Compute & overlay FPS counter.
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}  |  Objects: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Display.
            cv2.imshow(WINDOW_NAME, annotated)

            # Quit on 'q'.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # ── Cleanup ──────────────────────────────────────────────────
        print("[INFO] Releasing resources …")
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
