# Phase 1 — Project Initialization

**Date:** 2026-03-07  
**Status:** ✅ Complete

---

## Objectives

1. Establish the project repository structure.
2. Define the initial Python dependency list.
3. Create the camera capture module (`src/utils/camera.py`).
4. Create the YOLOv8 detection wrapper (`src/core/detector.py`).
5. Wire both modules into a working entry point (`main.py`).

---

## What Was Done

- Created the full directory tree: `data/`, `models/`, `src/core/`, `src/utils/`,
  `docs/`, and `tests/`.
- Added `requirements.txt` with OpenCV, Ultralytics (YOLOv8), NumPy, PyTorch,
  and torchvision.
- Implemented `CameraStream` — a threaded wrapper around OpenCV's
  `VideoCapture` that reads frames in a daemon thread to prevent UI lag.
- Implemented `ObjectDetector` — a lightweight wrapper around
  `ultralytics.YOLO` that returns structured detection results and can draw
  annotated bounding boxes on frames.
- Created `main.py` as the real-time simulation entry point.

---

## Next Steps (Phase 2)

- Integrate a monocular depth estimation model (MiDaS or Depth Anything) in
  `src/core/depth.py`.
- Build the decision engine (`src/core/decision.py`) that maps detection +
  depth data to control flags: **STOP**, **MOVE LEFT**, **MOVE RIGHT**.
- Add unit tests for the detection and decision logic in `tests/`.
