# Autonomous Navigation Support — Vision-Based Safety Intervention

> **Prototype collision avoidance system** that leverages a laptop webcam to detect
> obstacles in real time, estimate their distance, and output directional control
> flags (`STOP`, `MOVE LEFT`, `MOVE RIGHT`) for downstream navigation modules.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Tech Stack](#tech-stack)
3. [Repository Layout](#repository-layout)
4. [Quick Start](#quick-start)
5. [File Manifest](#file-manifest)
6. [Roadmap](#roadmap)
7. [License](#license)

---

## Abstract

This project demonstrates a **vision-only safety intervention layer** for
autonomous navigation platforms. Using a single monocular camera, the system:

1. **Detects obstacles** in the field of view via YOLOv8 object detection.
2. **Estimates relative depth** of each obstacle using a monocular depth model
   (MiDaS / Depth Anything).
3. **Generates control flags** — `STOP`, `MOVE LEFT`, or `MOVE RIGHT` — based on
   obstacle proximity and position in the frame.

The prototype runs entirely on a standard laptop and can serve as the safety
backbone for drones, mobile robots, or autonomous vehicles operating in
controlled environments.

---

## Tech Stack

| Component             | Technology                       |
| --------------------- | -------------------------------- |
| Language              | Python 3.10+                     |
| Computer Vision       | OpenCV 4.8+                      |
| Object Detection      | YOLOv8 (Ultralytics 8.3+)        |
| Depth Estimation      | MiDaS / Depth Anything (PyTorch) |
| Deep Learning Backend | PyTorch 2.1+                     |

---

## Repository Layout

```
Autonomous-Navigation-Safety-System/
├── data/                # Sample videos & images for offline testing
├── models/              # Downloaded YOLO & depth-estimation weights
├── src/
│   ├── core/            # Main logic modules
│   │   ├── detector.py  # YOLOv8 object detection wrapper
│   │   ├── depth.py     # Monocular depth estimation (future)
│   │   └── decision.py  # Decision engine: STOP / LEFT / RIGHT (future)
│   └── utils/
│       ├── camera.py    # Threaded webcam capture
│       ├── viz.py       # Drawing & visualization helpers (future)
│       └── config.py    # YAML/JSON config loader (future)
├── docs/                # Phase-by-phase progress documentation
├── tests/               # Unit & integration tests
├── main.py              # Real-time simulation entry point
├── requirements.txt     # Python dependencies
└── README.md            # ← You are here
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/Autonomous-Navigation-Safety-System.git
cd Autonomous-Navigation-Safety-System
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install PyTorch (CUDA 13.0)

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, Ultralytics will automatically download the YOLOv8
> nano weights (`yolov8n.pt`, ~6 MB) into the project directory.

### 5. Run the Real-Time Demo

```bash
python main.py
```

A window will open showing the live camera feed with YOLO bounding boxes
overlaid on detected objects. Press **`q`** to quit.

---

## File Manifest

| File                      | Purpose                                                                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/utils/camera.py`     | Threaded camera capture class — reads frames in a background thread to minimize latency.                                           |
| `src/utils/visualizer.py` | Safe corridor trapezoid and proximity heat bars.                                                                                   |
| `src/core/detector.py`    | YOLOv8 wrapper — loads a pretrained model and returns structured detection results (bounding boxes, class IDs, confidence scores). |
| `src/core/distance.py`    | Hybrid distance estimation — geometry (pinhole model) for known classes, MiDaS depth fallback for others.                          |
| `src/core/depth.py`       | MiDaS v2.1 Small wrapper with frame-skip cooldown for monocular depth estimation.                                                  |
| `src/core/decision.py`    | Safety intervention engine — STOP / AVOID / GO flags with temporal smoothing, weighted direction, and obstruction detection.       |
| `main.py`                 | Application entry point — full pipeline with `--save` recording option.                                                            |

---

## Final Features

| Feature                   | Description                                                                    |
| ------------------------- | ------------------------------------------------------------------------------ |
| **YOLOv8 Detection**      | Real-time object detection with bounding boxes and confidence scores.          |
| **Hybrid Distance**       | Geometry-based for known classes, MiDaS depth-map fallback for unknowns.       |
| **Safety Intervention**   | `STOP` / `AVOID LEFT` / `AVOID RIGHT` / `GO` flags with temporal smoothing.    |
| **Obstruction Detection** | Depth-map analysis triggers `STOP: OBSTRUCTION` even without YOLO detections.  |
| **Safe Corridor**         | Semi-transparent trapezoid showing navigable path (green = clear, red = stop). |
| **Proximity Heat Bars**   | Colour-coded bars next to each object (green / yellow / flashing red).         |
| **AI Depth Preview**      | Colourised MiDaS depth map inset (`AI DEPTH VIEW`) in the bottom-right.        |
| **Video Recording**       | `--save` flag records the full annotated feed to `/data` as AVI.               |

---

## Roadmap

| Phase | Milestone                            | Status      |
| ----- | ------------------------------------ | ----------- |
| 1     | Environment setup & project scaffold | ✅ Complete |
| 2     | Distance estimation + decision logic | ✅ Complete |
| 3     | MiDaS depth integration              | ✅ Complete |
| 4     | UI/UX polish & path planning         | ✅ Complete |

---

## Technical Limitations

> **Important:** This is a prototype system. The following limitations should
> be considered before deploying in any safety-critical context.

| Limitation                      | Details                                                                                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Monocular depth accuracy**    | MiDaS produces _relative_ depth, not metric. The linear scaling factor (`depth_scale`) is approximate and should be calibrated per environment.         |
| **Focal length assumption**     | Geometry-based distance assumes a fixed focal length (default 500 px). Accuracy improves significantly with proper camera calibration.                  |
| **CPU vs GPU latency**          | On CPU-only systems, MiDaS inference can drop FPS to 5–10. Frame-skip cooldown mitigates this but introduces depth-map staleness.                       |
| **YOLO class coverage**         | YOLOv8n is trained on COCO (80 classes). Unusual obstacles (e.g., construction barriers, debris) may not be detected; depth-map obstruction helps here. |
| **Single-camera field of view** | A single front-facing camera provides no rear or peripheral coverage. Multi-camera setups would be needed for full situational awareness.               |
| **Lighting conditions**         | Both YOLO and MiDaS performance degrade in low light, glare, or highly reflective environments.                                                         |

---