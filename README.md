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

| File                   | Purpose                                                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/utils/camera.py`  | Threaded camera capture class — reads frames in a background thread to minimize latency.                                           |
| `src/core/detector.py` | YOLOv8 wrapper — loads a pretrained model and returns structured detection results (bounding boxes, class IDs, confidence scores). |
| `src/core/depth.py`    | _(Planned)_ Monocular depth estimation using MiDaS or Depth Anything.                                                              |
| `src/core/decision.py` | _(Planned)_ Decision engine that maps detections + depth to control flags.                                                         |
| `src/utils/viz.py`     | _(Planned)_ Visualization utilities for drawing overlays on frames.                                                                |
| `src/utils/config.py`  | _(Planned)_ Configuration loader for YAML/JSON parameter files.                                                                    |
| `main.py`              | Application entry point — ties camera, detector, and display together.                                                             |

---

## Roadmap

| Phase | Milestone                             | Status         |
| ----- | ------------------------------------- | -------------- |
| 1     | Environment setup & project scaffold  | ✅ Complete    |
| 2     | Distance estimation + decision logic  | ✅ Complete    |
| 3     | Monocular depth estimation            | 🔄 In Progress |
| 4     | Decision engine (STOP / LEFT / RIGHT) | ⬜ Planned     |
| 5     | End-to-end integration & testing      | ⬜ Planned     |

---

## License

This project is developed as a private freelance engagement. All rights are
reserved by the client unless otherwise agreed upon.
