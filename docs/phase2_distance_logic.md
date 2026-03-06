# Phase 2 — Distance Estimation & Decision Logic

**Date:** 2026-03-07  
**Status:** ✅ Complete

---

## Distance Estimation (Pinhole Geometry)

We use the **pinhole camera model** to estimate metric distance from a single
monocular frame — no depth sensor or neural network required.

### Formula

```
D = (H_real × f) / h_pixels
```

| Symbol     | Meaning                              | Units  |
| ---------- | ------------------------------------ | ------ |
| `D`        | Estimated distance to the object     | metres |
| `H_real`   | Known real-world height of the class | metres |
| `f`        | Camera focal length                  | pixels |
| `h_pixels` | Bounding-box height in the image     | pixels |

### Reference Height Table (subset)

| Class   | Height (m) |
| ------- | ---------- |
| person  | 1.70       |
| car     | 1.50       |
| truck   | 3.00       |
| bus     | 3.20       |
| bicycle | 1.10       |
| chair   | 0.80       |

The full table lives in `src/core/distance.py → DEFAULT_REFERENCE_HEIGHTS`.

---

## Decision Engine

### Critical Zone

The middle **40 %** of the frame width is the critical zone (30 %–70 % of
pixel columns). Only objects whose bounding-box centre falls inside this
strip are treated as direct threats.

### Flag Rules

| Condition                                       | Flag               |
| ----------------------------------------------- | ------------------ |
| Object in zone **AND** distance < 1.5 m         | `STOP`             |
| Object in zone **AND** 1.5 m ≤ distance < 3.0 m | `AVOID LEFT/RIGHT` |
| No in-zone threats or all > 3.0 m               | `GO`               |

### Weighted AVOID Direction

When the flag is `AVOID`, a **threat weight** is computed per detection:

```
w = 1 / distance²
```

Weights are summed for the left half and the right half of the frame.
The side with the **lower** total threat weight is recommended as the
escape direction (it has fewer / more-distant obstacles).

### Temporal Smoothing

A rolling window of the last 5 raw flags is maintained. The output flag
is the **mode** (most frequent value) of the window, eliminating
single-frame flicker.

---

## Next Steps (Phase 3)

- Integrate MiDaS or Depth Anything for neural depth estimation.
- End-to-end integration and stress testing.
