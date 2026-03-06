# Phase 3 — MiDaS Depth Integration

**Date:** 2026-03-07  
**Status:** ✅ Complete

---

## MiDaS Depth Module

**Model:** MiDaS v2.1 Small (loaded via `torch.hub`).

- Returns a normalised depth map (0–255, uint8), where **higher = closer**.
- **Frame-skip cooldown:** inference runs every N-th frame (default 3);
  intermediate frames return a cached depth map to preserve FPS.

---

## Hybrid Distance Strategy

| Condition                            | Strategy | Formula / Method                       |
| ------------------------------------ | -------- | -------------------------------------- |
| Class in reference-height dictionary | Geometry | `D = (H_real × f) / h_pixels`          |
| Class NOT in dictionary + depth map  | MiDaS    | `D ≈ depth_scale × (1 − median / 255)` |
| Neither available                    | —        | `None`                                 |

Each detection gains a `"dist_source"` key (`"geometry"` or `"depth"`) for
traceability; depth-estimated distances show a `[D]` suffix in the UI.

---

## Obstruction Detection

Even when YOLO detects **no objects**, the depth map is checked:

1. Extract the critical-zone strip from the depth map.
2. Count pixels exceeding the brightness threshold (default 220).
3. If >30 % of pixels are "very close" → emit **`STOP: OBSTRUCTION`**.

This catches walls, large flat surfaces, or any obstacle that YOLO might miss.

---

## UI Additions

- **AI DEPTH VIEW** — colourised (MAGMA) depth preview inset, 200 × 150 px,
  bottom-right corner, labelled.

---

## Next Steps (Phase 4)

- End-to-end stress testing with varied environments.
- Focal-length calibration utility.
- Performance profiling (CPU vs. CUDA benchmarks).
