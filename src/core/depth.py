"""
depth.py — MiDaS Monocular Depth Estimation
=============================================
Wraps the **MiDaS v2.1 Small** model (loaded via ``torch.hub``) to produce a
normalised depth map from a single monocular RGB frame.

The depth map is returned as a uint8 image (0–255) where **higher values
indicate closer objects**.  This convention aligns with the way we visualise
"brightness = closeness" in the depth preview overlay.

A built-in **frame-skip cooldown** ensures the (relatively expensive) model
inference only runs every *N*-th frame.  On skipped frames the most recent
depth map is returned from cache, keeping the main loop's FPS high.

Usage
-----
    from src.core.depth import DepthEstimator

    depth = DepthEstimator(skip_frames=3)
    depth_map = depth.infer(frame)           # np.ndarray, uint8, H×W
    value     = depth.sample_depth(depth_map, bbox)  # float 0–255
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch


class DepthEstimator:
    """MiDaS v2.1 Small wrapper with frame-skip cooldown.

    Parameters
    ----------
    skip_frames : int
        Run the model only every *N*-th call to :meth:`infer`.
        Intermediate calls return the cached depth map.
    device : str | None
        ``"cpu"``, ``"cuda"``, etc.  ``None`` auto-selects CUDA if available.
    """

    def __init__(
        self,
        skip_frames: int = 3,
        device: Optional[str] = None,
    ) -> None:
        # Resolve device.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        # Load MiDaS Small via torch.hub.
        print("[INFO] Loading MiDaS v2.1 Small (this may download weights on first run) …")
        self._model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        )
        self._model.to(self._device).eval()

        # Load the matching MiDaS transforms.
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self._transform = midas_transforms.small_transform

        # Frame-skip state.
        self._skip_frames = max(1, skip_frames)
        self._frame_counter = 0
        self._cached_depth: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Produce a normalised depth map for the given BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3) from OpenCV.

        Returns
        -------
        np.ndarray
            Grayscale depth map (H × W), ``uint8``, 0–255.
            **Higher values = closer to the camera.**
        """
        self._frame_counter += 1

        # Return cached result on skipped frames.
        if (
            self._cached_depth is not None
            and self._frame_counter % self._skip_frames != 0
        ):
            return self._cached_depth

        # Pre-process: MiDaS expects RGB input.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        # Inference (no gradient computation needed).
        with torch.no_grad():
            prediction = self._model(input_batch)

            # Resize to original frame dimensions.
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalise to 0–255 uint8.
        # MiDaS outputs *inverse* depth (higher = closer), which is the
        # convention we want, so we just min-max normalise directly.
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)

        self._cached_depth = depth_norm
        return depth_norm

    @staticmethod
    def sample_depth(
        depth_map: np.ndarray,
        bbox: List[int],
        patch_size: int = 10,
    ) -> float:
        """Sample the depth at the centre of a bounding box.

        Takes the **median** of a small patch (default 10 × 10 pixels)
        centred on the bounding box to reduce noise.

        Parameters
        ----------
        depth_map : np.ndarray
            Normalised depth map (H × W, uint8).
        bbox : list[int]
            ``[x1, y1, x2, y2]`` in pixel coordinates.
        patch_size : int
            Side length of the square sampling patch (pixels).

        Returns
        -------
        float
            Median depth value in the range 0–255.
        """
        h, w = depth_map.shape[:2]
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

        half = patch_size // 2
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)

        patch = depth_map[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0

        return float(np.median(patch))

    @staticmethod
    def colorise(depth_map: np.ndarray) -> np.ndarray:
        """Apply a colour map to a grayscale depth map for visualisation.

        Parameters
        ----------
        depth_map : np.ndarray
            Grayscale depth map (H × W, uint8).

        Returns
        -------
        np.ndarray
            BGR colour-mapped image (H × W × 3).
        """
        return cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def skip_frames(self) -> int:
        """Current frame-skip interval."""
        return self._skip_frames

    @skip_frames.setter
    def skip_frames(self, value: int) -> None:
        self._skip_frames = max(1, value)
