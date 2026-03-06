"""
camera.py — Threaded Camera Capture
====================================
Provides a non-blocking video-capture interface built on top of OpenCV.
Frames are read in a background daemon thread so that the main loop never
stalls waiting for the next frame from the OS driver.

Usage
-----
    from src.utils.camera import CameraStream

    cam = CameraStream(src=0)
    cam.start()

    while True:
        frame = cam.read()
        if frame is None:
            break
        # ... process frame ...

    cam.release()
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    """Threaded wrapper around ``cv2.VideoCapture``.

    Reading frames in a dedicated thread prevents the main processing loop
    from blocking on I/O, which is especially important when inference adds
    per-frame latency.

    Parameters
    ----------
    src : int | str
        Camera index (``0`` for default webcam) or path to a video file.
    width : int | None
        Desired frame width.  ``None`` keeps the camera's default.
    height : int | None
        Desired frame height.  ``None`` keeps the camera's default.
    """

    def __init__(
        self,
        src: int | str = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {src}")

        # Optionally override resolution.
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read the very first frame so `self._frame` is never None at start.
        self._grabbed, self._frame = self._cap.read()

        # Thread-safety primitives.
        self._lock = threading.Lock()
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraStream":
        """Start the background frame-reading thread.

        Returns
        -------
        CameraStream
            ``self``, for convenient chaining (``cam = CameraStream().start()``).
        """
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def read(self) -> Optional[np.ndarray]:
        """Return the most recently captured frame.

        Returns
        -------
        np.ndarray | None
            BGR frame, or ``None`` if the stream has ended / was released.
        """
        with self._lock:
            if not self._grabbed:
                return None
            return self._frame.copy()

    def release(self) -> None:
        """Signal the background thread to stop and release the camera."""
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Frames-per-second reported by the capture device."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the captured frames."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update(self) -> None:
        """Continuously read frames until stopped or stream ends."""
        while not self._stopped:
            grabbed, frame = self._cap.read()
            with self._lock:
                self._grabbed = grabbed
                self._frame = frame

            if not grabbed:
                break
