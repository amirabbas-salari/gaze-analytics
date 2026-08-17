from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2

from src.config.settings import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    TARGET_FPS,
)


@dataclass
class CameraFrame:
    """
    Represents a single captured camera frame.
    """

    image: object
    frame_index: int
    timestamp_ms: int


class Camera:
    """
    Handles webcam initialization and frame capture.

    Responsibilities:
        - Open webcam
        - Configure resolution and FPS
        - Read frames
        - Track frame index
        - Release camera resources
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        target_fps: int = TARGET_FPS,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_index = 0

    def open(self) -> None:
        """
        Open and configure the webcam.
        """

        if self._capture is not None and self._capture.isOpened():
            return

        capture = cv2.VideoCapture(self.camera_index)

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open camera with index {self.camera_index}."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._capture = capture
        self._frame_index = 0

    def is_opened(self) -> bool:
        """
        Return True if the camera is currently open.
        """

        return (
            self._capture is not None
            and self._capture.isOpened()
        )

    def read(self) -> CameraFrame:
        """
        Capture and return one frame.

        Raises:
            RuntimeError:
                If the camera has not been opened or frame capture fails.
        """

        if not self.is_opened():
            raise RuntimeError(
                "Camera is not open. Call open() before read()."
            )

        assert self._capture is not None

        success, frame = self._capture.read()

        if not success or frame is None:
            raise RuntimeError(
                "Failed to capture frame from camera."
            )

        self._frame_index += 1

        timestamp_ms = int(
            self._capture.get(cv2.CAP_PROP_POS_MSEC)
        )

        if timestamp_ms <= 0:
            timestamp_ms = int(
                cv2.getTickCount()
                / cv2.getTickFrequency()
                * 1000
            )

        return CameraFrame(
            image=frame,
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
        )

    def get_actual_width(self) -> int:
        """
        Return the actual width reported by the camera.
        """

        if not self.is_opened():
            return 0

        assert self._capture is not None

        return int(
            self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

    def get_actual_height(self) -> int:
        """
        Return the actual height reported by the camera.
        """

        if not self.is_opened():
            return 0

        assert self._capture is not None

        return int(
            self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def get_actual_fps(self) -> float:
        """
        Return the FPS reported by the camera.
        """

        if not self.is_opened():
            return 0.0

        assert self._capture is not None

        return float(
            self._capture.get(cv2.CAP_PROP_FPS)
        )

    @property
    def frame_index(self) -> int:
        """
        Return the current frame index.
        """

        return self._frame_index

    def release(self) -> None:
        """
        Release the webcam resource.
        """

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        """
        Allow usage with a context manager.
        """

        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        Automatically release the camera.
        """

        self.release()