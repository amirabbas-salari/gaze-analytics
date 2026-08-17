from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass
class VideoFrame:
    """
    Represents a single frame read from a video file.
    """

    image: object
    frame_index: int
    timestamp_ms: int


class VideoReader:
    """
    Handles video-file input.

    Responsibilities:
        - Open a video file
        - Read frames sequentially
        - Expose video metadata
        - Track frame index
        - Release video resources
    """

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_index = 0

    def open(self) -> None:
        """
        Open the video file.
        """

        if self._capture is not None and self._capture.isOpened():
            return

        capture = cv2.VideoCapture(self.video_path)

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video file:\n{self.video_path}"
            )

        self._capture = capture
        self._frame_index = 0

    def is_opened(self) -> bool:
        """
        Return True if the video is currently open.
        """

        return (
            self._capture is not None
            and self._capture.isOpened()
        )

    def read(self) -> Optional[VideoFrame]:
        """
        Read the next frame from the video.

        Returns:
            VideoFrame:
                When a frame is successfully read.

            None:
                When the video reaches its end.

        Raises:
            RuntimeError:
                If the video has not been opened or reading fails unexpectedly.
        """

        if not self.is_opened():
            raise RuntimeError(
                "Video is not open. Call open() before read()."
            )

        assert self._capture is not None

        success, frame = self._capture.read()

        if not success or frame is None:
            return None

        self._frame_index += 1

        timestamp_ms = int(
            self._capture.get(cv2.CAP_PROP_POS_MSEC)
        )

        return VideoFrame(
            image=frame,
            frame_index=self._frame_index,
            timestamp_ms=timestamp_ms,
        )

    def get_width(self) -> int:
        """
        Return video width.
        """

        if not self.is_opened():
            return 0

        assert self._capture is not None

        return int(
            self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

    def get_height(self) -> int:
        """
        Return video height.
        """

        if not self.is_opened():
            return 0

        assert self._capture is not None

        return int(
            self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def get_fps(self) -> float:
        """
        Return source video FPS.
        """

        if not self.is_opened():
            return 0.0

        assert self._capture is not None

        return float(
            self._capture.get(cv2.CAP_PROP_FPS)
        )

    def get_frame_count(self) -> int:
        """
        Return total number of frames.
        """

        if not self.is_opened():
            return 0

        assert self._capture is not None

        return int(
            self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    def get_duration_seconds(self) -> float:
        """
        Return the approximate video duration in seconds.
        """

        fps = self.get_fps()
        frame_count = self.get_frame_count()

        if fps <= 0:
            return 0.0

        return frame_count / fps

    @property
    def frame_index(self) -> int:
        """
        Return current frame index.
        """

        return self._frame_index

    def release(self) -> None:
        """
        Release the video resource.
        """

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoReader":
        """
        Support usage as a context manager.
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
        Automatically release the video resource.
        """

        self.release()