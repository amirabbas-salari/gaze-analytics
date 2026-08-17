from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.config.settings import (
    FACE_LANDMARKER_MODEL,
    FACE_LANDMARKER_MIN_FACE_DETECTION_CONFIDENCE,
    FACE_LANDMARKER_MIN_FACE_PRESENCE_CONFIDENCE,
    FACE_LANDMARKER_MIN_TRACKING_CONFIDENCE,
    FACE_LANDMARKER_NUM_FACES,
)


@dataclass
class LandmarkPoint:
    """
    A normalized 3D facial landmark.
    """

    x: float
    y: float
    z: float
    visibility: Optional[float] = None
    presence: Optional[float] = None


@dataclass
class FaceLandmark:
    """
    Information related to one detected face.
    """

    landmarks: List[LandmarkPoint]
    blendshapes: List[dict]
    transformation_matrix: Optional[np.ndarray]


@dataclass
class FaceLandmarkerResult:
    """
    Result returned by the FaceLandmarker.
    """

    faces: List[FaceLandmark]
    timestamp_ms: int

    @property
    def face_count(self) -> int:
        return len(self.faces)


class FaceLandmarker:
    """
    MediaPipe Face Landmarker wrapper.

    Responsibilities:
        - Initialize MediaPipe Face Landmarker
        - Convert OpenCV frames to MediaPipe images
        - Detect facial landmarks
        - Extract blendshapes
        - Extract facial transformation matrices
        - Return project-level data structures
    """

    def __init__(
        self,
        model_path=FACE_LANDMARKER_MODEL,
        num_faces: int = FACE_LANDMARKER_NUM_FACES,
        min_face_detection_confidence: float = (
            FACE_LANDMARKER_MIN_FACE_DETECTION_CONFIDENCE
        ),
        min_face_presence_confidence: float = (
            FACE_LANDMARKER_MIN_FACE_PRESENCE_CONFIDENCE
        ),
        min_tracking_confidence: float = (
            FACE_LANDMARKER_MIN_TRACKING_CONFIDENCE
        ),
    ) -> None:

        self.model_path = str(model_path)
        self.num_faces = num_faces
        self.min_face_detection_confidence = (
            min_face_detection_confidence
        )
        self.min_face_presence_confidence = (
            min_face_presence_confidence
        )
        self.min_tracking_confidence = (
            min_tracking_confidence
        )

        self._landmarker: Optional[
            mp.tasks.vision.FaceLandmarker
        ] = None

        self._initialized = False

    def open(self) -> None:
        """
        Initialize the MediaPipe Face Landmarker.

        The VIDEO running mode is used because the project processes
        sequential frames from a video or camera stream.
        """

        if self._initialized:
            return

        if not self.model_path:
            raise ValueError(
                "MediaPipe Face Landmarker model path is empty."
            )

        model_path = str(self.model_path)

        if not cv2.os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face Landmarker model not found:\n{model_path}"
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=(
                self.min_face_detection_confidence
            ),
            min_face_presence_confidence=(
                self.min_face_presence_confidence
            ),
            min_tracking_confidence=(
                self.min_tracking_confidence
            ),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

        self._landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        )

        self._initialized = True

    def process(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> FaceLandmarkerResult:
        """
        Process one BGR OpenCV frame.

        Args:
            frame:
                OpenCV BGR image.

            timestamp_ms:
                Monotonically increasing timestamp in milliseconds.

        Returns:
            FaceLandmarkerResult
        """

        if not self._initialized or self._landmarker is None:
            raise RuntimeError(
                "FaceLandmarker is not initialized. "
                "Call open() before process()."
            )

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Input frame must be a numpy.ndarray."
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Input frame must have shape (height, width, 3)."
            )

        # OpenCV uses BGR while MediaPipe expects RGB/SRGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self._landmarker.detect_for_video(
            mp_image,
            int(timestamp_ms),
        )

        faces: List[FaceLandmark] = []

        face_landmarks = result.face_landmarks
        face_blendshapes = result.face_blendshapes
        transformation_matrices = (
            result.facial_transformation_matrixes
        )

        for face_index, raw_landmarks in enumerate(
            face_landmarks
        ):
            landmarks: List[LandmarkPoint] = []

            for landmark in raw_landmarks:
                landmarks.append(
                    LandmarkPoint(
                        x=float(landmark.x),
                        y=float(landmark.y),
                        z=float(landmark.z),
                        visibility=(
                            float(landmark.visibility)
                            if landmark.visibility is not None
                            else None
                        ),
                        presence=(
                            float(landmark.presence)
                            if landmark.presence is not None
                            else None
                        ),
                    )
                )

            blendshape_data: List[dict] = []

            if face_index < len(face_blendshapes):
                for category in face_blendshapes[face_index]:
                    blendshape_data.append(
                        {
                            "name": getattr(
                                category,
                                "category_name",
                                None,
                            ),
                            "score": float(
                                getattr(
                                    category,
                                    "score",
                                    0.0,
                                )
                            ),
                            "index": getattr(
                                category,
                                "index",
                                None,
                            ),
                        }
                    )

            transformation_matrix = None

            if face_index < len(transformation_matrices):
                matrix = np.asarray(
                    transformation_matrices[face_index],
                    dtype=np.float32,
                )

                transformation_matrix = matrix

            faces.append(
                FaceLandmark(
                    landmarks=landmarks,
                    blendshapes=blendshape_data,
                    transformation_matrix=(
                        transformation_matrix
                    ),
                )
            )

        return FaceLandmarkerResult(
            faces=faces,
            timestamp_ms=int(timestamp_ms),
        )

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """

        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

        self._initialized = False

    def __enter__(self) -> "FaceLandmarker":
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()