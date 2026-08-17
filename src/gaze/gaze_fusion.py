from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gaze.gaze_estimator import GazeDirection, GazeResult
from src.vision.head_pose import HeadPose


@dataclass
class FusedGaze:
    """
    Final gaze estimation after combining gaze and head pose.

    All angular values are represented in radians internally.
    """

    yaw: float
    pitch: float

    confidence: float

    gaze_yaw: float
    gaze_pitch: float

    head_yaw: Optional[float]
    head_pitch: Optional[float]
    head_roll: Optional[float]

    head_compensation_applied: bool

    @property
    def yaw_degrees(self) -> float:
        return float(np.degrees(self.yaw))

    @property
    def pitch_degrees(self) -> float:
        return float(np.degrees(self.pitch))

    @property
    def gaze_yaw_degrees(self) -> float:
        return float(np.degrees(self.gaze_yaw))

    @property
    def gaze_pitch_degrees(self) -> float:
        return float(np.degrees(self.gaze_pitch))

    def to_vector(self) -> np.ndarray:
        """
        Convert final yaw/pitch into a normalized 3D gaze vector.

        Coordinate convention used by this project:

            X → horizontal
            Y → vertical
            Z → forward
        """

        x = np.sin(self.yaw) * np.cos(self.pitch)
        y = np.sin(self.pitch)
        z = np.cos(self.yaw) * np.cos(self.pitch)

        vector = np.asarray(
            [x, y, z],
            dtype=np.float64,
        )

        norm = np.linalg.norm(vector)

        if norm <= 1e-8:
            return np.asarray(
                [0.0, 0.0, 1.0],
                dtype=np.float64,
            )

        return vector / norm

    def to_dict(self) -> dict:
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "yaw_degrees": self.yaw_degrees,
            "pitch_degrees": self.pitch_degrees,
            "gaze_yaw": self.gaze_yaw,
            "gaze_pitch": self.gaze_pitch,
            "gaze_yaw_degrees": self.gaze_yaw_degrees,
            "gaze_pitch_degrees": self.gaze_pitch_degrees,
            "head_yaw": self.head_yaw,
            "head_pitch": self.head_pitch,
            "head_roll": self.head_roll,
            "confidence": self.confidence,
            "head_compensation_applied": (
                self.head_compensation_applied
            ),
            "vector": self.to_vector().tolist(),
        }


class GazeFusion:
    """
    Combines L2CS gaze estimation with MediaPipe head pose.

    Important:
        This class does NOT perform screen calibration.

    The output is a camera-relative final gaze direction.

    The compensation strategy is intentionally explicit and
    configurable so it can later be replaced by a learned fusion
    model or a calibrated 3D geometry model.
    """

    def __init__(
        self,
        head_weight: float = 0.35,
        eye_weight: float = 0.65,
        max_head_yaw_correction: float = 45.0,
        max_head_pitch_correction: float = 30.0,
        min_confidence: float = 0.0,
    ) -> None:

        if head_weight < 0:
            raise ValueError(
                "head_weight cannot be negative."
            )

        if eye_weight < 0:
            raise ValueError(
                "eye_weight cannot be negative."
            )

        weight_sum = head_weight + eye_weight

        if weight_sum <= 0:
            raise ValueError(
                "At least one fusion weight must be positive."
            )

        if max_head_yaw_correction <= 0:
            raise ValueError(
                "max_head_yaw_correction must be positive."
            )

        if max_head_pitch_correction <= 0:
            raise ValueError(
                "max_head_pitch_correction must be positive."
            )

        self.head_weight = (
            head_weight / weight_sum
        )

        self.eye_weight = (
            eye_weight / weight_sum
        )

        self.max_head_yaw_correction = (
            float(max_head_yaw_correction)
        )

        self.max_head_pitch_correction = (
            float(max_head_pitch_correction)
        )

        self.min_confidence = float(
            np.clip(
                min_confidence,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # Public API
    # ========================================================

    def fuse(
        self,
        gaze_result: GazeResult,
    ) -> FusedGaze:
        """
        Fuse L2CS gaze direction with MediaPipe head pose.

        The initial implementation uses a simple explicit
        weighted compensation model.

        This is NOT the final scientific gaze model. It is a
        stable intermediate representation that we will later
        replace/upgrade using camera calibration and screen
        geometry.
        """

        if gaze_result is None:
            raise ValueError(
                "gaze_result cannot be None."
            )

        gaze = gaze_result.gaze
        head_pose = gaze_result.head_pose

        gaze_yaw = float(gaze.yaw)
        gaze_pitch = float(gaze.pitch)

        if head_pose is None:
            confidence = 0.65

            if confidence < self.min_confidence:
                confidence = 0.0

            return FusedGaze(
                yaw=gaze_yaw,
                pitch=gaze_pitch,
                confidence=confidence,
                gaze_yaw=gaze_yaw,
                gaze_pitch=gaze_pitch,
                head_yaw=None,
                head_pitch=None,
                head_roll=None,
                head_compensation_applied=False,
            )

        head_yaw = float(
            np.radians(head_pose.yaw)
        )

        head_pitch = float(
            np.radians(head_pose.pitch)
        )

        head_roll = float(
            np.radians(head_pose.roll)
        )

        corrected_yaw = self._compensate_yaw(
            gaze_yaw,
            head_yaw,
        )

        corrected_pitch = self._compensate_pitch(
            gaze_pitch,
            head_pitch,
        )

        confidence = self._calculate_confidence(
            gaze_yaw=gaze_yaw,
            gaze_pitch=gaze_pitch,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            head_roll=head_roll,
        )

        confidence = max(
            self.min_confidence,
            confidence,
        )

        return FusedGaze(
            yaw=corrected_yaw,
            pitch=corrected_pitch,
            confidence=confidence,
            gaze_yaw=gaze_yaw,
            gaze_pitch=gaze_pitch,
            head_yaw=float(head_pose.yaw),
            head_pitch=float(head_pose.pitch),
            head_roll=float(head_pose.roll),
            head_compensation_applied=True,
        )

    # ========================================================
    # Yaw Compensation
    # ========================================================

    def _compensate_yaw(
        self,
        gaze_yaw: float,
        head_yaw: float,
    ) -> float:
        """
        Compensate gaze yaw using head yaw.

        The compensation is bounded to prevent extreme head
        rotations from producing unstable output.

        The resulting angle remains relative to the camera.
        """

        head_yaw_degrees = float(
            np.degrees(head_yaw)
        )

        head_yaw_degrees = float(
            np.clip(
                head_yaw_degrees,
                -self.max_head_yaw_correction,
                self.max_head_yaw_correction,
            )
        )

        compensation = np.radians(
            head_yaw_degrees
        )

        return (
            gaze_yaw
            + (
                compensation
                * self.head_weight
            )
        )

    # ========================================================
    # Pitch Compensation
    # ========================================================

    def _compensate_pitch(
        self,
        gaze_pitch: float,
        head_pitch: float,
    ) -> float:
        """
        Compensate gaze pitch using head pitch.

        The correction is bounded for numerical stability.
        """

        head_pitch_degrees = float(
            np.degrees(head_pitch)
        )

        head_pitch_degrees = float(
            np.clip(
                head_pitch_degrees,
                -self.max_head_pitch_correction,
                self.max_head_pitch_correction,
            )
        )

        compensation = np.radians(
            head_pitch_degrees
        )

        return (
            gaze_pitch
            + (
                compensation
                * self.head_weight
            )
        )

    # ========================================================
    # Confidence
    # ========================================================

    def _calculate_confidence(
        self,
        gaze_yaw: float,
        gaze_pitch: float,
        head_yaw: float,
        head_pitch: float,
        head_roll: float,
    ) -> float:
        """
        Calculate a heuristic confidence score.

        This is intentionally NOT treated as a neural-network
        probability.

        It measures how reasonable the current combined pose
        is for later processing.
        """

        gaze_yaw_deg = abs(
            np.degrees(gaze_yaw)
        )

        gaze_pitch_deg = abs(
            np.degrees(gaze_pitch)
        )

        head_yaw_deg = abs(
            np.degrees(head_yaw)
        )

        head_pitch_deg = abs(
            np.degrees(head_pitch)
        )

        head_roll_deg = abs(
            np.degrees(head_roll)
        )

        # Gaze becomes less reliable at extreme angles.
        yaw_score = self._angle_score(
            gaze_yaw_deg,
            max_angle=80.0,
        )

        pitch_score = self._angle_score(
            gaze_pitch_deg,
            max_angle=60.0,
        )

        # Head rotation quality score.
        head_yaw_score = self._angle_score(
            head_yaw_deg,
            max_angle=70.0,
        )

        head_pitch_score = self._angle_score(
            head_pitch_deg,
            max_angle=60.0,
        )

        head_roll_score = self._angle_score(
            head_roll_deg,
            max_angle=45.0,
        )

        confidence = (
            0.30 * yaw_score
            + 0.20 * pitch_score
            + 0.25 * head_yaw_score
            + 0.15 * head_pitch_score
            + 0.10 * head_roll_score
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _angle_score(
        angle_degrees: float,
        max_angle: float,
    ) -> float:
        """
        Convert an absolute angle to a [0, 1] quality score.
        """

        if max_angle <= 0:
            return 0.0

        normalized = (
            abs(angle_degrees)
            / max_angle
        )

        return float(
            np.clip(
                1.0 - normalized,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # Convenience Methods
    # ========================================================

    def fuse_batch(
        self,
        gaze_results: list[GazeResult],
    ) -> list[FusedGaze]:
        """
        Fuse multiple gaze results.
        """

        return [
            self.fuse(result)
            for result in gaze_results
        ]