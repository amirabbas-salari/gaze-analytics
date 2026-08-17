from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gaze.l2cs import (
    GazePrediction,
    L2CSNet,
)
from src.vision.head_pose import HeadPose


@dataclass
class GazeDirection:
    """
    L2CS gaze direction.

    L2CS convention:
        yaw   -> horizontal direction
        pitch -> vertical direction
    """

    yaw: float
    pitch: float

    @property
    def yaw_degrees(self) -> float:
        return float(
            np.degrees(self.yaw)
        )

    @property
    def pitch_degrees(self) -> float:
        return float(
            np.degrees(self.pitch)
        )

    def to_vector(
        self,
        length: float = 1.0,
    ) -> np.ndarray:
        """
        Convert L2CS pitch/yaw to the same 2D direction
        convention used by the official L2CS renderer.

        Official L2CS:
            dx = -sin(pitch) * cos(yaw)
            dy = -sin(yaw)
        """

        if length <= 0:
            raise ValueError(
                "length must be positive."
            )

        dx = (
            -length
            * np.sin(self.pitch)
            * np.cos(self.yaw)
        )

        dy = (
            -length
            * np.sin(self.yaw)
        )

        vector = np.asarray(
            [
                dx,
                dy,
            ],
            dtype=np.float64,
        )

        return vector

    def to_3d_vector(
        self,
    ) -> np.ndarray:
        """
        Convert pitch/yaw to the L2CS Cartesian convention.

        This follows the official spherical conversion:

            x = -cos(pitch) * sin(yaw)
            y = -sin(pitch)
            z = -cos(pitch) * cos(yaw)
        """

        x = (
            -np.cos(self.pitch)
            * np.sin(self.yaw)
        )

        y = (
            -np.sin(self.pitch)
        )

        z = (
            -np.cos(self.pitch)
            * np.cos(self.yaw)
        )

        vector = np.asarray(
            [
                x,
                y,
                z,
            ],
            dtype=np.float64,
        )

        norm = np.linalg.norm(
            vector
        )

        if norm <= 1e-8:
            return np.asarray(
                [
                    0.0,
                    0.0,
                    -1.0,
                ],
                dtype=np.float64,
            )

        return (
            vector / norm
        )


@dataclass
class GazeResult:

    gaze: GazeDirection

    head_pose: Optional[
        HeadPose
    ]

    face_index: int = 0

    @property
    def yaw(self) -> float:
        return self.gaze.yaw

    @property
    def pitch(self) -> float:
        return self.gaze.pitch

    @property
    def yaw_degrees(self) -> float:
        return self.gaze.yaw_degrees

    @property
    def pitch_degrees(self) -> float:
        return self.gaze.pitch_degrees

    @property
    def vector(self) -> np.ndarray:
        return self.gaze.to_3d_vector()

    def to_dict(self) -> dict:
        return {
            "face_index": self.face_index,

            "gaze": {
                "yaw": self.yaw,
                "pitch": self.pitch,
                "yaw_degrees": self.yaw_degrees,
                "pitch_degrees": self.pitch_degrees,
                "vector": self.vector.tolist(),
            },

            "head_pose": (
                {
                    "yaw": self.head_pose.yaw,
                    "pitch": self.head_pose.pitch,
                    "roll": self.head_pose.roll,
                }
                if self.head_pose is not None
                else None
            ),
        }


class GazeEstimator:

    def __init__(
        self,
        l2cs: Optional[
            L2CSNet
        ] = None,
    ) -> None:

        self.l2cs = (
            l2cs
            if l2cs is not None
            else L2CSNet()
        )

    def load(self) -> None:

        if not self.l2cs.is_loaded:
            self.l2cs.load()

    def estimate(
        self,
        face_crop: np.ndarray,
        head_pose: Optional[
            HeadPose
        ] = None,
        face_index: int = 0,
    ) -> GazeResult:

        self.load()

        prediction = (
            self.l2cs.predict_single(
                face_crop
            )
        )

        direction = (
            self._prediction_to_direction(
                prediction
            )
        )

        return GazeResult(
            gaze=direction,
            head_pose=head_pose,
            face_index=face_index,
        )

    def estimate_batch(
        self,
        face_crops: list[np.ndarray],
        head_poses: Optional[
            list[Optional[HeadPose]]
        ] = None,
    ) -> list[GazeResult]:

        if not face_crops:
            return []

        self.load()

        predictions = (
            self.l2cs.predict(
                face_crops
            )
        )

        if head_poses is None:
            head_poses = [
                None
                for _ in face_crops
            ]

        if len(head_poses) != len(
            face_crops
        ):
            raise ValueError(
                "head_poses length must match "
                "face_crops length."
            )

        results = []

        for index, (
            prediction,
            head_pose,
        ) in enumerate(
            zip(
                predictions,
                head_poses,
            )
        ):

            direction = (
                self._prediction_to_direction(
                    prediction
                )
            )

            results.append(
                GazeResult(
                    gaze=direction,
                    head_pose=head_pose,
                    face_index=index,
                )
            )

        return results

    @staticmethod
    def _prediction_to_direction(
        prediction: GazePrediction,
    ) -> GazeDirection:

        return GazeDirection(
            yaw=float(
                prediction.yaw
            ),
            pitch=float(
                prediction.pitch
            ),
        )

    def unload(self) -> None:
        self.l2cs.unload()

    @property
    def device(self) -> str:
        return self.l2cs.device_name

    def __enter__(
        self,
    ) -> "GazeEstimator":

        self.load()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.unload()