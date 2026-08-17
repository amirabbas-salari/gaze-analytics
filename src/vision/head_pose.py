from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HeadPose:
    """
    Head orientation in degrees.

    yaw:
        Rotation around the vertical axis.
        Positive/negative values represent left/right rotation.

    pitch:
        Rotation around the horizontal axis.
        Positive/negative values represent up/down rotation.

    roll:
        Rotation around the camera axis.
        Positive/negative values represent clockwise/counter-clockwise tilt.
    """

    yaw: float
    pitch: float
    roll: float

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """
        Return pose as (yaw, pitch, roll).
        """

        return self.yaw, self.pitch, self.roll


class HeadPoseEstimator:
    """
    Estimates head pose from MediaPipe facial transformation matrices.

    The transformation matrix is expected to describe the rigid
    transformation from the canonical face model to the detected face.
    """

    def __init__(self) -> None:
        self._previous_pose: Optional[HeadPose] = None

    @staticmethod
    def _validate_matrix(
        transformation_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize the transformation matrix.
        """

        matrix = np.asarray(
            transformation_matrix,
            dtype=np.float64,
        )

        if matrix.size != 16:
            raise ValueError(
                "Transformation matrix must contain exactly "
                "16 values."
            )

        matrix = matrix.reshape(4, 4)

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Transformation matrix contains invalid values."
            )

        return matrix

    @staticmethod
    def _extract_rotation_matrix(
        transformation_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Extract the 3x3 rotation component from a 4x4 transform.

        The matrix may contain a small amount of numerical scale.
        SVD orthonormalization removes that scale and produces a
        proper rotation matrix.
        """

        matrix = HeadPoseEstimator._validate_matrix(
            transformation_matrix
        )

        rotation = matrix[:3, :3]

        # Remove numerical scale/shear and recover the closest
        # orthonormal rotation matrix.
        u, _, vt = np.linalg.svd(rotation)

        rotation = u @ vt

        # Ensure a proper rotation matrix with determinant +1.
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt

        return rotation

    @staticmethod
    def _rotation_matrix_to_euler(
        rotation_matrix: np.ndarray,
    ) -> HeadPose:
        """
        Convert a 3x3 rotation matrix to Euler angles.

        The implementation follows the common yaw/pitch/roll
        decomposition used for camera/face orientation.

        Returned angles are in degrees.
        """

        r = np.asarray(
            rotation_matrix,
            dtype=np.float64,
        )

        if r.shape != (3, 3):
            raise ValueError(
                "Rotation matrix must have shape (3, 3)."
            )

        # Check if the rotation is close to a valid rotation matrix.
        if not np.allclose(
            r.T @ r,
            np.eye(3),
            atol=1e-3,
        ):
            raise ValueError(
                "Invalid rotation matrix."
            )

        # Handle the near-gimbal-lock case.
        sy = np.sqrt(
            r[0, 0] * r[0, 0]
            + r[1, 0] * r[1, 0]
        )

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(r[2, 1], r[2, 2])
            y = np.arctan2(
                -r[2, 0],
                sy,
            )
            z = np.arctan2(
                r[1, 0],
                r[0, 0],
            )

        else:
            x = np.arctan2(
                -r[1, 2],
                r[1, 1],
            )
            y = np.arctan2(
                -r[2, 0],
                sy,
            )
            z = 0.0

        pitch = float(np.degrees(x))
        yaw = float(np.degrees(y))
        roll = float(np.degrees(z))

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

    def estimate(
        self,
        transformation_matrix: Optional[np.ndarray],
    ) -> Optional[HeadPose]:
        """
        Estimate head pose from MediaPipe's facial transformation matrix.

        Args:
            transformation_matrix:
                4x4 transformation matrix returned by MediaPipe.

        Returns:
            HeadPose or None when no matrix is available.
        """

        if transformation_matrix is None:
            return None

        rotation_matrix = self._extract_rotation_matrix(
            transformation_matrix
        )

        pose = self._rotation_matrix_to_euler(
            rotation_matrix
        )

        self._previous_pose = pose

        return pose

    def estimate_from_landmarks(
        self,
        landmarks: list,
        image_width: int,
        image_height: int,
    ) -> Optional[HeadPose]:
        """
        Optional fallback method for estimating head pose directly
        from facial landmarks using solvePnP.

        This method is useful when a transformation matrix is not
        available.

        Expected landmarks:
            MediaPipe normalized landmarks with x/y/z attributes.
        """

        if not landmarks:
            return None

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "Image dimensions must be positive."
            )

        # Representative MediaPipe face landmarks:
        # nose tip, chin, eye corners and mouth corners.
        landmark_indices = [
            1,    # Nose
            152,  # Chin
            33,   # Left eye outer/corner
            263,  # Right eye outer/corner
            61,   # Left mouth corner
            291,  # Right mouth corner
        ]

        if len(landmarks) <= max(landmark_indices):
            return None

        image_points = []

        for index in landmark_indices:
            landmark = landmarks[index]

            x = float(landmark.x) * image_width
            y = float(landmark.y) * image_height

            image_points.append([x, y])

        image_points = np.asarray(
            image_points,
            dtype=np.float64,
        )

        # Generic 3D canonical face approximation.
        #
        # These points are used only as a fallback and are not meant
        # to replace a calibrated camera model.
        model_points = np.asarray(
            [
                [0.0, 0.0, 0.0],          # Nose
                [0.0, -63.6, -12.5],      # Chin
                [-43.3, 32.7, -26.0],     # Left eye
                [43.3, 32.7, -26.0],      # Right eye
                [-28.9, -28.9, -24.1],    # Left mouth
                [28.9, -28.9, -24.1],     # Right mouth
            ],
            dtype=np.float64,
        )

        focal_length = float(image_width)

        center = (
            image_width / 2.0,
            image_height / 2.0,
        )

        camera_matrix = np.asarray(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

        success, rotation_vector, _translation_vector = (
            cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        pose = self._rotation_matrix_to_euler(
            rotation_matrix
        )

        self._previous_pose = pose

        return pose

    @property
    def previous_pose(self) -> Optional[HeadPose]:
        """
        Return the last successfully estimated pose.
        """

        return self._previous_pose