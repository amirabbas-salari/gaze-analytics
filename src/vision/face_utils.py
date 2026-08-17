from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from src.vision.face_landmarker import FaceLandmark, LandmarkPoint


@dataclass
class BoundingBox:
    """
    Pixel-space bounding box.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class EyeRegion:
    """
    Bounding region of one eye.
    """

    name: str
    bounding_box: BoundingBox


# MediaPipe Face Mesh landmark indices.
# These are kept in one place so future changes are easy.
LEFT_EYE_INDICES = (
    33,
    133,
    157,
    158,
    159,
    160,
    161,
    173,
    246,
)

RIGHT_EYE_INDICES = (
    362,
    263,
    384,
    385,
    386,
    387,
    388,
    398,
    466,
)

# A stable set of face contour points for estimating a face bounding box.
FACE_BOUNDING_INDICES = (
    10,
    33,
    54,
    67,
    103,
    109,
    127,
    136,
    150,
    152,
    234,
    251,
    284,
    297,
    323,
    332,
    356,
    366,
    379,
    389,
    447,
    454,
)


class FaceUtils:
    """
    Utility functions for MediaPipe facial landmarks.

    Responsibilities:
        - Convert normalized landmarks to pixel coordinates
        - Compute face bounding boxes
        - Compute eye regions
        - Crop faces and eyes
        - Clamp bounding boxes to image boundaries
        - Normalize crops for downstream models
    """

    @staticmethod
    def normalized_to_pixel(
        landmark: LandmarkPoint,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        """
        Convert normalized MediaPipe coordinates to pixel coordinates.
        """

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "Image width and height must be positive."
            )

        x = int(round(landmark.x * image_width))
        y = int(round(landmark.y * image_height))

        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))

        return x, y

    @staticmethod
    def landmarks_to_pixels(
        landmarks: Sequence[LandmarkPoint],
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Convert all normalized landmarks to an Nx2 pixel array.
        """

        points = []

        for landmark in landmarks:
            x, y = FaceUtils.normalized_to_pixel(
                landmark,
                image_width,
                image_height,
            )
            points.append([x, y])

        if not points:
            return np.empty(
                (0, 2),
                dtype=np.int32,
            )

        return np.asarray(
            points,
            dtype=np.int32,
        )

    @staticmethod
    def _box_from_points(
        points: Iterable[tuple[int, int]],
        image_width: int,
        image_height: int,
        padding_ratio: float = 0.0,
    ) -> Optional[BoundingBox]:
        """
        Build a bounding box from pixel points.
        """

        point_list = list(points)

        if not point_list:
            return None

        xs = [point[0] for point in point_list]
        ys = [point[1] for point in point_list]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        pad_x = int(round(width * padding_ratio))
        pad_y = int(round(height * padding_ratio))

        return FaceUtils.clamp_box(
            BoundingBox(
                x1=x1 - pad_x,
                y1=y1 - pad_y,
                x2=x2 + pad_x,
                y2=y2 + pad_y,
            ),
            image_width,
            image_height,
        )

    @staticmethod
    def clamp_box(
        box: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        """
        Clamp a bounding box to image boundaries.
        """

        x1 = max(0, min(box.x1, image_width - 1))
        y1 = max(0, min(box.y1, image_height - 1))
        x2 = max(x1 + 1, min(box.x2, image_width))
        y2 = max(y1 + 1, min(box.y2, image_height))

        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    @staticmethod
    def get_face_bounding_box(
        face: FaceLandmark,
        image_width: int,
        image_height: int,
        padding_ratio: float = 0.15,
    ) -> Optional[BoundingBox]:
        """
        Calculate a face bounding box from selected MediaPipe landmarks.
        """

        if not face.landmarks:
            return None

        points = []

        for index in FACE_BOUNDING_INDICES:
            if index >= len(face.landmarks):
                continue

            points.append(
                FaceUtils.normalized_to_pixel(
                    face.landmarks[index],
                    image_width,
                    image_height,
                )
            )

        return FaceUtils._box_from_points(
            points,
            image_width,
            image_height,
            padding_ratio=padding_ratio,
        )

    @staticmethod
    def get_eye_region(
        face: FaceLandmark,
        image_width: int,
        image_height: int,
        eye: str,
        padding_ratio: float = 0.35,
    ) -> Optional[EyeRegion]:
        """
        Calculate the bounding box of the left or right eye.

        Args:
            eye:
                "left" or "right"
        """

        eye = eye.lower().strip()

        if eye == "left":
            indices = LEFT_EYE_INDICES
        elif eye == "right":
            indices = RIGHT_EYE_INDICES
        else:
            raise ValueError(
                "eye must be either 'left' or 'right'."
            )

        if not face.landmarks:
            return None

        points = []

        for index in indices:
            if index >= len(face.landmarks):
                continue

            points.append(
                FaceUtils.normalized_to_pixel(
                    face.landmarks[index],
                    image_width,
                    image_height,
                )
            )

        box = FaceUtils._box_from_points(
            points,
            image_width,
            image_height,
            padding_ratio=padding_ratio,
        )

        if box is None:
            return None

        return EyeRegion(
            name=eye,
            bounding_box=box,
        )

    @staticmethod
    def get_both_eye_regions(
        face: FaceLandmark,
        image_width: int,
        image_height: int,
        padding_ratio: float = 0.35,
    ) -> tuple[
        Optional[EyeRegion],
        Optional[EyeRegion],
    ]:
        """
        Return left and right eye regions.
        """

        left = FaceUtils.get_eye_region(
            face,
            image_width,
            image_height,
            eye="left",
            padding_ratio=padding_ratio,
        )

        right = FaceUtils.get_eye_region(
            face,
            image_width,
            image_height,
            eye="right",
            padding_ratio=padding_ratio,
        )

        return left, right

    @staticmethod
    def crop(
        image: np.ndarray,
        box: BoundingBox,
    ) -> np.ndarray:
        """
        Crop an image using a pixel bounding box.
        """

        if image is None:
            raise ValueError("Image cannot be None.")

        if image.ndim != 3:
            raise ValueError(
                "Image must be a HxWxC array."
            )

        height, width = image.shape[:2]

        box = FaceUtils.clamp_box(
            box,
            width,
            height,
        )

        crop = image[
            box.y1:box.y2,
            box.x1:box.x2,
        ]

        if crop.size == 0:
            raise ValueError(
                "The requested crop is empty."
            )

        return crop.copy()

    @staticmethod
    def crop_face(
        image: np.ndarray,
        face: FaceLandmark,
        padding_ratio: float = 0.15,
    ) -> Optional[np.ndarray]:
        """
        Extract a face crop from a MediaPipe face.
        """

        height, width = image.shape[:2]

        box = FaceUtils.get_face_bounding_box(
            face,
            width,
            height,
            padding_ratio=padding_ratio,
        )

        if box is None:
            return None

        return FaceUtils.crop(
            image,
            box,
        )

    @staticmethod
    def crop_eye(
        image: np.ndarray,
        face: FaceLandmark,
        eye: str,
        padding_ratio: float = 0.35,
    ) -> Optional[np.ndarray]:
        """
        Extract one eye region from a face.
        """

        height, width = image.shape[:2]

        region = FaceUtils.get_eye_region(
            face,
            width,
            height,
            eye,
            padding_ratio=padding_ratio,
        )

        if region is None:
            return None

        return FaceUtils.crop(
            image,
            region.bounding_box,
        )

    @staticmethod
    def resize_with_padding(
        image: np.ndarray,
        target_size: tuple[int, int],
        pad_value: int = 0,
    ) -> np.ndarray:
        """
        Resize an image while preserving its aspect ratio.

        The remaining area is padded.
        """

        if image is None or image.size == 0:
            raise ValueError(
                "Input image cannot be empty."
            )

        target_width, target_height = target_size

        if target_width <= 0 or target_height <= 0:
            raise ValueError(
                "Target size must be positive."
            )

        source_height, source_width = image.shape[:2]

        scale = min(
            target_width / source_width,
            target_height / source_height,
        )

        resized_width = max(
            1,
            int(round(source_width * scale)),
        )

        resized_height = max(
            1,
            int(round(source_height * scale)),
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full(
            (
                target_height,
                target_width,
                image.shape[2],
            ),
            pad_value,
            dtype=image.dtype,
        )

        offset_x = (
            target_width - resized_width
        ) // 2

        offset_y = (
            target_height - resized_height
        ) // 2

        canvas[
            offset_y:offset_y + resized_height,
            offset_x:offset_x + resized_width,
        ] = resized

        return canvas

    @staticmethod
    def prepare_face_for_model(
        face_crop: np.ndarray,
        target_size: tuple[int, int] = (448, 448),
    ) -> np.ndarray:
        """
        Prepare a face crop for a 2D vision model.

        Returns:
            BGR image with the requested size.
        """

        if face_crop is None or face_crop.size == 0:
            raise ValueError(
                "Face crop cannot be empty."
            )

        return FaceUtils.resize_with_padding(
            face_crop,
            target_size,
            pad_value=0,
        )

    @staticmethod
    def get_landmark_pixel(
        face: FaceLandmark,
        index: int,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        """
        Return one landmark in pixel coordinates.
        """

        if index < 0 or index >= len(face.landmarks):
            raise IndexError(
                f"Landmark index {index} is out of range."
            )

        return FaceUtils.normalized_to_pixel(
            face.landmarks[index],
            image_width,
            image_height,
        )

    @staticmethod
    def average_landmark_position(
        face: FaceLandmark,
        indices: Sequence[int],
        image_width: int,
        image_height: int,
    ) -> Optional[tuple[float, float]]:
        """
        Calculate the average pixel position of selected landmarks.
        """

        points = []

        for index in indices:
            if index < 0 or index >= len(face.landmarks):
                continue

            x, y = FaceUtils.normalized_to_pixel(
                face.landmarks[index],
                image_width,
                image_height,
            )

            points.append((x, y))

        if not points:
            return None

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        return (
            float(np.mean(xs)),
            float(np.mean(ys)),
        )