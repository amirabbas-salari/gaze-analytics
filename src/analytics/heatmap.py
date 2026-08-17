from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from src.storage.database import Database
from src.storage.models import GazePoint


@dataclass
class HeatmapConfig:
    """
    Configuration for heatmap generation.
    """

    width: int = 1920
    height: int = 1080

    blur_kernel_size: int = 51
    blur_sigma: float = 0.0

    min_attention_score: float = 0.0

    normalize: bool = True

    intensity_power: float = 1.0

    # Radius used when adding an individual gaze point.
    point_radius: int = 35

    # Gaussian blur is applied after point accumulation.
    clip_percentile: float = 99.0

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(
                "width must be positive."
            )

        if self.height <= 0:
            raise ValueError(
                "height must be positive."
            )

        if self.blur_kernel_size <= 0:
            raise ValueError(
                "blur_kernel_size must be positive."
            )

        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1

        if self.blur_sigma < 0:
            raise ValueError(
                "blur_sigma cannot be negative."
            )

        if not 0.0 <= self.min_attention_score <= 1.0:
            raise ValueError(
                "min_attention_score must be between 0 and 1."
            )

        if self.intensity_power <= 0:
            raise ValueError(
                "intensity_power must be positive."
            )

        if self.point_radius <= 0:
            raise ValueError(
                "point_radius must be positive."
            )

        if not 0 < self.clip_percentile <= 100:
            raise ValueError(
                "clip_percentile must be in (0, 100]."
            )


@dataclass
class HeatmapResult:
    """
    Generated heatmap and related metadata.
    """

    heatmap: np.ndarray

    density: np.ndarray

    point_count: int

    total_weight: float

    width: int

    height: int

    def to_dict(self) -> dict:
        return {
            "point_count": self.point_count,
            "total_weight": self.total_weight,
            "width": self.width,
            "height": self.height,
        }


class HeatmapGenerator:
    """
    Generates gaze heatmaps from normalized gaze points.

    Input:
        GazePoint.x ∈ [0, 1]
        GazePoint.y ∈ [0, 1]

    Output:
        Grayscale density map
        +
        Optional colorized heatmap

    This class intentionally does not know anything about
    MediaPipe, L2CS, Tracking or Sessions.

    Its only responsibility is transforming gaze samples
    into a spatial attention representation.
    """

    def __init__(
        self,
        config: Optional[HeatmapConfig] = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else HeatmapConfig()
        )

    # ========================================================
    # Public API
    # ========================================================

    def generate(
        self,
        points: Iterable[GazePoint],
        background: Optional[np.ndarray] = None,
        colorize: bool = True,
    ) -> HeatmapResult:
        """
        Generate a heatmap from gaze points.

        Args:
            points:
                Iterable of GazePoint objects.

            background:
                Optional BGR advertisement image.

            colorize:
                If True, return a BGR colorized heatmap.
                Otherwise return a grayscale heatmap.

        Returns:
            HeatmapResult
        """

        point_list = [
            point
            for point in points
            if self._is_valid_point(point)
        ]

        density = self._build_density(
            point_list
        )

        if self.config.normalize:
            density = self._normalize_density(
                density
            )

        if colorize:
            heatmap = self._colorize(
                density
            )

            if background is not None:
                heatmap = self._overlay(
                    background,
                    heatmap,
                    density,
                )
        else:
            heatmap = density

        total_weight = float(
            sum(
                self._point_weight(point)
                for point in point_list
            )
        )

        return HeatmapResult(
            heatmap=heatmap,
            density=density,
            point_count=len(point_list),
            total_weight=total_weight,
            width=self.config.width,
            height=self.config.height,
        )

    def generate_from_database(
        self,
        database: Database,
        ad_id: Optional[str] = None,
        person_id: Optional[str] = None,
        background: Optional[np.ndarray] = None,
        colorize: bool = True,
    ) -> HeatmapResult:
        """
        Generate a heatmap directly from stored gaze points.
        """

        points = database.get_gaze_points(
            ad_id=ad_id,
            person_id=person_id,
        )

        return self.generate(
            points=points,
            background=background,
            colorize=colorize,
        )

    # ========================================================
    # Density
    # ========================================================

    def _build_density(
        self,
        points: list[GazePoint],
    ) -> np.ndarray:
        """
        Build the raw gaze density map.
        """

        density = np.zeros(
            (
                self.config.height,
                self.config.width,
            ),
            dtype=np.float32,
        )

        if not points:
            return density

        for point in points:
            x, y = self._normalized_to_pixel(
                point.x,
                point.y,
            )

            weight = self._point_weight(
                point
            )

            if weight <= 0:
                continue

            self._add_gaussian_point(
                density=density,
                x=x,
                y=y,
                weight=weight,
            )

        kernel_size = (
            self.config.blur_kernel_size
        )

        density = cv2.GaussianBlur(
            density,
            (
                kernel_size,
                kernel_size,
            ),
            sigmaX=self.config.blur_sigma,
            sigmaY=self.config.blur_sigma,
        )

        return density

    def _add_gaussian_point(
        self,
        density: np.ndarray,
        x: int,
        y: int,
        weight: float,
    ) -> None:
        """
        Add a weighted point using a circular Gaussian-like
        contribution.
        """

        radius = self.config.point_radius

        height, width = density.shape

        x1 = max(
            0,
            x - radius,
        )

        x2 = min(
            width,
            x + radius + 1,
        )

        y1 = max(
            0,
            y - radius,
        )

        y2 = min(
            height,
            y + radius + 1,
        )

        if x1 >= x2 or y1 >= y2:
            return

        local_width = x2 - x1
        local_height = y2 - y1

        grid_x = np.arange(
            x1,
            x2,
            dtype=np.float32,
        )

        grid_y = np.arange(
            y1,
            y2,
            dtype=np.float32,
        )

        mesh_x, mesh_y = np.meshgrid(
            grid_x,
            grid_y,
        )

        sigma = max(
            1.0,
            radius / 2.0,
        )

        gaussian = np.exp(
            -(
                (
                    (mesh_x - x) ** 2
                    + (mesh_y - y) ** 2
                )
                / (
                    2.0
                    * sigma
                    * sigma
                )
            )
        )

        density[
            y1:y2,
            x1:x2
        ] += (
            gaussian
            * weight
        )

    # ========================================================
    # Point Processing
    # ========================================================

    def _point_weight(
        self,
        point: GazePoint,
    ) -> float:
        """
        Calculate the contribution of one gaze point.
        """

        score = float(
            np.clip(
                point.attention_score,
                0.0,
                1.0,
            )
        )

        if (
            score
            < self.config.min_attention_score
        ):
            return 0.0

        return float(
            score
            ** self.config.intensity_power
        )

    def _is_valid_point(
        self,
        point: GazePoint,
    ) -> bool:
        """
        Validate one gaze point.
        """

        if point is None:
            return False

        if not np.isfinite(point.x):
            return False

        if not np.isfinite(point.y):
            return False

        if not np.isfinite(
            point.attention_score
        ):
            return False

        if not (
            0.0
            <= point.x
            <= 1.0
        ):
            return False

        if not (
            0.0
            <= point.y
            <= 1.0
        ):
            return False

        return True

    def _normalized_to_pixel(
        self,
        x: float,
        y: float,
    ) -> tuple[int, int]:
        """
        Convert normalized coordinates to pixels.
        """

        pixel_x = int(
            round(
                np.clip(
                    x,
                    0.0,
                    1.0,
                )
                * (
                    self.config.width
                    - 1
                )
            )
        )

        pixel_y = int(
            round(
                np.clip(
                    y,
                    0.0,
                    1.0,
                )
                * (
                    self.config.height
                    - 1
                )
            )
        )

        return pixel_x, pixel_y

    # ========================================================
    # Normalization
    # ========================================================

    def _normalize_density(
        self,
        density: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize density into [0, 1].

        A high percentile is used instead of the absolute maximum
        to prevent one extreme point from dominating the entire
        visualization.
        """

        if density.size == 0:
            return density

        positive = density[
            density > 0
        ]

        if positive.size == 0:
            return np.zeros_like(
                density,
                dtype=np.float32,
            )

        upper = np.percentile(
            positive,
            self.config.clip_percentile,
        )

        if upper <= 1e-8:
            return np.zeros_like(
                density,
                dtype=np.float32,
            )

        normalized = (
            density / upper
        )

        return np.clip(
            normalized,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    # ========================================================
    # Visualization
    # ========================================================

    @staticmethod
    def _colorize(
        density: np.ndarray,
    ) -> np.ndarray:
        """
        Convert grayscale density into a BGR heatmap.

        Uses OpenCV's built-in COLORMAP_JET.
        """

        grayscale = (
            np.clip(
                density,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(
            np.uint8
        )

        return cv2.applyColorMap(
            grayscale,
            cv2.COLORMAP_JET,
        )

    @staticmethod
    def _overlay(
        background: np.ndarray,
        heatmap: np.ndarray,
        density: np.ndarray,
        alpha: float = 0.55,
    ) -> np.ndarray:
        """
        Overlay heatmap on top of an advertisement image.

        Transparency is controlled by density:
            stronger density -> stronger heatmap.
        """

        if background is None:
            return heatmap

        if background.ndim != 3:
            raise ValueError(
                "Background must be a color image."
            )

        target_height, target_width = (
            background.shape[:2]
        )

        if (
            heatmap.shape[1] != target_width
            or heatmap.shape[0] != target_height
        ):
            heatmap = cv2.resize(
                heatmap,
                (
                    target_width,
                    target_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            density = cv2.resize(
                density,
                (
                    target_width,
                    target_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        background = background.copy()

        if background.shape[2] != 3:
            raise ValueError(
                "Background must have 3 channels."
            )

        if heatmap.shape[2] != 3:
            raise ValueError(
                "Heatmap must have 3 channels."
            )

        alpha_map = (
            np.clip(
                density,
                0.0,
                1.0,
            )
            * alpha
        ).astype(
            np.float32
        )

        alpha_map = alpha_map[
            :, :, None
        ]

        result = (
            background.astype(
                np.float32
            )
            * (
                1.0
                - alpha_map
            )
            +
            heatmap.astype(
                np.float32
            )
            * alpha_map
        )

        return np.clip(
            result,
            0,
            255,
        ).astype(
            np.uint8
        )

    # ========================================================
    # Output
    # ========================================================

    @staticmethod
    def save(
        image: np.ndarray,
        path: str | Path,
    ) -> None:
        """
        Save generated heatmap to disk.
        """

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(path),
            image,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save heatmap:\n{path}"
            )