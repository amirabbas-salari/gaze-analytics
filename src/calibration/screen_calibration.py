from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np

from src.gaze.gaze_fusion import FusedGaze


@dataclass
class CalibrationSample:
    """
    One calibration observation.

    gaze_yaw / gaze_pitch:
        Gaze direction estimated by L2CS.

    head_yaw / head_pitch / head_roll:
        Head orientation estimated from MediaPipe.

    face_x / face_y:
        Normalized face center position in the camera frame.

    screen_x / screen_y:
        Target screen position, normalized to [0, 1].
    """

    gaze_yaw: float
    gaze_pitch: float

    head_yaw: float
    head_pitch: float
    head_roll: float

    face_x: float
    face_y: float

    screen_x: float
    screen_y: float


@dataclass
class ScreenPoint:
    """
    Normalized screen coordinate.
    """

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y

    def to_pixel(
        self,
        screen_width: int,
        screen_height: int,
    ) -> tuple[int, int]:
        x = int(
            round(
                np.clip(
                    self.x,
                    0.0,
                    1.0,
                )
                * (screen_width - 1)
            )
        )

        y = int(
            round(
                np.clip(
                    self.y,
                    0.0,
                    1.0,
                )
                * (screen_height - 1)
            )
        )

        return x, y


class ScreenCalibration:
    """
    Data-driven screen calibration.

    The current implementation uses polynomial regression built
    only with NumPy. No additional machine-learning dependency is
    required.

    Input features:

        gaze_yaw
        gaze_pitch
        head_yaw
        head_pitch
        head_roll
        face_x
        face_y

    Output:

        screen_x
        screen_y

    The model can later be replaced by a more advanced geometric
    or neural calibration model without changing the public API.
    """

    FEATURE_COUNT = 7

    def __init__(
        self,
        degree: int = 2,
        regularization: float = 1e-4,
    ) -> None:

        if degree < 1:
            raise ValueError(
                "degree must be at least 1."
            )

        if regularization < 0:
            raise ValueError(
                "regularization cannot be negative."
            )

        self.degree = int(degree)
        self.regularization = float(
            regularization
        )

        self._coefficients_x: Optional[
            np.ndarray
        ] = None

        self._coefficients_y: Optional[
            np.ndarray
        ] = None

        self._feature_mean: Optional[
            np.ndarray
        ] = None

        self._feature_std: Optional[
            np.ndarray
        ] = None

        self._trained = False

    # ========================================================
    # Feature Extraction
    # ========================================================

    @staticmethod
    def _build_feature_vector(
        sample: CalibrationSample,
    ) -> np.ndarray:
        """
        Convert one calibration sample into a 7D feature vector.
        """

        return np.asarray(
            [
                sample.gaze_yaw,
                sample.gaze_pitch,
                sample.head_yaw,
                sample.head_pitch,
                sample.head_roll,
                sample.face_x,
                sample.face_y,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _validate_screen_coordinate(
        value: float,
    ) -> float:

        if not np.isfinite(value):
            raise ValueError(
                "Screen coordinate must be finite."
            )

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # Polynomial Features
    # ========================================================

    def _polynomial_features(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Generate polynomial features.

        Degree 1:
            x1 x2 x3 ...

        Degree 2:
            x1 x2 ...
            x1² x1*x2 ...
        """

        features = np.asarray(
            features,
            dtype=np.float64,
        )

        if features.ndim == 1:
            features = features.reshape(
                1,
                -1,
            )

        if features.shape[1] != self.FEATURE_COUNT:
            raise ValueError(
                f"Expected {self.FEATURE_COUNT} features, "
                f"received {features.shape[1]}."
            )

        columns = [
            np.ones(
                (features.shape[0], 1),
                dtype=np.float64,
            )
        ]

        # First-order terms.
        columns.append(features)

        if self.degree >= 2:
            quadratic = []

            for i in range(
                self.FEATURE_COUNT
            ):
                for j in range(
                    i,
                    self.FEATURE_COUNT
                ):
                    quadratic.append(
                        (
                            features[:, i]
                            * features[:, j]
                        ).reshape(-1, 1)
                    )

            columns.extend(quadratic)

        if self.degree >= 3:
            cubic = []

            for i in range(
                self.FEATURE_COUNT
            ):
                for j in range(
                    i,
                    self.FEATURE_COUNT
                ):
                    for k in range(
                        j,
                        self.FEATURE_COUNT
                    ):
                        cubic.append(
                            (
                                features[:, i]
                                * features[:, j]
                                * features[:, k]
                            ).reshape(-1, 1)
                        )

            columns.extend(cubic)

        return np.hstack(columns)

    # ========================================================
    # Normalization
    # ========================================================

    def _fit_normalization(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate normalization statistics.
        """

        self._feature_mean = np.mean(
            features,
            axis=0,
        )

        self._feature_std = np.std(
            features,
            axis=0,
        )

        # Prevent division by zero for constant features.
        self._feature_std[
            self._feature_std < 1e-8
        ] = 1.0

        return (
            features - self._feature_mean
        ) / self._feature_std

    def _normalize(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize using stored calibration statistics.
        """

        if (
            self._feature_mean is None
            or self._feature_std is None
        ):
            raise RuntimeError(
                "Calibration normalization parameters "
                "are not available."
            )

        return (
            features - self._feature_mean
        ) / self._feature_std

    # ========================================================
    # Training
    # ========================================================

    def fit(
        self,
        samples: list[CalibrationSample],
    ) -> None:
        """
        Train the calibration mapping.

        A minimum of 9 well-distributed calibration points
        is strongly recommended.

        For polynomial degree 2, more samples are preferred.
        """

        if len(samples) < 9:
            raise ValueError(
                "At least 9 calibration samples are required."
            )

        raw_features = np.vstack(
            [
                self._build_feature_vector(
                    sample
                )
                for sample in samples
            ]
        )

        targets_x = np.asarray(
            [
                self._validate_screen_coordinate(
                    sample.screen_x
                )
                for sample in samples
            ],
            dtype=np.float64,
        )

        targets_y = np.asarray(
            [
                self._validate_screen_coordinate(
                    sample.screen_y
                )
                for sample in samples
            ],
            dtype=np.float64,
        )

        normalized = (
            self._fit_normalization(
                raw_features
            )
        )

        design_matrix = (
            self._polynomial_features(
                normalized
            )
        )

        self._coefficients_x = (
            self._fit_ridge(
                design_matrix,
                targets_x,
            )
        )

        self._coefficients_y = (
            self._fit_ridge(
                design_matrix,
                targets_y,
            )
        )

        self._trained = True

    def _fit_ridge(
        self,
        design_matrix: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """
        Solve ridge regression using a numerically stable
        least-squares formulation.
        """

        feature_count = design_matrix.shape[1]

        regularizer = np.eye(
            feature_count,
            dtype=np.float64,
        )

        # Do not penalize the bias term.
        regularizer[0, 0] = 0.0

        lhs = (
            design_matrix.T
            @ design_matrix
            + self.regularization
            * regularizer
        )

        rhs = (
            design_matrix.T
            @ targets
        )

        try:
            coefficients = np.linalg.solve(
                lhs,
                rhs,
            )

        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(
                lhs,
                rhs,
                rcond=None,
            )[0]

        return coefficients

    # ========================================================
    # Prediction
    # ========================================================

    def predict(
        self,
        sample: CalibrationSample,
    ) -> ScreenPoint:
        """
        Predict normalized screen coordinates.
        """

        if not self._trained:
            raise RuntimeError(
                "Screen calibration model has not been trained."
            )

        features = (
            self._build_feature_vector(
                sample
            ).reshape(
                1,
                -1,
            )
        )

        normalized = self._normalize(
            features
        )

        design_matrix = (
            self._polynomial_features(
                normalized
            )
        )

        screen_x = float(
            design_matrix[0]
            @ self._coefficients_x
        )

        screen_y = float(
            design_matrix[0]
            @ self._coefficients_y
        )

        return ScreenPoint(
            x=float(
                np.clip(
                    screen_x,
                    0.0,
                    1.0,
                )
            ),
            y=float(
                np.clip(
                    screen_y,
                    0.0,
                    1.0,
                )
            ),
        )

    # ========================================================
    # Feature Creation From Runtime Gaze
    # ========================================================

    @staticmethod
    def sample_from_fused_gaze(
        gaze: FusedGaze,
        face_x: float,
        face_y: float,
        screen_x: float,
        screen_y: float,
    ) -> CalibrationSample:
        """
        Build a calibration sample from a runtime FusedGaze result.

        face_x / face_y:
            Normalized face-center coordinates.

        screen_x / screen_y:
            Known calibration target coordinates.
        """

        if not 0.0 <= face_x <= 1.0:
            raise ValueError(
                "face_x must be in [0, 1]."
            )

        if not 0.0 <= face_y <= 1.0:
            raise ValueError(
                "face_y must be in [0, 1]."
            )

        head_yaw = (
            float(gaze.head_yaw)
            if gaze.head_yaw is not None
            else 0.0
        )

        head_pitch = (
            float(gaze.head_pitch)
            if gaze.head_pitch is not None
            else 0.0
        )

        head_roll = (
            float(gaze.head_roll)
            if gaze.head_roll is not None
            else 0.0
        )

        return CalibrationSample(
            gaze_yaw=float(
                gaze.yaw
            ),
            gaze_pitch=float(
                gaze.pitch
            ),
            head_yaw=float(
                np.radians(
                    head_yaw
                )
                if abs(head_yaw) > np.pi
                else head_yaw
            ),
            head_pitch=float(
                np.radians(
                    head_pitch
                )
                if abs(head_pitch) > np.pi
                else head_pitch
            ),
            head_roll=float(
                np.radians(
                    head_roll
                )
                if abs(head_roll) > np.pi
                else head_roll
            ),
            face_x=float(face_x),
            face_y=float(face_y),
            screen_x=float(
                ScreenCalibration._validate_screen_coordinate(
                    screen_x
                )
            ),
            screen_y=float(
                ScreenCalibration._validate_screen_coordinate(
                    screen_y
                )
            ),
        )

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        samples: list[CalibrationSample],
    ) -> dict:
        """
        Evaluate calibration error on a sample set.
        """

        if not self._trained:
            raise RuntimeError(
                "Calibration model has not been trained."
            )

        if not samples:
            raise ValueError(
                "At least one sample is required."
            )

        errors = []

        for sample in samples:
            prediction = self.predict(
                sample
            )

            dx = (
                prediction.x
                - sample.screen_x
            )

            dy = (
                prediction.y
                - sample.screen_y
            )

            distance = float(
                np.sqrt(
                    dx * dx
                    + dy * dy
                )
            )

            errors.append(
                distance
            )

        errors = np.asarray(
            errors,
            dtype=np.float64,
        )

        return {
            "mean_error": float(
                np.mean(errors)
            ),
            "median_error": float(
                np.median(errors)
            ),
            "max_error": float(
                np.max(errors)
            ),
            "min_error": float(
                np.min(errors)
            ),
            "samples": len(samples),
        }

    # ========================================================
    # Persistence
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the calibration model as JSON.
        """

        if not self._trained:
            raise RuntimeError(
                "Cannot save an untrained calibration model."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "version": 1,
            "degree": self.degree,
            "regularization": self.regularization,
            "feature_mean": (
                self._feature_mean.tolist()
                if self._feature_mean is not None
                else None
            ),
            "feature_std": (
                self._feature_std.tolist()
                if self._feature_std is not None
                else None
            ),
            "coefficients_x": (
                self._coefficients_x.tolist()
                if self._coefficients_x is not None
                else None
            ),
            "coefficients_y": (
                self._coefficients_y.tolist()
                if self._coefficients_y is not None
                else None
            ),
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )

    def load(
        self,
        path: str | Path,
    ) -> None:
        """
        Load a previously trained calibration model.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Calibration file not found:\n{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if data.get("version") != 1:
            raise ValueError(
                "Unsupported calibration model version."
            )

        self.degree = int(
            data["degree"]
        )

        self.regularization = float(
            data["regularization"]
        )

        self._feature_mean = np.asarray(
            data["feature_mean"],
            dtype=np.float64,
        )

        self._feature_std = np.asarray(
            data["feature_std"],
            dtype=np.float64,
        )

        self._coefficients_x = np.asarray(
            data["coefficients_x"],
            dtype=np.float64,
        )

        self._coefficients_y = np.asarray(
            data["coefficients_y"],
            dtype=np.float64,
        )

        self._trained = True

    @property
    def is_calibrated(self) -> bool:
        return self._trained