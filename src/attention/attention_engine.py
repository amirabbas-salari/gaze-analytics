from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.gaze.gaze_fusion import FusedGaze
from src.calibration.screen_calibration import ScreenPoint


@dataclass
class AttentionResult:
    """
    Result of attention analysis for one face in one frame.

    The values in this object describe only the current frame.
    Temporal session management is handled separately.
    """

    is_looking_at_screen: bool

    attention_score: float

    gaze_confidence: float

    screen_point: Optional[ScreenPoint]

    head_pose_valid: bool

    gaze_valid: bool

    point_inside_screen: bool

    reason: str

    def to_dict(self) -> dict:
        return {
            "is_looking_at_screen": self.is_looking_at_screen,
            "attention_score": self.attention_score,
            "gaze_confidence": self.gaze_confidence,
            "screen_point": (
                {
                    "x": self.screen_point.x,
                    "y": self.screen_point.y,
                }
                if self.screen_point is not None
                else None
            ),
            "head_pose_valid": self.head_pose_valid,
            "gaze_valid": self.gaze_valid,
            "point_inside_screen": self.point_inside_screen,
            "reason": self.reason,
        }


class AttentionEngine:
    """
    Determines whether a person is looking at the display.

    The engine combines:

        1. Fused gaze confidence
        2. Gaze angle validity
        3. Head pose validity
        4. Calibrated screen point

    Important:
        This is a frame-level decision.

        It does NOT open or close look sessions.
        That responsibility belongs to SessionManager.

    Architecture:

        FusedGaze
             +
        ScreenPoint
             ↓
        AttentionEngine
             ↓
        AttentionResult
    """

    def __init__(
        self,
        min_attention_score: float = 0.60,
        min_gaze_confidence: float = 0.45,
        max_abs_gaze_yaw_degrees: float = 80.0,
        max_abs_gaze_pitch_degrees: float = 60.0,
        max_abs_head_yaw_degrees: float = 70.0,
        max_abs_head_pitch_degrees: float = 60.0,
        screen_margin: float = 0.05,
    ) -> None:

        if not 0.0 <= min_attention_score <= 1.0:
            raise ValueError(
                "min_attention_score must be between 0 and 1."
            )

        if not 0.0 <= min_gaze_confidence <= 1.0:
            raise ValueError(
                "min_gaze_confidence must be between 0 and 1."
            )

        if max_abs_gaze_yaw_degrees <= 0:
            raise ValueError(
                "max_abs_gaze_yaw_degrees must be positive."
            )

        if max_abs_gaze_pitch_degrees <= 0:
            raise ValueError(
                "max_abs_gaze_pitch_degrees must be positive."
            )

        if max_abs_head_yaw_degrees <= 0:
            raise ValueError(
                "max_abs_head_yaw_degrees must be positive."
            )

        if max_abs_head_pitch_degrees <= 0:
            raise ValueError(
                "max_abs_head_pitch_degrees must be positive."
            )

        if not 0.0 <= screen_margin < 0.5:
            raise ValueError(
                "screen_margin must be in [0, 0.5)."
            )

        self.min_attention_score = float(
            min_attention_score
        )

        self.min_gaze_confidence = float(
            min_gaze_confidence
        )

        self.max_abs_gaze_yaw_degrees = float(
            max_abs_gaze_yaw_degrees
        )

        self.max_abs_gaze_pitch_degrees = float(
            max_abs_gaze_pitch_degrees
        )

        self.max_abs_head_yaw_degrees = float(
            max_abs_head_yaw_degrees
        )

        self.max_abs_head_pitch_degrees = float(
            max_abs_head_pitch_degrees
        )

        self.screen_margin = float(
            screen_margin
        )

    # ========================================================
    # Public API
    # ========================================================

    def analyze(
        self,
        fused_gaze: FusedGaze,
        screen_point: Optional[ScreenPoint],
    ) -> AttentionResult:
        """
        Analyze one frame.

        Args:
            fused_gaze:
                Output of GazeFusion.

            screen_point:
                Calibrated screen coordinate.

        Returns:
            AttentionResult
        """

        if fused_gaze is None:
            raise ValueError(
                "fused_gaze cannot be None."
            )

        gaze_confidence = float(
            np.clip(
                fused_gaze.confidence,
                0.0,
                1.0,
            )
        )

        gaze_valid = self._validate_gaze(
            fused_gaze
        )

        head_pose_valid = self._validate_head_pose(
            fused_gaze
        )

        point_inside_screen = (
            self._is_point_inside_screen(
                screen_point
            )
        )

        directional_score = (
            self._calculate_direction_score(
                fused_gaze
            )
        )

        head_score = (
            self._calculate_head_pose_score(
                fused_gaze
            )
        )

        screen_score = (
            self._calculate_screen_score(
                screen_point
            )
        )

        attention_score = self._calculate_attention_score(
            gaze_confidence=gaze_confidence,
            directional_score=directional_score,
            head_score=head_score,
            screen_score=screen_score,
        )

        is_looking = (
            gaze_valid
            and head_pose_valid
            and point_inside_screen
            and gaze_confidence
            >= self.min_gaze_confidence
            and attention_score
            >= self.min_attention_score
        )

        reason = self._build_reason(
            is_looking=is_looking,
            gaze_valid=gaze_valid,
            head_pose_valid=head_pose_valid,
            point_inside_screen=point_inside_screen,
            gaze_confidence=gaze_confidence,
            attention_score=attention_score,
            screen_point=screen_point,
        )

        return AttentionResult(
            is_looking_at_screen=is_looking,
            attention_score=attention_score,
            gaze_confidence=gaze_confidence,
            screen_point=screen_point,
            head_pose_valid=head_pose_valid,
            gaze_valid=gaze_valid,
            point_inside_screen=point_inside_screen,
            reason=reason,
        )

    def analyze_without_screen_mapping(
        self,
        fused_gaze: FusedGaze,
    ) -> AttentionResult:
        """
        Analyze gaze when screen calibration is not available.

        This mode is useful during development.

        It cannot prove that the person is looking at the
        physical display; it only evaluates whether the gaze
        direction is plausible and frontal enough.

        Therefore the screen-point condition is intentionally
        not considered satisfied.
        """

        if fused_gaze is None:
            raise ValueError(
                "fused_gaze cannot be None."
            )

        gaze_confidence = float(
            np.clip(
                fused_gaze.confidence,
                0.0,
                1.0,
            )
        )

        gaze_valid = self._validate_gaze(
            fused_gaze
        )

        head_pose_valid = self._validate_head_pose(
            fused_gaze
        )

        directional_score = (
            self._calculate_direction_score(
                fused_gaze
            )
        )

        head_score = (
            self._calculate_head_pose_score(
                fused_gaze
            )
        )

        attention_score = float(
            np.clip(
                (
                    0.45 * gaze_confidence
                    + 0.35 * directional_score
                    + 0.20 * head_score
                ),
                0.0,
                1.0,
            )
        )

        is_looking = (
            gaze_valid
            and head_pose_valid
            and gaze_confidence
            >= self.min_gaze_confidence
            and attention_score
            >= self.min_attention_score
        )

        reason = (
            "Screen calibration unavailable; "
            "direction-only attention decision."
        )

        return AttentionResult(
            is_looking_at_screen=is_looking,
            attention_score=attention_score,
            gaze_confidence=gaze_confidence,
            screen_point=None,
            head_pose_valid=head_pose_valid,
            gaze_valid=gaze_valid,
            point_inside_screen=False,
            reason=reason,
        )

    # ========================================================
    # Validation
    # ========================================================

    def _validate_gaze(
        self,
        fused_gaze: FusedGaze,
    ) -> bool:
        """
        Validate gaze angles.
        """

        yaw = abs(
            fused_gaze.yaw_degrees
        )

        pitch = abs(
            fused_gaze.pitch_degrees
        )

        if not np.isfinite(yaw):
            return False

        if not np.isfinite(pitch):
            return False

        if (
            yaw
            > self.max_abs_gaze_yaw_degrees
        ):
            return False

        if (
            pitch
            > self.max_abs_gaze_pitch_degrees
        ):
            return False

        return True

    def _validate_head_pose(
        self,
        fused_gaze: FusedGaze,
    ) -> bool:
        """
        Validate the available head pose.

        If head pose is unavailable, the engine treats it as
        invalid rather than assuming a frontal face.
        """

        if fused_gaze.head_yaw is None:
            return False

        if fused_gaze.head_pitch is None:
            return False

        values = [
            fused_gaze.head_yaw,
            fused_gaze.head_pitch,
        ]

        if not all(
            np.isfinite(value)
            for value in values
        ):
            return False

        if (
            abs(fused_gaze.head_yaw)
            > self.max_abs_head_yaw_degrees
        ):
            return False

        if (
            abs(fused_gaze.head_pitch)
            > self.max_abs_head_pitch_degrees
        ):
            return False

        return True

    # ========================================================
    # Screen Validation
    # ========================================================

    def _is_point_inside_screen(
        self,
        screen_point: Optional[ScreenPoint],
    ) -> bool:
        """
        Check whether the predicted gaze point is inside the
        calibrated screen area.

        A small configurable margin is accepted around the
        screen boundary to account for calibration noise.
        """

        if screen_point is None:
            return False

        x = float(screen_point.x)
        y = float(screen_point.y)

        margin = self.screen_margin

        return (
            -margin <= x <= 1.0 + margin
            and
            -margin <= y <= 1.0 + margin
        )

    # ========================================================
    # Scores
    # ========================================================

    def _calculate_direction_score(
        self,
        fused_gaze: FusedGaze,
    ) -> float:
        """
        Score how plausible the gaze direction is.

        Smaller angular deviation from the forward direction
        produces a higher score.
        """

        yaw = abs(
            fused_gaze.yaw_degrees
        )

        pitch = abs(
            fused_gaze.pitch_degrees
        )

        yaw_score = self._linear_angle_score(
            yaw,
            self.max_abs_gaze_yaw_degrees,
        )

        pitch_score = self._linear_angle_score(
            pitch,
            self.max_abs_gaze_pitch_degrees,
        )

        return float(
            0.65 * yaw_score
            + 0.35 * pitch_score
        )

    def _calculate_head_pose_score(
        self,
        fused_gaze: FusedGaze,
    ) -> float:
        """
        Score the reliability of head orientation.
        """

        if (
            fused_gaze.head_yaw is None
            or fused_gaze.head_pitch is None
        ):
            return 0.0

        yaw_score = self._linear_angle_score(
            abs(fused_gaze.head_yaw),
            self.max_abs_head_yaw_degrees,
        )

        pitch_score = self._linear_angle_score(
            abs(fused_gaze.head_pitch),
            self.max_abs_head_pitch_degrees,
        )

        return float(
            0.60 * yaw_score
            + 0.40 * pitch_score
        )

    def _calculate_screen_score(
        self,
        screen_point: Optional[ScreenPoint],
    ) -> float:
        """
        Calculate screen-point quality.

        Points near the center have higher confidence than points
        very close to the boundary because edge predictions tend
        to be more sensitive to calibration errors.
        """

        if screen_point is None:
            return 0.0

        x = float(screen_point.x)
        y = float(screen_point.y)

        if not self._is_point_inside_screen(
            screen_point
        ):
            return 0.0

        # Distance from the nearest edge.
        edge_distance = min(
            x,
            1.0 - x,
            y,
            1.0 - y,
        )

        # Normalize the edge distance.
        # 0.0 = boundary
        # 0.5 = center
        score = min(
            1.0,
            edge_distance * 2.0,
        )

        # Never let screen position completely reject attention.
        return float(
            0.5 + 0.5 * score
        )

    @staticmethod
    def _linear_angle_score(
        angle: float,
        maximum: float,
    ) -> float:
        """
        Convert an angle to a [0, 1] score.
        """

        if maximum <= 0:
            return 0.0

        normalized = (
            abs(angle) / maximum
        )

        return float(
            np.clip(
                1.0 - normalized,
                0.0,
                1.0,
            )
        )

    def _calculate_attention_score(
        self,
        gaze_confidence: float,
        directional_score: float,
        head_score: float,
        screen_score: float,
    ) -> float:
        """
        Calculate the final frame-level attention score.

        Weights deliberately prioritize the actual gaze estimate.
        """

        score = (
            0.40 * gaze_confidence
            + 0.25 * directional_score
            + 0.15 * head_score
            + 0.20 * screen_score
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # Explanation
    # ========================================================

    def _build_reason(
        self,
        is_looking: bool,
        gaze_valid: bool,
        head_pose_valid: bool,
        point_inside_screen: bool,
        gaze_confidence: float,
        attention_score: float,
        screen_point: Optional[ScreenPoint],
    ) -> str:
        """
        Generate a human-readable explanation for the decision.
        """

        if is_looking:
            return (
                "Valid gaze toward the calibrated screen."
            )

        if not gaze_valid:
            return (
                "Gaze direction outside the accepted range."
            )

        if not head_pose_valid:
            return (
                "Head pose is missing or outside the "
                "accepted range."
            )

        if screen_point is None:
            return (
                "No calibrated screen point is available."
            )

        if not point_inside_screen:
            return (
                "Predicted gaze point is outside "
                "the screen boundary."
            )

        if (
            gaze_confidence
            < self.min_gaze_confidence
        ):
            return (
                "Gaze confidence is below the "
                "minimum threshold."
            )

        if (
            attention_score
            < self.min_attention_score
        ):
            return (
                "Final attention score is below "
                "the minimum threshold."
            )

        return (
            "Attention conditions were not satisfied."
        )