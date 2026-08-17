from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from src.attention.attention_engine import AttentionResult
from src.gaze.gaze_fusion import FusedGaze
from src.tracking.tracker import Track
from src.vision.face_landmarker import FaceLandmark
from src.vision.face_utils import BoundingBox, FaceUtils


@dataclass
class OverlayStyle:
    """
    Visual configuration for the runtime overlay.
    """

    face_box_thickness: int = 2
    landmark_radius: int = 1

    gaze_vector_length: int = 120
    gaze_vector_thickness: int = 3

    gaze_point_radius: int = 8
    gaze_point_thickness: int = 2

    text_scale: float = 0.55
    text_thickness: int = 2

    panel_alpha: float = 0.72

    show_landmarks: bool = True
    show_face_box: bool = True
    show_gaze_vector: bool = True
    show_gaze_point: bool = True
    show_text: bool = True

    # BGR colors
    face_box_color: tuple[int, int, int] = (
        0,
        220,
        0,
    )

    landmark_color: tuple[int, int, int] = (
        180,
        180,
        180,
    )

    gaze_vector_color: tuple[int, int, int] = (
        0,
        180,
        255,
    )

    gaze_point_color: tuple[int, int, int] = (
        255,
        80,
        80,
    )

    valid_attention_color: tuple[int, int, int] = (
        0,
        220,
        0,
    )

    invalid_attention_color: tuple[int, int, int] = (
        0,
        0,
        220,
    )

    panel_color: tuple[int, int, int] = (
        20,
        20,
        20,
    )


@dataclass
class OverlayFaceData:
    """
    All visualization information for one face.
    """

    face: FaceLandmark

    box: Optional[BoundingBox] = None

    track: Optional[Track] = None

    fused_gaze: Optional[FusedGaze] = None

    attention: Optional[AttentionResult] = None

    person_id: Optional[str] = None


class OverlayRenderer:
    """
    Draws system state on top of an OpenCV frame.

    This class has no recognition, tracking, gaze estimation or
    attention logic.

    It only visualizes already calculated results.
    """

    def __init__(
        self,
        style: Optional[OverlayStyle] = None,
    ) -> None:
        self.style = (
            style
            if style is not None
            else OverlayStyle()
        )

    # ========================================================
    # Main API
    # ========================================================

    def render(
        self,
        frame: np.ndarray,
        faces: Sequence[OverlayFaceData],
        fps: Optional[float] = None,
        active_viewer_count: Optional[int] = None,
        current_ad_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render all available information on a frame.
        """

        if frame is None:
            raise ValueError(
                "frame cannot be None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "frame must be a numpy.ndarray."
            )

        if frame.ndim != 3:
            raise ValueError(
                "frame must have shape HxWxC."
            )

        output = frame.copy()

        for face_data in faces:
            self._render_face(
                output,
                face_data,
            )

        self._render_global_panel(
            output,
            fps=fps,
            active_viewer_count=active_viewer_count,
            current_ad_name=current_ad_name,
        )

        return output

    # ========================================================
    # Face Rendering
    # ========================================================

    def _render_face(
        self,
        frame: np.ndarray,
        face_data: OverlayFaceData,
    ) -> None:
        """
        Draw all information related to one face.
        """

        face = face_data.face

        if (
            self.style.show_landmarks
        ):
            self._draw_landmarks(
                frame,
                face,
            )

        if (
            self.style.show_face_box
            and face_data.box is not None
        ):
            self._draw_face_box(
                frame,
                face_data,
            )

        if (
            self.style.show_gaze_vector
            and face_data.fused_gaze is not None
        ):
            self._draw_gaze_vector(
                frame,
                face_data,
            )

        if (
            self.style.show_gaze_point
            and face_data.attention is not None
        ):
            self._draw_gaze_point(
                frame,
                face_data,
            )

        if self.style.show_text:
            self._draw_face_information(
                frame,
                face_data,
            )

    # ========================================================
    # Landmarks
    # ========================================================

    def _draw_landmarks(
        self,
        frame: np.ndarray,
        face: FaceLandmark,
    ) -> None:
        """
        Draw MediaPipe landmarks.
        """

        height, width = frame.shape[:2]

        for landmark in face.landmarks:
            x, y = FaceUtils.normalized_to_pixel(
                landmark,
                width,
                height,
            )

            cv2.circle(
                frame,
                (x, y),
                self.style.landmark_radius,
                self.style.landmark_color,
                -1,
                lineType=cv2.LINE_AA,
            )

    # ========================================================
    # Face Box
    # ========================================================

    def _draw_face_box(
        self,
        frame: np.ndarray,
        face_data: OverlayFaceData,
    ) -> None:
        """
        Draw face bounding box and track state.
        """

        box = face_data.box

        if box is None:
            return

        attention_is_valid = (
            face_data.attention is not None
            and face_data.attention.is_looking_at_screen
        )

        color = (
            self.style.valid_attention_color
            if attention_is_valid
            else self.style.face_box_color
        )

        cv2.rectangle(
            frame,
            (box.x1, box.y1),
            (box.x2, box.y2),
            color,
            self.style.face_box_thickness,
            lineType=cv2.LINE_AA,
        )

    # ========================================================
    # Gaze Vector
    # ========================================================


    def _draw_gaze_vector(
        self,
        frame: np.ndarray,
        face_data: OverlayFaceData,
    ) -> None:
        """
        Draw gaze direction using the official L2CS 2D
        pitch/yaw visualization convention.
        """

        if face_data.fused_gaze is None:
            return

        if face_data.box is None:
            return

        box = face_data.box

        center_x, center_y = box.center

        yaw = np.radians(
            face_data.fused_gaze.yaw_degrees
        )

        pitch = np.radians(
            face_data.fused_gaze.pitch_degrees
        )

        length = float(
            self.style.gaze_vector_length
        )

        # Same convention used by official L2CS draw_gaze():
        #
        # dx = -length * sin(pitch) * cos(yaw)
        # dy = -length * sin(yaw)

        dx = (
            -length
            * np.sin(pitch)
            * np.cos(yaw)
        )

        dy = (
            -length
            * np.sin(yaw)
        )

        end_x = int(
            round(
                center_x + dx
            )
        )

        end_y = int(
            round(
                center_y + dy
            )
        )

        cv2.arrowedLine(
            frame,
            (center_x, center_y),
            (end_x, end_y),
            self.style.gaze_vector_color,
            self.style.gaze_vector_thickness,
            line_type=cv2.LINE_AA,
            tipLength=0.18,
        )



    # ========================================================
    # Gaze Point
    # ========================================================

    def _draw_gaze_point(
        self,
        frame: np.ndarray,
        face_data: OverlayFaceData,
    ) -> None:
        """
        Draw calibrated gaze point when available.

        AttentionResult.screen_point is normalized in [0, 1].
        """

        attention = face_data.attention

        if attention is None:
            return

        point = attention.screen_point

        if point is None:
            return

        height, width = frame.shape[:2]

        x = int(
            round(
                np.clip(
                    point.x,
                    0.0,
                    1.0,
                )
                * (width - 1)
            )
        )

        y = int(
            round(
                np.clip(
                    point.y,
                    0.0,
                    1.0,
                )
                * (height - 1)
            )
        )

        color = (
            self.style.valid_attention_color
            if attention.is_looking_at_screen
            else self.style.gaze_point_color
        )

        cv2.circle(
            frame,
            (x, y),
            self.style.gaze_point_radius,
            color,
            self.style.gaze_point_thickness,
            lineType=cv2.LINE_AA,
        )

        cv2.drawMarker(
            frame,
            (x, y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    # ========================================================
    # Face Text
    # ========================================================

    def _draw_face_information(
        self,
        frame: np.ndarray,
        face_data: OverlayFaceData,
    ) -> None:
        """
        Draw Track ID, Person ID, pose, gaze and attention state.
        """

        if face_data.box is None:
            return

        box = face_data.box

        x = box.x1
        y = max(
            20,
            box.y1 - 10,
        )

        lines: list[str] = []

        # ----------------------------------------------------
        # Tracking / Recognition
        # ----------------------------------------------------

        if face_data.track is not None:
            lines.append(
                f"Track: "
                f"{face_data.track.track_id}"
            )

        person_id = (
            face_data.person_id
            or (
                face_data.track.person_id
                if face_data.track is not None
                else None
            )
        )

        if person_id:
            lines.append(
                f"Person: {person_id}"
            )

        # ----------------------------------------------------
        # Gaze
        # ----------------------------------------------------

        if face_data.fused_gaze is not None:
            gaze = face_data.fused_gaze

            lines.append(
                "Gaze: "
                f"Y {gaze.yaw_degrees:+.1f}° "
                f"P {gaze.pitch_degrees:+.1f}°"
            )

            lines.append(
                f"Gaze conf: "
                f"{gaze.confidence:.2f}"
            )

        # ----------------------------------------------------
        # Head Pose
        # ----------------------------------------------------

        if (
            face_data.fused_gaze is not None
            and face_data.fused_gaze.head_yaw
            is not None
        ):
            gaze = face_data.fused_gaze

            lines.append(
                "Head: "
                f"Y {gaze.head_yaw:+.1f}° "
                f"P {gaze.head_pitch:+.1f}° "
                f"R {gaze.head_roll:+.1f}°"
            )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        if face_data.attention is not None:
            attention = face_data.attention

            state = (
                "LOOKING"
                if attention.is_looking_at_screen
                else "NOT LOOKING"
            )

            lines.append(
                f"Attention: "
                f"{attention.attention_score:.2f}"
            )

            lines.append(
                state
            )

        if not lines:
            return

        self._draw_text_box(
            frame,
            lines,
            x=x,
            y=y,
        )

    # ========================================================
    # Text Box
    # ========================================================

    def _draw_text_box(
        self,
        frame: np.ndarray,
        lines: Sequence[str],
        x: int,
        y: int,
    ) -> None:
        """
        Draw a semi-transparent information panel.
        """

        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX

        line_height = 22
        padding = 8

        widths = []

        for line in lines:
            text_size, _ = cv2.getTextSize(
                line,
                font,
                self.style.text_scale,
                self.style.text_thickness,
            )

            widths.append(
                text_size[0]
            )

        panel_width = max(
            widths
        ) + (
            padding * 2
        )

        panel_height = (
            len(lines)
            * line_height
            + padding * 2
        )

        frame_height, frame_width = (
            frame.shape[:2]
        )

        x1 = int(
            np.clip(
                x,
                0,
                max(
                    0,
                    frame_width
                    - panel_width,
                ),
            )
        )

        y1 = int(
            np.clip(
                y - panel_height,
                0,
                max(
                    0,
                    frame_height
                    - panel_height,
                ),
            )
        )

        x2 = min(
            frame_width,
            x1 + panel_width,
        )

        y2 = min(
            frame_height,
            y1 + panel_height,
        )

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            self.style.panel_color,
            -1,
        )

        cv2.addWeighted(
            overlay,
            self.style.panel_alpha,
            frame,
            1.0 - self.style.panel_alpha,
            0,
            frame,
        )

        text_y = (
            y1
            + padding
            + 16
        )

        for line in lines:
            cv2.putText(
                frame,
                line,
                (
                    x1 + padding,
                    text_y,
                ),
                font,
                self.style.text_scale,
                (255, 255, 255),
                self.style.text_thickness,
                lineType=cv2.LINE_AA,
            )

            text_y += line_height

    # ========================================================
    # Global Panel
    # ========================================================

    def _render_global_panel(
        self,
        frame: np.ndarray,
        fps: Optional[float],
        active_viewer_count: Optional[int],
        current_ad_name: Optional[str],
    ) -> None:
        """
        Draw global application status.
        """

        lines = []

        if fps is not None:
            lines.append(
                f"FPS: {fps:.1f}"
            )

        if active_viewer_count is not None:
            lines.append(
                f"Viewers: "
                f"{active_viewer_count}"
            )

        if current_ad_name:
            lines.append(
                f"Ad: "
                f"{current_ad_name}"
            )

        if not lines:
            return

        self._draw_global_box(
            frame,
            lines,
        )

    def _draw_global_box(
        self,
        frame: np.ndarray,
        lines: Sequence[str],
    ) -> None:
        """
        Draw global status panel in the top-left corner.
        """

        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX

        padding = 8
        line_height = 23

        widths = []

        for line in lines:
            text_size, _ = cv2.getTextSize(
                line,
                font,
                self.style.text_scale,
                self.style.text_thickness,
            )

            widths.append(
                text_size[0]
            )

        width = max(
            widths
        ) + (
            padding * 2
        )

        height = (
            len(lines)
            * line_height
            + padding * 2
        )

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, height),
            self.style.panel_color,
            -1,
        )

        cv2.addWeighted(
            overlay,
            self.style.panel_alpha,
            frame,
            1.0 - self.style.panel_alpha,
            0,
            frame,
        )

        y = padding + 17

        for line in lines:
            cv2.putText(
                frame,
                line,
                (
                    padding,
                    y,
                ),
                font,
                self.style.text_scale,
                (255, 255, 255),
                self.style.text_thickness,
                lineType=cv2.LINE_AA,
            )

            y += line_height