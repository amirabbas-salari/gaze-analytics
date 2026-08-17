from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.attention.attention_engine import (
    AttentionEngine,
)
from src.attention.session_manager import (
    SessionManager,
)
from src.calibration.screen_calibration import (
    ScreenCalibration,
)
from src.config.settings import (
    L2CS_MODEL_PATH,
    FACE_LANDMARKER_MODEL,
)
from src.gaze.gaze_estimator import (
    GazeEstimator,
)
from src.gaze.gaze_fusion import (
    GazeFusion,
)
from src.input.camera import Camera
from src.input.video import VideoReader
from src.analytics.statistics import (
    StatisticsEngine,
)
from src.recognition.recognizer import (
    FaceRecognizer,
)
from src.storage.database import Database
from src.tracking.tracker import (
    Detection,
    IoUTracker,
)
from src.visualization.overlay import (
    OverlayFaceData,
    OverlayRenderer,
)
from src.vision.face_landmarker import (
    FaceLandmarker,
)
from src.vision.face_utils import (
    FaceUtils,
)


# ============================================================
# Runtime Configuration
# ============================================================

DEFAULT_DATABASE_PATH = (
    "data/output/gaze_analytics.db"
)

DEFAULT_CALIBRATION_PATH = (
    "data/calibration/screen_calibration.json"
)


# ============================================================
# FPS Meter
# ============================================================

class FPSMeter:
    """
    Simple runtime FPS estimator.
    """

    def __init__(
        self,
        smoothing: float = 0.90,
    ) -> None:
        self.smoothing = float(
            np.clip(
                smoothing,
                0.0,
                0.99,
            )
        )

        self._last_time: Optional[float] = None
        self._fps: float = 0.0

    def update(self) -> float:
        now = time.perf_counter()

        if self._last_time is None:
            self._last_time = now
            return self._fps

        delta = now - self._last_time
        self._last_time = now

        if delta <= 0:
            return self._fps

        instant_fps = 1.0 / delta

        if self._fps <= 0:
            self._fps = instant_fps
        else:
            self._fps = (
                self.smoothing * self._fps
                + (1.0 - self.smoothing)
                * instant_fps
            )

        return self._fps


# ============================================================
# Advertisement Placeholder
# ============================================================

class AdvertisementContext:
    """
    Minimal advertisement context.

    The complete advertisement scheduler will be added later.

    For now, an optional ad_id can be supplied from CLI.
    """

    def __init__(
        self,
        ad_id: Optional[str] = None,
    ) -> None:
        self.ad_id = ad_id

    def current_ad_id(self) -> Optional[str]:
        return self.ad_id


# ============================================================
# Main Application
# ============================================================

class GazeAnalyticsApplication:
    """
    Main orchestration layer of the project.

    This class connects all independent modules while keeping
    their responsibilities separated.
    """

    def __init__(
        self,
        input_source: str = "camera",
        video_path: Optional[str] = None,
        camera_index: int = 0,
        database_path: str = DEFAULT_DATABASE_PATH,
        calibration_path: Optional[str] = (
            DEFAULT_CALIBRATION_PATH
        ),
        ad_id: Optional[str] = None,
        use_recognition: bool = True,
    ) -> None:

        self.input_source = input_source
        self.video_path = video_path
        self.camera_index = camera_index

        self.database_path = Path(
            database_path
        )

        self.calibration_path = (
            Path(calibration_path)
            if calibration_path
            else None
        )

        self.ad_context = AdvertisementContext(
            ad_id=ad_id
        )

        self.use_recognition = (
            use_recognition
        )

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        self.camera: Optional[Camera] = None
        self.video: Optional[VideoReader] = None

        # ----------------------------------------------------
        # Vision
        # ----------------------------------------------------

        self.face_landmarker = FaceLandmarker()

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        self.tracker = IoUTracker(
            iou_threshold=0.30,
            max_age=15,
            min_hits=2,
            max_tracks=20,
        )

        # ----------------------------------------------------
        # Recognition
        # ----------------------------------------------------

        self.recognizer: Optional[
            FaceRecognizer
        ] = None

        if self.use_recognition:
            self.recognizer = FaceRecognizer(
                model_name="buffalo_l",
                similarity_threshold=0.50,
                device_id=-1,
                gallery_path=(
                    "data/faces/gallery.json"
                ),
            )

        # ----------------------------------------------------
        # Gaze
        # ----------------------------------------------------

        self.gaze_estimator = GazeEstimator()

        self.gaze_fusion = GazeFusion(
            head_weight=0.35,
            eye_weight=0.65,
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        self.calibration = (
            ScreenCalibration()
        )

        self._calibration_loaded = False

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention_engine = AttentionEngine(
            min_attention_score=0.60,
            min_gaze_confidence=0.45,
        )

        self.session_manager = SessionManager(
            min_session_duration_ms=500,
            attention_timeout_ms=1000,
            max_session_gap_ms=3000,
        )

        # ----------------------------------------------------
        # Storage / Analytics
        # ----------------------------------------------------

        self.database = Database(
            self.database_path
        )

        self.statistics = StatisticsEngine(
            self.database
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        self.overlay = OverlayRenderer()

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        self.fps_meter = FPSMeter()

        self._running = False

    # ========================================================
    # Initialization
    # ========================================================

    def initialize(self) -> None:
        """
        Initialize all required modules.
        """

        self._validate_models()

        # Database
        self.statistics.initialize()

        # Face Landmarker
        self.face_landmarker.open()

        # L2CS
        self.gaze_estimator.load()

        # Recognition is optional.
        if self.recognizer is not None:
            self.recognizer.load()

        # Calibration is optional.
        self._load_calibration_if_available()

        # Input
        self._open_input()

    def _validate_models(self) -> None:
        """
        Validate model files before starting processing.
        """

        if not Path(
            FACE_LANDMARKER_MODEL
        ).exists():
            raise FileNotFoundError(
                "MediaPipe Face Landmarker model not found:\n"
                f"{FACE_LANDMARKER_MODEL}"
            )

        if not Path(
            L2CS_MODEL_PATH
        ).exists():
            raise FileNotFoundError(
                "L2CS model not found:\n"
                f"{L2CS_MODEL_PATH}"
            )

    def _open_input(self) -> None:
        """
        Open camera or video source.
        """

        if self.input_source == "camera":
            self.camera = Camera(
                camera_index=self.camera_index
            )

            self.camera.open()

            return

        if self.input_source == "video":
            if not self.video_path:
                raise ValueError(
                    "video_path is required when "
                    "input_source='video'."
                )

            self.video = VideoReader(
                self.video_path
            )

            self.video.open()

            return

        raise ValueError(
            "input_source must be 'camera' or 'video'."
        )

    def _load_calibration_if_available(
        self,
    ) -> None:
        """
        Load screen calibration when the file exists.

        Calibration is optional during development.
        """

        if self.calibration_path is None:
            return

        if not self.calibration_path.exists():
            return

        self.calibration.load(
            self.calibration_path
        )

        self._calibration_loaded = (
            self.calibration.is_calibrated
        )

    # ========================================================
    # Frame Input
    # ========================================================

    def _read_frame(
        self,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[int],
    ]:
        """
        Read a frame from the selected input source.
        """

        if self.input_source == "camera":
            if self.camera is None:
                raise RuntimeError(
                    "Camera is not initialized."
                )

            result = self.camera.read()

            return (
                result.image,
                result.timestamp_ms,
            )

        if self.video is None:
            raise RuntimeError(
                "Video reader is not initialized."
            )

        result = self.video.read()

        if result is None:
            return None, None

        return (
            result.image,
            result.timestamp_ms,
        )

    # ========================================================
    # Face Processing
    # ========================================================

    def _build_detections(
        self,
        frame: np.ndarray,
        faces,
    ) -> list[Detection]:
        """
        Convert MediaPipe faces into Tracker detections.
        """

        height, width = (
            frame.shape[:2]
        )

        detections: list[
            Detection
        ] = []

        for face_index, face in enumerate(
            faces
        ):
            box = (
                FaceUtils.get_face_bounding_box(
                    face,
                    width,
                    height,
                )
            )

            if box is None:
                continue

            detections.append(
                Detection(
                    bounding_box=box,
                    confidence=1.0,
                    face_index=face_index,
                )
            )

        return detections

    # ========================================================
    # Frame Pipeline
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> np.ndarray:
        """
        Execute the complete processing pipeline for one frame.
        """

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # 1. MediaPipe Face Landmarker
        # ----------------------------------------------------

        landmark_result = (
            self.face_landmarker.process(
                frame,
                timestamp_ms,
            )
        )

        faces = landmark_result.faces

        # ----------------------------------------------------
        # 2. Face Detection -> Tracker
        # ----------------------------------------------------

        detections = (
            self._build_detections(
                frame,
                faces,
            )
        )

        tracks = self.tracker.update(
            detections
        )

        # Map current detections to tracks.
        track_to_detection = (
            self._associate_tracks_to_faces(
                tracks,
                detections,
            )
        )

        overlay_faces: list[
            OverlayFaceData
        ] = []

        # ----------------------------------------------------
        # 3. Process Each Current Face
        # ----------------------------------------------------

        for track in tracks:

            if track.time_since_update != 0:
                continue

            detection_index = (
                track_to_detection.get(
                    track.track_id
                )
            )

            if detection_index is None:
                continue

            if detection_index >= len(faces):
                continue

            face = faces[
                detection_index
            ]

            box = detections[
                detection_index
            ].bounding_box

            # ------------------------------------------------
            # Face Crop
            # ------------------------------------------------

            face_crop = FaceUtils.crop_face(
                frame,
                face,
            )

            if face_crop is None:
                continue

            # ------------------------------------------------
            # 4. Head Pose
            # ------------------------------------------------

            head_pose = (
                self._estimate_head_pose(
                    face
                )
            )

            # ------------------------------------------------
            # 5. Face Recognition
            # ------------------------------------------------

            person_id = None

            if (
                self.recognizer is not None
            ):
                person_id = (
                    self._recognize_track(
                        track,
                        face_crop,
                    )
                )

            # ------------------------------------------------
            # 6. L2CS Gaze
            # ------------------------------------------------

            gaze_result = (
                self.gaze_estimator.estimate(
                    face_crop=face_crop,
                    head_pose=head_pose,
                    face_index=detection_index,
                )
            )

            # ------------------------------------------------
            # 7. Gaze Fusion
            # ------------------------------------------------

            fused_gaze = (
                self.gaze_fusion.fuse(
                    gaze_result
                )
            )

            # ------------------------------------------------
            # 8. Screen Mapping
            # ------------------------------------------------

            screen_point = (
                self._map_gaze_to_screen(
                    fused_gaze,
                    face,
                    frame_width,
                    frame_height,
                )
            )

            # ------------------------------------------------
            # 9. Attention
            # ------------------------------------------------

            if screen_point is not None:
                attention_result = (
                    self.attention_engine.analyze(
                        fused_gaze,
                        screen_point,
                    )
                )

            else:
                attention_result = (
                    self.attention_engine
                    .analyze_without_screen_mapping(
                        fused_gaze
                    )
                )

            # ------------------------------------------------
            # 10. Session
            # ------------------------------------------------

            completed_session = (
                self.session_manager.update(
                    track_id=track.track_id,
                    attention=attention_result,
                    timestamp_ms=timestamp_ms,
                    fused_gaze=fused_gaze,
                    person_id=person_id,
                )
            )

            if completed_session is not None:
                self._process_completed_session(
                    completed_session
                )

            # ------------------------------------------------
            # 11. Save Gaze Point
            # ------------------------------------------------

            if (
                screen_point is not None
                and attention_result.is_looking_at_screen
            ):
                self._save_gaze_point(
                    track_id=track.track_id,
                    person_id=person_id,
                    screen_point=screen_point,
                    attention_score=(
                        attention_result.attention_score
                    ),
                    timestamp_ms=timestamp_ms,
                )

            # ------------------------------------------------
            # 12. Overlay Data
            # ------------------------------------------------

            overlay_faces.append(
                OverlayFaceData(
                    face=face,
                    box=box,
                    track=track,
                    fused_gaze=fused_gaze,
                    attention=attention_result,
                    person_id=person_id,
                )
            )

        # ----------------------------------------------------
        # 13. Visualization
        # ----------------------------------------------------

        fps = self.fps_meter.update()

        active_sessions = (
            self.session_manager
            .active_session_count
        )

        output = self.overlay.render(
            frame,
            overlay_faces,
            fps=fps,
            active_viewer_count=(
                active_sessions
            ),
            current_ad_name=(
                self.ad_context.current_ad_id()
            ),
        )

        # ----------------------------------------------------
        # 14. Handle Tracks That Disappeared
        # ----------------------------------------------------

        self._close_lost_tracks(
            tracks,
            timestamp_ms,
        )

        return output

    # ========================================================
    # Head Pose
    # ========================================================

    def _estimate_head_pose(
        self,
        face,
    ):
        from src.vision.head_pose import (
            HeadPoseEstimator,
        )

        # Instantiate once lazily to keep the class independent.
        if not hasattr(
            self,
            "_head_pose_estimator",
        ):
            self._head_pose_estimator = (
                HeadPoseEstimator()
            )

        return (
            self._head_pose_estimator.estimate(
                face.transformation_matrix
            )
        )

    # ========================================================
    # Track / Face Association
    # ========================================================

    @staticmethod
    def _associate_tracks_to_faces(
        tracks,
        detections,
    ) -> dict[int, int]:
        """
        Associate active tracks with current detections.

        Uses IoU and ensures one detection belongs to only one track.
        """

        associations: dict[int, int] = {}

        used_detections: set[int] = set()

        for track in tracks:

            if track.time_since_update != 0:
                continue

            best_index = None
            best_iou = 0.0

            for index, detection in enumerate(
                detections
            ):
                if index in used_detections:
                    continue

                iou = (
                    IoUTrackerHelper.calculate_iou(
                        track.bounding_box,
                        detection.bounding_box,
                    )
                )

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is not None:
                associations[
                    track.track_id
                ] = best_index

                used_detections.add(
                    best_index
                )

        return associations

    # ========================================================
    # Recognition
    # ========================================================

    def _recognize_track(
        self,
        track,
        face_crop: np.ndarray,
    ) -> Optional[str]:
        """
        Recognize a tracked face and attach Person ID.

        Recognition is skipped when the track already has an
        identity. This prevents unnecessary inference on every frame.
        """

        if (
            track.person_id is not None
        ):
            return track.person_id

        if self.recognizer is None:
            return None

        try:
            result = self.recognizer.recognize(
                face_crop
            )

        except RuntimeError:
            return None

        if not result.is_known:
            return None

        if result.person_id is None:
            return None

        self.tracker.assign_person_id(
            track.track_id,
            result.person_id,
        )

        return result.person_id

    # ========================================================
    # Screen Mapping
    # ========================================================

    def _map_gaze_to_screen(
        self,
        fused_gaze,
        face,
        image_width: int,
        image_height: int,
    ):
        """
        Map fused gaze to normalized screen coordinates.

        For now this uses the calibrated polynomial model.

        When calibration is unavailable, no screen point is returned.
        """

        if not self._calibration_loaded:
            return None

        box = (
            FaceUtils.get_face_bounding_box(
                face,
                image_width,
                image_height,
            )
        )

        if box is None:
            return None

        face_center_x = (
            box.center[0]
            / image_width
        )

        face_center_y = (
            box.center[1]
            / image_height
        )

        try:
            from src.calibration.screen_calibration import (
                CalibrationSample,
            )

            sample = CalibrationSample(
                gaze_yaw=fused_gaze.yaw,
                gaze_pitch=fused_gaze.pitch,
                head_yaw=self._to_radians(
                    fused_gaze.head_yaw
                ),
                head_pitch=self._to_radians(
                    fused_gaze.head_pitch
                ),
                head_roll=self._to_radians(
                    fused_gaze.head_roll
                ),
                face_x=float(
                    np.clip(
                        face_center_x,
                        0.0,
                        1.0,
                    )
                ),
                face_y=float(
                    np.clip(
                        face_center_y,
                        0.0,
                        1.0,
                    )
                ),
                screen_x=0.0,
                screen_y=0.0,
            )

            return self.calibration.predict(
                sample
            )

        except (
            RuntimeError,
            ValueError,
            TypeError,
        ):
            return None

    @staticmethod
    def _to_radians(
        value: Optional[float],
    ) -> float:
        if value is None:
            return 0.0

        value = float(value)

        # Head pose is stored in degrees in FusedGaze.
        return float(
            np.radians(value)
        )

    # ========================================================
    # Gaze Point Persistence
    # ========================================================

    def _save_gaze_point(
        self,
        track_id: int,
        person_id: Optional[str],
        screen_point,
        attention_score: float,
        timestamp_ms: int,
    ) -> None:
        from src.storage.models import (
            GazePoint,
        )

        point = GazePoint(
            x=float(screen_point.x),
            y=float(screen_point.y),
            timestamp_ms=int(
                timestamp_ms
            ),
            person_id=person_id,
            track_id=track_id,
            ad_id=(
                self.ad_context.current_ad_id()
            ),
            attention_score=float(
                attention_score
            ),
        )

        try:
            self.database.save_gaze_point(
                point
            )

            self.database.commit()

        except Exception:
            self.database.rollback()
            raise

    # ========================================================
    # Completed Sessions
    # ========================================================

    def _process_completed_session(
        self,
        session,
    ) -> None:
        """
        Convert a completed LookSession into a persistent event.
        """

        self.statistics.process_session(
            look_session=session,
            person_id=session.person_id,
            ad_id=(
                self.ad_context.current_ad_id()
            ),
        )

    # ========================================================
    # Lost Tracks
    # ========================================================

    def _close_lost_tracks(
        self,
        tracks,
        timestamp_ms: int,
    ) -> None:
        """
        Close sessions for tracks that have expired.
        """

        for track in tracks:

            if (
                track.time_since_update
                <= self.tracker.max_age
            ):
                continue

            completed = (
                self.session_manager.close_track(
                    track_id=track.track_id,
                    timestamp_ms=timestamp_ms,
                )
            )

            if completed is not None:
                self._process_completed_session(
                    completed
                )

    # ========================================================
    # Shutdown
    # ========================================================

    def shutdown(self) -> None:
        """
        Gracefully stop all processing components.
        """

        self._running = False

        # Close all active attention sessions first.
        try:
            completed_sessions = (
                self.session_manager.close_all()
            )

            for session in completed_sessions:
                self._process_completed_session(
                    session
                )

        except Exception:
            pass

        if self.recognizer is not None:
            self.recognizer.close()

        self.gaze_estimator.unload()

        self.face_landmarker.close()

        if self.camera is not None:
            self.camera.release()

        if self.video is not None:
            self.video.release()

        self.database.close()

        cv2.destroyAllWindows()

    # ========================================================
    # Run
    # ========================================================

    def run(self) -> None:
        """
        Main application loop.
        """

        self.initialize()

        self._running = True

        try:
            while self._running:

                frame, timestamp_ms = (
                    self._read_frame()
                )

                if frame is None:
                    break

                if timestamp_ms is None:
                    break

                output = self.process_frame(
                    frame,
                    timestamp_ms,
                )

                cv2.imshow(
                    "Advertisement Gaze Analytics",
                    output,
                )

                key = cv2.waitKey(1) & 0xFF

                # ESC
                if key == 27:
                    break

                # Q
                if key in (
                    ord("q"),
                    ord("Q"),
                ):
                    break

        finally:
            self.shutdown()


# ============================================================
# IoU Helper
# ============================================================

class IoUTrackerHelper:
    """
    Small geometry helper used by the main pipeline.

    This keeps main.py independent of Tracker internals.
    """

    @staticmethod
    def calculate_iou(
        box_a,
        box_b,
    ) -> float:

        x1 = max(
            box_a.x1,
            box_b.x1,
        )

        y1 = max(
            box_a.y1,
            box_b.y1,
        )

        x2 = min(
            box_a.x2,
            box_b.x2,
        )

        y2 = min(
            box_a.y2,
            box_b.y2,
        )

        width = max(
            0,
            x2 - x1,
        )

        height = max(
            0,
            y2 - y1,
        )

        intersection = (
            width * height
        )

        if intersection <= 0:
            return 0.0

        area_a = (
            box_a.width
            * box_a.height
        )

        area_b = (
            box_b.width
            * box_b.height
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return float(
            intersection / union
        )


# ============================================================
# CLI
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Advertisement Gaze Analytics"
        )
    )

    parser.add_argument(
        "--source",
        choices=[
            "camera",
            "video",
        ],
        default="camera",
        help="Input source.",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index.",
    )

    parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE_PATH,
        help="SQLite database path.",
    )

    parser.add_argument(
        "--calibration",
        type=str,
        default=DEFAULT_CALIBRATION_PATH,
        help="Screen calibration JSON path.",
    )

    parser.add_argument(
        "--ad-id",
        type=str,
        default=None,
        help="Current advertisement ID.",
    )

    parser.add_argument(
        "--no-recognition",
        action="store_true",
        help="Disable Face Recognition.",
    )

    return parser


def main() -> None:
    parser = (
        build_argument_parser()
    )

    args = parser.parse_args()

    application = (
        GazeAnalyticsApplication(
            input_source=args.source,
            video_path=args.video,
            camera_index=args.camera,
            database_path=args.database,
            calibration_path=(
                args.calibration
            ),
            ad_id=args.ad_id,
            use_recognition=(
                not args.no_recognition
            ),
        )
    )

    application.run()


if __name__ == "__main__":
    main()