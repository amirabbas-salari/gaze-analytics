from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.attention.attention_engine import AttentionResult
from src.gaze.gaze_fusion import FusedGaze


@dataclass
class LookSession:
    """
    Represents one continuous period of visual attention.

    A LookSession belongs to a Track ID and may optionally be
    associated with a persistent Person ID.

    During the session we accumulate:
        - Attention scores
        - Gaze yaw/pitch
        - Head pose
        - Gaze screen coordinates
        - Number of valid gaze samples
    """

    session_id: int

    track_id: int

    person_id: Optional[str]

    start_time_ms: int

    last_seen_time_ms: int

    end_time_ms: Optional[int]

    duration_ms: int

    sample_count: int

    valid_gaze_sample_count: int

    # Attention statistics
    max_attention_score: float
    attention_score_sum: float
    average_attention_score: float

    # Gaze statistics
    gaze_yaw_sum: float
    gaze_pitch_sum: float

    gaze_yaw: float
    gaze_pitch: float

    max_gaze_confidence: float
    gaze_confidence_sum: float
    average_gaze_confidence: float

    # Head pose statistics
    head_yaw_sum: float
    head_pitch_sum: float
    head_roll_sum: float

    head_yaw: Optional[float]
    head_pitch: Optional[float]
    head_roll: Optional[float]

    # Screen gaze points
    start_screen_x: Optional[float] = None
    start_screen_y: Optional[float] = None

    end_screen_x: Optional[float] = None
    end_screen_y: Optional[float] = None

    last_screen_x: Optional[float] = None
    last_screen_y: Optional[float] = None

    # Session state
    is_active: bool = True

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def average_gaze_yaw(self) -> float:
        return self.gaze_yaw

    @property
    def average_gaze_pitch(self) -> float:
        return self.gaze_pitch

    def update_gaze(
        self,
        fused_gaze: Optional[FusedGaze],
    ) -> None:
        """
        Accumulate gaze information from one valid frame.
        """

        if fused_gaze is None:
            return

        self.valid_gaze_sample_count += 1

        # ----------------------------------------------------
        # Gaze
        # ----------------------------------------------------

        self.gaze_yaw_sum += float(
            fused_gaze.yaw
        )

        self.gaze_pitch_sum += float(
            fused_gaze.pitch
        )

        self.gaze_yaw = (
            self.gaze_yaw_sum
            / self.valid_gaze_sample_count
        )

        self.gaze_pitch = (
            self.gaze_pitch_sum
            / self.valid_gaze_sample_count
        )

        confidence = float(
            np.clip(
                fused_gaze.confidence,
                0.0,
                1.0,
            )
        )

        self.gaze_confidence_sum += confidence

        self.max_gaze_confidence = max(
            self.max_gaze_confidence,
            confidence,
        )

        self.average_gaze_confidence = (
            self.gaze_confidence_sum
            / self.valid_gaze_sample_count
        )

        # ----------------------------------------------------
        # Head Pose
        # ----------------------------------------------------

        if (
            fused_gaze.head_yaw is not None
            and fused_gaze.head_pitch is not None
        ):
            self.head_yaw_sum += float(
                fused_gaze.head_yaw
            )

            self.head_pitch_sum += float(
                fused_gaze.head_pitch
            )

            self.head_yaw = (
                self.head_yaw_sum
                / self.valid_gaze_sample_count
            )

            self.head_pitch = (
                self.head_pitch_sum
                / self.valid_gaze_sample_count
            )

            if fused_gaze.head_roll is not None:
                self.head_roll_sum += float(
                    fused_gaze.head_roll
                )

                self.head_roll = (
                    self.head_roll_sum
                    / self.valid_gaze_sample_count
                )

    def update_attention(
        self,
        attention: AttentionResult,
    ) -> None:
        """
        Accumulate attention statistics.
        """

        score = float(
            np.clip(
                attention.attention_score,
                0.0,
                1.0,
            )
        )

        self.sample_count += 1

        self.attention_score_sum += score

        self.average_attention_score = (
            self.attention_score_sum
            / self.sample_count
        )

        self.max_attention_score = max(
            self.max_attention_score,
            score,
        )

        # ----------------------------------------------------
        # Screen position
        # ----------------------------------------------------

        if attention.screen_point is not None:
            x = float(
                attention.screen_point.x
            )

            y = float(
                attention.screen_point.y
            )

            if (
                self.start_screen_x is None
                or self.start_screen_y is None
            ):
                self.start_screen_x = x
                self.start_screen_y = y

            self.last_screen_x = x
            self.last_screen_y = y

            self.end_screen_x = x
            self.end_screen_y = y

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "track_id": self.track_id,
            "person_id": self.person_id,
            "start_time_ms": self.start_time_ms,
            "last_seen_time_ms": self.last_seen_time_ms,
            "end_time_ms": self.end_time_ms,
            "duration_ms": self.duration_ms,
            "duration_seconds": self.duration_seconds,
            "sample_count": self.sample_count,
            "valid_gaze_sample_count": (
                self.valid_gaze_sample_count
            ),

            "attention": {
                "max": self.max_attention_score,
                "average": self.average_attention_score,
            },

            "gaze": {
                "yaw": self.gaze_yaw,
                "pitch": self.gaze_pitch,
                "yaw_degrees": float(
                    np.degrees(
                        self.gaze_yaw
                    )
                ),
                "pitch_degrees": float(
                    np.degrees(
                        self.gaze_pitch
                    )
                ),
                "max_confidence": (
                    self.max_gaze_confidence
                ),
                "average_confidence": (
                    self.average_gaze_confidence
                ),
            },

            "head_pose": {
                "yaw": self.head_yaw,
                "pitch": self.head_pitch,
                "roll": self.head_roll,
            },

            "start_screen_point": (
                {
                    "x": self.start_screen_x,
                    "y": self.start_screen_y,
                }
                if (
                    self.start_screen_x is not None
                    and self.start_screen_y is not None
                )
                else None
            ),

            "end_screen_point": (
                {
                    "x": self.end_screen_x,
                    "y": self.end_screen_y,
                }
                if (
                    self.end_screen_x is not None
                    and self.end_screen_y is not None
                )
                else None
            ),

            "is_active": self.is_active,
        }


class SessionManager:
    """
    Converts frame-level attention results into continuous
    gaze/attention sessions.

    Each Track ID has an independent session state.

    Main responsibilities:
        - Start sessions
        - Continue sessions
        - Accumulate gaze/head-pose data
        - Handle temporary attention loss
        - Close sessions
        - Reject sessions shorter than the minimum duration
    """

    def __init__(
        self,
        min_session_duration_ms: int = 500,
        attention_timeout_ms: int = 1000,
        max_session_gap_ms: int = 3000,
    ) -> None:

        if min_session_duration_ms < 0:
            raise ValueError(
                "min_session_duration_ms cannot be negative."
            )

        if attention_timeout_ms < 0:
            raise ValueError(
                "attention_timeout_ms cannot be negative."
            )

        if max_session_gap_ms < attention_timeout_ms:
            raise ValueError(
                "max_session_gap_ms must be >= "
                "attention_timeout_ms."
            )

        self.min_session_duration_ms = int(
            min_session_duration_ms
        )

        self.attention_timeout_ms = int(
            attention_timeout_ms
        )

        self.max_session_gap_ms = int(
            max_session_gap_ms
        )

        self._active_sessions: dict[
            int,
            LookSession,
        ] = {}

        self._last_positive_time: dict[
            int,
            int,
        ] = {}

        self._next_session_id = 1

        self._completed_sessions: list[
            LookSession
        ] = []

    # ========================================================
    # Update
    # ========================================================

    def update(
        self,
        track_id: int,
        attention: AttentionResult,
        timestamp_ms: int,
        fused_gaze: Optional[FusedGaze] = None,
        person_id: Optional[str] = None,
    ) -> Optional[LookSession]:
        """
        Update the session state for one tracked person.

        Args:
            track_id:
                Temporary tracking ID.

            attention:
                Frame-level attention result.

            timestamp_ms:
                Current frame timestamp.

            fused_gaze:
                Current fused gaze result.

            person_id:
                Persistent recognition ID, when available.

        Returns:
            A completed LookSession if one ended.
            Otherwise None.
        """

        if track_id < 0:
            raise ValueError(
                "track_id cannot be negative."
            )

        if timestamp_ms < 0:
            raise ValueError(
                "timestamp_ms cannot be negative."
            )

        if attention is None:
            raise ValueError(
                "attention cannot be None."
            )

        if attention.is_looking_at_screen:
            return self._handle_looking(
                track_id=track_id,
                attention=attention,
                timestamp_ms=timestamp_ms,
                fused_gaze=fused_gaze,
                person_id=person_id,
            )

        return self._handle_not_looking(
            track_id=track_id,
            timestamp_ms=timestamp_ms,
        )

    # ========================================================
    # Looking
    # ========================================================

    def _handle_looking(
        self,
        track_id: int,
        attention: AttentionResult,
        timestamp_ms: int,
        fused_gaze: Optional[FusedGaze],
        person_id: Optional[str],
    ) -> Optional[LookSession]:
        """
        Handle a positive attention frame.
        """

        session = self._active_sessions.get(
            track_id
        )

        if session is None:
            session = self._start_session(
                track_id=track_id,
                person_id=person_id,
                attention=attention,
                fused_gaze=fused_gaze,
                timestamp_ms=timestamp_ms,
            )

            return None

        # A recognition result may become available after
        # the session has already started.
        if (
            session.person_id is None
            and person_id is not None
        ):
            session.person_id = person_id

        session.last_seen_time_ms = timestamp_ms

        session.update_attention(
            attention
        )

        session.update_gaze(
            fused_gaze
        )

        self._last_positive_time[
            track_id
        ] = timestamp_ms

        session.duration_ms = max(
            0,
            timestamp_ms
            - session.start_time_ms,
        )

        return None

    # ========================================================
    # Not Looking
    # ========================================================

    def _handle_not_looking(
        self,
        track_id: int,
        timestamp_ms: int,
    ) -> Optional[LookSession]:
        """
        Handle a negative attention frame.

        Short gaps do not immediately terminate the session.
        """

        session = self._active_sessions.get(
            track_id
        )

        if session is None:
            return None

        last_positive = self._last_positive_time.get(
            track_id,
            session.last_seen_time_ms,
        )

        gap_ms = (
            timestamp_ms
            - last_positive
        )

        if gap_ms <= self.attention_timeout_ms:
            session.last_seen_time_ms = timestamp_ms
            return None

        if gap_ms <= self.max_session_gap_ms:
            session.last_seen_time_ms = timestamp_ms
            return None

        return self._close_session(
            track_id=track_id,
            end_time_ms=last_positive,
        )

    # ========================================================
    # Start Session
    # ========================================================

    def _start_session(
        self,
        track_id: int,
        person_id: Optional[str],
        attention: AttentionResult,
        fused_gaze: Optional[FusedGaze],
        timestamp_ms: int,
    ) -> LookSession:
        """
        Start a new look session.
        """

        session = LookSession(
            session_id=self._next_session_id,
            track_id=track_id,
            person_id=person_id,
            start_time_ms=timestamp_ms,
            last_seen_time_ms=timestamp_ms,
            end_time_ms=None,
            duration_ms=0,
            sample_count=0,
            valid_gaze_sample_count=0,

            max_attention_score=0.0,
            attention_score_sum=0.0,
            average_attention_score=0.0,

            gaze_yaw_sum=0.0,
            gaze_pitch_sum=0.0,
            gaze_yaw=0.0,
            gaze_pitch=0.0,

            max_gaze_confidence=0.0,
            gaze_confidence_sum=0.0,
            average_gaze_confidence=0.0,

            head_yaw_sum=0.0,
            head_pitch_sum=0.0,
            head_roll_sum=0.0,

            head_yaw=None,
            head_pitch=None,
            head_roll=None,

            start_screen_x=None,
            start_screen_y=None,
            end_screen_x=None,
            end_screen_y=None,
            last_screen_x=None,
            last_screen_y=None,

            is_active=True,
        )

        session.update_attention(
            attention
        )

        session.update_gaze(
            fused_gaze
        )

        self._active_sessions[
            track_id
        ] = session

        self._last_positive_time[
            track_id
        ] = timestamp_ms

        self._next_session_id += 1

        return session

    # ========================================================
    # Close Session
    # ========================================================

    def _close_session(
        self,
        track_id: int,
        end_time_ms: Optional[int] = None,
    ) -> Optional[LookSession]:
        """
        Close an active session.

        Sessions below the minimum duration are discarded.
        """

        session = self._active_sessions.get(
            track_id
        )

        if session is None:
            return None

        if end_time_ms is None:
            end_time_ms = (
                session.last_seen_time_ms
            )

        end_time_ms = max(
            int(end_time_ms),
            session.start_time_ms,
        )

        session.end_time_ms = end_time_ms

        session.duration_ms = (
            end_time_ms
            - session.start_time_ms
        )

        session.is_active = False

        self._active_sessions.pop(
            track_id,
            None,
        )

        self._last_positive_time.pop(
            track_id,
            None,
        )

        if (
            session.duration_ms
            < self.min_session_duration_ms
        ):
            return None

        self._completed_sessions.append(
            session
        )

        return session

    # ========================================================
    # Force Close
    # ========================================================

    def close_track(
        self,
        track_id: int,
        timestamp_ms: Optional[int] = None,
    ) -> Optional[LookSession]:
        """
        Force-close a track's active session.

        Used when a track permanently disappears.
        """

        session = self._active_sessions.get(
            track_id
        )

        if session is None:
            return None

        if timestamp_ms is None:
            timestamp_ms = (
                session.last_seen_time_ms
            )

        return self._close_session(
            track_id=track_id,
            end_time_ms=timestamp_ms,
        )

    def close_all(
        self,
        timestamp_ms: Optional[int] = None,
    ) -> list[LookSession]:
        """
        Close all active sessions.

        Used when:
            - video ends
            - camera stops
            - application exits
        """

        track_ids = list(
            self._active_sessions.keys()
        )

        completed: list[LookSession] = []

        for track_id in track_ids:
            session = self.close_track(
                track_id=track_id,
                timestamp_ms=timestamp_ms,
            )

            if session is not None:
                completed.append(
                    session
                )

        return completed

    # ========================================================
    # Retrieval
    # ========================================================

    def get_active_session(
        self,
        track_id: int,
    ) -> Optional[LookSession]:
        return self._active_sessions.get(
            track_id
        )

    def get_active_sessions(
        self,
    ) -> list[LookSession]:
        return list(
            self._active_sessions.values()
        )

    def consume_completed_sessions(
        self,
    ) -> list[LookSession]:
        """
        Return completed sessions and clear the internal queue.
        """

        sessions = list(
            self._completed_sessions
        )

        self._completed_sessions.clear()

        return sessions

    @property
    def active_session_count(self) -> int:
        return len(
            self._active_sessions
        )

    @property
    def completed_session_count(self) -> int:
        return len(
            self._completed_sessions
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(self) -> None:
        """
        Clear all session state.
        """

        self._active_sessions.clear()
        self._last_positive_time.clear()
        self._completed_sessions.clear()

        self._next_session_id = 1