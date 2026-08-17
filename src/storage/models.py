from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def utc_now_iso() -> str:
    """
    Return current UTC time in ISO-8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Person:
    """
    Persistent identity of a viewer.
    """

    person_id: str

    first_seen_at: str = field(
        default_factory=utc_now_iso
    )

    last_seen_at: str = field(
        default_factory=utc_now_iso
    )

    session_count: int = 0

    total_look_duration_ms: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "session_count": self.session_count,
            "total_look_duration_ms": (
                self.total_look_duration_ms
            ),
            "metadata": self.metadata,
        }


@dataclass
class Advertisement:
    """
    Advertisement metadata.
    """

    ad_id: str
    name: str

    duration_ms: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "ad_id": self.ad_id,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class GazeSnapshot:
    """
    Gaze information captured during one session/event.
    """

    yaw: float
    pitch: float

    head_yaw: Optional[float] = None
    head_pitch: Optional[float] = None
    head_roll: Optional[float] = None

    gaze_confidence: float = 0.0
    attention_score: float = 0.0

    gaze_x: Optional[float] = None
    gaze_y: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "head_yaw": self.head_yaw,
            "head_pitch": self.head_pitch,
            "head_roll": self.head_roll,
            "gaze_confidence": self.gaze_confidence,
            "attention_score": self.attention_score,
            "gaze_x": self.gaze_x,
            "gaze_y": self.gaze_y,
        }


@dataclass
class LookEvent:
    """
    A completed viewer attention event.

    This is the central analytical event in the system.
    """

    event_id: int

    person_id: Optional[str]

    track_id: int

    start_time_ms: int
    end_time_ms: int

    duration_ms: int

    ad_id: Optional[str] = None

    gaze: Optional[GazeSnapshot] = None

    start_gaze_x: Optional[float] = None
    start_gaze_y: Optional[float] = None

    end_gaze_x: Optional[float] = None
    end_gaze_y: Optional[float] = None

    average_attention_score: float = 0.0
    max_attention_score: float = 0.0

    created_at: str = field(
        default_factory=utc_now_iso
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "person_id": self.person_id,
            "track_id": self.track_id,
            "ad_id": self.ad_id,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "duration_ms": self.duration_ms,
            "duration_seconds": self.duration_seconds,
            "gaze": (
                self.gaze.to_dict()
                if self.gaze is not None
                else None
            ),
            "start_gaze_point": (
                {
                    "x": self.start_gaze_x,
                    "y": self.start_gaze_y,
                }
                if (
                    self.start_gaze_x is not None
                    and self.start_gaze_y is not None
                )
                else None
            ),
            "end_gaze_point": (
                {
                    "x": self.end_gaze_x,
                    "y": self.end_gaze_y,
                }
                if (
                    self.end_gaze_x is not None
                    and self.end_gaze_y is not None
                )
                else None
            ),
            "average_attention_score": (
                self.average_attention_score
            ),
            "max_attention_score": (
                self.max_attention_score
            ),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class PersonStatistics:
    """
    Aggregated statistics for one person.
    """

    person_id: str

    look_count: int = 0

    total_look_duration_ms: int = 0

    average_look_duration_ms: float = 0.0

    max_look_duration_ms: int = 0

    average_attention_score: float = 0.0

    last_look_at: Optional[str] = None

    def update_from_event(
        self,
        event: LookEvent,
    ) -> None:
        """
        Update statistics using one completed event.
        """

        previous_count = self.look_count

        self.look_count += 1

        self.total_look_duration_ms += (
            event.duration_ms
        )

        self.max_look_duration_ms = max(
            self.max_look_duration_ms,
            event.duration_ms,
        )

        if self.look_count > 0:
            self.average_look_duration_ms = (
                self.total_look_duration_ms
                / self.look_count
            )

        current_average = (
            self.average_attention_score
        )

        event_score = (
            event.average_attention_score
        )

        if previous_count == 0:
            self.average_attention_score = (
                event_score
            )
        else:
            self.average_attention_score = (
                (
                    current_average * previous_count
                )
                + event_score
            ) / self.look_count

        self.last_look_at = (
            event.created_at
        )

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "look_count": self.look_count,
            "total_look_duration_ms": (
                self.total_look_duration_ms
            ),
            "average_look_duration_ms": (
                self.average_look_duration_ms
            ),
            "max_look_duration_ms": (
                self.max_look_duration_ms
            ),
            "average_attention_score": (
                self.average_attention_score
            ),
            "last_look_at": self.last_look_at,
        }


@dataclass
class AdvertisementStatistics:
    """
    Aggregated statistics for one advertisement.
    """

    ad_id: str

    viewer_count: int = 0

    look_count: int = 0

    total_look_duration_ms: int = 0

    average_look_duration_ms: float = 0.0

    average_attention_score: float = 0.0

    max_attention_score: float = 0.0

    def update_from_event(
        self,
        event: LookEvent,
    ) -> None:
        """
        Update advertisement statistics from a look event.
        """

        previous_count = self.look_count

        self.look_count += 1

        self.total_look_duration_ms += (
            event.duration_ms
        )

        if self.look_count > 0:
            self.average_look_duration_ms = (
                self.total_look_duration_ms
                / self.look_count
            )

        current_average = (
            self.average_attention_score
        )

        self.average_attention_score = (
            (
                current_average * previous_count
            )
            + event.average_attention_score
        ) / self.look_count

        self.max_attention_score = max(
            self.max_attention_score,
            event.max_attention_score,
        )

    def to_dict(self) -> dict:
        return {
            "ad_id": self.ad_id,
            "viewer_count": self.viewer_count,
            "look_count": self.look_count,
            "total_look_duration_ms": (
                self.total_look_duration_ms
            ),
            "average_look_duration_ms": (
                self.average_look_duration_ms
            ),
            "average_attention_score": (
                self.average_attention_score
            ),
            "max_attention_score": (
                self.max_attention_score
            ),
        }


@dataclass
class GazePoint:
    """
    A point on the normalized advertisement/screen surface.
    """

    x: float
    y: float

    timestamp_ms: int

    person_id: Optional[str] = None

    track_id: Optional[int] = None

    ad_id: Optional[str] = None

    attention_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "timestamp_ms": self.timestamp_ms,
            "person_id": self.person_id,
            "track_id": self.track_id,
            "ad_id": self.ad_id,
            "attention_score": self.attention_score,
        }


@dataclass
class AnalyticsSession:
    """
    Describes one complete processing session of the application.

    This is different from LookSession.

    LookSession:
        one continuous period of looking.

    AnalyticsSession:
        one complete camera/video processing run.
    """

    session_id: str

    started_at: str

    ended_at: Optional[str] = None

    source_type: str = "camera"

    source_name: Optional[str] = None

    total_frames: int = 0

    total_viewers: int = 0

    total_look_events: int = 0

    total_look_duration_ms: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "total_frames": self.total_frames,
            "total_viewers": self.total_viewers,
            "total_look_events": (
                self.total_look_events
            ),
            "total_look_duration_ms": (
                self.total_look_duration_ms
            ),
            "metadata": self.metadata,
        }