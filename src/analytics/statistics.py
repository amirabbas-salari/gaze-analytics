from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

from src.attention.session_manager import LookSession
from src.storage.database import Database
from src.storage.models import (
    AdvertisementStatistics,
    GazeSnapshot,
    LookEvent,
    PersonStatistics,
)


class StatisticsEngine:
    """
    Converts completed LookSession objects into persistent
    LookEvent objects and maintains aggregated statistics.
    """

    def __init__(
        self,
        database: Database,
    ) -> None:
        self.database = database

        self._next_event_id = 1

        self._person_statistics: dict[
            str,
            PersonStatistics,
        ] = {}

        self._ad_statistics: dict[
            str,
            AdvertisementStatistics,
        ] = {}

        self._known_persons_per_ad: dict[
            str,
            set[str],
        ] = defaultdict(set)

    # ========================================================
    # Initialization
    # ========================================================

    def initialize(self) -> None:
        """
        Initialize database and recover the next event ID.
        """

        self.database.connect()
        self.database.initialize()

        self._recover_next_event_id()
        self.rebuild_statistics_cache()

    def _recover_next_event_id(self) -> None:
        """
        Recover the next event ID from the database.
        """

        row = self.database.connection.execute(
            """
            SELECT COALESCE(
                MAX(event_id),
                0
            ) AS max_event_id
            FROM look_events
            """
        ).fetchone()

        self._next_event_id = (
            int(row["max_event_id"]) + 1
        )

    # ========================================================
    # Main Processing
    # ========================================================

    def process_session(
        self,
        look_session: LookSession,
        person_id: Optional[str] = None,
        ad_id: Optional[str] = None,
    ) -> Optional[LookEvent]:
        """
        Convert one completed LookSession to a LookEvent.
        """

        if look_session is None:
            raise ValueError(
                "look_session cannot be None."
            )

        if look_session.is_active:
            raise ValueError(
                "Only completed sessions can be processed."
            )

        if look_session.duration_ms <= 0:
            return None

        resolved_person_id = (
            person_id
            if person_id is not None
            else look_session.person_id
        )

        if (
            resolved_person_id is not None
            and not resolved_person_id
        ):
            resolved_person_id = None

        event = self._session_to_event(
            look_session=look_session,
            person_id=resolved_person_id,
            ad_id=ad_id,
        )

        if event is None:
            return None

        try:
            self._persist_event(event)
            self._update_statistics(event)

        except Exception:
            self.database.rollback()
            raise

        return event

    def process_sessions(
        self,
        sessions: list[LookSession],
        person_id_resolver: Optional[
            Callable[[LookSession], Optional[str]]
        ] = None,
        ad_id_resolver: Optional[
            Callable[[LookSession], Optional[str]]
        ] = None,
    ) -> list[LookEvent]:
        """
        Process a collection of completed sessions.
        """

        events: list[LookEvent] = []

        for session in sessions:
            person_id = (
                person_id_resolver(session)
                if person_id_resolver is not None
                else session.person_id
            )

            ad_id = (
                ad_id_resolver(session)
                if ad_id_resolver is not None
                else None
            )

            event = self.process_session(
                look_session=session,
                person_id=person_id,
                ad_id=ad_id,
            )

            if event is not None:
                events.append(event)

        return events

    # ========================================================
    # Session -> Event
    # ========================================================

    def _session_to_event(
        self,
        look_session: LookSession,
        person_id: Optional[str],
        ad_id: Optional[str],
    ) -> Optional[LookEvent]:
        """
        Convert a completed LookSession to a persistent event.
        """

        start_time = int(
            look_session.start_time_ms
        )

        end_time = int(
            look_session.end_time_ms
            if look_session.end_time_ms is not None
            else look_session.last_seen_time_ms
        )

        duration = max(
            0,
            end_time - start_time,
        )

        if duration <= 0:
            return None

        # The Session stores angular information in radians.
        gaze = GazeSnapshot(
            yaw=float(
                look_session.gaze_yaw
            ),
            pitch=float(
                look_session.gaze_pitch
            ),
            head_yaw=(
                float(
                    self._degrees_to_radians(
                        look_session.head_yaw
                    )
                )
                if look_session.head_yaw is not None
                else None
            ),
            head_pitch=(
                float(
                    self._degrees_to_radians(
                        look_session.head_pitch
                    )
                )
                if look_session.head_pitch is not None
                else None
            ),
            head_roll=(
                float(
                    self._degrees_to_radians(
                        look_session.head_roll
                    )
                )
                if look_session.head_roll is not None
                else None
            ),
            gaze_confidence=float(
                look_session.average_gaze_confidence
            ),
            attention_score=float(
                look_session.average_attention_score
            ),
            gaze_x=look_session.last_screen_x,
            gaze_y=look_session.last_screen_y,
        )

        event = LookEvent(
            event_id=self._next_event_id,
            person_id=person_id,
            track_id=look_session.track_id,
            start_time_ms=start_time,
            end_time_ms=end_time,
            duration_ms=duration,
            ad_id=ad_id,
            gaze=gaze,
            start_gaze_x=look_session.start_screen_x,
            start_gaze_y=look_session.start_screen_y,
            end_gaze_x=look_session.end_screen_x,
            end_gaze_y=look_session.end_screen_y,
            average_attention_score=float(
                look_session.average_attention_score
            ),
            max_attention_score=float(
                look_session.max_attention_score
            ),
            metadata={
                "session_id": look_session.session_id,
                "sample_count": look_session.sample_count,
                "valid_gaze_sample_count": (
                    look_session.valid_gaze_sample_count
                ),
                "average_gaze_confidence": (
                    look_session.average_gaze_confidence
                ),
                "max_gaze_confidence": (
                    look_session.max_gaze_confidence
                ),
            },
        )

        self._next_event_id += 1

        return event

    @staticmethod
    def _degrees_to_radians(
        degrees: float,
    ) -> float:
        """
        Convert degrees to radians.

        HeadPose internally stores degrees, while GazeSnapshot
        stores angles in radians.
        """

        import numpy as np

        return float(
            np.radians(degrees)
        )

    # ========================================================
    # Persistence
    # ========================================================

    def _persist_event(
        self,
        event: LookEvent,
    ) -> None:
        """
        Persist the event.
        """

        self.database.save_look_event(
            event
        )

        self.database.commit()

    # ========================================================
    # Statistics
    # ========================================================

    def _update_statistics(
        self,
        event: LookEvent,
    ) -> None:
        """
        Update all relevant aggregation caches and persist them.
        """

        if event.person_id is not None:
            person_stats = (
                self._get_or_create_person_statistics(
                    event.person_id
                )
            )

            person_stats.update_from_event(
                event
            )

            self.database.upsert_person_statistics(
                person_stats
            )

        if event.ad_id is not None:
            ad_stats = (
                self._get_or_create_ad_statistics(
                    event.ad_id
                )
            )

            if event.person_id is not None:
                self._known_persons_per_ad[
                    event.ad_id
                ].add(
                    event.person_id
                )

            ad_stats.viewer_count = len(
                self._known_persons_per_ad[
                    event.ad_id
                ]
            )

            ad_stats.update_from_event(
                event
            )

            self.database.upsert_ad_statistics(
                ad_stats
            )

        self.database.commit()

    # ========================================================
    # Person Statistics
    # ========================================================

    def _get_or_create_person_statistics(
        self,
        person_id: str,
    ) -> PersonStatistics:
        if person_id not in self._person_statistics:
            self._person_statistics[
                person_id
            ] = PersonStatistics(
                person_id=person_id
            )

        return self._person_statistics[
            person_id
        ]

    def get_person_statistics(
        self,
        person_id: str,
    ) -> PersonStatistics:
        """
        Get person statistics.

        Rebuild from persisted events if not already cached.
        """

        cached = self._person_statistics.get(
            person_id
        )

        if cached is not None:
            return cached

        events = (
            self.database.get_person_look_events(
                person_id
            )
        )

        statistics = PersonStatistics(
            person_id=person_id
        )

        for event in events:
            statistics.update_from_event(
                event
            )

        self._person_statistics[
            person_id
        ] = statistics

        return statistics

    # ========================================================
    # Advertisement Statistics
    # ========================================================

    def _get_or_create_ad_statistics(
        self,
        ad_id: str,
    ) -> AdvertisementStatistics:
        cached = self._ad_statistics.get(
            ad_id
        )

        if cached is not None:
            return cached

        statistics = AdvertisementStatistics(
            ad_id=ad_id
        )

        self._ad_statistics[
            ad_id
        ] = statistics

        self._rebuild_ad_cache(
            ad_id
        )

        return self._ad_statistics[
            ad_id
        ]

    def _rebuild_ad_cache(
        self,
        ad_id: str,
    ) -> None:
        """
        Reconstruct advertisement statistics from stored events.
        """

        events = (
            self.database.get_ad_look_events(
                ad_id
            )
        )

        statistics = AdvertisementStatistics(
            ad_id=ad_id
        )

        unique_people: set[str] = set()

        for event in events:
            statistics.update_from_event(
                event
            )

            if event.person_id is not None:
                unique_people.add(
                    event.person_id
                )

        statistics.viewer_count = len(
            unique_people
        )

        self._known_persons_per_ad[
            ad_id
        ] = unique_people

        self._ad_statistics[
            ad_id
        ] = statistics

    def get_ad_statistics(
        self,
        ad_id: str,
    ) -> AdvertisementStatistics:
        """
        Get advertisement statistics.
        """

        if ad_id not in self._ad_statistics:
            self._rebuild_ad_cache(
                ad_id
            )

        return self._ad_statistics[
            ad_id
        ]

    # ========================================================
    # Gaze Analytics
    # ========================================================

    def get_average_gaze_for_person(
        self,
        person_id: str,
    ) -> dict:
        """
        Calculate average gaze direction for one person.
        """

        events = (
            self.database.get_person_look_events(
                person_id
            )
        )

        if not events:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
            }

        yaw_values = []
        pitch_values = []
        confidence_values = []

        for event in events:
            if event.gaze is None:
                continue

            yaw_values.append(
                event.gaze.yaw
            )

            pitch_values.append(
                event.gaze.pitch
            )

            confidence_values.append(
                event.gaze.gaze_confidence
            )

        if not yaw_values:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
            }

        import numpy as np

        return {
            "yaw": float(
                np.mean(yaw_values)
            ),
            "pitch": float(
                np.mean(pitch_values)
            ),
            "confidence": float(
                np.mean(confidence_values)
            ),
            "sample_count": len(
                yaw_values
            ),
        }

    def get_average_gaze_for_ad(
        self,
        ad_id: str,
    ) -> dict:
        """
        Calculate average gaze direction for one advertisement.
        """

        events = (
            self.database.get_ad_look_events(
                ad_id
            )
        )

        if not events:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
            }

        yaw_values = []
        pitch_values = []
        confidence_values = []

        for event in events:
            if event.gaze is None:
                continue

            yaw_values.append(
                event.gaze.yaw
            )

            pitch_values.append(
                event.gaze.pitch
            )

            confidence_values.append(
                event.gaze.gaze_confidence
            )

        if not yaw_values:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
            }

        import numpy as np

        return {
            "yaw": float(
                np.mean(yaw_values)
            ),
            "pitch": float(
                np.mean(pitch_values)
            ),
            "confidence": float(
                np.mean(confidence_values)
            ),
            "sample_count": len(
                yaw_values
            ),
        }

    # ========================================================
    # Global KPIs
    # ========================================================

    def get_global_statistics(self) -> dict:
        """
        Return high-level project KPIs.
        """

        row = self.database.connection.execute(
            """
            SELECT
                COUNT(*) AS look_count,
                COALESCE(
                    COUNT(
                        DISTINCT person_id
                    ),
                    0
                ) AS unique_viewers,
                COALESCE(
                    SUM(duration_ms),
                    0
                ) AS total_duration_ms,
                COALESCE(
                    AVG(duration_ms),
                    0
                ) AS average_duration_ms,
                COALESCE(
                    AVG(attention_score),
                    0
                ) AS average_attention_score
            FROM look_events
            """
        ).fetchone()

        ad_row = self.database.connection.execute(
            """
            SELECT COUNT(*) AS ad_count
            FROM ads
            """
        ).fetchone()

        return {
            "unique_viewers": int(
                row["unique_viewers"]
            ),
            "advertisement_count": int(
                ad_row["ad_count"]
            ),
            "look_count": int(
                row["look_count"]
            ),
            "total_look_duration_ms": int(
                row["total_duration_ms"]
            ),
            "average_look_duration_ms": float(
                row["average_duration_ms"]
            ),
            "average_attention_score": float(
                row["average_attention_score"]
            ),
        }

    # ========================================================
    # Ranking
    # ========================================================

    def get_top_advertisements(
        self,
        limit: int = 10,
    ) -> list[dict]:
        """
        Rank advertisements by total attention duration.
        """

        if limit <= 0:
            raise ValueError(
                "limit must be positive."
            )

        rows = self.database.connection.execute(
            """
            SELECT
                ads.ad_id,
                ads.name,
                COALESCE(
                    ad_statistics.viewer_count,
                    0
                ) AS viewer_count,
                COALESCE(
                    ad_statistics.look_count,
                    0
                ) AS look_count,
                COALESCE(
                    ad_statistics.total_look_duration_ms,
                    0
                ) AS total_look_duration_ms,
                COALESCE(
                    ad_statistics.average_attention_score,
                    0
                ) AS average_attention_score
            FROM ads
            LEFT JOIN ad_statistics
                ON ads.ad_id = ad_statistics.ad_id
            ORDER BY
                total_look_duration_ms DESC,
                average_attention_score DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "ad_id": row["ad_id"],
                "name": row["name"],
                "viewer_count": int(
                    row["viewer_count"]
                ),
                "look_count": int(
                    row["look_count"]
                ),
                "total_look_duration_ms": int(
                    row["total_look_duration_ms"]
                ),
                "average_attention_score": float(
                    row["average_attention_score"]
                ),
            }
            for row in rows
        ]

    def get_top_people(
        self,
        limit: int = 10,
    ) -> list[dict]:
        """
        Rank people by total gaze/attention duration.
        """

        if limit <= 0:
            raise ValueError(
                "limit must be positive."
            )

        rows = self.database.connection.execute(
            """
            SELECT
                person_id,
                look_count,
                total_look_duration_ms,
                average_look_duration_ms,
                average_attention_score
            FROM person_statistics
            ORDER BY
                total_look_duration_ms DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "person_id": row["person_id"],
                "look_count": int(
                    row["look_count"]
                ),
                "total_look_duration_ms": int(
                    row["total_look_duration_ms"]
                ),
                "average_look_duration_ms": float(
                    row["average_look_duration_ms"]
                ),
                "average_attention_score": float(
                    row["average_attention_score"]
                ),
            }
            for row in rows
        ]

    # ========================================================
    # Cache
    # ========================================================

    def clear_statistics_cache(self) -> None:
        """
        Clear in-memory caches without changing the database.
        """

        self._person_statistics.clear()
        self._ad_statistics.clear()
        self._known_persons_per_ad.clear()

    def rebuild_statistics_cache(self) -> None:
        """
        Rebuild all statistical caches from SQLite.
        """

        self.clear_statistics_cache()

        person_rows = self.database.connection.execute(
            """
            SELECT person_id
            FROM person_statistics
            """
        ).fetchall()

        for row in person_rows:
            self.get_person_statistics(
                row["person_id"]
            )

        ad_rows = self.database.connection.execute(
            """
            SELECT ad_id
            FROM ad_statistics
            """
        ).fetchall()

        for row in ad_rows:
            self.get_ad_statistics(
                row["ad_id"]
            )