from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from src.storage.models import (
    Advertisement,
    AdvertisementStatistics,
    GazePoint,
    LookEvent,
    Person,
    PersonStatistics,
)


class Database:
    """
    SQLite persistence layer for Advertisement Gaze Analytics.

    Responsibilities:
        - Create database schema
        - Store persons
        - Store advertisements
        - Store look events
        - Store gaze points
        - Store aggregated statistics
        - Execute analytical queries

    The rest of the application should not access sqlite3 directly.
    """

    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = Path(
            database_path
        )

        self._connection: Optional[
            sqlite3.Connection
        ] = None

    # ========================================================
    # Connection
    # ========================================================

    def connect(self) -> None:
        """
        Open the SQLite database.
        """

        if self._connection is not None:
            return

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
        )

        connection.row_factory = sqlite3.Row

        # Recommended SQLite settings for this application.
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        connection.execute(
            "PRAGMA synchronous = NORMAL"
        )

        self._connection = connection

    def close(self) -> None:
        """
        Close the database connection.
        """

        if self._connection is not None:
            self._connection.close()

            self._connection = None

    def __enter__(self) -> "Database":
        self.connect()
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError(
                "Database is not connected. "
                "Call connect() first."
            )

        return self._connection

    # ========================================================
    # Schema
    # ========================================================

    def initialize(self) -> None:
        """
        Create all database tables and indexes.
        """

        connection = self.connection

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,

                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,

                session_count INTEGER NOT NULL DEFAULT 0,

                total_look_duration_ms INTEGER NOT NULL DEFAULT 0,

                metadata_json TEXT NOT NULL DEFAULT '{}'
            );


            CREATE TABLE IF NOT EXISTS ads (
                ad_id TEXT PRIMARY KEY,

                name TEXT NOT NULL,

                duration_ms INTEGER NOT NULL DEFAULT 0,

                metadata_json TEXT NOT NULL DEFAULT '{}'
            );


            CREATE TABLE IF NOT EXISTS analytics_sessions (
                session_id TEXT PRIMARY KEY,

                started_at TEXT NOT NULL,

                ended_at TEXT,

                source_type TEXT NOT NULL,

                source_name TEXT,

                total_frames INTEGER NOT NULL DEFAULT 0,

                total_viewers INTEGER NOT NULL DEFAULT 0,

                total_look_events INTEGER NOT NULL DEFAULT 0,

                total_look_duration_ms INTEGER NOT NULL DEFAULT 0,

                metadata_json TEXT NOT NULL DEFAULT '{}'
            );


            CREATE TABLE IF NOT EXISTS look_events (
                event_id INTEGER PRIMARY KEY,

                person_id TEXT,

                track_id INTEGER NOT NULL,

                ad_id TEXT,

                start_time_ms INTEGER NOT NULL,

                end_time_ms INTEGER NOT NULL,

                duration_ms INTEGER NOT NULL,

                gaze_yaw REAL,

                gaze_pitch REAL,

                head_yaw REAL,

                head_pitch REAL,

                head_roll REAL,

                gaze_confidence REAL NOT NULL DEFAULT 0.0,

                attention_score REAL NOT NULL DEFAULT 0.0,

                gaze_x REAL,

                gaze_y REAL,

                start_gaze_x REAL,

                start_gaze_y REAL,

                end_gaze_x REAL,

                end_gaze_y REAL,

                created_at TEXT NOT NULL,

                metadata_json TEXT NOT NULL DEFAULT '{}',

                FOREIGN KEY (
                    person_id
                )
                REFERENCES persons (
                    person_id
                )
                ON DELETE SET NULL,

                FOREIGN KEY (
                    ad_id
                )
                REFERENCES ads (
                    ad_id
                )
                ON DELETE SET NULL
            );


            CREATE TABLE IF NOT EXISTS gaze_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp_ms INTEGER NOT NULL,

                person_id TEXT,

                track_id INTEGER,

                ad_id TEXT,

                x REAL NOT NULL,

                y REAL NOT NULL,

                attention_score REAL NOT NULL DEFAULT 0.0,

                FOREIGN KEY (
                    person_id
                )
                REFERENCES persons (
                    person_id
                )
                ON DELETE SET NULL,

                FOREIGN KEY (
                    ad_id
                )
                REFERENCES ads (
                    ad_id
                )
                ON DELETE SET NULL
            );


            CREATE TABLE IF NOT EXISTS person_statistics (
                person_id TEXT PRIMARY KEY,

                look_count INTEGER NOT NULL DEFAULT 0,

                total_look_duration_ms INTEGER NOT NULL DEFAULT 0,

                average_look_duration_ms REAL NOT NULL DEFAULT 0.0,

                max_look_duration_ms INTEGER NOT NULL DEFAULT 0,

                average_attention_score REAL NOT NULL DEFAULT 0.0,

                last_look_at TEXT,

                FOREIGN KEY (
                    person_id
                )
                REFERENCES persons (
                    person_id
                )
                ON DELETE CASCADE
            );


            CREATE TABLE IF NOT EXISTS ad_statistics (
                ad_id TEXT PRIMARY KEY,

                viewer_count INTEGER NOT NULL DEFAULT 0,

                look_count INTEGER NOT NULL DEFAULT 0,

                total_look_duration_ms INTEGER NOT NULL DEFAULT 0,

                average_look_duration_ms REAL NOT NULL DEFAULT 0.0,

                average_attention_score REAL NOT NULL DEFAULT 0.0,

                max_attention_score REAL NOT NULL DEFAULT 0.0,

                FOREIGN KEY (
                    ad_id
                )
                REFERENCES ads (
                    ad_id
                )
                ON DELETE CASCADE
            );


            CREATE INDEX IF NOT EXISTS idx_look_events_person
                ON look_events(person_id);


            CREATE INDEX IF NOT EXISTS idx_look_events_ad
                ON look_events(ad_id);


            CREATE INDEX IF NOT EXISTS idx_look_events_track
                ON look_events(track_id);


            CREATE INDEX IF NOT EXISTS idx_look_events_start_time
                ON look_events(start_time_ms);


            CREATE INDEX IF NOT EXISTS idx_look_events_end_time
                ON look_events(end_time_ms);


            CREATE INDEX IF NOT EXISTS idx_gaze_points_person
                ON gaze_points(person_id);


            CREATE INDEX IF NOT EXISTS idx_gaze_points_ad
                ON gaze_points(ad_id);


            CREATE INDEX IF NOT EXISTS idx_gaze_points_timestamp
                ON gaze_points(timestamp_ms);
            """
        )

        connection.commit()

    # ========================================================
    # Transaction
    # ========================================================

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    # ========================================================
    # Helper Methods
    # ========================================================

    @staticmethod
    def _serialize_metadata(
        metadata: Optional[dict[str, Any]],
    ) -> str:
        return json.dumps(
            metadata or {},
            ensure_ascii=False,
        )

    @staticmethod
    def _deserialize_metadata(
        value: Optional[str],
    ) -> dict[str, Any]:
        if not value:
            return {}

        try:
            result = json.loads(value)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        return {}

    # ========================================================
    # Persons
    # ========================================================

    def upsert_person(
        self,
        person: Person,
    ) -> None:
        """
        Insert or update a person.
        """

        self.connection.execute(
            """
            INSERT INTO persons (
                person_id,
                first_seen_at,
                last_seen_at,
                session_count,
                total_look_duration_ms,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT(person_id)
            DO UPDATE SET
                last_seen_at = excluded.last_seen_at,
                session_count = excluded.session_count,
                total_look_duration_ms =
                    excluded.total_look_duration_ms,
                metadata_json = excluded.metadata_json
            """,
            (
                person.person_id,
                person.first_seen_at,
                person.last_seen_at,
                person.session_count,
                person.total_look_duration_ms,
                self._serialize_metadata(
                    person.metadata
                ),
            ),
        )

    def get_person(
        self,
        person_id: str,
    ) -> Optional[Person]:
        """
        Retrieve a person by ID.
        """

        row = self.connection.execute(
            """
            SELECT *
            FROM persons
            WHERE person_id = ?
            """,
            (person_id,),
        ).fetchone()

        if row is None:
            return None

        return Person(
            person_id=row["person_id"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            session_count=row["session_count"],
            total_look_duration_ms=(
                row["total_look_duration_ms"]
            ),
            metadata=self._deserialize_metadata(
                row["metadata_json"]
            ),
        )

    # ========================================================
    # Advertisements
    # ========================================================

    def upsert_advertisement(
        self,
        ad: Advertisement,
    ) -> None:
        """
        Insert or update an advertisement.
        """

        self.connection.execute(
            """
            INSERT INTO ads (
                ad_id,
                name,
                duration_ms,
                metadata_json
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(ad_id)
            DO UPDATE SET
                name = excluded.name,
                duration_ms = excluded.duration_ms,
                metadata_json = excluded.metadata_json
            """,
            (
                ad.ad_id,
                ad.name,
                ad.duration_ms,
                self._serialize_metadata(
                    ad.metadata
                ),
            ),
        )

    def get_advertisement(
        self,
        ad_id: str,
    ) -> Optional[Advertisement]:
        """
        Retrieve an advertisement.
        """

        row = self.connection.execute(
            """
            SELECT *
            FROM ads
            WHERE ad_id = ?
            """,
            (ad_id,),
        ).fetchone()

        if row is None:
            return None

        return Advertisement(
            ad_id=row["ad_id"],
            name=row["name"],
            duration_ms=row["duration_ms"],
            metadata=self._deserialize_metadata(
                row["metadata_json"]
            ),
        )

    # ========================================================
    # Analytics Sessions
    # ========================================================

    def save_analytics_session(
        self,
        session_id: str,
        started_at: str,
        ended_at: Optional[str],
        source_type: str,
        source_name: Optional[str],
        total_frames: int,
        total_viewers: int,
        total_look_events: int,
        total_look_duration_ms: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Save one complete application processing session.
        """

        self.connection.execute(
            """
            INSERT INTO analytics_sessions (
                session_id,
                started_at,
                ended_at,
                source_type,
                source_name,
                total_frames,
                total_viewers,
                total_look_events,
                total_look_duration_ms,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(session_id)
            DO UPDATE SET
                ended_at = excluded.ended_at,
                total_frames = excluded.total_frames,
                total_viewers = excluded.total_viewers,
                total_look_events =
                    excluded.total_look_events,
                total_look_duration_ms =
                    excluded.total_look_duration_ms,
                metadata_json =
                    excluded.metadata_json
            """,
            (
                session_id,
                started_at,
                ended_at,
                source_type,
                source_name,
                total_frames,
                total_viewers,
                total_look_events,
                total_look_duration_ms,
                self._serialize_metadata(
                    metadata
                ),
            ),
        )

    # ========================================================
    # Look Events
    # ========================================================

    def save_look_event(
        self,
        event: LookEvent,
    ) -> None:
        """
        Persist one completed look event.
        """

        gaze = event.gaze

        self.connection.execute(
            """
            INSERT INTO look_events (
                event_id,
                person_id,
                track_id,
                ad_id,
                start_time_ms,
                end_time_ms,
                duration_ms,
                gaze_yaw,
                gaze_pitch,
                head_yaw,
                head_pitch,
                head_roll,
                gaze_confidence,
                attention_score,
                gaze_x,
                gaze_y,
                start_gaze_x,
                start_gaze_y,
                end_gaze_x,
                end_gaze_y,
                created_at,
                metadata_json
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                event.event_id,
                event.person_id,
                event.track_id,
                event.ad_id,
                event.start_time_ms,
                event.end_time_ms,
                event.duration_ms,
                gaze.yaw if gaze else None,
                gaze.pitch if gaze else None,
                gaze.head_yaw if gaze else None,
                gaze.head_pitch if gaze else None,
                gaze.head_roll if gaze else None,
                gaze.gaze_confidence if gaze else 0.0,
                event.average_attention_score,
                gaze.gaze_x if gaze else None,
                gaze.gaze_y if gaze else None,
                event.start_gaze_x,
                event.start_gaze_y,
                event.end_gaze_x,
                event.end_gaze_y,
                event.created_at,
                self._serialize_metadata(
                    event.metadata
                ),
            ),
        )

    def get_look_event(
        self,
        event_id: int,
    ) -> Optional[LookEvent]:
        """
        Retrieve a persisted look event.
        """

        row = self.connection.execute(
            """
            SELECT *
            FROM look_events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_look_event(
            row
        )

    def _row_to_look_event(
        self,
        row: sqlite3.Row,
    ) -> LookEvent:
        """
        Convert SQLite row into LookEvent.
        """

        from src.storage.models import (
            GazeSnapshot,
        )

        gaze_exists = (
            row["gaze_yaw"] is not None
            or row["gaze_pitch"] is not None
        )

        gaze = None

        if gaze_exists:
            gaze = GazeSnapshot(
                yaw=float(
                    row["gaze_yaw"] or 0.0
                ),
                pitch=float(
                    row["gaze_pitch"] or 0.0
                ),
                head_yaw=(
                    float(row["head_yaw"])
                    if row["head_yaw"] is not None
                    else None
                ),
                head_pitch=(
                    float(row["head_pitch"])
                    if row["head_pitch"] is not None
                    else None
                ),
                head_roll=(
                    float(row["head_roll"])
                    if row["head_roll"] is not None
                    else None
                ),
                gaze_confidence=float(
                    row["gaze_confidence"]
                ),
                attention_score=float(
                    row["attention_score"]
                ),
                gaze_x=(
                    float(row["gaze_x"])
                    if row["gaze_x"] is not None
                    else None
                ),
                gaze_y=(
                    float(row["gaze_y"])
                    if row["gaze_y"] is not None
                    else None
                ),
            )

        return LookEvent(
            event_id=row["event_id"],
            person_id=row["person_id"],
            track_id=row["track_id"],
            ad_id=row["ad_id"],
            start_time_ms=row["start_time_ms"],
            end_time_ms=row["end_time_ms"],
            duration_ms=row["duration_ms"],
            gaze=gaze,
            start_gaze_x=(
                float(row["start_gaze_x"])
                if row["start_gaze_x"] is not None
                else None
            ),
            start_gaze_y=(
                float(row["start_gaze_y"])
                if row["start_gaze_y"] is not None
                else None
            ),
            end_gaze_x=(
                float(row["end_gaze_x"])
                if row["end_gaze_x"] is not None
                else None
            ),
            end_gaze_y=(
                float(row["end_gaze_y"])
                if row["end_gaze_y"] is not None
                else None
            ),
            average_attention_score=float(
                row["attention_score"]
            ),
            max_attention_score=float(
                row["attention_score"]
            ),
            created_at=row["created_at"],
            metadata=self._deserialize_metadata(
                row["metadata_json"]
            ),
        )

    # ========================================================
    # Gaze Points
    # ========================================================

    def save_gaze_point(
        self,
        point: GazePoint,
    ) -> None:
        """
        Save one gaze sample.

        Gaze points are intentionally separate from look events.
        This allows heatmap generation without reconstructing
        the original video.
        """

        self.connection.execute(
            """
            INSERT INTO gaze_points (
                timestamp_ms,
                person_id,
                track_id,
                ad_id,
                x,
                y,
                attention_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                point.timestamp_ms,
                point.person_id,
                point.track_id,
                point.ad_id,
                point.x,
                point.y,
                point.attention_score,
            ),
        )

    # ========================================================
    # Statistics
    # ========================================================

    def upsert_person_statistics(
        self,
        statistics: PersonStatistics,
    ) -> None:
        """
        Persist aggregated person statistics.
        """

        self.connection.execute(
            """
            INSERT INTO person_statistics (
                person_id,
                look_count,
                total_look_duration_ms,
                average_look_duration_ms,
                max_look_duration_ms,
                average_attention_score,
                last_look_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(person_id)
            DO UPDATE SET
                look_count = excluded.look_count,
                total_look_duration_ms =
                    excluded.total_look_duration_ms,
                average_look_duration_ms =
                    excluded.average_look_duration_ms,
                max_look_duration_ms =
                    excluded.max_look_duration_ms,
                average_attention_score =
                    excluded.average_attention_score,
                last_look_at =
                    excluded.last_look_at
            """,
            (
                statistics.person_id,
                statistics.look_count,
                statistics.total_look_duration_ms,
                statistics.average_look_duration_ms,
                statistics.max_look_duration_ms,
                statistics.average_attention_score,
                statistics.last_look_at,
            ),
        )

    def upsert_ad_statistics(
        self,
        statistics: AdvertisementStatistics,
    ) -> None:
        """
        Persist aggregated advertisement statistics.
        """

        self.connection.execute(
            """
            INSERT INTO ad_statistics (
                ad_id,
                viewer_count,
                look_count,
                total_look_duration_ms,
                average_look_duration_ms,
                average_attention_score,
                max_attention_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(ad_id)
            DO UPDATE SET
                viewer_count =
                    excluded.viewer_count,
                look_count =
                    excluded.look_count,
                total_look_duration_ms =
                    excluded.total_look_duration_ms,
                average_look_duration_ms =
                    excluded.average_look_duration_ms,
                average_attention_score =
                    excluded.average_attention_score,
                max_attention_score =
                    excluded.max_attention_score
            """,
            (
                statistics.ad_id,
                statistics.viewer_count,
                statistics.look_count,
                statistics.total_look_duration_ms,
                statistics.average_look_duration_ms,
                statistics.average_attention_score,
                statistics.max_attention_score,
            ),
        )

    # ========================================================
    # Queries
    # ========================================================

    def get_person_look_events(
        self,
        person_id: str,
    ) -> list[LookEvent]:
        """
        Get all look events belonging to one person.
        """

        rows = self.connection.execute(
            """
            SELECT *
            FROM look_events
            WHERE person_id = ?
            ORDER BY start_time_ms ASC
            """,
            (person_id,),
        ).fetchall()

        return [
            self._row_to_look_event(row)
            for row in rows
        ]

    def get_ad_look_events(
        self,
        ad_id: str,
    ) -> list[LookEvent]:
        """
        Get all look events for one advertisement.
        """

        rows = self.connection.execute(
            """
            SELECT *
            FROM look_events
            WHERE ad_id = ?
            ORDER BY start_time_ms ASC
            """,
            (ad_id,),
        ).fetchall()

        return [
            self._row_to_look_event(row)
            for row in rows
        ]

    def get_gaze_points(
        self,
        ad_id: Optional[str] = None,
        person_id: Optional[str] = None,
    ) -> list[GazePoint]:
        """
        Retrieve gaze points for heatmap generation.
        """

        query = """
            SELECT
                timestamp_ms,
                person_id,
                track_id,
                ad_id,
                x,
                y,
                attention_score
            FROM gaze_points
            WHERE 1 = 1
        """

        parameters: list[Any] = []

        if ad_id is not None:
            query += """
                AND ad_id = ?
            """

            parameters.append(ad_id)

        if person_id is not None:
            query += """
                AND person_id = ?
            """

            parameters.append(person_id)

        query += """
            ORDER BY timestamp_ms ASC
        """

        rows = self.connection.execute(
            query,
            parameters,
        ).fetchall()

        return [
            GazePoint(
                timestamp_ms=row["timestamp_ms"],
                person_id=row["person_id"],
                track_id=row["track_id"],
                ad_id=row["ad_id"],
                x=float(row["x"]),
                y=float(row["y"]),
                attention_score=float(
                    row["attention_score"]
                ),
            )
            for row in rows
        ]

    # ========================================================
    # Maintenance
    # ========================================================

    def delete_person(
        self,
        person_id: str,
    ) -> None:
        """
        Delete one person and associated dependent records.

        look_events and gaze_points use SET NULL for person_id,
        while person_statistics is removed through CASCADE.
        """

        self.connection.execute(
            """
            DELETE FROM persons
            WHERE person_id = ?
            """,
            (person_id,),
        )

    def delete_advertisement(
        self,
        ad_id: str,
    ) -> None:
        """
        Delete one advertisement.
        """

        self.connection.execute(
            """
            DELETE FROM ads
            WHERE ad_id = ?
            """,
            (ad_id,),
        )

    def vacuum(self) -> None:
        """
        Rebuild the SQLite database file and reclaim unused space.
        """

        self.connection.execute(
            "VACUUM"
        )

    def clear_all_data(
        self,
    ) -> None:
        """
        Delete analytical data while keeping table definitions.
        """

        self.connection.executescript(
            """
            DELETE FROM gaze_points;
            DELETE FROM look_events;
            DELETE FROM ad_statistics;
            DELETE FROM person_statistics;
            DELETE FROM analytics_sessions;
            DELETE FROM persons;
            DELETE FROM ads;
            """
        )

        self.commit()