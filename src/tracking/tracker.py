from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.vision.face_utils import BoundingBox


@dataclass
class Detection:
    """
    One face detection for the current frame.

    track-specific information is intentionally not stored here.
    Detection represents only the observation coming from the
    vision layer.
    """

    bounding_box: BoundingBox

    confidence: float = 1.0

    face_index: int = 0


@dataclass
class Track:
    """
    Persistent track representing one physical face across frames.

    Important:
        track_id is NOT a Person ID.

    track_id:
        Temporary identity used by the tracker.

    person_id:
        Optional persistent identity assigned later by the
        face recognition layer.
    """

    track_id: int

    bounding_box: BoundingBox

    confidence: float

    person_id: Optional[str] = None

    age: int = 1
    hits: int = 1
    time_since_update: int = 0

    is_confirmed: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return (
            (
                self.bounding_box.x1
                + self.bounding_box.x2
            )
            / 2.0,
            (
                self.bounding_box.y1
                + self.bounding_box.y2
            )
            / 2.0,
        )

    @property
    def width(self) -> int:
        return self.bounding_box.width

    @property
    def height(self) -> int:
        return self.bounding_box.height

    @property
    def is_lost(self) -> bool:
        return self.time_since_update > 0


@dataclass
class TrackMatch:
    """
    Result of matching a detection to an existing track.
    """

    track_id: int
    detection_index: int
    iou: float


class IoUTracker:
    """
    Lightweight multi-face tracker based on IoU matching.

    Responsibilities:
        - Create tracks for new detections
        - Match detections to existing tracks
        - Keep Track IDs stable across consecutive frames
        - Temporarily retain missing tracks
        - Remove stale tracks
        - Keep optional Person IDs separate from Track IDs

    This class is intentionally independent of Face Recognition.

    Pipeline:

        Detection
            ↓
        IoUTracker
            ↓
        Track ID
            ↓
        Recognition layer
            ↓
        Person ID

    Later, this implementation can be replaced by ByteTrack or
    another MOT algorithm without changing the rest of the project.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 15,
        min_hits: int = 2,
        max_tracks: int = 50,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        if max_age < 0:
            raise ValueError(
                "max_age cannot be negative."
            )

        if min_hits < 1:
            raise ValueError(
                "min_hits must be at least 1."
            )

        if max_tracks < 1:
            raise ValueError(
                "max_tracks must be at least 1."
            )

        self.iou_threshold = float(
            iou_threshold
        )

        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.max_tracks = int(max_tracks)

        self._tracks: dict[int, Track] = {}

        self._next_track_id = 1

        self._frame_index = 0

    # ========================================================
    # Public API
    # ========================================================

    def update(
        self,
        detections: list[Detection],
    ) -> list[Track]:
        """
        Update tracker with detections from the current frame.

        Returns:
            Active tracks for the current frame.
        """

        self._frame_index += 1

        # Limit the number of detections to avoid pathological
        # situations in noisy frames.
        detections = detections[
            : self.max_tracks
        ]

        if not self._tracks:
            self._create_initial_tracks(
                detections
            )

            return self.get_active_tracks()

        if not detections:
            self._age_all_tracks()

            self._remove_stale_tracks()

            return self.get_active_tracks()

        matches, unmatched_tracks, unmatched_detections = (
            self._match_detections(detections)
        )

        self._update_matched_tracks(
            detections,
            matches,
        )

        self._age_unmatched_tracks(
            unmatched_tracks
        )

        self._create_new_tracks(
            detections,
            unmatched_detections,
        )

        self._remove_stale_tracks()

        return self.get_active_tracks()

    def get_active_tracks(self) -> list[Track]:
        """
        Return active tracks sorted by Track ID.
        """

        tracks = [
            track
            for track in self._tracks.values()
            if track.time_since_update <= self.max_age
        ]

        tracks.sort(
            key=lambda track: track.track_id
        )

        return tracks

    def get_confirmed_tracks(self) -> list[Track]:
        """
        Return only confirmed tracks.
        """

        tracks = [
            track
            for track in self.get_active_tracks()
            if track.is_confirmed
        ]

        return tracks

    def get_track(
        self,
        track_id: int,
    ) -> Optional[Track]:
        """
        Return one track by Track ID.
        """

        return self._tracks.get(track_id)

    def assign_person_id(
        self,
        track_id: int,
        person_id: str,
    ) -> None:
        """
        Attach a persistent Person ID to a Track.

        Track ID and Person ID intentionally remain separate.
        """

        track = self._tracks.get(track_id)

        if track is None:
            raise KeyError(
                f"Unknown track ID: {track_id}"
            )

        if not person_id:
            raise ValueError(
                "person_id cannot be empty."
            )

        track.person_id = person_id

    def remove_track(
        self,
        track_id: int,
    ) -> None:
        """
        Remove a specific track.
        """

        self._tracks.pop(
            track_id,
            None,
        )

    def reset(self) -> None:
        """
        Remove all tracks and reset the tracker.
        """

        self._tracks.clear()

        self._next_track_id = 1

        self._frame_index = 0

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    # ========================================================
    # Matching
    # ========================================================

    def _match_detections(
        self,
        detections: list[Detection],
    ) -> tuple[
        list[TrackMatch],
        list[int],
        list[int],
    ]:
        """
        Greedy IoU-based matching.

        Returns:
            matches
            unmatched_track_ids
            unmatched_detection_indices
        """

        track_items = list(
            self._tracks.items()
        )

        if not track_items:
            return (
                [],
                [],
                list(range(len(detections))),
            )

        iou_candidates: list[
            tuple[float, int, int]
        ] = []

        for track_id, track in track_items:
            for detection_index, detection in enumerate(
                detections
            ):
                iou = self._calculate_iou(
                    track.bounding_box,
                    detection.bounding_box,
                )

                if iou >= self.iou_threshold:
                    iou_candidates.append(
                        (
                            iou,
                            track_id,
                            detection_index,
                        )
                    )

        # Highest IoU first.
        iou_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        matches: list[TrackMatch] = []

        for (
            iou,
            track_id,
            detection_index,
        ) in iou_candidates:
            if track_id in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            used_tracks.add(track_id)
            used_detections.add(
                detection_index
            )

            matches.append(
                TrackMatch(
                    track_id=track_id,
                    detection_index=detection_index,
                    iou=iou,
                )
            )

        unmatched_tracks = [
            track_id
            for track_id, _ in track_items
            if track_id not in used_tracks
        ]

        unmatched_detections = [
            detection_index
            for detection_index in range(
                len(detections)
            )
            if detection_index not in used_detections
        ]

        return (
            matches,
            unmatched_tracks,
            unmatched_detections,
        )

    # ========================================================
    # Track Updates
    # ========================================================

    def _update_matched_tracks(
        self,
        detections: list[Detection],
        matches: list[TrackMatch],
    ) -> None:
        """
        Update matched tracks.
        """

        for match in matches:
            track = self._tracks.get(
                match.track_id
            )

            if track is None:
                continue

            detection = detections[
                match.detection_index
            ]

            # Basic temporal smoothing.
            smoothed_box = self._smooth_box(
                track.bounding_box,
                detection.bounding_box,
                alpha=0.65,
            )

            track.bounding_box = smoothed_box

            track.confidence = (
                0.7 * track.confidence
                + 0.3 * detection.confidence
            )

            track.age += 1
            track.hits += 1
            track.time_since_update = 0

            if track.hits >= self.min_hits:
                track.is_confirmed = True

    def _age_unmatched_tracks(
        self,
        unmatched_track_ids: list[int],
    ) -> None:
        """
        Age tracks that were not matched this frame.
        """

        for track_id in unmatched_track_ids:
            track = self._tracks.get(
                track_id
            )

            if track is None:
                continue

            track.age += 1
            track.time_since_update += 1

    def _age_all_tracks(self) -> None:
        """
        Age every existing track when there are no detections.
        """

        for track in self._tracks.values():
            track.age += 1
            track.time_since_update += 1

    def _create_initial_tracks(
        self,
        detections: list[Detection],
    ) -> None:
        """
        Create tracks when tracker is initially empty.
        """

        for detection in detections:
            self._create_track(
                detection
            )

    def _create_new_tracks(
        self,
        detections: list[Detection],
        detection_indices: list[int],
    ) -> None:
        """
        Create tracks for unmatched detections.
        """

        available_slots = (
            self.max_tracks
            - len(self._tracks)
        )

        if available_slots <= 0:
            return

        for detection_index in detection_indices[
            :available_slots
        ]:
            self._create_track(
                detections[detection_index]
            )

    def _create_track(
        self,
        detection: Detection,
    ) -> Track:
        """
        Create one new Track.
        """

        track = Track(
            track_id=self._next_track_id,
            bounding_box=detection.bounding_box,
            confidence=float(
                np.clip(
                    detection.confidence,
                    0.0,
                    1.0,
                )
            ),
            age=1,
            hits=1,
            time_since_update=0,
            is_confirmed=(
                self.min_hits <= 1
            ),
        )

        self._tracks[
            self._next_track_id
        ] = track

        self._next_track_id += 1

        return track

    def _remove_stale_tracks(self) -> None:
        """
        Remove tracks that have been missing for too long.
        """

        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.time_since_update
            > self.max_age
        ]

        for track_id in stale_ids:
            del self._tracks[track_id]

    # ========================================================
    # Geometry
    # ========================================================

    @staticmethod
    def _calculate_iou(
        box_a: BoundingBox,
        box_b: BoundingBox,
    ) -> float:
        """
        Calculate Intersection over Union.
        """

        intersection_x1 = max(
            box_a.x1,
            box_b.x1,
        )

        intersection_y1 = max(
            box_a.y1,
            box_b.y1,
        )

        intersection_x2 = min(
            box_a.x2,
            box_b.x2,
        )

        intersection_y2 = min(
            box_a.y2,
            box_b.y2,
        )

        intersection_width = max(
            0,
            intersection_x2
            - intersection_x1,
        )

        intersection_height = max(
            0,
            intersection_y2
            - intersection_y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        if intersection_area <= 0:
            return 0.0

        area_a = max(
            0,
            box_a.width,
        ) * max(
            0,
            box_a.height,
        )

        area_b = max(
            0,
            box_b.width,
        ) * max(
            0,
            box_b.height,
        )

        union_area = (
            area_a
            + area_b
            - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return float(
            intersection_area
            / union_area
        )

    @staticmethod
    def _smooth_box(
        old_box: BoundingBox,
        new_box: BoundingBox,
        alpha: float,
    ) -> BoundingBox:
        """
        Temporally smooth the bounding box.

        alpha close to 1:
            follow the new detection faster.

        alpha close to 0:
            keep more of the previous position.
        """

        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0,
            )
        )

        x1 = int(
            round(
                (
                    old_box.x1
                    * (1.0 - alpha)
                )
                + (
                    new_box.x1
                    * alpha
                )
            )
        )

        y1 = int(
            round(
                (
                    old_box.y1
                    * (1.0 - alpha)
                )
                + (
                    new_box.y1
                    * alpha
                )
            )
        )

        x2 = int(
            round(
                (
                    old_box.x2
                    * (1.0 - alpha)
                )
                + (
                    new_box.x2
                    * alpha
                )
            )
        )

        y2 = int(
            round(
                (
                    old_box.y2
                    * (1.0 - alpha)
                )
                + (
                    new_box.y2
                    * alpha
                )
            )
        )

        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=max(x1 + 1, x2),
            y2=max(y1 + 1, y2),
        )