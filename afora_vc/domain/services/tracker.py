from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..schemas import Detection, TrackHistory


class VehicleTracker:
    """Gestiona el historial de posiciones de vehiculos (post-tracking)."""

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self._tracks: dict[int, TrackHistory] = {}

    def update(self, detection: Detection) -> TrackHistory:
        track_id = detection.track_id
        cx, cy = detection.center

        if track_id not in self._tracks:
            self._tracks[track_id] = TrackHistory(
                track_id=track_id, vehicle_type=detection.class_name
            )

        track = self._tracks[track_id]
        track.add_position(cx, cy)

        if len(track.positions) > self.max_history:
            track.positions = track.positions[-self.max_history :]

        if track.vehicle_type is None:
            track.vehicle_type = detection.class_name

        return track

    def update_batch(self, detections: list[Detection]) -> None:
        for detection in detections:
            self.update(detection)

    def get_track(self, track_id: int) -> TrackHistory | None:
        return self._tracks.get(track_id)

    def get_trajectory(
        self, track_id: int, last_n: int | None = None
    ) -> list[tuple[int, int]]:
        track = self._tracks.get(track_id)
        if track is None:
            return []
        positions = track.positions
        if last_n is not None and len(positions) > last_n:
            return positions[-last_n:]
        return positions

    def get_trajectory_array(
        self, track_id: int, last_n: int | None = None
    ) -> np.ndarray:
        positions = self.get_trajectory(track_id, last_n)
        if not positions:
            return np.array([])
        return np.array(positions, dtype=np.int32).reshape((-1, 1, 2))

    def has_previous_position(self, track_id: int) -> bool:
        track = self._tracks.get(track_id)
        return track is not None and len(track.positions) >= 2

    def get_movement(
        self, track_id: int
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        track = self._tracks.get(track_id)
        if track is None or len(track.positions) < 2:
            return None
        return (track.positions[-2], track.positions[-1])

    def clear(self) -> None:
        self._tracks.clear()

    def remove_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def get_active_tracks(self) -> list[int]:
        return list(self._tracks.keys())

    def __len__(self) -> int:
        return len(self._tracks)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._tracks

    def __iter__(self) -> Iterator[TrackHistory]:
        return iter(self._tracks.values())
