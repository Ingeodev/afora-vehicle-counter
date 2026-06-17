from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrackInfo:
    track_id: int
    vehicle_type: str
    positions: list[tuple[int, int]] = field(default_factory=list)
    last_seen_frame: int = 0
    zones_visited: list[str] = field(default_factory=list)
    is_duplicate: bool = False
    merged_with: int | None = None


class TrackDeduplicator:
    def __init__(
        self,
        max_distance: float = 60.0,
        lookback_frames: int = 45,
        min_positions_to_compare: int = 5,
        max_tracks_to_keep: int = 200,
        position_match_threshold: float = 0.6,
    ):
        self.max_distance = max_distance
        self.lookback_frames = lookback_frames
        self.min_positions = min_positions_to_compare
        self.max_tracks = max_tracks_to_keep
        self.position_match_threshold = position_match_threshold

        self._tracks: dict[int, TrackInfo] = {}
        self._id_mapping: dict[int, int] = {}
        self._current_frame = 0

    def get_original_id(
        self, track_id: int, position: tuple[int, int], frame_num: int, vehicle_type: str
    ) -> int:
        self._current_frame = frame_num

        if track_id in self._id_mapping:
            original_id = self._id_mapping[track_id]
            if original_id in self._tracks:
                self._tracks[original_id].positions.append(position)
                self._tracks[original_id].last_seen_frame = frame_num
            return original_id

        if track_id in self._tracks:
            self._tracks[track_id].positions.append(position)
            self._tracks[track_id].last_seen_frame = frame_num
            return track_id

        similar_track = self._find_similar_track(position, frame_num, vehicle_type)

        if similar_track is not None:
            self._id_mapping[track_id] = similar_track
            self._tracks[similar_track].positions.append(position)
            self._tracks[similar_track].last_seen_frame = frame_num
            return similar_track

        self._tracks[track_id] = TrackInfo(
            track_id=track_id,
            vehicle_type=vehicle_type,
            positions=[position],
            last_seen_frame=frame_num,
        )

        self._cleanup_old_tracks()
        return track_id

    def _find_similar_track(
        self, position: tuple[int, int], frame_num: int, vehicle_type: str
    ) -> int | None:
        min_frame = frame_num - self.lookback_frames
        candidates: list[tuple[int, float]] = []

        for tid, track in self._tracks.items():
            if track.last_seen_frame < min_frame:
                continue
            if track.last_seen_frame == frame_num:
                continue
            if track.vehicle_type != vehicle_type:
                continue

            if len(track.positions) < self.min_positions:
                last_pos = track.positions[-1]
                dist = self._distance(position, last_pos)
                if dist < self.max_distance:
                    candidates.append((tid, dist))
            else:
                expected_pos = self._predict_position(
                    track.positions, frame_num - track.last_seen_frame
                )
                dist = self._distance(position, expected_pos)
                if dist < self.max_distance * 1.5:
                    candidates.append((tid, dist))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def _predict_position(
        self, positions: list[tuple[int, int]], frames_ahead: int
    ) -> tuple[int, int]:
        if len(positions) < 2:
            return positions[-1]

        n = min(10, len(positions))
        recent = positions[-n:]

        dx = (recent[-1][0] - recent[0][0]) / (n - 1)
        dy = (recent[-1][1] - recent[0][1]) / (n - 1)

        predicted_x = recent[-1][0] + dx * frames_ahead
        predicted_y = recent[-1][1] + dy * frames_ahead

        return (int(predicted_x), int(predicted_y))

    def _distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _cleanup_old_tracks(self) -> None:
        if len(self._tracks) <= self.max_tracks:
            return

        sorted_tracks = sorted(self._tracks.items(), key=lambda x: x[1].last_seen_frame)
        to_remove = len(self._tracks) - self.max_tracks
        for tid, _ in sorted_tracks[:to_remove]:
            del self._tracks[tid]
            self._id_mapping = {k: v for k, v in self._id_mapping.items() if v != tid}

    def register_zone_entry(self, track_id: int, zone: str) -> bool:
        original_id = self._id_mapping.get(track_id, track_id)
        if original_id not in self._tracks:
            return True
        track = self._tracks[original_id]
        if zone in track.zones_visited:
            return False
        track.zones_visited.append(zone)
        return True

    def get_zones_for_track(self, track_id: int) -> list[str]:
        original_id = self._id_mapping.get(track_id, track_id)
        if original_id in self._tracks:
            return self._tracks[original_id].zones_visited.copy()
        return []

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_tracks": len(self._tracks),
            "duplicates_detected": len(self._id_mapping),
            "active_mappings": dict(self._id_mapping),
        }

    def clear(self) -> None:
        self._tracks.clear()
        self._id_mapping.clear()
        self._current_frame = 0
