from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ZoneEntryRecord:
    track_id: int
    vehicle_type: str
    zone: str
    position: tuple[int, int]
    timestamp: float
    frame_num: int
    is_duplicate: bool = False


class ZoneDeduplicator:
    def __init__(
        self,
        spatial_threshold: float = 150.0,
        temporal_threshold: float = 5.0,
        same_type_only: bool = True,
    ):
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.same_type_only = same_type_only

        self._zone_entries: dict[str, list[ZoneEntryRecord]] = {}
        self._duplicate_ids: set[int] = set()

        self._stats = {"total_attempts": 0, "duplicates_detected": 0, "valid_entries": 0}

    def should_count_entry(
        self,
        track_id: int,
        zone: str,
        position: tuple[int, int],
        timestamp: float,
        vehicle_type: str,
        frame_num: int,
    ) -> bool:
        self._stats["total_attempts"] += 1

        if zone not in self._zone_entries:
            self._zone_entries[zone] = []

        is_duplicate = self._check_duplicate(zone, track_id, position, timestamp, vehicle_type)

        entry = ZoneEntryRecord(
            track_id=track_id,
            vehicle_type=vehicle_type,
            zone=zone,
            position=position,
            timestamp=timestamp,
            frame_num=frame_num,
            is_duplicate=is_duplicate,
        )
        self._zone_entries[zone].append(entry)

        if is_duplicate:
            self._stats["duplicates_detected"] += 1
            self._duplicate_ids.add(track_id)
            return False

        self._stats["valid_entries"] += 1
        return True

    def _check_duplicate(
        self, zone: str, track_id: int, position: tuple[int, int], timestamp: float, vehicle_type: str
    ) -> bool:
        if zone not in self._zone_entries:
            return False

        recent_entries = self._zone_entries[zone]

        for entry in reversed(recent_entries):
            if entry.track_id != track_id:
                continue
            time_diff = timestamp - entry.timestamp
            if time_diff > self.temporal_threshold:
                break
            if time_diff < 0:
                continue
            if self.same_type_only and entry.vehicle_type != vehicle_type:
                continue
            dist = self._distance(position, entry.position)
            if dist < self.spatial_threshold:
                return True

        return False

    def _distance(self, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_stats(self) -> dict[str, Any]:
        return self._stats.copy()

    def get_duplicate_ids(self) -> set[int]:
        return self._duplicate_ids.copy()

    def clear(self) -> None:
        self._zone_entries.clear()
        self._duplicate_ids.clear()
        self._stats = {"total_attempts": 0, "duplicates_detected": 0, "valid_entries": 0}

    def cleanup_old_entries(self, current_timestamp: float) -> None:
        cutoff = current_timestamp - (self.temporal_threshold * 3)
        for zone in self._zone_entries:
            self._zone_entries[zone] = [
                e for e in self._zone_entries[zone] if e.timestamp > cutoff
            ]
