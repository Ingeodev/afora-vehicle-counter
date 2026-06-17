import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..schemas import ZoneConfig, ZoneEntry
from ..ports.storage import StorageReader
from .zone_deduplicator import ZoneDeduplicator


@dataclass
class ZoneCheckResult:
    entered_zone: str | None
    is_new_entry: bool
    all_zones: list[str]


class ZoneManager:
    def __init__(
        self,
        enable_deduplication: bool = True,
        dedup_spatial_threshold: float = 150.0,
        dedup_temporal_threshold: float = 5.0,
    ):
        self._zones_original: list[ZoneConfig] = []
        self._zones_scaled: list[ZoneConfig] = []
        self._vehicle_zone_history: dict[int, list[str]] = {}
        self._detection_log: list[ZoneEntry] = []

        self._enable_dedup = enable_deduplication
        self._zone_dedup = (
            ZoneDeduplicator(
                spatial_threshold=dedup_spatial_threshold,
                temporal_threshold=dedup_temporal_threshold,
                same_type_only=False,
            )
            if enable_deduplication
            else None
        )

        self._frame_num = 0

    def load_from_json(self, storage: StorageReader, path: str) -> int:
        data = storage.read_json(path)
        return self.load_from_dict(data)

    def load_from_dict(self, data: dict[str, Any]) -> int:
        zones_data = data.get("zones", [])
        self._zones_original = [ZoneConfig.from_dict(z) for z in zones_data]
        self._zones_scaled = self._zones_original.copy()
        return len(self._zones_original)

    def scale_to_resolution(
        self, original_width: int, original_height: int, new_width: int, new_height: int
    ) -> None:
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        self._zones_scaled = [zone.scale(scale_x, scale_y) for zone in self._zones_original]

    def point_in_zone(self, point: tuple[int, int], zone: ZoneConfig) -> bool:
        pt = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(zone.points, pt, False) >= 0

    def check_entry(
        self,
        track_id: int,
        x: int,
        y: int,
        timestamp_seconds: float,
        vehicle_type: str,
        date: str = "",
        exact_time: str = "",
    ) -> ZoneCheckResult:
        point = (x, y)

        if track_id not in self._vehicle_zone_history:
            self._vehicle_zone_history[track_id] = []

        entered_zone = None
        is_new_entry = False

        for zone in self._zones_scaled:
            if self.point_in_zone(point, zone):
                if zone.label not in self._vehicle_zone_history[track_id]:
                    if self._enable_dedup and self._zone_dedup:
                        is_valid = self._zone_dedup.should_count_entry(
                            track_id=track_id,
                            zone=zone.label,
                            position=point,
                            timestamp=timestamp_seconds,
                            vehicle_type=vehicle_type,
                            frame_num=self._frame_num,
                        )
                        if not is_valid:
                            self._vehicle_zone_history[track_id].append(zone.label)
                            continue

                    self._vehicle_zone_history[track_id].append(zone.label)
                    entered_zone = zone.label
                    is_new_entry = True

                    minutes = int(timestamp_seconds // 60)
                    secs = int(timestamp_seconds % 60)
                    timestamp_formatted = f"{minutes:02d}:{secs:02d}"

                    entry = ZoneEntry(
                        vehicle_id=track_id,
                        vehicle_type=vehicle_type,
                        zone=zone.label,
                        date=date,
                        exact_time=exact_time,
                        timestamp_formatted=timestamp_formatted,
                    )
                    self._detection_log.append(entry)
                    break

        self._frame_num += 1

        return ZoneCheckResult(
            entered_zone=entered_zone,
            is_new_entry=is_new_entry,
            all_zones=self._vehicle_zone_history.get(track_id, []).copy(),
        )

    def get_zones_for_vehicle(self, track_id: int) -> list[str]:
        return self._vehicle_zone_history.get(track_id, []).copy()

    def get_detection_log(self) -> list[ZoneEntry]:
        return self._detection_log.copy()

    def get_log_count(self) -> int:
        return len(self._detection_log)

    def get_zone_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._detection_log:
            counts[entry.zone] = counts.get(entry.zone, 0) + 1
        return counts

    def clear_history(self) -> None:
        self._vehicle_zone_history.clear()
        self._detection_log.clear()

    @property
    def zones(self) -> list[ZoneConfig]:
        return self._zones_scaled

    @property
    def zone_labels(self) -> list[str]:
        return [z.label for z in self._zones_scaled]

    def __len__(self) -> int:
        return len(self._zones_scaled)

    def __bool__(self) -> bool:
        return len(self._zones_scaled) > 0
