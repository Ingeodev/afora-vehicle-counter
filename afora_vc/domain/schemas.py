from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ZoneConfig:
    """Configuracion de una zona poligonal de interes."""
    label: str
    points: np.ndarray

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneConfig":
        return cls(
            label=data["label"],
            points=np.array(data["points"], dtype=np.int32),
        )

    def scale(self, scale_x: float, scale_y: float) -> "ZoneConfig":
        scaled = self.points.copy().astype(float)
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        return ZoneConfig(label=self.label, points=scaled.astype(np.int32))


@dataclass
class VideoInfo:
    """Metadatos de un video a procesar."""
    path: str
    hora_inicial: str
    mask_path: str | None = None
    zones_path: str | None = None
    output_folder: str | None = None
    output_name: str | None = None
    video_id: str | None = None

    fps: float | None = None
    width: int | None = None
    height: int | None = None
    total_frames: int | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.fps and self.total_frames:
            return self.total_frames / self.fps
        return None

    @property
    def duration_minutes(self) -> float | None:
        if self.duration_seconds:
            return self.duration_seconds / 60
        return None


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class RawDetection:
    """Deteccion cruda SIN track_id. Salida del detector."""
    class_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float = 1.0
    mask_polygon: np.ndarray | None = None
    mask_center: tuple[int, int] | None = None

    @property
    def center(self) -> tuple[int, int]:
        if self.mask_center is not None:
            return self.mask_center
        return self.bbox.center


@dataclass
class Detection:
    """Deteccion con track_id asignado. Salida del tracker."""
    track_id: int
    class_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float = 1.0
    mask_polygon: np.ndarray | None = None
    mask_center: tuple[int, int] | None = None

    @property
    def center(self) -> tuple[int, int]:
        if self.mask_center is not None:
            return self.mask_center
        return self.bbox.center

    @classmethod
    def from_raw(cls, raw: RawDetection, track_id: int) -> "Detection":
        return cls(
            track_id=track_id,
            class_id=raw.class_id,
            class_name=raw.class_name,
            bbox=raw.bbox,
            confidence=raw.confidence,
            mask_polygon=raw.mask_polygon,
            mask_center=raw.mask_center,
        )


@dataclass
class ZoneEntry:
    vehicle_id: int
    vehicle_type: str
    zone: str
    date: str
    exact_time: str
    timestamp_formatted: str


@dataclass
class ProcessingResult:
    video_path: str
    output_video_path: str | None
    csv_path: str | None
    total_detections: int
    zone_counts: dict[str, int]
    processing_time_seconds: float
    frames_processed: int
    detection_log: list[ZoneEntry] = field(default_factory=list)

    @property
    def processing_time_minutes(self) -> float:
        return self.processing_time_seconds / 60


@dataclass
class TrackHistory:
    track_id: int
    positions: list[tuple[int, int]] = field(default_factory=list)
    zones_visited: list[ZoneEntry] = field(default_factory=list)
    vehicle_type: str | None = None

    def add_position(self, x: int, y: int) -> None:
        self.positions.append((x, y))

    @property
    def last_position(self) -> tuple[int, int] | None:
        return self.positions[-1] if self.positions else None

    @property
    def previous_position(self) -> tuple[int, int] | None:
        if len(self.positions) >= 2:
            return self.positions[-2]
        return None


@dataclass
class FrameData:
    frame: np.ndarray
    index: int
    timestamp_seconds: float
