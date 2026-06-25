from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AbsoluteZoneEntry:
    """ZoneEntry con tiempo absoluto derivado del nombre del video.

    El vehicle_id incluye video, id original y tipo porque la deteccion
    puede saltar de clase por error y un mismo ID numerico aparecer con
    distintos tipos. El composite key asegura que cada combinacion
    se trackee independientemente.
    """
    vehicle_id: str
    vehicle_type: str
    zone: str
    date: str
    absolute_time_seconds: float
    absolute_time_str: str
    video_name: str


@dataclass
class VehiclePath:
    """Trayectoria completa de un vehiculo a traves de zonas."""
    vehicle_id: str
    vehicle_type: str
    zones: list[str]
    first_seen_seconds: float
    last_seen_seconds: float
    first_seen_str: str = ""
    last_seen_str: str = ""
    flow_name: str | None = None


@dataclass
class CountGroupConfig:
    """Configuracion de un grupo de conteo del YAML."""
    name: str
    include: dict[int, str] | Literal["all"]
    exclude: dict[int, str] | Literal["none"]
    access: str
    zone_paths: list[list[str]]


@dataclass
class CountConfig:
    """Configuracion completa de la seccion count: del YAML."""
    interval_minutes: int
    groups: list[CountGroupConfig]
    graph: str = "web"  # "none" | "image" | "web"


@dataclass
class MatchedFlow:
    """Un vehiculo emparejado con un flujo especifico."""
    vehicle_id: str
    vehicle_type: str
    flow_name: str
    access: str
    absolute_seconds: float
    interval_key: str


@dataclass
class AggregatedCell:
    """Una celda en la tabla final de conteo."""
    access: str
    flow_name: str
    vehicle_type: str
    interval: str
    count: int
