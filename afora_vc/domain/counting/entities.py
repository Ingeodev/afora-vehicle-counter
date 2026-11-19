from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# Separador con el que se serializa una trayectoria para matchearla.
# No puede aparecer en el nombre de una zona (se valida al cargar el YAML).
ZONE_SEPARATOR = ","

# Comodines de los patrones de flujo:
#   WILDCARD_ONE  -> exactamente una zona
#   WILDCARD_MANY -> una o mas zonas
# Se eligieron por ser validos en YAML sin comillas ('*' y '%' no lo son).
WILDCARD_ONE = "_"
WILDCARD_MANY = "+"


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
    # Patron con el que matcheo (para auditoria), p.ej. "[A,+,D]".
    flow_pattern: str | None = None


@dataclass(frozen=True)
class ZonePattern:
    """Secuencia de zonas que define un flujo.

    `_` matchea exactamente una zona y `+` matchea una o mas. El match es
    completo: el patron tiene que cubrir toda la trayectoria, sin zonas
    sobrantes ni faltantes.
    """

    zones: tuple[str, ...]
    regex: re.Pattern[str]
    literals: int
    singles: int
    runs: int

    @classmethod
    def compile(cls, zones: Iterable[str]) -> ZonePattern:
        parts = tuple(str(z) for z in zones)
        regex_parts: list[str] = []
        literals = singles = runs = 0
        for z in parts:
            if ZONE_SEPARATOR in z:
                raise ValueError(
                    f"La zona '{z}' no puede contener '{ZONE_SEPARATOR}'"
                )
            if z == WILDCARD_ONE:
                regex_parts.append("[^,]+")
                singles += 1
            elif z == WILDCARD_MANY:
                regex_parts.append("[^,]+(?:,[^,]+)*")
                runs += 1
            else:
                regex_parts.append(re.escape(z))
                literals += 1
        return cls(
            zones=parts,
            regex=re.compile(ZONE_SEPARATOR.join(regex_parts)),
            literals=literals,
            singles=singles,
            runs=runs,
        )

    def matches(self, path: Sequence[str]) -> bool:
        return self.regex.fullmatch(ZONE_SEPARATOR.join(path)) is not None

    def precedence_key(
        self, group_index: int, pattern_index: int
    ) -> tuple[int, int, int, int, int]:
        """Menor es mas especifico.

        Primero gana el patron con mas zonas literales (el 100% definido le
        gana al que usa comodines), despues el que tiene menos `+`, despues
        el que fija mas posiciones con `_`, y al final el orden del YAML.
        """
        return (-self.literals, self.runs, -self.singles, group_index, pattern_index)

    def __str__(self) -> str:
        return "[" + ZONE_SEPARATOR.join(self.zones) + "]"


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
