from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Literal

import yaml

from ..domain.counting.entities import (
    ZONE_SEPARATOR,
    WILDCARD_MANY,
    WILDCARD_ONE,
    CountConfig,
    CountGroupConfig,
)
from ..domain.exceptions import ConfigurationError

_VALID_COUNT_KEYS = {"interval", "graph", "groups"}
_VALID_GROUP_KEYS = {"name", "access", "include", "exclude", "zones"}


def _check_unknown_keys(
    data: dict[str, Any], dataclass_type: type, origin: str, skip: set[str] | None = None
) -> None:
    """Falla con un mensaje util ante un typo en el YAML.

    Sin esto, un campo mal escrito termina en
    ``BatchConfig.__init__() got an unexpected keyword argument``, que no dice
    ni donde ni cual es la alternativa valida.
    """
    skip = skip or set()
    valid = {f.name for f in fields(dataclass_type)} - skip
    unknown = sorted(set(data) - valid - skip)
    if unknown:
        raise ConfigurationError(
            f"Campo(s) desconocido(s) en {origin}: {', '.join(unknown)}. "
            f"Validos: {', '.join(sorted(valid))}"
        )


def _parse_zone_pattern(group_name: str, raw_pattern: Any) -> list[str]:
    """Valida un patron de zonas y lo normaliza a una lista de strings.

    `_` matchea una zona y `+` una o mas; el resto son zonas literales.
    """
    if not isinstance(raw_pattern, list) or not raw_pattern:
        raise ConfigurationError(
            f"El grupo '{group_name}' tiene un patron de zonas vacio o invalido: "
            f"{raw_pattern!r}"
        )

    pattern: list[str] = []
    for raw_zone in raw_pattern:
        zone = str(raw_zone).strip()
        if not zone:
            raise ConfigurationError(
                f"El grupo '{group_name}' tiene una zona vacia en {raw_pattern!r}"
            )
        if zone in {"*", "%"}:
            raise ConfigurationError(
                f"El grupo '{group_name}' usa '{zone}' en {raw_pattern!r}. "
                f"Usa '{WILDCARD_ONE}' para exactamente una zona y "
                f"'{WILDCARD_MANY}' para una o mas ('*' y '%' no son "
                f"validos en YAML sin comillas)"
            )
        if ZONE_SEPARATOR in zone:
            raise ConfigurationError(
                f"La zona '{zone}' del grupo '{group_name}' no puede contener "
                f"'{ZONE_SEPARATOR}'"
            )
        pattern.append(zone)

    if not any(z not in (WILDCARD_ONE, WILDCARD_MANY) for z in pattern):
        raise ConfigurationError(
            f"El patron {pattern} del grupo '{group_name}' no tiene zonas "
            f"literales: matchearia cualquier trayectoria"
        )

    for current, following in zip(pattern, pattern[1:]):
        if current == WILDCARD_MANY and following == WILDCARD_MANY:
            raise ConfigurationError(
                f"El patron {pattern} del grupo '{group_name}' repite "
                f"'{WILDCARD_MANY}': con uno solo alcanza"
            )

    return pattern


@dataclass
class VideoOverride:
    path: str
    mask: str | None = None
    zones: str | None = None
    reduce: int | None = None
    batch_size: int | None = None
    pipeline_mode: str | None = None
    model: str | None = None
    backend: str | None = None
    tracker: str | None = None
    device: str | None = None
    strategy: str | None = None
    max_minutes: float | None = None
    night_enhance: bool | None = None
    deblurring: bool | None = None
    no_video: bool | None = None
    codec: str | None = None
    crf: int | None = None
    preset: str | None = None
    output_name: str | None = None
    model_config: str | None = None
    tracker_config: str | None = None
    base_time: str | None = None
    date: str | None = None
    queue_maxsize: int | None = None
    profile: bool | None = None
    profile_path: str | None = None
    audit: bool | None = None


@dataclass
class BatchConfig:
    input: str = ""
    output: str = "./output"
    recursive: bool = True
    mask: str | None = None
    zones: str | None = None

    backend: str = "yolo"
    model: str = "models/yolov8s.pt"
    pipeline_mode: str = "sequential"
    batch_size: int = 1
    reduce: int = 1
    device: str | None = None
    strategy: str = "box"
    tracker: str = "botsort"
    max_minutes: float | None = None
    night_enhance: bool = False
    deblurring: bool = False
    no_video: bool = False
    output_name: str | None = None
    quiet: bool = False

    # Video de salida (solo verificacion): avc1/h264, hevc/h265 o mp4v.
    # crf mas alto = archivo mas chico; preset mas rapido = menos CPU.
    codec: str = "avc1"
    crf: int = 30
    preset: str = "veryfast"

    model_config: str | None = None
    tracker_config: str | None = None
    base_time: str = "00:00"
    date: str = "2025-01-01"
    queue_maxsize: int = 4
    profile: bool = False
    profile_path: str | None = None
    audit: bool = False

    overrides: list[VideoOverride] = field(default_factory=list)
    count: CountConfig | None = None
    progress_file: str | None = None
    _output_viz: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str) -> BatchConfig:
        if not os.path.isfile(path):
            raise ConfigurationError(
                f"No existe el archivo de configuracion: {path}"
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error de sintaxis YAML en {path}: {e}") from e
        config = cls.from_dict(data)
        # Deja la ruta del YAML en el objeto: process_from_config la usa para
        # ubicar progress.csv (reanudacion) aunque se pase el objeto directo.
        config._source_path = os.path.abspath(path)
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchConfig:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigurationError(
                f"El YAML debe ser un mapa de campos, no {type(data).__name__}"
            )
        data = dict(data)

        # Si output es un dict (con visualization adentro), extraer output_folder.
        if isinstance(data.get("output"), dict):
            out_dict = data.pop("output")
            data["output"] = out_dict.get("output_folder", "./output")
            # Aceptar visualization sin aplicarlo (compatibilidad con visual-config).
            if "visualization" in out_dict:
                data["_output_viz"] = out_dict["visualization"]

        ov_data = data.pop("overrides", [])
        overrides = []
        for item in ov_data:
            if isinstance(item, str):
                overrides.append(VideoOverride(path=item))
            elif isinstance(item, dict):
                _check_unknown_keys(item, VideoOverride, "un override")
                overrides.append(VideoOverride(**item))

        count_data = data.pop("count", None)
        count_config: CountConfig | None = None
        if count_data is not None:
            count_config = cls._parse_count(count_data)

        _check_unknown_keys(data, cls, "el YAML", skip={"_output_viz"})
        return cls(overrides=overrides, count=count_config, **data)

    @staticmethod
    def _parse_count(data: dict[str, Any]) -> CountConfig:
        unknown = sorted(set(data) - _VALID_COUNT_KEYS)
        if unknown:
            raise ConfigurationError(
                f"Campo(s) desconocido(s) en 'count': {', '.join(unknown)}. "
                f"Validos: {', '.join(sorted(_VALID_COUNT_KEYS))}"
            )

        interval = int(data.get("interval", 15))
        raw_groups = data.get("groups", [])
        groups = []
        seen_patterns: dict[tuple[str, ...], str] = {}

        for g in raw_groups:
            if not isinstance(g, dict):
                raise ConfigurationError(
                    f"Cada grupo de 'count.groups' debe ser un mapa, no {type(g).__name__}"
                )
            name = str(g.get("name", ""))
            unknown = sorted(set(g) - _VALID_GROUP_KEYS)
            if unknown:
                raise ConfigurationError(
                    f"Campo(s) desconocido(s) en el grupo '{name}': {', '.join(unknown)}. "
                    f"Validos: {', '.join(sorted(_VALID_GROUP_KEYS))}"
                )

            raw_zones = g.get("zones", [])
            if not isinstance(raw_zones, list) or not raw_zones:
                raise ConfigurationError(
                    f"El grupo '{name}' no tiene 'zones' definidas"
                )

            zone_paths: list[list[str]] = []
            for raw_pattern in raw_zones:
                pattern = _parse_zone_pattern(name, raw_pattern)
                key = tuple(pattern)
                if key in seen_patterns:
                    raise ConfigurationError(
                        f"El patron {pattern} esta repetido en los grupos "
                        f"'{seen_patterns[key]}' y '{name}'. Cada trayectoria "
                        f"debe matchear con un solo flujo"
                    )
                seen_patterns[key] = name
                zone_paths.append(pattern)

            include: dict[int, str] | Literal["all"] = g.get("include", "all")
            exclude: dict[int, str] | Literal["none"] = g.get("exclude", "none")
            if isinstance(include, dict):
                include = {int(k): str(v) for k, v in include.items()}
            if isinstance(exclude, dict):
                exclude = {int(k): str(v) for k, v in exclude.items()}
            groups.append(CountGroupConfig(
                name=name,
                include=include,
                exclude=exclude,
                access=str(g.get("access", "")),
                zone_paths=zone_paths,
            ))
        graph = str(data.get("graph", "web")).lower()
        return CountConfig(interval_minutes=interval, groups=groups, graph=graph)

    def get_override(self, video_rel_path: str) -> VideoOverride | None:
        norm = video_rel_path.replace("\\", "/")
        for ov in self.overrides:
            ov_norm = ov.path.replace("\\", "/")
            if ov_norm == norm:
                return ov
        return None
