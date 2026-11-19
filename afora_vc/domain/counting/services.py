from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from .entities import (
    AbsoluteZoneEntry,
    VehiclePath,
    CountGroupConfig,
    CountConfig,
    MatchedFlow,
    ZonePattern,
)

# Patron para extraer timestamp del nombre del video:
#   roosvelt_2026-06-01_15.40.13_detections.csv  (YYYY-MM-DD)
#   43hiv00085_10-07-2026_21.31.26_detections.csv  (DD-MM-YYYY)
#   -> HH=15, MM=40, SS=13
_VIDEO_TS_PATTERN = re.compile(
    r".*?_(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})_(\d{2})\.(\d{2})\.(\d{2})_detections\.csv$"
)

_DEFAULT_VEHICLE_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ---------------------------------------------------------------------------
# VehicleClassesResolver
# ---------------------------------------------------------------------------

class VehicleClassesResolver:
    """Resuelve class_id -> vehicle_type desde model-config o defaults."""

    def __init__(self, classes: dict[int, str] | None = None):
        self._classes = classes or dict(_DEFAULT_VEHICLE_CLASSES)

    @classmethod
    def from_model_config(cls, path: str | None) -> VehicleClassesResolver:
        if path and os.path.exists(path):
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            vc = data.get("vehicle_classes")
            if vc:
                return cls({int(k): str(v) for k, v in vc.items()})
        return cls()

    def resolve(self, class_id: int) -> str | None:
        return self._classes.get(class_id)

    def resolve_group_include(
        self,
        include: dict[int, str] | str | list[str],
    ) -> list[str]:
        if isinstance(include, str) and include == "all":
            return list(self._classes.values())
        if isinstance(include, str):
            canonical = self._resolve_name(include)
            return [canonical] if canonical else []
        if isinstance(include, list):
            resolved = []
            for name in include:
                canonical = self._resolve_name(str(name))
                if canonical:
                    resolved.append(canonical)
            return resolved
        if isinstance(include, dict):
            resolved = []
            for cid, name in include.items():
                cid = int(cid)
                canonical = self._resolve_name(str(name)) or self.resolve(cid)
                if canonical is not None:
                    resolved.append(canonical)
            return resolved
        return []

    def resolve_group_exclude(
        self,
        exclude: dict[int, str] | str | list[str],
    ) -> set[str]:
        if isinstance(exclude, str) and exclude == "none":
            return set()
        if isinstance(exclude, str):
            canonical = self._resolve_name(exclude)
            return {canonical} if canonical else set()
        if isinstance(exclude, list):
            excluded = set()
            for name in exclude:
                canonical = self._resolve_name(str(name))
                if canonical:
                    excluded.add(canonical)
            return excluded
        if isinstance(exclude, dict):
            excluded = set()
            for cid, name in exclude.items():
                cid = int(cid)
                canonical = self._resolve_name(str(name)) or self.resolve(cid)
                if canonical is not None:
                    excluded.add(canonical)
            return excluded
        return set()

    def _resolve_name(self, name: str) -> str | None:
        name_lower = name.strip().lower()
        for cls_name in self._classes.values():
            if cls_name.lower() == name_lower:
                return cls_name
        return None

    def get_all_types(self) -> list[str]:
        return list(self._classes.values())


# ---------------------------------------------------------------------------
# AbsoluteTimeConverter
# ---------------------------------------------------------------------------

class AbsoluteTimeConverter:
    """Lee detections.csv, convierte tiempo relativo a absoluto
    y modifica vehicle_id para que sea unico entre videos."""

    def process(self, folder_path: str) -> list[AbsoluteZoneEntry]:
        entries: list[AbsoluteZoneEntry] = []
        for fname in os.listdir(folder_path):
            if not fname.endswith("_detections.csv"):
                continue
            fpath = os.path.join(folder_path, fname)
            self._process_file(fpath, entries)
        return entries

    def _process_file(self, fpath: str, out: list[AbsoluteZoneEntry]) -> None:
        m = _VIDEO_TS_PATTERN.match(os.path.basename(fpath))
        if not m:
            return
        vid_date = m.group(1)
        base_h, base_m, base_s = int(m.group(2)), int(m.group(3)), int(m.group(4))
        base_dt = datetime(2000, 1, 1, base_h, base_m, base_s)
        video_name = os.path.basename(fpath).replace("_detections.csv", "")

        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_raw = row.get("timestamp_formatted", "00:00").strip()
                rel_seconds = self._parse_mmss(ts_raw)
                abs_dt = base_dt + timedelta(seconds=rel_seconds)
                abs_seconds = abs_dt.hour * 3600 + abs_dt.minute * 60 + abs_dt.second

                vt = row.get("vehicle_type", "unknown").strip()
                vid = row.get("vehicle_id", "0").strip()
                composite_id = f"{video_name}_{vid}_{vt}"

                out.append(AbsoluteZoneEntry(
                    vehicle_id=composite_id,
                    vehicle_type=vt,
                    zone=row.get("zone", "").strip(),
                    date=vid_date,
                    absolute_time_seconds=abs_seconds,
                    absolute_time_str=abs_dt.strftime("%H:%M:%S"),
                    video_name=video_name,
                ))

    @staticmethod
    def _parse_mmss(ts: str) -> float:
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0.0


# ---------------------------------------------------------------------------
# VehiclePathBuilder
# ---------------------------------------------------------------------------

class VehiclePathBuilder:
    """Agrupa AbsoluteZoneEntry por vehicle_id y construye VehiclePath."""

    def process(self, entries: list[AbsoluteZoneEntry]) -> list[VehiclePath]:
        groups: dict[str, list[AbsoluteZoneEntry]] = defaultdict(list)
        for e in entries:
            groups[e.vehicle_id].append(e)

        paths: list[VehiclePath] = []
        for vid, group in groups.items():
            group.sort(key=lambda x: x.absolute_time_seconds)
            zones = [e.zone for e in group]
            vt = group[0].vehicle_type
            first_sec = group[0].absolute_time_seconds
            last_sec = group[-1].absolute_time_seconds

            def _sec_to_str(s: float) -> str:
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                sec = int(s % 60)
                return f"{h:02d}:{m:02d}:{sec:02d}"

            paths.append(VehiclePath(
                vehicle_id=vid,
                vehicle_type=vt,
                zones=zones,
                first_seen_seconds=first_sec,
                last_seen_seconds=last_sec,
                first_seen_str=_sec_to_str(first_sec),
                last_seen_str=_sec_to_str(last_sec),
            ))
        return paths


# ---------------------------------------------------------------------------
# FlowMatcher
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FlowRule:
    """Grupo de conteo con los tipos de vehiculo ya resueltos."""

    name: str
    include: frozenset[str]
    exclude: frozenset[str]


class FlowMatcher:
    """Empareja VehiclePath con los flujos definidos en count.groups[].

    El match es completo: el patron debe cubrir toda la trayectoria. Cuando
    varios patrones matchean, gana el mas especifico (ver
    ``ZonePattern.precedence_key``); el orden del YAML solo desempata.
    """

    def __init__(
        self,
        groups: list[CountGroupConfig],
        resolver: VehicleClassesResolver,
    ):
        self._rules: list[_FlowRule] = []
        patterns: list[tuple[tuple[int, int, int, int, int], int, ZonePattern]] = []

        for group_index, g in enumerate(groups):
            self._rules.append(_FlowRule(
                name=g.name,
                include=frozenset(resolver.resolve_group_include(g.include)),
                exclude=frozenset(resolver.resolve_group_exclude(g.exclude)),
            ))
            for pattern_index, zone_path in enumerate(g.zone_paths):
                pattern = ZonePattern.compile(zone_path)
                patterns.append((
                    pattern.precedence_key(group_index, pattern_index),
                    group_index,
                    pattern,
                ))

        patterns.sort(key=lambda item: item[0])
        self._patterns = patterns
        self._hits = [0] * len(patterns)

    def process(self, paths: list[VehiclePath]) -> list[VehiclePath]:
        for path in paths:
            flow_name, pattern = self._match(path)
            path.flow_name = flow_name
            path.flow_pattern = str(pattern) if pattern is not None else None
        return paths

    def _match(self, path: VehiclePath) -> tuple[str | None, ZonePattern | None]:
        if not path.zones:
            return None, None
        for index, (_, group_index, pattern) in enumerate(self._patterns):
            rule = self._rules[group_index]
            if not self._type_matches(path.vehicle_type, rule):
                continue
            if pattern.matches(path.zones):
                self._hits[index] += 1
                return rule.name, pattern
        return None, None

    def _type_matches(self, vt: str, rule: _FlowRule) -> bool:
        if vt in rule.exclude:
            return False
        if rule.include:
            return vt in rule.include
        return True

    def dead_patterns(self) -> list[tuple[str, str]]:
        """Patrones que no matchearon ninguna trayectoria (config muerta)."""
        return [
            (self._rules[group_index].name, str(pattern))
            for index, (_, group_index, pattern) in enumerate(self._patterns)
            if self._hits[index] == 0
        ]


# ---------------------------------------------------------------------------
# InterventionFileExporter
# ---------------------------------------------------------------------------

class InterventionFileExporter:
    """Crea detections.csv con todos los VehiclePath, escribe el reporte de
    trayectorias sin match y retorna los MatchedFlow para el siguiente paso."""

    UNMATCHED_FILENAME = "unmatched_paths.csv"

    def process(
        self,
        paths: list[VehiclePath],
        output_dir: str,
    ) -> list[MatchedFlow]:
        rows: list[dict[str, str]] = []
        matched: list[MatchedFlow] = []

        for p in paths:
            flow_str = p.flow_name if p.flow_name is not None else "UNKNOWN"
            rows.append({
                "vehicle_id": p.vehicle_id,
                "vehicle_type": p.vehicle_type,
                "zones": "[" + ", ".join(p.zones) + "]",
                "hora_inicial": p.first_seen_str,
                "hora_final": p.last_seen_str,
                "path_asignado": flow_str,
                "patron": p.flow_pattern or "",
            })
            if p.flow_name is not None:
                matched.append(self._to_matched_flow(p))

        out_path = os.path.join(output_dir, "detections.csv")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "vehicle_id", "vehicle_type", "zones",
                "hora_inicial", "hora_final", "path_asignado", "patron",
            ])
            writer.writeheader()
            writer.writerows(rows)

        self._write_unmatched(paths, output_dir)
        return matched

    def _write_unmatched(
        self,
        paths: list[VehiclePath],
        output_dir: str,
    ) -> None:
        """Agrupa las trayectorias UNKNOWN para poder ajustar los patrones."""
        out_path = os.path.join(output_dir, self.UNMATCHED_FILENAME)
        stats: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}

        for p in paths:
            if p.flow_name is not None:
                continue
            key = (tuple(p.zones), p.vehicle_type)
            item = stats.get(key)
            if item is None:
                stats[key] = {
                    "count": 1,
                    "hora_min": p.first_seen_str,
                    "hora_max": p.last_seen_str,
                }
                continue
            item["count"] += 1
            if p.first_seen_str < item["hora_min"]:
                item["hora_min"] = p.first_seen_str
            if p.last_seen_str > item["hora_max"]:
                item["hora_max"] = p.last_seen_str

        if not stats:
            # Evitar dejar un reporte viejo de una corrida anterior.
            if os.path.exists(out_path):
                os.remove(out_path)
            return

        rows = [
            {
                "zones": "[" + ", ".join(zones) + "]",
                "vehicle_type": vehicle_type,
                "count": item["count"],
                "hora_min": item["hora_min"],
                "hora_max": item["hora_max"],
            }
            for (zones, vehicle_type), item in stats.items()
        ]
        rows.sort(key=lambda r: (-r["count"], r["zones"], r["vehicle_type"]))

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "zones", "vehicle_type", "count", "hora_min", "hora_max",
            ])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _to_matched_flow(p: VehiclePath) -> MatchedFlow:
        flow_name = p.flow_name or "UNKNOWN"
        # Buscar el access de este flow en los grupos
        access = ""
        return MatchedFlow(
            vehicle_id=p.vehicle_id,
            vehicle_type=p.vehicle_type,
            flow_name=flow_name,
            access=access,
            absolute_seconds=p.first_seen_seconds,
            interval_key="",
        )

    @staticmethod
    def enrich_matched_with_access(
        matched: list[MatchedFlow],
        groups: list[CountGroupConfig],
    ) -> list[MatchedFlow]:
        name_to_access = {g.name: g.access for g in groups}
        for m in matched:
            m.access = name_to_access.get(m.flow_name, "")
        return matched


# ---------------------------------------------------------------------------
# IntervalGrouper
# ---------------------------------------------------------------------------

class IntervalGrouper:
    """Agrupa MatchedFlow por intervalos de tiempo."""

    def __init__(self, interval_minutes: int):
        self._interval_seconds = interval_minutes * 60

    def process(
        self,
        matched: list[MatchedFlow],
    ) -> dict[tuple[str, str, str, str], int]:
        result: dict[tuple[str, str, str, str], int] = defaultdict(int)
        for m in matched:
            slot = int(m.absolute_seconds // self._interval_seconds)
            start = slot * self._interval_seconds
            key = self._interval_key(start)
            result[(m.access, m.flow_name, m.vehicle_type, key)] += 1
        return dict(result)

    def _interval_key(self, start_seconds: float) -> str:
        h = int(start_seconds // 3600)
        m = int((start_seconds % 3600) // 60)
        end = start_seconds + self._interval_seconds
        h2 = int(end // 3600)
        m2 = int((end % 3600) // 60)
        return f"{h:02d}:{m:02d}-{h2:02d}:{m2:02d}"

    def all_intervals(self) -> list[str]:
        day_seconds = 24 * 3600
        intervals: list[str] = []
        for start in range(0, day_seconds, self._interval_seconds):
            intervals.append(self._interval_key(start))
        return intervals
