from __future__ import annotations

import csv
import os
from typing import Protocol

from ..domain.counting.entities import CountConfig, CountGroupConfig
from ..domain.counting.enums import Access
from ..domain.counting.services import IntervalGrouper, VehicleClassesResolver


class CountExporterStrategy(Protocol):
    """Strategy para el formato de exportacion."""

    def write(
        self,
        data: dict[tuple[str, str, str, str], int],
        config: CountConfig,
        resolver: VehicleClassesResolver,
        path: str,
    ) -> None:
        ...


class DefaultFlowExporter:
    """
    Formato por defecto: CSV con 3 filas de encabezado.

    Fila 1: Acceso (norte, sur, oeste, este) - celdas combinadas
    Fila 2: Nombre del flujo (1A, 4, ...) - celdas combinadas
    Fila 3: Tipo de vehiculo (car, motorcycle, bus, truck)
    Columna fija: intervalo (00:00-00:14, ..., 23:45-23:59)
    """

    def write(
        self,
        data: dict[tuple[str, str, str, str], int],
        config: CountConfig,
        resolver: VehicleClassesResolver,
        path: str,
    ) -> None:
        if not data:
            return

        # Construir estructura de columnas: [(access, flow_name, vehicle_type), ...]
        columns = self._build_columns(config, resolver)

        # Generar todos los intervalos del dia
        interval_grouper = IntervalGrouper(config.interval_minutes)
        intervals = interval_grouper.all_intervals()

        # Construir filas
        rows: list[list[str]] = []
        for interval in intervals:
            row = [interval]
            for access, flow_name, vt in columns:
                key = (access, flow_name, vt, interval)
                row.append(str(data.get(key, 0)))
            rows.append(row)

        # Generar las 3 filas de encabezado
        header_row1: list[str] = ["intervalo"]
        header_row2: list[str] = [""]
        header_row3: list[str] = [""]

        i = 0
        ncols = len(columns)
        while i < ncols:
            current_access = columns[i][0]
            # Contar cuantas columnas ocupa este acceso
            j = i
            while j < ncols and columns[j][0] == current_access:
                j += 1
            access_span = j - i
            header_row1.append(current_access)
            for _ in range(access_span - 1):
                header_row1.append("")

            # Flujos dentro de este acceso
            k = i
            while k < j:
                current_flow = columns[k][1]
                l = k
                while l < j and columns[l][1] == current_flow:
                    l += 1
                flow_span = l - k
                header_row2.append(current_flow)
                for _ in range(flow_span - 1):
                    header_row2.append("")
                # Tipos dentro de este flujo
                for m in range(k, l):
                    header_row3.append(columns[m][2])
                k = l
            i = j

        # Escribir CSV
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header_row1)
            writer.writerow(header_row2)
            writer.writerow(header_row3)
            writer.writerows(rows)

    def _build_columns(
        self,
        config: CountConfig,
        resolver: VehicleClassesResolver,
    ) -> list[tuple[str, str, str]]:
        access_order = Access.display_order()
        # Agrupar flujos por acceso
        grouped: dict[str, list[CountGroupConfig]] = {}
        for g in config.groups:
            grouped.setdefault(g.access, []).append(g)

        columns: list[tuple[str, str, str]] = []
        for acc in access_order:
            if acc not in grouped:
                continue
            for flow in grouped[acc]:
                types = resolver.resolve_group_include(flow.include)
                # Si include es "all" o no se pudo resolver, usar exclude como base negativa
                if not types:
                    types = resolver.get_all_types()
                excluded = resolver.resolve_group_exclude(flow.exclude)
                for vt in types:
                    if vt not in excluded:
                        columns.append((acc, flow.name, vt))
        return columns
