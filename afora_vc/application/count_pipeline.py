from __future__ import annotations

import os

from ..domain.counting.entities import CountConfig, CountGroupConfig
from ..domain.counting.services import (
    AbsoluteTimeConverter,
    VehiclePathBuilder,
    FlowMatcher,
    InterventionFileExporter,
    IntervalGrouper,
    VehicleClassesResolver,
)


def export_chart(
    data: dict[tuple[str, str, str, str], int],
    config: CountConfig,
    resolver: VehicleClassesResolver,
    output_dir: str,
    date: str = "",
) -> str | None:
    """Exporta grafico segun config.graph ("none", "image", "web").
    Retorna la ruta del archivo generado o None."""
    mode = config.graph
    if mode == "none":
        return None

    if mode == "image":
        from ..output.chart_exporter import StackedBarChartExporter
        path = os.path.join(output_dir, "aggregated_chart.png")
        StackedBarChartExporter().write(data, config, resolver, path, date=date)
        return path

    if mode == "web":
        from ..output.chart_exporter import ChartJsExporter
        path = os.path.join(output_dir, "aggregated_chart.html")
        ChartJsExporter().write(data, config, resolver, path, date=date)
        return path

    return None


class GroupPipeline:
    """Pipeline que encadena las agrupaciones jerarquicamente."""

    def __init__(
        self,
        config: CountConfig,
        resolver: VehicleClassesResolver | None = None,
    ):
        self._config = config
        self._resolver = resolver or VehicleClassesResolver()
        self._converter = AbsoluteTimeConverter()
        self._path_builder = VehiclePathBuilder()
        self._flow_matcher = FlowMatcher(config.groups, self._resolver)
        self._intervention = InterventionFileExporter()
        self._interval_grouper = IntervalGrouper(config.interval_minutes)

    def execute(
        self,
        folder_path: str,
        output_dir: str | None = None,
    ) -> dict[tuple[str, str, str, str], int]:
        out = output_dir or folder_path

        # Level 1: tiempo absoluto + composite ID
        entries = self._converter.process(folder_path)

        # Level 2: construir paths por vehiculo
        paths = self._path_builder.process(entries)

        # Level 3: matchear con flujos
        paths = self._flow_matcher.process(paths)

        # Level 4: archivo de interventoria + filtrar matched
        matched = self._intervention.process(paths, out)
        matched = InterventionFileExporter.enrich_matched_with_access(
            matched, self._config.groups
        )

        # Level 5: agrupar por intervalos
        grouped = self._interval_grouper.process(matched)

        return grouped
