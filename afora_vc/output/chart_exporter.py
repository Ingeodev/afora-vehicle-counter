from __future__ import annotations

import json
import os
from typing import Protocol

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from ..domain.counting.entities import CountConfig
from ..domain.counting.services import IntervalGrouper, VehicleClassesResolver


# Paleta categorica para tipos de vehiculo
_VEHICLE_COLORS: dict[str, str] = {
    "car": "#1f77b4",
    "motorcycle": "#ff7f0e",
    "bus": "#2ca02c",
    "truck": "#d62728",
    "bicycle": "#9467bd",
    "van": "#8c564b",
    "pickup": "#e377c2",
    "trailer": "#7f7f7f",
    "unknown": "#bcbd22",
}


class ChartExporterStrategy(Protocol):
    """Strategy para exportar graficos estadisticos."""

    def write(
        self,
        data: dict[tuple[str, str, str, str], int],
        config: CountConfig,
        resolver: VehicleClassesResolver,
        path: str,
        date: str = "",
    ) -> None: ...


class StackedBarChartExporter:
    """
    Genera una imagen PNG con graficos de barras apiladas,
    uno por flujo, ordenados verticalmente.

    Cada grafico muestra:
      - Eje X: intervalos de tiempo
      - Eje Y: cantidad de vehiculos
      - Barras apiladas por tipo de vehiculo (color categorizado)
      - Titulo: nombre del flujo + fecha
    """

    _FALLBACK_COLORS = [
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]

    def write(
        self,
        data: dict[tuple[str, str, str, str], int],
        config: CountConfig,
        resolver: VehicleClassesResolver,
        path: str,
        date: str = "",
    ) -> None:
        flows = config.groups
        if not flows:
            return

        interval_grouper = IntervalGrouper(config.interval_minutes)
        intervals = interval_grouper.all_intervals()

        n_flows = len(flows)
        fig_height = max(400, 250 * n_flows)

        fig = make_subplots(
            rows=n_flows,
            cols=1,
            subplot_titles=[self._title(g.name, date) for g in flows],
            vertical_spacing=0.06,
            shared_xaxes=False,
        )

        legend_added: set[str] = set()

        for i, flow in enumerate(flows, 1):
            types = resolver.resolve_group_include(flow.include)
            if not types:
                types = resolver.get_all_types()
            excluded = resolver.resolve_group_exclude(flow.exclude)
            visible_types = [t for t in types if t not in excluded]

            for vt in visible_types:
                y = [
                    data.get((flow.access, flow.name, vt, inv), 0)
                    for inv in intervals
                ]
                show_legend = vt not in legend_added
                if show_legend:
                    legend_added.add(vt)
                color = _VEHICLE_COLORS.get(
                    vt,
                    self._FALLBACK_COLORS[
                        hash(vt) % len(self._FALLBACK_COLORS)
                    ],
                )
                fig.add_trace(
                    go.Bar(
                        name=vt,
                        x=intervals,
                        y=y,
                        marker_color=color,
                        legendgroup=vt,
                        showlegend=show_legend,
                    ),
                    row=i,
                    col=1,
                )

            fig.update_xaxes(
                tickmode="array",
                tickvals=self._every_nth(intervals, 4),
                tickangle=45,
                row=i,
                col=1,
            )
            fig.update_yaxes(title_text="Cantidad", row=i, col=1)

        fig.update_layout(
            barmode="stack",
            height=fig_height,
            title_text=f"Conteo por Flujo - {date}" if date else "Conteo por Flujo",
            legend_title_text="Tipo",
            hovermode="x unified",
            margin=dict(l=60, r=30, t=60, b=40),
        )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pio.write_image(fig, path, format="png", width=1200, height=fig_height)

    @staticmethod
    def _title(name: str, date: str) -> str:
        return f"{name} - {date}" if date else name

    @staticmethod
    def _every_nth(items: list[str], n: int) -> list[str]:
        return [items[i] for i in range(0, len(items), n)]


class ChartJsExporter:
    """
    Genera una pagina HTML autonoma con graficos de barras apiladas
    usando Chart.js (vía CDN), uno por flujo.

    Cada grafico usa:
      - Eje X: intervalos de tiempo
      - Eje Y: cantidad de vehiculos
      - Barras apiladas por tipo de vehiculo (color categorizado)
      - Titulo: nombre del flujo + fecha
    """

    _FALLBACK_COLORS = [
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]

    _HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Conteo por Flujo - {date}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f7fa; padding: 24px; color: #333; }}
h1 {{ text-align: center; margin-bottom: 32px; font-size: 1.6rem; color: #1a1a2e; }}
.chart-wrapper {{ background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                 padding: 20px; margin-bottom: 32px; }}
.chart-wrapper h2 {{ margin-bottom: 16px; font-size: 1.1rem; color: #1a1a2e;
                     border-left: 4px solid #4a6cf7; padding-left: 12px; }}
canvas {{ max-height: 400px; }}
</style>
</head>
<body>
<h1>Conteo por Flujo - {date}</h1>
<div id="charts">{charts_html}</div>
<script>
const DATA = {json_data};
DATA.charts.forEach((chart, i) => {{
    const ctx = document.getElementById('chart-' + i).getContext('2d');
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: chart.labels,
            datasets: chart.datasets,
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ intersect: false, mode: 'index' }},
            scales: {{
                x: {{ stacked: true, title: {{ display: true, text: 'Intervalo' }} }},
                y: {{ stacked: true, title: {{ display: true, text: 'Cantidad' }} }},
            }},
            plugins: {{
                legend: {{ position: 'bottom' }},
                tooltip: {{ callbacks: {{ footer: (items) => 'Total: ' + items.reduce((s, i) => s + i.parsed.y, 0) }} }},
            }},
        }},
    }});
}});
</script>
</body>
</html>"""

    _CHART_HTML = """<div class="chart-wrapper">
<h2>{title}</h2>
<div style="position:relative;height:350px;"><canvas id="chart-{idx}"></canvas></div>
</div>"""

    def write(
        self,
        data: dict[tuple[str, str, str, str], int],
        config: CountConfig,
        resolver: VehicleClassesResolver,
        path: str,
        date: str = "",
    ) -> None:
        flows = config.groups
        if not flows:
            return

        interval_grouper = IntervalGrouper(config.interval_minutes)
        intervals = interval_grouper.all_intervals()

        charts_data: list[dict] = []
        charts_html_parts: list[str] = []

        for idx, flow in enumerate(flows):
            types = resolver.resolve_group_include(flow.include)
            if not types:
                types = resolver.get_all_types()
            excluded = resolver.resolve_group_exclude(flow.exclude)
            visible_types = [t for t in types if t not in excluded]

            datasets: list[dict] = []
            for vt in visible_types:
                y = [
                    data.get((flow.access, flow.name, vt, inv), 0)
                    for inv in intervals
                ]
                color = _VEHICLE_COLORS.get(
                    vt,
                    self._FALLBACK_COLORS[hash(vt) % len(self._FALLBACK_COLORS)],
                )
                datasets.append({
                    "label": vt,
                    "data": y,
                    "backgroundColor": color,
                })

            title = f"{flow.name} - {date}" if date else flow.name
            charts_data.append({
                "title": title,
                "labels": intervals,
                "datasets": datasets,
            })
            charts_html_parts.append(
                self._CHART_HTML.format(title=title, idx=idx)
            )

        html = self._HTML_TEMPLATE.format(
            date=date or "Sin fecha",
            charts_html="\n".join(charts_html_parts),
            json_data=json.dumps({"charts": charts_data}, ensure_ascii=False),
        )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
