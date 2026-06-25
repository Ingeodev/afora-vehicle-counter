import os
from pathlib import Path

import pytest

from afora_vc.domain.counting.entities import CountConfig, CountGroupConfig
from afora_vc.domain.counting.services import VehicleClassesResolver
from afora_vc.output.chart_exporter import (
    StackedBarChartExporter,
    ChartJsExporter,
)
from afora_vc.application.count_pipeline import export_chart


@pytest.fixture
def sample_config() -> CountConfig:
    return CountConfig(
        interval_minutes=15,
        graph="none",
        groups=[
            CountGroupConfig(
                name="Flujo Norte",
                include="all",
                exclude="none",
                access="norte",
                zone_paths=[["a", "b"]],
            ),
            CountGroupConfig(
                name="Flujo Sur",
                include="all",
                exclude="none",
                access="sur",
                zone_paths=[["c", "d"]],
            ),
        ],
    )


@pytest.fixture
def resolver() -> VehicleClassesResolver:
    return VehicleClassesResolver()


@pytest.fixture
def sample_data() -> dict[tuple[str, str, str, str], int]:
    return {
        ("norte", "Flujo Norte", "car", "00:00-00:15"): 5,
        ("norte", "Flujo Norte", "truck", "00:00-00:15"): 2,
        ("norte", "Flujo Norte", "car", "00:15-00:30"): 3,
        ("sur", "Flujo Sur", "bus", "00:00-00:15"): 1,
        ("sur", "Flujo Sur", "car", "00:00-00:15"): 4,
    }


class TestStackedBarChartExporter:
    def test_write_png(self, sample_config, sample_data, resolver, tmp_path):
        exporter = StackedBarChartExporter()
        out_path = str(tmp_path / "chart.png")
        exporter.write(sample_data, sample_config, resolver, out_path, date="2026-06-25")
        assert os.path.exists(out_path)
        assert out_path.endswith(".png")

    def test_write_empty_data(self, sample_config, resolver, tmp_path):
        exporter = StackedBarChartExporter()
        out_path = str(tmp_path / "empty_chart.png")
        exporter.write({}, sample_config, resolver, out_path)
        assert os.path.exists(out_path)

    def test_write_no_flows(self, tmp_path, resolver):
        config = CountConfig(interval_minutes=15, groups=[], graph="none")
        exporter = StackedBarChartExporter()
        out_path = str(tmp_path / "no_flows.png")
        exporter.write({}, config, resolver, out_path)
        assert not os.path.exists(out_path)


class TestChartJsExporter:
    def test_write_html(self, sample_config, sample_data, resolver, tmp_path):
        exporter = ChartJsExporter()
        out_path = str(tmp_path / "chart.html")
        exporter.write(sample_data, sample_config, resolver, out_path, date="2026-06-25")
        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Conteo por Flujo - 2026-06-25" in content
        assert "chart.js" in content
        assert "Flujo Norte" in content
        assert "Flujo Sur" in content

    def test_write_empty_data(self, sample_config, resolver, tmp_path):
        exporter = ChartJsExporter()
        out_path = str(tmp_path / "empty_chart.html")
        exporter.write({}, sample_config, resolver, out_path, date="2026-06-25")
        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Flujo Norte" in content
        assert "Flujo Sur" in content

    def test_write_no_flows(self, tmp_path, resolver):
        config = CountConfig(interval_minutes=15, groups=[], graph="none")
        exporter = ChartJsExporter()
        out_path = str(tmp_path / "no_flows.html")
        exporter.write({}, config, resolver, out_path)
        assert not os.path.exists(out_path)

    def test_html_contains_json_data(self, sample_config, sample_data, resolver, tmp_path):
        exporter = ChartJsExporter()
        out_path = str(tmp_path / "chart_data.html")
        exporter.write(sample_data, sample_config, resolver, out_path, date="2026-06-25")
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "car" in content
        assert "truck" in content
        assert "bus" in content


class TestExportChartDispatcher:
    def test_graph_none_returns_none(self, sample_config, sample_data, resolver, tmp_path):
        sample_config.graph = "none"
        result = export_chart(sample_data, sample_config, resolver, str(tmp_path), date="test")
        assert result is None

    def test_graph_image_dispatches_stacked_bar(self, sample_config, sample_data, resolver, tmp_path):
        sample_config.graph = "image"
        result = export_chart(sample_data, sample_config, resolver, str(tmp_path), date="test")
        assert result is not None
        assert result.endswith("aggregated_chart.png")
        assert os.path.exists(result)

    def test_graph_web_dispatches_chartjs(self, sample_config, sample_data, resolver, tmp_path):
        sample_config.graph = "web"
        result = export_chart(sample_data, sample_config, resolver, str(tmp_path), date="test")
        assert result is not None
        assert result.endswith("aggregated_chart.html")
        assert os.path.exists(result)
