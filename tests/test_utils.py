import numpy as np

from afora_vc.utils.geometry import (
    point_in_polygon,
    scale_points,
    calculate_center,
    polygon_area,
    polygon_centroid,
    line_intersection,
)
from afora_vc.utils.time_utils import (
    format_timestamp,
    calculate_exact_time,
    seconds_to_hms,
    parse_time,
)


class TestGeometry:
    def test_point_in_polygon_inside(self):
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        assert point_in_polygon((50, 50), polygon) is True

    def test_point_in_polygon_outside(self):
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        assert point_in_polygon((200, 200), polygon) is False

    def test_point_in_polygon_on_edge(self):
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        assert point_in_polygon((50, 0), polygon) is True

    def test_scale_points_immutable(self):
        points = np.array([[100, 200], [300, 400]], dtype=np.int32)
        scaled = scale_points(points, 0.5, 0.5)
        assert points[0][0] == 100
        assert scaled[0][0] == 50

    def test_scale_points_correct(self):
        points = np.array([[100, 200], [300, 400]], dtype=np.int32)
        scaled = scale_points(points, 2.0, 0.5)
        assert scaled[0][0] == 200
        assert scaled[0][1] == 100
        assert scaled[1][0] == 600
        assert scaled[1][1] == 200

    def test_calculate_center(self):
        cx, cy = calculate_center(100, 200, 300, 400)
        assert cx == 200
        assert cy == 300

    def test_polygon_area_square(self):
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        area = polygon_area(points)
        assert area == 10000.0

    def test_polygon_area_triangle(self):
        points = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.int32)
        area = polygon_area(points)
        assert area == 5000.0

    def test_polygon_centroid(self):
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        cx, cy = polygon_centroid(points)
        assert cx == 50
        assert cy == 50

    def test_line_intersection(self):
        result = line_intersection(
            (0, 0), (100, 100),
            (0, 100), (100, 0),
        )
        assert result is not None
        x, y = result
        assert abs(x - 50) < 0.001
        assert abs(y - 50) < 0.001

    def test_line_intersection_parallel(self):
        result = line_intersection(
            (0, 0), (100, 100),
            (0, 50), (100, 150),
        )
        assert result is None


class TestTimeUtils:
    def test_format_timestamp(self):
        assert format_timestamp(0) == "00:00"
        assert format_timestamp(60) == "01:00"
        assert format_timestamp(125.5) == "02:05"
        assert format_timestamp(3600) == "60:00"

    def test_calculate_exact_time_hh_mm(self):
        result = calculate_exact_time("14:30", 125)
        assert result == "14:32:05"

    def test_calculate_exact_time_hh_mm_ss(self):
        result = calculate_exact_time("14:30:00", 125)
        assert result == "14:32:05"

    def test_calculate_exact_time_wraps_hour(self):
        result = calculate_exact_time("14:30", 3600)
        assert result == "15:30:00"

    def test_calculate_exact_time_invalid(self):
        result = calculate_exact_time("invalid", 100)
        assert result is None

    def test_seconds_to_hms(self):
        assert seconds_to_hms(0) == "00:00:00"
        assert seconds_to_hms(3661) == "01:01:01"
        assert seconds_to_hms(86400) == "24:00:00"

    def test_parse_time_hh_mm(self):
        result = parse_time("14:30")
        assert result == (14, 30, 0)

    def test_parse_time_hh_mm_ss(self):
        result = parse_time("14:30:15")
        assert result == (14, 30, 15)

    def test_parse_time_invalid(self):
        assert parse_time("abc") is None
