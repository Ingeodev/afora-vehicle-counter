import cv2
import numpy as np


def point_in_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def scale_points(points: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = points.copy().astype(float)
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled.astype(np.int32)


def calculate_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def line_intersection(
    line1_start: tuple[int, int],
    line1_end: tuple[int, int],
    line2_start: tuple[int, int],
    line2_end: tuple[int, int],
) -> tuple[float, float] | None:
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)


def polygon_area(points: np.ndarray) -> float:
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def polygon_centroid(points: np.ndarray) -> tuple[int, int]:
    centroid = points.mean(axis=0).astype(int)
    return (int(centroid[0]), int(centroid[1]))
