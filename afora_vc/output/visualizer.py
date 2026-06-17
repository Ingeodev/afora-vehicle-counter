from typing import Sequence, Callable

import cv2
import numpy as np

from ..domain.schemas import Detection, ZoneConfig


ZONE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 128, 0),
    (128, 255, 0),
]


class Visualizer:
    def __init__(
        self,
        box_color: tuple[int, int, int] = (0, 255, 0),
        box_color_with_zone: tuple[int, int, int] = (0, 255, 0),
        box_color_no_zone: tuple[int, int, int] = (255, 0, 0),
        trajectory_color: tuple[int, int, int] = (255, 0, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.box_color = box_color
        self.box_color_with_zone = box_color_with_zone
        self.box_color_no_zone = box_color_no_zone
        self.trajectory_color = trajectory_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_zones(
        self,
        frame: np.ndarray,
        zones: Sequence[ZoneConfig],
        alpha: float = 0.3,
        draw_labels: bool = True,
    ) -> np.ndarray:
        overlay = frame.copy()
        for i, zone in enumerate(zones):
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            cv2.fillPoly(overlay, [zone.points], color)
            cv2.polylines(frame, [zone.points], True, color, self.line_thickness)
            if draw_labels:
                centroid = zone.points.mean(axis=0).astype(int)
                cv2.putText(
                    frame, zone.label, tuple(centroid), self.font, 1.0, (255, 255, 255), 2
                )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        zones_visited: list[str] | None = None,
        trajectory: np.ndarray | None = None,
    ) -> np.ndarray:
        bbox = detection.bbox
        has_zones = zones_visited and len(zones_visited) > 0
        color = self.box_color_with_zone if has_zones else self.box_color_no_zone

        if detection.mask_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [detection.mask_polygon], color)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.polylines(frame, [detection.mask_polygon], True, color, 2)
            cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 1)
        else:
            cv2.rectangle(
                frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, self.line_thickness
            )

        cx, cy = detection.center
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        label = f"{detection.class_name}-{detection.track_id}"
        cv2.putText(
            frame, label, (bbox.x1, bbox.y1 - 5), self.font, self.font_scale, self.text_color, 2
        )

        if has_zones:
            zones_text = f"Zonas: {','.join(zones_visited)}"
            cv2.putText(
                frame, zones_text, (bbox.x1, bbox.y2 + 15),
                self.font, self.font_scale * 0.8, self.box_color_with_zone, 1,
            )

        if trajectory is not None and len(trajectory) > 0:
            cv2.polylines(frame, [trajectory], False, self.trajectory_color, self.line_thickness)

        return frame

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        get_zones: Callable[[int], list[str] | None] | None = None,
        get_trajectory: Callable[[int], np.ndarray | None] | None = None,
    ) -> np.ndarray:
        for det in detections:
            zones = get_zones(det.track_id) if get_zones else None
            trajectory = get_trajectory(det.track_id) if get_trajectory else None
            self.draw_detection(frame, det, zones, trajectory)
        return frame

    def draw_stats(
        self,
        frame: np.ndarray,
        stats: dict[str, int | str],
        position: tuple[int, int] = (10, 30),
        line_height: int = 30,
    ) -> np.ndarray:
        x, y = position
        for label, value in stats.items():
            text = f"{label}: {value}"
            cv2.putText(frame, text, (x, y), self.font, 0.7, self.text_color, 2)
            y += line_height
        return frame
