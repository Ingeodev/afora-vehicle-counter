from typing import Sequence, Callable

import cv2
import numpy as np

from ..domain.schemas import Detection, ZoneConfig


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


COUNTED_COLOR = _hex_to_bgr("#2EFFD5")
UNCOUNTED_COLOR = _hex_to_bgr("#0F1667")

ZONE_COLORS = [
    _hex_to_bgr("#0DC9F8"),
    _hex_to_bgr("#F92815"),
    _hex_to_bgr("#79F207"),
    _hex_to_bgr("#2809F1"),
    _hex_to_bgr("#BE77CF"),
    _hex_to_bgr("#FEB643"),
    _hex_to_bgr("#18FB7E"),
    _hex_to_bgr("#F410BB"),
    _hex_to_bgr("#3D69C7"),
    _hex_to_bgr("#77D4BD"),
    _hex_to_bgr("#BC6D4E"),
    _hex_to_bgr("#50B955"),
    _hex_to_bgr("#A31DFC"),
    _hex_to_bgr("#08F724"),
    _hex_to_bgr("#B5DE68"),
    _hex_to_bgr("#F74A81"),
    _hex_to_bgr("#30FDD7"),
    _hex_to_bgr("#FB56F5"),
    _hex_to_bgr("#F96915"),
    _hex_to_bgr("#7839C0"),
]


class Visualizer:
    def __init__(
        self,
        trajectory_color: tuple[int, int, int] = (255, 0, 0),
        text_color: tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.trajectory_color = trajectory_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _get_color(counted: bool) -> tuple[int, int, int]:
        return COUNTED_COLOR if counted else UNCOUNTED_COLOR

    @staticmethod
    def _get_alpha(num_zones: int) -> float:
        if num_zones == 0:
            return 0.6
        elif num_zones == 1:
            return 0.8
        return 1.0

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
        counted = bool(zones_visited)
        color = self._get_color(counted)
        num_zones = len(zones_visited) if zones_visited else 0
        alpha = self._get_alpha(num_zones)

        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        sub_frame = frame[max(y1, 0):y2, max(x1, 0):x2]
        if sub_frame.size > 0:
            sub_overlay = sub_frame.copy()
            cv2.rectangle(sub_overlay, (0, 0), (sub_frame.shape[1], sub_frame.shape[0]), color, cv2.FILLED)
            blended = cv2.addWeighted(sub_overlay, alpha * 0.35, sub_frame, 1 - alpha * 0.35, 0)
            sub_frame[:] = blended
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

        cx, cy = detection.center
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        label = f"{detection.class_name}-{detection.track_id}"
        cv2.putText(
            frame, label, (x1, y1 - 5), self.font, self.font_scale, color, 3
        )

        if zones_visited:
            zones_text = ",".join(zones_visited)
            text_size = cv2.getTextSize(zones_text, self.font, self.font_scale * 2, 2)[0]
            text_x = (x1 + x2 - text_size[0]) // 2
            text_y = y1 + text_size[1] + 4
            cv2.putText(
                frame, zones_text, (text_x, text_y),
                self.font, self.font_scale * 2, color, 3,
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
        position: tuple[int, int] | None = None,
        line_height: int = 30,
    ) -> np.ndarray:
        if position is None:
            h = frame.shape[0]
            position = (10, h - 60)
        x, y = position
        for label, value in stats.items():
            text = f"{label}: {value}"
            cv2.putText(frame, text, (x, y), self.font, 0.7, self.text_color, 2)
            y += line_height
        return frame
