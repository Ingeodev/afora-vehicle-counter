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

DEFAULT_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": _hex_to_bgr("#FF1E8C"),         # Rosa fuerte
    "truck": _hex_to_bgr("#FF8F3E"),       # Naranja intenso
    "bus": _hex_to_bgr("#FFE600"),         # Amarillo
    "motorcycle": _hex_to_bgr("#00D8FF"),  # Cian
    "bicycle": _hex_to_bgr("#B14DFF"),     # Violeta claro
    "person": _hex_to_bgr("#FF3B30"),      # Rojo
}


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

    def _get_color(self, counted: bool) -> tuple[int, int, int]:
        return COUNTED_COLOR if counted else UNCOUNTED_COLOR

    def _get_class_color(self, class_name: str) -> tuple[int, int, int]:
        return DEFAULT_CLASS_COLORS.get(class_name, (255, 255, 255))

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
        counted = bool(zones_visited)
        color = self._get_color(counted)

        cx, cy = detection.center
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        # Label estático pequeño arriba del punto central.
        label = detection.class_name
        cv2.putText(
            frame, label, (cx, cy - 10), self.font, 0.4, (255, 255, 255), 1
        )

        if trajectory is not None and len(trajectory) > 0:
            traj_color = self._get_class_color(detection.class_name)
            cv2.polylines(frame, [trajectory], False, traj_color, self.line_thickness)

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
