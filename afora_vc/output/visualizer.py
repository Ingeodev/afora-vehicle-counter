from typing import Sequence, Callable

import cv2
import numpy as np

from ..domain.schemas import Detection, ZoneConfig
from ..application.config import VisualizationConfig


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


COUNTED_COLOR = _hex_to_bgr("#2EFFD5")
UNCOUNTED_COLOR = _hex_to_bgr("#0F1667")
DEFAULT_CENTER_COLOR = (0, 255, 255)  # Amarillo.

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

# Colores por defecto por clase si no se configuran en el YAML.
DEFAULT_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": _hex_to_bgr("#FFFF00"),         # Amarillo
    "truck": _hex_to_bgr("#FF0000"),       # Rojo puro
    "bus": _hex_to_bgr("#FFFFFF"),         # Blanco
    "motorcycle": _hex_to_bgr("#FF00FF"),  # Magenta
    "bicycle": _hex_to_bgr("#00FF00"),     # Verde lima
    "person": _hex_to_bgr("#00FFFF"),      # Cian
}


class Visualizer:
    def __init__(
        self,
        config: VisualizationConfig | None = None,
    ):
        self.cfg = config or VisualizationConfig()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_class_color(self, class_name: str) -> tuple[int, int, int]:
        """Color por clase de vehiculo."""
        colors = self.cfg.class_colors or DEFAULT_CLASS_COLORS
        return colors.get(class_name, (255, 255, 255))

    def _get_color(self, counted: bool) -> tuple[int, int, int]:
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
        alpha: float | None = None,
        draw_labels: bool | None = None,
    ) -> np.ndarray:
        mode = self.cfg.zone_render_mode
        a = alpha if alpha is not None else self.cfg.zone_alpha
        labels = draw_labels if draw_labels is not None else self.cfg.draw_zone_labels

        overlay = frame.copy()
        for i, zone in enumerate(zones):
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            if mode in ("fill", "both"):
                cv2.fillPoly(overlay, [zone.points], color)
            if mode in ("outline", "both"):
                cv2.polylines(frame, [zone.points], True, color, 2)
            if labels:
                centroid = zone.points.mean(axis=0).astype(int)
                cv2.putText(
                    frame, zone.label, tuple(centroid), self.font, 1.0, (255, 255, 255), 2
                )
        if mode in ("fill", "both"):
            cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)
        return frame

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        zones_visited: list[str] | None = None,
        trajectory: np.ndarray | None = None,
    ) -> np.ndarray:
        bbox = detection.bbox
        class_color = self._get_class_color(detection.class_name)
        counted = bool(zones_visited)
        color = self._get_color(counted)
        num_zones = len(zones_visited) if zones_visited else 0
        alpha = self._get_alpha(num_zones)

        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        # ── BBox ──────────────────────────────────────────
        if self.cfg.draw_bbox:
            sub_frame = frame[max(y1, 0):y2, max(x1, 0):x2]
            if sub_frame.size > 0:
                sub_overlay = sub_frame.copy()
                cv2.rectangle(sub_overlay, (0, 0), (sub_frame.shape[1], sub_frame.shape[0]), color, cv2.FILLED)
                blended = cv2.addWeighted(sub_overlay, alpha * 0.35, sub_frame, 1 - alpha * 0.35, 0)
                sub_frame[:] = blended
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.cfg.bbox_thickness)

        # ── Centro ────────────────────────────────────────
        if self.cfg.draw_center:
            cx, cy = detection.center
            center_color = self.cfg.center_color or DEFAULT_CENTER_COLOR
            cv2.circle(frame, (cx, cy), self.cfg.center_radius, center_color, -1)

        # ── Class label ──────────────────────────────────
        if self.cfg.draw_class_label:
            label = f"{detection.class_name}-{detection.track_id}"
            anchor = self.cfg.label_anchor
            scale = self.cfg.label_scale
            thick = self.cfg.label_thickness
            if anchor == "above_bbox":
                pos = (x1, y1 - 5)
            elif anchor == "below_bbox":
                text_h = int(scale * 20) + 8
                pos = (x1, y2 + text_h)
            elif anchor == "center":
                cx, cy = detection.center
                pos = (cx, cy)
            else:
                pos = (x1, y1 - 5)
            cv2.putText(
                frame, label, pos, self.font, scale, class_color, thick
            )

        # ── Zones visited (texto sobre el bbox) ──────────
        if zones_visited and self.cfg.draw_class_label:
            zones_text = ",".join(zones_visited)
            text_size = cv2.getTextSize(zones_text, self.font, scale * 2, 2)[0]
            text_x = (x1 + x2 - text_size[0]) // 2
            text_y = y1 + text_size[1] + 4
            cv2.putText(
                frame, zones_text, (text_x, text_y),
                self.font, scale * 2, class_color, 3,
            )

        # ── Trayectoria ──────────────────────────────────
        if self.cfg.draw_trajectory and trajectory is not None and len(trajectory) > 0:
            traj_color = class_color if self.cfg.trajectory_class_colors else self.cfg.trajectory_color if hasattr(self.cfg, 'trajectory_color') else (255, 0, 0)
            cv2.polylines(
                frame, [trajectory], False,
                traj_color, self.cfg.trajectory_thickness,
            )

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
        if not self.cfg.draw_stats:
            return frame
        if position is None:
            h = frame.shape[0]
            position = (10, h - 60)
        x, y = position
        text_color = (255, 255, 255)
        for label, value in stats.items():
            text = f"{label}: {value}"
            cv2.putText(frame, text, (x, y), self.font, 0.7, text_color, 2)
            y += line_height
        return frame
