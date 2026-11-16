from typing import Sequence, Callable

import cv2
import numpy as np

from ..domain.schemas import Detection, ZoneConfig
from ..application.config import VisualizationConfig, _LABEL_FONT_SIZES


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


COUNTED_COLOR = _hex_to_bgr("#2EFFD5")
UNCOUNTED_COLOR = _hex_to_bgr("#0F1667")
DEFAULT_CENTER_COLOR = (255, 255, 255)  # Blanco fijo.

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


def _render_label(template: str, detection: Detection, zones: list[str] | None) -> str:
    """Renderiza un template de label con las variables del detection."""
    text = template
    text = text.replace("$id", str(detection.track_id))
    text = text.replace("$class", detection.class_name)
    text = text.replace("$zones", ",".join(zones) if zones else "")
    return text


class Visualizer:
    def __init__(
        self,
        config: VisualizationConfig | None = None,
    ):
        self.cfg = config or VisualizationConfig()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_class_color(self, class_name: str) -> tuple[int, int, int]:
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

    def _draw_label(self, frame: np.ndarray, text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
        """Dibuja texto con fondo semitransparente multilinea."""
        scale, thick = _LABEL_FONT_SIZES.get(self.cfg.label_font_size, _LABEL_FONT_SIZES["s"])
        lines = text.split("\n")

        # Calcular dimensiones del bloque de texto.
        line_h = int(scale * 25) + 4
        max_w = 0
        for line in lines:
            w, _ = cv2.getTextSize(line, self.font, scale, thick)[0]
            max_w = max(max_w, w)

        total_h = line_h * len(lines)
        bg_padding = 4
        x0, y0 = pos[0] - bg_padding, pos[1] - total_h - bg_padding + line_h - 4

        # Fondo semitransparente.
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + max_w + bg_padding * 2, y0 + total_h + bg_padding), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.cfg.label_bg_alpha, frame, 1 - self.cfg.label_bg_alpha, 0, frame)

        # Texto.
        for i, line in enumerate(lines):
            y = y0 + line_h * (i + 1) - 2
            cv2.putText(frame, line, (x0 + bg_padding, y), self.font, scale, color, thick)

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

        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        cx, cy = detection.center

        # ── BBox ──────────────────────────────────────────
        if self.cfg.draw_bbox:
            sub_frame = frame[max(y1, 0):y2, max(x1, 0):x2]
            if sub_frame.size > 0:
                sub_overlay = sub_frame.copy()
                cv2.rectangle(sub_overlay, (0, 0), (sub_frame.shape[1], sub_frame.shape[0]), color, cv2.FILLED)
                blended = cv2.addWeighted(sub_overlay, self.cfg.bbox_alpha * 0.35, sub_frame, 1 - self.cfg.bbox_alpha * 0.35, 0)
                sub_frame[:] = blended
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.cfg.bbox_thickness)

        # ── Centro ────────────────────────────────────────
        if self.cfg.draw_center:
            cv2.circle(frame, (cx, cy), self.cfg.center_radius, DEFAULT_CENTER_COLOR, -1)

        # ── Label con template ───────────────────────────
        if self.cfg.draw_class_label:
            label_text = _render_label(self.cfg.label_template, detection, zones_visited)
            ref = self.cfg.label_reference
            anchor = self.cfg.label_anchor
            off = self.cfg.label_offset

            if ref == "center":
                base_x, base_y = cx, cy
            else:
                # bbox: punto medio del borde superior/inferior.
                base_x = (x1 + x2) // 2
                base_y = y1 if anchor in ("above",) else y2

            if ref == "center":
                if anchor == "above":
                    pos = (base_x, base_y - off)
                elif anchor == "below":
                    pos = (base_x, base_y + off)
                elif anchor == "left":
                    pos = (base_x - off, base_y)
                else:
                    pos = (base_x + off, base_y)
            else:
                # bbox: above/below del borde.
                if anchor == "above":
                    pos = (x1, y1 - off)
                elif anchor == "below":
                    pos = (x1, y2 + off + 16)
                elif anchor == "left":
                    pos = (x1 - off, y1)
                else:
                    pos = (x2 + off, y1)

            self._draw_label(frame, label_text, pos, (255, 255, 255))

        # ── Trayectoria ──────────────────────────────────
        if self.cfg.draw_trajectory and trajectory is not None and len(trajectory) > 0:
            traj_color = class_color if self.cfg.trajectory_class_colors else (255, 0, 0)
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
