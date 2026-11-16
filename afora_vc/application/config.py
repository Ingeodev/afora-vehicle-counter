from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import yaml

from ..domain.enums import DeviceType, DetectorBackend, TrackerBackend, PreprocessMode, DetectorStrategy, PipelineMode
from ..utils.device import detect_device


DEFAULT_VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class BatchConfig:
    enabled: bool = False
    batch_size: int = 4
    adaptive: bool = True
    max_batch_size: int = 16
    target_latency_ms: float = 100.0
    queue_maxsize: int = 4


@dataclass
class VideoConfig:
    reduce_factor: int = 1
    max_minutes: float | None = None
    skip_rate: int = 1
    output_fps: float = 15.0
    codec: str = "avc1"
    # Calidad/velocidad del encoder H.264. Los videos de salida son para
    # verificacion: crf alto y preset rapido pesan mucho menos.
    crf: int = 30
    preset: str = "veryfast"


@dataclass
class DetectorConfig:
    backend: DetectorBackend = DetectorBackend.YOLO
    model_path: str = "models/yolov8s.pt"
    device: DeviceType = field(default_factory=detect_device)
    confidence_threshold: float = 0.5
    strategy: DetectorStrategy = DetectorStrategy.BOX
    class_thresholds: dict[str, float] | None = None
    vehicle_classes: dict[int, str] | None = None
    half_precision: bool = True


@dataclass
class TrackerConfig:
    backend: TrackerBackend = TrackerBackend.BOTSORT
    config_path: str | None = None
    max_age: int = 30
    min_hits: int = 3


@dataclass
class VisualizationConfig:
    """Configuracion granular de que y como renderizar en el video de salida."""
    # Que elementos renderizar.
    draw_bbox: bool = False
    draw_center: bool = True
    draw_trajectory: bool = True
    draw_class_label: bool = True
    draw_zones: bool = True
    draw_zone_labels: bool = True
    draw_stats: bool = False

    # Estilo del bbox.
    bbox_thickness: int = 2
    bbox_alpha: float = 0.15  # Transparencia del relleno.

    # Estilo del punto central.
    center_radius: int = 5
    center_color: tuple[int, int, int] | None = None  # None = amarillo por defecto.

    # Estilo de la trayectoria.
    trajectory_thickness: int = 6
    trajectory_max_points: int = 30
    trajectory_class_colors: bool = True  # Usar color por clase de vehiculo.

    # Colores por clase (BGR). Se usa si trajectory_class_colors = True.
    class_colors: dict[str, tuple[int, int, int]] = field(default_factory=lambda: {
        "car": (0, 165, 255),       # Naranja
        "truck": (0, 100, 255),     # Naranja oscuro
        "bus": (0, 255, 255),       # Amarillo
        "motorcycle": (255, 0, 0),  # Azul
        "bicycle": (0, 255, 0),     # Verde
        "person": (255, 255, 255),  # Blanco
    })

    # Label: template con variables $id, $class, $zones.
    # Ejemplo: "$id-$zones\n$class" → "42-ZA,ZB\ncar"
    label_template: str = "$id-$zones\n$class"
    # Referencia para posicionar: "bbox" o "center".
    label_reference: str = "center"
    # Anchor relativo a la referencia: "above", "below", "left", "right".
    label_anchor: str = "above"
    label_offset: int = 10  # Pixeles de separacion respecto a la referencia.
    # Tamano de fuente: "xs", "s", "m", "l".
    label_font_size: str = "m"
    label_thickness: int = 2
    label_bg_alpha: float = 0.5  # Transparencia del fondo del label.

    # Zonas: "fill" (poligono relleno), "outline" (solo borde), "both".
    zone_render_mode: str = "both"
    zone_alpha: float = 0.3
    # Como detectar si un vehiculo entro en una zona: "center" o "bbox".
    zone_detection_mode: str = "center"


# Mapas de tamaño de fuente a escala y grosor.
_LABEL_FONT_SIZES: dict[str, tuple[float, int]] = {
    "xs": (0.7, 2),
    "s": (1.0, 4),
    "m": (1.4, 4),
    "l": (2.0, 6),
}


@dataclass
class OutputConfig:
    output_folder: str = "./output"
    save_video: bool = True
    save_csv: bool = True
    draw_zones: bool = True
    draw_trajectories: bool = True
    max_trajectory_points: int = 20
    progress_interval: int = 100
    verbose: bool = True
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


@dataclass
class PreprocessConfig:
    mode: PreprocessMode = PreprocessMode.NONE
    deblur_mode: str = "night"


@dataclass
class PipelineConfig:
    pipeline_mode: PipelineMode = PipelineMode.SEQUENTIAL
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        pipeline_mode_str = data.get("pipeline_mode", "sequential")
        if isinstance(pipeline_mode_str, str):
            pipeline_mode = PipelineMode(pipeline_mode_str)
        else:
            pipeline_mode = pipeline_mode_str
        video_data = data.get("video", {})
        detector_data = data.get("detector", {})
        tracker_data = data.get("tracker", {})
        preprocess_data = data.get("preprocess", {})
        output_data = data.get("output", {})
        batch_data = data.get("batch", {})

        # Extract visualization sub-config if present.
        viz_data = output_data.pop("visualization", {})
        if viz_data:
            # class_colors en YAML es dict[str, str] con hex colors; convertir a BGR.
            raw_colors = viz_data.get("class_colors", {})
            if raw_colors:
                from ..output.visualizer import _hex_to_bgr
                viz_data["class_colors"] = {
                    k: _hex_to_bgr(v) if isinstance(v, str) else tuple(v)
                    for k, v in raw_colors.items()
                }
            visualization = VisualizationConfig(**viz_data)
        else:
            visualization = VisualizationConfig()

        # Convert string backend to enum
        if "backend" in detector_data and isinstance(detector_data["backend"], str):
            detector_data["backend"] = DetectorBackend(detector_data["backend"])
        if "backend" in tracker_data and isinstance(tracker_data["backend"], str):
            tracker_data["backend"] = TrackerBackend(tracker_data["backend"])
        if "mode" in preprocess_data and isinstance(preprocess_data["mode"], str):
            preprocess_data["mode"] = PreprocessMode(preprocess_data["mode"])
        if "strategy" in detector_data and isinstance(detector_data["strategy"], str):
            detector_data["strategy"] = DetectorStrategy(detector_data["strategy"])

        return cls(
            pipeline_mode=pipeline_mode,
            video=VideoConfig(**video_data),
            detector=DetectorConfig(**detector_data),
            tracker=TrackerConfig(**tracker_data),
            preprocess=PreprocessConfig(**preprocess_data),
            output=OutputConfig(**output_data, visualization=visualization),
            batch=BatchConfig(**batch_data),
        )

    def to_yaml(self, path: str) -> None:
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_mode": self.pipeline_mode.value,
            "video": {
                "reduce_factor": self.video.reduce_factor,
                "max_minutes": self.video.max_minutes,
                "skip_rate": self.video.skip_rate,
                "output_fps": self.video.output_fps,
                "codec": self.video.codec,
                "crf": self.video.crf,
                "preset": self.video.preset,
            },
            "detector": {
                "backend": self.detector.backend.value,
                "model_path": self.detector.model_path,
                "device": self.detector.device.value,
                "confidence_threshold": self.detector.confidence_threshold,
                "strategy": self.detector.strategy.value,
                "class_thresholds": self.detector.class_thresholds,
                "vehicle_classes": self.detector.vehicle_classes,
            },
            "tracker": {
                "backend": self.tracker.backend.value,
                "config_path": self.tracker.config_path,
                "max_age": self.tracker.max_age,
                "min_hits": self.tracker.min_hits,
            },
            "preprocess": {
                "mode": self.preprocess.mode.value,
                "deblur_mode": self.preprocess.deblur_mode,
            },
            "output": {
                "output_folder": self.output.output_folder,
                "save_video": self.output.save_video,
                "save_csv": self.output.save_csv,
                "draw_zones": self.output.draw_zones,
                "draw_trajectories": self.output.draw_trajectories,
                "max_trajectory_points": self.output.max_trajectory_points,
                "progress_interval": self.output.progress_interval,
                "verbose": self.output.verbose,
                "visualization": {
                    "draw_bbox": self.output.visualization.draw_bbox,
                    "draw_center": self.output.visualization.draw_center,
                    "draw_trajectory": self.output.visualization.draw_trajectory,
                    "draw_class_label": self.output.visualization.draw_class_label,
                    "draw_zones": self.output.visualization.draw_zones,
                    "draw_zone_labels": self.output.visualization.draw_zone_labels,
                    "draw_stats": self.output.visualization.draw_stats,
                    "bbox_thickness": self.output.visualization.bbox_thickness,
                    "bbox_alpha": self.output.visualization.bbox_alpha,
                    "center_radius": self.output.visualization.center_radius,
                    "trajectory_thickness": self.output.visualization.trajectory_thickness,
                    "trajectory_max_points": self.output.visualization.trajectory_max_points,
                    "trajectory_class_colors": self.output.visualization.trajectory_class_colors,
                    "label_template": self.output.visualization.label_template,
                    "label_reference": self.output.visualization.label_reference,
                    "label_anchor": self.output.visualization.label_anchor,
                    "label_offset": self.output.visualization.label_offset,
                    "label_font_size": self.output.visualization.label_font_size,
                    "label_thickness": self.output.visualization.label_thickness,
                    "label_bg_alpha": self.output.visualization.label_bg_alpha,
                    "zone_render_mode": self.output.visualization.zone_render_mode,
                    "zone_alpha": self.output.visualization.zone_alpha,
                    "zone_detection_mode": self.output.visualization.zone_detection_mode,
                },
            },
            "batch": {
                "enabled": self.batch.enabled,
                "batch_size": self.batch.batch_size,
                "adaptive": self.batch.adaptive,
                "max_batch_size": self.batch.max_batch_size,
                "queue_maxsize": self.batch.queue_maxsize,
            },
        }


def get_default_config() -> PipelineConfig:
    return PipelineConfig()
