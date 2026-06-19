from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import yaml

from ..domain.enums import DeviceType, DetectorBackend, TrackerBackend, PreprocessMode, DetectorStrategy, PipelineMode
from ..utils.device import detect_device


DEFAULT_VEHICLE_CLASSES = {
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


@dataclass
class DetectorConfig:
    backend: DetectorBackend = DetectorBackend.YOLO
    model_path: str = "models/yolov8s.pt"
    device: DeviceType = field(default_factory=detect_device)
    confidence_threshold: float = 0.5
    strategy: DetectorStrategy = DetectorStrategy.BOX
    class_thresholds: dict[str, float] | None = None
    vehicle_classes: dict[int, str] = field(
        default_factory=lambda: DEFAULT_VEHICLE_CLASSES.copy()
    )
    half_precision: bool = True


@dataclass
class TrackerConfig:
    backend: TrackerBackend = TrackerBackend.BOTSORT
    config_path: str | None = None
    max_age: int = 30
    min_hits: int = 3


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
            output=OutputConfig(**output_data),
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
