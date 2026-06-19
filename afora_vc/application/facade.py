from pathlib import Path
from typing import Optional, Literal

import yaml

from ..domain.schemas import ProcessingResult
from ..domain.exceptions import (
    VideoNotFoundError,
    VideoOpenError,
    ConfigurationError,
    OutputDirectoryError,
    BackendNotRegisteredError,
)
from ..domain.enums import DeviceType, DetectorBackend, TrackerBackend, PreprocessMode, PipelineMode
from ..adapters import DetectorRegistry, TrackerRegistry
from .config import PipelineConfig
from .pipeline import Pipeline
from .pipelined_pipeline import PipelinedPipeline


def _build_pipeline(
    config: PipelineConfig,
    model_path: str,
    config_type: str,
) -> Pipeline | PipelinedPipeline:
    from ..adapters.preprocessors.noop import NoOpPreprocessor
    from ..adapters.preprocessors.deblurring import DeblurringPreprocessor

    # Create detector
    try:
        detector = DetectorRegistry.create(
            config.detector.backend.value,
            model_path=model_path,
            device=config.detector.device,
            vehicle_classes=config.detector.vehicle_classes,
            confidence_threshold=config.detector.confidence_threshold,
            class_thresholds=config.detector.class_thresholds,
            half_precision=config.detector.half_precision,
        )
    except ValueError as e:
        available = DetectorRegistry.available()
        raise BackendNotRegisteredError(
            config.detector.backend.value, available
        ) from e

    # Create tracker
    try:
        tracker = TrackerRegistry.create(
            config.tracker.backend.value,
            config_path=config.tracker.config_path,
            device=config.detector.device,
        )
    except ValueError as e:
        available = TrackerRegistry.available()
        raise BackendNotRegisteredError(config.tracker.backend.value, available) from e

    # Preprocessor
    if config.preprocess.mode == PreprocessMode.DEBLUR:
        preprocessor = DeblurringPreprocessor(mode=config.preprocess.deblur_mode)
    elif config.preprocess.mode == PreprocessMode.NIGHT_ENHANCE:
        preprocessor = DeblurringPreprocessor.create_night_enhance()
    else:
        preprocessor = NoOpPreprocessor()

    pipeline_cls = PipelinedPipeline if config.pipeline_mode == PipelineMode.PIPELINED else Pipeline
    return pipeline_cls(
        config=config,
        detector=detector,
        tracker=tracker,
        preprocessor=preprocessor,
    )


def process_video(
    video: str,
    zones: Optional[str] = None,
    mask: Optional[str] = None,
    output: str = "./output",
    base_time: str = "00:00",
    date: str = "2025-01-01",
    device: Optional[Literal["cpu", "cuda", "mps"]] = None,
    model: str = "models/yolov8s.pt",
    reduce: int = 1,
    max_minutes: Optional[float] = None,
    strategy: Literal["box", "seg"] = "box",
    night_enhance: bool = False,
    model_config: Optional[str] = None,
    deblurring: bool = False,
    no_video: bool = False,
    quiet: bool = False,
    backend: str = "yolo",
    tracker_backend: str = "botsort",
    batch_size: int = 1,
    output_name: Optional[str] = None,
    tracker_config: Optional[str] = None,
    pipeline_mode: str = "sequential",
    queue_maxsize: int = 4,
    profile: bool = False,
    profile_path: str | None = None,
) -> ProcessingResult:
    import os

    if not os.path.exists(video):
        raise VideoNotFoundError(video)

    if not os.path.exists(output):
        try:
            os.makedirs(output, exist_ok=True)
        except OSError as e:
            raise OutputDirectoryError(output, str(e))

    config = PipelineConfig()

    # Detector
    config.detector.backend = DetectorBackend(backend)
    config.detector.model_path = model
    if device:
        config.detector.device = DeviceType(device)

    # Tracker
    config.tracker.backend = TrackerBackend(tracker_backend)
    if tracker_config:
        config.tracker.config_path = tracker_config

    # Video
    config.video.reduce_factor = reduce
    config.video.max_minutes = max_minutes

    # Output
    config.output.save_video = not no_video
    config.output.verbose = not quiet
    config.output.output_folder = output

    # Batch
    config.batch.enabled = batch_size > 1
    config.batch.batch_size = batch_size
    config.batch.queue_maxsize = queue_maxsize

    # Pipeline mode
    config.pipeline_mode = PipelineMode(pipeline_mode)

    # Model config overrides
    if model_config:
        if not os.path.exists(model_config):
            raise ConfigurationError("Archivo de configuracion no encontrado", model_config)

        try:
            with open(model_config, "r") as f:
                model_params = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error de sintaxis YAML: {e}", model_config)

        if "default_threshold" in model_params:
            config.detector.confidence_threshold = float(model_params["default_threshold"])
        if "class_thresholds" in model_params:
            config.detector.class_thresholds = model_params["class_thresholds"]
        if "vehicle_classes" in model_params:
            config.detector.vehicle_classes = {
                int(k): str(v) for k, v in model_params["vehicle_classes"].items()
            }

    from ..utils.profiler import Profiler

    _profiler: Profiler | None = None
    if profile:
        _profiler = Profiler(csv_path=profile_path or f"{output}/profile.csv")

    pipeline = _build_pipeline(config, model, strategy)

    result = pipeline.process_video(
        video_path=video,
        zones_path=zones,
        mask_path=mask,
        output_folder=output,
        output_name=output_name,
        base_time=base_time,
        date=date,
        enable_deblurring=deblurring,
        enable_night_enhance=night_enhance,
        profiler=_profiler,
    )

    if not quiet:
        print(f"\n✅ Completado: {result.total_detections} detecciones")

    return result


def process_directory(
    root_path: str,
    output_base: str | None = None,
    recursive: bool = True,
    **kwargs,
) -> list[ProcessingResult]:
    from ..adapters.storage.local import LocalStorageReader

    reader = LocalStorageReader()
    results: list[ProcessingResult] = []

    scan_result = _scan_directory(root_path, reader, recursive)

    for video_info in scan_result:
        output_folder = output_base or video_info.get("output_folder", "./output")
        try:
            result = process_video(
                video=video_info["path"],
                zones=video_info.get("zones_path"),
                mask=video_info.get("mask_path"),
                output=output_folder,
                **{k: v for k, v in kwargs.items() if k in [
                    "base_time", "date", "device", "model", "reduce",
                    "max_minutes", "strategy", "night_enhance", "model_config",
                    "deblurring", "no_video", "quiet", "backend",
                    "tracker_backend", "batch_size", "tracker_config",
                    "output_name", "pipeline_mode", "queue_maxsize",
                    "profile", "profile_path",
                ]},
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error procesando {video_info['path']}: {e}")

    return results


def _scan_directory(
    root_path: str,
    reader: "LocalStorageReader",
    recursive: bool = True,
) -> list[dict]:
    from ..adapters.storage.local import LocalStorageReader

    entries = []
    paths_to_scan = [root_path]

    while paths_to_scan:
        current = paths_to_scan.pop(0)
        files = reader.list_files(current, "*")

        mask_path = None
        zones_path = None

        for fi in files:
            ext = fi.extension.lower()
            if ext in {".png", ".jpg", ".jpeg"} and "mask" in fi.name.lower():
                mask_path = fi.path
            if ext == ".json":
                zones_path = fi.path

        for fi in files:
            if fi.extension.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                import os
                entries.append({
                    "path": fi.path,
                    "mask_path": mask_path,
                    "zones_path": zones_path,
                    "output_folder": os.path.join(root_path, "output"),
                })

        if recursive:
            dirs = reader.list_files(current, "*")
            for d in dirs:
                if not d.path.endswith((".mp4", ".avi", ".mov", ".mkv", ".png", ".jpg", ".jpeg", ".json")):
                    # check if it's actually a directory
                    full_path = reader.get_video_path(d.path)
                    import os as _os
                    if _os.path.isdir(full_path):
                        paths_to_scan.append(d.path)

    return entries
