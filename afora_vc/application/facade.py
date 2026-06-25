import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Literal

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
from ..domain.counting.services import VehicleClassesResolver
from ..adapters import DetectorRegistry, TrackerRegistry
from ..adapters.storage.local import LocalStorageReader
from ..utils.display import DisplayManager
from ..utils.progress import ProgressTracker
from .config import PipelineConfig
from .pipeline import Pipeline
from .pipelined_pipeline import PipelinedPipeline
from .batch_config import BatchConfig, VideoOverride


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
    audit: bool = False,
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

    from ..utils.auditor import Auditor
    from ..utils.profiler import Profiler

    _profiler: Profiler | None = None
    if profile:
        _profiler = Profiler(csv_path=profile_path or f"{output}/profile.csv")

    _auditor: Auditor | None = None
    if audit:
        _auditor = Auditor()

    _display = DisplayManager(verbose=not quiet, quiet=quiet)

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
        auditor=_auditor,
        display=_display,
    )

    if _auditor:
        _display.show_audit_report(_auditor.report())

    return result


def process_directory(
    root_path: str,
    output_base: str | None = None,
    recursive: bool = True,
    global_mask: str | None = None,
    global_zones: str | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    skip_paths: set[str] | None = None,
    progress_tracker: ProgressTracker | None = None,
    count_config=None,
    count_resolver: VehicleClassesResolver | None = None,
    **kwargs,
) -> list[ProcessingResult]:
    reader = LocalStorageReader()
    results: list[ProcessingResult] = []
    overrides = overrides or {}
    skip_paths = skip_paths or set()

    scan_result = _scan_directory(root_path, reader, recursive)

    quiet = kwargs.get("quiet", False)
    _display = DisplayManager(verbose=not quiet, quiet=quiet)
    total_videos = len(scan_result)
    pending = [e for e in scan_result if e["path"] not in skip_paths]
    skipped_count = total_videos - len(pending)

    if skipped_count:
        _display.show_info(f"Procesados previamente: {skipped_count} videos (saltados)")
    _display.show_info(f"Escaneando: {root_path} -- {len(pending)} pendientes de {total_videos}")

    if not pending:
        _display.show_success("Todos los videos ya fueron procesados")
        return results

    # Agrupar pendientes por subdirectorio
    dir_groups: dict[str, list[dict]] = defaultdict(list)
    for entry in pending:
        dir_key = os.path.dirname(entry["rel_path"]) or "__root__"
        dir_groups[dir_key].append(entry)

    completed_counts: dict[str, int] = defaultdict(int)
    count_threads: list[threading.Thread] = []

    _display.start_scan(total_videos=len(pending))

    for i, entry in enumerate(pending, 1):
        video_path = entry["path"]
        rel_path = entry["rel_path"]
        video_name = entry["name"]
        dir_key = os.path.dirname(rel_path) or "__root__"
        _display.start_video_progress(total=100, description=f"[{i}/{len(pending)}] {video_name}")

        # Preserve directory structure in output
        if output_base:
            out_dir = os.path.join(output_base, os.path.dirname(rel_path)) if os.path.dirname(rel_path) else output_base
        else:
            out_dir = entry["output_folder"]
        os.makedirs(out_dir, exist_ok=True)

        # Resolve mask and zones: override > global > auto-detected
        ov = overrides.get(rel_path.replace("\\", "/"), {}) if overrides else {}
        final_mask = ov.get("mask") or global_mask or entry.get("mask_path")
        final_zones = ov.get("zones") or global_zones or entry.get("zones_path")

        # Build per-video kwargs with overrides merged on top
        video_kwargs = dict(kwargs)
        for k in ("reduce", "batch_size", "pipeline_mode", "model", "backend",
                  "tracker_backend", "device", "strategy", "max_minutes",
                  "night_enhance", "deblurring", "no_video", "output_name"):
            if k in ov and ov[k] is not None:
                video_kwargs[k] = ov[k]

        try:
            result = process_video(
                video=video_path,
                zones=final_zones,
                mask=final_mask,
                output=out_dir,
                **{k: v for k, v in video_kwargs.items() if k in [
                    "base_time", "date", "device", "model", "reduce",
                    "max_minutes", "strategy", "night_enhance", "model_config",
                    "deblurring", "no_video", "quiet", "backend",
                    "tracker_backend", "batch_size", "tracker_config",
                    "output_name", "pipeline_mode", "queue_maxsize",
                    "profile", "profile_path", "audit",
                ]},
            )
            results.append(result)
            if progress_tracker is not None:
                progress_tracker.mark_completed(
                    video_path=video_path,
                    output_path=out_dir,
                    processing_time_seconds=result.processing_time_seconds,
                    total_detections=result.total_detections,
                )
        except Exception as e:
            _display.show_error(str(e), context=video_name)
        finally:
            _display.stop_video_progress()
            _display.update_scan(advance=1)

        # Verificar si este subdirectorio completo su procesamiento
        completed_counts[dir_key] += 1
        if count_config and dir_key != "__root__" and completed_counts[dir_key] == len(dir_groups[dir_key]):
            from .count_pipeline import GroupPipeline
            from ..output.count_exporter import DefaultFlowExporter
            t = threading.Thread(
                target=_run_count_pipeline,
                args=(count_config, out_dir, dir_key, count_resolver),
                daemon=True,
            )
            t.start()
            count_threads.append(t)

    _display.stop_scan()

    total_detections = sum(r.total_detections for r in results)
    all_count = len(pending)
    _display.show_success(f"Procesados {all_count} videos, {total_detections} detecciones totales")

    for t in count_threads:
        t.join()

    return results


def _run_count_pipeline(count_config, out_dir, dir_key, resolver):
    from .count_pipeline import GroupPipeline, export_chart
    from ..output.count_exporter import DefaultFlowExporter
    pipeline = GroupPipeline(count_config, resolver=resolver)
    data = pipeline.execute(out_dir, output_dir=out_dir)
    if data:
        exporter = DefaultFlowExporter()
        out_path = os.path.join(out_dir, "aggregated_counts.csv")
        exporter.write(data, count_config, resolver or VehicleClassesResolver(), out_path)
        export_chart(data, count_config, resolver or VehicleClassesResolver(), out_dir, date=dir_key)


def process_from_config(config: BatchConfig | str) -> list[ProcessingResult]:
    config_path: str | None = None
    if isinstance(config, str):
        config_path = config
        config = BatchConfig.from_yaml(config)
    else:
        config_path = getattr(config, "_source_path", None)

    overrides_dict: dict[str, dict[str, Any]] = {}
    for ov in config.overrides:
        d = {k: v for k, v in vars(ov).items() if k != "path" and v is not None}
        overrides_dict[ov.path.replace("\\", "/")] = d

    progress_tracker: ProgressTracker | None = None
    skip_paths: set[str] = set()
    if config_path:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        progress_path = os.path.join(config_dir, "progress.csv")
        progress_tracker = ProgressTracker(progress_path)
        skip_paths = progress_tracker.completed_paths

    resolver = VehicleClassesResolver.from_model_config(config.model_config)

    return process_directory(
        root_path=config.input,
        output_base=config.output,
        recursive=config.recursive,
        global_mask=config.mask,
        global_zones=config.zones,
        overrides=overrides_dict,
        skip_paths=skip_paths,
        progress_tracker=progress_tracker,
        count_config=config.count,
        count_resolver=resolver,
        device=config.device,
        model=config.model,
        reduce=config.reduce,
        max_minutes=config.max_minutes,
        strategy=config.strategy,
        night_enhance=config.night_enhance,
        deblurring=config.deblurring,
        no_video=config.no_video,
        quiet=config.quiet,
        backend=config.backend,
        tracker_backend=config.tracker,
        batch_size=config.batch_size,
        output_name=config.output_name,
        pipeline_mode=config.pipeline_mode,
        model_config=config.model_config,
        tracker_config=config.tracker_config,
        base_time=config.base_time,
        date=config.date,
        queue_maxsize=config.queue_maxsize,
        profile=config.profile,
        profile_path=config.profile_path,
        audit=config.audit,
    )


def _scan_directory(
    root_path: str,
    reader: "LocalStorageReader",
    recursive: bool = True,
) -> list[dict]:
    from pathlib import Path

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
                rel = os.path.relpath(fi.path, root_path)
                entries.append({
                    "path": fi.path,
                    "rel_path": rel,
                    "name": fi.name,
                    "mask_path": mask_path,
                    "zones_path": zones_path,
                    "output_folder": os.path.join(root_path, "output"),
                })

        if recursive:
            dir_entries = [e for e in os.scandir(current) if e.is_dir()]
            for d in dir_entries:
                paths_to_scan.append(d.path)

    entries.sort(key=lambda e: e["name"])
    return entries
