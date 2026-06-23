import time
from collections import Counter
from typing import Callable
from queue import Queue
from threading import Thread

import cv2
import numpy as np
from tqdm import tqdm

from ..domain.schemas import Detection, ProcessingResult, FrameData
from ..domain.services.zone_manager import ZoneManager
from ..domain.services.mask_manager import MaskManager
from ..domain.services.tracker import VehicleTracker
from ..domain.ports.detector import Detector
from ..domain.ports.tracker import Tracker
from ..domain.ports.preprocessor import FramePreprocessor
from ..domain.ports.storage import StorageReader, StorageWriter
from ..adapters.video.source import VideoSource
from ..adapters.video.writer import AnnotatedVideoWriter, VideoWriterConfig
from ..adapters.storage.local import LocalStorageReader, LocalStorageWriter
from ..output.visualizer import Visualizer
from ..output.csv_exporter import CSVExporter
from ..utils.auditor import Auditor
from ..utils.profiler import Profiler
from ..utils.time_utils import calculate_exact_time, format_timestamp
from .config import PipelineConfig, DetectorConfig, TrackerConfig


class PipelineContext:
    def __init__(self, frame_index: int, timestamp_seconds: float):
        self.frame_index = frame_index
        self.timestamp_seconds = timestamp_seconds
        self.detections: list[Detection] = []


def _write_frame(
    fd: FrameData,
    detections: list[Detection],
    output_queue: Queue,
    video_writer: "AnnotatedVideoWriter",
    out_cfg: "OutputConfig",
    zone_manager: ZoneManager,
    visualizer: Visualizer,
    vehicle_tracker: VehicleTracker,
) -> None:
    frame = fd.frame.copy()
    if out_cfg.draw_zones and zone_manager:
        visualizer.draw_zones(frame, zone_manager.zones)
    for det in detections:
        zones_visited = zone_manager.get_zones_for_vehicle(det.track_id)
        trajectory = vehicle_tracker.get_trajectory_array(
            det.track_id, out_cfg.max_trajectory_points
        )
        visualizer.draw_detection(frame, det, zones_visited, trajectory)
    stats = {
        "Detecciones": zone_manager.get_log_count() if zone_manager else 0,
        "Tiempo": format_timestamp(fd.timestamp_seconds),
    }
    visualizer.draw_stats(frame, stats)
    output_queue.put(frame)


def _update_progress(
    frames_processed: int,
    max_frames: int,
    start_time: float,
    pbar: "tqdm",
    zone_manager: ZoneManager,
    on_progress: Callable[[int, int, float], None] | None,
    out_cfg: "OutputConfig",
) -> None:
    if on_progress and frames_processed % out_cfg.progress_interval == 0:
        elapsed = time.time() - start_time
        on_progress(frames_processed, max_frames, elapsed)
    if frames_processed % 10 == 0:
        postfix = {}
        if zone_manager:
            postfix["detections"] = zone_manager.get_log_count()
        pbar.set_postfix(postfix)


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        detector: Detector,
        tracker: Tracker,
        preprocessor: FramePreprocessor | None = None,
        storage_reader: StorageReader | None = None,
        storage_writer: StorageWriter | None = None,
    ):
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.preprocessor = preprocessor
        self.reader = storage_reader or LocalStorageReader()
        self.writer = storage_writer or LocalStorageWriter()
        self.zone_manager = ZoneManager()
        self.mask_manager = MaskManager()
        self.visualizer = Visualizer()
        self.csv_exporter = CSVExporter(self.writer)

    def process_video(
        self,
        video_path: str,
        zones_path: str | None = None,
        mask_path: str | None = None,
        output_folder: str | None = None,
        output_name: str | None = None,
        base_time: str = "00:00",
        date: str = "Unknown",
        on_progress: Callable[[int, int, float], None] | None = None,
        enable_deblurring: bool = False,
        enable_night_enhance: bool = False,
        profiler: "Profiler | None" = None,
        auditor: "Auditor | None" = None,
    ) -> ProcessingResult:
        cfg = self.config
        out_cfg = cfg.output

        import os as _os
        video_name = _os.path.basename(video_path)
        _pf = profiler
        _au = auditor

        output_folder = output_folder or out_cfg.output_folder
        self.writer.makedirs(output_folder)

        # ZoneManager
        if zones_path and self.reader.exists(zones_path):
            self.zone_manager.load_from_json(self.reader, zones_path)

        # MaskManager
        if mask_path and self.reader.exists(mask_path):
            self.mask_manager.load(self.reader, mask_path)

        # Open video
        video_path_local = self.reader.get_video_path(video_path)
        video = VideoSource.from_path(video_path_local)

        # Reset tracker between videos
        self.tracker.reset()

        try:
            reduce_factor = cfg.video.reduce_factor
            new_width = video.width // reduce_factor
            new_height = video.height // reduce_factor

            if reduce_factor > 1:
                self.zone_manager.scale_to_resolution(
                    video.width, video.height, new_width, new_height
                )

            if self.mask_manager.is_loaded:
                self.mask_manager.resize_to_frame(new_width, new_height)

            if cfg.video.max_minutes is not None and cfg.video.max_minutes > 0:
                max_frames = int(cfg.video.max_minutes * 60 * video.fps)
            else:
                max_frames = video.total_frames

            # Create VehicleTracker (history)
            vehicle_tracker = VehicleTracker()

            # Video writer
            output_video_path = None
            video_writer = None

            if out_cfg.save_video:
                if output_name is None:
                    import os
                    base_name = os.path.basename(video_path)
                    output_name = base_name.replace(".mp4", "_processed.mp4")

                output_video_path = f"{output_folder}/{output_name}"

                writer_config = VideoWriterConfig(
                    fps=cfg.video.output_fps,
                    codec=cfg.video.codec,
                    width=new_width,
                    height=new_height,
                )
                video_writer = AnnotatedVideoWriter.from_path(
                    output_video_path, writer_config
                )

            frames_processed = 0
            start_time = time.time()
            resize = (new_width, new_height) if reduce_factor > 1 else None

            # Preprocessors
            from ..adapters.preprocessors.noop import NoOpPreprocessor
            from ..adapters.preprocessors.deblurring import DeblurringPreprocessor

            if enable_night_enhance:
                preprocessor = DeblurringPreprocessor.create_night_enhance()
            elif enable_deblurring:
                preprocessor = DeblurringPreprocessor(mode="night")
            else:
                preprocessor = NoOpPreprocessor()

            pbar = tqdm(
                total=max_frames,
                unit="frames",
                desc=f"Procesando {output_name or 'video'}",
                disable=not out_cfg.verbose,
            )

            # Warmup: run a dummy inference to load CUDA kernels, ONNX runtime, etc.
            if hasattr(self.detector, "warmup"):
                self.detector.warmup((new_height, new_width, 3))

            # Threaded reader
            qsize = cfg.batch.queue_maxsize
            input_queue: Queue = Queue(maxsize=qsize)

            def _reader_worker():
                try:
                    for fd in video.frames(max_frames=max_frames, resize=resize, skip_rate=cfg.video.skip_rate):
                        input_queue.put(fd)
                finally:
                    input_queue.put(None)

            reader_thread = Thread(target=_reader_worker, daemon=True)
            reader_thread.start()

            # Threaded writer
            output_queue: Queue | None = None
            writer_thread: Thread | None = None

            if video_writer:
                output_queue = Queue(maxsize=qsize)

                def _writer_worker():
                    try:
                        while True:
                            f = output_queue.get()
                            if f is None:
                                break
                            video_writer.write(f)
                    finally:
                        video_writer.close()

                writer_thread = Thread(target=_writer_worker, daemon=True)
                writer_thread.start()

            # ─── Main loop ────────────────
            batch_enabled = cfg.batch.enabled
            batch_buffer: list[tuple[FrameData, np.ndarray, float]] = []

            for item in iter(input_queue.get, None):
                fd: FrameData = item
                fi = fd.index

                t = time.perf_counter()
                processed = preprocessor.process(fd.frame)
                if _pf: _pf.record(video_name, fi, "preprocess", (time.perf_counter() - t) * 1000)
                ts = fd.timestamp_seconds

                if batch_enabled:
                    batch_buffer.append((fd, processed, ts))
                    if len(batch_buffer) < cfg.batch.batch_size:
                        continue

                    batch_frames = [b[1] for b in batch_buffer]
                    batch_ts = [b[2] for b in batch_buffer]
                    batch_originals = [b[0] for b in batch_buffer]
                    first_fi = batch_originals[0].index

                    t = time.perf_counter()
                    raw_detections_batch = self.detector.detect_batch(batch_frames)
                    if _pf: _pf.record(video_name, first_fi, "detect", (time.perf_counter() - t) * 1000, batch=len(batch_frames))
                    if _au:
                        for i, (rd, fd_orig) in enumerate(zip(raw_detections_batch, batch_originals)):
                            _au.snapshot("detect", fd_orig.index, len(rd), class_counts=dict(_Counter(d.class_name for d in rd)))

                    for raw_detections, ts_orig, fd_orig, proc_frame in zip(
                        raw_detections_batch, batch_ts, batch_originals, batch_frames
                    ):
                        t = time.perf_counter()
                        detections = self.tracker.update(raw_detections, proc_frame)
                        if _pf: _pf.record(video_name, fd_orig.index, "track", (time.perf_counter() - t) * 1000)
                        vehicle_tracker.update_batch(detections)

                        if self.mask_manager.is_loaded:
                            detections = self.mask_manager.filter_detections(detections)

                        if _au:
                            _au.snapshot("track", fd_orig.index, len(detections), class_counts=dict(_Counter(d.class_name for d in detections)), track_ids={d.track_id for d in detections})

                        zone_count_before = len(self.zone_manager.get_detection_log()) if self.zone_manager else 0

                        if self.zone_manager:
                            t = time.perf_counter()
                            for det in detections:
                                exact_time = calculate_exact_time(base_time, ts_orig) or "00:00:00"
                                self.zone_manager.check_entry(
                                    track_id=det.track_id,
                                    x=det.center[0],
                                    y=det.center[1],
                                    timestamp_seconds=ts_orig,
                                    vehicle_type=det.class_name,
                                    date=date,
                                    exact_time=exact_time,
                                )
                            if _pf: _pf.record(video_name, fd_orig.index, "zone_check", (time.perf_counter() - t) * 1000)

                        if _au and self.zone_manager:
                            zone_entries = len(self.zone_manager.get_detection_log()) - zone_count_before
                            if zone_entries > 0:
                                _new_log = self.zone_manager.get_detection_log()[zone_count_before:]
                                _au.snapshot("zone_check", fd_orig.index, zone_entries,
                                              class_counts=dict(Counter(e.zone for e in _new_log)),
                                              track_ids={e.vehicle_id for e in _new_log})
                            else:
                                _au.snapshot("zone_check", fd_orig.index, 0)

                        if _au and self.zone_manager:
                            for det in detections:
                                zv = self.zone_manager.get_zones_for_vehicle(det.track_id)
                                if zv:
                                    for z in zv:
                                        _au.snapshot("zone_display", fd_orig.index, 1,
                                                      class_counts={z: 1}, track_ids={det.track_id})

                        if video_writer and output_queue is not None:
                            t = time.perf_counter()
                            _write_frame(
                                fd_orig, detections, output_queue,
                                video_writer, out_cfg, self.zone_manager,
                                self.visualizer, vehicle_tracker,
                            )
                            if _pf: _pf.record(video_name, fd_orig.index, "visualize", (time.perf_counter() - t) * 1000)

                    batch_buffer.clear()
                    frames_processed += len(batch_frames)
                    pbar.update(len(batch_frames))
                    _update_progress(frames_processed, max_frames, start_time, pbar, self.zone_manager, on_progress, out_cfg)
                    continue

                # ─── Single-frame path ────────────────
                t = time.perf_counter()
                raw_detections = self.detector.detect(processed)
                if _pf: _pf.record(video_name, fi, "detect", (time.perf_counter() - t) * 1000)
                if _au:
                    from collections import Counter as _Counter
                    _au.snapshot("detect", fi, len(raw_detections), class_counts=dict(_Counter(d.class_name for d in raw_detections)))

                t = time.perf_counter()
                detections = self.tracker.update(raw_detections, processed)
                if _pf: _pf.record(video_name, fi, "track", (time.perf_counter() - t) * 1000)

                vehicle_tracker.update_batch(detections)

                if self.mask_manager.is_loaded:
                    detections = self.mask_manager.filter_detections(detections)

                if _au:
                    _au.snapshot("track", fi, len(detections), class_counts=dict(_Counter(d.class_name for d in detections)), track_ids={d.track_id for d in detections})

                zone_count_before = len(self.zone_manager.get_detection_log()) if self.zone_manager else 0

                if self.zone_manager:
                    t = time.perf_counter()
                    for det in detections:
                        exact_time = calculate_exact_time(base_time, ts) or "00:00:00"
                        self.zone_manager.check_entry(
                            track_id=det.track_id,
                            x=det.center[0],
                            y=det.center[1],
                            timestamp_seconds=ts,
                            vehicle_type=det.class_name,
                            date=date,
                            exact_time=exact_time,
                        )
                    if _pf: _pf.record(video_name, fi, "zone_check", (time.perf_counter() - t) * 1000)

                if _au and self.zone_manager:
                    zone_entries = len(self.zone_manager.get_detection_log()) - zone_count_before
                    if zone_entries > 0:
                        _new_log = self.zone_manager.get_detection_log()[zone_count_before:]
                        _au.snapshot("zone_check", fi, zone_entries,
                                          class_counts=dict(Counter(e.zone for e in _new_log)),
                                      track_ids={e.vehicle_id for e in _new_log})
                    else:
                        _au.snapshot("zone_check", fi, 0)

                if _au and self.zone_manager:
                    for det in detections:
                        zv = self.zone_manager.get_zones_for_vehicle(det.track_id)
                        if zv:
                            for z in zv:
                                _au.snapshot("zone_display", fi, 1,
                                              class_counts={z: 1}, track_ids={det.track_id})

                if video_writer and output_queue is not None:
                    t = time.perf_counter()
                    _write_frame(
                        fd, detections, output_queue,
                        video_writer, out_cfg, self.zone_manager,
                        self.visualizer, vehicle_tracker,
                    )
                    if _pf: _pf.record(video_name, fi, "visualize", (time.perf_counter() - t) * 1000)

                frames_processed += 1
                _update_progress(frames_processed, max_frames, start_time, pbar, self.zone_manager, on_progress, out_cfg)
                pbar.update(1)

            # Flush remaining batch buffer
            if batch_enabled and batch_buffer:
                batch_frames = [b[1] for b in batch_buffer]
                batch_ts = [b[2] for b in batch_buffer]
                batch_originals = [b[0] for b in batch_buffer]
                first_fi = batch_originals[0].index

                t = time.perf_counter()
                raw_detections_batch = self.detector.detect_batch(batch_frames)
                if _pf: _pf.record(video_name, first_fi, "detect", (time.perf_counter() - t) * 1000, batch=len(batch_frames))
                if _au:
                    for i, (rd, fd_orig) in enumerate(zip(raw_detections_batch, batch_originals)):
                        _au.snapshot("detect", fd_orig.index, len(rd), class_counts=dict(_Counter(d.class_name for d in rd)))

                for raw_detections, ts_orig, fd_orig, proc_frame in zip(
                    raw_detections_batch, batch_ts, batch_originals, batch_frames
                ):
                    t = time.perf_counter()
                    detections = self.tracker.update(raw_detections, proc_frame)
                    if _pf: _pf.record(video_name, fd_orig.index, "track", (time.perf_counter() - t) * 1000)
                    vehicle_tracker.update_batch(detections)
                    if self.mask_manager.is_loaded:
                        detections = self.mask_manager.filter_detections(detections)
                    if _au:
                        _au.snapshot("track", fd_orig.index, len(detections), class_counts=dict(_Counter(d.class_name for d in detections)), track_ids={d.track_id for d in detections})

                    _zone_count_before = len(self.zone_manager.get_detection_log()) if self.zone_manager else 0

                    if self.zone_manager:
                        t = time.perf_counter()
                        for det in detections:
                            exact_time = calculate_exact_time(base_time, ts_orig) or "00:00:00"
                            self.zone_manager.check_entry(
                                track_id=det.track_id,
                                x=det.center[0],
                                y=det.center[1],
                                timestamp_seconds=ts_orig,
                                vehicle_type=det.class_name,
                                date=date,
                                exact_time=exact_time,
                            )
                        if _pf: _pf.record(video_name, fd_orig.index, "zone_check", (time.perf_counter() - t) * 1000)

                    if _au and self.zone_manager:
                        _zone_entries = len(self.zone_manager.get_detection_log()) - _zone_count_before
                        if _zone_entries > 0:
                            _new_log = self.zone_manager.get_detection_log()[_zone_count_before:]
                            _au.snapshot("zone_check", fd_orig.index, _zone_entries,
                                      class_counts=dict(Counter(e.zone for e in _new_log)),
                                          track_ids={e.vehicle_id for e in _new_log})
                        else:
                            _au.snapshot("zone_check", fd_orig.index, 0)
                        for det in detections:
                            zv = self.zone_manager.get_zones_for_vehicle(det.track_id)
                            if zv:
                                for z in zv:
                                    _au.snapshot("zone_display", fd_orig.index, 1,
                                                  class_counts={z: 1}, track_ids={det.track_id})

                    if video_writer and output_queue is not None:
                        t = time.perf_counter()
                        _write_frame(
                            fd_orig, detections, output_queue,
                            video_writer, out_cfg, self.zone_manager,
                            self.visualizer, vehicle_tracker,
                        )
                        if _pf: _pf.record(video_name, fd_orig.index, "visualize", (time.perf_counter() - t) * 1000)
                batch_buffer.clear()
                frames_processed += len(batch_frames)

            pbar.close()

            # Close writer
            if writer_thread is not None and output_queue is not None:
                output_queue.put(None)
                writer_thread.join(timeout=10)

            # Export CSV
            csv_path = None
            if out_cfg.save_csv and self.zone_manager:
                csv_dir = f"{output_folder}/csv"
                self.writer.makedirs(csv_dir)
                detection_log = self.zone_manager.get_detection_log()
                zone_counts_map = self.zone_manager.get_zone_counts()

                det_path = f"{csv_dir}/detections.csv"
                self.csv_exporter.export_detections(detection_log, det_path)

                summary_path = f"{csv_dir}/summary.csv"
                self.csv_exporter.export_summary(zone_counts_map, summary_path)

                types_path = f"{csv_dir}/vehicle_types.csv"
                self.csv_exporter.export_vehicle_types(detection_log, types_path)

                csv_path = det_path

            if _pf:
                profile_saved = _pf.save()
                if out_cfg.verbose:
                    print(f"📊 Profile: {profile_saved}")

            processing_time = time.time() - start_time
            fps = frames_processed / processing_time if processing_time > 0 else 0
            detection_log = self.zone_manager.get_detection_log() if self.zone_manager else []
            zone_counts = self.zone_manager.get_zone_counts() if self.zone_manager else {}

            if out_cfg.verbose:
                print("\n" + "=" * 70)
                print("✅ PROCESO COMPLETADO")
                print("=" * 70)
                print(f"⏱️ Tiempo: {processing_time / 60:.2f} min ({processing_time:.1f}s)")
                print(f"🎞️ Frames: {frames_processed} | {fps:.1f} FPS avg")
                if output_video_path:
                    print(f"📁 Video: {output_video_path}")
                if csv_path:
                    print(f"📊 CSV: {csv_path}")
                print(f"\n🚗 Detecciones totales: {len(detection_log)}")
                if zone_counts:
                    print("\n📍 POR ZONA:")
                    for zone, count in sorted(zone_counts.items()):
                        print(f"   Zona {zone}: {count}")
                print("=" * 70)

            return ProcessingResult(
                video_path=video_path,
                output_video_path=output_video_path,
                csv_path=csv_path,
                total_detections=len(detection_log),
                zone_counts=zone_counts,
                processing_time_seconds=processing_time,
                frames_processed=frames_processed,
                detection_log=detection_log,
            )

        finally:
            video.close()
