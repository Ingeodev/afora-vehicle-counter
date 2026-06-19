import time
from typing import Callable
from queue import Queue
from threading import Thread

import numpy as np
from tqdm import tqdm

from ..adapters.pipes import QueuePipe
from ..domain.pipes import IdentityMapper
from ..domain.schemas import RawDetection, ProcessingResult, FrameData
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
from ..utils.profiler import Profiler
from ..utils.time_utils import calculate_exact_time, format_timestamp
from .config import PipelineConfig
from .pipeline import _write_frame, _update_progress


class PipelinedPipeline:
    """Pipeline con 4 hilos independientes:
       Reader → [Queue] → Detector → [Queue] → Tracker+Vis → [Queue] → Writer
       Cada hilo corre al maximo sin esperar a los demas."""

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
        profiler: Profiler | None = None,
    ) -> ProcessingResult:
        cfg = self.config
        out_cfg = cfg.output

        import os as _os
        video_name = _os.path.basename(video_path)
        _pf = profiler

        output_folder = output_folder or out_cfg.output_folder
        self.writer.makedirs(output_folder)

        if zones_path and self.reader.exists(zones_path):
            self.zone_manager.load_from_json(self.reader, zones_path)

        if mask_path and self.reader.exists(mask_path):
            self.mask_manager.load(self.reader, mask_path)

        video_path_local = self.reader.get_video_path(video_path)
        video = VideoSource.from_path(video_path_local)

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

            vehicle_tracker = VehicleTracker()

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

            from ..adapters.preprocessors.noop import NoOpPreprocessor
            from ..adapters.preprocessors.deblurring import DeblurringPreprocessor

            if enable_night_enhance:
                reader_preprocessor = DeblurringPreprocessor.create_night_enhance()
            elif enable_deblurring:
                reader_preprocessor = DeblurringPreprocessor(mode="night")
            else:
                reader_preprocessor = NoOpPreprocessor()

            pbar = tqdm(
                total=max_frames,
                unit="frames",
                desc=f"Procesando {output_name or 'video'}",
                disable=not out_cfg.verbose,
            )

            if hasattr(self.detector, "warmup"):
                self.detector.warmup((new_height, new_width, 3))

            qsize = cfg.batch.queue_maxsize
            batch_enabled = cfg.batch.enabled
            batch_size = cfg.batch.batch_size

            # ─── 3 pipes: reader→detector, detector→tracker, tracker→writer ──
            detect_pipe: QueuePipe[tuple[np.ndarray, FrameData]] = QueuePipe(
                maxsize=qsize,
                input_mapper=IdentityMapper(),
                output_mapper=IdentityMapper(),
            )
            raw_pipe: QueuePipe[tuple[list[RawDetection], np.ndarray, FrameData]] = QueuePipe(
                maxsize=qsize,
                input_mapper=IdentityMapper(),
                output_mapper=IdentityMapper(),
            )
            output_queue: Queue | None = None

            # ─── Thread 1: Reader + Preprocessor ─────────────────────────
            def _reader_worker():
                try:
                    for fd in video.frames(max_frames=max_frames, resize=resize, skip_rate=cfg.video.skip_rate):
                        t = time.perf_counter()
                        processed = reader_preprocessor.process(fd.frame)
                        if _pf: _pf.record(video_name, fd.index, "preprocess", (time.perf_counter() - t) * 1000)
                        detect_pipe.send((processed, fd))
                finally:
                    detect_pipe.close()

            reader_thread = Thread(target=_reader_worker, daemon=True)
            reader_thread.start()

            # ─── Thread 2: Detector ──────────────────────────────────────
            def _detector_worker():
                buffer: list[tuple[np.ndarray, FrameData]] = []
                try:
                    while True:
                        item = detect_pipe.receive()
                        if item is None:
                            break
                        processed, fd = item
                        buffer.append((processed, fd))

                        if not batch_enabled or len(buffer) >= batch_size:
                            self._flush_raw(buffer, raw_pipe, video_name, _pf)
                            buffer = []

                    if buffer:
                        self._flush_raw(buffer, raw_pipe, video_name, _pf)
                except Exception:
                    import traceback
                    traceback.print_exc()
                finally:
                    raw_pipe.close()

            detector_thread = Thread(target=_detector_worker, daemon=True)
            detector_thread.start()

            # ─── Thread 3: Tracker + Visualizer ──────────────────────────
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

            def _tracker_worker():
                nonlocal frames_processed, start_time
                try:
                    while True:
                        item = raw_pipe.receive()
                        if item is None:
                            break
                        raw_detections, processed_frame, fd = item
                        ts = fd.timestamp_seconds

                        t = time.perf_counter()
                        detections = self.tracker.update(raw_detections, processed_frame)
                        if _pf: _pf.record(video_name, fd.index, "track", (time.perf_counter() - t) * 1000)
                        vehicle_tracker.update_batch(detections)

                        if self.mask_manager.is_loaded:
                            detections = self.mask_manager.filter_detections(detections)

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
                            if _pf: _pf.record(video_name, fd.index, "zone_check", (time.perf_counter() - t) * 1000)

                        if video_writer and output_queue is not None:
                            t = time.perf_counter()
                            _write_frame(
                                fd, detections, output_queue,
                                video_writer, out_cfg, self.zone_manager,
                                self.visualizer, vehicle_tracker,
                            )
                            if _pf: _pf.record(video_name, fd.index, "visualize", (time.perf_counter() - t) * 1000)

                        frames_processed += 1
                        _update_progress(frames_processed, max_frames, start_time, pbar, self.zone_manager, on_progress, out_cfg)
                        pbar.update(1)
                except Exception:
                    import traceback
                    traceback.print_exc()
                finally:
                    if output_queue is not None:
                        output_queue.put(None)

            tracker_thread = Thread(target=_tracker_worker, daemon=True)
            tracker_thread.start()

            # ─── Esperar a que todos los hilos terminen ──────────
            reader_thread.join(timeout=300)
            detector_thread.join(timeout=300)
            tracker_thread.join(timeout=300)

            if writer_thread is not None and output_queue is not None:
                writer_thread.join(timeout=10)

            pbar.close()

            # ─── CSV export ──────────────────────────────────────
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
                _pf.save()

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

    def _flush_raw(
        self,
        buffer: list[tuple[np.ndarray, FrameData]],
        raw_pipe: QueuePipe,
        video_name: str = "",
        profiler: "Profiler | None" = None,
    ) -> None:
        batch_frames = [p for p, _ in buffer]
        originals = [fd for _, fd in buffer]
        first_fi = originals[0].index if originals else -1
        batch = len(buffer)

        t = time.perf_counter()
        if batch == 1:
            raw_list = [self.detector.detect(batch_frames[0])]
        else:
            raw_list = self.detector.detect_batch(batch_frames)
        if profiler:
            profiler.record(video_name, first_fi, "detect", (time.perf_counter() - t) * 1000, batch=batch)

        for raw_detections, processed_frame, fd_orig in zip(raw_list, batch_frames, originals):
            raw_pipe.send((raw_detections, processed_frame, fd_orig))
