"""Tests para PipelinedPipeline con mocks.
   Valida comunicacion entre hilos, invocacion a detect/tracker, y limpieza."""

import time

import numpy as np
import pytest

from afora_vc.domain.schemas import RawDetection, Detection, BoundingBox, FrameData
from afora_vc.domain.enums import PipelineMode
from afora_vc.application.config import PipelineConfig
from afora_vc.application.pipelined_pipeline import PipelinedPipeline


def _make_frame(height=480, width=640) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _raw_detection(class_id=2, class_name="car") -> RawDetection:
    return RawDetection(
        class_id=class_id, class_name=class_name, confidence=0.9,
        bbox=BoundingBox(10, 20, 100, 200),
    )


def _detection(track_id=1, class_id=2, class_name="car") -> Detection:
    return Detection(
        track_id=track_id, class_id=class_id, class_name=class_name,
        confidence=0.9, bbox=BoundingBox(10, 20, 100, 200),
    )


def _make_frames(n: int, fps: float = 30) -> list[FrameData]:
    return [
        FrameData(frame=_make_frame(), index=i, timestamp_seconds=i / fps)
        for i in range(n)
    ]


class TestPipelinedPipelineInit:
    def test_default_pipeline_mode_is_sequential(self):
        cfg = PipelineConfig()
        assert cfg.pipeline_mode == PipelineMode.SEQUENTIAL

    def test_pipelined_pipeline_accepts_detector_and_tracker(self, mocker):
        cfg = PipelineConfig()
        detector = mocker.MagicMock()
        tracker = mocker.MagicMock()
        pipeline = PipelinedPipeline(config=cfg, detector=detector, tracker=tracker)
        assert pipeline.detector is detector
        assert pipeline.tracker is tracker


class TestPipelinedPipelineFlush:
    """_flush_raw usa detect() para 1 frame y detect_batch() para varios."""

    def test_flush_single_frame_calls_detect(self, mocker):
        cfg = PipelineConfig()
        detector = mocker.MagicMock()
        detector.detect.return_value = [_raw_detection()]
        detector.detect_batch.return_value = [[_raw_detection()]]
        tracker = mocker.MagicMock()

        from afora_vc.adapters.pipes import QueuePipe
        raw_pipe = QueuePipe(maxsize=4)

        pipeline = PipelinedPipeline(config=cfg, detector=detector, tracker=tracker)
        frame = _make_frame()
        fd = _make_frames(1)[0]
        pipeline._flush_raw([(frame, fd)], raw_pipe)

        detector.detect.assert_called_once()
        assert detector.detect_batch.call_count == 0
        raw, processed_frame, result_fd = raw_pipe.receive()
        assert len(raw) == 1
        assert result_fd is fd
        raw_pipe.close()

    def test_flush_multiple_frames_calls_detect_batch(self, mocker):
        cfg = PipelineConfig()
        detector = mocker.MagicMock()
        detector.detect_batch.return_value = [[_raw_detection()], [_raw_detection()]]
        detector.detect.return_value = [_raw_detection()]
        tracker = mocker.MagicMock()

        from afora_vc.adapters.pipes import QueuePipe
        raw_pipe = QueuePipe(maxsize=4)

        pipeline = PipelinedPipeline(config=cfg, detector=detector, tracker=tracker)
        frame1, frame2 = _make_frame(), _make_frame()
        fd1, fd2 = _make_frames(2)
        pipeline._flush_raw([(frame1, fd1), (frame2, fd2)], raw_pipe)

        assert detector.detect.call_count == 0
        detector.detect_batch.assert_called_once()
        assert len(detector.detect_batch.call_args[0][0]) == 2


class TestPipelinedPipelineSingleFrame:
    """Con batch_size=1 y modo pipelined, cada frame se envia individualmente."""

    @pytest.fixture
    def make_mocks(self, mocker):
        detector = mocker.MagicMock()
        detector.detect.return_value = [_raw_detection()]
        detector.supports_batch = False

        tracker = mocker.MagicMock()
        tracker.update.return_value = [_detection()]
        return detector, tracker

    def test_process_calls_detect_per_frame(self, make_mocks, mocker):
        detector, tracker = make_mocks
        config = PipelineConfig()
        config.pipeline_mode = PipelineMode.PIPELINED
        config.output.save_video = False
        config.output.save_csv = False

        pipeline = PipelinedPipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 5
        mock_video.frames.return_value = _make_frames(5)

        mocker.patch(
            "afora_vc.application.pipelined_pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        result = pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        assert result.frames_processed == 5
        assert detector.detect.call_count == 5
        assert tracker.update.call_count == 5

    def test_raw_detection_flows_to_tracker(self, make_mocks, mocker):
        detector, tracker = make_mocks
        config = PipelineConfig()
        config.pipeline_mode = PipelineMode.PIPELINED
        config.output.save_video = False
        config.output.save_csv = False

        pipeline = PipelinedPipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 3
        mock_video.frames.return_value = _make_frames(3)

        mocker.patch(
            "afora_vc.application.pipelined_pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        for call_args in tracker.update.call_args_list:
            raw_dets = call_args[0][0]
            for det in raw_dets:
                assert isinstance(det, RawDetection)

    def test_single_frame_returns_detection(self, make_mocks, mocker):
        detector, tracker = make_mocks
        config = PipelineConfig()
        config.pipeline_mode = PipelineMode.PIPELINED
        config.output.save_video = False
        config.output.save_csv = False

        pipeline = PipelinedPipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 1
        mock_video.frames.return_value = _make_frames(1)

        mocker.patch(
            "afora_vc.application.pipelined_pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        result = pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        assert result.frames_processed == 1
        tracker.update.assert_called_once()
        output = tracker.update.return_value
        assert all(isinstance(d, Detection) for d in output)


class TestPipelinedPipelineBatchMode:
    """Con batch_size>1, el hilo detector acumula y llama detect_batch."""

    def test_batch_mode_calls_detect_batch(self, mocker):
        detector = mocker.MagicMock()
        detector.supports_batch = True
        detector.detect_batch.return_value = [
            [_raw_detection()] for _ in range(4)
        ]

        tracker = mocker.MagicMock()
        tracker.update.return_value = [_detection()]

        config = PipelineConfig()
        config.pipeline_mode = PipelineMode.PIPELINED
        config.output.save_video = False
        config.output.save_csv = False
        config.batch.enabled = True
        config.batch.batch_size = 4

        pipeline = PipelinedPipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 8
        mock_video.frames.return_value = _make_frames(8)

        mocker.patch(
            "afora_vc.application.pipelined_pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        result = pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        assert result.frames_processed == 8
        assert detector.detect.call_count == 0
        assert detector.detect_batch.call_count == 2
        assert tracker.update.call_count == 8

    def test_batch_disabled_uses_single_detect(self, mocker):
        detector = mocker.MagicMock()
        detector.detect.return_value = [_raw_detection()]
        detector.supports_batch = False

        tracker = mocker.MagicMock()
        tracker.update.return_value = [_detection()]

        config = PipelineConfig()
        config.pipeline_mode = PipelineMode.PIPELINED
        config.output.save_video = False
        config.output.save_csv = False
        config.batch.enabled = False
        config.batch.batch_size = 1

        pipeline = PipelinedPipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 4
        mock_video.frames.return_value = _make_frames(4)

        mocker.patch(
            "afora_vc.application.pipelined_pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        result = pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        assert result.frames_processed == 4
        assert detector.detect.call_count == 4
        assert detector.detect_batch.call_count == 0
        assert tracker.update.call_count == 4


class TestPipelinedPipelineCleanup:
    """Validar que los hilos se cierran correctamente y no quedan colgados."""

    def test_threads_terminate_after_video_ends(self, mocker):
        detector = mocker.MagicMock()
        detector.detect.return_value = [_raw_detection()]
        detector.supports_batch = False

        tracker = mocker.MagicMock()
        tracker.update.return_value = [_detection()]

        config = PipelineConfig()
        config.pipeline_mode = PipelineMode.PIPELINED
        config.output.save_video = False
        config.output.save_csv = False

        pipeline = PipelinedPipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 10
        mock_video.frames.return_value = _make_frames(10)

        mocker.patch(
            "afora_vc.application.pipelined_pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        import threading
        active_before = threading.active_count()

        pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        time.sleep(0.5)
        active_after = threading.active_count()
        assert active_after <= active_before + 1
