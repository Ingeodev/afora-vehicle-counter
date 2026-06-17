"""Tests de integracion del pipeline con detector y tracker mock."""

import numpy as np
import pytest

from afora_vc.domain.schemas import RawDetection, Detection, BoundingBox
from afora_vc.domain.enums import DeviceType, DetectorBackend, TrackerBackend
from afora_vc.application.config import PipelineConfig
from afora_vc.application.pipeline import Pipeline


class TestPipelineDetectTrack:
    """Verifica que detect+track como fases separadas producen Detection."""

    @pytest.fixture
    def make_mocks(self, mocker):
        detector = mocker.MagicMock()
        detector.detect.return_value = [
            RawDetection(class_id=2, class_name="car", confidence=0.9,
                         bbox=BoundingBox(10, 20, 100, 200)),
            RawDetection(class_id=7, class_name="truck", confidence=0.8,
                         bbox=BoundingBox(50, 60, 150, 250)),
        ]
        detector.supports_batch = False

        tracker = mocker.MagicMock()
        tracker.update.return_value = [
            Detection(track_id=1, class_id=2, class_name="car", confidence=0.9,
                      bbox=BoundingBox(10, 20, 100, 200)),
            Detection(track_id=2, class_id=7, class_name="truck", confidence=0.8,
                      bbox=BoundingBox(50, 60, 150, 250)),
        ]

        return detector, tracker

    def test_detect_then_track_produces_detections(self, make_mocks, mocker):
        """El pipeline ejecuta detect -> track como fases separadas y produce Detection con track_id."""
        detector, tracker = make_mocks
        config = PipelineConfig()
        config.output.save_video = False
        config.output.save_csv = False

        pipeline = Pipeline(config=config, detector=detector, tracker=tracker)

        # Mock video source
        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 10

        from afora_vc.domain.schemas import FrameData
        mock_video.frames.return_value = [
            FrameData(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                      index=i, timestamp_seconds=i / 30)
            for i in range(10)
        ]

        mocker.patch(
            "afora_vc.application.pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        result = pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        assert result.frames_processed == 10
        assert detector.detect.call_count == 10
        assert tracker.update.call_count == 10

    def test_raw_detection_flows_through_tracker(self, make_mocks, mocker):
        """Verifica que lo que detect retorna (RawDetection) se pasa al tracker."""
        detector, tracker = make_mocks
        config = PipelineConfig()
        config.output.save_video = False
        config.output.save_csv = False

        pipeline = Pipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 3

        from afora_vc.domain.schemas import FrameData
        mock_video.frames.return_value = [
            FrameData(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                      index=i, timestamp_seconds=i / 30)
            for i in range(3)
        ]

        mocker.patch(
            "afora_vc.application.pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        # Cada llamada a tracker.update recibe list[RawDetection]
        for call_args in tracker.update.call_args_list:
            raw_dets = call_args[0][0]  # primer arg posicional
            for det in raw_dets:
                assert isinstance(det, RawDetection)
                assert not hasattr(det, "track_id")

        # Y retorna list[Detection]
        for call in tracker.update.call_args_list:
            dets = tracker.update.return_value
            for d in dets:
                assert isinstance(d, Detection)

    def test_pipeline_without_tracker_returns_raw(self, mocker):
        """Verifica el flujo detector->RawDetection sin tracker."""
        detector = mocker.MagicMock()
        detector.detect.return_value = [
            RawDetection(class_id=2, class_name="car", confidence=0.9,
                         bbox=BoundingBox(10, 20, 100, 200)),
        ]
        detector.supports_batch = False

        tracker = mocker.MagicMock()
        tracker.update.return_value = [
            Detection(track_id=42, class_id=2, class_name="car", confidence=0.9,
                      bbox=BoundingBox(10, 20, 100, 200)),
        ]

        config = PipelineConfig()
        config.output.save_video = False
        config.output.save_csv = False

        pipeline = Pipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 1

        from afora_vc.domain.schemas import FrameData
        mock_video.frames.return_value = [
            FrameData(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                      index=0, timestamp_seconds=0)
        ]

        mocker.patch(
            "afora_vc.application.pipeline.VideoSource.from_path",
            return_value=mock_video,
        )

        result = pipeline.process_video(
            video_path="/fake/video.mp4",
            output_folder="/tmp/test_output",
            base_time="00:00",
            date="2025-01-01",
        )

        # tracker debe haber recibido RawDetection y retornado Detection
        tracker.update.assert_called_once()
        raw_input = tracker.update.call_args[0][0]
        assert all(isinstance(d, RawDetection) for d in raw_input)


class TestPipelineBatchMode:
    """Verifica que batch mode acumula frames y llama a detect_batch."""

    def test_batch_mode_accumulates(self, mocker):
        detector = mocker.MagicMock()
        detector.supports_batch = True
        detector.detect_batch.return_value = [
            [RawDetection(class_id=2, class_name="car", confidence=0.9,
                          bbox=BoundingBox(10, 20, 100, 200))]
            for _ in range(4)
        ]

        tracker = mocker.MagicMock()
        tracker.update.return_value = [
            Detection(track_id=1, class_id=2, class_name="car", confidence=0.9,
                      bbox=BoundingBox(10, 20, 100, 200)),
        ]

        config = PipelineConfig()
        config.output.save_video = False
        config.output.save_csv = False
        config.batch.enabled = True
        config.batch.batch_size = 4

        pipeline = Pipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 8

        from afora_vc.domain.schemas import FrameData
        mock_video.frames.return_value = [
            FrameData(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                      index=i, timestamp_seconds=i / 30)
            for i in range(8)
        ]

        mocker.patch(
            "afora_vc.application.pipeline.VideoSource.from_path",
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

    def test_batch_disabled_still_uses_detect(self, mocker):
        detector = mocker.MagicMock()
        detector.detect.return_value = [
            RawDetection(class_id=2, class_name="car", confidence=0.9,
                         bbox=BoundingBox(10, 20, 100, 200)),
            RawDetection(class_id=7, class_name="truck", confidence=0.8,
                         bbox=BoundingBox(50, 60, 150, 250)),
        ]
        detector.supports_batch = False

        tracker = mocker.MagicMock()
        tracker.update.return_value = [
            Detection(track_id=1, class_id=2, class_name="car", confidence=0.9,
                      bbox=BoundingBox(10, 20, 100, 200)),
            Detection(track_id=2, class_id=7, class_name="truck", confidence=0.8,
                      bbox=BoundingBox(50, 60, 150, 250)),
        ]

        config = PipelineConfig()
        config.output.save_video = False
        config.output.save_csv = False
        config.batch.enabled = False

        pipeline = Pipeline(config=config, detector=detector, tracker=tracker)

        mock_video = mocker.MagicMock()
        mock_video.fps = 30
        mock_video.width = 640
        mock_video.height = 480
        mock_video.total_frames = 4

        from afora_vc.domain.schemas import FrameData
        mock_video.frames.return_value = [
            FrameData(frame=np.zeros((480, 640, 3), dtype=np.uint8),
                      index=i, timestamp_seconds=i / 30)
            for i in range(4)
        ]

        mocker.patch(
            "afora_vc.application.pipeline.VideoSource.from_path",
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
        assert tracker.update.call_count == 4
