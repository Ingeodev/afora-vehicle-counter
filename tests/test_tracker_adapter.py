"""Tests para BotSORTTrackerAdapter con mock del tracker interno."""

import numpy as np
import pytest

from afora_vc.domain.schemas import RawDetection, Detection, BoundingBox


class TestBotSORTTrackerAdapter:
    def test_assigns_track_ids(self, mocker):
        from afora_vc.adapters.trackers.botsort_adapter import BotSORTTrackerAdapter

        mock_tracker = mocker.MagicMock()
        mock_tracker.update.return_value = np.array([
            [10, 20, 100, 200, 1, 0.9, 2],
            [50, 60, 150, 250, 2, 0.8, 7],
        ])

        adapter = BotSORTTrackerAdapter()
        adapter._tracker = mock_tracker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = [
            RawDetection(class_id=2, class_name="car", confidence=0.9,
                         bbox=BoundingBox(10, 20, 100, 200)),
            RawDetection(class_id=7, class_name="truck", confidence=0.8,
                         bbox=BoundingBox(50, 60, 150, 250)),
        ]

        detections = adapter.update(raw, frame)

        assert len(detections) == 2
        assert all(isinstance(d, Detection) for d in detections)
        assert detections[0].track_id == 1
        assert detections[1].track_id == 2
        assert detections[0].class_name == "car"
        assert detections[1].class_name == "truck"

    def test_consistent_ids_across_frames(self, mocker):
        from afora_vc.adapters.trackers.botsort_adapter import BotSORTTrackerAdapter

        mock_tracker = mocker.MagicMock()
        mock_tracker.update.side_effect = [
            np.array([[10, 20, 100, 200, 1, 0.9, 2],
                      [50, 60, 150, 250, 2, 0.8, 7]]),
            np.array([[12, 22, 102, 202, 1, 0.9, 2],
                      [52, 62, 152, 252, 2, 0.8, 7]]),
        ]

        adapter = BotSORTTrackerAdapter()
        adapter._tracker = mock_tracker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = [
            RawDetection(class_id=2, class_name="car", confidence=0.9,
                         bbox=BoundingBox(10, 20, 100, 200)),
            RawDetection(class_id=7, class_name="truck", confidence=0.8,
                         bbox=BoundingBox(50, 60, 150, 250)),
        ]

        frame1 = adapter.update(raw, frame)
        frame2 = adapter.update(raw, frame)

        assert len(frame1) == 2
        assert len(frame2) == 2
        assert frame1[0].track_id == frame2[0].track_id
        assert frame1[1].track_id == frame2[1].track_id

    def test_reset_clears_state(self, mocker):
        from afora_vc.adapters.trackers.botsort_adapter import BotSORTTrackerAdapter

        mock_tracker = mocker.MagicMock()
        mock_tracker.update.return_value = np.array([
            [10, 20, 100, 200, 1, 0.9, 2],
        ])

        adapter = BotSORTTrackerAdapter()
        adapter._tracker = mock_tracker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = [
            RawDetection(class_id=2, class_name="car", confidence=0.9,
                         bbox=BoundingBox(10, 20, 100, 200)),
        ]

        adapter.update(raw, frame)
        old_offset = adapter._id_offset

        adapter.reset()

        assert adapter._id_offset > old_offset
        assert adapter._tracker is None
        assert adapter._frame_count == 0

    def test_handles_empty_detections(self, mocker):
        from afora_vc.adapters.trackers.botsort_adapter import BotSORTTrackerAdapter

        adapter = BotSORTTrackerAdapter()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = adapter.update([], frame)
        assert detections == []

    def test_handles_single_detection(self, mocker):
        from afora_vc.adapters.trackers.botsort_adapter import BotSORTTrackerAdapter

        mock_tracker = mocker.MagicMock()
        mock_tracker.update.return_value = np.array([
            [10, 20, 100, 200, 1, 0.9, 2],
        ])

        adapter = BotSORTTrackerAdapter()
        adapter._tracker = mock_tracker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = [
            RawDetection(class_id=2, class_name="car", confidence=0.9,
                         bbox=BoundingBox(10, 20, 100, 200)),
        ]

        detections = adapter.update(raw, frame)
        assert len(detections) == 1
        assert detections[0].track_id == 1
        assert detections[0].class_name == "car"

    def test_resolve_class_name_from_raw(self, mocker):
        """Verifica que el class_name se resuelve del raw_detection aun si cls_id es distinto."""
        from afora_vc.adapters.trackers.botsort_adapter import BotSORTTrackerAdapter

        mock_tracker = mocker.MagicMock()
        # El tracker retorna cls_id=2, pero raw tiene class_id=99 mapeado a "custom"
        mock_tracker.update.return_value = np.array([
            [10, 20, 100, 200, 42, 0.9, 99],
        ])

        adapter = BotSORTTrackerAdapter()
        adapter._tracker = mock_tracker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = [
            RawDetection(class_id=99, class_name="custom_type", confidence=0.9,
                         bbox=BoundingBox(10, 20, 100, 200)),
        ]

        detections = adapter.update(raw, frame)
        assert len(detections) == 1
        assert detections[0].class_name == "custom_type"
        assert detections[0].track_id == 42 + 0  # primer frame, id_offset=0
