import numpy as np

from afora_vc.domain.schemas import (
    BoundingBox,
    RawDetection,
    Detection,
    ZoneConfig,
    ZoneEntry,
    TrackHistory,
    ProcessingResult,
    VideoInfo,
    FrameData,
)


class TestBoundingBox:
    def test_center_calculation(self):
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        assert bbox.center == (150, 150)

    def test_center_odd_dimensions(self):
        bbox = BoundingBox(x1=101, y1=102, x2=203, y2=204)
        cx, cy = bbox.center
        assert cx == (101 + 203) // 2
        assert cy == (102 + 204) // 2

    def test_dimensions(self):
        bbox = BoundingBox(x1=100, y1=50, x2=300, y2=200)
        assert bbox.width == 200
        assert bbox.height == 150

    def test_dimensions_always_positive(self):
        bbox = BoundingBox(x1=50, y1=50, x2=100, y2=100)
        assert bbox.width >= 0
        assert bbox.height >= 0


class TestRawDetection:
    def test_no_track_id(self):
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        det = RawDetection(class_id=2, class_name="car", bbox=bbox, confidence=0.9)
        assert not hasattr(det, "track_id")

    def test_center_from_bbox(self):
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        det = RawDetection(class_id=2, class_name="car", bbox=bbox)
        assert det.center == (150, 150)

    def test_center_from_mask(self):
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        det = RawDetection(
            class_id=2, class_name="car", bbox=bbox,
            mask_center=(120, 130),
        )
        assert det.center == (120, 130)


class TestDetection:
    def test_requires_track_id(self):
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(track_id=5, class_id=2, class_name="car", bbox=bbox)
        assert det.track_id == 5

    def test_center_from_bbox(self):
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        det = Detection(track_id=1, class_id=2, class_name="car", bbox=bbox)
        assert det.center == (150, 150)

    def test_center_prefers_mask_center(self):
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        det = Detection(
            track_id=1, class_id=2, class_name="car", bbox=bbox,
            mask_center=(300, 400),
        )
        assert det.center == (300, 400)

    def test_from_raw(self):
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        raw = RawDetection(class_id=2, class_name="car", bbox=bbox, confidence=0.9)
        det = Detection.from_raw(raw, track_id=42)
        assert det.track_id == 42
        assert det.class_id == 2
        assert det.class_name == "car"
        assert det.confidence == 0.9
        assert det.center == (5, 5)

    def test_from_raw_with_mask(self):
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
        raw = RawDetection(
            class_id=2, class_name="car", bbox=bbox,
            confidence=0.9, mask_polygon=poly, mask_center=(5, 5),
        )
        det = Detection.from_raw(raw, track_id=7)
        assert det.mask_polygon is not None
        assert det.mask_center == (5, 5)


class TestZoneConfig:
    def test_from_dict(self):
        data = {"label": "A", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}
        zone = ZoneConfig.from_dict(data)
        assert zone.label == "A"
        assert zone.points.shape == (4, 2)

    def test_scale_immutable(self):
        original = ZoneConfig.from_dict({
            "label": "A",
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
        })
        scaled = original.scale(0.5, 0.5)
        assert original.points[1][0] == 100
        assert scaled.points[1][0] == 50

    def test_scale_returns_new_instance(self):
        original = ZoneConfig.from_dict({
            "label": "A",
            "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
        })
        scaled = original.scale(2.0, 2.0)
        assert scaled is not original
        assert scaled.label == original.label


class TestProcessingResult:
    def test_time_minutes(self):
        result = ProcessingResult(
            video_path="test.mp4",
            output_video_path="out.mp4",
            csv_path="out.csv",
            total_detections=10,
            zone_counts={"A": 5, "B": 5},
            processing_time_seconds=120.0,
            frames_processed=1000,
        )
        assert result.processing_time_minutes == 2.0

    def test_zero_time(self):
        result = ProcessingResult(
            video_path="test.mp4",
            output_video_path=None,
            csv_path=None,
            total_detections=0,
            zone_counts={},
            processing_time_seconds=0.0,
            frames_processed=0,
        )
        assert result.processing_time_minutes == 0.0


class TestTrackHistory:
    def test_add_position(self):
        th = TrackHistory(track_id=1, vehicle_type="car")
        th.add_position(100, 200)
        assert len(th.positions) == 1
        assert th.positions[0] == (100, 200)

    def test_last_position(self):
        th = TrackHistory(track_id=1, vehicle_type="car")
        th.add_position(100, 200)
        th.add_position(150, 250)
        assert th.last_position == (150, 250)

    def test_last_position_empty(self):
        th = TrackHistory(track_id=1)
        assert th.last_position is None

    def test_previous_position(self):
        th = TrackHistory(track_id=1, vehicle_type="car")
        th.add_position(100, 200)
        th.add_position(150, 250)
        assert th.previous_position == (100, 200)

    def test_previous_position_insufficient(self):
        th = TrackHistory(track_id=1, vehicle_type="car")
        th.add_position(100, 200)
        assert th.previous_position is None


class TestVideoInfo:
    def test_duration_from_fps_frames(self):
        info = VideoInfo(path="test.mp4", hora_inicial="00:00", fps=30.0, total_frames=300)
        assert info.duration_seconds == 10.0
        assert info.duration_minutes == 10.0 / 60

    def test_duration_none_when_no_fps(self):
        info = VideoInfo(path="test.mp4", hora_inicial="00:00", total_frames=300)
        assert info.duration_seconds is None
        assert info.duration_minutes is None


class TestZoneEntry:
    def test_create_entry(self):
        entry = ZoneEntry(
            vehicle_id=1,
            vehicle_type="car",
            zone="A",
            date="2026-01-01",
            exact_time="14:30:00",
            timestamp_formatted="00:30",
        )
        assert entry.vehicle_id == 1
        assert entry.zone == "A"
        assert entry.vehicle_type == "car"


class TestFrameData:
    def test_create_frame_data(self):
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        fd = FrameData(frame=frame, index=42, timestamp_seconds=1.4)
        assert fd.index == 42
        assert fd.timestamp_seconds == 1.4
        assert fd.frame.shape == (100, 100, 3)
