"""Tests para YOLODetectorAdapter usando mock del modelo."""

import numpy as np
import pytest

from afora_vc.domain.schemas import RawDetection
from afora_vc.domain.enums import DeviceType
from afora_vc.domain.exceptions import BackendNotInstalledError


@pytest.fixture
def mock_tensor():
    class MockTensor:
        def __init__(self, data):
            self._data = data if isinstance(data, np.ndarray) else np.array(data)
        def cpu(self):
            return self
        def numpy(self):
            return self._data
    return MockTensor


@pytest.fixture
def make_mock_result(mock_tensor):
    def _make(boxes_data, cls_data, conf_data, masks_xy=None):
        class MockBoxes:
            def __init__(self):
                self.xyxy = mock_tensor(boxes_data)
                self.cls = mock_tensor(cls_data)
                self.conf = mock_tensor(conf_data)

        class MockMasks:
            def __init__(self, xy):
                self.xy = xy

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes()
                self.masks = None
                if masks_xy is not None:
                    self.masks = MockMasks(masks_xy)

        return MockResult()
    return _make


class TestYOLODetectorParseResults:
    """Tests de _parseResults con resultados mock."""

    def test_returns_raw_detections(self, make_mock_result):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        adapter._classes = {2: "car", 7: "truck"}
        adapter._allowed_ids = [2, 7]
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = make_mock_result(
            [[10, 20, 100, 200], [50, 60, 150, 250]],
            [2, 7],
            [0.9, 0.8],
        )

        detections = adapter._parse_results([result])

        assert len(detections) == 2
        for det in detections:
            assert isinstance(det, RawDetection)
            assert not hasattr(det, "track_id")

    def test_has_no_track_id(self, make_mock_result):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        adapter._classes = {2: "car"}
        adapter._allowed_ids = [2]
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = make_mock_result([[10, 20, 100, 200]], [2], [0.9])
        detections = adapter._parse_results([result])

        assert len(detections) == 1
        det = detections[0]
        assert det.class_id == 2
        assert det.class_name == "car"
        assert det.confidence == 0.9
        assert det.bbox.x1 == 10
        assert det.bbox.y1 == 20
        assert det.bbox.x2 == 100
        assert det.bbox.y2 == 200

    def test_filters_low_confidence(self, make_mock_result):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        adapter._classes = {2: "car"}
        adapter._allowed_ids = [2]
        adapter._confidence = 0.7
        adapter._class_thresholds = {}

        result = make_mock_result([[10, 20, 100, 200]], [2], [0.5])
        detections = adapter._parse_results([result])

        assert len(detections) == 0

    def test_filters_disallowed_classes(self, make_mock_result):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        adapter._classes = {2: "car"}
        adapter._allowed_ids = [2]
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        # class_id=7 (truck) should be filtered out
        result = make_mock_result(
            [[10, 20, 100, 200], [50, 60, 150, 250]],
            [2, 7],
            [0.9, 0.8],
        )

        detections = adapter._parse_results([result])
        assert len(detections) == 1
        assert detections[0].class_id == 2

    def test_class_thresholds_override(self, make_mock_result):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        adapter._classes = {2: "car", 7: "truck"}
        adapter._allowed_ids = [2, 7]
        adapter._confidence = 0.5
        # truck has higher threshold
        adapter._class_thresholds = {"truck": 0.9}

        result = make_mock_result(
            [[10, 20, 100, 200], [50, 60, 150, 250]],
            [2, 7],
            [0.9, 0.85],
        )

        detections = adapter._parse_results([result])
        assert len(detections) == 1
        assert detections[0].class_id == 2  # car (0.9 >= 0.5), truck (0.85 < 0.9)

    def test_empty_result(self, make_mock_result):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        assert adapter._parse_results([]) == []

    def test_result_without_boxes(self, make_mock_result):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        class ResultNoBoxes:
            def __init__(self):
                self.boxes = None

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        assert adapter._parse_results([ResultNoBoxes()]) == []

    def test_returns_raw_detections_has_class_name(self, make_mock_result):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        adapter._classes = {3: "motorcycle"}
        adapter._allowed_ids = [3]
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = make_mock_result([[10, 20, 100, 200]], [3], [0.9])
        detections = adapter._parse_results([result])

        assert detections[0].class_name == "motorcycle"


class TestYOLODetectorAdapterInit:
    def test_import_error_on_missing_ultralytics(self, mocker):
        mocker.patch.dict("sys.modules", {"ultralytics": None})
        import sys
        # Force the import to fail by removing ultralytics
        orig_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "ultralytics":
                raise ImportError("No module named 'ultralytics'")
            return orig_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=mock_import)

        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        with pytest.raises(BackendNotInstalledError):
            YOLODetectorAdapter()

    def test_warmup_runs_without_error(self, mocker):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        mock_model = mocker.MagicMock()

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        adapter._model = mock_model
        adapter._device_str = "cpu"
        adapter._half = False
        adapter._allowed_ids = [2]
        adapter._classes = {2: "car"}
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        adapter.warmup((640, 480, 3))
        mock_model.assert_called_once()

    def test_supports_batch_is_true(self):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        assert adapter.supports_batch is True

    def test_class_names_returns_copy(self):
        from afora_vc.adapters.detectors.yolo_adapter import YOLODetectorAdapter

        adapter = YOLODetectorAdapter.__new__(YOLODetectorAdapter)
        classes = {2: "car", 7: "truck"}
        adapter._classes = classes

        result = adapter.class_names
        assert result == classes
        result[2] = "modified"
        assert adapter._classes[2] == "car"
