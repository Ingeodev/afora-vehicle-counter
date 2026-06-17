import numpy as np
import pytest
import torch

from afora_vc.domain.schemas import RawDetection
from afora_vc.domain.enums import DeviceType
from afora_vc.domain.exceptions import BackendNotInstalledError


@pytest.fixture
def make_mock_result():
    def _make(scores, labels, boxes):
        return {
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }

    return _make


class TestDFineParseResults:
    def test_returns_raw_detections(self, make_mock_result):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._classes = {2: "car", 7: "truck"}
        adapter._allowed_ids = {2, 7}
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = make_mock_result(
            [0.9, 0.8],
            [2, 7],
            [[10.2, 20.7, 100.3, 200.1], [50.8, 60.3, 150.9, 250.4]],
        )

        detections = adapter._parse_results(result)

        assert len(detections) == 2
        for det in detections:
            assert isinstance(det, RawDetection)
            assert not hasattr(det, "track_id")

    def test_has_no_track_id(self, make_mock_result):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._classes = {2: "car"}
        adapter._allowed_ids = {2}
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = make_mock_result([0.9], [2], [[10.0, 20.0, 100.0, 200.0]])
        detections = adapter._parse_results(result)

        assert len(detections) == 1
        det = detections[0]
        assert det.class_id == 2
        assert det.class_name == "car"
        assert det.confidence == pytest.approx(0.9)
        assert det.bbox.x1 == 10
        assert det.bbox.y1 == 20
        assert det.bbox.x2 == 100
        assert det.bbox.y2 == 200

    def test_rounds_floats_to_ints(self, make_mock_result):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._classes = {2: "car"}
        adapter._allowed_ids = {2}
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = make_mock_result([0.9], [2], [[10.7, 20.3, 99.5, 199.2]])
        detections = adapter._parse_results(result)

        assert detections[0].bbox.x1 == 11
        assert detections[0].bbox.y1 == 20
        assert detections[0].bbox.x2 == 100
        assert detections[0].bbox.y2 == 199

    def test_filters_low_confidence(self, make_mock_result):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._classes = {2: "car"}
        adapter._allowed_ids = {2}
        adapter._confidence = 0.7
        adapter._class_thresholds = {}

        result = make_mock_result([0.5], [2], [[10, 20, 100, 200]])
        detections = adapter._parse_results(result)

        assert len(detections) == 0

    def test_filters_disallowed_classes(self, make_mock_result):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._classes = {2: "car"}
        adapter._allowed_ids = {2}
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = make_mock_result(
            [0.9, 0.8],
            [2, 7],
            [[10, 20, 100, 200], [50, 60, 150, 250]],
        )

        detections = adapter._parse_results(result)
        assert len(detections) == 1
        assert detections[0].class_id == 2

    def test_class_thresholds_override(self, make_mock_result):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._classes = {2: "car", 7: "truck"}
        adapter._allowed_ids = {2, 7}
        adapter._confidence = 0.5
        adapter._class_thresholds = {"truck": 0.9}

        result = make_mock_result(
            [0.9, 0.85],
            [2, 7],
            [[10, 20, 100, 200], [50, 60, 150, 250]],
        )

        detections = adapter._parse_results(result)
        assert len(detections) == 1
        assert detections[0].class_id == 2

    def test_empty_result(self):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._classes = {2: "car"}
        adapter._allowed_ids = {2}
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = {"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.empty(0, 4)}
        assert adapter._parse_results(result) == []

    def test_result_without_scores(self):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        assert adapter._parse_results({}) == []

    def test_class_name_resolution(self, make_mock_result):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._classes = {3: "motorcycle"}
        adapter._allowed_ids = {3}
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        result = make_mock_result([0.9], [3], [[10, 20, 100, 200]])
        detections = adapter._parse_results(result)

        assert detections[0].class_name == "motorcycle"


class TestDFineDetectorAdapterInit:
    def test_import_error_on_missing_transformers(self, mocker):
        orig_import = __import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("transformers"):
                raise ImportError(f"No module named '{name}'")
            return orig_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=mock_import)

        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        with pytest.raises(BackendNotInstalledError):
            DFineDetectorAdapter()

    def test_warmup_runs_without_error(self, mocker):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._processor = mocker.MagicMock()
        adapter._model = mocker.MagicMock()
        adapter._device_str = "cpu"
        adapter._classes = {2: "car"}
        adapter._allowed_ids = {2}
        adapter._confidence = 0.5
        adapter._class_thresholds = {}

        mock_result = {
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([2]),
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0]]),
        }
        adapter._processor.post_process_object_detection.return_value = [mock_result]

        adapter.warmup((640, 480, 3))
        adapter._model.assert_called_once()

    def test_supports_batch_is_true(self):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        assert adapter.supports_batch is True

    def test_detect_batch_count_matches_input(self, mocker):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        mock_processor = mocker.MagicMock()
        mock_model = mocker.MagicMock()
        mock_result = {
            "scores": mocker.MagicMock(),
            "labels": mocker.MagicMock(),
            "boxes": mocker.MagicMock(),
        }
        mock_result["scores"].__len__.return_value = 0
        mock_result["labels"].__len__.return_value = 0
        mock_result["boxes"].__len__.return_value = 0
        mock_processor.post_process_object_detection.return_value = [mock_result] * 3

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        adapter._processor = mock_processor
        adapter._model = mock_model
        adapter._device_str = "cpu"
        adapter._device = DeviceType.CPU
        adapter._confidence = 0.5
        adapter._allowed_ids = {2, 7}
        adapter._classes = {2: "car", 7: "truck"}
        adapter._class_thresholds = {}

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = adapter.detect_batch(frames)
        assert len(results) == 3

    def test_class_names_returns_copy(self):
        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter.__new__(DFineDetectorAdapter)
        classes = {2: "car", 7: "truck"}
        adapter._classes = classes

        result = adapter.class_names
        assert result == classes
        result[2] = "modified"
        assert adapter._classes[2] == "car"

    def test_device_cpu_creates_model(self, mocker):
        mocker.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "transformers.DFineForObjectDetection.from_pretrained",
            return_value=mocker.MagicMock(),
        )

        from afora_vc.adapters.detectors.dfine_adapter import DFineDetectorAdapter

        adapter = DFineDetectorAdapter(
            model_path="ustc-community/dfine-nano-coco",
            device=DeviceType.CPU,
        )

        assert adapter._device == DeviceType.CPU
        assert adapter._classes == {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
