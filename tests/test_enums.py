from afora_vc.domain.enums import (
    DeviceType,
    DetectorBackend,
    TrackerBackend,
    PreprocessMode,
    DetectorStrategy,
)


class TestDeviceType:
    def test_values(self):
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"

    def test_from_string(self):
        assert DeviceType("cpu") == DeviceType.CPU
        assert DeviceType("cuda") == DeviceType.CUDA
        assert DeviceType("mps") == DeviceType.MPS


class TestDetectorBackend:
    def test_values(self):
        assert DetectorBackend.YOLO.value == "yolo"
        assert DetectorBackend.DFINE.value == "dfine"
        assert DetectorBackend.RTDETR.value == "rtdetr"

    def test_from_string(self):
        assert DetectorBackend("yolo") == DetectorBackend.YOLO


class TestTrackerBackend:
    def test_values(self):
        assert TrackerBackend.BOTSORT.value == "botsort"
        assert TrackerBackend.BYTETRACK.value == "bytetrack"


class TestPreprocessMode:
    def test_values(self):
        assert PreprocessMode.NONE.value == "none"
        assert PreprocessMode.DEBLUR.value == "deblur"
        assert PreprocessMode.NIGHT_ENHANCE.value == "night_enhance"
        assert PreprocessMode.CUSTOM.value == "custom"


class TestDetectorStrategy:
    def test_values(self):
        assert DetectorStrategy.BOX.value == "box"
        assert DetectorStrategy.SEG.value == "seg"
