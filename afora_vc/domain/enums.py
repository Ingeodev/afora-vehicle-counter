from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class DetectorBackend(str, Enum):
    YOLO = "yolo"
    DFINE = "dfine"
    RTDETR = "rtdetr"


class TrackerBackend(str, Enum):
    BOTSORT = "botsort"
    BYTETRACK = "bytetrack"


class PreprocessMode(str, Enum):
    NONE = "none"
    DEBLUR = "deblur"
    NIGHT_ENHANCE = "night_enhance"
    CUSTOM = "custom"


class DetectorStrategy(str, Enum):
    BOX = "box"
    SEG = "seg"
