from importlib.metadata import PackageNotFoundError, version as _package_version

try:  # en un clon sin instalar no hay metadata: se usa la version del pyproject
    __version__ = _package_version("afora-vehicle-counter")
except PackageNotFoundError:  # pragma: no cover - depende del entorno
    __version__ = "2.0.0-alpha.1"

from afora_vc.application.config import PipelineConfig, VideoConfig, DetectorConfig, TrackerConfig, OutputConfig, PreprocessConfig, BatchConfig as PipelineBatchConfig
from afora_vc.application.batch_config import BatchConfig, VideoOverride
from afora_vc.application.facade import process_video, process_directory, process_from_config
from afora_vc.utils.progress import ProgressTracker
from afora_vc.domain.enums import DeviceType, DetectorBackend, TrackerBackend, PreprocessMode, DetectorStrategy, PipelineMode
from afora_vc.domain.schemas import ZoneConfig, VideoInfo, BoundingBox, RawDetection, Detection, ZoneEntry, ProcessingResult, TrackHistory, FrameData
from afora_vc.domain.exceptions import AforosError, VideoNotFoundError, VideoOpenError, ConfigurationError, OutputDirectoryError, ProcessingError, BackendNotInstalledError, BackendNotRegisteredError, DeviceNotAvailableError

__all__ = [
    "__version__",
    "PipelineConfig", "VideoConfig", "DetectorConfig", "TrackerConfig", "OutputConfig", "PreprocessConfig",
    "PipelineBatchConfig",
    "BatchConfig", "VideoOverride",
    "ProgressTracker",
    "process_video", "process_directory", "process_from_config",
    "DeviceType", "DetectorBackend", "TrackerBackend", "PreprocessMode", "DetectorStrategy", "PipelineMode",
    "ZoneConfig", "VideoInfo", "BoundingBox", "RawDetection", "Detection", "ZoneEntry", "ProcessingResult", "TrackHistory", "FrameData",
    "AforosError", "VideoNotFoundError", "VideoOpenError", "ConfigurationError", "OutputDirectoryError",
    "ProcessingError", "BackendNotInstalledError", "BackendNotRegisteredError", "DeviceNotAvailableError",
]
