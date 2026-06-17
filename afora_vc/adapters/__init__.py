from .detectors._registry import DetectorRegistry
from .trackers._registry import TrackerRegistry


def _register_default_detectors():
    try:
        from .detectors.yolo_adapter import YOLODetectorAdapter
        DetectorRegistry.register("yolo", YOLODetectorAdapter)
    except ImportError:
        pass

    try:
        from .detectors.dfine_adapter import DFineDetectorAdapter
        DetectorRegistry.register("dfine", DFineDetectorAdapter)
    except ImportError:
        pass

    try:
        from .detectors.rtdetr_adapter import RTDETRDetectorAdapter
        DetectorRegistry.register("rtdetr", RTDETRDetectorAdapter)
    except ImportError:
        pass


def _register_default_trackers():
    try:
        from .trackers.botsort_adapter import BotSORTTrackerAdapter
        TrackerRegistry.register("botsort", BotSORTTrackerAdapter)
    except ImportError:
        pass

    try:
        from .trackers.bytetrack_adapter import ByteTrackTrackerAdapter
        TrackerRegistry.register("bytetrack", ByteTrackTrackerAdapter)
    except ImportError:
        pass


_register_default_detectors()
_register_default_trackers()


__all__ = ["DetectorRegistry", "TrackerRegistry"]
