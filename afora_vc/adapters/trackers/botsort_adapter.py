import os
from types import SimpleNamespace

import numpy as np

from ...domain.schemas import RawDetection, Detection, BoundingBox
from ...domain.enums import DeviceType


class _DetectionsBridge:
    """Adapta list[RawDetection] a la interfaz que BOTSORT.update() espera.

    BOTSORT.update() accede a: results.conf, results.cls, results.xyxy,
    results.xywh, results[mask], len(results).
    """

    def __init__(self, raw_detections: list[RawDetection]):
        n = len(raw_detections)
        self._xyxy = np.zeros((n, 4), dtype=np.float32)
        self._conf = np.zeros(n, dtype=np.float32)
        self._cls = np.zeros(n, dtype=np.float32)

        for i, det in enumerate(raw_detections):
            self._xyxy[i] = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            self._conf[i] = det.confidence
            self._cls[i] = det.class_id

        # xywh derivado de xyxy
        self._xywh = np.zeros((n, 4), dtype=np.float32)
        self._xywh[:, 0] = (self._xyxy[:, 0] + self._xyxy[:, 2]) / 2  # cx
        self._xywh[:, 1] = (self._xyxy[:, 1] + self._xyxy[:, 3]) / 2  # cy
        self._xywh[:, 2] = self._xyxy[:, 2] - self._xyxy[:, 0]  # w
        self._xywh[:, 3] = self._xyxy[:, 3] - self._xyxy[:, 1]  # h

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    @property
    def xywh(self) -> np.ndarray:
        return self._xywh

    @property
    def conf(self) -> np.ndarray:
        return self._conf

    @property
    def cls(self) -> np.ndarray:
        return self._cls

    def __len__(self) -> int:
        return len(self._conf)

    def __getitem__(self, mask):
        """Soporta indexado booleano/entero que usa BOTSORT internamente."""
        subset = _DetectionsBridge.__new__(_DetectionsBridge)
        subset._xyxy = self._xyxy[mask]
        subset._xywh = self._xywh[mask]
        subset._conf = self._conf[mask]
        subset._cls = self._cls[mask]
        return subset


class BotSORTTrackerAdapter:
    """Adapter del tracker BotSORT usando la implementacion de Ultralytics."""

    def __init__(
        self,
        config_path: str | None = None,
        device: DeviceType = DeviceType.CPU,
    ):
        self._device_str = device.value
        self._tracker = None
        self._tracker_config = config_path
        self._frame_count = 0
        self._id_offset = 0

    def _init_tracker(self, fps: int = 30) -> None:
        from ultralytics.trackers.bot_sort import BOTSORT

        cfg = {
            "tracker_type": "botsort",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.25,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "with_reid": False,
            "model": "auto",
        }

        if self._tracker_config and os.path.exists(self._tracker_config):
            import yaml
            with open(self._tracker_config, "r") as f:
                user_cfg = yaml.safe_load(f)
            if user_cfg:
                cfg.update(user_cfg)

        args = SimpleNamespace(**cfg)
        self._tracker = BOTSORT(args)

    def update(self, raw_detections: list[RawDetection], frame: np.ndarray) -> list[Detection]:
        self._frame_count += 1

        if not raw_detections:
            return []

        if self._tracker is None:
            self._init_tracker()

        bridge = _DetectionsBridge(raw_detections)
        tracked = self._tracker.update(bridge, img=frame)

        # tracked es np.ndarray (N, 7+): [x1, y1, x2, y2, track_id, conf, cls_id]
        if tracked is None or len(tracked) == 0:
            return []

        # Mapa class_id -> class_name desde las detecciones raw
        cls_name_map: dict[int, str] = {}
        for det in raw_detections:
            cls_name_map[det.class_id] = det.class_name

        results: list[Detection] = []
        for t in tracked:
            if t is None or len(t) < 7:
                continue

            x1, y1, x2, y2 = map(int, t[:4])
            track_id = int(t[4]) + self._id_offset
            conf = float(t[5])
            cls_id = int(t[6])

            cls_name = cls_name_map.get(cls_id, "unknown")

            results.append(
                Detection(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                )
            )

        return results

    def reset(self) -> None:
        self._tracker = None
        self._frame_count = 0
        self._id_offset += 10000
