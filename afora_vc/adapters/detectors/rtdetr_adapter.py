import re
import numpy as np

from ...domain.schemas import RawDetection, BoundingBox
from ...domain.enums import DeviceType
from ...domain.exceptions import BackendNotInstalledError


class RTDetrDetectorAdapter:
    _engine_batch = 1

    def __init__(
        self,
        model_path: str = "models/rtdetr-l.pt",
        device: DeviceType = DeviceType.CPU,
        vehicle_classes: dict[int, str] | None = None,
        confidence_threshold: float = 0.5,
        class_thresholds: dict[str, float] | None = None,
        half_precision: bool = True,
    ):
        self._device = device
        self._confidence = confidence_threshold
        self._class_thresholds = class_thresholds or {}

        self._classes = vehicle_classes or {
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
        self._allowed_ids = list(self._classes.keys())

        try:
            from ultralytics import RTDETR
        except ImportError:
            raise BackendNotInstalledError("rtdetr", "afora-vehicle-counter[rtdetr]")

        self._model = RTDETR(model_path)
        self._device_str = device.value
        is_engine = model_path.lower().endswith(".engine")
        if is_engine:
            self._half = half_precision and device != DeviceType.CPU
        else:
            self._half = False

        self._engine_batch = self._probe_engine_batch_size() if is_engine else 1

    def _probe_engine_batch_size(self) -> int:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self._model(dummy, verbose=False)
            return 1
        except AssertionError as e:
            m = re.search(r"max model size \((\d+),", str(e))
            return int(m.group(1)) if m else 1

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        if self._engine_batch > 1:
            padded = [frame] + [np.zeros_like(frame)] * (self._engine_batch - 1)
            all_results = self.detect_batch(padded)
            return all_results[0]
        results = self._model(
            frame,
            device=self._device_str,
            classes=self._allowed_ids,
            verbose=False,
            half=self._half,
        )
        return self._parse_results(results)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        n_actual = len(frames)
        if self._engine_batch > 1 and n_actual != self._engine_batch:
            dummy = np.zeros_like(frames[0])
            padded = list(frames) + [dummy] * (self._engine_batch - n_actual)
            results = self._model(
                padded,
                device=self._device_str,
                classes=self._allowed_ids,
                verbose=False,
                half=self._half,
            )
            parsed = [self._parse_results([r]) for r in results]
            return parsed[:n_actual]
        results = self._model(
            frames,
            device=self._device_str,
            classes=self._allowed_ids,
            verbose=False,
            half=self._half,
        )
        return [self._parse_results([r]) for r in results]

    def warmup(self, input_shape: tuple[int, int, int]) -> None:
        dummy = np.zeros(input_shape, dtype=np.uint8)
        if self._engine_batch > 1:
            self.detect_batch([dummy] * self._engine_batch)
        else:
            self._model(dummy, verbose=False)

    def _parse_results(self, results) -> list[RawDetection]:
        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        detections: list[RawDetection] = []

        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            if cls_id not in self._classes:
                continue

            cls_name = self._classes[int(cls_id)]
            threshold = self._class_thresholds.get(cls_name, self._confidence)
            if conf < threshold:
                continue

            x1, y1, x2, y2 = box.astype(int)

            detections.append(
                RawDetection(
                    class_id=int(cls_id),
                    class_name=cls_name,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(conf),
                    mask_polygon=None,
                    mask_center=None,
                )
            )

        return detections

    @property
    def class_names(self) -> dict[int, str]:
        return self._classes.copy()

    @property
    def supports_batch(self) -> bool:
        return True
