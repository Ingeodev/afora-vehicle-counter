import os
from pathlib import Path

import numpy as np

from ...domain.schemas import RawDetection, BoundingBox
from ...domain.enums import DeviceType
from ...domain.exceptions import BackendNotInstalledError


class YOLODetectorAdapter:
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        device: DeviceType = DeviceType.CPU,
        vehicle_classes: dict[int, str] | None = None,
        confidence_threshold: float = 0.5,
        class_thresholds: dict[str, float] | None = None,
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
            from ultralytics import YOLO
        except ImportError:
            raise BackendNotInstalledError("yolo", "afora-vehicle-counter[yolo]")

        self._model = YOLO(model_path)
        self._device_str = device.value
        self._half = device != DeviceType.CPU

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        results = self._model(
            frame,
            device=self._device_str,
            classes=self._allowed_ids,
            verbose=False,
            half=self._half,
        )
        return self._parse_results(results)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
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

            mask_polygon = None
            mask_center = None
            if result.masks is not None:
                poly = result.masks.xy[len(detections)]
                if len(poly) > 0:
                    import cv2
                    M = cv2.moments(poly.astype(np.float32))
                    if M["m00"] > 0:
                        mask_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    mask_polygon = poly.astype(np.int32)

            detections.append(
                RawDetection(
                    class_id=int(cls_id),
                    class_name=cls_name,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(conf),
                    mask_polygon=mask_polygon,
                    mask_center=mask_center,
                )
            )

        return detections

    @property
    def class_names(self) -> dict[int, str]:
        return self._classes.copy()

    @property
    def supports_batch(self) -> bool:
        return True
