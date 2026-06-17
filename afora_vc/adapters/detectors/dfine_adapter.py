import numpy as np

from ...domain.schemas import RawDetection, BoundingBox
from ...domain.enums import DeviceType
from ...domain.exceptions import BackendNotInstalledError


class DFineDetectorAdapter:
    def __init__(
        self,
        model_path: str = "ustc-community/dfine-medium-coco",
        device: DeviceType = DeviceType.CPU,
        vehicle_classes: dict[int, str] | None = None,
        confidence_threshold: float = 0.5,
        class_thresholds: dict[str, float] | None = None,
        half_precision: bool = True,
    ):
        self._device = device
        self._confidence = confidence_threshold
        self._class_thresholds = class_thresholds or {}
        self._half_precision = half_precision

        self._classes = vehicle_classes or {
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
        self._allowed_ids = set(self._classes.keys())

        try:
            from transformers import AutoImageProcessor, DFineForObjectDetection
        except ImportError:
            raise BackendNotInstalledError("dfine", "afora-vehicle-counter[dfine]")

        self._processor = AutoImageProcessor.from_pretrained(model_path)
        self._model = DFineForObjectDetection.from_pretrained(model_path)
        self._model.eval()

        device_str = device.value
        if device == DeviceType.CUDA:
            import torch
            if torch.cuda.is_available():
                self._model.to("cuda")
                if half_precision:
                    self._model.half()
            else:
                device_str = "cpu"
                self._model.to("cpu")
        else:
            self._model.to(device_str)

        self._device_str = device_str

    def _to_dtype(self, inputs: dict) -> dict:
        if getattr(self, "_half_precision", False) and self._device == DeviceType.CUDA:
            import torch
            return {k: v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in inputs.items()}
        return inputs

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        from PIL import Image

        img = Image.fromarray(frame)
        inputs = self._processor(images=img, return_tensors="pt").to(self._device_str)
        inputs = self._to_dtype(inputs)

        import torch
        with torch.no_grad():
            outputs = self._model(**inputs)

        result = self._processor.post_process_object_detection(
            outputs,
            target_sizes=[(img.height, img.width)],
            threshold=self._confidence,
        )[0]

        return self._parse_results(result)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device_str)
        inputs = self._to_dtype(inputs)

        import torch
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = [(img.height, img.width) for img in images]
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self._confidence,
        )

        return [self._parse_results(r) for r in results]

    def warmup(self, input_shape: tuple[int, int, int]) -> None:
        dummy = np.zeros(input_shape, dtype=np.uint8)
        self.detect(dummy)

    def _parse_results(self, result: dict) -> list[RawDetection]:
        scores = result.get("scores")
        if scores is None or len(scores) == 0:
            return []

        detections: list[RawDetection] = []

        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            cls_id = int(label_id)
            conf = float(score)

            if cls_id not in self._allowed_ids:
                continue

            cls_name = self._classes[cls_id]
            threshold = self._class_thresholds.get(cls_name, self._confidence)
            if conf < threshold:
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in box.tolist())

            detections.append(
                RawDetection(
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
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
