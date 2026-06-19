import cv2
import numpy as np

from ..ports.storage import StorageReader
from ..schemas import Detection


def _squeeze(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is 2D (H, W); ultralytics import can make cv2 return (H, W, 1)."""
    return np.squeeze(mask) if mask.ndim == 3 else mask


class MaskManager:
    def __init__(self):
        self._mask_original: np.ndarray | None = None
        self._mask_resized: np.ndarray | None = None
        self._current_size: tuple[int, int] | None = None

    def load(self, storage: StorageReader, path: str) -> bool:
        try:
            local_path = storage.get_video_path(path)
            mask = cv2.imread(local_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return False
            mask = _squeeze(mask)
            self._mask_original = mask
            self._mask_resized = mask.copy()
            self._current_size = (mask.shape[1], mask.shape[0])
            return True
        except Exception:
            return False

    def load_from_path(self, path: str) -> bool:
        try:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return False
            mask = _squeeze(mask)
            self._mask_original = mask
            self._mask_resized = mask.copy()
            self._current_size = (mask.shape[1], mask.shape[0])
            return True
        except Exception:
            return False

    def resize_to_frame(self, width: int, height: int) -> None:
        if self._mask_original is None:
            return
        if self._current_size == (width, height):
            return
        resized = cv2.resize(
            self._mask_original, (width, height), interpolation=cv2.INTER_NEAREST
        )
        self._mask_resized = _squeeze(resized)
        self._current_size = (width, height)

    def is_point_valid(self, x: int, y: int) -> bool:
        if self._mask_resized is None:
            return True
        h, w = self._mask_resized.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        return self._mask_resized[y, x] > 0

    def filter_detections(self, detections: list[Detection]) -> list[Detection]:
        if self._mask_resized is None:
            return detections
        return [d for d in detections if self.is_point_valid(*d.center)]

    def apply_to_frame(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        if self._mask_resized is None:
            return frame
        overlay = frame.copy()
        mask_inv = cv2.bitwise_not(self._mask_resized)
        overlay[mask_inv > 0] = [0, 0, 255]
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    @property
    def is_loaded(self) -> bool:
        return self._mask_original is not None

    @property
    def original_size(self) -> tuple[int, int] | None:
        if self._mask_original is None:
            return None
        return (self._mask_original.shape[1], self._mask_original.shape[0])

    @property
    def current_size(self) -> tuple[int, int] | None:
        return self._current_size

    def __bool__(self) -> bool:
        return self.is_loaded
