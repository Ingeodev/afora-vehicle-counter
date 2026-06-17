from typing import Protocol, runtime_checkable

import numpy as np

from ..schemas import RawDetection


@runtime_checkable
class Detector(Protocol):
    """Puerto de entrada para cualquier backend de deteccion."""

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        ...

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        ...

    def warmup(self, input_shape: tuple[int, int, int]) -> None:
        ...

    @property
    def class_names(self) -> dict[int, str]:
        ...

    @property
    def supports_batch(self) -> bool:
        ...
