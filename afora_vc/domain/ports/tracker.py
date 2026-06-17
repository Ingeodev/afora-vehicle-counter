from typing import Protocol, runtime_checkable

import numpy as np

from ..schemas import RawDetection, Detection


@runtime_checkable
class Tracker(Protocol):
    """Puerto de entrada para algoritmos de tracking."""

    def update(self, raw_detections: list[RawDetection], frame: np.ndarray) -> list[Detection]:
        ...

    def reset(self) -> None:
        ...
