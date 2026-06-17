from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FramePreprocessor(Protocol):
    """Puerto de entrada para preprocesamiento de frames."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        ...
