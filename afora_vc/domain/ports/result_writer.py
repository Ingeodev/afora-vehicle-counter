from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ResultWriter(Protocol):
    """Puerto de salida para escritura de resultados."""

    def write_video_frame(self, frame: np.ndarray) -> None:
        ...

    def close(self) -> None:
        ...
