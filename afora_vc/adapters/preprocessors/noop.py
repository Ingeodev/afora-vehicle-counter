import numpy as np

from ...domain.ports.preprocessor import FramePreprocessor


class NoOpPreprocessor:
    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame
