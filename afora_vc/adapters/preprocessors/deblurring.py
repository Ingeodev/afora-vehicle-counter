import cv2
import numpy as np

from ...domain.ports.preprocessor import FramePreprocessor


KERNELS = {
    "soft": np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32),
    "medium": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "strong": np.array(
        [[-0.5, -1, -0.5], [-1, 7, -1], [-0.5, -1, -0.5]], dtype=np.float32
    ),
}


class DeblurringPreprocessor:
    def __init__(
        self,
        mode: str = "night",
        use_bilateral: bool = True,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50,
        bilateral_sigma_space: float = 50,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        sharpen_strength: str = "medium",
        gamma_correction: float = 1.2,
    ):
        if mode == "night":
            use_clahe = True
            clahe_clip_limit = 2.5
            sharpen_strength = "medium"
            gamma_correction = 1.3
        elif mode == "fast":
            use_bilateral = False
            use_clahe = True
            clahe_clip_limit = 2.0
            sharpen_strength = "soft"
        elif mode == "quality":
            use_bilateral = True
            bilateral_d = 7
            use_clahe = True
            clahe_clip_limit = 3.0
            sharpen_strength = "strong"
        elif mode == "enhance_only":
            use_bilateral = False
            use_clahe = True
            clahe_clip_limit = 3.0
            sharpen_strength = "soft"
            gamma_correction = 1.4

        self.use_bilateral = use_bilateral
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.use_clahe = use_clahe
        self.gamma_correction = gamma_correction
        self.kernel = KERNELS.get(sharpen_strength, KERNELS["medium"])

        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        else:
            self.clahe = None

        if gamma_correction != 1.0:
            inv_gamma = 1.0 / gamma_correction
            self.gamma_table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
            ).astype(np.uint8)
        else:
            self.gamma_table = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        result = frame
        if self.gamma_table is not None:
            result = cv2.LUT(result, self.gamma_table)
        if self.use_bilateral:
            result = cv2.bilateralFilter(
                result, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space
            )
        if self.clahe is not None:
            result = self._apply_clahe(result)
        result = cv2.filter2D(result, -1, self.kernel)
        return result

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @classmethod
    def create_night_enhance(cls) -> "DeblurringPreprocessor":
        return cls(mode="enhance_only")
