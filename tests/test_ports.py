"""Tests estructurales de los Protocols."""

import numpy as np

from afora_vc.domain.ports import (
    Detector,
    Tracker,
    FramePreprocessor,
    ResultWriter,
    StorageReader,
    StorageWriter,
)
from afora_vc.domain.schemas import RawDetection, Detection, BoundingBox


class TestDetectorProtocol:
    def test_protocol_structural(self):
        """Verifica que un mock cumple estructuralmente con el Protocol."""

        class MockDetector:
            def detect(self, frame: np.ndarray) -> list[RawDetection]:
                return []

            def detect_batch(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
                return [[] for _ in frames]

            def warmup(self, input_shape: tuple[int, int, int]) -> None:
                pass

            @property
            def class_names(self) -> dict[int, str]:
                return {2: "car"}

            @property
            def supports_batch(self) -> bool:
                return False

        mock = MockDetector()
        assert isinstance(mock, Detector)


class TestTrackerProtocol:
    def test_protocol_structural(self):
        class MockTracker:
            def update(self, raw_detections: list[RawDetection], frame: np.ndarray) -> list[Detection]:
                return []

            def reset(self) -> None:
                pass

        mock = MockTracker()
        assert isinstance(mock, Tracker)


class TestPreprocessorProtocol:
    def test_protocol_structural(self):
        class MockPreprocessor:
            def process(self, frame: np.ndarray) -> np.ndarray:
                return frame.copy()

        mock = MockPreprocessor()
        assert isinstance(mock, FramePreprocessor)


class TestResultWriterProtocol:
    def test_protocol_structural(self):
        class MockWriter:
            def write_video_frame(self, frame: np.ndarray) -> None:
                pass

            def close(self) -> None:
                pass

        mock = MockWriter()
        assert isinstance(mock, ResultWriter)


class TestStorageReaderProtocol:
    def test_protocol_structural(self):
        class MockReader:
            def list_files(self, directory: str, pattern: str = "*") -> list:
                return []

            def read_file(self, path: str) -> bytes:
                return b""

            def read_text(self, path: str, encoding: str = "utf-8") -> str:
                return ""

            def read_json(self, path: str) -> dict:
                return {}

            def exists(self, path: str) -> bool:
                return False

            def is_file(self, path: str) -> bool:
                return False

            def get_video_path(self, path: str) -> str:
                return path

        mock = MockReader()
        assert isinstance(mock, StorageReader)


class TestStorageWriterProtocol:
    def test_protocol_structural(self):
        class MockWriter:
            def write_file(self, path: str, content: bytes) -> None:
                pass

            def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
                pass

            def write_json(self, path: str, data: dict, indent: int = 2) -> None:
                pass

            def makedirs(self, path: str, exist_ok: bool = True) -> None:
                pass

            def get_video_writer_path(self, path: str) -> str:
                return path

            def finalize_video(self, temp_path: str, final_path: str) -> None:
                pass

            def copy_file(self, source: str, destination: str) -> None:
                pass

            def delete_file(self, path: str) -> None:
                pass

        mock = MockWriter()
        assert isinstance(mock, StorageWriter)
