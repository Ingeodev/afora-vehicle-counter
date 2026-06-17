import sys
from contextlib import contextmanager
from typing import Iterator

import cv2
import numpy as np

from ...domain.schemas import FrameData
from ...domain.ports.storage import StorageReader


class VideoSource:
    def __init__(self, cap: cv2.VideoCapture, video_path: str):
        self._cap = cap
        self._video_path = video_path
        self._frame_index = 0

        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @classmethod
    @contextmanager
    def open(cls, storage: StorageReader, path: str) -> Iterator["VideoSource"]:
        video_path = storage.get_video_path(path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {path}")
        try:
            yield cls(cap, video_path)
        finally:
            cap.release()

    @classmethod
    def from_path(cls, path: str) -> "VideoSource":
        backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG]
        if sys.platform == "darwin":
            backends.append(cv2.CAP_AVFOUNDATION)

        for backend in backends:
            cap = cv2.VideoCapture(path, backend)
            if cap.isOpened():
                return cls(cap, path)
            cap.release()

        raise ValueError(f"No se pudo abrir el video: {path}")

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def duration_seconds(self) -> float:
        return self._total_frames / self._fps if self._fps > 0 else 0

    @property
    def current_frame_index(self) -> int:
        return self._frame_index

    def frames(
        self,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
        skip_rate: int = 1,
    ) -> Iterator[FrameData]:
        frames_read = 0
        while True:
            if max_frames is not None and frames_read >= max_frames:
                break

            if skip_rate > 1 and frames_read > 0:
                for _ in range(skip_rate - 1):
                    if not self._cap.grab():
                        break
                    self._frame_index += 1

            ret, frame = self._cap.read()
            if not ret:
                break

            self._frame_index += 1

            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

            frames_read += 1

            timestamp = self._frame_index / self._fps if self._fps > 0 else 0

            yield FrameData(
                frame=frame,
                index=self._frame_index - 1,
                timestamp_seconds=timestamp,
            )

    def seek(self, frame_index: int) -> bool:
        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if success:
            self._frame_index = frame_index
        return success
