import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from ...domain.ports.storage import StorageWriter


logger = logging.getLogger(__name__)

_FALLBACK_CODEC = "mp4v"


def _create_video_writer(
    path: str, codec: str, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if writer.isOpened():
        return writer
    if codec != _FALLBACK_CODEC:
        logger.warning(
            "Codec '%s' no soportado. Usando fallback '%s'.", codec, _FALLBACK_CODEC
        )
        fourcc = cv2.VideoWriter_fourcc(*_FALLBACK_CODEC)
        writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer


@dataclass
class VideoWriterConfig:
    fps: float = 30.0
    codec: str = "avc1"
    width: int = 1920
    height: int = 1080


class AnnotatedVideoWriter:
    def __init__(
        self,
        cv_writer: cv2.VideoWriter,
        output_path: str,
        storage: StorageWriter | None = None,
        temp_path: str | None = None,
    ):
        self._writer = cv_writer
        self._output_path = output_path
        self._storage = storage
        self._temp_path = temp_path
        self._frame_count = 0

    @classmethod
    @contextmanager
    def open(
        cls, storage: StorageWriter, path: str, config: VideoWriterConfig
    ) -> Iterator["AnnotatedVideoWriter"]:
        write_path = storage.get_video_writer_path(path)
        writer = _create_video_writer(
            write_path, config.codec, config.fps, (config.width, config.height)
        )
        if not writer.isOpened():
            raise ValueError(f"No se pudo crear VideoWriter: {path}")

        video_writer = cls(writer, path, storage, write_path)
        try:
            yield video_writer
        finally:
            writer.release()
            if write_path != path:
                storage.finalize_video(write_path, path)

    @classmethod
    def from_path(cls, path: str, config: VideoWriterConfig) -> "AnnotatedVideoWriter":
        writer = _create_video_writer(
            path, config.codec, config.fps, (config.width, config.height)
        )
        if not writer.isOpened():
            raise ValueError(f"No se pudo crear VideoWriter: {path}")
        return cls(writer, path)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._frame_count += 1

    def close(self) -> None:
        self._writer.release()
        if self._storage and self._temp_path and self._temp_path != self._output_path:
            self._storage.finalize_video(self._temp_path, self._output_path)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def output_path(self) -> str:
        return self._output_path
