from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Iterator

import cv2
import numpy as np

from ...domain.ports.storage import StorageWriter


logger = logging.getLogger(__name__)

# FourCC de OpenCV que no pasan por PyAV.
_OPENCV_CODECS = {"mp4v", "mjpg", "xvid"}
_FALLBACK_CODEC = "mp4v"

# Alias de usuario -> encoder de FFmpeg via PyAV.
_PYAV_ENCODERS = {
    "avc1": "libx264",
    "h264": "libx264",
    "hevc": "libx265",
    "h265": "libx265",
}

_X264_PRESETS = {
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow", "placebo",
}


def _load_av():
    """Importa PyAV de forma diferida para no penalizar el arranque del CLI."""
    import av

    return av


@dataclass
class VideoWriterConfig:
    fps: float = 30.0
    codec: str = "avc1"
    width: int = 1920
    height: int = 1080
    # Los videos de salida son para verificacion, no para archivo final:
    # crf alto + preset rapido minimiza peso sin perder legibilidad.
    crf: int = 30
    preset: str = "veryfast"


class _PyAvBackend:
    """Codifica H.264/H.265 con libx264/libx265 (wheels de PyAV traen ambos)."""

    def __init__(self, path: str, config: VideoWriterConfig, encoder: str):
        if not 0 <= int(config.crf) <= 51:
            raise ValueError(f"crf fuera de rango (0-51): {config.crf}")

        self._av = _load_av()
        self._fps = float(config.fps) or 15.0
        # PyAV espera una fraccion (o int), no un float: 29.97 -> 30000/1001.
        self._rate = Fraction(self._fps).limit_denominator(1001)
        self._in_size = (int(config.width), int(config.height))
        # yuv420p requiere dimensiones pares; se rellena 1 px si hace falta.
        self._out_size = (self._in_size[0] + self._in_size[0] % 2,
                          self._in_size[1] + self._in_size[1] % 2)
        self._lock = threading.Lock()
        self._closed = False
        self._frame_count = 0

        if self._out_size != self._in_size:
            logger.info(
                "Dimensiones impares %dx%d: se rellena a %dx%d para %s.",
                *self._in_size, *self._out_size, encoder,
            )

        self._container = self._av.open(path, mode="w")
        try:
            stream = self._container.add_stream(encoder, rate=self._rate)
            stream.width = self._out_size[0]
            stream.height = self._out_size[1]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(int(config.crf)), "preset": config.preset}
            self._stream = stream
        except Exception:
            self._container.close()
            raise

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame invalido: dtype={frame.dtype} shape={frame.shape}. "
                "Se esperaba una imagen BGR uint8 de 3 canales."
            )
        h, w = frame.shape[:2]
        if (w, h) != self._in_size:
            raise ValueError(
                f"Frame de {w}x{h} distinto al configurado "
                f"{self._in_size[0]}x{self._in_size[1]}."
            )
        if self._out_size != self._in_size:
            pad_h = self._out_size[1] - h
            pad_w = self._out_size[0] - w
            frame = cv2.copyMakeBorder(
                frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        return frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("El VideoWriter ya fue cerrado.")
            prepared = self._prepare(frame)
            video_frame = self._av.VideoFrame.from_ndarray(prepared, format="bgr24")
            video_frame.pts = self._frame_count
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)
            self._frame_count += 1

    def release(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                # Sin este flush se pierden los ultimos frames del GOP.
                for packet in self._stream.encode():
                    self._container.mux(packet)
            finally:
                self._container.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count


class _OpenCvBackend:
    """Backend historico de OpenCV. Solo para codecs que no maneja PyAV."""

    def __init__(self, path: str, config: VideoWriterConfig):
        self._codec = config.codec
        self._writer = self._open(path, config)
        self._lock = threading.Lock()
        self._frame_count = 0

    def _open(self, path: str, config: VideoWriterConfig) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*config.codec)
        writer = cv2.VideoWriter(path, fourcc, config.fps, (config.width, config.height))
        if writer.isOpened():
            return writer
        if config.codec != _FALLBACK_CODEC:
            logger.warning(
                "Codec '%s' no soportado por OpenCV. Usando fallback '%s'.",
                config.codec, _FALLBACK_CODEC,
            )
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*_FALLBACK_CODEC),
                config.fps,
                (config.width, config.height),
            )
            self._codec = _FALLBACK_CODEC
        if not writer.isOpened():
            raise ValueError(f"No se pudo crear VideoWriter: {path}")
        return writer

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            self._writer.write(frame)
            self._frame_count += 1

    def release(self) -> None:
        with self._lock:
            self._writer.release()

    @property
    def frame_count(self) -> int:
        return self._frame_count


def _create_writer(path: str, config: VideoWriterConfig) -> _PyAvBackend | _OpenCvBackend:
    codec = (config.codec or "avc1").lower()
    if codec in _OPENCV_CODECS:
        return _OpenCvBackend(path, config)

    encoder = _PYAV_ENCODERS.get(codec)
    if encoder is None:
        raise ValueError(
            f"Codec desconocido: {config.codec!r}. "
            "Usar avc1, h264, hevc, h265 o mp4v."
        )
    if not 0 <= int(config.crf) <= 51:
        raise ValueError(f"crf fuera de rango (0-51): {config.crf}")
    if config.preset not in _X264_PRESETS:
        raise ValueError(
            f"preset desconocido: {config.preset!r}. "
            f"Usar uno de: {', '.join(sorted(_X264_PRESETS))}."
        )

    try:
        return _PyAvBackend(path, config, encoder)
    except ImportError:
        logger.warning(
            "PyAV no esta instalado: se usa '%s' de OpenCV (archivos mas pesados).",
            _FALLBACK_CODEC,
        )
    except Exception as exc:
        logger.warning(
            "No se pudo inicializar '%s' via PyAV (%s). Usando fallback '%s'.",
            encoder, exc, _FALLBACK_CODEC,
        )
    return _OpenCvBackend(path, replace(config, codec=_FALLBACK_CODEC))


class AnnotatedVideoWriter:
    def __init__(
        self,
        backend: _PyAvBackend | _OpenCvBackend,
        output_path: str,
        storage: StorageWriter | None = None,
        temp_path: str | None = None,
    ):
        self._backend = backend
        self._output_path = output_path
        self._storage = storage
        self._temp_path = temp_path
        self._closed = False
        self._close_lock = threading.Lock()

    @classmethod
    @contextmanager
    def open(
        cls, storage: StorageWriter, path: str, config: VideoWriterConfig
    ) -> Iterator["AnnotatedVideoWriter"]:
        write_path = storage.get_video_writer_path(path)
        backend = _create_writer(write_path, config)

        video_writer = cls(backend, path, storage, write_path)
        try:
            yield video_writer
        finally:
            video_writer.close()

    @classmethod
    def from_path(cls, path: str, config: VideoWriterConfig) -> "AnnotatedVideoWriter":
        backend = _create_writer(path, config)
        return cls(backend, path)

    def write(self, frame: np.ndarray) -> None:
        self._backend.write(frame)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._backend.release()
        if self._storage and self._temp_path and self._temp_path != self._output_path:
            self._storage.finalize_video(self._temp_path, self._output_path)

    @property
    def frame_count(self) -> int:
        return self._backend.frame_count

    @property
    def output_path(self) -> str:
        return self._output_path
