"""Soporte nativo para grabaciones DVR de Dahua (``.dav``).

Los DVR de Dahua escriben la grabacion en segmentos de pocos minutos con una
ruta y un nombre predecibles::

    <raiz>/AAAA-MM-DD/001/dav/HH/HH.MM.SS-HH.MM.SS[R][0@0][0][-18000].dav

El nombre codifica la hora de inicio y de fin del segmento; la carpeta padre
codifica la fecha.  El demuxer ``dhav`` de FFmpeg (disponible desde FFmpeg 4.2)
permite abrir cada archivo directamente con OpenCV, sin conversion previa.

Este modulo agrupa los segmentos contiguos de un mismo dia en una sola
"grabacion logica" y expone ``DvrVideoSource``, que recorre todos los archivos
de forma secuencial manteniendo la linea de tiempo real (hora de pared).  De
ese modo el tracker no se reinicia cada pocos minutos y los tiempos absolutos
del CSV siguen siendo correctos.
"""

from __future__ import annotations

import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, time as time_cls, timedelta
from typing import Iterator, Sequence

import cv2

from ...domain.schemas import FrameData

# Extensiones reconocidas como grabacion DVR.
DVR_EXTENSIONS = {".dav"}

# HH.MM.SS-HH.MM.SS[...].dav
_DVR_FILENAME_RE = re.compile(
    r"^(?P<sh>\d{2})\.(?P<sm>\d{2})\.(?P<ss>\d{2})"
    r"-(?P<eh>\d{2})\.(?P<em>\d{2})\.(?P<es>\d{2})"
)

# Carpetas de fecha: AAAA-MM-DD
_DATE_DIR_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

# Segundos de tolerancia para considerar dos segmentos como contiguos.
DEFAULT_GAP_SECONDS = 2.0

_DHAV_HINT = (
    "No se pudo abrir el archivo DVR: {path}\n"
    "El backend FFmpeg de OpenCV debe incluir el demuxer 'dhav' (FFmpeg >= 4.2).\n"
    "Los wheels oficiales de opencv-python-headless ya lo incluyen; si usa una\n"
    "build personalizada, actualice OpenCV/FFmpeg.\n"
    'Verifique con: python -c "import cv2; print(cv2.getBuildInformation())"'
)


def is_dvr_path(path: str | os.PathLike) -> bool:
    """True si la ruta corresponde a una grabacion DVR soportada."""
    return os.path.splitext(os.fspath(path))[1].lower() in DVR_EXTENSIONS


def dvr_logical_name(start: datetime) -> str:
    """Nombre logico de una grabacion (compatible con ``_VIDEO_TS_PATTERN``).

    El pipeline deriva el nombre del CSV de ``splitext(video_name)[0]`` y el
    paso de conteo exige ``<prefijo>_AAAA-MM-DD_HH.MM.SS_detections.csv``.
    """
    return f"dav_{start:%Y-%m-%d_%H.%M.%S}.dav"


def date_dir_key(path: str | os.PathLike) -> str | None:
    """Devuelve ``AAAA-MM-DD`` si el ultimo componente es una carpeta de fecha."""
    name = os.path.basename(os.path.normpath(os.fspath(path)))
    m = _DATE_DIR_RE.match(name)
    if not m:
        return None
    try:
        date_cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    return name


def _find_date_in_parents(path: str) -> date_cls | None:
    current = os.path.dirname(os.path.abspath(path))
    while current and current != os.path.dirname(current):
        key = date_dir_key(current)
        if key is not None:
            y, m, d = key.split("-")
            return date_cls(int(y), int(m), int(d))
        current = os.path.dirname(current)
    return None


def _valid_time(h: int, m: int, s: int) -> bool:
    return 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59


@dataclass(frozen=True)
class DvrSegment:
    """Un archivo ``.dav`` con su ventana temporal."""

    path: str
    start: datetime
    end: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        if self.end is None:
            return 0.0
        return max(0.0, (self.end - self.start).total_seconds())

    @classmethod
    def from_path(cls, path: str | os.PathLike) -> "DvrSegment":
        """Construye el segmento a partir de la ruta.

        Si el nombre no sigue la convencion de Dahua se usa la fecha de
        modificacion del archivo como aproximacion, de modo que el pipeline
        siga funcionando con grabaciones renombradas.
        """
        path = os.fspath(path)
        name = os.path.basename(path)
        m = _DVR_FILENAME_RE.match(name)
        if m:
            sh, sm, ss = int(m.group("sh")), int(m.group("sm")), int(m.group("ss"))
            eh, em, es = int(m.group("eh")), int(m.group("em")), int(m.group("es"))
            if _valid_time(sh, sm, ss) and _valid_time(eh, em, es):
                day = _find_date_in_parents(path)
                if day is None:
                    day = datetime.fromtimestamp(os.path.getmtime(path)).date()
                start = datetime.combine(day, time_cls(sh, sm, ss))
                end = datetime.combine(day, time_cls(eh, em, es))
                if end <= start:
                    # El segmento cruza la medianoche.
                    end += timedelta(days=1)
                return cls(path=path, start=start, end=end)

        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return cls(path=path, start=mtime, end=None)


@dataclass
class DvrRecording:
    """Grabacion logica formada por uno o mas segmentos contiguos."""

    segments: list[DvrSegment]

    def __post_init__(self) -> None:
        self.segments.sort(key=lambda s: s.start)

    @property
    def start(self) -> datetime:
        return self.segments[0].start

    @property
    def end(self) -> datetime:
        last = self.start
        for seg in self.segments:
            candidate = seg.end or seg.start
            if candidate > last:
                last = candidate
        return last

    @property
    def logical_name(self) -> str:
        return dvr_logical_name(self.start)

    @property
    def duration_seconds(self) -> float:
        """Duracion de la union de intervalos (sin contar solapamientos)."""
        total = 0.0
        last_end: datetime | None = None
        for seg in sorted(self.segments, key=lambda s: s.start):
            seg_end = seg.end or seg.start
            if last_end is None or seg.start > last_end:
                total += max(0.0, (seg_end - seg.start).total_seconds())
            elif seg_end > last_end:
                total += (seg_end - last_end).total_seconds()
            if last_end is None or seg_end > last_end:
                last_end = seg_end
        return total


def group_dvr_segments(
    segments: Sequence[DvrSegment],
    max_gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> list[DvrRecording]:
    """Agrupa segmentos contiguos de un mismo dia en grabaciones logicas.

    Un segmento sin ``end`` conocido (grabacion renombrada) nunca se fusiona,
    para no inventar continuidad.
    """
    ordered = sorted(segments, key=lambda s: s.start)
    groups: list[list[DvrSegment]] = []
    current: list[DvrSegment] = []

    for seg in ordered:
        if not current:
            current = [seg]
            continue

        prev = current[-1]
        same_day = seg.start.date() == current[0].start.date()
        contiguous = (
            prev.end is not None
            and seg.start <= prev.end + timedelta(seconds=max_gap_seconds)
        )
        if same_day and contiguous:
            current.append(seg)
        else:
            groups.append(current)
            current = [seg]

    if current:
        groups.append(current)

    return [DvrRecording(segments=g) for g in groups]


def group_dvr_paths(
    paths: Sequence[str | os.PathLike],
    max_gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> list[DvrRecording]:
    """Agrupa una lista de rutas ``.dav`` en grabaciones logicas."""
    return group_dvr_segments(
        [DvrSegment.from_path(p) for p in paths],
        max_gap_seconds=max_gap_seconds,
    )


@dataclass(frozen=True)
class VideoIdentity:
    """Identidad de un video de entrada (simple o grabacion DVR multi-archivo)."""

    paths: tuple[str, ...]
    name: str
    is_dvr: bool
    start: datetime | None = None

    @property
    def primary_path(self) -> str:
        return self.paths[0]


def video_identity(video: str | os.PathLike | Sequence[str | os.PathLike]) -> VideoIdentity:
    """Resuelve rutas de entrada a una identidad estable para el pipeline."""
    if isinstance(video, (str, os.PathLike)):
        paths = (os.fspath(video),)
    else:
        paths = tuple(os.fspath(p) for p in video)

    if not paths:
        raise ValueError("Se requiere al menos una ruta de video")

    if len(paths) > 1:
        if not all(is_dvr_path(p) for p in paths):
            raise ValueError(
                "Solo se admiten multiples archivos para secuencias .dav (DVR)"
            )
        segments = sorted(
            (DvrSegment.from_path(p) for p in paths), key=lambda s: s.start
        )
        start = segments[0].start
        return VideoIdentity(
            paths=tuple(s.path for s in segments),
            name=dvr_logical_name(start),
            is_dvr=True,
            start=start,
        )

    path = paths[0]
    if is_dvr_path(path):
        segment = DvrSegment.from_path(path)
        return VideoIdentity(
            paths=(segment.path,),
            name=dvr_logical_name(segment.start),
            is_dvr=True,
            start=segment.start,
        )

    return VideoIdentity(
        paths=(path,),
        name=os.path.basename(path),
        is_dvr=False,
        start=None,
    )


class DvrVideoSource:
    """Fuente de video que encadena varios ``.dav`` como un solo video.

    Mantiene indices de frame secuenciales (para progreso y perfilado) y
    ``timestamp_seconds`` alineado al reloj de pared desde el inicio de la
    grabacion, de modo que ``base_time + timestamp`` reconstruya la hora real.
    Los frames solapados entre segmentos se descartan una sola vez.
    """

    def __init__(
        self,
        segments: Sequence[DvrSegment],
        fps: float,
        width: int,
        height: int,
        total_frames: int,
        start: datetime,
    ) -> None:
        self._segments = list(segments)
        self._fps = float(fps)
        self._width = int(width)
        self._height = int(height)
        self._total_frames = int(total_frames)
        self._start = start
        self._frame_index = 0

    @classmethod
    def from_paths(cls, paths: Sequence[str | os.PathLike]) -> "DvrVideoSource":
        segments = sorted(
            (DvrSegment.from_path(p) for p in paths), key=lambda s: s.start
        )
        if not segments:
            raise ValueError("Se requiere al menos un archivo .dav")

        probe = cv2.VideoCapture(segments[0].path)
        if not probe.isOpened():
            probe.release()
            raise ValueError(_DHAV_HINT.format(path=segments[0].path))
        try:
            fps = float(probe.get(cv2.CAP_PROP_FPS))
            width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            probe.release()

        if fps <= 0:
            fps = 15.0

        recording = DvrRecording(segments=list(segments))
        total_frames = int(round(recording.duration_seconds * fps))
        return cls(
            segments=segments,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            start=recording.start,
        )

    @property
    def recording_start(self) -> datetime:
        return self._start

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
        return self._total_frames / self._fps if self._fps > 0 else 0.0

    @property
    def current_frame_index(self) -> int:
        return self._frame_index

    def close(self) -> None:
        # Cada segmento se abre y cierra dentro de ``frames``.
        return None

    @contextmanager
    def _open_segment(self, seg: DvrSegment) -> Iterator[cv2.VideoCapture]:
        cap = cv2.VideoCapture(seg.path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(_DHAV_HINT.format(path=seg.path))
        try:
            yield cap
        finally:
            cap.release()

    def frames(
        self,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
        skip_rate: int = 1,
    ) -> Iterator[FrameData]:
        if skip_rate < 1:
            skip_rate = 1

        frames_read = 0
        last_ts: float | None = None

        for seg in self._segments:
            with self._open_segment(seg) as cap:
                seg_fps = float(cap.get(cv2.CAP_PROP_FPS))
                if seg_fps <= 0:
                    seg_fps = self._fps

                seg_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                seg_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (seg_width, seg_height) != (self._width, self._height) and (
                    seg_width > 0 and seg_height > 0
                ):
                    warnings.warn(
                        f"Resolucion distinta en {seg.path}: "
                        f"{seg_width}x{seg_height} vs {self._width}x{self._height}. "
                        "Se reescala al tamano del primer segmento.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                base_offset = (seg.start - self._start).total_seconds()
                local = 0

                while True:
                    if max_frames is not None and frames_read >= max_frames:
                        return

                    if skip_rate > 1 and frames_read > 0:
                        for _ in range(skip_rate - 1):
                            if not cap.grab():
                                break
                            local += 1

                    ret, frame = cap.read()
                    if not ret:
                        break

                    ts = base_offset + local / seg_fps
                    local += 1

                    # Segmentos consecutivos pueden solaparse ~1s: descartar
                    # los frames ya emitidos para no duplicar conteos.
                    if last_ts is not None and ts <= last_ts:
                        continue
                    last_ts = ts

                    if (seg_width, seg_height) != (self._width, self._height) and (
                        seg_width > 0 and seg_height > 0
                    ):
                        frame = cv2.resize(
                            frame, (self._width, self._height),
                            interpolation=cv2.INTER_AREA,
                        )

                    if resize is not None:
                        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

                    frames_read += 1
                    self._frame_index = frames_read

                    yield FrameData(
                        frame=frame,
                        index=frames_read - 1,
                        timestamp_seconds=ts,
                    )

    def seek(self, frame_index: int) -> bool:
        # El pipeline es secuencial y nunca solicita seek sobre DVR.
        return False


__all__ = [
    "DEFAULT_GAP_SECONDS",
    "DVR_EXTENSIONS",
    "DvrRecording",
    "DvrSegment",
    "DvrVideoSource",
    "VideoIdentity",
    "date_dir_key",
    "dvr_logical_name",
    "group_dvr_paths",
    "group_dvr_segments",
    "is_dvr_path",
    "video_identity",
]
