"""Fuentes de video: archivos estandar y grabaciones DVR (``.dav``)."""

from __future__ import annotations

import os
from typing import Sequence

from .dvr import (
    DvrRecording,
    DvrSegment,
    DvrVideoSource,
    VideoIdentity,
    group_dvr_paths,
    group_dvr_segments,
    is_dvr_path,
    video_identity,
)
from .source import VideoSource

__all__ = [
    "DvrRecording",
    "DvrSegment",
    "DvrVideoSource",
    "VideoIdentity",
    "VideoSource",
    "group_dvr_paths",
    "group_dvr_segments",
    "is_dvr_path",
    "open_video",
    "video_identity",
]


def open_video(
    video: str | os.PathLike | Sequence[str | os.PathLike],
    identity: VideoIdentity | None = None,
) -> VideoSource | DvrVideoSource:
    """Abre una fuente de video a partir de una ruta o secuencia de ``.dav``.

    Es el unico punto de entrada que deberian usar los pipelines: elige
    automaticamente entre ``VideoSource`` (MP4/AVI/etc.) y ``DvrVideoSource``
    (grabaciones Dahua), sin que el usuario tenga que convertir nada.
    """
    ident = identity or video_identity(video)
    if ident.is_dvr:
        return DvrVideoSource.from_paths(ident.paths)
    return VideoSource.from_path(ident.primary_path)
