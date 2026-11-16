"""Tests del writer de video anotado (PyAV/libx264 y fallback OpenCV)."""

import shutil

import numpy as np
import pytest

av = pytest.importorskip("av")

import afora_vc.adapters.video.writer as writer_module
from afora_vc.adapters.video.writer import AnnotatedVideoWriter, VideoWriterConfig


def _frames(count: int, width: int, height: int):
    """Frames BGR deterministas (sin aleatoriedad para que sean reproducibles)."""
    for i in range(count):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :, 0] = (i * 20) % 256
        frame[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
        frame[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
        yield frame


def _decode(path):
    container = av.open(str(path))
    stream = container.streams.video[0]
    codec = stream.codec_context.name
    width = stream.codec_context.width
    height = stream.codec_context.height
    frames = [frame for frame in container.decode(video=0)]
    container.close()
    return codec, width, height, frames


def _write(path, config, count=10, width=None, height=None):
    writer = AnnotatedVideoWriter.from_path(str(path), config)
    for frame in _frames(count, width or config.width, height or config.height):
        writer.write(frame)
    writer.close()
    return writer


class TestPyAvBackend:
    def test_h264_output_is_decodable(self, tmp_path):
        path = tmp_path / "out.mp4"
        config = VideoWriterConfig(fps=10, codec="avc1", width=64, height=48)
        writer = _write(path, config)

        assert path.exists() and path.stat().st_size > 0
        assert writer.frame_count == 10

        codec, width, height, frames = _decode(path)
        assert codec == "h264"
        assert (width, height) == (64, 48)
        assert len(frames) == 10

    def test_hevc_alias(self, tmp_path):
        path = tmp_path / "out.mp4"
        config = VideoWriterConfig(fps=10, codec="hevc", width=64, height=48, preset="ultrafast")
        _write(path, config, count=5)

        codec, _, _, frames = _decode(path)
        assert codec == "hevc"
        assert len(frames) == 5

    def test_odd_dimensions_are_padded(self, tmp_path):
        path = tmp_path / "out.mp4"
        config = VideoWriterConfig(fps=10, codec="h264", width=65, height=49, preset="ultrafast")
        _write(path, config, count=3, width=65, height=49)

        codec, width, height, frames = _decode(path)
        assert codec == "h264"
        assert (width, height) == (66, 50)
        assert len(frames) == 3

    def test_frame_size_mismatch_raises(self, tmp_path):
        config = VideoWriterConfig(fps=10, codec="h264", width=64, height=48)
        writer = AnnotatedVideoWriter.from_path(str(tmp_path / "out.mp4"), config)
        try:
            with pytest.raises(ValueError, match="distinto al configurado"):
                writer.write(np.zeros((96, 128, 3), np.uint8))
        finally:
            writer.close()

    def test_invalid_dtype_raises(self, tmp_path):
        config = VideoWriterConfig(fps=10, codec="h264", width=64, height=48)
        writer = AnnotatedVideoWriter.from_path(str(tmp_path / "out.mp4"), config)
        try:
            with pytest.raises(ValueError, match="Frame invalido"):
                writer.write(np.zeros((48, 64, 3), np.float32))
        finally:
            writer.close()

    def test_crf_controls_size(self, tmp_path):
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(15)]

        def encode(crf, name):
            path = tmp_path / name
            config = VideoWriterConfig(
                fps=10, codec="h264", width=128, height=96, crf=crf, preset="ultrafast"
            )
            writer = AnnotatedVideoWriter.from_path(str(path), config)
            for frame in frames:
                writer.write(frame)
            writer.close()
            return path.stat().st_size

        high_quality = encode(16, "hq.mp4")
        low_quality = encode(45, "lq.mp4")
        assert low_quality < high_quality

    def test_close_is_idempotent(self, tmp_path):
        path = tmp_path / "out.mp4"
        config = VideoWriterConfig(fps=10, codec="h264", width=64, height=48)
        writer = _write(path, config)
        writer.close()
        writer.close()
        assert path.exists()

    def test_write_after_close_raises(self, tmp_path):
        config = VideoWriterConfig(fps=10, codec="h264", width=64, height=48)
        writer = _write(tmp_path / "out.mp4", config)
        with pytest.raises(RuntimeError, match="cerrado"):
            writer.write(np.zeros((48, 64, 3), np.uint8))


class TestOpenCvBackend:
    def test_mp4v_still_works(self, tmp_path):
        path = tmp_path / "out.mp4"
        config = VideoWriterConfig(fps=10, codec="mp4v", width=64, height=48)
        _write(path, config, count=5)

        codec, width, height, frames = _decode(path)
        assert codec == "mpeg4"
        assert (width, height) == (64, 48)
        assert len(frames) == 5

    def test_falls_back_when_pyav_unavailable(self, tmp_path, monkeypatch, caplog):
        def _raise():
            raise ImportError("no av")

        monkeypatch.setattr(writer_module, "_load_av", _raise)

        path = tmp_path / "out.mp4"
        config = VideoWriterConfig(fps=10, codec="avc1", width=64, height=48)
        _write(path, config, count=3)

        codec, _, _, frames = _decode(path)
        assert codec == "mpeg4"
        assert len(frames) == 3
        assert "PyAV no esta instalado" in caplog.text

    def test_unknown_codec_raises(self, tmp_path):
        config = VideoWriterConfig(fps=10, codec="nope", width=64, height=48)
        with pytest.raises(ValueError, match="Codec desconocido"):
            AnnotatedVideoWriter.from_path(str(tmp_path / "out.mp4"), config)


class TestConfigValidation:
    def test_invalid_preset_raises(self, tmp_path):
        config = VideoWriterConfig(fps=10, codec="h264", width=64, height=48, preset="nope")
        with pytest.raises(ValueError, match="preset desconocido"):
            AnnotatedVideoWriter.from_path(str(tmp_path / "out.mp4"), config)

    def test_invalid_crf_raises(self, tmp_path):
        config = VideoWriterConfig(fps=10, codec="h264", width=64, height=48, crf=99)
        with pytest.raises(ValueError, match="crf fuera de rango"):
            AnnotatedVideoWriter.from_path(str(tmp_path / "out.mp4"), config)


class _TempStorage:
    """Storage de prueba: escribe a un temporal y lo mueve al final."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path

    def get_video_writer_path(self, path):
        return str(self.tmp_path / "temp_video.mp4")

    def finalize_video(self, temp_path, final_path):
        shutil.move(temp_path, final_path)


class TestOpenContextManager:
    def test_moves_temp_file_on_close(self, tmp_path):
        final_path = tmp_path / "final.mp4"
        storage = _TempStorage(tmp_path)
        config = VideoWriterConfig(fps=10, codec="h264", width=64, height=48)

        with AnnotatedVideoWriter.open(storage, str(final_path), config) as writer:
            for frame in _frames(4, 64, 48):
                writer.write(frame)

        assert final_path.exists()
        assert not (tmp_path / "temp_video.mp4").exists()

        _, _, _, frames = _decode(final_path)
        assert len(frames) == 4
