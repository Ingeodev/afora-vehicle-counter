"""Tests de la configuracion de logging y flags del CLI."""

import logging
import sys

import pytest

import afora_vc.cli.main as cli
import afora_vc.application.facade as facade
from afora_vc.application.batch_config import BatchConfig


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """basicConfig(force=True) reemplaza handlers: se restauran al terminar."""
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


class TestConfigureLogging:
    def test_verbose_sends_info_to_stdout(self, capsys):
        cli._configure_logging("WARNING", verbose=True)
        logging.getLogger("afora_vc.demo").info("mensaje-info")

        assert "mensaje-info" in capsys.readouterr().out

    def test_default_level_skips_info(self, capsys):
        cli._configure_logging()
        logger = logging.getLogger("afora_vc.demo")
        logger.info("mensaje-info")
        logger.warning("mensaje-warning")

        out = capsys.readouterr().out
        assert "mensaje-info" not in out
        assert "mensaje-warning" in out

    def test_debug_level_shows_debug(self, capsys):
        cli._configure_logging("DEBUG")
        logging.getLogger("afora_vc.demo").debug("mensaje-debug")

        assert "mensaje-debug" in capsys.readouterr().out


class TestRunFlags:
    def test_no_resume_skips_progress_tracker(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("input: .\noutput: ./out\n", encoding="utf-8")

        captured = {}

        def fake_process_from_config(config, resume=True):
            captured["resume"] = resume
            return []

        monkeypatch.setattr(cli, "process_from_config", fake_process_from_config)
        monkeypatch.setattr(sys, "argv", ["afora", "run", str(config_path)])

        assert cli.main() == 0
        assert captured["resume"] is True

        monkeypatch.setattr(
            sys, "argv", ["afora", "run", str(config_path), "--no-resume"]
        )
        assert cli.main() == 0
        assert captured["resume"] is False


class TestProcessFromConfigResume:
    def _config(self, tmp_path):
        input_dir = tmp_path / "videos"
        input_dir.mkdir()
        video = input_dir / "video.mp4"
        video.write_bytes(b"fake")

        config_path = tmp_path / "config.yaml"
        config = BatchConfig(input=str(input_dir), output=str(tmp_path / "out"))
        config._source_path = str(config_path)

        (tmp_path / "progress.csv").write_text(
            "input_path,output_path,processing_time_minutes,total_detections\n"
            f"{video.resolve()},{tmp_path},1.2,10\n",
            encoding="utf-8",
        )
        return config, video

    def _patch_process_directory(self, monkeypatch):
        captured = {}

        def fake_process_directory(**kwargs):
            captured["skip_paths"] = kwargs.get("skip_paths")
            captured["progress_tracker"] = kwargs.get("progress_tracker")
            return []

        monkeypatch.setattr(facade, "process_directory", fake_process_directory)
        return captured

    def test_resume_true_uses_progress_file(self, tmp_path, monkeypatch):
        config, video = self._config(tmp_path)
        captured = self._patch_process_directory(monkeypatch)

        facade.process_from_config(config)

        assert video.resolve().as_posix() in captured["skip_paths"]
        assert captured["progress_tracker"] is not None

    def test_resume_false_ignores_progress_file(self, tmp_path, monkeypatch):
        config, _ = self._config(tmp_path)
        captured = self._patch_process_directory(monkeypatch)

        facade.process_from_config(config, resume=False)

        assert captured["skip_paths"] == set()
        assert captured["progress_tracker"] is None
