"""Tests del DisplayManager en entornos con y sin TTY (Colab, CI)."""

import sys

import pytest

from afora_vc.utils.display import DisplayManager


class TestInteractiveDetection:
    def test_env_var_disables_progress(self, monkeypatch):
        monkeypatch.setenv("AFORA_PROGRESS", "0")
        assert DisplayManager().interactive is False

    def test_env_var_enables_progress(self, monkeypatch):
        monkeypatch.setenv("AFORA_PROGRESS", "1")
        assert DisplayManager().interactive is True

    def test_non_tty_is_detected(self, monkeypatch):
        monkeypatch.delenv("AFORA_PROGRESS", raising=False)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
        assert DisplayManager().interactive is False


class TestPlainProgress:
    def _display(self, **kwargs):
        display = DisplayManager(interactive=False, **kwargs)
        # Sin throttle para que cada update imprima una linea.
        display.progress_log_interval = 0.0
        return display

    def test_prints_percentage_lines(self, capsys):
        display = self._display()
        display.start_progress(total=100, description="Procesando video")
        display.update_progress(advance=50)
        display.update_progress(advance=50)
        display.stop_progress()

        out = capsys.readouterr().out
        assert "Procesando video: 50% (50/100)" in out
        assert "Procesando video: 100% (100/100)" in out

    def test_prints_frames_when_total_unknown(self, capsys):
        display = self._display()
        display.start_progress(total=0, description="Procesando video")
        display.update_progress(advance=120)
        display.update_progress(advance=10)
        display.stop_progress()

        out = capsys.readouterr().out
        assert "Procesando video: 120 frames" in out
        assert "Procesando video: 130 frames" in out

    def test_quiet_suppresses_plain_progress(self, capsys):
        display = self._display(quiet=True)
        display.start_progress(total=10)
        display.update_progress(advance=10)
        display.stop_progress()

        assert capsys.readouterr().out == ""

    def test_interactive_mode_does_not_print_plain_lines(self, capsys):
        display = DisplayManager(interactive=True)
        display.progress_log_interval = 0.0
        display.start_progress(total=10)
        display.update_progress(advance=10)
        display.stop_progress()

        # tqdm escribe a stderr; stdout queda limpio.
        assert capsys.readouterr().out == ""


class TestVideoFinished:
    def test_prints_compact_line(self, capsys):
        display = DisplayManager(interactive=False)
        display.video_finished(1, 4, "video.mp4", frames=45, seconds=3.24)

        out = capsys.readouterr().out
        assert "[1/4] video.mp4 -> OK (45 frames, 3.2s)" in out

    def test_prints_error_line(self, capsys):
        display = DisplayManager(interactive=False)
        display.video_finished(2, 4, "roto.mp4", ok=False)

        out = capsys.readouterr().out
        assert "[2/4] roto.mp4 -> ERROR (0 frames, 0.0s)" in out

    def test_silent_only_when_interactive(self, capsys):
        display = DisplayManager(interactive=True)
        display.video_finished(1, 1, "video.mp4", frames=10, seconds=1.0)

        # En modo interactivo las barras muestran el progreso; stdout queda limpio.
        assert capsys.readouterr().out == ""

    def test_quiet_still_prints_compact_line(self, capsys):
        display = DisplayManager(quiet=True)
        display.video_finished(1, 1, "video.mp4", frames=10, seconds=1.0)

        out = capsys.readouterr().out
        assert "[1/1] video.mp4 -> OK (10 frames, 10.0 fps, 0.02 min)" in out
