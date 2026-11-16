from __future__ import annotations

import os
import sys
import time
from collections import Counter
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tqdm import tqdm


_BAR_FMT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}"
_DEFAULT_TQDM_KW = dict(ascii=True, ncols=100, bar_format=_BAR_FMT)

_STYLE_HEADER = "bold cyan"
_STYLE_OK = "bold green"
_STYLE_WARN = "bold yellow"
_STYLE_ERROR = "bold red"
_STYLE_INFO = "white"
_STYLE_ACCENT = "bold magenta"
_STYLE_DIM = "dim white"


class DisplayManager:
    def __init__(
        self,
        verbose: bool = True,
        quiet: bool = False,
        interactive: bool | None = None,
    ):
        self.verbose = verbose
        self.quiet = quiet
        self.console = Console(highlight=False)
        self._err_console = Console(highlight=False, stderr=True)
        # En Colab/CI stderr no es TTY: tqdm con "\r" se ve mal o no se ve.
        self.interactive = (
            self._detect_interactive() if interactive is None else interactive
        )
        # Segundos minimos entre lineas de progreso en modo no interactivo.
        self.progress_log_interval = 30.0
        self._tqdm_bar: tqdm | None = None
        self._scan_tqdm: tqdm | None = None
        self._video_tqdm: tqdm | None = None
        self._plain_active = False
        self._plain_desc = ""
        self._plain_total = 0
        self._plain_count = 0
        self._plain_last_emit = 0.0

    @staticmethod
    def _detect_interactive() -> bool:
        env = os.environ.get("AFORA_PROGRESS")
        if env is not None:
            return env.strip().lower() not in ("0", "false", "no", "off")
        try:
            return bool(sys.stderr.isatty())
        except (AttributeError, ValueError):
            return False

    def print(self, *args: Any, **kwargs: Any) -> None:
        if not self.quiet:
            self.console.print(*args, **kwargs)

    # ── Start summary ──────────────────────────────────────

    def show_start_summary(
        self,
        video_name: str,
        total_frames: int,
        fps: float,
        resolution: tuple[int, int],
        backend: str,
        model: str,
        device: str,
        batch_size: int,
        pipeline_mode: str,
        reduce_factor: int,
        zones_count: int,
        save_video: bool,
        save_csv: bool,
    ) -> None:
        if self.quiet:
            return

        info = Table.grid(padding=(0, 2))
        info.add_column(style=_STYLE_DIM)
        info.add_column(style=_STYLE_INFO)

        info.add_row("Video", f"[bold]{video_name}[/]")
        info.add_row("Resolucion", f"{resolution[0]}x{resolution[1]}")
        info.add_row("Frames", f"{total_frames} @ {fps:.1f} FPS")
        info.add_row("Backend", f"[{_STYLE_ACCENT}]{backend}[/] ({model})")
        info.add_row("Dispositivo", device)
        info.add_row("Batch size", str(batch_size))
        info.add_row("Pipeline", pipeline_mode)
        info.add_row("Reduce", str(reduce_factor))
        info.add_row("Zonas", str(zones_count))
        info.add_row("Video out", "[OK]" if save_video else "[NO]")
        info.add_row("CSV out", "[OK]" if save_csv else "[NO]")

        panel = Panel(
            info,
            title=f"[{_STYLE_HEADER}]> INICIANDO PROCESAMIENTO[/]",
            border_style=_STYLE_ACCENT,
            padding=(1, 2),
        )
        self.console.print(panel)

    # ── Single-video progress bar ───────────────────────────

    def start_progress(self, total: int, description: str = "Procesando") -> None:
        if self.quiet:
            return
        self._plain_desc = description
        self._plain_total = int(total or 0)
        self._plain_count = 0
        # Primera linea de progreso recien pasado el intervalo: el panel de
        # inicio ya avisa que el video arranco.
        self._plain_last_emit = time.monotonic()
        self._plain_active = True
        if not self.interactive:
            return
        self._tqdm_bar = tqdm(
            total=total,
            desc=description,
            unit="frames",
            **_DEFAULT_TQDM_KW,
        )

    def update_progress(
        self,
        advance: int = 1,
        total: int | None = None,
        extra: str = "",
    ) -> None:
        if total is not None:
            self._plain_total = int(total)
            if self._tqdm_bar is not None:
                self._tqdm_bar.total = total
                self._tqdm_bar.refresh()
        if not self.interactive:
            if not self._plain_active:
                return
            self._plain_count += advance
            now = time.monotonic()
            if (
                self._plain_last_emit
                and now - self._plain_last_emit < self.progress_log_interval
            ):
                return
            self._plain_last_emit = now
            self._print_plain_progress(extra)
            return
        if self._tqdm_bar is None:
            return
        if extra:
            self._tqdm_bar.set_postfix_str(extra)
        self._tqdm_bar.update(advance)

    def _plain_status(self) -> str:
        total = self._plain_total
        if total > 0:
            # El total puede quedar corto (max_minutes, frames finales del
            # contenedor): se recorta para no mostrar 107%.
            count = min(self._plain_count, total)
            pct = min(100, count * 100 // total)
            return f"{self._plain_desc}: {pct}% ({count}/{total})"
        return f"{self._plain_desc}: {self._plain_count} frames"

    def _print_plain_progress(self, extra: str = "") -> None:
        line = self._plain_status()
        if extra:
            line = f"{line} {extra}"
        self.console.print(line)

    def stop_progress(self) -> None:
        if not self.interactive:
            if self._plain_active and not self.quiet:
                self.console.print(self._plain_status())
            self._plain_active = False
            return
        if self._tqdm_bar is not None:
            self._tqdm_bar.close()
            self._tqdm_bar = None

    # ── End-of-video summary ───────────────────────────────

    def show_end_summary(
        self,
        video_name: str,
        processing_time: float,
        frames_processed: int,
        fps: float,
        total_detections: int,
        zone_counts: dict[str, int] | None = None,
        csv_path: str | None = None,
        video_path: str | None = None,
        profile_path: str | None = None,
    ) -> None:
        if self.quiet:
            # En quiet la linea por video ya la imprime video_finished en la
            # facade; aca no repetimos.
            return

        self.console.print()

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style=_STYLE_DIM)
        summary.add_column(style=_STYLE_INFO)

        minutes = int(processing_time // 60)
        seconds = int(processing_time % 60)
        time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

        summary.add_row("Video", f"[bold]{video_name}[/]")
        summary.add_row("Tiempo", time_str)
        summary.add_row("Frames", str(frames_processed))
        summary.add_row("FPS", f"{fps:.1f}")
        summary.add_row(
            "Detecciones",
            f"[{_STYLE_OK}]{total_detections}[/]",
        )
        if video_path:
            summary.add_row("Video out", f"[dim]{video_path}[/]")
        if csv_path:
            summary.add_row("CSV", f"[dim]{csv_path}[/]")
        if profile_path:
            summary.add_row("Profile", f"[dim]{profile_path}[/]")

        if zone_counts:
            zone_table = Table.grid(padding=(0, 2))
            zone_table.add_column(style=_STYLE_ACCENT)
            zone_table.add_column(style=_STYLE_INFO, justify="right")
            for zone, count in sorted(zone_counts.items()):
                zone_table.add_row(f"  Zona {zone}", str(count))

            panel = Panel(
                zone_table,
                title="[bold]DETECCION POR ZONA[/]",
                border_style=_STYLE_OK,
                padding=(1, 2),
            )
            summary.add_row("", panel)

        final_panel = Panel(
            summary,
            title=f"[{_STYLE_OK}]PROCESO COMPLETADO[/]",
            border_style=_STYLE_OK,
            padding=(1, 2),
        )
        self.console.print(final_panel)

    # ── Scan mode (nested progress) ────────────────────────

    def start_scan(
        self,
        total_videos: int,
        description: str = "Procesando videos",
    ) -> None:
        if self.quiet or not self.interactive:
            return
        self._scan_tqdm = tqdm(
            total=total_videos,
            desc=description,
            position=0,
            leave=True,
            **_DEFAULT_TQDM_KW,
        )

    def update_scan(self, advance: int = 1, extra: str = "") -> None:
        if not self.interactive or self._scan_tqdm is None:
            return
        if extra:
            self._scan_tqdm.set_postfix_str(extra)
        self._scan_tqdm.update(advance)

    def start_video_progress(
        self,
        total: int,
        description: str = "Video actual",
    ) -> None:
        if not self.interactive or self._scan_tqdm is None:
            return
        if self._video_tqdm is not None:
            self._video_tqdm.close()
        self._video_tqdm = tqdm(
            total=total,
            desc=f"  {description}",
            position=1,
            leave=False,
            **_DEFAULT_TQDM_KW,
        )

    def update_video_progress(self, advance: int = 1, extra: str = "") -> None:
        if not self.interactive or self._video_tqdm is None:
            return
        if extra:
            self._video_tqdm.set_postfix_str(extra)
        self._video_tqdm.update(advance)

    def stop_video_progress(self) -> None:
        if self._video_tqdm is not None:
            self._video_tqdm.close()
            self._video_tqdm = None

    def stop_scan(self) -> None:
        if self._video_tqdm is not None:
            self._video_tqdm.close()
            self._video_tqdm = None
        if self._scan_tqdm is not None:
            self._scan_tqdm.close()
            self._scan_tqdm = None

    def video_finished(
        self,
        index: int,
        total: int,
        name: str,
        frames: int = 0,
        seconds: float = 0.0,
        ok: bool = True,
    ) -> None:
        """Linea compacta por video.

        En modo **quiet** imprime siempre una linea de estado; en entornos sin
        TTY (Colab/CI) también la emite; en una terminal interactiva no imprime
        nada porque las barras ya muestran el progreso.
        """
        if self.interactive and not self.quiet:
            return
        status = "OK" if ok else "ERROR"
        if self.quiet:
            minutes = seconds / 60
            fps = frames / seconds if seconds > 0 else 0
            self.console.print(
                f"[{index}/{total}] {name} -> {status} "
                f"({frames} frames, {fps:.1f} fps, {minutes:.2f} min)"
            )
            if not self.interactive:
                self.console.file.flush()
        else:
            self.console.print(
                f"[{index}/{total}] {name} -> {status} ({frames} frames, {seconds:.1f}s)"
            )

    # ── Error display ──────────────────────────────────────

    def show_error(self, message: str, context: str = "") -> None:
        text = Text()
        text.append("[ERROR] ", _STYLE_ERROR)
        if context:
            text.append(f" [{context}]", _STYLE_DIM)
        text.append(f": {message}", _STYLE_INFO)
        self._error_console.print(text)

    @property
    def _error_console(self) -> Console:
        """Consola para errores.

        En una terminal van a stderr (convencion Unix), pero en Colab/CI stderr
        no siempre se muestra: ahi el error tiene que quedar en stdout o el
        comando parece no haber hecho nada.
        """
        return self.console if not self.interactive else self._err_console

    def show_warning(self, message: str) -> None:
        text = Text()
        text.append("[WARN] ", _STYLE_WARN)
        text.append(message, _STYLE_WARN)
        self.console.print(text)

    def show_success(self, message: str) -> None:
        text = Text()
        text.append("[OK] ", _STYLE_OK)
        text.append(message, _STYLE_OK)
        self.console.print(text)
        if not self.interactive:
            self.console.file.flush()

    def show_info(self, message: str) -> None:
        if not self.quiet:
            if not self.interactive:
                # En Colab/CI rich puede bufferizar; flush explicito evita que
                # el comando parezca colgado durante operaciones lentas (scan
                # de Drive por FUSE, descarga de modelos, etc.).
                self.console.print(message, style=_STYLE_INFO)
                self.console.file.flush()
            else:
                self.console.print(message, style=_STYLE_INFO)

    # ── Audit report ───────────────────────────────────────

    def show_audit_report(self, report: str) -> None:
        if not self.quiet:
            panel = Panel(
                report,
                title="[bold]AUDITOR REPORT[/]",
                border_style=_STYLE_WARN,
                padding=(1, 2),
            )
            self.console.print(panel)
