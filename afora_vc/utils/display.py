from __future__ import annotations

import time
from collections import Counter
from typing import Any, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


_STYLE_HEADER = "bold cyan"
_STYLE_OK = "bold green"
_STYLE_WARN = "bold yellow"
_STYLE_ERROR = "bold red"
_STYLE_INFO = "white"
_STYLE_ACCENT = "bold magenta"
_STYLE_DIM = "dim white"


class DisplayManager:
    def __init__(self, verbose: bool = True, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.console = Console(highlight=False)
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._live: Live | None = None
        self._scan_progress: Progress | None = None
        self._scan_task: TaskID | None = None
        self._video_progress: Progress | None = None
        self._video_task: TaskID | None = None
        self._scan_live: Live | None = None

    # ── Helpers ────────────────────────────────────────────

    def _emoji(self, name: str) -> str:
        return {}.get(name, "")

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
        info.add_row("Resolución", f"{resolution[0]}×{resolution[1]}")
        info.add_row("Frames", f"{total_frames} @ {fps:.1f} FPS")
        info.add_row("Backend", f"[{_STYLE_ACCENT}]{backend}[/] ({model})")
        info.add_row("Dispositivo", device)
        info.add_row("Batch size", str(batch_size))
        info.add_row("Pipeline", pipeline_mode)
        info.add_row("Reduce", str(reduce_factor))
        info.add_row("Zonas", str(zones_count))
        info.add_row("Video out", "✅" if save_video else "❌")
        info.add_row("CSV out", "✅" if save_csv else "❌")

        panel = Panel(
            info,
            title=f"[{_STYLE_HEADER}]▶ INICIANDO PROCESAMIENTO[/]",
            border_style=_STYLE_ACCENT,
            padding=(1, 2),
        )
        self.console.print(panel)

    # ── Single-video progress bar ───────────────────────────

    def start_progress(self, total: int, description: str = "Procesando") -> None:
        if self.quiet:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/]"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[extra]}[/]"),
            console=self.console,
        )
        self._task_id = self._progress.add_task(
            description, total=total, extra=""
        )
        self._progress.start()

    def update_progress(
        self,
        advance: int = 1,
        total: int | None = None,
        extra: str = "",
    ) -> None:
        if self._progress is None or self._task_id is None:
            return
        if total is not None:
            self._progress.update(self._task_id, total=total)
        if extra:
            self._progress.update(self._task_id, extra=extra)
        self._progress.update(self._task_id, advance=advance)

    def stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

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
                title="[bold]📍 DETECCIÓN POR ZONA[/]",
                border_style=_STYLE_OK,
                padding=(1, 2),
            )
            summary.add_row("", panel)

        final_panel = Panel(
            summary,
            title=f"[{_STYLE_OK}]✅ PROCESO COMPLETADO[/]",
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
        if self.quiet:
            return
        self._scan_progress = Progress(
            TextColumn("[bold]{task.description}[/]"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[extra]}[/]"),
            console=self.console,
        )
        self._video_progress = Progress(
            TextColumn("  [dim]{task.description}[/]"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[extra]}[/]"),
            console=self.console,
        )

        self._scan_task = self._scan_progress.add_task(
            description, total=total_videos, extra=""
        )

        self._scan_live = Live(
            Panel(
                self._build_scan_display(),
                title=f"[{_STYLE_HEADER}]▶ PROCESANDO CARPETA[/]",
                border_style=_STYLE_ACCENT,
            ),
            console=self.console,
            refresh_per_second=4,
        )
        self._scan_live.start()

    def _build_scan_display(self) -> Table:
        table = Table.grid(padding=(0, 1))
        if self._scan_progress:
            table.add_row(self._scan_progress)
        if self._video_progress:
            table.add_row(self._video_progress)
        return table

    def update_scan(self, advance: int = 1, extra: str = "") -> None:
        if self._scan_progress is not None and self._scan_task is not None:
            if extra:
                self._scan_progress.update(self._scan_task, extra=extra)
            self._scan_progress.update(self._scan_task, advance=advance)
        if self._scan_live:
            self._scan_live.update(
                Panel(
                    self._build_scan_display(),
                    title=f"[{_STYLE_HEADER}]▶ PROCESANDO CARPETA[/]",
                    border_style=_STYLE_ACCENT,
                )
            )

    def start_video_progress(
        self,
        total: int,
        description: str = "Video actual",
    ) -> None:
        if self._video_progress is None:
            return
        self._video_task = self._video_progress.add_task(
            description, total=total, extra=""
        )
        if self._scan_live:
            self._scan_live.update(
                Panel(
                    self._build_scan_display(),
                    title=f"[{_STYLE_HEADER}]▶ PROCESANDO CARPETA[/]",
                    border_style=_STYLE_ACCENT,
                )
            )

    def update_video_progress(self, advance: int = 1, extra: str = "") -> None:
        if self._video_progress is not None and self._video_task is not None:
            if extra:
                self._video_progress.update(self._video_task, extra=extra)
            self._video_progress.update(self._video_task, advance=advance)

    def stop_video_progress(self) -> None:
        if self._video_progress is not None and self._video_task is not None:
            self._video_progress.remove_task(self._video_task)
            self._video_task = None

    def stop_scan(self) -> None:
        if self._scan_live is not None:
            self._scan_live.stop()
            self._scan_live = None
        self._scan_progress = None
        self._scan_task = None
        self._video_progress = None
        self._video_task = None

    # ── Error display ──────────────────────────────────────

    def show_error(self, message: str, context: str = "") -> None:
        import sys
        err_console = Console(highlight=False, stderr=True)
        text = Text()
        text.append("❌ ", _STYLE_ERROR)
        text.append("Error", _STYLE_ERROR)
        if context:
            text.append(f" [{context}]", _STYLE_DIM)
        text.append(f": {message}", _STYLE_INFO)
        err_console.print(text)

    def show_warning(self, message: str) -> None:
        text = Text()
        text.append("⚠️  ", _STYLE_WARN)
        text.append(message, _STYLE_WARN)
        self.console.print(text)

    def show_success(self, message: str) -> None:
        text = Text()
        text.append("✅ ", _STYLE_OK)
        text.append(message, _STYLE_OK)
        self.console.print(text)

    def show_info(self, message: str) -> None:
        if not self.quiet:
            self.console.print(message, style=_STYLE_INFO)

    # ── Audit report ───────────────────────────────────────

    def show_audit_report(self, report: str) -> None:
        if not self.quiet:
            panel = Panel(
                report,
                title="[bold]📋 AUDITOR REPORT[/]",
                border_style=_STYLE_WARN,
                padding=(1, 2),
            )
            self.console.print(panel)
