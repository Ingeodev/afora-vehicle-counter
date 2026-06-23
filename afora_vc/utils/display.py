from __future__ import annotations

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
    def __init__(self, verbose: bool = True, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.console = Console(highlight=False)
        self._err_console = Console(highlight=False, stderr=True)
        self._tqdm_bar: tqdm | None = None
        self._scan_tqdm: tqdm | None = None
        self._video_tqdm: tqdm | None = None

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
        if self._tqdm_bar is None:
            return
        if total is not None:
            self._tqdm_bar.total = total
            self._tqdm_bar.refresh()
        if extra:
            self._tqdm_bar.set_postfix_str(extra)
        self._tqdm_bar.update(advance)

    def stop_progress(self) -> None:
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
        if self.quiet:
            return
        self._scan_tqdm = tqdm(
            total=total_videos,
            desc=description,
            position=0,
            leave=True,
            **_DEFAULT_TQDM_KW,
        )

    def update_scan(self, advance: int = 1, extra: str = "") -> None:
        if self._scan_tqdm is None:
            return
        if extra:
            self._scan_tqdm.set_postfix_str(extra)
        self._scan_tqdm.update(advance)

    def start_video_progress(
        self,
        total: int,
        description: str = "Video actual",
    ) -> None:
        if self._scan_tqdm is None:
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
        if self._video_tqdm is None:
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

    # ── Error display ──────────────────────────────────────

    def show_error(self, message: str, context: str = "") -> None:
        text = Text()
        text.append("[ERROR] ", _STYLE_ERROR)
        if context:
            text.append(f" [{context}]", _STYLE_DIM)
        text.append(f": {message}", _STYLE_INFO)
        self._err_console.print(text)

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

    def show_info(self, message: str) -> None:
        if not self.quiet:
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
