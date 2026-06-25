"""
Utility to organize video files into date-based subdirectories.

Parses timestamps embedded in filenames (pattern: ``_YYYYMMDDHHMMSS_``),
moves each video into a subfolder named by its date, and logs every
move so the operation can be rolled back.

Usage (via CLI):
    afora utils group-by-date <path>
    afora utils group-by-date <path> --rollback
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime


VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
LOG_FILENAME = "_separation_log.txt"


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass
class FileMove:
    original_path: str
    destination_path: str


@dataclass
class SortResult:
    moved: list[FileMove] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _parse_timestamp_segment(segment: str) -> tuple[str, str]:
    """Extract date and formatted datetime from a 14-digit segment.

    >>> _parse_timestamp_segment("20260409124115")
    ('2026-04-09', '2026-04-09_12.41.15')
    """
    if len(segment) < 14:
        raise ValueError(f"Segment too short: {segment}")
    date_part = f"{segment[0:4]}-{segment[4:6]}-{segment[6:8]}"
    formatted = f"{date_part}_{segment[8:10]}.{segment[10:12]}.{segment[12:14]}"
    return date_part, formatted


def _extract_timestamp(filename: str) -> tuple[str, str] | None:
    """Extract date and formatted timestamp from a video filename.

    Expects the 3rd underscore-delimited part to contain a 14-digit
    timestamp like ``YYYYMMDDHHMMSS``.

    Returns ``(date_str, formatted_ts)`` or ``None``.
    """
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    raw = parts[2]
    raw = re.sub(r"\..+$", "", raw)
    try:
        return _parse_timestamp_segment(raw)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _unique_path(destination: str) -> str:
    """Append a counter suffix if the destination already exists."""
    if not os.path.exists(destination):
        return destination
    base, ext = os.path.splitext(destination)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"


def sort_videos_by_date(folder_path: str) -> SortResult:
    """Move all videos in *folder_path* into subdirectories named by date.

    Each file is renamed to ``{parent_folder}_{YYYY-MM-DD_HH.MM.SS}.ext``
    and a log file (``_separation_log.txt``) is written to *folder_path*
    to enable rollback.

    Returns a ``SortResult`` with the list of moves, skips, and errors.
    """
    result = SortResult()
    folder_name = os.path.basename(folder_path.rstrip("\\/"))

    if not os.path.isdir(folder_path):
        result.errors.append(f"Path does not exist: {folder_path}")
        return result

    videos = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        result.errors.append(f"No video files found in: {folder_path}")
        return result

    moves: list[FileMove] = []
    current_date: str | None = None
    current_dir: str | None = None

    for filename in videos:
        ts = _extract_timestamp(filename)
        if ts is None:
            result.skipped.append(filename)
            print(f"[SKIP] Could not parse timestamp: {filename}")
            continue

        date_part, formatted_ts = ts

        if date_part != current_date:
            current_dir = _ensure_dir(os.path.join(folder_path, date_part))
            current_date = date_part

        ext = os.path.splitext(filename)[1]
        new_name = f"{folder_name}_{formatted_ts}{ext}"
        src = os.path.join(folder_path, filename)
        dst = _unique_path(os.path.join(current_dir, new_name))

        shutil.move(src, dst)
        moves.append(FileMove(original_path=src, destination_path=dst))
        print(f"[OK] {filename} -> {os.path.basename(dst)}  [{date_part}]")

    if moves:
        log_path = os.path.join(folder_path, LOG_FILENAME)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Video separation log\n")
            f.write(f"# Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: original_path | destination_path\n")
            f.write("-" * 60 + "\n")
            for m in moves:
                f.write(f"{m.original_path} | {m.destination_path}\n")
        print(f"[OK] Log saved: {LOG_FILENAME}")

    result.moved = moves
    return result


def rollback_sort(folder_path: str) -> SortResult:
    """Reverse a previous :func:`sort_videos_by_date` operation.

    Reads the ``_separation_log.txt`` from *folder_path* and moves every
    file back to its original location. Empty date folders are removed.
    The log is deleted on success.
    """
    result = SortResult()
    log_path = os.path.join(folder_path, LOG_FILENAME)

    if not os.path.isfile(log_path):
        result.errors.append(f"Log file not found: {log_path}")
        return result

    moves: list[tuple[str, str]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            orig = parts[0].strip()
            dest = parts[1].strip()
            moves.append((orig, dest))

    errors = 0
    for orig, dest in reversed(moves):
        if not os.path.isfile(dest):
            print(f"[SKIP] Does not exist: {os.path.basename(dest)}")
            continue
        try:
            os.makedirs(os.path.dirname(orig), exist_ok=True)
            shutil.move(dest, orig)
            print(f"[ROLLBACK] {os.path.basename(dest)} -> {os.path.basename(orig)}")
            result.moved.append(FileMove(original_path=dest, destination_path=orig))
        except Exception as e:
            print(f"[ERROR] Could not restore {os.path.basename(dest)}: {e}")
            errors += 1
            result.errors.append(str(e))

    # Remove empty date folders
    for item in os.listdir(folder_path):
        full = os.path.join(folder_path, item)
        if os.path.isdir(full) and item != os.path.basename(folder_path):
            try:
                os.rmdir(full)
            except OSError:
                pass

    if errors == 0:
        os.remove(log_path)
        print(f"[OK] Log removed: {LOG_FILENAME}")
    else:
        print(f"[WARN] {errors} error(s) during rollback. Check manually.")

    return result
