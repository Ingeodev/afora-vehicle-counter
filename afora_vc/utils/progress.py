from __future__ import annotations

import csv
import os
from typing import Set


class ProgressTracker:
    def __init__(self, path: str):
        self.path = path
        self._completed: Set[str] = set()
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["input_path", "output_path", "processing_time_minutes", "total_detections"])
            return
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("input_path", "").strip()
                if p:
                    self._completed.add(os.path.abspath(p))

    def is_completed(self, video_path: str) -> bool:
        return os.path.abspath(video_path) in self._completed

    def mark_completed(self, video_path: str, output_path: str, processing_time_seconds: float, total_detections: int) -> None:
        abs_video = os.path.abspath(video_path)
        minutes = processing_time_seconds / 60
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([abs_video, os.path.abspath(output_path), f"{minutes:.2f}", str(total_detections)])
        self._completed.add(abs_video)

    @property
    def completed_paths(self) -> Set[str]:
        return self._completed

    @property
    def completed_count(self) -> int:
        return len(self._completed)
