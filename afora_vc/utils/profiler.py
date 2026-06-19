import csv
import threading
import time
from dataclasses import dataclass, fields, asdict
from pathlib import Path


@dataclass
class ProfileRecord:
    video: str
    frame: int
    step: str
    duration_ms: float
    batch: int = 1


class Profiler:
    """Profiler thread-safe que registra tiempos por paso en el pipeline."""

    def __init__(self, csv_path: str):
        self._path = csv_path
        self._records: list[ProfileRecord] = []
        self._lock = threading.Lock()

    def record(
        self,
        video: str,
        frame: int,
        step: str,
        duration_ms: float,
        batch: int = 1,
    ) -> None:
        with self._lock:
            self._records.append(
                ProfileRecord(
                    video=video,
                    frame=frame,
                    step=step,
                    duration_ms=round(duration_ms, 3),
                    batch=batch,
                )
            )

    def save(self) -> str:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        field_names = [f.name for f in fields(ProfileRecord)]
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            with self._lock:
                for r in self._records:
                    writer.writerow(asdict(r))
        return self._path

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)
