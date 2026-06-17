from typing import Sequence

from ..domain.schemas import ZoneEntry
from ..domain.ports.storage import StorageWriter


class CSVExporter:
    def __init__(self, storage: StorageWriter):
        self.storage = storage

    def export_detections(
        self, entries: Sequence[ZoneEntry], path: str, include_header: bool = True
    ) -> str:
        if not entries:
            return path

        rows = [
            {
                "vehicle_id": e.vehicle_id,
                "vehicle_type": e.vehicle_type,
                "zone": e.zone,
                "date": e.date,
                "exact_time": e.exact_time,
                "timestamp_formatted": e.timestamp_formatted,
            }
            for e in entries
        ]

        lines: list[str] = []

        if include_header:
            headers = list(rows[0].keys())
            lines.append(",".join(headers))

        for row in rows:
            values = [str(v) for v in row.values()]
            lines.append(",".join(values))

        content = "\n".join(lines)
        self.storage.write_text(path, content)

        return path

    def export_summary(
        self, zone_counts: dict[str, int], path: str, include_total: bool = True
    ) -> str:
        lines = ["zone,count"]

        total = 0
        for zone, count in sorted(zone_counts.items()):
            lines.append(f"{zone},{count}")
            total += count

        if include_total:
            lines.append(f"TOTAL,{total}")

        content = "\n".join(lines)
        self.storage.write_text(path, content)

        return path

    def export_vehicle_types(self, entries: Sequence[ZoneEntry], path: str) -> str:
        type_counts: dict[str, int] = {}
        for entry in entries:
            vtype = entry.vehicle_type
            type_counts[vtype] = type_counts.get(vtype, 0) + 1

        lines = ["vehicle_type,count"]
        for vtype, count in sorted(type_counts.items()):
            lines.append(f"{vtype},{count}")

        content = "\n".join(lines)
        self.storage.write_text(path, content)

        return path

    def export_all(
        self, entries: Sequence[ZoneEntry], zone_counts: dict[str, int], base_path: str
    ) -> dict[str, str]:
        paths = {}
        det_path = f"{base_path}_detections.csv"
        self.export_detections(entries, det_path)
        paths["detections"] = det_path

        summary_path = f"{base_path}_summary.csv"
        self.export_summary(zone_counts, summary_path)
        paths["summary"] = summary_path

        types_path = f"{base_path}_types.csv"
        self.export_vehicle_types(entries, types_path)
        paths["types"] = types_path

        return paths
