import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class StageSnapshot:
    stage: str
    frame: int
    total: int
    class_counts: dict[str, int] = field(default_factory=dict)
    track_ids: set[int] = field(default_factory=set)


class Auditor:
    def __init__(self):
        self._snapshots: list[StageSnapshot] = []
        self._lock = threading.Lock()

    def snapshot(
        self,
        stage: str,
        frame: int,
        total: int,
        class_counts: dict[str, int] | None = None,
        track_ids: set[int] | None = None,
    ) -> None:
        with self._lock:
            self._snapshots.append(
                StageSnapshot(
                    stage=stage,
                    frame=frame,
                    total=total,
                    class_counts=class_counts or {},
                    track_ids=track_ids or set(),
                )
            )

    @property
    def snapshots_size(self) -> int:
        with self._lock:
            return len(self._snapshots)

    def report(self) -> str:
        with self._lock:
            if not self._snapshots:
                return "No audit data."

        stages_order = ["detect", "track", "mask", "zone_check", "csv"]
        stage_frames: dict[str, list[StageSnapshot]] = defaultdict(list)
        for s in self._snapshots:
            stage_frames[s.stage].append(s)

        lines = []
        lines.append("=" * 72)
        lines.append("AUDIT REPORT - Detection pipeline stage analysis")
        lines.append("=" * 72)

        frames_set = {s.frame for s in self._snapshots}
        lines.append(f"Total frames audited: {len(frames_set)}")

        for stage in stages_order:
            snaps = stage_frames.get(stage)
            if not snaps:
                continue

            total = sum(s.total for s in snaps)
            avg = total / len(snaps)
            all_classes: Counter[str] = Counter()
            all_track_ids: set[int] = set()
            non_zero_frames = sum(1 for s in snaps if s.total > 0)
            max_in_frame = max(s.total for s in snaps)

            for s in snaps:
                all_classes.update(s.class_counts)
                if s.track_ids:
                    all_track_ids |= s.track_ids

            lines.append("")
            lines.append(f"[{stage}]")
            lines.append(f"  Total detections : {total}")
            lines.append(f"  Per frame avg    : {avg:.1f}")
            lines.append(f"  Frames w/ dets   : {non_zero_frames} / {len(snaps)}")
            lines.append(f"  Max in frame     : {max_in_frame}")
            if all_classes:
                classes_str = ", ".join(f"{k}={v}" for k, v in all_classes.most_common())
                lines.append(f"  By class         : {classes_str}")
            if all_track_ids:
                lines.append(f"  Unique track IDs : {len(all_track_ids)}")

        lines.append("")
        lines.append("=" * 72)
        lines.append("STAGE TRANSITION (per-frame deltas)")
        lines.append("=" * 72)
        lines.append(f"  {'Frame':>6}  {'detect':>7}  {'track':>7}  {'mask':>7}  {'zone':>7}  {'csv':>7}")
        lines.append(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

        # Find common frames across stages
        frame_stages: dict[int, dict[str, int]] = defaultdict(dict)
        for s in self._snapshots:
            frame_stages[s.frame][s.stage] = s.total

        min_frames = sorted(frame_stages.keys())
        for fi in min_frames:
            stages_data = frame_stages[fi]
            line = f"  {fi:>6}"
            for st in stages_order:
                val = stages_data.get(st, "")
                if val != "":
                    line += f"  {val:>5}"
                else:
                    line += f"  {'-':>5}"
            lines.append(line)

        return "\n".join(lines)
