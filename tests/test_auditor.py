import pytest

from afora_vc.utils.auditor import Auditor, StageSnapshot


class TestStageSnapshot:
    def test_fields(self):
        s = StageSnapshot(stage="detect", frame=5, total=3, class_counts={"car": 2, "truck": 1}, track_ids={1, 2})
        assert s.stage == "detect"
        assert s.frame == 5
        assert s.total == 3
        assert s.class_counts == {"car": 2, "truck": 1}
        assert s.track_ids == {1, 2}

    def test_defaults(self):
        s = StageSnapshot(stage="detect", frame=0, total=0)
        assert s.class_counts == {}
        assert s.track_ids == set()


class TestAuditor:
    def test_snapshot_and_count(self):
        au = Auditor()
        au.snapshot("detect", 0, 5, class_counts={"car": 5})
        au.snapshot("track", 0, 3, class_counts={"car": 3}, track_ids={1, 2, 3})
        assert au.snapshots_size == 2

    def test_snapshot_empty(self):
        au = Auditor()
        assert au.snapshots_size == 0

    def test_thread_safety(self):
        from threading import Thread

        au = Auditor()
        errors = []

        def _worker():
            try:
                for i in range(500):
                    au.snapshot("detect", i, i)
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert au.snapshots_size == 2000
        assert not errors

    def test_report_empty(self):
        au = Auditor()
        assert "No audit data." in au.report()

    def test_report_basic(self):
        au = Auditor()
        au.snapshot("detect", 0, 5, class_counts={"car": 5})
        au.snapshot("track", 0, 3, class_counts={"car": 3}, track_ids={1, 2, 3})
        au.snapshot("zone_check", 0, 2)
        report = au.report()
        assert "detect" in report
        assert "track" in report
        assert "zone_check" in report
        assert "Total detections : 5" in report
        assert "Unique track IDs : 3" in report

    def test_report_per_frame_table(self):
        au = Auditor()
        au.snapshot("detect", 0, 5)
        au.snapshot("track", 0, 3)
        au.snapshot("detect", 1, 4)
        au.snapshot("track", 1, 4)
        report = au.report()
        assert "STAGE TRANSITION" in report
        assert "     0" in report
        assert "     1" in report

    def test_report_multiple_frames(self):
        au = Auditor()
        for i in range(10):
            au.snapshot("detect", i, i + 1)
            au.snapshot("track", i, i)
        report = au.report()
        assert "Total frames audited: 10" in report

    def test_report_with_zone_and_csv_stages(self):
        au = Auditor()
        au.snapshot("detect", 0, 5)
        au.snapshot("track", 0, 3, track_ids={1, 2})
        au.snapshot("zone_check", 0, 2)
        au.snapshot("zone_display", 0, 3)
        report = au.report()
        assert "[zone_display]" in report
