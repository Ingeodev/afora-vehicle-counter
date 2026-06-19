import csv
import os
import tempfile
import time
from pathlib import Path
from threading import Thread

import pytest

from afora_vc.utils.profiler import Profiler, ProfileRecord


class TestProfileRecord:
    def test_fields(self):
        r = ProfileRecord(video="test.mp4", frame=5, step="detect", duration_ms=12.3, batch=1)
        assert r.video == "test.mp4"
        assert r.frame == 5
        assert r.step == "detect"
        assert r.duration_ms == 12.3
        assert r.batch == 1

    def test_default_batch(self):
        r = ProfileRecord(video="v", frame=0, step="s", duration_ms=1.0)
        assert r.batch == 1


class TestProfiler:
    def test_record_and_count(self):
        pf = Profiler(csv_path=":memory:")
        pf.record("v.mp4", 1, "preprocess", 10.5)
        pf.record("v.mp4", 1, "detect", 45.2)
        assert pf.record_count == 2

    def test_record_rounds_duration(self):
        pf = Profiler(csv_path=":memory:")
        pf.record("v.mp4", 1, "step", 1.23456)
        assert pf._records[0].duration_ms == 1.235

    def test_thread_safety(self):
        pf = Profiler(csv_path=":memory:")
        errors = []

        def _worker():
            try:
                for i in range(500):
                    pf.record("v.mp4", i, "detect", float(i))
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert pf.record_count == 2000
        assert not errors

    def test_save_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "profile.csv")
            pf = Profiler(csv_path=csv_path)
            pf.record("v.mp4", 0, "preprocess", 1.0)
            pf.record("v.mp4", 0, "detect", 2.0)

            result = pf.save()
            assert result == csv_path
            assert os.path.exists(csv_path)

            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[0]["step"] == "preprocess"
            assert rows[0]["duration_ms"] == "1.0"
            assert rows[1]["step"] == "detect"

    def test_save_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "sub", "profile.csv")
            pf = Profiler(csv_path=csv_path)
            pf.record("v.mp4", 0, "preprocess", 1.0)
            pf.save()
            assert os.path.exists(csv_path)

    def test_save_without_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "profile.csv")
            pf = Profiler(csv_path=csv_path)
            pf.save()
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 0

    def test_csv_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "profile.csv")
            pf = Profiler(csv_path=csv_path)
            pf.record("v.mp4", 1, "detect", 10.0, batch=3)
            pf.save()

            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["video"] == "v.mp4"
            assert rows[0]["frame"] == "1"
            assert rows[0]["step"] == "detect"
            assert rows[0]["duration_ms"] == "10.0"
            assert rows[0]["batch"] == "3"

    def test_record_count_empty(self):
        pf = Profiler(csv_path=":memory:")
        assert pf.record_count == 0

    def test_multiple_records_same_frame(self):
        pf = Profiler(csv_path=":memory:")
        pf.record("v.mp4", 1, "preprocess", 10.0)
        pf.record("v.mp4", 1, "detect", 20.0)
        pf.record("v.mp4", 1, "track", 5.0)
        pf.record("v.mp4", 1, "visualize", 15.0)
        assert pf.record_count == 4
