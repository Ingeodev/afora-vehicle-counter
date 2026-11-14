"""Tests del soporte nativo para grabaciones DVR de Dahua (``.dav``)."""

import os
from datetime import datetime

from afora_vc.adapters.video.dvr import (
    DvrSegment,
    dvr_logical_name,
    group_dvr_paths,
    group_dvr_segments,
    is_dvr_path,
    video_identity,
)
from afora_vc.domain.counting.services import _VIDEO_TS_PATTERN


def _touch_dav(tmp_path, day: str, filename: str) -> str:
    folder = tmp_path / day / "001" / "dav" / filename[:2]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_bytes(b"")
    return str(path)


def _csv_name(video_name: str) -> str:
    return os.path.splitext(video_name)[0] + "_detections.csv"


def test_is_dvr_path() -> None:
    assert is_dvr_path("grabacion.DAV")
    assert is_dvr_path("grabacion.dav")
    assert not is_dvr_path("grabacion.mp4")


def test_parse_segment_uses_filename_and_date_folder(tmp_path) -> None:
    path = _touch_dav(
        tmp_path, "2026-11-04", "10.00.00-10.05.00[R][0@0][0][-18000].dav"
    )

    segment = DvrSegment.from_path(path)

    assert segment.start == datetime(2026, 11, 4, 10, 0, 0)
    assert segment.end == datetime(2026, 11, 4, 10, 5, 0)
    assert segment.duration_seconds == 300


def test_parse_segment_crossing_midnight(tmp_path) -> None:
    path = _touch_dav(tmp_path, "2026-11-04", "23.58.00-00.03.00[R].dav")

    segment = DvrSegment.from_path(path)

    assert segment.start == datetime(2026, 11, 4, 23, 58, 0)
    assert segment.end == datetime(2026, 11, 5, 0, 3, 0)


def test_group_contiguous_segments(tmp_path) -> None:
    first = DvrSegment.from_path(
        _touch_dav(tmp_path, "2026-11-04", "10.00.00-10.05.00[R].dav")
    )
    second = DvrSegment.from_path(
        _touch_dav(tmp_path, "2026-11-04", "10.05.00-10.10.00[R].dav")
    )

    recordings = group_dvr_segments([second, first])

    assert len(recordings) == 1
    assert recordings[0].start == datetime(2026, 11, 4, 10, 0, 0)
    assert recordings[0].duration_seconds == 600
    assert recordings[0].logical_name == "dav_2026-11-04_10.00.00.dav"


def test_group_union_duration_ignores_overlap(tmp_path) -> None:
    first = DvrSegment.from_path(
        _touch_dav(tmp_path, "2026-11-04", "10.00.00-10.05.00[R].dav")
    )
    second = DvrSegment.from_path(
        _touch_dav(tmp_path, "2026-11-04", "10.04.00-10.10.00[R].dav")
    )

    recordings = group_dvr_segments([first, second])

    assert len(recordings) == 1
    assert recordings[0].duration_seconds == 600


def test_group_respects_real_gaps(tmp_path) -> None:
    first = DvrSegment.from_path(
        _touch_dav(tmp_path, "2026-11-04", "10.00.00-10.05.00[R].dav")
    )
    second = DvrSegment.from_path(
        _touch_dav(tmp_path, "2026-11-04", "10.20.00-10.25.00[R].dav")
    )

    assert len(group_dvr_segments([first, second])) == 2


def test_group_splits_across_days(tmp_path) -> None:
    first = DvrSegment.from_path(
        _touch_dav(tmp_path, "2026-11-04", "23.55.00-23.59.00[R].dav")
    )
    second = DvrSegment.from_path(
        _touch_dav(tmp_path, "2026-11-05", "00.00.00-00.05.00[R].dav")
    )

    recordings = group_dvr_segments([first, second])

    assert len(recordings) == 2
    assert recordings[0].logical_name.startswith("dav_2026-11-04_")
    assert recordings[1].logical_name.startswith("dav_2026-11-05_")


def test_identity_multi_segment_matches_csv_pattern(tmp_path) -> None:
    first = _touch_dav(tmp_path, "2026-11-04", "10.00.00-10.05.00[R].dav")
    second = _touch_dav(tmp_path, "2026-11-04", "10.05.00-10.10.00[R].dav")

    identity = video_identity([second, first])

    assert identity.is_dvr
    assert identity.name == "dav_2026-11-04_10.00.00.dav"
    assert identity.start == datetime(2026, 11, 4, 10, 0, 0)
    assert len(identity.paths) == 2
    assert _VIDEO_TS_PATTERN.match(_csv_name(identity.name))


def test_identity_single_dav_is_logical(tmp_path) -> None:
    path = _touch_dav(tmp_path, "2026-11-04", "10.00.00-10.05.00[R].dav")

    identity = video_identity(path)

    assert identity.is_dvr
    assert identity.name == "dav_2026-11-04_10.00.00.dav"
    assert _VIDEO_TS_PATTERN.match(_csv_name(identity.name))


def test_identity_regular_video_is_unchanged(tmp_path) -> None:
    path = tmp_path / "camara.mp4"
    path.write_bytes(b"")

    identity = video_identity(str(path))

    assert not identity.is_dvr
    assert identity.name == "camara.mp4"
    assert identity.start is None


def test_fallback_for_renamed_dav_keeps_csv_pattern(tmp_path) -> None:
    path = tmp_path / "grabacion_renombrada.dav"
    path.write_bytes(b"")

    segment = DvrSegment.from_path(str(path))
    recordings = group_dvr_paths([str(path)])
    identity = video_identity(str(path))

    assert segment.end is None
    assert len(recordings) == 1
    assert identity.is_dvr
    assert _VIDEO_TS_PATTERN.match(_csv_name(identity.name))


def test_logical_name_format() -> None:
    name = dvr_logical_name(datetime(2026, 1, 2, 3, 4, 5))

    assert name == "dav_2026-01-02_03.04.05.dav"


def test_scan_directory_groups_dav_by_day(tmp_path) -> None:
    from afora_vc.adapters.storage.local import LocalStorageReader
    from afora_vc.application.facade import _scan_directory

    first = _touch_dav(tmp_path, "2026-11-04", "10.00.00-10.05.00[R].dav")
    second = _touch_dav(tmp_path, "2026-11-04", "10.05.00-10.10.00[R].dav")
    later = _touch_dav(tmp_path, "2026-11-04", "10.20.00-10.25.00[R].dav")
    (tmp_path / "2026-11-04" / "zones.json").write_text("{}", encoding="utf-8")

    entries = _scan_directory(str(tmp_path), LocalStorageReader(), recursive=True)

    assert len(entries) == 2
    assert entries[0]["paths"] == [first, second]
    assert entries[1]["paths"] == [later]
    assert entries[0]["name"] == "dav_2026-11-04_10.00.00.dav"
    assert entries[0]["rel_path"] == os.path.join("2026-11-04", entries[0]["name"])
    assert entries[0]["date"] == "2026-11-04"
    assert entries[0]["base_time"] == "10:00:00"
    assert entries[0]["output_folder"] == os.path.join(
        str(tmp_path), "output", "2026-11-04"
    )
    assert entries[0]["zones_path"].endswith("zones.json")
    assert entries[0]["path"] == first
