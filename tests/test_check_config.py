from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from afora_vc.application.batch_config import BatchConfig
from afora_vc.application.facade import check_config, process_from_config


def _make_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake video")


def _make_zones(path: Path, label: str = "A") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "zones": [
            {
                "label": label,
                "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            }
        ],
    }
    with open(path, "w") as f:
        yaml.dump(data, f)


def test_check_config_overrides(tmp_path: Path) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    global_zones = tmp_path / "global_zones.json"
    override_zones = tmp_path / "override_zones.json"
    model = tmp_path / "model.pt"
    config_path = tmp_path / "config.yaml"

    _make_zones(global_zones, "A")
    _make_zones(override_zones, "B")
    model.write_text("fake model")

    videos = {
        "cam1_2026-07-10_08.00.00.mp4": None,
        "sub/cam2_2026-07-10_09.00.00.mp4": None,
        "cam3_2026-07-10_10.00.00.mp4": None,
    }
    for name in videos:
        _make_video(input_dir / name)

    config = {
        "input": str(input_dir),
        "output": str(output_dir),
        "recursive": True,
        "backend": "yolo",
        "model": str(model),
        "zones": str(global_zones),
        "overrides": [
            {
                "path": os.path.abspath(str(input_dir / "cam1_2026-07-10_08.00.00.mp4")),
                "zones": str(override_zones),
            },
            {
                "path": os.path.abspath(str(input_dir / "sub/cam2_2026-07-10_09.00.00.mp4")),
                "zones": str(override_zones),
            },
        ],
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    batch = BatchConfig.from_yaml(str(config_path))
    input_abs = os.path.abspath(batch.input)
    overrides_dict: dict[str, dict[str, Any]] = {}
    for ov in batch.overrides:
        d = {k: v for k, v in vars(ov).items() if k != "path" and v is not None}
        ov_path = ov.path.replace("\\", "/")
        if os.path.isabs(ov.path):
            ov_path = os.path.relpath(ov.path, input_abs).replace("\\", "/")
        overrides_dict[ov_path] = d

    assert "cam1_2026-07-10_08.00.00.mp4" in overrides_dict
    assert overrides_dict["cam1_2026-07-10_08.00.00.mp4"]["zones"] == str(override_zones)

    assert "sub/cam2_2026-07-10_09.00.00.mp4" in overrides_dict
    assert overrides_dict["sub/cam2_2026-07-10_09.00.00.mp4"]["zones"] == str(override_zones)

    all_rel = {"cam1_2026-07-10_08.00.00.mp4", "sub/cam2_2026-07-10_09.00.00.mp4", "cam3_2026-07-10_10.00.00.mp4"}
    assert set(overrides_dict.keys()).issubset(all_rel)


def test_check_config_runs(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    input_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    global_zones = tmp_path / "global_zones.json"
    model = tmp_path / "model.pt"
    config_path = tmp_path / "config.yaml"

    _make_zones(global_zones)
    model.write_text("fake model")

    _make_video(input_dir / "test_2026-07-10_08.00.00.mp4")
    _make_video(input_dir / "sub/test2_2026-07-10_09.00.00.mp4")

    config = {
        "input": str(input_dir),
        "output": str(output_dir),
        "recursive": True,
        "backend": "yolo",
        "model": str(model),
        "zones": str(global_zones),
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    check_config(str(config_path))

    captured = capsys.readouterr()
    assert "Video 2/2" in captured.out
    assert "global" in captured.out
    assert "override" not in captured.out
