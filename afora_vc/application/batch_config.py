from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from ..domain.counting.entities import CountConfig, CountGroupConfig


@dataclass
class VideoOverride:
    path: str
    mask: str | None = None
    zones: str | None = None
    reduce: int | None = None
    batch_size: int | None = None
    pipeline_mode: str | None = None
    model: str | None = None
    backend: str | None = None
    tracker: str | None = None
    device: str | None = None
    strategy: str | None = None
    max_minutes: float | None = None
    night_enhance: bool | None = None
    deblurring: bool | None = None
    no_video: bool | None = None
    output_name: str | None = None


@dataclass
class BatchConfig:
    input: str = ""
    output: str = "./output"
    recursive: bool = True
    mask: str | None = None
    zones: str | None = None

    backend: str = "yolo"
    model: str = "models/yolov8s.pt"
    pipeline_mode: str = "sequential"
    batch_size: int = 1
    reduce: int = 1
    device: str | None = None
    strategy: str = "box"
    tracker: str = "botsort"
    max_minutes: float | None = None
    night_enhance: bool = False
    deblurring: bool = False
    no_video: bool = False
    output_name: str | None = None
    quiet: bool = False

    overrides: list[VideoOverride] = field(default_factory=list)
    count: CountConfig | None = None

    @classmethod
    def from_yaml(cls, path: str) -> BatchConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchConfig:
        ov_data = data.pop("overrides", [])
        overrides = []
        for item in ov_data:
            if isinstance(item, str):
                overrides.append(VideoOverride(path=item))
            elif isinstance(item, dict):
                overrides.append(VideoOverride(**item))

        count_data = data.pop("count", None)
        count_config: CountConfig | None = None
        if count_data is not None:
            count_config = cls._parse_count(count_data)

        return cls(overrides=overrides, count=count_config, **data)

    @staticmethod
    def _parse_count(data: dict[str, Any]) -> CountConfig:
        interval = int(data.get("interval", 15))
        raw_groups = data.get("groups", [])
        groups = []
        for g in raw_groups:
            include: dict[int, str] | Literal["all"] = g.get("include", "all")
            exclude: dict[int, str] | Literal["none"] = g.get("exclude", "none")
            if isinstance(include, dict):
                include = {int(k): str(v) for k, v in include.items()}
            if isinstance(exclude, dict):
                exclude = {int(k): str(v) for k, v in exclude.items()}
            groups.append(CountGroupConfig(
                name=str(g.get("name", "")),
                include=include,
                exclude=exclude,
                access=str(g.get("acess", "")),
                zone_paths=[[str(z) for z in zp] for zp in g.get("zones", [])],
            ))
        graph = str(data.get("graph", "none")).lower()
        return CountConfig(interval_minutes=interval, groups=groups, graph=graph)

    def get_override(self, video_rel_path: str) -> VideoOverride | None:
        norm = video_rel_path.replace("\\", "/")
        for ov in self.overrides:
            ov_norm = ov.path.replace("\\", "/")
            if ov_norm == norm:
                return ov
        return None
