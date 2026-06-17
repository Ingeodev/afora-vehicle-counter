import json
import os
import shutil
from pathlib import Path
from typing import Any

from ...domain.ports.storage import FileInfo


class LocalStorageReader:
    def __init__(self, base_path: str | None = None):
        self.base_path = Path(base_path) if base_path else None

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if self.base_path and not p.is_absolute():
            return self.base_path / p
        return p

    def list_files(self, directory: str, pattern: str = "*") -> list[FileInfo]:
        dir_path = self._resolve_path(directory)
        if not dir_path.exists():
            return []
        files: list[FileInfo] = []
        for fp in dir_path.glob(pattern):
            if fp.is_file():
                files.append(
                    FileInfo(
                        path=str(fp),
                        name=fp.name,
                        extension=fp.suffix.lower(),
                        size_bytes=fp.stat().st_size,
                    )
                )
        return files

    def read_file(self, path: str) -> bytes:
        return self._resolve_path(path).read_bytes()

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self._resolve_path(path).read_text(encoding=encoding)

    def read_json(self, path: str) -> dict[str, Any]:
        content = self.read_text(path)
        return json.loads(content)

    def exists(self, path: str) -> bool:
        return self._resolve_path(path).exists()

    def is_file(self, path: str) -> bool:
        return self._resolve_path(path).is_file()

    def get_video_path(self, path: str) -> str:
        return str(self._resolve_path(path))


class LocalStorageWriter:
    def __init__(self, base_path: str | None = None):
        self.base_path = Path(base_path) if base_path else None

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if self.base_path and not p.is_absolute():
            return self.base_path / p
        return p

    def write_file(self, path: str, content: bytes) -> None:
        fp = self._resolve_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(content)

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        fp = self._resolve_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding=encoding)

    def write_json(self, path: str, data: dict[str, Any], indent: int = 2) -> None:
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        self.write_text(path, content)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        dir_path = self._resolve_path(path)
        dir_path.mkdir(parents=True, exist_ok=exist_ok)

    def get_video_writer_path(self, path: str) -> str:
        fp = self._resolve_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        return str(fp)

    def finalize_video(self, temp_path: str, final_path: str) -> None:
        if temp_path != final_path:
            src = self._resolve_path(temp_path)
            dst = self._resolve_path(final_path)
            shutil.move(str(src), str(dst))

    def copy_file(self, source: str, destination: str) -> None:
        src = self._resolve_path(source)
        dst = self._resolve_path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    def delete_file(self, path: str) -> None:
        fp = self._resolve_path(path)
        if fp.exists():
            fp.unlink()
