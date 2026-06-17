from typing import Protocol, runtime_checkable, Any

from dataclasses import dataclass


@dataclass
class FileInfo:
    path: str
    name: str
    extension: str
    size_bytes: int | None = None


@runtime_checkable
class StorageReader(Protocol):
    def list_files(self, directory: str, pattern: str = "*") -> list[FileInfo]:
        ...

    def read_file(self, path: str) -> bytes:
        ...

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        ...

    def read_json(self, path: str) -> dict[str, Any]:
        ...

    def exists(self, path: str) -> bool:
        ...

    def is_file(self, path: str) -> bool:
        ...

    def get_video_path(self, path: str) -> str:
        ...


@runtime_checkable
class StorageWriter(Protocol):
    def write_file(self, path: str, content: bytes) -> None:
        ...

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        ...

    def write_json(self, path: str, data: dict[str, Any], indent: int = 2) -> None:
        ...

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        ...

    def get_video_writer_path(self, path: str) -> str:
        ...

    def finalize_video(self, temp_path: str, final_path: str) -> None:
        ...

    def copy_file(self, source: str, destination: str) -> None:
        ...

    def delete_file(self, path: str) -> None:
        ...
