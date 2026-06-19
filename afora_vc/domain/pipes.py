from typing import Protocol, TypeVar, Generic, runtime_checkable

T = TypeVar("T")
U = TypeVar("U")


@runtime_checkable
class PipeMapper(Protocol, Generic[T, U]):
    def map(self, data: T) -> U:
        ...


class IdentityMapper(Generic[T]):
    def map(self, data: T) -> T:
        return data


@runtime_checkable
class DataPipe(Protocol, Generic[T]):
    def send(self, item: T) -> None:
        ...

    def receive(self) -> T | None:
        ...

    def close(self) -> None:
        ...
