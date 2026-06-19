from queue import Queue
from threading import Event
from typing import Generic, TypeVar

from ...domain.pipes import DataPipe, PipeMapper, IdentityMapper

T = TypeVar("T")


class QueuePipe(Generic[T]):
    """Pipe concreto basado en queue.Queue con mappers opcionales y centinela None."""

    def __init__(
        self,
        maxsize: int = 0,
        input_mapper: PipeMapper | None = None,
        output_mapper: PipeMapper | None = None,
    ):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._input_mapper = input_mapper or IdentityMapper()
        self._output_mapper = output_mapper or IdentityMapper()
        self._closed = Event()

    def send(self, item: T) -> None:
        if self._closed.is_set():
            raise RuntimeError("Pipe closed")
        mapped = self._input_mapper.map(item)
        self._queue.put(mapped)

    def receive(self) -> T | None:
        item = self._queue.get()
        if item is None:
            return None
        return self._output_mapper.map(item)

    def close(self) -> None:
        self._closed.set()
        self._queue.put(None)
