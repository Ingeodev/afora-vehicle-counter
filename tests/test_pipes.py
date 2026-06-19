"""Tests para domain/pipes.py y adapters/pipes/queue_pipe.py."""

import time
from threading import Thread

import pytest

from afora_vc.domain.pipes import IdentityMapper, DataPipe, PipeMapper
from afora_vc.adapters.pipes import QueuePipe


class TestIdentityMapper:
    def test_returns_same(self):
        mapper = IdentityMapper()
        assert mapper.map(42) == 42
        assert mapper.map("hello") == "hello"
        assert mapper.map([1, 2, 3]) == [1, 2, 3]
        data = {"key": "value"}
        assert mapper.map(data) is data


class TestQueuePipeBasic:
    def test_send_receive(self):
        pipe = QueuePipe[int](maxsize=4)
        pipe.send(1)
        pipe.send(2)
        assert pipe.receive() == 1
        assert pipe.receive() == 2

    def test_send_receive_order_fifo(self):
        pipe = QueuePipe[str](maxsize=4)
        for item in ["a", "b", "c"]:
            pipe.send(item)
        for expected in ["a", "b", "c"]:
            assert pipe.receive() == expected

    def test_close_sends_none_sentinel(self):
        pipe = QueuePipe[int](maxsize=4)
        pipe.send(99)
        pipe.close()
        assert pipe.receive() == 99
        assert pipe.receive() is None

    def test_receive_blocks_until_send(self):
        pipe = QueuePipe[int](maxsize=2)
        results = []

        def _sender():
            time.sleep(0.05)
            pipe.send(42)
            pipe.close()

        t = Thread(target=_sender, daemon=True)
        t.start()
        val = pipe.receive()
        assert val == 42
        assert pipe.receive() is None

    def test_close_then_close_no_error(self):
        pipe = QueuePipe[int](maxsize=4)
        pipe.send(1)
        pipe.close()
        pipe.close()
        pipe.close()
        assert pipe.receive() == 1
        assert pipe.receive() is None


class TestQueuePipeMappers:
    def test_input_mapper(self):
        class DoubleMapper:
            def map(self, x):
                return x * 2
        pipe = QueuePipe[int](maxsize=4, input_mapper=DoubleMapper())
        pipe.send(5)
        assert pipe.receive() == 10

    def test_output_mapper(self):
        class StrMapper:
            def map(self, x):
                return f"num_{x}"
        pipe = QueuePipe[int](maxsize=4, output_mapper=StrMapper())
        pipe.send(42)
        assert pipe.receive() == "num_42"

    def test_both_mappers_chain_input_then_output(self):
        class DoubleMapper:
            def map(self, x): return x * 2
        class StrMapper:
            def map(self, x): return f"_{x}_"
        pipe = QueuePipe[int](maxsize=4, input_mapper=DoubleMapper(), output_mapper=StrMapper())
        pipe.send(3)
        assert pipe.receive() == "_6_"


class TestQueuePipeClose:
    def test_send_after_close_raises(self):
        pipe = QueuePipe[int](maxsize=4)
        pipe.close()
        with pytest.raises(RuntimeError, match="Pipe closed"):
            pipe.send(1)

    def test_receive_after_close_returns_none(self):
        pipe = QueuePipe[int](maxsize=4)
        pipe.close()
        assert pipe.receive() is None


class TestDataPipeProtocol:
    def test_queuepipe_is_datapipe(self):
        pipe = QueuePipe[int](maxsize=4)
        assert isinstance(pipe, DataPipe)

    def test_identitymapper_is_pipemapper(self):
        assert isinstance(IdentityMapper(), PipeMapper)
