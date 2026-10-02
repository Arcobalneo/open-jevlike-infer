import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from jevlike_infer.batching import MicroBatcher


def test_concurrent_items_share_a_batch():
    sizes = []
    gate = threading.Event()

    def run(items):
        gate.wait(1)  # hold the first batch so the rest queue up behind it
        sizes.append(len(items))
        return [item * 2 for item in items]

    batcher = MicroBatcher(run, max_batch=8, window=0.05)
    with ThreadPoolExecutor(16) as pool:
        futures = [pool.submit(batcher.submit, i) for i in range(16)]
        time.sleep(0.2)
        gate.set()
        assert [f.result() for f in futures] == [i * 2 for i in range(16)]
    assert max(sizes) > 1 and max(sizes) <= 8 and sum(sizes) == 16


def test_exception_fails_only_its_item():
    batcher = MicroBatcher(lambda items: [ValueError("bad") if i < 0 else i for i in items], window=0.05)
    with ThreadPoolExecutor(4) as pool:
        good, bad = pool.submit(batcher.submit, 1), pool.submit(batcher.submit, -1)
        assert good.result() == 1
        with pytest.raises(ValueError, match="bad"):
            bad.result()


def test_batch_crash_fails_all_items_and_loop_survives():
    calls = []

    def run(items):
        calls.append(items)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return items

    batcher = MicroBatcher(run)
    with pytest.raises(RuntimeError, match="boom"):
        batcher.submit(1)
    assert batcher.submit(2) == 2


def test_wrong_result_count_is_an_error():
    batcher = MicroBatcher(lambda items: [])
    with pytest.raises(RuntimeError, match="0 results for 1"):
        batcher.submit(1)


def test_invalid_max_batch():
    with pytest.raises(ValueError):
        MicroBatcher(lambda items: items, max_batch=0)
