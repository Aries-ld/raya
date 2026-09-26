import asyncio
import threading

import pytest

from raya_maas.errors import APIError
from raya_maas.worker import InferenceWorker


async def test_timeout_preserves_single_model_owner_and_discards_expired_queue():
    entered, release = threading.Event(), threading.Event()

    class Engine:
        calls = 0

        def predict(self, request):
            self.calls += 1
            entered.set()
            release.wait(2)
            return {}

    engine = Engine()
    worker = InferenceWorker(engine, queue_size=1, timeout=0.05)
    await worker.start()
    first = asyncio.create_task(worker.submit("first"))
    await asyncio.to_thread(entered.wait, 1)
    second = asyncio.create_task(worker.submit("second"))
    await asyncio.sleep(0)
    with pytest.raises(APIError) as full:
        await worker.submit("third")
    assert full.value.status == 429
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(x, APIError) and x.status == 504 for x in results)
    assert worker.active
    assert engine.calls == 1
    release.set()
    await worker.close()
    assert engine.calls == 1


async def test_worker_recovers_after_bad_input():
    class Engine:
        def predict(self, request):
            if request == "bad":
                raise APIError("bad media")
            return {"label": "A"}

    worker = InferenceWorker(Engine(), 2, 1)
    await worker.start()
    with pytest.raises(APIError, match="bad media"):
        await worker.submit("bad")
    assert (await worker.submit("good"))["label"] == "A"
    await worker.close()


async def test_request_timings_are_snapshotted_before_next_job():
    class Engine:
        def predict(self, value):
            self.last_diagnostics = {"questions": {"gesture": {"timing_ms": {"forward": value}}}}
            return {"value": value}

    worker = InferenceWorker(Engine(), 2, 1)
    await worker.start()
    first, second = await asyncio.gather(
        worker.submit(12, with_metrics=True), worker.submit(34, with_metrics=True)
    )
    assert first[0]["value"] == 12 and first[1]["forward_ms"] == 12
    assert second[0]["value"] == 34 and second[1]["forward_ms"] == 34
    assert first[1]["processing_ms"] >= 0 and first[1]["queue_ms"] >= 0
    await worker.close()
