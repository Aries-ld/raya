import asyncio
import logging
import time

from .errors import APIError

logger = logging.getLogger(__name__)


class InferenceWorker:
    """One model owner. A timed-out request never releases a still-running GPU job."""

    def __init__(self, engine, queue_size: int, timeout: float):
        self.engine = engine
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.timeout = timeout
        self.active = False
        self.closed = False
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self._run())

    async def submit(self, request):
        if self.closed:
            raise APIError("Service shutting down", 503, "service_unavailable")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((request, future, time.monotonic() + self.timeout))
        except asyncio.QueueFull as exc:
            raise APIError("Inference queue full; retry later", 429, "rate_limit_exceeded") from exc
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.timeout)
        except TimeoutError as exc:
            future.cancel()
            raise APIError("Inference deadline exceeded", 504, "request_timeout") from exc
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _run(self):
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                request, future, deadline = item
                if future.cancelled() or time.monotonic() >= deadline:
                    if not future.done():
                        future.set_exception(
                            APIError("Request expired in queue", 504, "request_timeout")
                        )
                    continue
                self.active = True
                try:
                    result = await asyncio.to_thread(self.engine.predict, request)
                    if not future.done():
                        future.set_result(result)
                except Exception as exc:
                    if not isinstance(exc, APIError):
                        # No request body, media URLs or credentials in logs/errors.
                        logger.error("Inference failed (%s)", type(exc).__name__)
                        exc = APIError("Inference failed; see service logs", 500, "inference_error")
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    self.active = False
            finally:
                self.queue.task_done()

    async def close(self):
        self.closed = True
        # Drain/reject queued work, then await the in-flight job before freeing the model.
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if item is not None and not item[1].done():
                item[1].set_exception(APIError("Service shutting down", 503, "service_unavailable"))
            self.queue.task_done()
        await self.queue.put(None)
        await self.task
