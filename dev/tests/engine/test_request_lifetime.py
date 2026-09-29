import gc
import threading
import time
import unittest
import weakref
from unittest import mock

from dev.tests.engine.test_runtime import FakeFactory, request, send_success
from dev.tests.test_server import FakeTokenizer
from server import backend as backend_api
from server import images
from server import protocol as wire
from server import runtime as engine_runtime
from server import server as api


class RequestLifetimeTests(unittest.TestCase):
    def setUp(self):
        enabled = gc.isenabled()
        gc.disable()
        if enabled:
            self.addCleanup(gc.enable)

    def backend(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        backend = backend_api.NativeBackend(runtime, FakeTokenizer())
        self.addCleanup(backend.close)
        return backend, factory.processes[0]

    def job(self):
        cache = images.ImageCache(request_budget_bytes=1024)
        owner = cache.request_batch()
        owner.append(images.PreparedImage(2, 2, b"pixels", 0, 0))
        job = backend_api.Job(
            request_id=1,
            prompt_tokens=[101, 102],
            max_new_tokens=16,
            seed=0,
            temperature=0,
            top_p=1,
            top_k=1,
            deadline=time.monotonic() + 30,
            image_owner=owner,
        )
        return job, cache, weakref.ref(owner)

    def terminal(self, job):
        while True:
            event = job.events.get(timeout=2)
            if event[0] in ("done", "error"):
                return event

    def assert_released(self, cache, owner):
        deadline = time.monotonic() + 1
        while owner() is not None and time.monotonic() < deadline:
            threading.Event().wait(0.001)
        self.assertIsNone(owner(), "request retained after all active holders left")
        self.assertEqual(cache.stats()["request_bytes"], 0)

    def test_native_terminals_release_images_without_cycle_collection(self):
        for outcome in (
            "complete",
            "cancel",
            "capacity",
            "callback_error",
            "runtime_callback_error",
            "shutdown",
        ):
            with self.subTest(outcome=outcome):
                backend, process = self.backend()
                job, cache, owner = self.job()
                if outcome == "callback_error":

                    class Constraint:
                        def consume(self, _tokens):
                            raise api.APIError(400, "constraint callback failed")

                    job.constraint = Constraint()
                elif outcome == "runtime_callback_error":

                    def fail_event(*_args):
                        raise RuntimeError("event callback failed")

                    backend._on_event = fail_event
                self.assertTrue(backend.submit(job))
                call = backend.active[job.request_id].call
                if outcome in ("complete", "runtime_callback_error"):
                    send_success(process, call, tokens=(4,))
                elif outcome == "capacity":
                    process.send(
                        wire.CapacityExhaustedEvent(call.request_id, 40, 12, 50000)
                    )
                elif outcome == "shutdown":
                    backend.close()
                else:
                    count = 0
                    if outcome == "callback_error":
                        process.send(
                            wire.StartEvent(
                                call.request_id, wire.CacheDisposition.MISS, 0, 0, 4096
                            )
                        )
                        process.send(wire.TokensEvent(call.request_id, 0, (4,)))
                        process.stdin.wait_for(wire.CancelFrame)
                        count = 1
                    else:
                        backend.cancel(job)
                    process.send(
                        wire.DoneEvent(
                            call.request_id,
                            wire.FinishReason.CANCELLED,
                            2,
                            count,
                            100,
                            0,
                            100,
                        )
                    )
                kind, result = self.terminal(job)
                self.assertEqual(
                    kind, "done" if outcome in ("complete", "cancel") else "error"
                )
                if outcome == "callback_error":
                    self.assertEqual(result.message, "constraint callback failed")
                    self.assertIsNone(result.__traceback__)
                elif outcome == "runtime_callback_error":
                    self.assertEqual(result.message, "event callback failed")
                    self.assertIsNone(call.callback_errors[0].__traceback__)
                del call, job
                self.assert_released(cache, owner)
                self.assertFalse(backend.active)

    def test_pre_admission_and_submit_failures_release_images(self):
        for outcome in ("expired", "closed", "full", "write_error"):
            with self.subTest(outcome=outcome):
                backend, _ = self.backend()
                job, cache, owner = self.job()
                if outcome == "expired":
                    job.deadline = time.monotonic() - 1
                    self.assertTrue(backend.submit(job))
                elif outcome == "closed":
                    backend.close()
                    self.assertTrue(backend.submit(job))
                elif outcome == "full":
                    slots = backend.runtime._admission_slots
                    for _ in range(backend.runtime.pending_limit):
                        self.assertTrue(slots.acquire(blocking=False))
                    try:
                        self.assertFalse(backend.submit(job))
                    finally:
                        for _ in range(backend.runtime.pending_limit):
                            slots.release()
                else:

                    def fail_write(*_args, **_kwargs):
                        raise ValueError("write failed")

                    with mock.patch.object(
                        backend.runtime,
                        "_write_bytes",
                        new=fail_write,
                    ):
                        self.assertTrue(backend.submit(job))
                if outcome != "full":
                    kind, error = self.terminal(job)
                    self.assertEqual(kind, "error")
                    self.assertIsNone(error.__traceback__)
                del job
                self.assert_released(cache, owner)
                self.assertFalse(backend.active)

    def test_mask_worker_keeps_image_lease_until_it_exits(self):
        backend, process = self.backend()
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        class Constraint:
            def masks(self, tokens):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test mask was not released")
                return b"\xff" * (8 * (len(tokens) + 1))

        job, cache, owner = self.job()
        job.constraint = Constraint()
        self.assertTrue(backend.submit(job))
        call = backend.active[job.request_id].call
        process.send(
            wire.StartEvent(call.request_id, wire.CacheDisposition.MISS, 0, 0, 4096)
        )
        process.send(wire.MaskRequestEvent(call.request_id, 88, 2, ()))
        self.assertTrue(entered.wait(1))
        backend.cancel(job)
        process.send(
            wire.DoneEvent(
                call.request_id, wire.FinishReason.CANCELLED, 2, 0, 100, 0, 100
            )
        )
        self.assertEqual(self.terminal(job)[0], "done")
        del call, job
        self.assertIsNotNone(owner())
        self.assertEqual(cache.stats()["request_bytes"], len(b"pixels"))
        release.set()
        self.assert_released(cache, owner)

    def test_inflight_callback_owns_images_across_terminal_cleanup(self):
        job, cache, owner = self.job()
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def callback(_call, _event, batch=job.image_owner):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test callback was not released")
            self.assertEqual(batch[0].pixels, b"pixels")

        call = engine_runtime.RuntimeCall(None, 1, 1, request(100), callback, None)
        del job, callback
        thread = threading.Thread(target=call._emit, args=(None,))
        thread.start()
        self.assertTrue(entered.wait(1))
        call._set_terminal(error=engine_runtime.RuntimeClosed("finished"))
        self.assertIsNotNone(owner())
        release.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertFalse(call.callback_errors)
        self.assert_released(cache, owner)

    def test_callback_failure_after_native_failure_does_not_retain_retired_state(self):
        backend, process = self.backend()
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        class Constraint:
            def consume(self, _tokens):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test callback was not released")
                raise api.APIError(400, "late callback failure")

        job, cache, owner = self.job()
        job.constraint = Constraint()
        self.assertTrue(backend.submit(job))
        call = backend.active[job.request_id].call
        process.send(
            wire.StartEvent(call.request_id, wire.CacheDisposition.MISS, 0, 0, 4096)
        )
        process.send(wire.TokensEvent(call.request_id, 0, (4,)))
        self.assertTrue(entered.wait(1))
        backend.runtime._fail_generation(
            call.generation, engine_runtime.EngineUnhealthy("native failed")
        )
        self.assertEqual(self.terminal(job)[0], "error")
        self.assertIsNotNone(owner())
        release.set()
        backend.runtime._reader_thread.join(1)
        self.assertFalse(backend.runtime._reader_thread.is_alive())
        del call, job
        self.assert_released(cache, owner)

    def test_callback_diagnostics_do_not_retain_request_frames(self):
        for completion in (False, True):
            with self.subTest(completion=completion):
                job, cache, owner = self.job()

                def callback(*_args):
                    try:
                        raise ValueError("inner failure")
                    except ValueError as cause:
                        raise RuntimeError("callback failed") from cause

                call = engine_runtime.RuntimeCall(
                    None,
                    1,
                    1,
                    request(100, image_owner=job.image_owner),
                    None if completion else callback,
                    callback if completion else None,
                )
                if not completion:
                    call._emit(None)
                call._set_terminal(error=engine_runtime.RuntimeClosed("finished"))
                (failure,) = call.callback_errors
                self.assertIsInstance(failure, RuntimeError)
                self.assertEqual(str(failure), "callback failed")
                self.assertIsNone(failure.__traceback__)
                self.assertIsNone(failure.__context__)
                self.assertIsNone(failure.__cause__)
                del call, job
                self.assert_released(cache, owner)


if __name__ == "__main__":
    unittest.main()
