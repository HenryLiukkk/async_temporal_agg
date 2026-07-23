"""Asynchronous policy inference helpers for real-time ACT evaluation."""

from dataclasses import dataclass
import queue
import threading
import time
import traceback

import numpy as np


@dataclass
class AsyncInferenceResult:
    step: int
    value: object = None
    elapsed: float = 0.0
    error: BaseException = None
    error_traceback: str = ''


class AsyncInferenceError(RuntimeError):
    pass


class LatestObservationWorker:
    """Run one inference at a time without ever queueing stale observations."""

    _STOP = object()

    def __init__(self, infer_fn, name='act-policy-worker'):
        self._infer_fn = infer_fn
        self._requests = queue.Queue(maxsize=1)
        self._results = queue.Queue()
        self._ready = threading.Event()
        self._ready.set()
        self._thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=True,
        )
        self._started = False

    @property
    def ready(self):
        return self._ready.is_set()

    def start(self):
        if self._started:
            raise RuntimeError('Async inference worker has already been started')
        self._started = True
        self._thread.start()

    def submit(self, step, payload_factory):
        """Submit only when idle; payload_factory is skipped while inference is busy."""
        if not self._started:
            raise RuntimeError('Async inference worker has not been started')
        if not self._ready.is_set():
            return False

        self._ready.clear()
        try:
            payload = payload_factory()
            self._requests.put_nowait((int(step), payload))
        except BaseException:
            self._ready.set()
            raise
        return True

    def poll(self):
        results = []
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            self._raise_if_failed(result)
            results.append(result)
        return results

    def wait(self, timeout):
        try:
            result = self._results.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                f'Async policy inference did not finish within {timeout:.1f} seconds'
            ) from exc
        self._raise_if_failed(result)
        return result

    def close(self, timeout=5.0):
        if not self._started:
            return
        try:
            self._requests.put_nowait(self._STOP)
        except queue.Full:
            # At most one request can be pending. Drop it so shutdown is prompt.
            try:
                self._requests.get_nowait()
            except queue.Empty:
                pass
            self._requests.put_nowait(self._STOP)
        self._thread.join(timeout=timeout)

    def _run(self):
        while True:
            request = self._requests.get()
            if request is self._STOP:
                return

            step, payload = request
            started_at = time.perf_counter()
            try:
                value = self._infer_fn(payload)
                result = AsyncInferenceResult(
                    step=step,
                    value=value,
                    elapsed=time.perf_counter() - started_at,
                )
            except BaseException as exc:
                result = AsyncInferenceResult(
                    step=step,
                    elapsed=time.perf_counter() - started_at,
                    error=exc,
                    error_traceback=traceback.format_exc(),
                )
            self._results.put(result)
            self._ready.set()

    @staticmethod
    def _raise_if_failed(result):
        if result.error is None:
            return
        raise AsyncInferenceError(
            'Background ACT inference failed:\n' + result.error_traceback
        ) from result.error


def aggregate_action_chunks(action_history, step, decay):
    """Blend overlapping chunks using the same weighting order as ACT++."""
    contributors = []
    active_history = []

    for query_step, chunk in sorted(action_history, key=lambda item: item[0]):
        chunk = np.asarray(chunk, dtype=np.float32)
        offset = int(step) - int(query_step)
        if offset < 0:
            active_history.append((query_step, chunk))
            continue
        if offset < len(chunk):
            active_history.append((query_step, chunk))
            contributors.append(chunk[offset])

    if not contributors:
        return None, active_history, 0

    weights = np.exp(-float(decay) * np.arange(len(contributors), dtype=np.float64))
    weights /= weights.sum()
    stacked = np.stack(contributors, axis=0)
    action = np.sum(stacked * weights[:, None], axis=0, dtype=np.float64)
    return action.astype(np.float32), active_history, len(contributors)
