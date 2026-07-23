import threading
import unittest

import numpy as np

from async_temporal_agg import (
    AsyncInferenceError,
    LatestObservationWorker,
    aggregate_action_chunks,
)


class AggregateActionChunksTest(unittest.TestCase):
    def test_blends_overlapping_chunks_with_original_act_weight_order(self):
        old_chunk = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        new_chunk = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)

        action, history, count = aggregate_action_chunks(
            [(0, old_chunk), (1, new_chunk)],
            step=1,
            decay=0.01,
        )

        weights = np.exp(-0.01 * np.arange(2))
        weights /= weights.sum()
        expected = 2.0 * weights[0] + 10.0 * weights[1]
        self.assertAlmostEqual(float(action[0]), expected, places=5)
        self.assertEqual(count, 2)
        self.assertEqual(len(history), 2)

    def test_removes_expired_chunks(self):
        chunk = np.zeros((2, 7), dtype=np.float32)
        action, history, count = aggregate_action_chunks(
            [(0, chunk)],
            step=2,
            decay=0.01,
        )
        self.assertIsNone(action)
        self.assertEqual(history, [])
        self.assertEqual(count, 0)


class LatestObservationWorkerTest(unittest.TestCase):
    def test_does_not_build_payload_while_worker_is_busy(self):
        release = threading.Event()

        def infer_fn(value):
            release.wait(timeout=2)
            return value * 2

        worker = LatestObservationWorker(infer_fn)
        worker.start()
        self.assertTrue(worker.submit(3, lambda: 5))

        payload_built = False

        def build_payload():
            nonlocal payload_built
            payload_built = True
            return 8

        self.assertFalse(worker.submit(4, build_payload))
        self.assertFalse(payload_built)

        release.set()
        result = worker.wait(timeout=2)
        self.assertEqual(result.step, 3)
        self.assertEqual(result.value, 10)
        worker.close()

    def test_propagates_background_error(self):
        def infer_fn(_):
            raise ValueError('bad inference')

        worker = LatestObservationWorker(infer_fn)
        worker.start()
        self.assertTrue(worker.submit(0, lambda: None))
        with self.assertRaisesRegex(AsyncInferenceError, 'bad inference'):
            worker.wait(timeout=2)
        worker.close()


if __name__ == '__main__':
    unittest.main()
