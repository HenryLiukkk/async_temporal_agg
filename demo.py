"""Runnable simulation of asynchronous chunk inference and control."""

import time

import numpy as np

from async_temporal_agg import LatestObservationWorker, aggregate_action_chunks


CONTROL_HZ = 50
CONTROL_STEPS = 200
CHUNK_SIZE = 100
ACTION_DIM = 7
TEMPORAL_DECAY = 0.01


def observe(step):
    """Return a small stand-in for robot state and camera observations."""
    return {
        'step': step,
        'qpos': np.full(ACTION_DIM, step / 1000, dtype=np.float32),
    }


def infer_action_chunk(observation):
    """Simulate one model forward pass that predicts a full action chunk."""
    time.sleep(0.025)
    horizon = np.arange(CHUNK_SIZE, dtype=np.float32)[:, None]
    base = observation['qpos'][None, :]
    return base + 0.0005 * horizon


def main():
    period = 1 / CONTROL_HZ

    initial_observation = observe(0)
    initial_chunk = infer_action_chunk(initial_observation)
    history = [(0, initial_chunk)]
    last_action = initial_chunk[0]

    worker = LatestObservationWorker(infer_action_chunk)
    worker.start()

    started_at = time.perf_counter()
    submitted = 0
    completed = 0
    try:
        for step in range(CONTROL_STEPS):
            tick_started = time.perf_counter()

            for result in worker.poll():
                history.append((result.step, result.value))
                completed += 1

            if step > 0 and worker.submit(
                step,
                lambda current_step=step: observe(current_step),
            ):
                submitted += 1

            action, history, contributors = aggregate_action_chunks(
                history,
                step=step,
                decay=TEMPORAL_DECAY,
            )
            if action is None:
                action = last_action.copy()
            else:
                last_action = action.copy()

            # Replace this line with a non-blocking robot command in production.
            if step % 20 == 0:
                print(
                    f'step={step:03d} contributors={contributors:02d} '
                    f'action[0]={action[0]:.4f}'
                )

            remaining = period - (time.perf_counter() - tick_started)
            time.sleep(max(0, remaining))
    finally:
        worker.close()

    elapsed = time.perf_counter() - started_at
    print(
        f'completed {CONTROL_STEPS} control steps in {elapsed:.2f}s; '
        f'submitted={submitted}, completed={completed}'
    )


if __name__ == '__main__':
    main()
