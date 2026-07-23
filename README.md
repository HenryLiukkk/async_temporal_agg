# 🦾 Async Temporal Aggregation

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?logo=numpy&logoColor=white)
![Tests](https://img.shields.io/badge/tests-4%20passed-brightgreen)

> 🐣 A tiny helper that lets a robot keep moving while its policy is still
> thinking.

This project runs **chunked policy inference** in a background thread while the
main thread keeps sending robot commands at a steady control rate.

It is useful for policies such as **ACT**, where one model forward pass predicts
a whole chunk of future actions.

<br/>

## 💡 Why does this exist?

Imagine that your robot should receive a command every `20 ms`:

```text
50 Hz control loop = one command every 20 ms
```

But your policy needs `25 ms` to produce a new action chunk.

With normal synchronous inference, the robot has to wait:

```text
observe → wait for policy → command robot → observe → wait again...
```

That waiting can make motion slow or jerky.

This project separates the work:

```text
🦾 Robot thread
observe → blend ready actions → command robot → sleep
   │
   └── sends the newest observation when the worker is free

🧠 Policy thread
preprocess → model inference → full action chunk → result queue
```

The robot thread does not wait for routine policy inference.

<br/>

## ✨ What does it do?

- 🧵 Runs policy inference on a background thread.
- 📸 Uses the newest observation instead of building a stale image queue.
- ⏱️ Remembers when each observation was captured.
- 🧩 Aligns overlapping action chunks to the current control step.
- 🫧 Blends matching actions with temporal aggregation.
- 🚨 Sends background exceptions back to the control thread.
- 🧰 Works with PyTorch, ONNX Runtime, TensorFlow, or custom inference code.

The core module only depends on NumPy.

<br/>

## 🚀 Quick Start

### 1. Install

```bash
git clone <your-github-repository-url>
cd async_temporal_agg

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Run the tests

```bash
python -m unittest -v test_async_temporal_agg.py
```

Expected result:

```text
Ran 4 tests

OK
```

### 3. Run the cute little robot simulation 🤖

```bash
python demo.py
```

The demo simulates:

- a `50 Hz` robot-control loop;
- a `25 ms` policy forward pass;
- chunks containing `100` future actions.

Example output:

```text
step=000 contributors=01 action[0]=0.0000
step=020 contributors=10 action[0]=0.0140
step=040 contributors=20 action[0]=0.0287
...
completed 200 control steps in 4.04s
```

<br/>

## 🧠 What is an action chunk?

A normal policy may predict one action:

```text
[joint_1, joint_2, ..., gripper]
```

A chunked policy predicts many future actions in one forward pass:

```text
chunk[0]   → action for the first future step
chunk[1]   → action for the second future step
...
chunk[99]  → action for the 100th future step
```

For a seven-dimensional robot action and a chunk size of 100:

```python
chunk.shape == (100, 7)
```

> 🌟 One chunk does **not** mean 100 separate model calls.  
> One model forward pass returns all 100 actions together.

<br/>

## ⏰ How are delayed chunks aligned?

Suppose an observation is captured at control step `20`.

The background worker finishes its chunk at step `22`. The controller should
not restart that delayed chunk from index zero. It uses:

```python
offset = current_step - observation_step
action = chunk[offset]
```

In this example:

```python
action = chunk[22 - 20]
action = chunk[2]
```

This small timestamp trick keeps delayed predictions on the correct timeline.

<br/>

## 🫧 How does temporal aggregation work?

Several chunks can predict an action for the same current step:

```text
chunk A:  A[current_step - A.query_step]
chunk B:  B[current_step - B.query_step]
chunk C:  C[current_step - C.query_step]
```

The project selects those matching actions and computes a weighted average:

```python
action, history, contributors = aggregate_action_chunks(
    history,
    step=current_step,
    decay=0.01,
)
```

The result is usually smoother than switching from one chunk to the next.

### A small weighting note 🍪

This copied implementation preserves the ACT++ ordering:

```python
weight[i] = exp(-decay * i)
```

Chunks are ordered from earlier to later query steps, so earlier chunks receive
more weight.

This is smooth, but a very long history can make stopping feel delayed. For
faster stopping, consider:

- keeping only recent chunks;
- using freshness-based weights;
- giving newer chunks more weight.

<br/>

## 🛠️ Minimal Integration

First, write a function that receives one observation and returns one complete
action chunk:

```python
def infer_action_chunk(observation):
    # Run preprocessing and your model here.
    return chunk  # shape: [chunk_size, action_dim]
```

Generate the first chunk before robot motion:

```python
first_observation = observe()
first_chunk = infer_action_chunk(first_observation)

history = [(0, first_chunk)]
last_action = first_chunk[0]
```

Start the worker:

```python
from async_temporal_agg import LatestObservationWorker

worker = LatestObservationWorker(infer_action_chunk)
worker.start()
```

Then use it inside the control loop:

```python
import time

from async_temporal_agg import aggregate_action_chunks

control_hz = 50
period = 1 / control_hz

try:
    for step in range(max_steps):
        tick_started = time.perf_counter()

        # Collect results that are already finished. This never waits.
        for result in worker.poll():
            history.append((result.step, result.value))

        # Submit only when the worker is idle.
        if step > 0:
            worker.submit(
                step,
                lambda: copy_latest_observation(),
            )

        # Pick and blend the actions that belong to this step.
        action, history, contributors = aggregate_action_chunks(
            history,
            step=step,
            decay=0.01,
        )

        # Choose a safe fallback for your robot.
        if action is None:
            action = last_action
        else:
            last_action = action

        command_robot_nonblocking(action)

        remaining = period - (time.perf_counter() - tick_started)
        time.sleep(max(0, remaining))
finally:
    worker.close()
```

<br/>

## 🔥 PyTorch + ACT Example

Put GPU work inside the worker's inference function:

```python
def infer_action_chunk(observation):
    with torch.inference_mode():
        qpos = preprocess_qpos(observation["qpos"]).to("cuda")
        images = preprocess_images(observation["images"]).to("cuda")

        chunk = policy(qpos, images)

        return (
            chunk.squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
```

The control thread only receives the completed CPU chunk. It does not call the
policy directly after initialization.

<br/>

## 📁 Project Structure

```text
async_temporal_agg/
├── async_temporal_agg.py       # Worker + aggregation logic
├── demo.py                     # Runnable simulated controller
├── test_async_temporal_agg.py  # Four unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

<br/>

## 📚 Tiny API Guide

### `LatestObservationWorker`

```python
worker = LatestObservationWorker(infer_fn)
```

| Method | What it does |
|---|---|
| `start()` | Starts the background thread |
| `submit(step, factory)` | Submits an observation if the worker is idle |
| `poll()` | Returns finished results without waiting |
| `wait(timeout)` | Waits for one result, mainly for setup or tests |
| `close()` | Stops and joins the worker |
| `ready` | Tells whether a new request can be accepted |

### `AsyncInferenceResult`

Each completed result contains:

| Field | Meaning |
|---|---|
| `step` | Step at which the observation was captured |
| `value` | The predicted action chunk |
| `elapsed` | Inference time in seconds |
| `error` | Background exception, if one occurred |
| `error_traceback` | Full background traceback |

### `aggregate_action_chunks`

```python
action, active_history, contributor_count = aggregate_action_chunks(
    action_history,
    step=current_step,
    decay=0.01,
)
```

It removes expired chunks, selects timestamp-aligned actions, and returns their
weighted average.

<br/>

## ❓ Friendly FAQ

### Does the worker still predict the full chunk?

Yes. One background model call still returns the complete chunk, such as
`(100, 7)`.

### Does the robot wait until all 100 chunk actions are executed?

No. While the robot executes the current aggregated action, the worker can
already predict another full chunk from a newer observation.

### Is this implemented with `asyncio`?

No. It uses a Python `threading.Thread`, a request queue, a result queue, and a
small readiness event.

### Is temporal aggregation also calculated in the worker?

No. Model inference runs in the worker. Temporal aggregation stays in the
control thread because it is tiny and must use the exact current control step.

### What happens when inference is busy?

The new request is skipped. The controller keeps using valid actions from
existing chunks. Old camera frames do not pile up.

<br/>

## 🦺 Safety First

This project schedules inference. It is **not** a complete robot safety system.

Before using it on hardware, add:

- joint and workspace limits;
- velocity and acceleration limits;
- rejected-command handling;
- a maximum accepted chunk age;
- emergency-stop monitoring;
- a safe fallback for missing actions;
- reduced-speed testing.

> ⚠️ Please test with a simulator or an observation-only mode before enabling
> robot motion.

<br/>

## 🌱 Known Limitations

- Only one inference request runs at a time.
- Busy workers skip new requests instead of queueing them.
- An inference already running cannot be interrupted by `close()`.
- Temporal aggregation runs in the control thread by design.
- The default weighting order favors earlier chunks.

<br/>

## ❤️ Acknowledgment

This standalone implementation was extracted from a real-robot ACT++ inference
workflow. It is inspired by the temporal-aggregation idea used in ACT-style
chunked robot policies.

Happy robot building! 🦾✨
