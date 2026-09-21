"""Unit tests for `ActionExecutor`, decoupled from any policy via a fake `plan_fn` -- proves
execution strategy (queueing, temporal ensembling, completion-gated replanning) works
independently of which model generated the action chunk."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lerobot.policies.safediff_vla.execution import ActionExecutor, detect_gripper_close_event

ACTION_DIM = 3
HORIZON = 4
GRIPPER_IX = 2  # last channel, matching FakeExecutorConfig.gripper_action_index below


@dataclass
class FakeExecutorConfig:
    action_horizon: int = HORIZON
    execute_horizon: int = 2
    use_temporal_ensembling: bool = False
    temporal_ensemble_coeff: float = 0.01
    use_completion_gate: bool = True
    completion_threshold: float = 0.25
    max_replan_retries: int = 4
    replan_on_gripper_close: bool = False
    gripper_action_index: int = 2  # last channel of this file's ACTION_DIM=3 fake actions
    gripper_open_threshold: float = 0.6
    gripper_close_threshold: float = 0.4
    use_phase_conditioning: bool = False


def make_chunk(batch_size: int = 2, fill: float = 1.0) -> torch.Tensor:
    return torch.full((batch_size, HORIZON, ACTION_DIM), fill)


def make_gripper_chunk(
    gripper_values: list[float], batch_size: int = 1, other_fill: float = 0.0
) -> torch.Tensor:
    """A `[batch_size, HORIZON, ACTION_DIM]` chunk with every non-gripper channel set to
    `other_fill` and the gripper channel (`GRIPPER_IX`) set per-horizon-step from `gripper_values`
    (must have length `HORIZON`)."""
    assert len(gripper_values) == HORIZON
    chunk = torch.full((batch_size, HORIZON, ACTION_DIM), other_fill)
    chunk[:, :, GRIPPER_IX] = torch.tensor(gripper_values)
    return chunk


def make_plan_fn(chunk: torch.Tensor, metrics: dict | None = None):
    calls = {"count": 0}

    def plan_fn():
        calls["count"] += 1
        return chunk, (metrics or {})

    return plan_fn, calls


def make_sequential_plan_fn(chunks: list[torch.Tensor]):
    """Returns a new chunk from `chunks` (in order) on each successive `plan_fn()` call -- for
    tests that need the *replanned* chunk to differ from the original."""
    calls = {"count": 0}

    def plan_fn():
        chunk = chunks[min(calls["count"], len(chunks) - 1)]
        calls["count"] += 1
        return chunk, {}

    return plan_fn, calls


def test_select_action_queue_drains_and_replans() -> None:
    executor = ActionExecutor(FakeExecutorConfig(execute_horizon=2))
    chunk = make_chunk()
    plan_fn, calls = make_plan_fn(chunk)
    current_state = torch.zeros(2, 5)

    first = executor.select_action(current_state, plan_fn)
    assert torch.equal(first, chunk[:, 0])
    assert len(executor._action_queue) == 1
    assert calls["count"] == 1

    second = executor.select_action(current_state, plan_fn)
    assert torch.equal(second, chunk[:, 1])
    assert len(executor._action_queue) == 0
    assert calls["count"] == 1  # queue not empty on the second call, so plan_fn wasn't re-invoked

    executor.select_action(current_state, plan_fn)
    assert calls["count"] == 2  # queue was empty -- fresh chunk requested


def test_reset_clears_all_state() -> None:
    executor = ActionExecutor(FakeExecutorConfig())
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={"predicted_subgoal_state": torch.ones(2, 5)})
    current_state = torch.zeros(2, 5)
    executor.select_action(current_state, plan_fn)

    executor.reset()
    assert len(executor._action_queue) == 0
    assert len(executor._ensemble_buffer) == 0
    assert executor._pending_target_state is None
    assert executor._last_gap is None
    assert executor._replan_retries == 0


def test_temporal_ensembling_bypasses_queue_and_blends_predictions() -> None:
    executor = ActionExecutor(FakeExecutorConfig(use_temporal_ensembling=True, execute_horizon=1))
    chunk = make_chunk(fill=2.0)
    plan_fn, calls = make_plan_fn(chunk)
    current_state = torch.zeros(2, 5)

    action = executor.select_action(current_state, plan_fn)
    assert calls["count"] == 1
    assert len(executor._action_queue) == 0  # queue never used on this path
    # A single buffered chunk: the ensembled action is exactly that chunk's own "now" prediction.
    assert torch.allclose(action, chunk[:, 0])

    executor.select_action(current_state, plan_fn)
    assert calls["count"] == 2  # temporal ensembling replans every step


def test_completion_gate_does_not_trigger_when_far_but_not_diverging() -> None:
    """Regression test for the original (timescale-mismatched) gate design: it flagged
    "not complete" whenever the gap merely still exceeded `completion_threshold`, which fired on
    almost every commit. With a fixed (never-changing) plan, the measured gap can't grow between
    checks, so the gate must commit a fresh full-length chunk even with `completion_threshold=0.0`
    (i.e. nowhere near "arrived")."""
    executor = ActionExecutor(FakeExecutorConfig(completion_threshold=0.0, max_replan_retries=2))
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={"predicted_subgoal_state": torch.ones(2, 5)})
    current_state = torch.zeros(2, 5)

    executor.select_action(current_state, plan_fn)
    while executor._action_queue:
        executor.select_action(current_state, plan_fn)
    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 0
    assert len(executor._action_queue) == executor._execute_horizon - 1


def test_completion_gate_triggers_single_step_replan_when_gap_grows() -> None:
    executor = ActionExecutor(FakeExecutorConfig(completion_threshold=0.0, max_replan_retries=2))
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={"predicted_subgoal_state": torch.ones(2, 5)})
    current_state = torch.zeros(2, 5)

    executor.select_action(current_state, plan_fn)
    while executor._action_queue:
        executor.select_action(current_state, plan_fn)
    # Force the next check to look like the gap grew since the last one.
    executor._last_gap = torch.zeros_like(executor._last_gap)

    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 1
    assert len(executor._action_queue) == 0

    executor._last_gap = torch.zeros_like(executor._last_gap)
    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 2
    assert len(executor._action_queue) == 0

    # Retry budget exhausted: this call must fall back to a fresh full-length commit regardless.
    executor._last_gap = torch.zeros_like(executor._last_gap)
    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 0
    assert len(executor._action_queue) == executor._execute_horizon - 1


def test_use_completion_gate_false_disables_gating_even_with_subgoal() -> None:
    """With the gate off, `_pending_target_state` is never even populated (nothing to gate with),
    and every chunk boundary commits a fresh full-length chunk regardless."""
    executor = ActionExecutor(
        FakeExecutorConfig(completion_threshold=0.0, max_replan_retries=2, use_completion_gate=False)
    )
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={"predicted_subgoal_state": torch.ones(2, 5)})
    current_state = torch.zeros(2, 5)

    for _ in range(3):
        executor.select_action(current_state, plan_fn)
        while executor._action_queue:
            executor.select_action(current_state, plan_fn)
        assert executor._replan_retries == 0
        assert executor._pending_target_state is None


def test_select_action_never_gates_without_subgoal_signal() -> None:
    """A `plan_fn` whose metrics never carry `predicted_subgoal_state` (mirrors `temporal_decoder`
    / `smolvla_nominal`) must never gate, regardless of `completion_threshold`."""
    executor = ActionExecutor(FakeExecutorConfig(completion_threshold=0.0, max_replan_retries=2))
    chunk = make_chunk()
    plan_fn, _ = make_plan_fn(chunk, metrics={})
    current_state = torch.zeros(2, 5)

    executor.select_action(current_state, plan_fn)
    while executor._action_queue:
        executor.select_action(current_state, plan_fn)
    executor.select_action(current_state, plan_fn)
    assert executor._replan_retries == 0
    assert len(executor._action_queue) == executor._execute_horizon - 1


# ---- replan_on_gripper_close (execution-only grasp-event experiment) --------------------------


def test_replan_on_gripper_close_discards_queue_and_forces_replan_on_open_to_close() -> None:
    executor = ActionExecutor(FakeExecutorConfig(execute_horizon=HORIZON, replan_on_gripper_close=True))
    # open, open, close (grasp event here, at horizon index 2), close.
    old_chunk = make_gripper_chunk([1.0, 1.0, 0.0, 0.0], other_fill=1.0)
    new_chunk = make_gripper_chunk([1.0, 1.0, 1.0, 1.0], other_fill=9.0)
    plan_fn, calls = make_sequential_plan_fn([old_chunk, new_chunk])
    current_state = torch.zeros(1, 5)

    first = executor.select_action(current_state, plan_fn)
    assert torch.equal(first, old_chunk[:, 0])
    second = executor.select_action(current_state, plan_fn)
    assert torch.equal(second, old_chunk[:, 1])
    assert calls["count"] == 1
    assert executor.gripper_events == []

    # Third dispatched action is the closing one (horizon index 2) -- it must still execute
    # normally, and the event must fire right after.
    third = executor.select_action(current_state, plan_fn)
    assert torch.equal(third, old_chunk[:, 2])
    assert len(executor.gripper_events) == 1
    event = executor.gripper_events[0]
    assert event["step_index"] == 2
    assert event["queue_len_at_event"] == 1
    assert len(event["old_remaining_actions_after_grasp"]) == 1
    assert torch.equal(event["old_remaining_actions_after_grasp"][0], old_chunk[:, 3])
    # The old chunk's last (never-executed) action was discarded, not merely queued.
    assert len(executor._action_queue) == 0
    assert calls["count"] == 1  # plan_fn not yet re-invoked -- that happens next call

    # Next call: queue is empty -> forces a fresh plan_fn() call instead of executing the
    # discarded old_chunk[:, 3]; the dispatched action is the *new* chunk's first action.
    fourth = executor.select_action(current_state, plan_fn)
    assert calls["count"] == 2
    assert torch.equal(fourth, new_chunk[:, 0])
    assert not torch.equal(fourth, old_chunk[:, 3])


def test_replan_on_gripper_close_only_fires_once_per_episode() -> None:
    """Guard against repeated replanning from gripper chatter: a second genuine open->close
    transition later in the same episode must not trigger a second discard."""
    executor = ActionExecutor(FakeExecutorConfig(execute_horizon=HORIZON, replan_on_gripper_close=True))
    chunk1 = make_gripper_chunk([1.0, 0.0, 1.0, 0.0])  # transition at index 1 (fires)
    chunk2 = make_gripper_chunk([1.0, 0.0, 1.0, 0.0])  # transition at index 1 again (must NOT fire)
    plan_fn, calls = make_sequential_plan_fn([chunk1, chunk2])
    current_state = torch.zeros(1, 5)

    executor.select_action(current_state, plan_fn)  # index 0 (open)
    executor.select_action(current_state, plan_fn)  # index 1 (close) -- event fires, queue cleared
    assert len(executor.gripper_events) == 1
    assert calls["count"] == 1

    executor.select_action(current_state, plan_fn)  # queue empty -> replans into chunk2, index 0 (open)
    assert calls["count"] == 2
    executor.select_action(current_state, plan_fn)  # chunk2 index 1 (close) -- guard must suppress this
    assert len(executor.gripper_events) == 1  # still just the one event
    assert len(executor._action_queue) == 2  # queue drains normally now, not cleared again


def test_replan_on_gripper_close_disabled_leaves_execution_unchanged() -> None:
    """Default (`replan_on_gripper_close=False`) must reproduce plain open-loop draining exactly,
    even for a chunk containing an open->close transition -- no event recorded, nothing discarded."""
    executor = ActionExecutor(FakeExecutorConfig(execute_horizon=HORIZON, replan_on_gripper_close=False))
    chunk = make_gripper_chunk([1.0, 1.0, 0.0, 0.0])
    plan_fn, calls = make_plan_fn(chunk)
    current_state = torch.zeros(1, 5)

    dispatched = [executor.select_action(current_state, plan_fn) for _ in range(HORIZON)]
    for i, action in enumerate(dispatched):
        assert torch.equal(action, chunk[:, i])
    assert executor.gripper_events == []
    assert calls["count"] == 1  # whole chunk drained before any replan, as normal


def test_detect_gripper_close_event_matches_first_debounced_open_to_close() -> None:
    values = torch.tensor([1.0, 0.55, 0.0, 0.0, 1.0])  # 0.55 is in the hysteresis band -> stays "open"
    assert detect_gripper_close_event(values, open_threshold=0.6, close_threshold=0.4) == 2


def test_detect_gripper_close_event_returns_none_without_a_transition() -> None:
    values = torch.tensor([1.0, 1.0, 0.55, 1.0])
    assert detect_gripper_close_event(values, open_threshold=0.6, close_threshold=0.4) is None


# ---- use_phase_conditioning (live phase tracking) ----------------------------------------------


def test_phase_starts_at_zero_and_flips_to_one_on_first_grasp() -> None:
    executor = ActionExecutor(FakeExecutorConfig(execute_horizon=HORIZON, use_phase_conditioning=True))
    chunk = make_gripper_chunk([1.0, 1.0, 0.0, 0.0])  # close at horizon index 2
    plan_fn, _ = make_plan_fn(chunk)
    current_state = torch.zeros(1, 5)

    assert executor.phase == 0
    executor.select_action(current_state, plan_fn)  # index 0 (open)
    assert executor.phase == 0
    executor.select_action(current_state, plan_fn)  # index 1 (open)
    assert executor.phase == 0
    executor.select_action(current_state, plan_fn)  # index 2 (close) -- grasp event
    assert executor.phase == 1
    executor.select_action(current_state, plan_fn)  # index 3 -- stays 1 for the rest of the episode
    assert executor.phase == 1


def test_phase_tracking_is_independent_of_replan_on_gripper_close() -> None:
    """`use_phase_conditioning` must flip `phase` even when `replan_on_gripper_close=False` --
    phase conditioning and immediate replanning are separate, independently toggleable knobs."""
    executor = ActionExecutor(
        FakeExecutorConfig(
            execute_horizon=HORIZON, use_phase_conditioning=True, replan_on_gripper_close=False
        )
    )
    chunk = make_gripper_chunk([1.0, 1.0, 0.0, 0.0])
    plan_fn, calls = make_plan_fn(chunk)
    current_state = torch.zeros(1, 5)

    dispatched = [executor.select_action(current_state, plan_fn) for _ in range(HORIZON)]
    for i, action in enumerate(dispatched):
        assert torch.equal(action, chunk[:, i])  # queue never discarded -- plain draining
    assert executor.phase == 1  # but phase still flipped on the grasp event
    assert calls["count"] == 1  # and no extra replan was forced


def test_phase_stays_zero_when_use_phase_conditioning_is_off() -> None:
    executor = ActionExecutor(FakeExecutorConfig(execute_horizon=HORIZON, use_phase_conditioning=False))
    chunk = make_gripper_chunk([1.0, 1.0, 0.0, 0.0])
    plan_fn, _ = make_plan_fn(chunk)
    current_state = torch.zeros(1, 5)

    for _ in range(HORIZON):
        executor.select_action(current_state, plan_fn)
    assert executor.phase == 0
    assert executor.gripper_events == []  # tracking never even ran


def test_phase_resets_to_zero() -> None:
    executor = ActionExecutor(FakeExecutorConfig(execute_horizon=HORIZON, use_phase_conditioning=True))
    chunk = make_gripper_chunk([1.0, 1.0, 0.0, 0.0])
    plan_fn, _ = make_plan_fn(chunk)
    current_state = torch.zeros(1, 5)
    for _ in range(3):
        executor.select_action(current_state, plan_fn)
    assert executor.phase == 1

    executor.reset()
    assert executor.phase == 0
