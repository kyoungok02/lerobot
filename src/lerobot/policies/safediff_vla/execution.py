"""Inference-time action execution: turns a policy's per-chunk `plan_fn()` output into one action
per environment step. Deliberately independent of any model/checkpoint weights (see
`ActionExecutor`'s docstring) -- generation (`SafeDiffVLAPolicy.plan_action_chunk` /
`legacy.modeling_legacy_diffusion.LegacySafeDiffVLAPolicy.plan_action_chunk`) and execution
strategy (queueing, temporal ensembling, completion-gated replanning) are separate concerns that
can vary independently: the same trained checkpoint can be evaluated open-loop for the full
`action_horizon`, replanned every few steps, or temporally ensembled, without retraining.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import Tensor

from .state_predictor import completion_gap

logger = logging.getLogger(__name__)


class ExecutorConfig(Protocol):
    """The subset of `SafeDiffVLAConfig` / `legacy.LegacySafeDiffVLAConfig` fields `ActionExecutor`
    reads. Both configs satisfy this structurally (duck typing) -- no shared base class needed."""

    action_horizon: int
    execute_horizon: int
    use_temporal_ensembling: bool
    temporal_ensemble_coeff: float
    use_completion_gate: bool
    completion_threshold: float
    max_replan_retries: int
    # Execution-time-only experiment (no retraining): see `ActionExecutor`'s gripper-event handling
    # below and `replan_on_gripper_close`'s own docstring in `configuration_safediff_vla.py`.
    replan_on_gripper_close: bool
    gripper_action_index: int
    gripper_open_threshold: float
    gripper_close_threshold: float
    # Live phase tracking for `TemporalActionDecoder`'s `use_phase` conditioning -- see
    # `use_phase_conditioning`'s own docstring in `configuration_safediff_vla.py`.
    use_phase_conditioning: bool


def detect_gripper_close_event(
    gripper_values: Tensor, open_threshold: float, close_threshold: float
) -> int | None:
    """Step index of the first hysteresis-debounced open->closed gripper transition in a full
    per-step gripper-channel trajectory. Pure post-hoc analysis function mirroring
    `ActionExecutor`'s own incremental detection (`_update_gripper_close_state` below) one-for-one,
    for use on an *already-executed* action stream (e.g. a `replan_on_gripper_close=False` baseline
    rollout, which never populates `ActionExecutor.gripper_events`).

    `gripper_values`: 1-D `[T]` tensor, one scalar per executed step. Convention (VLABench,
    `envs/vlabench.py`'s `_build_ctrl_from_action`: `finger_qpos = gripper * FINGER_OPEN`): higher
    value = more open (1 = fully open, 0 = fully closed) -- same channel/convention
    `examples/safediff_vla/eval_baseline_rollout.py`'s `GRIPPER_INDEX`/`GRIPPER_THRESHOLD` already
    document. Hysteresis: a value `>= open_threshold` is confidently "open", `<= close_threshold`
    confidently "closed"; values in between hold whatever state was last confidently observed
    (debounces chatter instead of flipping on every small oscillation around a single midpoint).

    Returns the index of the first step where the debounced state is "closed" while the
    immediately preceding step's debounced state was "open" -- i.e. the episode's first genuine
    grasp attempt -- or `None` if the gripper never (confidently) closes after being open.
    """
    state: bool | None = None
    for i, value in enumerate(gripper_values.tolist()):
        prev_state = state
        if value >= open_threshold:
            state = True
        elif value <= close_threshold or state is None:
            state = False
        if prev_state is True and state is False:
            return i
    return None


class ActionExecutor:
    """Owns all per-episode execution state (action queue, temporal-ensembling buffer, completion
    gate) and the policy for turning a fresh `plan_fn()` chunk into the action for *this* step.
    Holds no model parameters of its own.
    """

    def __init__(self, config: ExecutorConfig) -> None:
        self._action_horizon = config.action_horizon
        self._execute_horizon = config.execute_horizon
        self._use_temporal_ensembling = config.use_temporal_ensembling
        self._temporal_ensemble_coeff = config.temporal_ensemble_coeff
        self._use_completion_gate = config.use_completion_gate
        self._completion_threshold = config.completion_threshold
        self._max_replan_retries = config.max_replan_retries
        self._replan_on_gripper_close = config.replan_on_gripper_close
        self._gripper_action_index = config.gripper_action_index
        self._gripper_open_threshold = config.gripper_open_threshold
        self._gripper_close_threshold = config.gripper_close_threshold
        self._use_phase_conditioning = config.use_phase_conditioning
        # Either feature needs the same live open->closed grasp-event detection; only
        # `replan_on_gripper_close` additionally discards the queue on that event (see
        # `_handle_gripper_close_event`).
        self._track_gripper_event = self._replan_on_gripper_close or self._use_phase_conditioning
        if self._track_gripper_event:
            logger.info(
                "ActionExecutor: replan_on_gripper_close=%s use_phase_conditioning=%s -- gripper "
                "convention: action channel %d, higher value = more open (VLABench "
                "envs/vlabench.py: finger_qpos = gripper * FINGER_OPEN; 1=fully open, 0=fully "
                "closed), hysteresis open>=%.2f / close<=%.2f. On the episode's first debounced "
                "open->closed transition: %s%s",
                self._replan_on_gripper_close,
                self._use_phase_conditioning,
                self._gripper_action_index,
                self._gripper_open_threshold,
                self._gripper_close_threshold,
                "the action queue is discarded right after that closing action executes, forcing a "
                "fresh plan_fn() call next step"
                if self._replan_on_gripper_close
                else "no queue change",
                " and `phase` switches from 0 (pre-grasp) to 1 (post-grasp/transport) for every "
                "later plan_fn() call this episode"
                if self._use_phase_conditioning
                else "",
            )
        self.reset()

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self._execute_horizon)
        # Holds up to `action_horizon` past chunk predictions, oldest first, for temporal
        # ensembling: the k-th most recently appended chunk was queried k steps ago, so its
        # prediction for "now" lives at its own index k (`_ensembled_action` below).
        self._ensemble_buffer: deque[Tensor] = deque(maxlen=self._action_horizon)
        # Subgoal state predicted as of the last *fully committed* chunk, and the gap to it
        # measured at that same moment (see `select_action`'s completion gate). Both stay None
        # forever for architectures/policies with no subgoal signal.
        self._pending_target_state: Tensor | None = None
        self._last_gap: Tensor | None = None
        self._replan_retries = 0
        # `replan_on_gripper_close` / `use_phase_conditioning` shared state (see
        # `_handle_gripper_close_event` below). `_gripper_is_open` is None until the first
        # confidently-open-or-closed action is seen.
        self._gripper_is_open: bool | None = None
        self._gripper_event_fired = False
        self._step_index = -1
        self.gripper_events: list[dict[str, Any]] = []
        # `use_phase_conditioning` only: 0 = pre-grasp, 1 = post-grasp/transport, read by
        # `SafeDiffVLAPolicy.select_action` on every call. Stays 0 (harmlessly unused) when the
        # flag is off.
        self.phase = 0

    def _ensembled_action(self, chunk: Tensor) -> Tensor:
        """Blend "now"-predictions from every buffered chunk with exponential-decay weights.

        `chunk` (this step's fresh prediction) is pushed last, so iterating the buffer newest
        -> oldest via `reversed()` lines up positional age with the offset each chunk holds its
        prediction for "now" at: age 0 is `chunk` itself (offset 0), age 1 is last step's chunk
        (offset 1, since it was queried one step ago), and so on.
        """
        self._ensemble_buffer.append(chunk)
        predictions, weights = [], []
        for age, past_chunk in enumerate(reversed(self._ensemble_buffer)):
            predictions.append(past_chunk[:, age])
            weights.append(math.exp(-self._temporal_ensemble_coeff * age))
        weights_t = torch.tensor(weights, device=chunk.device, dtype=chunk.dtype)
        weights_t /= weights_t.sum()
        return (torch.stack(predictions, dim=0) * weights_t[:, None, None]).sum(dim=0)

    def select_action(
        self, current_state: Tensor, plan_fn: Callable[[], tuple[Tensor, dict[str, Any]]]
    ) -> Tensor:
        """`current_state`: `[B, state_dim]`, the current (t=0) state, used only for the
        completion gate. `plan_fn`: calls the policy's `plan_action_chunk` (or
        `predict_action_chunk`, for the temporal-ensembling path) and returns its
        `(actions, metrics)` pair."""
        self._step_index += 1
        if self._use_temporal_ensembling:
            actions, _ = plan_fn()
            return self._ensembled_action(actions)
        if not self._action_queue:
            # A subgoal can legitimately be many chunks away, so "not yet arrived after one
            # execute_horizon" is the normal case, not a problem: gating on that (as an earlier
            # version of this method did) made the gate fire on almost every commit, collapsing
            # execution into a near-permanent single-step replan loop and producing visibly jerky
            # motion. What actually signals trouble is the gap *growing* since the last check: the
            # last chunk moved away from the target it was aiming for, while still being
            # meaningfully far from it.
            gap = (
                completion_gap(self._pending_target_state, current_state)
                if self._pending_target_state is not None
                else None
            )
            diverging = (
                gap is not None
                and self._last_gap is not None
                and bool((gap > self._last_gap).any())
                and bool((gap > self._completion_threshold).any())
            )
            chunk, metrics = plan_fn()
            # Policies with no subgoal signal never gate: there's nothing to measure progress
            # against, so always commit a fresh chunk.
            has_subgoal = self._use_completion_gate and "predicted_subgoal_state" in metrics
            if has_subgoal and diverging and self._replan_retries < self._max_replan_retries:
                # Take one corrective step towards the *same* still-pending target and reassess
                # on the very next call, instead of silently moving on to whatever the model
                # proposes next.
                self._action_queue.extend(chunk.transpose(0, 1)[:1])
                self._replan_retries += 1
                self._last_gap = gap
            else:
                if has_subgoal:
                    self._pending_target_state = metrics["predicted_subgoal_state"]
                    self._last_gap = completion_gap(self._pending_target_state, current_state)
                self._action_queue.extend(chunk.transpose(0, 1)[: self._execute_horizon])
                self._replan_retries = 0
        action = self._action_queue.popleft()
        if self._track_gripper_event:
            self._handle_gripper_close_event(action)
        return action

    def _handle_gripper_close_event(self, dispatched_action: Tensor) -> None:
        """Shared `replan_on_gripper_close` / `use_phase_conditioning` execution-time logic (no
        model/training changes from this method alone): on this episode's first debounced
        open->closed gripper transition,
          - `use_phase_conditioning`: flip `self.phase` from 0 to 1 for every later `plan_fn()`
            call this episode.
          - `replan_on_gripper_close`: additionally discard whatever the *old* plan still had
            queued right after `dispatched_action` (the closing action itself) executes, so
            `select_action`'s own `if not self._action_queue` branch calls `plan_fn()` again on the
            very next step -- a fresh chunk conditioned on the post-grasp observation/state,
            instead of blindly continuing the pre-grasp chunk.
        Guarded by `_gripper_event_fired` so a single episode only ever does this once, even with
        further open<->close chatter later on.
        """
        if self._gripper_event_fired:
            return
        assert dispatched_action.shape[0] == 1, (
            "replan_on_gripper_close/use_phase_conditioning only support batch size 1 "
            f"(single-environment closed-loop eval); got batch size {dispatched_action.shape[0]}"
        )
        value = dispatched_action[0, self._gripper_action_index].item()
        prev_open = self._gripper_is_open
        if value >= self._gripper_open_threshold:
            self._gripper_is_open = True
        elif value <= self._gripper_close_threshold or prev_open is None:
            self._gripper_is_open = False
        if prev_open is True and self._gripper_is_open is False:
            self._gripper_event_fired = True
            self.phase = 1
            old_remaining = [a.detach().clone() for a in list(self._action_queue)[:5]]
            self.gripper_events.append(
                {
                    "step_index": self._step_index,
                    "queue_len_at_event": len(self._action_queue),
                    "old_remaining_actions_after_grasp": old_remaining,
                }
            )
            if self._replan_on_gripper_close:
                self._action_queue.clear()
