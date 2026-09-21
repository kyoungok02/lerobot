#!/usr/bin/env python
"""Read-only, no-rollout post-hoc analysis of the execute_horizon={50,20,10} ablation already
saved at `outputs/eval/safediff_vla_execute_horizon_ablation/execute_horizon_ablation.json`
(canonical checkpoint: `sincos_20k_padfix`, canonical execution baseline: execute_horizon=50).
Runs no policy, no environment, no training -- pure statistics over each episode's already-saved
`gripper_transition_steps` (all threshold crossings, direction unlabeled) / `first_grasp_timestep`
(first open->close) / `episode_len` / `success` / `seed` / `mean_abs_delta_action(2)`.

Direction reconstruction: gripper transitions strictly alternate direction. Every episode's first
transition equals its `first_grasp_timestep` (verified for all 30 episodes before writing this
script -- the env always starts with an open gripper), so the full direction-labeled sequence is
recoverable exactly: open->close, close->open, open->close, ... with no ambiguity. No raw
per-step gripper values are needed for transition timing/counting/duration/boundary-proximity
analysis.

Known gap: raw per-step gripper channel values were never persisted by the eval harness (only
derived transition-step indices + 7-dim-pooled mean_abs_delta_action/2), so a literal
gripper-only mean_abs_delta_gripper/2 cannot be reconstructed here without re-running rollouts,
which this analysis intentionally does not do. `transitions_per_step` (density) and
`transitions_per_chunk` are reported as well-justified proxies instead (for a near-binary
threshold-crossing channel, delta magnitude is dominated by flip frequency) -- see the final
report's caveat and next-experiment candidates.

Outputs:
    outputs/eval/safediff_vla_execute_horizon_ablation/gripper_dithering_analysis.json

Usage:
    python examples/safediff_vla/execute_horizon_gripper_analysis.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ABLATION_JSON = Path("outputs/eval/safediff_vla_execute_horizon_ablation/execute_horizon_ablation.json")
OUTPUT_JSON = Path("outputs/eval/safediff_vla_execute_horizon_ablation/gripper_dithering_analysis.json")
BOUNDARY_WINDOW = 5  # "+/-5 step" as specified
REAL_HOLD_MIN_STEPS = 20  # threshold to call a closed segment a "real" grasp hold vs a transient touch


def reconstruct_segments(transitions: list[int], episode_len: int) -> list[dict]:
    """`transitions` (all threshold crossings, ascending, alternating direction starting with
    open->close per this dataset's verified convention) -> closed/open segments spanning
    `[0, episode_len)`. Segment `i` starts at `transitions[i-1]` (or 0) and ends just before
    `transitions[i]` (or at `episode_len`); segment parity: even index (0, 2, ...) = open (gripper
    starts open), odd index = closed."""
    bounds = [0, *transitions, episode_len]
    segments = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        state = "closed" if i % 2 == 1 else "open"
        segments.append({"state": state, "start": start, "end": end, "duration": end - start})
    return segments


def episode_metrics(ep: dict, eh: int) -> dict:
    transitions = ep["gripper_transition_steps"]
    episode_len = ep["episode_len"]
    first_close = ep["first_grasp_timestep"]

    assert (len(transitions) == 0) == (first_close is None)
    assert first_close is None or transitions[0] == first_close, "direction-reconstruction assumption violated"

    open_to_close = transitions[0::2]  # indices 0, 2, 4, ... per the starts-open-then-closes convention
    close_to_open = transitions[1::2]
    first_reopen = close_to_open[0] if close_to_open else None

    segments = reconstruct_segments(transitions, episode_len)
    closed_durations = [s["duration"] for s in segments if s["state"] == "closed"]
    open_durations = [s["duration"] for s in segments if s["state"] == "open"]

    # Chunk/replan boundaries: a fresh `execute_horizon`-step chunk is committed at env steps
    # eh, 2*eh, 3*eh, ... (see execution.py::ActionExecutor -- architecture=temporal_decoder has
    # no subgoal, so the completion-gate branch never fires and exactly `execute_horizon` steps
    # are committed every replan, unconditionally). Step 0 (the very first chunk) is excluded --
    # there is no prior chunk for it to be inconsistent with.
    boundaries = list(range(eh, episode_len, eh))
    n_replans = len(boundaries)

    def dist_to_nearest_boundary(t: int) -> int:
        return min((abs(t - b) for b in boundaries), default=10**9)

    near_boundary = [t for t in transitions if dist_to_nearest_boundary(t) <= BOUNDARY_WINDOW]
    # Modular-position check (robust even when eh <= 2*BOUNDARY_WINDOW, where the +/-5 window
    # trivially covers ~100% of the timeline -- see report caveat): position of each transition
    # within its own inter-boundary cycle, folded to [0, eh/2] (distance from the *nearest*
    # boundary edge of that cycle, not just the following one).
    mod_positions = [min(t % eh, eh - (t % eh)) for t in transitions]

    return {
        "seed": ep["seed"],
        "success": ep["success"],
        "episode_len": episode_len,
        "mean_abs_delta_action": ep["mean_abs_delta_action"],
        "mean_abs_delta2_action": ep["mean_abs_delta2_action"],
        "first_close_timestep": first_close,
        "first_reopen_timestep": first_reopen,
        "total_gripper_transitions": len(transitions),
        "open_to_close_count": len(open_to_close),
        "close_to_open_count": len(close_to_open),
        "reopen_count_after_first_grasp": len(close_to_open),  # all closes-to-opens happen after the first grasp by construction
        "longest_continuous_closed_duration": max(closed_durations, default=0),
        "longest_continuous_open_duration": max(open_durations, default=0),
        "n_replans_after_first_chunk": n_replans,
        "transitions_near_boundary_pm5": len(near_boundary),
        "transitions_near_boundary_pm5_fraction_of_total": (len(near_boundary) / len(transitions)) if transitions else None,
        "transitions_per_replan": (len(transitions) / n_replans) if n_replans else None,
        "transitions_per_step": len(transitions) / episode_len,
        "mean_dist_to_nearest_boundary_folded": statistics.mean(mod_positions) if mod_positions else None,
        "ends_closed": segments[-1]["state"] == "closed" if segments else None,
        "closed_segments": closed_durations,
        "open_segments_after_first_close": open_durations[1:],  # exclude the initial pre-grasp open segment
    }


def failure_phase(m: dict) -> str:
    """Coarse, explicitly-thresholded classification of *where* a failed episode's gripper
    behavior broke down -- not a ground-truth task-phase label (none is available from
    gripper-only data), just a read of the reconstructed segments."""
    if m["first_close_timestep"] is None:
        return "never_attempted_grasp"
    if not m["closed_segments"] or max(m["closed_segments"]) < REAL_HOLD_MIN_STEPS:
        return "grasp_never_held"  # touched closed but never held >= REAL_HOLD_MIN_STEPS steps
    if m["ends_closed"]:
        return "held_grasp_to_episode_end"  # gripper mechanics look fine; task failed for another reason
    return "grasped_then_reopened"  # held a real grasp at some point, then let go before episode end


def main() -> None:
    data = json.loads(ABLATION_JSON.read_text())
    report: dict = {"source": str(ABLATION_JSON), "conditions": {}}

    for eh_str in ("50", "20", "10"):
        eh = int(eh_str)
        episodes = data["conditions"][eh_str]["closed_loop_rollout"]["per_episode"]
        per_ep = [episode_metrics(ep, eh) for ep in episodes]
        for m in per_ep:
            m["failure_phase"] = None if m["success"] else failure_phase(m)

        def agg(key: str) -> float:
            vals = [m[key] for m in per_ep if m[key] is not None]
            return statistics.mean(vals) if vals else None

        near_total = sum(m["transitions_near_boundary_pm5"] for m in per_ep)
        trans_total = sum(m["total_gripper_transitions"] for m in per_ep)

        report["conditions"][eh_str] = {
            "execute_horizon": eh,
            "n_episodes": len(per_ep),
            "n_success": sum(1 for m in per_ep if m["success"]),
            "per_episode": per_ep,
            "aggregate": {
                "mean_total_gripper_transitions": agg("total_gripper_transitions"),
                "mean_open_to_close_count": agg("open_to_close_count"),
                "mean_close_to_open_count": agg("close_to_open_count"),
                "mean_first_close_timestep": agg("first_close_timestep"),
                "mean_first_reopen_timestep": agg("first_reopen_timestep"),
                "mean_longest_closed_duration": agg("longest_continuous_closed_duration"),
                "mean_longest_open_duration": agg("longest_continuous_open_duration"),
                "mean_abs_delta_action": agg("mean_abs_delta_action"),
                "mean_abs_delta2_action": agg("mean_abs_delta2_action"),
                "mean_transitions_per_replan": agg("transitions_per_replan"),
                "mean_transitions_per_step": agg("transitions_per_step"),
                "pooled_transitions_near_boundary_pm5_fraction": (near_total / trans_total) if trans_total else None,
                "mean_dist_to_nearest_boundary_folded": agg("mean_dist_to_nearest_boundary_folded"),
                "failure_phase_counts": {
                    phase: sum(1 for m in per_ep if m["failure_phase"] == phase)
                    for phase in ("never_attempted_grasp", "grasp_never_held", "grasped_then_reopened", "held_grasp_to_episode_end")
                },
            },
        }

    # Null/chance baseline for the +/-5 boundary window, per eh: fraction of ALL timesteps in
    # [0, episode_len) that fall within +/-5 of *some* boundary, i.e. what a uniformly-random
    # transition timing would give "for free" -- needed because for eh<=10 the window's radius
    # (5) is comparable to or larger than half the boundary spacing, so raw "% near boundary"
    # alone is not comparable across conditions (see report).
    episode_len_typical = 500
    for eh_str in ("50", "20", "10"):
        eh = int(eh_str)
        boundaries = list(range(eh, episode_len_typical, eh))
        covered = set()
        for b in boundaries:
            for t in range(max(0, b - BOUNDARY_WINDOW), min(episode_len_typical, b + BOUNDARY_WINDOW + 1)):
                covered.add(t)
        report["conditions"][eh_str]["null_baseline_pm5_coverage_fraction"] = len(covered) / episode_len_typical

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
