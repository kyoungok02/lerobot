#!/usr/bin/env python
"""Post-hoc, read-only augmentation of the already-saved episodes_raw.json files (produced by
`replan_consistency_logging.py`): adds a second "plan-vs-executed-reality" consistency metric that
is well-defined for ALL execute_horizon values, including eh=50, where the original "plan-vs-old-
plan" metric (`summary.json`'s `replan_consistency`) is structurally undefined -- at
execute_horizon == action_horizon, a chunk's entire 50-step prediction is fully consumed before the
next replan is requested, so there is no overlapping "old chunk's prediction for this same future
step" to compare the new chunk against (`n_replans_total` comes out 0 there, not a bug).

`replan_vs_executed_reality_consistency` instead compares the new chunk's first-step gripper
prediction against the actual last-executed gripper value just before the boundary (what really
just happened, rather than a stale unconsumed old-plan prediction) -- this is well-defined for
every execute_horizon and is the metric this analysis uses for cross-condition comparison.

Runs no policy, no environment; only reads/writes JSON already on disk under
outputs/eval/safediff_vla_replan_consistency/.

Usage:
    python examples/safediff_vla/replan_consistency_augment.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("outputs/eval/safediff_vla_replan_consistency")
GRIPPER_THRESHOLD = 0.5


def main() -> None:
    for eh in (50, 20, 10):
        d = ROOT / f"execute_horizon_{eh}"
        records = json.loads((d / "episodes_raw.json").read_text())
        all_b = []
        for rec in records:
            executed = rec["executed_gripper"]
            for c in rec["chunks"][1:]:  # skip chunk 0 (no predecessor to compare against)
                boundary_t = c["start_timestep"]
                if boundary_t == 0 or boundary_t - 1 >= len(executed) or boundary_t >= rec["episode_len"]:
                    continue
                new_g0 = c["gripper_50step_raw"][0]
                executed_prev = executed[boundary_t - 1]
                new_binary = new_g0 > GRIPPER_THRESHOLD
                prev_binary = executed_prev > GRIPPER_THRESHOLD
                transitions = rec["gripper_transitions_exact"]
                dist = min((abs(boundary_t - t) for t in transitions), default=None)
                all_b.append(
                    {
                        "seed": rec["seed"],
                        "boundary_timestep": boundary_t,
                        "new_plan_gripper_first": new_g0,
                        "executed_gripper_just_before_boundary": executed_prev,
                        "abs_diff": abs(new_g0 - executed_prev),
                        "binary_disagree": new_binary != prev_binary,
                        "actual_transition_at_boundary": boundary_t in transitions,
                        "distance_to_nearest_actual_transition": dist,
                    }
                )

        disagree = [r for r in all_b if r["binary_disagree"]]
        agree = [r for r in all_b if not r["binary_disagree"]]

        def rate(subset: list[dict]) -> float | None:
            if not subset:
                return None
            return sum(1 for r in subset if r["actual_transition_at_boundary"]) / len(subset)

        summary_path = d / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["replan_vs_executed_reality_consistency"] = {
            "definition": "compares new_plan_gripper[0] (new chunk's first-step raw prediction) against "
            "executed_gripper[boundary_t - 1] (the actual last-executed gripper value, i.e. what really "
            "just happened) rather than the old chunk's own stale prediction for that index -- well-defined "
            "for every execute_horizon including 50, unlike the plan-vs-old-plan metric above which requires "
            "execute_horizon < action_horizon (chunk overlap) to exist at all.",
            "n_boundaries": len(all_b),
            "n_binary_disagreements": len(disagree),
            "disagreement_rate": (len(disagree) / len(all_b)) if all_b else None,
            "mean_abs_diff": statistics.mean(r["abs_diff"] for r in all_b) if all_b else None,
            "mean_abs_diff_when_disagree": statistics.mean(r["abs_diff"] for r in disagree) if disagree else None,
            "mean_abs_diff_when_agree": statistics.mean(r["abs_diff"] for r in agree) if agree else None,
            "p_actual_transition_at_boundary_given_disagree": rate(disagree),
            "p_actual_transition_at_boundary_given_agree": rate(agree),
        }
        summary["replan_consistency"]["note_if_undefined"] = (
            "n_replans_total=0 at eh=action_horizon (50): consecutive chunks never overlap (old chunk's "
            "50-step prediction is fully consumed before the new chunk is requested), so there is no "
            "'old chunk's prediction for this same future step' to compare against. See "
            "replan_vs_executed_reality_consistency for the well-defined, cross-condition-comparable metric."
            if eh == 50
            else None
        )
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"eh={eh}: n_boundaries={len(all_b)} disagreement_rate={summary['replan_vs_executed_reality_consistency']['disagreement_rate']}")


if __name__ == "__main__":
    main()
