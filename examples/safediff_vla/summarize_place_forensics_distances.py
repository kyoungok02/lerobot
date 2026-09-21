#!/usr/bin/env python
"""Read-only post-processing of A's `place_phase_forensics.py` output (`episodes_raw.json`) --
derives the extra spatial-alignment metrics it records per-step but doesn't itself aggregate:
EE<->card minimum 3D distance, distance at the first commanded-close step, distance 5/10/20 steps
before that, per-axis (dx, dy, dz) residual at the minimum-distance step, that step's index and its
`execute_horizon=50` chunk offset (`step_ix % 50`), max card height, and whether the 0.9m lift
threshold was ever reached. Does not touch `place_phase_forensics.py` or re-run any policy/env
code -- pure arithmetic over its already-written JSON.

Usage:
    uv run python examples/safediff_vla/summarize_place_forensics_distances.py \
        outputs/eval/safediff_vla_b_padfix20k_place_forensics/episodes_raw.json
"""

import argparse
import json
from pathlib import Path

LIFT_THRESHOLD = 0.9
EXECUTE_HORIZON = 50


def summarize_episode(ep: dict) -> dict:
    steps = ep["steps"]
    dists = [s["ee_to_card_dist"] for s in steps]
    min_ix = min(range(len(steps)), key=lambda i: dists[i])
    min_step = steps[min_ix]

    close_ix = None
    if ep["first_commanded_close_step"] is not None:
        # first_commanded_close_step is 1-based (see place_phase_forensics.py's
        # `commanded_binary.index(False) + 1`); `steps` is 0-indexed by step_ix, and step_ix 0 is
        # the pre-first-action reset snapshot, so step_ix == first_commanded_close_step lines up
        # with the physics state recorded *after* that close action executed.
        target_ix = ep["first_commanded_close_step"]
        close_ix = next((i for i, s in enumerate(steps) if s["step_ix"] == target_ix), None)

    def dist_at(step_ix_target: int | None) -> float | None:
        if step_ix_target is None or step_ix_target < 0:
            return None
        match = next((s for s in steps if s["step_ix"] == step_ix_target), None)
        return match["ee_to_card_dist"] if match else None

    dx, dy, dz = None, None, None
    if min_step is not None:
        ee = min_step["ee_pos"]
        card = min_step["card_pos"]
        dx, dy, dz = ee[0] - card[0], ee[1] - card[1], ee[2] - card[2]

    close_dist = dist_at(ep["first_commanded_close_step"]) if close_ix is not None else None
    close_step = ep["first_commanded_close_step"]

    return {
        "seed": ep["seed"],
        "success": ep["success"],
        "ee_to_card_min_dist_m": min(dists) if dists else None,
        "ee_to_card_min_dist_step_ix": min_step["step_ix"] if min_step else None,
        "ee_to_card_min_dist_chunk_offset": (min_step["step_ix"] % EXECUTE_HORIZON) if min_step else None,
        "dx_at_min_dist": dx,
        "dy_at_min_dist": dy,
        "dz_at_min_dist": dz,
        "first_commanded_close_step": close_step,
        "ee_to_card_dist_at_commanded_close": close_dist,
        "ee_to_card_dist_5step_before_close": dist_at(close_step - 5) if close_step is not None else None,
        "ee_to_card_dist_10step_before_close": dist_at(close_step - 10) if close_step is not None else None,
        "ee_to_card_dist_20step_before_close": dist_at(close_step - 20) if close_step is not None else None,
        "max_card_height_m": ep["max_card_height"],
        "reached_lift_threshold_0_9m": (ep["max_card_height"] or 0) >= LIFT_THRESHOLD,
        "first_physics_grasp_step": ep["first_physics_grasp_step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("episodes_raw_json", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    episodes = json.loads(Path(args.episodes_raw_json).read_text())
    summaries = [summarize_episode(ep) for ep in episodes]

    out_path = (
        Path(args.output) if args.output else Path(args.episodes_raw_json).with_name("distance_summary.json")
    )
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"wrote {out_path}")
    for s in summaries:
        print(
            f"seed={s['seed']} success={s['success']} "
            f"min_dist={s['ee_to_card_min_dist_m']:.4f}m@step{s['ee_to_card_min_dist_step_ix']}"
            f"(chunk_offset={s['ee_to_card_min_dist_chunk_offset']}) "
            f"dxyz=({s['dx_at_min_dist']:.4f},{s['dy_at_min_dist']:.4f},{s['dz_at_min_dist']:.4f}) "
            f"dist_at_close={s['ee_to_card_dist_at_commanded_close']} "
            f"max_height={s['max_card_height_m']:.4f} reached_0.9m={s['reached_lift_threshold_0_9m']}"
        )


if __name__ == "__main__":
    main()
