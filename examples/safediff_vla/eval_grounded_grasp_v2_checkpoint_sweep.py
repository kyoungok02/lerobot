#!/usr/bin/env python
"""Runs `eval_grounded_grasp_v2_5k.py`'s closed-loop rollout evaluation (select_poker seeds
1000-1002, unchanged -- no architecture/loss/execution code touched by this script or that one)
against every checkpoint saved by `train_grounded_grasp_v2_20k.py`
(`outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/{005000,010000,015000,020000}`),
each as its own subprocess (isolates each checkpoint's GPU memory/policy state cleanly), and
aggregates all four (plus, for context, the already-evaluated original 5k pilot from
`train_grounded_grasp_v2_5k.py`, read from its existing
`outputs/eval/safediff_vla_grounded_grasp_v2_5k/summary.json` if present -- not re-run) into one
`comparison.json`.

Purpose: the 5k pilot was the first design in this line of experiments where closed-loop target
grounding and task success improved together (2/3 seeds succeeded with 100% per-step
target_is_nearest; the third failed with 0%). This sweep checks whether that signal holds,
improves, or collapses as training gets longer -- 20k is NOT assumed to be the best checkpoint
going in.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v2_checkpoint_sweep.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

SWEEP_STEPS = [5000, 10000, 15000, 20000]
SWEEP_TRAIN_DIR = Path("outputs/train/safediff_vla_grounded_grasp_v2_20k")
SWEEP_EVAL_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_20k_sweep")
PILOT_SUMMARY = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5k/summary.json")
SEEDS = [1000, 1001, 1002]
EVAL_SCRIPT = Path(__file__).parent / "eval_grounded_grasp_v2_5k.py"


def run_one_checkpoint(step: int, device: str) -> dict:
    checkpoint = SWEEP_TRAIN_DIR / "checkpoints" / f"{step:06d}" / "pretrained_model"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint} -- did training finish saving it?")
    out_dir = SWEEP_EVAL_DIR / f"step_{step:06d}"
    logger.info("=== evaluating step %d (%s) ===", step, checkpoint)
    subprocess.run(
        [
            "uv", "run", "python", str(EVAL_SCRIPT),
            "--checkpoint", str(checkpoint),
            "--seeds", *[str(s) for s in SEEDS],
            "--device", device,
            "--output-dir", str(out_dir),
        ],
        check=True,
    )
    summary = json.loads((out_dir / "summary.json").read_text())
    summary["step"] = step
    summary["checkpoint"] = str(checkpoint)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, nargs="+", default=SWEEP_STEPS)
    parser.add_argument("--output", default="outputs/eval/safediff_vla_grounded_grasp_v2_20k_sweep/comparison.json")
    args = parser.parse_args()

    per_checkpoint = [run_one_checkpoint(step, args.device) for step in args.steps]

    pilot_5k = None
    if PILOT_SUMMARY.exists():
        pilot_5k = json.loads(PILOT_SUMMARY.read_text())
        pilot_5k["step"] = 5000
        pilot_5k["label"] = "original_5k_pilot (train_grounded_grasp_v2_5k.py -- preserved, not re-run)"

    def row(summary: dict) -> dict:
        agg = summary["aggregate"]
        per_ep = summary["per_episode"]
        return {
            "step": summary["step"],
            "checkpoint": summary["checkpoint"],
            "n_success": agg["n_success"],
            "n_grasp_success": agg["n_grasp_success"],
            "n_wrong_object_grasped_ever": agg["n_wrong_object_grasped_ever"],
            "mean_min_ee_to_card_dist": agg["mean_min_ee_to_card_dist"],
            "n_wrong_object_grounding_at_min_distance": agg["n_wrong_object_grounding_at_min_distance"],
            "n_wrong_object_grounding_majority_of_steps": agg["n_wrong_object_grounding_majority_of_steps"],
            "n_phase_transition_matches_proximity_trigger": agg["n_phase_transition_matches_proximity_trigger"],
            "per_seed": [
                {
                    "seed": e["seed"],
                    "target_card_identity": e["target_card_identity"],
                    "success": e["success"],
                    "grasp_success": e["grasp_success"],
                    "wrong_object_grasped_ever": e["wrong_object_grasped_ever"],
                    "fraction_steps_target_is_nearest": e["fraction_steps_target_is_nearest"],
                    "episode_min_ee_to_card_dist": e["episode_min_ee_to_card_dist"],
                    "predicted_target_dist_to_gt_card_at_proximity_trigger": (
                        e["at_proximity_trigger"]["predicted_target_dist_to_gt_card"] if e["at_proximity_trigger"] else None
                    ),
                    "predicted_target_dist_to_gt_card_at_min_distance": (
                        e["at_min_distance"]["predicted_target_dist_to_gt_card"] if e["at_min_distance"] else None
                    ),
                    "proximity_trigger_step": e["proximity_trigger_step"],
                    "post_grasp_phase_transition_step": e["post_grasp_phase_transition_step"],
                    "original_decoder_close_step": e["original_decoder_close_step"],
                    "video": e["video"],
                }
                for e in per_ep
            ],
        }

    comparison = {
        "task": "select_poker",
        "seeds": SEEDS,
        "sweep_train_dir": str(SWEEP_TRAIN_DIR),
        "rows": ([row(pilot_5k)] if pilot_5k else []) + [row(s) for s in per_checkpoint],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2))
    logger.info("=== wrote %s ===", out_path)
    for r in comparison["rows"]:
        logger.info(
            "step=%s success=%d/3 grasp=%d/3 wrong_obj_ever=%d/3 mean_min_dist=%.4f",
            r["step"], r["n_success"], r["n_grasp_success"], r["n_wrong_object_grasped_ever"], r["mean_min_ee_to_card_dist"],
        )


if __name__ == "__main__":
    main()
