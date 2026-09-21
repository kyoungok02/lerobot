#!/usr/bin/env python
"""Does minimal binary phase conditioning (`SafeDiffVLAConfig.use_phase_conditioning`, see its
docstring in `configuration_safediff_vla.py`) fix what immediate replan-on-grasp alone couldn't?

`eval_grasp_event_replan.py` (the previous experiment, same 3 seeds, same checkpoint family)
already showed: with `replan_on_gripper_close=True` on the plain (non-phase) sin/cos 20k
checkpoint, the freshly replanned chunk's position resets sensibly near the current state, but in
2/3 seeds the gripper channel immediately trends back open right after the grasp -- i.e. immediate
replanning alone wasn't enough to commit to a transport phase. This script re-runs the *same*
`replan_on_gripper_close=True` execution condition, on the *new* `safediff_vla_temporal_decoder_phase_5k`
checkpoint (5k-step quick training run, `use_phase_conditioning=True`, otherwise identical
architecture/hyperparameters -- see `train_phase_conditioning_5k.py`), and reuses
`eval_grasp_event_replan.py`'s own `run_condition()` unchanged so the recorded fields (grasp event
step, queue length at event, pre/post-grasp state, old-plan-vs-actually-executed next 5 actions
split xyz/rotation/gripper, smoothness, video) are directly comparable to that run's own
`experiment` condition, already saved at
`outputs/eval/safediff_vla_grasp_event_replan/comparison_seeds1000-1002.json`.

Fixed condition (unchanged from the prior experiment): execute_horizon=action_horizon=50,
temporal ensembling off, replan_on_gripper_close=True. No architecture/loss/dataset/env change
from what `safediff_vla_temporal_decoder_phase_5k` was trained with -- this script only exercises
execution-time behavior on an already-trained checkpoint.

Usage:
    uv run python examples/safediff_vla/eval_phase_conditioning.py
    uv run python examples/safediff_vla/eval_phase_conditioning.py --seeds 1000 1001 1002
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_grasp_event_replan as ger  # noqa: E402  (reuse run_condition, don't reimplement)

from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

PHASE_CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_phase_5k/checkpoints/005000/pretrained_model"
PRIOR_EXPERIMENT_REPORT = "outputs/eval/safediff_vla_grasp_event_replan/comparison_seeds1000-1002.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=PHASE_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/safediff_vla_phase_conditioning/comparison.json")
    parser.add_argument("--videos-dir", default="outputs/eval/safediff_vla_phase_conditioning/videos")
    parser.add_argument("--prior-experiment-report", default=PRIOR_EXPERIMENT_REPORT)
    args = parser.parse_args()
    ger.CHECKPOINT = args.checkpoint

    set_seed(args.seeds[0])

    logger.info("=== loading policy from %s ===", args.checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()

    assert policy.config.architecture == "temporal_decoder", policy.config.architecture
    assert policy.config.action_horizon == 50, policy.config.action_horizon
    assert policy.config.use_phase_conditioning, "this checkpoint was not trained with phase conditioning"

    # Same execution condition as the prior (non-phase) `replan_on_gripper_close=True` experiment,
    # for a direct, apples-to-apples comparison -- only the checkpoint (and thus phase
    # conditioning) differs.
    policy.config.execute_horizon = 50
    policy.config.use_temporal_ensembling = False
    policy.config.replan_on_gripper_close = True
    logger.info(
        "condition confirmed: architecture=%s action_horizon=%d execute_horizon=%d "
        "use_temporal_ensembling=%s replan_on_gripper_close=%s use_phase_conditioning=%s",
        policy.config.architecture,
        policy.config.action_horizon,
        policy.config.execute_horizon,
        policy.config.use_temporal_ensembling,
        policy.config.replan_on_gripper_close,
        policy.config.use_phase_conditioning,
    )

    episodes = ger.run_condition(
        policy, args.device, args.seeds, "phase_conditioned_replan", videos_dir=Path(args.videos_dir)
    )

    prior_experiment = None
    prior_path = Path(args.prior_experiment_report)
    if prior_path.exists():
        prior_experiment = json.loads(prior_path.read_text())["experiment"]
    else:
        logger.warning("prior experiment report not found at %s -- comparison table will be phase-only", prior_path)

    report = {
        "checkpoint": args.checkpoint,
        "task": ger.TASK,
        "seeds": args.seeds,
        "condition": {
            "architecture": policy.config.architecture,
            "action_horizon": policy.config.action_horizon,
            "execute_horizon": policy.config.execute_horizon,
            "use_temporal_ensembling": policy.config.use_temporal_ensembling,
            "replan_on_gripper_close": policy.config.replan_on_gripper_close,
            "use_phase_conditioning": policy.config.use_phase_conditioning,
        },
        "phase_conditioned_replan": episodes,
        "prior_experiment_no_phase_replan": prior_experiment,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("=== wrote report to %s ===", out_path)

    for ep in episodes:
        seed = args.seeds[ep["episode_ix"]]
        prior_ep = prior_experiment[ep["episode_ix"]] if prior_experiment else None
        gripper_after = ep["actually_executed_next5_actions_after_grasp"]["gripper"]
        stays_closed = bool(gripper_after) and max(gripper_after) < 0.5
        logger.info(
            "seed=%d success=%s grasp_step=%s gripper_stays_closed_after_replan=%s "
            "(prior no-phase run: success=%s grasp_step=%s)",
            seed,
            ep["success"],
            ep["grasp_event_step"],
            stays_closed,
            prior_ep["success"] if prior_ep else "n/a",
            prior_ep["grasp_event_step"] if prior_ep else "n/a",
        )


if __name__ == "__main__":
    main()
