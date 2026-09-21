#!/usr/bin/env python
"""Execution-layer-only ablation: does discarding the action queue and immediately replanning on
this episode's first gripper open->close event (a completed grasp) let the sin/cos 20k
`temporal_decoder` checkpoint move into the next task phase (transport/place), or does it keep
producing open/approach-phase actions regardless?

No model/loss/checkpoint/dataset/env change -- this is purely
`SafeDiffVLAConfig.replan_on_gripper_close` (`policies/safediff_vla/execution.py`'s
`ActionExecutor`), flipped at eval time. Both conditions below fix
`execute_horizon=action_horizon=50` (a single full open-loop chunk between replans, same as the
existing `execute_horizon_50_ensembling_off.json` baseline) -- unlike the earlier
`execute_horizon=10/5` ablations, where replanning every few steps was frequent enough that the
model kept re-predicting the open/approach phase and never reached the grasp action at all. Here
the *only* extra replan an episode can ever get is the one immediately after its first completed
grasp.

Two conditions, same 3 seeds each (`--seeds`, default 1000 1001 1002 -- intentionally small; this
is a first look, not the full 10-seed eval):
  A. baseline:   replan_on_gripper_close=False (identical to the existing execute_horizon=50 run)
  B. experiment: replan_on_gripper_close=True

For every episode we record: success; the grasp (first open->close) event step; how many actions
were still queued at that moment; robot state immediately before/after the event step; the 5
actions that were going to execute right after the grasp under the *old* plan (for A, these are
also the actions that actually executed, since nothing gets discarded; for B, these are the
discarded plan `ActionExecutor.gripper_events` captured before clearing the queue); the 5 actions
that *actually* executed right after the grasp (for A this is identical to "old plan's next 5"; for
B it's the freshly replanned chunk's own first 5); each split into xyz / rotation(euler) / gripper;
smoothness metrics; and a video.

Usage:
    uv run python examples/safediff_vla/eval_grasp_event_replan.py
    uv run python examples/safediff_vla/eval_grasp_event_replan.py --seeds 1000 1001 1002
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))
from eval_baseline_rollout import mean_abs_delta, probe_video  # noqa: E402  (reuse, don't reimplement)

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.execution import detect_gripper_close_event  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_sincos_20k/checkpoints/020000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}


def split_xyz_rot_gripper(actions: Tensor) -> dict[str, list]:
    """`actions`: `[N, 7]` (pos(3) + euler(3) + gripper(1), VLABench's own convention). Split for
    reporting -- never mixed back into one blob."""
    return {
        "xyz": actions[:, :3].tolist(),
        "rotation_euler_xyz": actions[:, 3:6].tolist(),
        "gripper": actions[:, 6].tolist(),
    }


def run_condition(
    policy: SafeDiffVLAPolicy,
    device: str,
    seeds: list[int],
    condition: str,
    videos_dir: Path,
) -> list[dict]:
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config
    )

    open_th = policy.config.gripper_open_threshold
    close_th = policy.config.gripper_close_threshold
    gripper_ix = policy.config.gripper_action_index
    execute_horizon = policy.config.execute_horizon

    episodes: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        actions_ep = rollout_data[ACTION][env_idx, :ep_len]
        states_ep = rollout_data[OBS_STR][OBS_STATE][env_idx, :ep_len]  # aligned with actions_ep
        success = bool(rollout_data["success"][env_idx, :ep_len].any().item())
        mad, mad2 = mean_abs_delta(actions_ep)

        # Whether *this* rollout actually discarded-and-replanned on the grasp event -- read the
        # policy's own live config rather than assuming a hardcoded condition name, so this
        # callback works for any caller's condition label (e.g. `eval_phase_conditioning.py`'s
        # `"phase_conditioned_replan"`, not just this script's own "experiment").
        replanned = bool(policy.config.replan_on_gripper_close)
        executor_events = list(policy._executor.gripper_events) if replanned else []

        if replanned and executor_events:
            event_step = executor_events[0]["step_index"]
            queue_len_at_event = executor_events[0]["queue_len_at_event"]
            old_remaining = executor_events[0]["old_remaining_actions_after_grasp"]
            # Empty exactly when the grasp action was the chunk's very last -- nothing was left
            # to discard.
            old_plan_next5 = torch.cat(old_remaining, dim=0) if old_remaining else actions_ep[:0]
        else:
            # Baseline (and any experiment episode with no detected grasp): recover the same
            # information from the *executed* action stream, since with execute_horizon=50 nothing
            # was ever discarded -- what was planned is exactly what ran.
            event_step = detect_gripper_close_event(actions_ep[:, gripper_ix], open_th, close_th)
            queue_len_at_event = (
                execute_horizon - 1 - (event_step % execute_horizon) if event_step is not None else None
            )
            old_plan_next5 = (
                actions_ep[event_step + 1 : event_step + 6] if event_step is not None else actions_ep[:0]
            )

        actual_next5 = (
            actions_ep[event_step + 1 : event_step + 6] if event_step is not None else actions_ep[:0]
        )
        pre_state = states_ep[event_step].tolist() if event_step is not None else None
        post_state = (
            states_ep[event_step + 1].tolist() if event_step is not None and event_step + 1 < ep_len else None
        )

        record = {
            "episode_ix": episode_ix,
            "condition": condition,
            "success": success,
            "episode_len": ep_len,
            "grasp_event_step": event_step,
            "queue_len_at_event": queue_len_at_event,
            "pre_grasp_state": pre_state,
            "post_grasp_state": post_state,
            "old_plan_next5_actions_after_grasp": split_xyz_rot_gripper(old_plan_next5),
            "actually_executed_next5_actions_after_grasp": split_xyz_rot_gripper(actual_next5),
            "mean_abs_delta_action": mad,
            "mean_abs_delta2_action": mad2,
        }
        episodes.append(record)
        logger.info(
            "[%s] episode %d (seed %d): success=%s grasp_event_step=%s queue_len_at_event=%s "
            "|da/dt|=%.4f |d2a/dt2|=%.4f",
            condition,
            episode_ix,
            seeds[episode_ix],
            success,
            event_step,
            queue_len_at_event,
            mad,
            mad2,
        )

    try:
        info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=len(seeds),
            max_episodes_rendered=len(seeds),
            videos_dir=videos_dir,
            return_episode_data=True,
            start_seed=seeds[0],
            episode_callback=episode_callback,
        )
    finally:
        env.close()

    video_paths = info.get("video_paths", [])
    for ep in episodes:
        ix = ep["episode_ix"]
        if ix < len(video_paths):
            ep["video_path"] = video_paths[ix]
            first_shape, last_shape, n_frames = probe_video(video_paths[ix])
            logger.info("[%s] episode %d video: %d frames -> %s", condition, ix, n_frames, video_paths[ix])
    return episodes


def main() -> None:
    global CHECKPOINT

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/safediff_vla_grasp_event_replan/comparison.json")
    parser.add_argument("--videos-dir", default="outputs/eval/safediff_vla_grasp_event_replan/videos")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    set_seed(args.seeds[0])

    logger.info("=== loading policy from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    assert policy.config.architecture == "temporal_decoder", policy.config.architecture
    assert policy.config.action_horizon == 50, policy.config.action_horizon

    # Fixed condition for both arms, per the experiment spec: execute_horizon == action_horizon
    # (one full open-loop 50-step chunk between ordinary replans), temporal ensembling off, no
    # subgoal signal (architecture has none). Only `replan_on_gripper_close` differs between arms.
    policy.config.execute_horizon = 50
    policy.config.use_temporal_ensembling = False
    logger.info(
        "condition confirmed: architecture=%s action_horizon=%d execute_horizon=%d "
        "use_temporal_ensembling=%s gripper_action_index=%d open_threshold=%.2f close_threshold=%.2f",
        policy.config.architecture,
        policy.config.action_horizon,
        policy.config.execute_horizon,
        policy.config.use_temporal_ensembling,
        policy.config.gripper_action_index,
        policy.config.gripper_open_threshold,
        policy.config.gripper_close_threshold,
    )

    videos_root = Path(args.videos_dir)
    results = {}
    for condition, replan_flag in (("baseline", False), ("experiment", True)):
        policy.config.replan_on_gripper_close = replan_flag
        logger.info(
            "=== condition=%s replan_on_gripper_close=%s seeds=%s ===", condition, replan_flag, args.seeds
        )
        episodes = run_condition(
            policy, args.device, args.seeds, condition, videos_dir=videos_root / condition
        )
        results[condition] = episodes

    report = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "seeds": args.seeds,
        "condition": {
            "architecture": policy.config.architecture,
            "action_horizon": policy.config.action_horizon,
            "execute_horizon": policy.config.execute_horizon,
            "use_temporal_ensembling": policy.config.use_temporal_ensembling,
            "gripper_action_index": policy.config.gripper_action_index,
            "gripper_open_threshold": policy.config.gripper_open_threshold,
            "gripper_close_threshold": policy.config.gripper_close_threshold,
            "gripper_convention": "higher = more open (VLABench envs/vlabench.py: "
            "finger_qpos = gripper * FINGER_OPEN); 1 = fully open, 0 = fully closed",
        },
        "baseline": results["baseline"],
        "experiment": results["experiment"],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("=== wrote report to %s ===", out_path)

    for ep_a, ep_b in zip(results["baseline"], results["experiment"], strict=True):
        logger.info(
            "seed=%d baseline: success=%s grasp_step=%s | experiment: success=%s grasp_step=%s",
            args.seeds[ep_a["episode_ix"]],
            ep_a["success"],
            ep_a["grasp_event_step"],
            ep_b["success"],
            ep_b["grasp_event_step"],
        )


if __name__ == "__main__":
    main()
