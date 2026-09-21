#!/usr/bin/env python
"""2x2 execution-time ablation isolating `use_phase_conditioning` from `replan_on_gripper_close`
-- the previous `eval_phase_conditioning.py` run had both on together in its "phase" condition,
so any effect couldn't be attributed to phase conditioning specifically vs. immediate replanning
alone. This script separates them into four independent cells, same 5 seeds throughout, same
checkpoint *training budget* (5k steps) on both sides of the phase/no-phase split so step count
isn't a confound either:

    A. sincos_5k  (phase OFF, no phase_embedding at all) / replan OFF  -- plain baseline
    B. sincos_5k  (phase OFF)                            / replan ON  -- replan alone
    C. phase_5k   (phase ON)                             / replan OFF -- phase alone
    D. phase_5k   (phase ON)                             / replan ON  -- both together

Key comparisons (see the module docstring's own analysis at the end of `main()`):
  - B vs D: adding phase conditioning *on top of* replan-on-grasp -- isolates phase embedding's
    marginal effect on the immediate-replan behavior already characterized before.
  - A vs C: pure phase-conditioning effect with no replanning at all (phase conditioning can only
    change the chunk generated at the next *ordinary* execute_horizon=50 boundary, not react
    mid-chunk).

Fixed across all four cells: task=select_poker, action_horizon=execute_horizon=50, temporal
ensembling off, no subgoal signal (`temporal_decoder`, not `temporal_decoder_subgoal`).

IMPORTANT caveat this script's own record makes explicit per episode: the phase transition (and
`replan_on_gripper_close`'s discard) both fire on the gripper action's own open->close signal --
there is no contact/force sensing and no object-in-gripper check. A *failed* grasp attempt (gripper
closes on empty air) flips `phase` to 1 / triggers a discard exactly the same as a real one --
`grasp_may_be_failed` below flags every episode where this happened but the episode never
succeeded, so this is visible in the record rather than silently assumed away.

Usage:
    uv run python examples/safediff_vla/eval_phase_replan_2x2.py
    uv run python examples/safediff_vla/eval_phase_replan_2x2.py --seeds 1000 1001 1002 1003 1004
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

SINCOS_5K_CHECKPOINT = (
    "outputs/train/safediff_vla_temporal_decoder_sincos_5k/checkpoints/005000/pretrained_model"
)
PHASE_5K_CHECKPOINT = (
    "outputs/train/safediff_vla_temporal_decoder_phase_5k/checkpoints/005000/pretrained_model"
)
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
DISPLACEMENT_WINDOW = 20
GRIPPER_STAY_CLOSED_WINDOW = 20

CELLS = [
    # (label, checkpoint, use_phase_conditioning, replan_on_gripper_close)
    ("A_sincos5k_phaseOFF_replanOFF", SINCOS_5K_CHECKPOINT, False, False),
    ("B_sincos5k_phaseOFF_replanON", SINCOS_5K_CHECKPOINT, False, True),
    ("C_phase5k_phaseON_replanOFF", PHASE_5K_CHECKPOINT, True, False),
    ("D_phase5k_phaseON_replanON", PHASE_5K_CHECKPOINT, True, True),
]


def run_cell(
    policy: SafeDiffVLAPolicy, checkpoint: str, device: str, seeds: list[int], label: str, videos_dir: Path
) -> list[dict]:
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
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
    replan_enabled = bool(policy.config.replan_on_gripper_close)

    episodes: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        actions_ep = rollout_data[ACTION][env_idx, :ep_len]
        states_ep: Tensor = rollout_data[OBS_STR][OBS_STATE][env_idx, :ep_len]  # aligned w/ actions_ep
        success = bool(rollout_data["success"][env_idx, :ep_len].any().item())
        mad, mad2 = mean_abs_delta(actions_ep)

        executor_events = list(policy._executor.gripper_events) if replan_enabled else []
        if replan_enabled and executor_events:
            event_step = executor_events[0]["step_index"]
            queue_len_at_event = executor_events[0]["queue_len_at_event"]
            discarded = queue_len_at_event > 0
        else:
            event_step = detect_gripper_close_event(actions_ep[:, gripper_ix], open_th, close_th)
            queue_len_at_event = (
                execute_horizon - 1 - (event_step % execute_horizon) if event_step is not None else None
            )
            discarded = False  # replan disabled (or no grasp detected) -- nothing ever discarded

        xyz_displacement = None
        xyz_displacement_norm = None
        gripper_stays_closed = None
        grasp_may_be_failed = None
        if event_step is not None:
            end_ix = min(event_step + DISPLACEMENT_WINDOW, ep_len - 1)
            start_xyz = states_ep[event_step, :3]
            end_xyz = states_ep[end_ix, :3]
            delta = end_xyz - start_xyz
            xyz_displacement = delta.tolist()
            xyz_displacement_norm = float(delta.norm().item())

            window_end = min(event_step + 1 + GRIPPER_STAY_CLOSED_WINDOW, ep_len)
            gripper_window = actions_ep[event_step + 1 : window_end, gripper_ix]
            # "Stays closed": never confidently re-opens (>= open_threshold) in the window --
            # matches the same hysteresis convention as event detection itself, not a new one.
            gripper_stays_closed = (
                bool((gripper_window < open_th).all().item()) if gripper_window.numel() > 0 else None
            )
            # The gripper-close event is a pure action-channel signal (VLABench exposes no
            # contact/force sensing) -- it fires whether or not anything was actually grasped, so
            # `phase` can switch to 1 (and, when replan is on, a discard can fire) on a *failed*
            # grasp attempt too. Flag every such episode explicitly rather than assuming success.
            grasp_may_be_failed = not success

        record = {
            "episode_ix": episode_ix,
            "cell": label,
            "success": success,
            "episode_len": ep_len,
            "first_gripper_close_step": event_step,
            "queue_discarded": discarded,
            "queue_len_at_event": queue_len_at_event,
            f"xyz_displacement_{DISPLACEMENT_WINDOW}step": xyz_displacement,
            f"xyz_displacement_{DISPLACEMENT_WINDOW}step_norm": xyz_displacement_norm,
            f"gripper_stays_closed_{GRIPPER_STAY_CLOSED_WINDOW}step": gripper_stays_closed,
            "grasp_may_be_failed_but_phase_or_discard_still_triggered": grasp_may_be_failed,
            "mean_abs_delta_action": mad,
            "mean_abs_delta2_action": mad2,
        }
        episodes.append(record)
        logger.info(
            "[%s] episode %d (seed %d): success=%s close_step=%s discarded=%s "
            "displacement_norm_%dstep=%s stays_closed_%dstep=%s grasp_may_be_failed=%s",
            label,
            episode_ix,
            seeds[episode_ix],
            success,
            event_step,
            discarded,
            DISPLACEMENT_WINDOW,
            None if xyz_displacement_norm is None else round(xyz_displacement_norm, 4),
            GRIPPER_STAY_CLOSED_WINDOW,
            gripper_stays_closed,
            grasp_may_be_failed,
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
            _, _, n_frames = probe_video(video_paths[ix])
            logger.info("[%s] episode %d video: %d frames -> %s", label, ix, n_frames, video_paths[ix])
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/safediff_vla_phase_replan_2x2/comparison.json")
    parser.add_argument("--videos-dir", default="outputs/eval/safediff_vla_phase_replan_2x2/videos")
    args = parser.parse_args()

    set_seed(args.seeds[0])

    results: dict[str, list[dict]] = {}
    loaded_checkpoint = None
    policy = None
    for label, checkpoint, use_phase, replan in CELLS:
        if checkpoint != loaded_checkpoint:
            logger.info("=== loading policy from %s ===", checkpoint)
            policy = SafeDiffVLAPolicy.from_pretrained(checkpoint)
            policy = policy.to(args.device)
            policy.eval()
            loaded_checkpoint = checkpoint
            assert policy.config.architecture == "temporal_decoder", policy.config.architecture
            assert policy.config.action_horizon == 50, policy.config.action_horizon

        policy.config.execute_horizon = 50
        policy.config.use_temporal_ensembling = False
        policy.config.use_phase_conditioning = use_phase
        policy.config.replan_on_gripper_close = replan
        if use_phase:
            assert hasattr(policy.decoder, "phase_embedding"), (
                f"{checkpoint} has no phase_embedding but use_phase=True was requested"
            )
        logger.info(
            "=== cell=%s checkpoint=%s use_phase_conditioning=%s replan_on_gripper_close=%s seeds=%s ===",
            label,
            checkpoint,
            use_phase,
            replan,
            args.seeds,
        )
        episodes = run_cell(
            policy, checkpoint, args.device, args.seeds, label, videos_dir=Path(args.videos_dir) / label
        )
        results[label] = episodes

    report = {
        "task": TASK,
        "seeds": args.seeds,
        "fixed_condition": {
            "action_horizon": 50,
            "execute_horizon": 50,
            "use_temporal_ensembling": False,
            "architecture": "temporal_decoder",
        },
        "gripper_close_event_caveat": (
            "The gripper open->close event driving both `phase` and `replan_on_gripper_close`'s "
            "discard is a pure action-channel signal (VLABench exposes no contact/force sensing) "
            "-- it does NOT mean the grasp succeeded. See each episode's "
            "'grasp_may_be_failed_but_phase_or_discard_still_triggered' field."
        ),
        "cells": {
            label: {"checkpoint": ckpt, "use_phase_conditioning": p, "replan_on_gripper_close": r}
            for label, ckpt, p, r in CELLS
        },
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("=== wrote report to %s ===", out_path)

    logger.info("=== summary ===")
    for label, _, _, _ in CELLS:
        eps = results[label]
        n_success = sum(1 for e in eps if e["success"])
        n_stays_closed = sum(
            1 for e in eps if e.get(f"gripper_stays_closed_{GRIPPER_STAY_CLOSED_WINDOW}step")
        )
        n_with_close_event = sum(1 for e in eps if e["first_gripper_close_step"] is not None)
        logger.info(
            "%s: success=%d/%d gripper_stays_closed_%dstep=%d/%d (of %d episodes with a close event)",
            label,
            n_success,
            len(eps),
            GRIPPER_STAY_CLOSED_WINDOW,
            n_stays_closed,
            n_with_close_event,
            n_with_close_event,
        )


if __name__ == "__main__":
    main()
