#!/usr/bin/env python
"""Read-only x-axis bias source audit: traces one rollout episode's action pipeline step by step --

    raw env state (robot-frame) -> normalized state (model input)
        -> model raw output (normalized) -> denormalized action (robot-frame, env input)
        -> resulting EE pose (physics, world-frame, converted back to robot-frame for comparison)

No model/execution/env code is edited. Physics readout uses the exact same
`VLABenchEnv.step`/`.reset` monkeypatch pattern as `place_phase_forensics.py` (call the original,
read extra state after, return the original result unmodified) plus one direct read of the
cached `env._robot_base_xyz` (the same offset `_build_ctrl_from_action`/`_get_obs` already use to
convert between robot-frame (dataset/action/state convention) and world-frame (dm_control/physics)
-- see `envs/vlabench.py`'s own comments, read but not changed here).

Convention audit (from reading, not changing, the relevant code):
  - `envs/vlabench.py::_get_obs`: `observation.state[:3] = pos_world - base` (robot-frame).
  - `envs/vlabench.py::_build_ctrl_from_action`: `pos_world = action[:3] + base` -- i.e. action
    xyz is an ABSOLUTE target position in robot-frame (not a delta), using the exact same `base`
    offset as the state conversion above (same env instance, cached once per episode at
    `_ensure_env()`). So action and state share one frame/convention by construction.
  - `rotation_encoding.py::encode`/`decode`: `xyz = x[..., :3]` copied through unchanged in both
    directions -- the sin/cos rotation encoding never touches the position channels.
  - This checkpoint has `use_backbone_domain_adapter=False` (config.json), so
    `SafeDiffVLAPolicy.domain_adapter` is None -- no adapter-side transform is in play at all.
  - Normalization is plain per-channel MEAN_STD (`policy_preprocessor_step_5_normalizer_processor
    .safetensors`), loaded straight from saved stats, same channel order both directions.

This script adds a fourth, empirical check on top of that: does the model's own denormalized
action-x, at the point closest to the card, actually converge toward the card's own x position
(both compared in the same robot-frame), or is there a residual, and if so is it plausibly a sign
flip / constant offset / frame mismatch, or just imprecision?

Usage:
    uv run python examples/safediff_vla/audit_xyz_pipeline_trace.py --seed 1001
    uv run python examples/safediff_vla/audit_xyz_pipeline_trace.py --seed 1008
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_padfix_20k/checkpoints/020000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_INDEX = 6


def physics_snapshot(env: VLABenchEnvImpl) -> dict[str, Any]:
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target = task.entities[task.target_entity]
    ee_pos_world = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
    card_pos_world = np.asarray(target.get_xpos(physics), dtype=float)
    base = env._robot_base_xyz if env._robot_base_xyz is not None else np.zeros(3)
    return {
        "ee_pos_world": ee_pos_world.tolist(),
        "ee_pos_robot_frame": (ee_pos_world - base).tolist(),
        "card_pos_world": card_pos_world.tolist(),
        "card_pos_robot_frame": (card_pos_world - base).tolist(),
        "robot_base_xyz": base.tolist(),
        "ee_to_card_dist": float(np.linalg.norm(ee_pos_world - card_pos_world)),
    }


def run_trace(policy: SafeDiffVLAPolicy, device: str, seed: int, window: tuple[int, int]) -> list[dict]:
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]  # gymnasium SyncVectorEnv, num_envs=1
    env_impl: VLABenchEnvImpl = env.envs[0]  # the actual single env instance, for physics reads

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    trace: list[dict] = []
    try:
        policy.reset()
        observation, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
        pre = physics_snapshot(env_impl)

        step_ix = 0
        max_steps = window[1] + 5
        while step_ix < max_steps:
            obs_t = preprocess_observation(observation)
            obs_t["task"] = [env.call("task_description")[0]] if hasattr(env, "call") else [""]
            obs_t = env_preprocessor(obs_t)
            obs_norm = preprocessor(obs_t)  # normalized -- fed to the model

            raw_state_robot_frame = obs_t[OBS_STATE].clone().cpu().numpy().reshape(-1).tolist()
            norm_state = obs_norm[OBS_STATE].clone().cpu().numpy().reshape(-1).tolist()

            with torch.no_grad():
                action_norm = policy.select_action(obs_norm)  # normalized action, [1, action_dim]
            action_norm_np = action_norm.clone().cpu().numpy().reshape(-1)

            action_raw = postprocessor(action_norm)  # denormalized -- robot-frame, env's own units
            action_raw_np = action_raw.clone().cpu().numpy().reshape(-1)

            action_transition = env_postprocessor({"action": action_raw})
            action_env_np = action_transition["action"].clone().cpu().numpy().reshape(-1)

            if window[0] <= step_ix <= window[1]:
                trace.append(
                    {
                        "step_ix": step_ix,
                        "pre_step_physics": pre,
                        "raw_state_robot_frame_xyz": raw_state_robot_frame[:3],
                        "normalized_state_xyz": norm_state[:3],
                        "model_raw_output_normalized_xyz": action_norm_np[:3].tolist(),
                        "denormalized_action_robot_frame_xyz": action_raw_np[:3].tolist(),
                        "env_input_action_xyz": action_env_np[:3].tolist(),
                        "commanded_gripper": float(action_env_np[GRIPPER_INDEX]),
                    }
                )

            observation, reward, terminated, truncated, info = env.step(action_env_np[None, :])
            post = physics_snapshot(env_impl)
            if window[0] <= step_ix <= window[1]:
                trace[-1]["post_step_physics"] = post
                trace[-1]["ee_pos_robot_frame_moved_to"] = post["ee_pos_robot_frame"]
                trace[-1]["card_pos_robot_frame_now"] = post["card_pos_robot_frame"]
                trace[-1]["action_minus_actual_ee_xyz_robot_frame"] = [
                    a - e for a, e in zip(action_env_np[:3].tolist(), post["ee_pos_robot_frame"], strict=True)
                ]
                trace[-1]["action_minus_card_xyz_robot_frame"] = [
                    a - c for a, c in zip(action_env_np[:3].tolist(), post["card_pos_robot_frame"], strict=True)
                ]

            pre = post
            step_ix += 1
            if terminated.any() or truncated.any():
                break
    finally:
        env.close()
    return trace


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--window-start", type=int, default=None, help="defaults to seed-appropriate window")
    parser.add_argument("--window-len", type=int, default=15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_xyz_pipeline_trace")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    set_seed(args.seed)
    logger.info("=== loading policy from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    assert not policy.config.use_temporal_ensembling
    assert policy.config.execute_horizon == 50
    assert not hasattr(policy.config, "use_phase_conditioning")
    assert not hasattr(policy.config, "replan_on_gripper_close")

    window_start = args.window_start if args.window_start is not None else max(0, 60 - args.window_len // 2)
    window = (window_start, window_start + args.window_len)
    logger.info("=== tracing seed=%d, step window=%s ===", args.seed, window)
    trace = run_trace(policy, args.device, args.seed, window)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trace_seed{args.seed}.json"
    out_path.write_text(json.dumps(trace, indent=2))
    logger.info("wrote %s", out_path)

    for t in trace:
        logger.info(
            "step=%d raw_state_x=%.4f norm_state_x=%.4f model_out_norm_x=%.4f denorm_action_x=%.4f "
            "ee_after_x=%.4f card_x=%.4f (action-ee)_x=%.4f (action-card)_x=%.4f grip=%.3f",
            t["step_ix"],
            t["raw_state_robot_frame_xyz"][0],
            t["normalized_state_xyz"][0],
            t["model_raw_output_normalized_xyz"][0],
            t["denormalized_action_robot_frame_xyz"][0],
            t["ee_pos_robot_frame_moved_to"][0],
            t["card_pos_robot_frame_now"][0],
            t["action_minus_actual_ee_xyz_robot_frame"][0],
            t["action_minus_card_xyz_robot_frame"][0],
            t["commanded_gripper"],
        )


if __name__ == "__main__":
    main()
