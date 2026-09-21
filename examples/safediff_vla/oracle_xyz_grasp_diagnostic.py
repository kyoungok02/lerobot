#!/usr/bin/env python
"""Causal diagnostic, NOT a deployable method: during the grasp-approach window, override only
the policy's commanded xyz with the *true* target card's own live position (read directly from
physics -- an oracle no real deployment would have access to), while leaving the policy's own
rotation, gripper, and timing completely untouched. If grasp/lift then succeeds where the
unmodified policy failed, spatial (visual) localization was the bottleneck; if it still fails, the
bottleneck lies elsewhere (orientation / gripper timing / grasp execution).

No model/execution/env code is edited -- this script builds its own manual per-step rollout loop
(same primitives `audit_xyz_pipeline_trace.py` already uses: `preprocess_observation` ->
`env_preprocessor` -> `preprocessor` -> `policy.select_action` -> `postprocessor` ->
`env_postprocessor` -> `env.step`) and only swaps `action[:3]` for the oracle target position
*after* the policy/postprocessor have already produced their own action, before it reaches the
env. Everything else in the pipeline (rotation channels 3:6, gripper channel 6, and every
upstream component) is exactly what the checkpoint itself predicts.

Window definition: `[close_step_hint - WINDOW_LEAD, episode_end]`, where `close_step_hint` is that
seed's own *unmodified*-policy first-commanded-close step, already measured earlier this session
(`target_grounding_diagnostic.py` / `place_phase_forensics.py` runs against this same B checkpoint)
-- hardcoded per seed below for exactly the 3 seeds requested, not a general mechanism. Steps
before the window run the policy's own xyz unmodified, so the natural pre-grasp approach shape is
preserved; from the window onward (through episode end, to also cover the lift attempt) xyz is
oracle-locked onto the target's current position every step.

Usage:
    uv run python examples/safediff_vla/oracle_xyz_grasp_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.utils.io_utils import write_video
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
GRIPPER_THRESHOLD = 0.5
WINDOW_LEAD = 20  # steps of oracle-corrected approach before the unmodified policy's own close

# Unmodified-policy first_commanded_close_step for these seeds, already measured this session
# (target_grounding_diagnostic.py / place_phase_forensics.py on this same B checkpoint).
CLOSE_STEP_HINT = {1000: 48, 1001: 69, 1002: 65}


def is_success_from_info(info: dict, num_envs: int) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            v = final_info.get("is_success", [False] * num_envs)
            return bool(v.tolist()[0] if hasattr(v, "tolist") else v)
        if len(final_info) > 0 and isinstance(final_info[0], dict):
            return bool(final_info[0].get("is_success", False))
        return False
    if "is_success" in info:
        v = info["is_success"]
        return bool(v.tolist()[0] if hasattr(v, "tolist") else v)
    return False


def run_oracle_episode(policy: SafeDiffVLAPolicy, device: str, seed: int, videos_dir: Path) -> dict:
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    env_impl: VLABenchEnvImpl = env.envs[0]

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

    close_hint = CLOSE_STEP_HINT[seed]
    window_start = max(1, close_hint - WINDOW_LEAD)

    policy.reset()
    observation, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    target_name = env_impl._env.task.target_entity
    base = env_impl._robot_base_xyz if env_impl._robot_base_xyz is not None else np.zeros(3)

    frames = []
    step_records = []
    success = False
    close_step_actual = None
    dist_at_close = None
    step_ix = 0
    max_steps = 500
    try:
        while step_ix < max_steps:
            frames.append(env_impl.render())

            obs_t = preprocess_observation(observation)
            obs_t["task"] = [env.call("task_description")[0]] if hasattr(env, "call") else [""]
            obs_t = env_preprocessor(obs_t)
            obs_norm = preprocessor(obs_t)

            with torch.no_grad():
                action_norm = policy.select_action(obs_norm)
            action_raw = postprocessor(action_norm)
            action_raw_np = action_raw.clone().cpu().numpy().reshape(-1)
            policy_gripper = float(action_raw_np[GRIPPER_INDEX])

            corrected = step_ix >= window_start
            if corrected:
                physics = env_impl._env.physics
                target_ent = env_impl._env.task.entities[target_name]
                card_world = np.asarray(target_ent.get_xpos(physics), dtype=float)
                action_raw_np[:3] = card_world - base

            action_transition = env_postprocessor(
                {"action": torch.from_numpy(action_raw_np[None, :]).to(action_raw.dtype)}
            )
            action_env_np = action_transition["action"].clone().cpu().numpy().reshape(-1)

            # Pre-step physics readout (for "distance at close" -- the state the closing action
            # was actually issued from).
            physics = env_impl._env.physics
            ee_pos_world = np.asarray(env_impl._env.task.robot.get_end_effector_pos(physics), dtype=float)
            card_pos_world = np.asarray(
                env_impl._env.task.entities[target_name].get_xpos(physics), dtype=float
            )
            ee_to_card_dist = float(np.linalg.norm(ee_pos_world - card_pos_world))

            if close_step_actual is None and policy_gripper <= GRIPPER_THRESHOLD:
                close_step_actual = step_ix + 1
                dist_at_close = ee_to_card_dist

            step_records.append(
                {
                    "step_ix": step_ix,
                    "oracle_corrected": corrected,
                    "policy_gripper": policy_gripper,
                    "ee_to_card_dist": ee_to_card_dist,
                }
            )

            observation, reward, terminated, truncated, info = env.step(action_env_np[None, :])
            step_ix += 1
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                success = is_success_from_info(info, 1)
                break
    finally:
        env.close()

    fps = env_impl.metadata.get("render_fps", 20) if hasattr(env_impl, "metadata") else 20
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_path = videos_dir / f"oracle_seed{seed}.mp4"
    if frames:
        write_video(video_path, frames, fps=fps)

    return {
        "seed": seed,
        "success": success,
        "episode_len": step_ix,
        "window_start_step": window_start,
        "close_step_hint_unmodified_policy": close_hint,
        "close_step_actual_this_run": close_step_actual,
        "ee_to_card_dist_at_close": dist_at_close,
        "video_path": str(video_path),
        "step_records": step_records,
    }


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_oracle_xyz_diagnostic")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    set_seed(args.seeds[0])
    logger.info("=== loading policy from %s (unmodified) ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"
    assert policy.config.execute_horizon == 50
    assert not policy.config.use_temporal_ensembling

    out_dir = Path(args.output_dir)
    results = []
    for seed in args.seeds:
        logger.info(
            "=== seed=%d: oracle xyz correction from step %d onward (close_step_hint=%d) ===",
            seed,
            max(1, CLOSE_STEP_HINT[seed] - WINDOW_LEAD),
            CLOSE_STEP_HINT[seed],
        )
        result = run_oracle_episode(policy, args.device, seed, out_dir / "videos")
        results.append(result)
        logger.info(
            "seed=%d success=%s close_step_actual=%s dist_at_close=%s video=%s",
            seed,
            result["success"],
            result["close_step_actual_this_run"],
            result["ee_to_card_dist_at_close"],
            result["video_path"],
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out_dir / "summary.json")
    logger.info(
        "=== oracle-corrected success: %d/%d ===", sum(1 for r in results if r["success"]), len(results)
    )


if __name__ == "__main__":
    main()
