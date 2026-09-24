#!/usr/bin/env python
"""Closed-loop rollout evaluation of the canonical `architecture="temporal_decoder"` (sin/cos +
padfix, no grounded-grasp machinery of any kind) checkpoint, using the SAME physics
instrumentation and per-episode metrics as `eval_grounded_grasp_v2_5k.py`, for a direct comparison
against `temporal_decoder_grounded_grasp_v2` on identical held-out rollout seeds. No
architecture/loss/execution code is touched by this script.

This architecture has no `TargetPointHead`, no reactive close, no grasp-phase conditioning -- "the
predicted target" is read off the decoder's own predicted action-chunk trajectory instead (the
first step the postprocessed gripper channel crosses closed, falling back to the chunk's last-step
xyz if it never closes), exactly `instruction_counterfactual_audit.py`'s convention for a
checkpoint with no explicit target head. This is captured once per fresh chunk plan (an
instance-level wrap of `plan_action_chunk`, read-only -- observes, does not change, its output)
and held constant for every step until the next chunk is planned, mirroring how
`temporal_decoder_grounded_grasp_v2`'s reactive-close mechanism caches its own last prediction.

Fields with no equivalent for this architecture (proximity_trigger_step, post_grasp_phase_
transition_step) are always None -- reported explicitly, not omitted, so the comparison script can
tell "not applicable" apart from "did not occur".

Default checkpoint is `safediff_vla_temporal_decoder_padfix_20k` -- the canonical sin/cos+padfix
`temporal_decoder` 20k run already used as the reference baseline elsewhere in this codebase (e.g.
`instruction_counterfactual_audit.py`). NOT `safediff_vla_temporal_decoder_v2_baseline_20k`, despite
its name: that checkpoint's `decoder.state_encoder`/`action_head` are shaped for the raw 7-D
layout (pre-sin/cos-encoding), so loading it into the current `TemporalActionDecoder`
(ENCODED_DIM=10) fails a state_dict shape check outright -- confirmed by actually trying.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_temporal_decoder_baseline_holdout.py \
        --seeds 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009
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
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.scripts.lerobot_eval import eval_policy
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
GRIPPER_THRESHOLD = 0.5
SEEDS = list(range(2000, 2010))


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def all_card_positions(env: VLABenchEnvImpl) -> dict[str, list[float]]:
    physics = env._env.physics
    task = env._env.task
    out = {}
    for name, ent in task.entities.items():
        if type(ent).__name__ != "Poker":
            continue
        try:
            out[name] = np.asarray(ent.get_xpos(physics), dtype=float).tolist()
        except Exception:  # noqa: BLE001 - best-effort diagnostic only
            pass
    return out


def extract_physics_record(env: VLABenchEnvImpl, step_ix: int) -> dict[str, Any]:
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target_name = task.target_entity
    # WORLD frame throughout (matches `all_card_positions`'s own `ent.get_xpos(physics)`) -- the
    # policy's own cached predicted target is converted from robot-base-relative to WORLD via
    # `+base` at the one place it's compared against these (`merge_policy_log`), not here.
    ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
    cards = all_card_positions(env)
    card_pos = np.array(cards.get(target_name, [np.nan, np.nan, np.nan]))
    try:
        is_grasped = bool(task.entities[target_name].is_grasped(physics, robot))
    except Exception:  # noqa: BLE001 - best-effort diagnostic only
        is_grasped = False
    other_card_grasped = False
    for name, ent in task.entities.items():
        if name == target_name or type(ent).__name__ != "Poker":
            continue
        try:
            if ent.is_grasped(physics, robot):
                other_card_grasped = True
        except Exception:  # noqa: BLE001
            pass
    return {
        "step_ix": step_ix,
        "ee_pos": ee_pos.tolist(),
        "target_entity": target_name,
        "card_pos": card_pos.tolist(),
        "all_card_positions": cards,
        "is_grasped": is_grasped,
        "wrong_object_grasped_this_step": other_card_grasped,
        "ee_to_card_dist": float(np.linalg.norm(ee_pos - card_pos)),
    }


def install_policy_instrumentation(policy: SafeDiffVLAPolicy, postprocessor) -> dict:
    """Instance-level wrap (not a class patch) of this one policy's own `plan_action_chunk` --
    observes (doesn't change) its output. Captures, once per fresh chunk plan: the postprocessed
    (physical-units) gripper-crossing xyz (the "predicted target" proxy for an architecture with
    no explicit target head), robot-base-relative (converted to WORLD frame later, in
    `merge_policy_log`, exactly like `_v2_last_target_xyz`'s cached value)."""
    orig = SafeDiffVLAPolicy.plan_action_chunk
    log: list[dict] = []

    def wrapped(self, batch):
        actions, metrics = orig(self, batch)
        with torch.no_grad():
            actions_physical = postprocessor(actions[0].clone()).cpu().numpy()  # [H, 7], physical units
        gripper = actions_physical[:, GRIPPER_INDEX_RAW]
        closed = gripper <= GRIPPER_THRESHOLD
        close_idx = int(np.argmax(closed)) if closed.any() else None
        close_xyz = actions_physical[close_idx, :3] if close_idx is not None else actions_physical[-1, :3]
        log.append({"predicted_target_robot_frame": close_xyz.tolist(), "chunk_close_idx": close_idx})
        return actions, metrics

    policy.plan_action_chunk = wrapped.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log, "_original": orig}


def uninstall_policy_instrumentation(policy: SafeDiffVLAPolicy) -> None:
    if "plan_action_chunk" in policy.__dict__:
        del policy.__dict__["plan_action_chunk"]


def install_env_instrumentation(policy_log: list[dict]) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def merge_policy_log(record: dict, base: np.ndarray) -> None:
        if not policy_log:
            record["chunk_close_idx"] = None
            record["predicted_target_world"] = None
            record["predicted_target_dist_to_gt_card"] = None
            record["ee_to_predicted_target_dist"] = None
            record["nearest_card_to_predicted_target"] = None
            record["target_is_nearest"] = None
            return
        p = policy_log[-1]
        record["chunk_close_idx"] = p["chunk_close_idx"]
        predicted_target_world = np.asarray(p["predicted_target_robot_frame"], dtype=float) + base
        card_pos = np.asarray(record["card_pos"], dtype=float)
        ee_pos = np.asarray(record["ee_pos"], dtype=float)
        record["predicted_target_world"] = predicted_target_world.tolist()
        record["predicted_target_dist_to_gt_card"] = float(np.linalg.norm(predicted_target_world - card_pos))
        record["ee_to_predicted_target_dist"] = float(np.linalg.norm(ee_pos - predicted_target_world))
        dists = {
            name: float(np.linalg.norm(predicted_target_world - np.asarray(pos, dtype=float)))
            for name, pos in record["all_card_positions"].items()
        }
        nearest = min(dists, key=dists.get) if dists else None
        record["nearest_card_to_predicted_target"] = nearest
        record["target_is_nearest"] = nearest == record["target_entity"]

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            record = extract_physics_record(self, step_ix=0)
            merge_policy_log(record, robot_base(self))
            state["current_episode_steps"] = [record]
        return result

    def patched_step(self, action):
        result = orig_step(self, action)
        terminated = result[2]
        if not terminated:
            step_ix = len(state["current_episode_steps"])
            record = extract_physics_record(self, step_ix=step_ix)
            merge_policy_log(record, robot_base(self))
            state["current_episode_steps"].append(record)
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_env_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def summarize_episode(steps: list[dict], seed: int, success: bool, episode_len: int, video: Path) -> dict:
    grasp_success = any(s["is_grasped"] for s in steps)
    wrong_object_grasped_ever = any(s.get("wrong_object_grasped_this_step") for s in steps)

    orig_close_step = next((s["step_ix"] for s in steps if s.get("chunk_close_idx") is not None), None)

    min_idx = min(range(len(steps)), key=lambda i: steps[i]["ee_to_card_dist"])
    min_step = steps[min_idx]

    def ref(step_ix: int | None) -> dict | None:
        if step_ix is None:
            return None
        s = next((x for x in steps if x["step_ix"] == step_ix), None)
        if s is None:
            return None
        return {
            "step_ix": step_ix,
            "predicted_target_world": s.get("predicted_target_world"),
            "actual_target_card_world": s["card_pos"],
            "predicted_target_dist_to_gt_card": s.get("predicted_target_dist_to_gt_card"),
            "nearest_card_to_predicted_target": s.get("nearest_card_to_predicted_target"),
            "target_is_nearest": s.get("target_is_nearest"),
            "ee_to_card_dist": s["ee_to_card_dist"],
        }

    target_is_nearest_values = [s["target_is_nearest"] for s in steps if s.get("target_is_nearest") is not None]
    frac_target_is_nearest = (
        sum(target_is_nearest_values) / len(target_is_nearest_values) if target_is_nearest_values else None
    )
    target_dist_values = sorted(
        s["predicted_target_dist_to_gt_card"] for s in steps if s.get("predicted_target_dist_to_gt_card") is not None
    )
    mean_target_dist = sum(target_dist_values) / len(target_dist_values) if target_dist_values else None
    median_target_dist = (
        target_dist_values[len(target_dist_values) // 2]
        if target_dist_values and len(target_dist_values) % 2 == 1
        else (
            (target_dist_values[len(target_dist_values) // 2 - 1] + target_dist_values[len(target_dist_values) // 2]) / 2
            if target_dist_values
            else None
        )
    )

    return {
        "seed": seed,
        "success": success,
        "grasp_success": grasp_success,
        "wrong_object_grasped_ever": wrong_object_grasped_ever,
        "target_card_identity": steps[0]["target_entity"],
        "episode_len": episode_len,
        "original_decoder_close_step": orig_close_step,
        # No equivalent for this architecture (no reactive close / no phase) -- reported
        # explicitly as None rather than omitted, so the comparison script can tell
        # "not applicable" apart from "did not occur".
        "proximity_trigger_step": None,
        "post_grasp_phase_transition_step": None,
        "episode_min_ee_to_card_dist": min_step["ee_to_card_dist"],
        "episode_min_dist_step_ix": min_step["step_ix"],
        "at_first_chunk": ref(1),
        "at_min_distance": ref(min_step["step_ix"]),
        "fraction_steps_target_is_nearest": frac_target_is_nearest,
        "wrong_object_grounding_at_min_distance": (
            (not min_step.get("target_is_nearest")) if min_step.get("target_is_nearest") is not None else None
        ),
        "wrong_object_grounding_majority_of_steps": (
            frac_target_is_nearest < 0.5 if frac_target_is_nearest is not None else None
        ),
        "mean_predicted_target_dist_to_gt_card": mean_target_dist,
        "median_predicted_target_dist_to_gt_card": median_target_dist,
        "video": str(video),
    }


def run(policy: SafeDiffVLAPolicy, device: str, seeds: list[int], out_dir: Path, checkpoint: str) -> None:
    assert policy.config.architecture == "temporal_decoder"
    assert policy.config.use_temporal_ensembling is False
    set_seed(seeds[0])

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
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    policy_inst = install_policy_instrumentation(policy, postprocessor)
    env_state = install_env_instrumentation(policy_inst["log"])
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(env_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps})
        policy_inst["log"].clear()  # per-episode: episode boundary == policy.reset() call inside rollout()
        logger.info("  episode %d len=%d n_steps=%d", episode_ix, ep_len, len(steps))

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
            start_seed=seeds[0],
            episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)
        uninstall_policy_instrumentation(policy)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))

    per_episode_summary = [
        summarize_episode(rec["steps"], rec["seed"], rec["success"], rec["episode_len"], videos_dir / f"eval_episode_{rec['episode_ix']}.mp4")
        for rec in episode_records
    ]

    summary = {
        "checkpoint": checkpoint,
        "task": TASK,
        "architecture": "temporal_decoder",
        "n_episodes": len(seeds),
        "seeds": seeds,
        "per_episode": per_episode_summary,
        "aggregate": {
            "n_success": sum(1 for e in per_episode_summary if e["success"]),
            "n_grasp_success": sum(1 for e in per_episode_summary if e["grasp_success"]),
            "n_wrong_object_grasped_ever": sum(1 for e in per_episode_summary if e["wrong_object_grasped_ever"]),
            "mean_min_ee_to_card_dist": sum(e["episode_min_ee_to_card_dist"] for e in per_episode_summary) / len(per_episode_summary),
            "n_wrong_object_grounding_at_min_distance": sum(
                1 for e in per_episode_summary if e["wrong_object_grounding_at_min_distance"]
            ),
            "n_wrong_object_grounding_majority_of_steps": sum(
                1 for e in per_episode_summary if e["wrong_object_grounding_majority_of_steps"]
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    logger.info("aggregate: %s", json.dumps(summary["aggregate"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_temporal_decoder_baseline_holdout")
    args = parser.parse_args()

    logger.info("=== loading temporal_decoder checkpoint from %s ===", args.checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()

    run(policy, args.device, args.seeds, Path(args.output_dir), args.checkpoint)


if __name__ == "__main__":
    main()
