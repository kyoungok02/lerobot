#!/usr/bin/env python
"""Closed-loop rollout evaluation of the fresh `train_grounded_grasp_v2_5k.py` checkpoint
(`temporal_decoder_grounded_grasp_v2`) on `select_poker` seeds 1000-1002. Self-contained: reads
live MuJoCo physics via a non-invasive `VLABenchEnv.step`/`.reset` monkeypatch (same pattern as
every prior rollout script this session), plus an instance-level wrap of this one policy object's
own `_grounded_grasp_v2_reactive_close` (observes, does not change, its already-implemented
proximity-trigger/phase-flip/one-shot-guard behavior and the decoder's raw pre-override gripper
channel). No model/loss/execution-policy code is touched by this script.

Per episode, records:
  - task success (the env's own success condition, via `eval_policy`).
  - grasp/contact success (`entity.is_grasped(physics, robot)` on the REAL target card -- ground
    truth, independent of the model).
  - target card identity, predicted target xyz, actual target card xyz, predicted<->GT distance,
    nearest-card-to-predicted-target identity, and target_is_nearest -- each at three reference
    steps: the first chunk, the proximity-close-trigger step, and the step of minimum EE<->target
    distance (mirrors the fields the earlier `eval_grounded_grasp_5k.py`/`grasp_approach_precision_
    *.py` diagnostics used, for direct comparability).
  - proximity close trigger timestep, the decoder's OWN raw predicted close timestep (before the
    reactive-close override -- "original_decoder_close_step"), and the PRE_GRASP->POST_GRASP phase
    transition timestep (by construction the same step as the proximity trigger, tracked
    independently here to confirm that).
  - min EE<->target-card distance over the whole episode.
  - video path.

"wrong_object_grounding" (comparable to the baseline's reported 6/10): counted BOTH at the episode's
minimum-EE-distance step (arguably the most decision-relevant "closest approach" moment) and as
"nearest-target a minority of steps" (`fraction_steps_target_is_nearest < 0.5`) -- both from the
same underlying `target_is_nearest` per-step signal, reported explicitly so either definition can
be checked against whatever the baseline used.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v2_5k.py
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

CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_5k/checkpoints/005000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_THRESHOLD = 0.5
SEEDS = [1000, 1001, 1002]


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
    # WORLD frame throughout (matches `all_card_positions`'s own `ent.get_xpos(physics)`, which
    # is not base-relative) -- the policy's own cached predicted target is converted from its
    # robot-base-relative frame to WORLD via `+base` at the one place it's compared against these
    # (`merge_policy_log`), not here.
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


def install_policy_instrumentation(policy: SafeDiffVLAPolicy) -> dict:
    """Instance-level wrap (not a class patch) of this one policy's own
    `_grounded_grasp_v2_reactive_close` -- observes (doesn't change) its already-implemented
    proximity-trigger/phase-flip/one-shot-guard behavior, plus the decoder's raw pre-override
    gripper channel."""
    orig = SafeDiffVLAPolicy._grounded_grasp_v2_reactive_close
    log: list[dict] = []

    def wrapped(self, action, current_state):
        original_gripper = float(action[0, GRIPPER_INDEX_RAW].item())
        phase_before = self._v2_grasp_phase
        was_triggered_before = self._v2_close_triggered
        predicted_target_world = None
        if self._v2_last_target_xyz is not None:
            target_pos_m = self._v2_last_target_xyz * self.action_pos_std + self.action_pos_mean
            predicted_target_world = target_pos_m[0].detach().cpu().numpy()  # robot-base frame
        result = orig(self, action, current_state)
        log.append(
            {
                "original_gripper": original_gripper,
                "final_gripper": float(result[0, GRIPPER_INDEX_RAW].item()),
                "phase_before": phase_before,
                "phase_after": self._v2_grasp_phase,
                "close_triggered_after": bool(self._v2_close_triggered),
                "just_triggered_this_step": bool((not was_triggered_before) and self._v2_close_triggered),
                "predicted_target_robot_frame": predicted_target_world.tolist() if predicted_target_world is not None else None,
            }
        )
        return result

    policy._grounded_grasp_v2_reactive_close = wrapped.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log, "_original": orig}


def uninstall_policy_instrumentation(policy: SafeDiffVLAPolicy) -> None:
    if "_grounded_grasp_v2_reactive_close" in policy.__dict__:
        del policy.__dict__["_grounded_grasp_v2_reactive_close"]


def install_env_instrumentation(policy_log: list[dict]) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def merge_policy_log(record: dict, base: np.ndarray) -> None:
        if not policy_log:
            record["original_decoder_gripper"] = None
            record["applied_gripper"] = None
            record["phase_after_this_step"] = 0
            record["close_triggered_after_this_step"] = False
            record["proximity_trigger_fired_this_step"] = False
            record["predicted_target_world"] = None
            record["predicted_target_dist_to_gt_card"] = None
            record["ee_to_predicted_target_dist"] = None
            record["nearest_card_to_predicted_target"] = None
            record["target_is_nearest"] = None
            return
        p = policy_log[-1]
        record["original_decoder_gripper"] = p["original_gripper"]
        record["applied_gripper"] = p["final_gripper"]
        record["phase_after_this_step"] = p["phase_after"]
        record["close_triggered_after_this_step"] = p["close_triggered_after"]
        record["proximity_trigger_fired_this_step"] = p["just_triggered_this_step"]
        # `extract_physics_record`'s `ee_pos`/`card_pos`/`all_card_positions` are WORLD frame
        # (`ent.get_xpos(physics)`, un-shifted). The cached predicted target is in the policy's
        # own robot-BASE-relative frame (un-normalized via `action_pos_std/mean` in
        # `_grounded_grasp_v2_reactive_close`, the same frame `observation.state[:3]`/`action[:3]`
        # use) -- `+base` converts it to WORLD frame so it can be compared against the other two.
        if p["predicted_target_robot_frame"] is not None:
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
        else:
            record["predicted_target_world"] = None
            record["predicted_target_dist_to_gt_card"] = None
            record["ee_to_predicted_target_dist"] = None
            record["nearest_card_to_predicted_target"] = None
            record["target_is_nearest"] = None

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

    orig_close_step = next(
        (s["step_ix"] for s in steps[1:] if s.get("original_decoder_gripper") is not None and s["original_decoder_gripper"] <= GRIPPER_THRESHOLD),
        None,
    )
    proximity_trigger_step = next((s["step_ix"] for s in steps if s.get("proximity_trigger_fired_this_step")), None)
    phase_transition_step = next(
        (s["step_ix"] for s in steps if s.get("phase_after_this_step") == 1 and steps[max(s["step_ix"] - 1, 0)].get("phase_after_this_step") == 0),
        None,
    )

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

    return {
        "seed": seed,
        "success": success,
        "grasp_success": grasp_success,
        "wrong_object_grasped_ever": wrong_object_grasped_ever,
        "target_card_identity": steps[0]["target_entity"],
        "episode_len": episode_len,
        "original_decoder_close_step": orig_close_step,
        "proximity_trigger_step": proximity_trigger_step,
        "post_grasp_phase_transition_step": phase_transition_step,
        "phase_transition_matches_proximity_trigger": phase_transition_step == proximity_trigger_step,
        "episode_min_ee_to_card_dist": min_step["ee_to_card_dist"],
        "episode_min_dist_step_ix": min_step["step_ix"],
        "at_first_chunk": ref(1),
        "at_proximity_trigger": ref(proximity_trigger_step),
        "at_min_distance": ref(min_step["step_ix"]),
        "fraction_steps_target_is_nearest": frac_target_is_nearest,
        "wrong_object_grounding_at_min_distance": (
            (not min_step.get("target_is_nearest")) if min_step.get("target_is_nearest") is not None else None
        ),
        "wrong_object_grounding_majority_of_steps": (
            frac_target_is_nearest < 0.5 if frac_target_is_nearest is not None else None
        ),
        "video": str(video),
    }


def run(policy: SafeDiffVLAPolicy, device: str, seeds: list[int], out_dir: Path) -> None:
    assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2"
    assert policy.config.use_temporal_ensembling is False
    set_seed(seeds[0])

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
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    policy_inst = install_policy_instrumentation(policy)
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
        "checkpoint": CHECKPOINT,
        "task": TASK,
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
            "n_phase_transition_matches_proximity_trigger": sum(
                1 for e in per_episode_summary if e["phase_transition_matches_proximity_trigger"]
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    logger.info("aggregate: %s", json.dumps(summary["aggregate"], indent=2))


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_v2_5k")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    logger.info("=== loading temporal_decoder_grounded_grasp_v2 checkpoint from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    run(policy, args.device, args.seeds, Path(args.output_dir))


if __name__ == "__main__":
    main()
