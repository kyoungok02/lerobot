#!/usr/bin/env python
"""Offline/rollout diagnostic: does the policy's commanded grasp-approach xyz miss the *target*
card because it's imprecisely localized (still nearest to the right card, just not close enough),
because it's grounded on the *wrong* card/object entirely, or because it isn't targeting any
specific object at all?

Read-only: no model/execution/env code is edited. Uses the exact same `VLABenchEnv.step`/`.reset`
monkeypatch pattern as `place_phase_forensics.py` (call the original, read extra state after,
return the original result unmodified), extended to record every `Poker`-class entity's own xyz
and identity (not just the target's) at every step, plus the exact `commanded xyz` already sent to
`env.step()` (the executed, denormalized, robot-frame action -- same convention audited end-to-end
in `audit_xyz_pipeline_trace.py`, which showed the action->physics chain itself has no sign
inversion / offset / frame bug in x for seed=1001 and seed=1008).

At each failed episode's first commanded-close step (same definition as
`place_phase_forensics.py`'s `first_commanded_close_step`: first index the gripper action crosses
from open to closed), records: commanded xyz, the true target card's xyz, every card/object's xyz
+ identity, distance from commanded xyz to the target and to every distractor, which object is
*nearest* to commanded xyz, whether that nearest object is the target, and the nearest distractor's
distance.

Failure classification (thresholds are a deliberate, documented choice -- not derived from any
existing repo constant, since none exists for this quantity):
  - `GRASP_TOLERANCE = 0.05m`: close enough that a real grasp attempt would plausibly succeed (the
    one seed-1001 near-miss already on record here closed at 4.8cm; the A/B success episodes on
    record closed at 2-4cm).
  - `PROXIMITY_RADIUS = 0.10m`: closer than roughly half the typical spacing between adjacent
    card holders on this table (~12-21cm in the scenes sampled here) -- "plausibly grounded on
    *some* object" cutoff.
  - `wrong_object_grounding`: nearest object is a distractor (not the target) and within
    `PROXIMITY_RADIUS`.
  - `target_localization_error`: nearest object is the target, but the distance to it exceeds
    `GRASP_TOLERANCE`.
  - `free_space_or_average_target`: nearest object (target or distractor) is beyond
    `PROXIMITY_RADIUS` -- commanded xyz isn't grounded on anything in particular.
  - `within_grasp_tolerance_of_target`: not a failure of grounding at all (nearest object is the
    target, within `GRASP_TOLERANCE`) -- if the episode still failed, the bottleneck is elsewhere
    (orientation / gripper timing / grasp execution), flagged explicitly rather than mislabeled.

seed=1001 and seed=1008 additionally get their *full* per-step trace (every recorded physics step,
not just the close-moment snapshot) written to their own file, per request.

Usage:
    uv run python examples/safediff_vla/target_grounding_diagnostic.py
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
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import ACTION
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
GRASP_TOLERANCE = 0.05  # meters -- see module docstring
PROXIMITY_RADIUS = 0.10  # meters -- see module docstring
DETAILED_TRACE_SEEDS = (1001, 1008)


def extract_scene_record(env: VLABenchEnvImpl, step_ix: int) -> dict[str, Any]:
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target_name = task.target_entity
    base = env._robot_base_xyz if env._robot_base_xyz is not None else np.zeros(3, dtype=float)

    ee_pos_world = np.asarray(robot.get_end_effector_pos(physics), dtype=float)

    objects: dict[str, Any] = {}
    for name, ent in task.entities.items():
        if type(ent).__name__ != "Poker":  # cards only -- excludes table/card_holder infrastructure
            continue
        pos_world = np.asarray(ent.get_xpos(physics), dtype=float)
        objects[name] = {
            "xyz_world": pos_world.tolist(),
            "xyz_robot_frame": (pos_world - base).tolist(),
            "is_target": name == target_name,
        }

    return {
        "step_ix": step_ix,
        "target_entity": target_name,
        "ee_pos_world": ee_pos_world.tolist(),
        "ee_pos_robot_frame": (ee_pos_world - base).tolist(),
        "robot_base_xyz": base.tolist(),
        "objects": objects,
    }


def install_instrumentation() -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": [], "step_ix": 0}

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            state["current_episode_steps"] = [extract_scene_record(self, step_ix=0)]
            state["step_ix"] = 0
        return result

    def patched_step(self, action):
        result = orig_step(self, action)
        terminated = result[2]
        state["step_ix"] += 1
        if not terminated:
            # Success-episode limitation identical to place_phase_forensics.py's (self._env is
            # already rebuilt for the next episode by the time control returns on the terminal
            # step) -- documented there, not repeated in depth here.
            state["current_episode_steps"].append(extract_scene_record(self, step_ix=state["step_ix"]))
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def classify_grounding(
    dist_to_target: float, nearest_name: str, nearest_dist: float, target_name: str
) -> str:
    if nearest_dist > PROXIMITY_RADIUS:
        return "free_space_or_average_target"
    if nearest_name != target_name:
        return "wrong_object_grounding"
    if dist_to_target > GRASP_TOLERANCE:
        return "target_localization_error"
    return "within_grasp_tolerance_of_target"


def analyze_episode(rec: dict, executed_action: torch.Tensor) -> dict:
    steps = rec["steps"]
    gripper = executed_action[:, GRIPPER_INDEX]
    commanded_binary = (gripper > GRIPPER_THRESHOLD).tolist()
    close_step = (commanded_binary.index(False) + 1) if False in commanded_binary else None

    if close_step is None:
        return {"first_commanded_close_step": None, "note": "gripper never commanded closed"}

    action_ix = close_step - 1  # 0-indexed into executed_action
    if action_ix >= executed_action.shape[0]:
        return {"first_commanded_close_step": close_step, "note": "close step beyond recorded episode length"}
    commanded_xyz = executed_action[action_ix, :3].tolist()

    scene = next((s for s in steps if s["step_ix"] == close_step), None)
    if scene is None:
        return {"first_commanded_close_step": close_step, "note": "no physics snapshot at close step"}

    target_name = scene["target_entity"]
    distances = {
        name: float(np.linalg.norm(np.array(commanded_xyz) - np.array(info["xyz_robot_frame"])))
        for name, info in scene["objects"].items()
    }
    dist_to_target = distances.get(target_name)
    nearest_name = min(distances, key=distances.get)
    nearest_dist = distances[nearest_name]
    distractor_distances = {n: d for n, d in distances.items() if n != target_name}
    nearest_distractor_name = (
        min(distractor_distances, key=distractor_distances.get) if distractor_distances else None
    )
    nearest_distractor_dist = (
        distractor_distances.get(nearest_distractor_name) if nearest_distractor_name else None
    )

    return {
        "first_commanded_close_step": close_step,
        "commanded_xyz_robot_frame": commanded_xyz,
        "target_entity": target_name,
        "target_xyz_robot_frame": scene["objects"][target_name]["xyz_robot_frame"],
        "all_objects_xyz_robot_frame": {n: info["xyz_robot_frame"] for n, info in scene["objects"].items()},
        "dist_commanded_to_target": dist_to_target,
        "dist_commanded_to_each_distractor": distractor_distances,
        "nearest_object": nearest_name,
        "nearest_object_dist": nearest_dist,
        "target_is_nearest": nearest_name == target_name,
        "nearest_distractor": nearest_distractor_name,
        "nearest_distractor_dist": nearest_distractor_dist,
        "grounding_classification": classify_grounding(
            dist_to_target, nearest_name, nearest_dist, target_name
        ),
    }


def run(policy: SafeDiffVLAPolicy, device: str, n_episodes: int, start_seed: int, out_dir: Path) -> None:
    policy.config.execute_horizon = 50
    assert policy.config.action_horizon == 50
    assert not policy.config.use_temporal_ensembling
    assert not hasattr(policy.config, "use_phase_conditioning")
    assert not hasattr(policy.config, "replan_on_gripper_close")
    set_seed(start_seed)

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

    inst_state = install_instrumentation()
    episode_records: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        executed_action = rollout_data[ACTION][env_idx, :ep_len].clone()
        steps = list(inst_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "steps": steps, "executed_action": executed_action})

    try:
        info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=0,
            videos_dir=None,
            start_seed=start_seed,
            episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_instrumentation(inst_state)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "grasp_tolerance_m": GRASP_TOLERANCE,
        "proximity_radius_m": PROXIMITY_RADIUS,
        "n_episodes": n_episodes,
        "episodes": [],
    }
    classification_counts: dict[str, int] = {}
    for rec in episode_records:
        analysis = analyze_episode(rec, rec["executed_action"])
        entry = {"seed": rec["seed"], "success": rec["success"], **analysis}
        summary["episodes"].append(entry)
        cls_label = analysis.get("grounding_classification", "no_close_event")
        if not rec["success"]:
            classification_counts[cls_label] = classification_counts.get(cls_label, 0) + 1
        logger.info(
            "seed=%s success=%s close_step=%s classification=%s dist_to_target=%s nearest=%s(%s) target_is_nearest=%s",
            rec["seed"],
            rec["success"],
            analysis.get("first_commanded_close_step"),
            cls_label,
            analysis.get("dist_commanded_to_target"),
            analysis.get("nearest_object"),
            analysis.get("nearest_object_dist"),
            analysis.get("target_is_nearest"),
        )

        if rec["seed"] in DETAILED_TRACE_SEEDS:
            detail_path = out_dir / f"detailed_trace_seed{rec['seed']}.json"
            detail_path.write_text(
                json.dumps(
                    {
                        "seed": rec["seed"],
                        "success": rec["success"],
                        "executed_action": rec["executed_action"].tolist(),
                        "steps": rec["steps"],
                        "close_moment_analysis": analysis,
                    },
                    indent=2,
                )
            )
            logger.info("wrote detailed trace: %s", detail_path)

    summary["failure_grounding_classification_counts"] = classification_counts
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("=== failure_grounding_classification_counts: %s ===", classification_counts)
    logger.info("wrote %s", out_dir / "summary.json")


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_target_grounding_diagnostic")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    logger.info("=== loading policy from %s (unmodified) ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"

    run(policy, args.device, args.n_episodes, args.start_seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
