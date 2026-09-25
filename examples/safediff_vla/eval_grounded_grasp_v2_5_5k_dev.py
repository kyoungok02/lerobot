#!/usr/bin/env python
"""DEVELOPMENT-ONLY closed-loop rollout evaluation of `train_grounded_grasp_v2_5_5k.py`'s
checkpoint (`temporal_decoder_grounded_grasp_v2_5`) on `select_poker` seeds 1000-1002, compared
DIRECTLY against a v2 checkpoint on the SAME seeds. These are already-seen seeds (development use
only, per this pilot's own scope) -- held-out seeds are not evaluated here.

Self-contained: reads live MuJoCo physics via a non-invasive `VLABenchEnv.step`/`.reset`
monkeypatch (same pattern as every prior rollout script on this branch), plus instance-level wraps
of the v2.5 policy's own `_plan_temporal_decoder` (captures v2's own `predicted_target_xyz`, cached
every GLOBAL-stage replan), `_maybe_enter_local_grasp` (captures the PRE_GRASP->LOCAL_GRASP
transition), `_plan_local_grasp_step` (captures each LOCAL_GRASP step's delta/target), and
`_select_action_local_grasp` (captures the gripper latch / LOCAL_GRASP->POST_GRASP transition). No
model/loss/execution-policy code is touched by this script.

Per episode, records (v2.5) or a reduced set (v2, for direct comparison):
  - task success, grasp/contact success (ground truth `is_grasped`, independent of the model).
  - LOCAL_GRASP entry: whether it happened, at what timestep, and total step count spent in it.
  - min EE<->target-card distance BEFORE local entry vs AFTER (over the whole episode, and
    specifically within the LOCAL_GRASP window) -- the core "did refinement actually help" metric.
  - mean |delta_xyz| magnitude during LOCAL_GRASP.
  - gripper close timestep, grasp (is_grasped) timestep, LOCAL_GRASP->POST_GRASP transition
    timestep.
  - target_is_nearest (nearest-card-to-predicted-target identity check), same convention as prior
    eval scripts on this branch.
  - video path.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_v2_5_5k_dev.py
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
from lerobot.policies.safediff_vla.local_grasp_refiner import LOCAL_GRASP, POST_GRASP, PRE_GRASP
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

V2_5_CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_5_5k/checkpoints/005000/pretrained_model"
V2_CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/015000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
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
    ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
    cards = all_card_positions(env)
    card_pos = np.array(cards.get(target_name, [np.nan, np.nan, np.nan]))
    try:
        is_grasped = bool(task.entities[target_name].is_grasped(physics, robot))
    except Exception:  # noqa: BLE001
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


def install_env_instrumentation(policy_log: list[dict]) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def merge_policy_log(record: dict, base: np.ndarray) -> None:
        keys = (
            "stage", "predicted_target_world", "predicted_target_dist_to_gt_card",
            "nearest_card_to_predicted_target", "target_is_nearest", "local_delta_xyz_m",
            "just_entered_local", "just_latched_gripper", "just_transitioned_post_grasp",
            "applied_gripper", "distance_to_predicted_target_at_entry_m",
        )
        for k in keys:
            record[k] = None
        if not policy_log:
            return
        p = policy_log[-1]
        record["stage"] = p["stage"]
        record["just_entered_local"] = p.get("just_entered_local", False)
        record["just_latched_gripper"] = p.get("just_latched_gripper", False)
        record["just_transitioned_post_grasp"] = p.get("just_transitioned_post_grasp", False)
        record["applied_gripper"] = p.get("applied_gripper")
        record["local_delta_xyz_m"] = p.get("local_delta_xyz_m")
        record["distance_to_predicted_target_at_entry_m"] = p.get("distance_to_predicted_target_at_entry_m")
        if p.get("predicted_target_robot_frame") is not None:
            predicted_target_world = np.asarray(p["predicted_target_robot_frame"], dtype=float) + base
            card_pos = np.asarray(record["card_pos"], dtype=float)
            record["predicted_target_world"] = predicted_target_world.tolist()
            record["predicted_target_dist_to_gt_card"] = float(np.linalg.norm(predicted_target_world - card_pos))
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


def install_v2_5_policy_instrumentation(policy: SafeDiffVLAPolicy) -> dict:
    """Instance-level wraps (not class patches) of v2.5's own stage-transition methods --
    observes, does not change, their already-implemented behavior."""
    orig_plan = SafeDiffVLAPolicy._plan_temporal_decoder
    orig_enter_local = SafeDiffVLAPolicy._maybe_enter_local_grasp
    orig_local_step = SafeDiffVLAPolicy._plan_local_grasp_step
    orig_select_local = SafeDiffVLAPolicy._select_action_local_grasp
    log: list[dict] = []

    def wrapped_plan(self, batch):
        actions, metrics = orig_plan(self, batch)
        if "predicted_target_xyz" in metrics:
            target_pos_m = metrics["predicted_target_xyz"] * self.action_pos_std + self.action_pos_mean
            log.append({"stage": int(self._v25_stage), "predicted_target_robot_frame": target_pos_m[0].detach().cpu().numpy().tolist()})
        return actions, metrics

    def wrapped_enter_local(self, action, current_state):
        stage_before = self._v25_stage
        # Read-only: replicate `_maybe_enter_local_grasp`'s OWN distance computation (not the
        # method's return value, which is just the unmodified `action`) so we can log the exact
        # quantity the controller thresholds against (EE <-> v2's PREDICTED target, in meters) --
        # distinct from EE <-> the actual ground-truth card, which `ee_to_card_dist` measures for
        # task-performance diagnostics elsewhere in this script. Purely observational; does not
        # call or duplicate any decision logic, just the same un-normalization arithmetic.
        distance_to_predicted_target_m = None
        if self._v25_stage == PRE_GRASP and self._v25_last_target_xyz is not None:
            ee_pos_m = current_state[..., :3] * self.state_pos_std + self.state_pos_mean
            target_pos_m = self._v25_last_target_xyz * self.action_pos_std + self.action_pos_mean
            distance_to_predicted_target_m = float((ee_pos_m - target_pos_m).norm(dim=-1)[0].item())
        result = orig_enter_local(self, action, current_state)
        if stage_before == PRE_GRASP and self._v25_stage == LOCAL_GRASP:
            entry = {
                "stage": int(self._v25_stage),
                "predicted_target_robot_frame": None,
                "just_entered_local": True,
                "distance_to_predicted_target_at_entry_m": distance_to_predicted_target_m,
            }
            if log:
                log[-1]["just_entered_local"] = True
                log[-1]["distance_to_predicted_target_at_entry_m"] = distance_to_predicted_target_m
            else:
                log.append(entry)
        return result

    def wrapped_local_step(self, batch):
        action, metrics = orig_local_step(self, batch)
        target_pos_m = metrics["predicted_target_xyz"] * self.action_pos_std + self.action_pos_mean
        log.append(
            {
                "stage": int(self._v25_stage),
                "predicted_target_robot_frame": target_pos_m[0].detach().cpu().numpy().tolist(),
                "local_delta_xyz_m": metrics["delta_xyz_m"][0].detach().cpu().numpy().tolist(),
            }
        )
        return action, metrics

    def wrapped_select_local(self, batch, current_state):
        was_latched = self._v25_gripper_closed_latch
        action = orig_select_local(self, batch, current_state)
        just_latched = (not was_latched) and self._v25_gripper_closed_latch
        if log:
            log[-1]["stage"] = int(self._v25_stage)
            log[-1]["just_latched_gripper"] = just_latched
            log[-1]["just_transitioned_post_grasp"] = just_latched  # same event, see architecture docstring
            log[-1]["applied_gripper"] = float(action[0, GRIPPER_INDEX_RAW].item())
        return action

    policy._plan_temporal_decoder = wrapped_plan.__get__(policy, SafeDiffVLAPolicy)
    policy._maybe_enter_local_grasp = wrapped_enter_local.__get__(policy, SafeDiffVLAPolicy)
    policy._plan_local_grasp_step = wrapped_local_step.__get__(policy, SafeDiffVLAPolicy)
    policy._select_action_local_grasp = wrapped_select_local.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log}


def uninstall_v2_5_policy_instrumentation(policy: SafeDiffVLAPolicy) -> None:
    for attr in ("_plan_temporal_decoder", "_maybe_enter_local_grasp", "_plan_local_grasp_step", "_select_action_local_grasp"):
        if attr in policy.__dict__:
            del policy.__dict__[attr]


def summarize_episode_v2_5(
    steps: list[dict], seed: int, success: bool, episode_len: int, video: Path, local_grasp_radius_m: float
) -> dict:
    """NOTE on metric naming (fixed after an earlier report mislabeled this): every
    `*_ee_to_gt_card_dist_*` field below is EE<->ACTUAL ground-truth card distance -- a
    task-performance diagnostic, computed against the real card position, NEVER what the
    controller itself thresholds against. The controller's own PRE_GRASP->LOCAL_GRASP trigger
    compares EE<->v2's own PREDICTED target (`local_grasp_radius_m`, see
    `_maybe_enter_local_grasp`) -- that quantity is reported separately below as
    `distance_to_predicted_target_at_entry_m`, which is what should be checked against
    `local_grasp_radius_m` to confirm the controller fired correctly. The two are expected to
    differ whenever v2's own grounding isn't pixel-perfect, and neither is "more correct" than the
    other -- they just answer different questions (control-loop math vs. real-world outcome).
    """
    grasp_success = any(s["is_grasped"] for s in steps)
    local_steps = [s for s in steps if s.get("stage") == LOCAL_GRASP]
    local_entry_step = next((s["step_ix"] for s in steps if s.get("just_entered_local")), None)
    gripper_close_timestep = next((s["step_ix"] for s in steps if s.get("just_latched_gripper")), None)
    post_grasp_transition_step = next((s["step_ix"] for s in steps if s.get("just_transitioned_post_grasp")), None)
    grasp_timestep = next((s["step_ix"] for s in steps if s["is_grasped"]), None)

    entry_step_record = next((s for s in steps if s["step_ix"] == local_entry_step), None) if local_entry_step is not None else None
    distance_to_predicted_target_at_entry_m = (
        entry_step_record.get("distance_to_predicted_target_at_entry_m") if entry_step_record is not None else None
    )
    entry_within_local_grasp_radius = (
        distance_to_predicted_target_at_entry_m <= local_grasp_radius_m
        if distance_to_predicted_target_at_entry_m is not None
        else None
    )
    ee_to_gt_card_dist_at_local_entry = entry_step_record["ee_to_card_dist"] if entry_step_record is not None else None

    before_local = [s for s in steps if local_entry_step is None or s["step_ix"] < local_entry_step]
    after_local = [s for s in steps if local_entry_step is not None and s["step_ix"] >= local_entry_step]
    min_ee_to_gt_card_dist_before_local_entry = min((s["ee_to_card_dist"] for s in before_local), default=None)
    min_ee_to_gt_card_dist_after_local_entry = min((s["ee_to_card_dist"] for s in after_local), default=None)
    min_ee_to_gt_card_dist_during_local_steps = min((s["ee_to_card_dist"] for s in local_steps), default=None) if local_steps else None
    # The actual "did refinement help" comparison: distance to the real card AT the moment of
    # entry vs. the best distance achieved anywhere from that point on -- both anchored to the
    # SAME reference point (local entry), unlike comparing two different windows' minimums.
    local_refinement_improved_distance = (
        min_ee_to_gt_card_dist_after_local_entry < ee_to_gt_card_dist_at_local_entry
        if ee_to_gt_card_dist_at_local_entry is not None and min_ee_to_gt_card_dist_after_local_entry is not None
        else None
    )

    delta_mags = [float(np.linalg.norm(s["local_delta_xyz_m"])) for s in local_steps if s.get("local_delta_xyz_m") is not None]
    target_is_nearest_values = [s["target_is_nearest"] for s in steps if s.get("target_is_nearest") is not None]
    fraction_steps_target_is_nearest = (
        sum(target_is_nearest_values) / len(target_is_nearest_values) if target_is_nearest_values else None
    )
    # Grounding correctness must be judged on the APPROACH only (PRE_GRASP + LOCAL_GRASP), never
    # POST_GRASP: once the grasp attempt is over, the target is presumably already held/lifted (or
    # the attempt has moved on) and `target_point_head`'s prediction over the CURRENT observation
    # is answering a different, out-of-distribution question -- v2's own model was never trained
    # to "re-ground" a card that's no longer sitting on the table. POST_GRASP is typically the
    # large majority of an episode's steps (the arm keeps executing after grasping until the
    # episode's step budget runs out), so a naive whole-episode majority vote is dominated by this
    # irrelevant tail and can disagree sharply with what the approach itself actually showed -- v2
    # has no equivalent phase to exclude, so mixing the two would also make the v2-vs-v2.5
    # comparison apples-to-oranges.
    approach_steps = [s for s in steps if s.get("stage") != POST_GRASP]
    target_is_nearest_approach = [s["target_is_nearest"] for s in approach_steps if s.get("target_is_nearest") is not None]
    fraction_steps_target_is_nearest_approach = (
        sum(target_is_nearest_approach) / len(target_is_nearest_approach) if target_is_nearest_approach else None
    )
    # Majority-vote grounding-correctness rule -- same convention as this branch's earlier
    # `wrong_object_grounding_majority_of_steps` fields (see `eval_grounded_grasp_v2_5k.py`), but
    # restricted to the approach (see above).
    correct_grounding = (
        fraction_steps_target_is_nearest_approach >= 0.5 if fraction_steps_target_is_nearest_approach is not None else None
    )

    return {
        "seed": seed,
        "success": success,
        "grasp_success": grasp_success,
        "target_card_identity": steps[0]["target_entity"],
        "episode_len": episode_len,
        "correct_grounding": correct_grounding,
        "fraction_steps_target_is_nearest_approach": fraction_steps_target_is_nearest_approach,
        "fraction_steps_target_is_nearest_whole_episode": fraction_steps_target_is_nearest,
        "entered_local_grasp": local_entry_step is not None,
        "local_entry_step": local_entry_step,
        "n_local_steps": len(local_steps),
        "distance_to_predicted_target_at_entry_m": distance_to_predicted_target_at_entry_m,
        "entry_within_local_grasp_radius": entry_within_local_grasp_radius,
        "ee_to_gt_card_dist_at_local_entry": ee_to_gt_card_dist_at_local_entry,
        "gripper_close_timestep": gripper_close_timestep,
        "post_grasp_transition_step": post_grasp_transition_step,
        "grasp_timestep": grasp_timestep,
        "min_ee_to_gt_card_dist_before_local_entry": min_ee_to_gt_card_dist_before_local_entry,
        "min_ee_to_gt_card_dist_after_local_entry": min_ee_to_gt_card_dist_after_local_entry,
        "min_ee_to_gt_card_dist_during_local_steps": min_ee_to_gt_card_dist_during_local_steps,
        "local_refinement_improved_distance": local_refinement_improved_distance,
        "episode_min_ee_to_gt_card_dist": min(s["ee_to_card_dist"] for s in steps),
        "mean_local_delta_xyz_magnitude_m": (sum(delta_mags) / len(delta_mags)) if delta_mags else None,
        "video": str(video),
    }


def run(policy: SafeDiffVLAPolicy, device: str, seeds: list[int], out_dir: Path) -> dict:
    assert policy.config.architecture == "temporal_decoder_grounded_grasp_v2_5"
    set_seed(seeds[0])

    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=V2_5_CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    policy_inst = install_v2_5_policy_instrumentation(policy)
    env_state = install_env_instrumentation(policy_inst["log"])
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(env_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps})
        policy_inst["log"].clear()
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
        uninstall_v2_5_policy_instrumentation(policy)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))

    per_episode_summary = [
        summarize_episode_v2_5(
            rec["steps"], rec["seed"], rec["success"], rec["episode_len"],
            videos_dir / f"eval_episode_{rec['episode_ix']}.mp4", policy.config.local_grasp_radius_m,
        )
        for rec in episode_records
    ]

    def agg_mean(key: str) -> float | None:
        vals = [e[key] for e in per_episode_summary if e.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "checkpoint": V2_5_CHECKPOINT,
        "task": TASK,
        "n_episodes": len(seeds),
        "seeds": seeds,
        "note": "DEVELOPMENT eval on already-seen seeds only -- held-out seeds not evaluated in this pilot.",
        "per_episode": per_episode_summary,
        "aggregate": {
            "n_success": sum(1 for e in per_episode_summary if e["success"]),
            "n_grasp_success": sum(1 for e in per_episode_summary if e["grasp_success"]),
            "n_correct_grounding": sum(1 for e in per_episode_summary if e["correct_grounding"]),
            "n_correct_grounding_success": sum(1 for e in per_episode_summary if e["correct_grounding"] and e["success"]),
            "n_correct_grounding_failure": sum(1 for e in per_episode_summary if e["correct_grounding"] and not e["success"]),
            "n_wrong_grounding": sum(1 for e in per_episode_summary if e["correct_grounding"] is False),
            "n_entered_local_grasp": sum(1 for e in per_episode_summary if e["entered_local_grasp"]),
            "n_entry_within_local_grasp_radius": sum(1 for e in per_episode_summary if e["entry_within_local_grasp_radius"]),
            "n_local_refinement_improved_distance": sum(1 for e in per_episode_summary if e["local_refinement_improved_distance"]),
            "mean_distance_to_predicted_target_at_entry_m": agg_mean("distance_to_predicted_target_at_entry_m"),
            "mean_ee_to_gt_card_dist_at_local_entry": agg_mean("ee_to_gt_card_dist_at_local_entry"),
            "mean_min_ee_to_gt_card_dist_before_local_entry": agg_mean("min_ee_to_gt_card_dist_before_local_entry"),
            "mean_min_ee_to_gt_card_dist_after_local_entry": agg_mean("min_ee_to_gt_card_dist_after_local_entry"),
            "mean_min_ee_to_gt_card_dist_during_local_steps": agg_mean("min_ee_to_gt_card_dist_during_local_steps"),
            "mean_episode_min_ee_to_gt_card_dist": agg_mean("episode_min_ee_to_gt_card_dist"),
            "mean_local_delta_xyz_magnitude_m": agg_mean("mean_local_delta_xyz_magnitude_m"),
            "mean_fraction_steps_target_is_nearest_approach": agg_mean("fraction_steps_target_is_nearest_approach"),
            "mean_fraction_steps_target_is_nearest_whole_episode": agg_mean("fraction_steps_target_is_nearest_whole_episode"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    logger.info("aggregate: %s", json.dumps(summary["aggregate"], indent=2))
    return summary


def run_v2_baseline(device: str, seeds: list[int], out_dir: Path) -> dict:
    """Same seeds, the v2 checkpoint v2.5 was initialized from -- for a direct, apples-to-apples
    comparison. Reuses `eval_grounded_grasp_v2_5k.py`'s own instrumentation approach, trimmed to
    just what's needed for the comparison table (task success, grasp success, min EE-to-card
    distance, target_is_nearest fraction)."""
    from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy as _Policy

    policy = _Policy.from_pretrained(V2_CHECKPOINT)
    policy = policy.to(device)
    policy.eval()
    set_seed(seeds[0])

    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=V2_CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    orig_close = SafeDiffVLAPolicy._grounded_grasp_v2_reactive_close
    plog: list[dict] = []

    def wrapped_close(self, action, current_state):
        target_pos_m = None
        if self._v2_last_target_xyz is not None:
            target_pos_m = (self._v2_last_target_xyz * self.action_pos_std + self.action_pos_mean)[0].detach().cpu().numpy()
        was_triggered_before = self._v2_close_triggered
        result = orig_close(self, action, current_state)
        plog.append({
            "predicted_target_robot_frame": target_pos_m.tolist() if target_pos_m is not None else None,
            "just_triggered_close": (not was_triggered_before) and self._v2_close_triggered,
        })
        return result

    policy._grounded_grasp_v2_reactive_close = wrapped_close.__get__(policy, SafeDiffVLAPolicy)

    def merge(record, base):
        record["predicted_target_world"] = None
        record["target_is_nearest"] = None
        record["just_triggered_close"] = False
        if not plog:
            return
        p = plog[-1]
        record["just_triggered_close"] = p.get("just_triggered_close", False)
        if p.get("predicted_target_robot_frame") is None:
            return
        predicted_target_world = np.asarray(p["predicted_target_robot_frame"], dtype=float) + base
        record["predicted_target_world"] = predicted_target_world.tolist()
        dists = {
            name: float(np.linalg.norm(predicted_target_world - np.asarray(pos, dtype=float)))
            for name, pos in record["all_card_positions"].items()
        }
        nearest = min(dists, key=dists.get) if dists else None
        record["target_is_nearest"] = nearest == record["target_entity"]

    cls = VLABenchEnvImpl
    orig_step, orig_reset = cls.step, cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            record = extract_physics_record(self, step_ix=0)
            merge(record, robot_base(self))
            state["current_episode_steps"] = [record]
        return result

    def patched_step(self, action):
        result = orig_step(self, action)
        if not result[2]:
            record = extract_physics_record(self, step_ix=len(state["current_episode_steps"]))
            merge(record, robot_base(self))
            state["current_episode_steps"].append(record)
        return result

    cls.step, cls.reset = patched_step, patched_reset
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix, rollout_data, env_idx, done_index):
        episode_records.append({"episode_ix": episode_ix, "episode_len": done_index + 1, "steps": list(state["current_episode_steps"])})
        plog.clear()

    try:
        info = eval_policy(
            env=env, policy=policy, env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=len(seeds),
            max_episodes_rendered=len(seeds), videos_dir=videos_dir, start_seed=seeds[0], episode_callback=episode_callback,
        )
    finally:
        env.close()
        cls.step, cls.reset = orig_step, orig_reset
        if "_grounded_grasp_v2_reactive_close" in policy.__dict__:
            del policy.__dict__["_grounded_grasp_v2_reactive_close"]

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"], rec["success"] = ep["seed"], ep["success"]

    per_episode_summary = []
    for rec in episode_records:
        steps = rec["steps"]
        gripper_close_timestep = next((s["step_ix"] for s in steps if s.get("just_triggered_close")), None)
        # Same "approach only" restriction as v2.5's own classification (see
        # `summarize_episode_v2_5`): once the gripper has closed, v2's own `target_point_head`
        # prediction is answering an out-of-distribution question (the card may already be
        # lifted), so it must not be allowed to drag down a grounding-correctness judgment that's
        # supposed to describe the APPROACH. v2 has no separate stage marker, so this uses the
        # gripper-close timestep as the approach/post-grasp boundary instead.
        approach_steps = [s for s in steps if gripper_close_timestep is None or s["step_ix"] <= gripper_close_timestep]
        target_is_nearest_approach = [s["target_is_nearest"] for s in approach_steps if s.get("target_is_nearest") is not None]
        fraction_steps_target_is_nearest_approach = (
            sum(target_is_nearest_approach) / len(target_is_nearest_approach) if target_is_nearest_approach else None
        )
        per_episode_summary.append(
            {
                "seed": rec["seed"],
                "success": rec["success"],
                "grasp_success": any(s["is_grasped"] for s in steps),
                "grasp_timestep": next((s["step_ix"] for s in steps if s["is_grasped"]), None),
                "target_card_identity": steps[0]["target_entity"],
                "correct_grounding": (
                    fraction_steps_target_is_nearest_approach >= 0.5
                    if fraction_steps_target_is_nearest_approach is not None
                    else None
                ),
                "fraction_steps_target_is_nearest_approach": fraction_steps_target_is_nearest_approach,
                "gripper_close_timestep": gripper_close_timestep,
                "episode_min_ee_to_gt_card_dist": min(s["ee_to_card_dist"] for s in steps),
                "video": str(videos_dir / f"eval_episode_{rec['episode_ix']}.mp4"),
            }
        )
    summary = {
        "checkpoint": V2_CHECKPOINT,
        "task": TASK,
        "seeds": seeds,
        "per_episode": per_episode_summary,
        "aggregate": {
            "n_success": sum(1 for e in per_episode_summary if e["success"]),
            "n_grasp_success": sum(1 for e in per_episode_summary if e["grasp_success"]),
            "n_correct_grounding": sum(1 for e in per_episode_summary if e["correct_grounding"]),
            "n_correct_grounding_success": sum(1 for e in per_episode_summary if e["correct_grounding"] and e["success"]),
            "n_correct_grounding_failure": sum(1 for e in per_episode_summary if e["correct_grounding"] and not e["success"]),
            "n_wrong_grounding": sum(1 for e in per_episode_summary if e["correct_grounding"] is False),
            "mean_episode_min_ee_to_gt_card_dist": sum(e["episode_min_ee_to_gt_card_dist"] for e in per_episode_summary) / len(per_episode_summary),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("v2 baseline aggregate: %s", json.dumps(summary["aggregate"], indent=2))
    return summary


def build_comparison_table(v2_summary: dict, v2_5_summary: dict, out_dir: Path) -> dict:
    """Joins v2's and v2.5's per-episode results by seed and answers the core question this
    comparison is for: does v2.5's `LocalGraspRefiner` turn a v2 episode that had CORRECT
    grounding but still FAILED to grasp into an actual success? Classifies every seed into
    correct_grounding+success / correct_grounding+failure / wrong_grounding (per v2's OWN
    grounding, since that's the policy whose failures this experiment is trying to fix)."""
    v2_by_seed = {e["seed"]: e for e in v2_summary["per_episode"]}
    v2_5_by_seed = {e["seed"]: e for e in v2_5_summary["per_episode"]}
    rows = []
    for seed in sorted(set(v2_by_seed) & set(v2_5_by_seed)):
        v2e, v25e = v2_by_seed[seed], v2_5_by_seed[seed]
        category = (
            "wrong_grounding" if v2e["correct_grounding"] is False
            else ("correct_grounding_success" if v2e["success"] else "correct_grounding_failure")
        )
        rows.append(
            {
                "seed": seed,
                "target_card_identity": v2e["target_card_identity"],
                "category_by_v2_grounding": category,
                "v2_success": v2e["success"],
                "v2_5_success": v25e["success"],
                "v2_grasp_success": v2e["grasp_success"],
                "v2_5_grasp_success": v25e["grasp_success"],
                "v2_5_entered_local_grasp": v25e["entered_local_grasp"],
                "v2_5_local_refinement_improved_distance": v25e["local_refinement_improved_distance"],
                # The core question this comparison exists to answer.
                "v2_correct_grounding_failure_fixed_by_v2_5": category == "correct_grounding_failure" and v25e["success"],
            }
        )
    n_correct_grounding_failure = sum(1 for r in rows if r["category_by_v2_grounding"] == "correct_grounding_failure")
    n_fixed = sum(1 for r in rows if r["v2_correct_grounding_failure_fixed_by_v2_5"])
    table = {
        "rows": rows,
        "n_seeds_compared": len(rows),
        "n_v2_correct_grounding_success": sum(1 for r in rows if r["category_by_v2_grounding"] == "correct_grounding_success"),
        "n_v2_correct_grounding_failure": n_correct_grounding_failure,
        "n_v2_wrong_grounding": sum(1 for r in rows if r["category_by_v2_grounding"] == "wrong_grounding"),
        "n_v2_correct_grounding_failure_fixed_by_v2_5": n_fixed,
        "fraction_v2_correct_grounding_failure_fixed_by_v2_5": (
            n_fixed / n_correct_grounding_failure if n_correct_grounding_failure else None
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v2_vs_v2_5_comparison.json").write_text(json.dumps(table, indent=2))
    logger.info("wrote %s", out_dir / "v2_vs_v2_5_comparison.json")
    logger.info(
        "=== core question: does v2.5's local refiner fix v2's correct-grounding-but-failed episodes? ===\n"
        "%d/%d such episodes fixed (%s)",
        n_fixed, n_correct_grounding_failure,
        f"{table['fraction_v2_correct_grounding_failure_fixed_by_v2_5']:.0%}" if n_correct_grounding_failure else "N/A -- none in this seed set",
    )
    return table


def main() -> None:
    global V2_5_CHECKPOINT, V2_CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2-5-checkpoint", default=V2_5_CHECKPOINT)
    parser.add_argument("--v2-checkpoint", default=V2_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_v2_5_5k_dev")
    parser.add_argument("--v2-output-dir", default="outputs/eval/safediff_vla_grounded_grasp_v2_5_5k_dev/v2_baseline")
    parser.add_argument("--skip-v2-baseline", action="store_true")
    args = parser.parse_args()
    V2_5_CHECKPOINT = args.v2_5_checkpoint
    V2_CHECKPOINT = args.v2_checkpoint

    logger.info("=== loading temporal_decoder_grounded_grasp_v2_5 checkpoint from %s ===", V2_5_CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(V2_5_CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    v2_5_summary = run(policy, args.device, args.seeds, Path(args.output_dir))

    if not args.skip_v2_baseline:
        logger.info("=== loading temporal_decoder_grounded_grasp_v2 baseline checkpoint from %s ===", V2_CHECKPOINT)
        v2_summary = run_v2_baseline(args.device, args.seeds, Path(args.v2_output_dir))
        logger.info(
            "=== v2 vs v2.5 (seeds %s) ===\nv2:   %s\nv2.5: %s",
            args.seeds,
            json.dumps(v2_summary["aggregate"]),
            json.dumps(v2_5_summary["aggregate"]),
        )
        build_comparison_table(v2_summary, v2_5_summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
