#!/usr/bin/env python
"""Measurement-only forensics of canonical execute_horizon=50 select_poker failures, using the
*real* VLABench success-condition primitives (contact-based grasp detection + lift-height check)
rather than the gripper-action-channel proxy used by earlier analyses in this directory.

IMPORTANT TASK-MODEL CORRECTION (verified against VLABench source, ~/VLABench, before writing
this script -- see `SelectPokerTask`/`LM4ManipBaseTask` in
VLABench/tasks/hierarchical_tasks/primitive/select_poker_series.py and VLABench/tasks/dm_task.py,
and live-checked against a running env below): `select_poker` has **no target/receptacle
location** at all. Its success condition (`should_terminate_episode` -> `ConditionSet.is_met`,
AND of two sub-conditions) is:
  1. `IsGraspedCondition`: contact detection between the gripper and the target card's geoms
     (`Entity.is_grasped`, pure MuJoCo contact query -- no distance threshold).
  2. `LiftCondition`: the target card's world z-position >= `target_height=0.9` (meters).
The episode auto-terminates (`env.reset()`) the instant both are simultaneously true.

This means "object<->target distance", "target/receptacle pose", and "release near target" as
literally requested don't apply to this task -- there is no place location to release *at*. This
script remaps the same underlying question ("does it get the object to the goal condition but fail
right at the end, or does it never get close at all?") onto what select_poker actually checks:
card height vs. the 0.9m lift threshold, and grasp-contact continuity, both read directly from
live `physics`/`task` state (the same objects/methods `env.step()`'s own success check uses) via a
non-invasive `VLABenchEnv.step`/`.reset` monkeypatch -- call the original, read extra state
afterward, return the original result completely unmodified. No policy or execution code is
touched; `execution.py`, `modeling_safediff_vla.py`, and `envs/vlabench.py` are all read, not
edited.

One documented limitation: on the SUCCESS episode, `VLABenchEnv.step` auto-calls `self.reset()`
internally the instant success fires (before returning), which -- because `reset` is also
monkeypatched to start a fresh per-episode buffer -- replaces `self._env` with the *next* episode's
freshly-reset state before this script's wrapper regains control. The literal success-instant
physics snapshot is therefore not captured for that one episode (all preceding steps are); the
`success` flag itself (from `rollout_data`, entirely independent of this instrumentation) is
unaffected.

Failure classification (7-way, each with an explicit numeric rule; anything not matching a rule
below is `unknown` -- never guessed):
  - never_grasped: `is_grasped` was never True.
  - never_reached_lift_region: was grasped at some point, but max(card_height while grasped) < 0.75m
    (never got within 15cm of the 0.9m threshold while actually holding it).
  - grasp_lost_before_threshold: grasp contact was lost (True->False) at a card height < 0.85m
    (more than 5cm short of the threshold when it let go).
  - grasp_lost_near_threshold: grasp contact was lost at a card height >= 0.85m (within 5cm of the
    threshold) but success never fired -- a near-miss.
  - height_threshold_reached_without_grasp: card height reached >= 0.9m at some point while
    `is_grasped` was False (the card got disturbed/knocked up without being properly held) --
    would be geometrically impossible for a *successful* episode (success requires both
    simultaneously), so this is only checked among failures, where it's an anomaly worth flagging.
  - wrong_object_grasped: a *different* card entity (not `task.target_entity`) was both grasped and
    reached >= 0.9m height -- checked against every other card-like entity in `task.entities`.
  - unknown: none of the above numeric rules matched.

Outputs:
    outputs/eval/safediff_vla_place_forensics/episodes_raw.json   (per-episode, per-step trace)
    outputs/eval/safediff_vla_place_forensics/summary.json        (per-episode classification + divergence-from-success analysis)

Usage:
    MUJOCO_GL=egl python examples/safediff_vla/place_phase_forensics.py
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

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_sincos_20k_padfix/checkpoints/020000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_INDEX = 6
GRIPPER_THRESHOLD = 0.5
LIFT_THRESHOLD = 0.9  # meters, VLABench's own SelectPokerTask condition_config target_height
NEAR_LIFT_MARGIN = 0.05  # "grasp lost within 5cm of threshold" -> near-miss
LIFT_REGION_MARGIN = 0.15  # "never got within 15cm of threshold" -> never_reached_lift_region


def extract_physics_record(env: VLABenchEnvImpl, step_ix: int) -> dict[str, Any]:
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target_name = task.target_entity
    target = task.entities[target_name]

    ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
    ee_quat = np.asarray(robot.get_end_effector_quat(physics), dtype=float)
    card_pos = np.asarray(target.get_xpos(physics), dtype=float)
    card_quat = np.asarray(target.get_xqaut(physics), dtype=float)
    is_grasped = bool(target.is_grasped(physics, robot))

    other_cards = {}
    for name, ent in task.entities.items():
        if name == target_name or name == "table" or name.startswith("card_holder"):
            continue
        try:
            other_cards[name] = {
                "is_grasped": bool(ent.is_grasped(physics, robot)),
                "height": float(np.asarray(ent.get_xpos(physics))[2]),
            }
        except Exception:  # noqa: BLE001 — best-effort diagnostic only, never lets a step fail
            pass

    return {
        "step_ix": step_ix,
        "ee_pos": ee_pos.tolist(),
        "ee_quat": ee_quat.tolist(),
        "target_entity": target_name,
        "card_pos": card_pos.tolist(),
        "card_quat": card_quat.tolist(),
        "card_height": float(card_pos[2]),
        "ee_to_card_dist": float(np.linalg.norm(ee_pos - card_pos)),
        "is_grasped": is_grasped,
        "success_condition_met_now": bool(task.should_terminate_episode(physics)),
        "other_card_entities": other_cards,
    }


def install_env_instrumentation() -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": [], "step_ix": 0}

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            # A real new-episode reset from the eval loop (`env.reset(seed=..., options=...)`,
            # see lerobot_eval.py's rollout()). Start a fresh per-episode buffer.
            #
            # `VLABenchEnv.step()` also calls `self.reset()` (no `seed=`) *internally*, the
            # instant success fires, before returning to its own caller (see module docstring).
            # That call must NOT clear the buffer here -- the just-finished episode's
            # `episode_callback` hasn't read it yet, and clobbering it would silently replace the
            # just-succeeded episode's whole recorded trajectory with a single throwaway
            # unseeded-reset snapshot belonging to no real episode. Guarding on `seed is not None`
            # is what distinguishes the two call sites.
            state["current_episode_steps"] = [extract_physics_record(self, step_ix=0)]
            state["step_ix"] = 0
        return result

    def patched_step(self, action):
        result = orig_step(self, action)  # unmodified call
        terminated = result[2]
        state["step_ix"] += 1
        if not terminated:
            # See module docstring: on the terminal (success) step, `orig_step` has already
            # rebuilt `self._env` for the *next* episode by the time control returns here, so
            # reading physics now would silently record the wrong episode's initial state.
            # Skipping it is a documented, narrow limitation (only ever affects the one success
            # episode's final step), not a silent correctness bug.
            state["current_episode_steps"].append(extract_physics_record(self, step_ix=state["step_ix"]))
        return result  # unmodified return value — no transformation

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_env_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def classify_failure(steps: list[dict]) -> dict:
    grasped_flags = [s["is_grasped"] for s in steps]
    heights = [s["card_height"] for s in steps]

    if not any(grasped_flags):
        return {"phase": "never_grasped", "evidence": "is_grasped was False for every recorded step"}

    heights_while_grasped = [h for g, h in zip(grasped_flags, heights, strict=True) if g]
    max_height_while_grasped = max(heights_while_grasped)

    # wrong_object_grasped: some OTHER card entity was both grasped and lifted near/above threshold
    for s in steps:
        for name, info in s["other_card_entities"].items():
            if info["is_grasped"] and info["height"] >= LIFT_THRESHOLD - NEAR_LIFT_MARGIN:
                return {
                    "phase": "wrong_object_grasped",
                    "evidence": f"entity '{name}' (not the target) was grasped and reached height {info['height']:.3f}m at step {s['step_ix']}",
                }

    # height_threshold_reached_without_grasp: anomaly check (should be geometrically impossible on
    # a genuine success, since success requires simultaneity -- but this is a *failure* episode)
    for s in steps:
        if not s["is_grasped"] and s["card_height"] >= LIFT_THRESHOLD:
            return {
                "phase": "height_threshold_reached_without_grasp",
                "evidence": f"card reached height {s['card_height']:.3f}m >= {LIFT_THRESHOLD}m at step {s['step_ix']} while is_grasped=False",
            }

    if max_height_while_grasped < LIFT_THRESHOLD - LIFT_REGION_MARGIN:
        return {
            "phase": "never_reached_lift_region",
            "evidence": f"max card height while grasped was {max_height_while_grasped:.3f}m, never within {LIFT_REGION_MARGIN}m of the {LIFT_THRESHOLD}m threshold",
        }

    # Find the step where grasp was lost (True -> False) nearest the max-height moment, to report
    # "height at grasp loss" -- use the *last* such transition, since a re-grasp attempt after an
    # early drop is common and the last release is the one that determined the episode's fate.
    loss_step = None
    for i in range(1, len(steps)):
        if grasped_flags[i - 1] and not grasped_flags[i]:
            loss_step = i
    if loss_step is not None:
        height_at_loss = heights[loss_step - 1]  # height at the last instant it was still grasped
        if height_at_loss >= LIFT_THRESHOLD - NEAR_LIFT_MARGIN:
            return {
                "phase": "grasp_lost_near_threshold",
                "evidence": f"grasp contact lost at step {steps[loss_step]['step_ix']}, card height {height_at_loss:.3f}m "
                f"(within {NEAR_LIFT_MARGIN}m of the {LIFT_THRESHOLD}m threshold)",
            }
        return {
            "phase": "grasp_lost_before_threshold",
            "evidence": f"grasp contact lost at step {steps[loss_step]['step_ix']}, card height only {height_at_loss:.3f}m "
            f"(more than {NEAR_LIFT_MARGIN}m short of the {LIFT_THRESHOLD}m threshold)",
        }

    return {
        "phase": "unknown",
        "evidence": f"grasped throughout to episode end, max height while grasped {max_height_while_grasped:.3f}m, "
        "did not clearly match any other rule",
    }


def first_true_step(flags: list[bool], steps: list[dict]) -> int | None:
    for f, s in zip(flags, steps, strict=True):
        if f:
            return s["step_ix"]
    return None


def divergence_from_reference(steps: list[dict], ref_steps: list[dict]) -> dict:
    """First step index at which this episode's (ee_to_card_dist, is_grasped, card_height) starts
    diverging meaningfully from the success episode's own trajectory at the same step index.
    "Meaningfully" = ee_to_card_dist differs by > 0.05m, or is_grasped differs, or card_height
    differs by > 0.05m -- first index where ANY of those holds, compared pointwise by step_ix."""
    n = min(len(steps), len(ref_steps))
    for i in range(n):
        s, r = steps[i], ref_steps[i]
        dist_diff = abs(s["ee_to_card_dist"] - r["ee_to_card_dist"])
        height_diff = abs(s["card_height"] - r["card_height"])
        grasp_diff = s["is_grasped"] != r["is_grasped"]
        if dist_diff > 0.05 or height_diff > 0.05 or grasp_diff:
            return {
                "first_divergence_step": s["step_ix"],
                "ee_to_card_dist_diff": dist_diff,
                "card_height_diff": height_diff,
                "grasp_state_differs": grasp_diff,
            }
    return {"first_divergence_step": None, "note": f"no divergence found within first {n} steps (shorter trajectory length)"}


def run(policy: SafeDiffVLAPolicy, device: str, n_episodes: int, start_seed: int, out_dir: Path) -> None:
    policy.config.execute_horizon = 50  # canonical baseline, explicit even though it's already the loaded default
    assert policy.config.action_horizon == 50
    assert policy.config.use_temporal_ensembling is False
    assert not hasattr(policy.config, "phase_conditioning")
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
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    inst_state = install_env_instrumentation()
    episode_records: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        executed_action = rollout_data[ACTION][env_idx, :ep_len].clone()
        commanded_gripper = executed_action[:, GRIPPER_INDEX]
        # `> GRIPPER_THRESHOLD` = open (matches the "side=1 is open" convention used throughout
        # this directory's other gripper analyses, e.g. execute_horizon_gripper_analysis.py) --
        # the episode always starts open, so the *first close* is the first `False`, not `True`.
        commanded_binary = (commanded_gripper > GRIPPER_THRESHOLD).tolist()
        steps = list(inst_state["current_episode_steps"])  # copy before the next reset touches it

        grasped_flags = [s["is_grasped"] for s in steps]
        heights = [s["card_height"] for s in steps]
        episode_records.append(
            {
                "episode_ix": episode_ix,
                "episode_len": ep_len,
                "n_physics_steps_recorded": len(steps),
                "first_physics_grasp_step": first_true_step(grasped_flags, steps),
                "first_commanded_close_step": (commanded_binary.index(False) + 1) if False in commanded_binary else None,
                "min_card_height": min(heights) if heights else None,
                "max_card_height": max(heights) if heights else None,
                "final_card_height": heights[-1] if heights else None,
                "steps": steps,
                "commanded_gripper_binary": commanded_binary,
            }
        )
        logger.info(
            "  episode %d len=%d n_physics_steps=%d first_physics_grasp=%s max_height=%s",
            episode_ix,
            ep_len,
            len(steps),
            episode_records[-1]["first_physics_grasp_step"],
            f"{episode_records[-1]['max_card_height']:.3f}" if episode_records[-1]["max_card_height"] is not None else None,
        )

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
        uninstall_env_instrumentation(inst_state)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))

    ref = next((r for r in episode_records if r["success"]), None)
    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "execute_horizon": 50,
        "action_horizon": 50,
        "lift_threshold_m": LIFT_THRESHOLD,
        "success_condition": "is_grasped(target_card, robot) AND target_card.height_z >= lift_threshold_m, simultaneously (VLABench SelectPokerTask condition_config, verified against ~/VLABench source)",
        "n_episodes": n_episodes,
        "n_success": sum(1 for r in episode_records if r["success"]),
        "reference_success_seed": ref["seed"] if ref else None,
        "episodes": [],
    }
    for rec in episode_records:
        entry = {
            "seed": rec["seed"],
            "success": rec["success"],
            "episode_len": rec["episode_len"],
            "first_physics_grasp_step": rec["first_physics_grasp_step"],
            "first_commanded_close_step": rec["first_commanded_close_step"],
            "min_card_height": rec["min_card_height"],
            "max_card_height": rec["max_card_height"],
            "final_card_height": rec["final_card_height"],
        }
        if not rec["success"]:
            entry["failure_classification"] = classify_failure(rec["steps"])
            if ref is not None:
                entry["divergence_from_reference"] = divergence_from_reference(rec["steps"], ref["steps"])
        summary["episodes"].append(entry)

    summary["failure_phase_counts"] = {
        phase: sum(1 for e in summary["episodes"] if not e["success"] and e["failure_classification"]["phase"] == phase)
        for phase in (
            "never_grasped",
            "never_reached_lift_region",
            "grasp_lost_before_threshold",
            "grasp_lost_near_threshold",
            "height_threshold_reached_without_grasp",
            "wrong_object_grasped",
            "unknown",
        )
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("\n=== failure_phase_counts: %s ===", summary["failure_phase_counts"])
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_place_forensics")
    args = parser.parse_args()

    logger.info("=== loading canonical checkpoint from %s (unmodified) ===", args.checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"

    run(policy, args.device, args.n_episodes, args.start_seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
