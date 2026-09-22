#!/usr/bin/env python
"""Re-evaluate the canonical `sincos_20k_padfix` `temporal_decoder` checkpoint on `select_poker`
with the eval-harness instruction bug fixed (see `lerobot/envs/vlabench.py`'s
`VLABenchEnv._resolve_task_description`), and compare against the preserved pre-fix rollouts.

Background: `lerobot_eval.py` (and every example script in this directory) feeds the policy
`observation["task"] = env.call("task_description")` every step. Before the fix,
`VLABenchEnv.task_description` fell back to the generic task NAME ("select_poker") for every
episode, regardless of which card was actually the target -- verified against the installed
VLABench package: no task class ever defines `task_description`/`language_instruction`, and the
training dataset's own task strings (`lerobot/vlabench_unified`) are per-episode-specific (e.g.
"primitive: Please pick the poker 7 of spades"). So every closed-loop rollout of this checkpoint
prior to the fix ran with zero card-identity information in the language channel. See
`instruction_grounding_audit.py`'s module docstring for the original discovery, and
`tests/envs/test_vlabench.py` for the regression tests guarding the fix itself.

This script does NOT retrain, and does not touch the VLM, pooling, or TargetPointHead -- it only
loads the existing checkpoint and re-runs the standard closed-loop rollout via
`lerobot.scripts.lerobot_eval.eval_policy` (same helper `eval_baseline_rollout.py` and
`place_phase_forensics.py` use), now with the corrected instruction.

Same fixed condition as the preserved pre-fix runs (`outputs/eval/sincos_20k_padfix/report.json`
and `outputs/eval/safediff_vla_place_forensics/`, both left untouched by this script): select_poker,
seeds 1000-1009, execute_horizon=action_horizon=50 (full open-loop chunk), temporal ensembling OFF.
This checkpoint has no subgoal/phase-conditioning/corrective-replan config at all (asserted below),
so those ablations are correctly "OFF" simply by not applying any of the monkeypatches those other
scripts in this directory (`corrective_replan_ablation.py`, `eval_ablation_rollout.py`) add.

Physics instrumentation (`extract_physics_record`) mirrors `place_phase_forensics.py`'s
non-invasive `VLABenchEnv.step`/`.reset` monkeypatch -- call the original, read extra state
afterward, return the original result unmodified -- extended to also record EE distance to every
other real card (not just the target), which `place_phase_forensics.py` didn't need. That extra
per-card distance is what makes a real "did it approach the CORRECT card" metric possible: distance
to the target card alone can't distinguish "got close to the target" from "got close to whichever
card was in that direction anyway".

Metrics per episode, and aggregated:
  - task success (VLABench's own `should_terminate_episode`, from `eval_policy`'s `success`)
  - actual target-card grasp (`is_grasped(target_card)` True at any point)
  - wrong-object grasp (a different card entity grasped AND lifted near/above the lift threshold)
  - never-grasped (target card never grasped)
  - min EE<->correct-card distance (over the whole episode)
  - EE<->correct-card distance at the "close" instant (first commanded gripper-close step)
  - NEW: approached_correct_card -- at the step of this episode's global minimum EE<->ANY-card
    distance, was the nearest card actually the target? (fraction across episodes)

Provenance logged per episode: seed, task name, exact instruction string fed to the policy (now
returned directly by `eval_policy`'s `per_episode` entries after the `lerobot_eval.py` provenance
change).

Outputs:
    <output-dir>/episodes_raw.json   per-episode, per-step physics trace
    <output-dir>/summary.json        per-episode metrics + aggregate + comparison against the
                                      preserved pre-fix runs
    <output-dir>/videos/eval_episode_{i}.mp4

Usage:
    MUJOCO_GL=egl python examples/safediff_vla/eval_canonical_corrected_instruction.py
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
NEAR_LIFT_MARGIN = 0.05

# Preserved pre-fix (buggy-instruction) runs for this exact checkpoint/task/seed range -- read-only,
# never modified or deleted by this script.
BUGGY_BASELINE_PLACE_FORENSICS = "outputs/eval/safediff_vla_place_forensics/episodes_raw.json"
BUGGY_BASELINE_REPORT = "outputs/eval/sincos_20k_padfix/report.json"


def extract_physics_record(env: VLABenchEnvImpl, step_ix: int) -> dict[str, Any]:
    physics = env._env.physics
    task = env._env.task
    robot = task.robot
    target_name = task.target_entity
    target = task.entities[target_name]

    ee_pos = np.asarray(robot.get_end_effector_pos(physics), dtype=float)
    card_pos = np.asarray(target.get_xpos(physics), dtype=float)
    is_grasped = bool(target.is_grasped(physics, robot))

    other_cards = {}
    for name, ent in task.entities.items():
        if name == target_name or name == "table" or name.startswith("card_holder"):
            continue
        try:
            pos = np.asarray(ent.get_xpos(physics), dtype=float)
            other_cards[name] = {
                "is_grasped": bool(ent.is_grasped(physics, robot)),
                "height": float(pos[2]),
                "ee_to_card_dist": float(np.linalg.norm(ee_pos - pos)),
            }
        except Exception:  # noqa: BLE001 — best-effort diagnostic only, never lets a step fail
            pass

    return {
        "step_ix": step_ix,
        "ee_pos": ee_pos.tolist(),
        "target_entity": target_name,
        "card_height": float(card_pos[2]),
        "ee_to_card_dist": float(np.linalg.norm(ee_pos - card_pos)),
        "is_grasped": is_grasped,
        "other_card_entities": other_cards,
    }


def nearest_card_at_step(step: dict) -> tuple[str, float]:
    """(card_name, dist) of whichever real card the EE was nearest to at this step, target card
    included."""
    candidates = {step["target_entity"]: step["ee_to_card_dist"]}
    for name, info in step["other_card_entities"].items():
        candidates[name] = info["ee_to_card_dist"]
    nearest = min(candidates, key=candidates.get)
    return nearest, candidates[nearest]


def install_env_instrumentation() -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": [], "step_ix": 0}

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            # See place_phase_forensics.py's docstring for why this must be guarded on
            # `seed is not None`: VLABenchEnv.step() also calls self.reset() (no seed=) internally
            # the instant success fires, and that call must not clobber the just-finished episode's
            # buffer before episode_callback has read it.
            state["current_episode_steps"] = [extract_physics_record(self, step_ix=0)]
            state["step_ix"] = 0
        return result

    def patched_step(self, action):
        result = orig_step(self, action)
        terminated = result[2]
        state["step_ix"] += 1
        if not terminated:
            state["current_episode_steps"].append(extract_physics_record(self, step_ix=state["step_ix"]))
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_env_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def episode_metrics(steps: list[dict], commanded_gripper_binary: list[bool]) -> dict:
    grasped_flags = [s["is_grasped"] for s in steps]
    ee_to_target_dists = [s["ee_to_card_dist"] for s in steps]

    never_grasped = not any(grasped_flags)
    target_card_grasped = any(grasped_flags)

    wrong_object_grasp = False
    for s in steps:
        for info in s["other_card_entities"].values():
            if info["is_grasped"] and info["height"] >= LIFT_THRESHOLD - NEAR_LIFT_MARGIN:
                wrong_object_grasp = True
                break
        if wrong_object_grasp:
            break

    min_ee_to_target_dist = min(ee_to_target_dists) if ee_to_target_dists else None

    first_close_step = commanded_gripper_binary.index(False) + 1 if False in commanded_gripper_binary else None
    ee_to_target_dist_at_close = (
        steps[first_close_step]["ee_to_card_dist"]
        if first_close_step is not None and first_close_step < len(steps)
        else None
    )

    # NEW metric: at the step where EE was closest to *any* real card, was that card actually the
    # target? A model with no card-identity information should land near chance here (whatever the
    # base rate of "target happens to be the geometrically nearest card" is for this scene
    # distribution); a genuinely instruction-grounded model should exceed it.
    approached_correct_card = None
    if steps:
        nearest_by_step = [nearest_card_at_step(s) for s in steps]
        min_step_ix = min(range(len(steps)), key=lambda i: nearest_by_step[i][1])
        nearest_name, nearest_dist = nearest_by_step[min_step_ix]
        approached_correct_card = nearest_name == steps[min_step_ix]["target_entity"]

    return {
        "target_card_grasp": target_card_grasped,
        "never_grasped": never_grasped,
        "wrong_object_grasp": wrong_object_grasp,
        "min_ee_to_correct_card_dist": min_ee_to_target_dist,
        "ee_to_correct_card_dist_at_close": ee_to_target_dist_at_close,
        "first_commanded_close_step": first_close_step,
        "approached_correct_card": approached_correct_card,
    }


def run(policy: SafeDiffVLAPolicy, device: str, n_episodes: int, start_seed: int, out_dir: Path) -> dict:
    # Also accepts `temporal_decoder_grounded_grasp` (the modality-aware-pooling checkpoint) --
    # this function reads only live physics/gripper state, nothing architecture-internal, so the
    # same corrected-instruction re-eval applies unchanged to either checkpoint.
    assert policy.config.architecture in ("temporal_decoder", "temporal_decoder_grounded_grasp"), policy.config.architecture
    assert policy.config.action_horizon == 50, policy.config.action_horizon
    assert policy.config.execute_horizon == 50, policy.config.execute_horizon
    assert policy.config.use_temporal_ensembling is False, policy.config.use_temporal_ensembling
    assert not hasattr(policy.config, "phase_conditioning")
    assert not hasattr(policy.config, "replan_on_gripper_close")
    logger.info(
        "condition confirmed: architecture=%s action_horizon=%d execute_horizon=%d "
        "use_temporal_ensembling=%s (no subgoal/phase/replan config on this checkpoint)",
        policy.config.architecture,
        policy.config.action_horizon,
        policy.config.execute_horizon,
        policy.config.use_temporal_ensembling,
    )
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
        commanded_gripper = rollout_data[ACTION][env_idx, :ep_len, GRIPPER_INDEX]
        commanded_binary = (commanded_gripper > GRIPPER_THRESHOLD).tolist()
        steps = list(inst_state["current_episode_steps"])  # copy before the next reset touches it
        episode_records.append(
            {
                "episode_ix": episode_ix,
                "episode_len": ep_len,
                "n_physics_steps_recorded": len(steps),
                "steps": steps,
                "commanded_gripper_binary": commanded_binary,
            }
        )

    videos_dir = out_dir / "videos"
    try:
        info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=n_episodes,
            videos_dir=videos_dir,
            start_seed=start_seed,
            episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(inst_state)

    video_paths = info.get("video_paths", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    per_episode_summary = []
    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        metrics = episode_metrics(rec["steps"], rec["commanded_gripper_binary"])
        entry = {
            "episode_ix": ep["episode_ix"],
            "seed": ep["seed"],
            "task_name": ep["task_name"],
            "instruction": ep["instruction"],
            "success": ep["success"],
            "episode_len": rec["episode_len"],
            "video_path": video_paths[ep["episode_ix"]] if ep["episode_ix"] < len(video_paths) else None,
            **metrics,
        }
        per_episode_summary.append(entry)
        rec["episode_ix"] = ep["episode_ix"]
        rec["seed"] = ep["seed"]
        rec["instruction"] = ep["instruction"]
        logger.info(
            "episode %d seed=%d instruction=%r success=%s target_card_grasp=%s wrong_object_grasp=%s "
            "min_ee_to_correct_card_dist=%.4f approached_correct_card=%s",
            entry["episode_ix"],
            entry["seed"],
            entry["instruction"],
            entry["success"],
            entry["target_card_grasp"],
            entry["wrong_object_grasp"],
            entry["min_ee_to_correct_card_dist"] or float("nan"),
            entry["approached_correct_card"],
        )

    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))

    n = len(per_episode_summary)
    aggregate = {
        "n_episodes": n,
        "n_success": sum(1 for e in per_episode_summary if e["success"]),
        "n_target_card_grasp": sum(1 for e in per_episode_summary if e["target_card_grasp"]),
        "n_wrong_object_grasp": sum(1 for e in per_episode_summary if e["wrong_object_grasp"]),
        "n_never_grasped": sum(1 for e in per_episode_summary if e["never_grasped"]),
        "n_approached_correct_card": sum(1 for e in per_episode_summary if e["approached_correct_card"]),
        "fraction_approached_correct_card": sum(1 for e in per_episode_summary if e["approached_correct_card"]) / n
        if n
        else None,
        "mean_min_ee_to_correct_card_dist": float(
            np.mean([e["min_ee_to_correct_card_dist"] for e in per_episode_summary if e["min_ee_to_correct_card_dist"] is not None])
        )
        if any(e["min_ee_to_correct_card_dist"] is not None for e in per_episode_summary)
        else None,
    }

    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "condition": {
            "architecture": policy.config.architecture,
            "action_horizon": policy.config.action_horizon,
            "execute_horizon": policy.config.execute_horizon,
            "use_temporal_ensembling": policy.config.use_temporal_ensembling,
            "ensembling_off": True,
            "subgoal_off": True,
            "phase_off": True,
            "replan_off": True,
        },
        "instruction_fix": "lerobot/envs/vlabench.py: VLABenchEnv.task_description now uses "
        "task_obj.get_instruction() (suite-prefixed) instead of falling back to the generic task name",
        "n_episodes": n_episodes,
        "start_seed": start_seed,
        "per_episode": per_episode_summary,
        "aggregate": aggregate,
    }
    summary["comparison_vs_preserved_buggy_baseline"] = build_comparison(per_episode_summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("=== aggregate: %s ===", aggregate)
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    return summary


def build_comparison(fixed_per_episode: list[dict]) -> dict:
    """Recompute the same metrics from the preserved pre-fix (buggy-instruction) runs' own
    already-recorded per-step traces -- both preserved runs used this exact checkpoint/task/seed
    range/condition, so no new buggy rollout needs to be generated to compare against."""
    forensics_path = Path(BUGGY_BASELINE_PLACE_FORENSICS)
    report_path = Path(BUGGY_BASELINE_REPORT)
    if not forensics_path.exists() or not report_path.exists():
        return {"available": False, "reason": "preserved pre-fix baseline files not found"}

    buggy_episodes_raw = json.loads(forensics_path.read_text())
    buggy_report = json.loads(report_path.read_text())
    buggy_per_episode_report = {e["seed"]: e for e in buggy_report["closed_loop_rollout"]["per_episode"]}

    buggy_summary = []
    for rec in buggy_episodes_raw:
        steps = rec["steps"]
        # place_phase_forensics.py's steps don't carry `other_card_entities[*]["ee_to_card_dist"]`
        # (added only in this script) -- so `approached_correct_card` can't be recomputed for the
        # buggy baseline from its saved trace; every other metric can.
        grasped_flags = [s["is_grasped"] for s in steps]
        ee_to_target_dists = [s["ee_to_card_dist"] for s in steps]
        wrong_object_grasp = False
        for s in steps:
            for info in s["other_card_entities"].values():
                if info["is_grasped"] and info["height"] >= LIFT_THRESHOLD - NEAR_LIFT_MARGIN:
                    wrong_object_grasp = True
        report_entry = buggy_per_episode_report.get(rec["seed"], {})
        buggy_summary.append(
            {
                "seed": rec["seed"],
                "success": rec["success"],
                "target_card_grasp": any(grasped_flags),
                "never_grasped": not any(grasped_flags),
                "wrong_object_grasp": wrong_object_grasp,
                "min_ee_to_correct_card_dist": min(ee_to_target_dists) if ee_to_target_dists else None,
                "buggy_instruction_seen_by_policy": "select_poker",  # verified generic fallback, see module docstring
                "gripper_transition_steps": report_entry.get("gripper_transition_steps"),
            }
        )

    n_buggy = len(buggy_summary)
    n_fixed = len(fixed_per_episode)
    return {
        "available": True,
        "preserved_buggy_baseline_files": [str(forensics_path), str(report_path)],
        "buggy_per_episode": buggy_summary,
        "buggy_aggregate": {
            "n_episodes": n_buggy,
            "n_success": sum(1 for e in buggy_summary if e["success"]),
            "n_target_card_grasp": sum(1 for e in buggy_summary if e["target_card_grasp"]),
            "n_wrong_object_grasp": sum(1 for e in buggy_summary if e["wrong_object_grasp"]),
            "n_never_grasped": sum(1 for e in buggy_summary if e["never_grasped"]),
            "mean_min_ee_to_correct_card_dist": float(
                np.mean([e["min_ee_to_correct_card_dist"] for e in buggy_summary if e["min_ee_to_correct_card_dist"] is not None])
            )
            if any(e["min_ee_to_correct_card_dist"] is not None for e in buggy_summary)
            else None,
        },
        "fixed_aggregate": {
            "n_episodes": n_fixed,
            "n_success": sum(1 for e in fixed_per_episode if e["success"]),
            "n_target_card_grasp": sum(1 for e in fixed_per_episode if e["target_card_grasp"]),
            "n_wrong_object_grasp": sum(1 for e in fixed_per_episode if e["wrong_object_grasp"]),
            "n_never_grasped": sum(1 for e in fixed_per_episode if e["never_grasped"]),
            "mean_min_ee_to_correct_card_dist": float(
                np.mean([e["min_ee_to_correct_card_dist"] for e in fixed_per_episode if e["min_ee_to_correct_card_dist"] is not None])
            )
            if any(e["min_ee_to_correct_card_dist"] is not None for e in fixed_per_episode)
            else None,
        },
    }


def main() -> None:
    global CHECKPOINT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/sincos_20k_padfix_instruction_fixed")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    logger.info("=== loading canonical checkpoint from %s (unmodified) ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    run(policy, args.device, args.n_episodes, args.start_seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
