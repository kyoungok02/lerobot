#!/usr/bin/env python
"""Diagnostic-only oracle position-correction experiment to test whether the target-localization
bias found by `grasp_approach_precision_closedloop.py` (commanded EE target consistently offset
from the card, with IK-tracking error <1cm) is the *causal bottleneck* behind `never_grasped`
closed-loop failures, or whether close-timing/orientation/multimodality issues would remain even
with perfect xyz aim.

THIS IS A PRIVILEGED DIAGNOSTIC ORACLE, NOT A DEPLOYABLE METHOD. It uses the simulator's own live
target-card position -- information a real policy never has -- to override *only* the xyz
component of the action already produced by the unmodified canonical policy. No model weights,
architecture, loss, checkpoint, or `execution.ActionExecutor` queueing/replanning logic are
touched. Rotation (`action[3:6]`) and the gripper command (`action[6]`) are always passed through
exactly as the policy predicted them, in both conditions.

Two closed-loop conditions, same seeds, same everything else:
  - baseline: the policy's own `plan_action_chunk` output, executed unmodified (identical to
    `grasp_approach_precision_closedloop.py`'s own run).
  - oracle_xyz_correction: every env step, while the target card has not yet been grasped (live
    `is_grasped` contact query, updated after each step -- non-anticipatory, only ever uses the
    *current* physics state), the xyz component of the commanded action is replaced by the card's
    live world position (converted to the same robot-base frame `action[:3]` already uses -- see
    `grasp_approach_precision_closedloop.py`'s docstring on `VLABenchEnv._build_ctrl_from_action`).
    Once contact is detected, correction is disabled for the rest of the episode (the "grasp 이후
    correction 해제" requirement) and the policy's own xyz resumes.

Interpretation (not automated -- read the printed/saved numbers):
  - If oracle_xyz_correction substantially improves min EE<->card distance / grasp rate / success
    vs baseline (same seeds), target localization bias is a real causal bottleneck.
  - If it does NOT improve things much, the bottleneck is elsewhere (close timing, orientation,
    phase/multimodality ambiguity) and localization bias, while real, is not what's blocking grasp.

Outputs:
    outputs/eval/safediff_vla_grasp_oracle_correction/summary.json
    outputs/eval/safediff_vla_grasp_oracle_correction/episodes_raw.json
    outputs/eval/safediff_vla_grasp_oracle_correction/videos_baseline/eval_episode_*.mp4
    outputs/eval/safediff_vla_grasp_oracle_correction/videos_oracle_xyz_correction/eval_episode_*.mp4

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/grasp_approach_oracle_correction.py --start-seed 1000 --n-episodes 3
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

sys.path.insert(0, str(Path(__file__).parent))
import place_phase_forensics as ppf  # noqa: E402
from grasp_approach_precision_closedloop import (  # noqa: E402
    EXECUTE_HORIZON,
    classify_approach,
    min_distance_step,
)

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = ppf.CHECKPOINT  # canonical sincos_20k_padfix, unchanged
TASK = ppf.TASK
RENAME_MAP = ppf.RENAME_MAP
CONDITIONS = ("baseline", "oracle_xyz_correction")


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def install_instrumentation(mode: str) -> dict:
    assert mode in CONDITIONS
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": [], "step_ix": 0, "grasped_so_far": False}

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            record = ppf.extract_physics_record(self, step_ix=0)
            state["current_episode_steps"] = [record]
            state["step_ix"] = 0
            state["grasped_so_far"] = bool(record["is_grasped"])
        return result

    def patched_step(self, action):
        base = robot_base(self)
        original_action = np.asarray(action, dtype=float).copy()
        applied_action = original_action.copy()
        corrected_this_step = False
        if mode == "oracle_xyz_correction" and not state["grasped_so_far"]:
            # Privileged diagnostic only: read the LIVE (pre-step) simulator state for the true
            # target card position -- same quantity `extract_physics_record` calls `card_pos`
            # throughout the A analysis, so this is directly comparable. Only `applied_action[:3]`
            # (xyz) is touched; rotation/gripper (indices 3:7) are untouched, still the policy's
            # own prediction.
            physics = self._env.physics
            task = self._env.task
            target = task.entities[task.target_entity]
            card_pos_world = np.asarray(target.get_xpos(physics), dtype=float)
            applied_action[:3] = card_pos_world - base
            corrected_this_step = True

        result = orig_step(self, applied_action)  # same call shape as the unmodified eval loop
        terminated = result[2]
        chunk_step = state["step_ix"]
        state["step_ix"] += 1
        if not terminated:
            record = ppf.extract_physics_record(self, step_ix=state["step_ix"])
            card_pos = np.asarray(record["card_pos"], dtype=float)
            ee_pos = np.asarray(record["ee_pos"], dtype=float)
            ee_err = ee_pos - card_pos
            record["ee_axis_error"] = {"dx": float(ee_err[0]), "dy": float(ee_err[1]), "dz": float(ee_err[2])}

            orig_target_world = original_action[:3] + base
            pred_err = orig_target_world - card_pos
            record["predicted_target_axis_error_before_correction"] = {
                "dx": float(pred_err[0]),
                "dy": float(pred_err[1]),
                "dz": float(pred_err[2]),
            }
            record["oracle_corrected_this_step"] = corrected_this_step
            record["original_predicted_action_raw"] = original_action.tolist()
            record["applied_action_raw"] = applied_action.tolist()
            record["chunk_id"] = chunk_step // EXECUTE_HORIZON
            record["chunk_offset"] = chunk_step % EXECUTE_HORIZON
            state["current_episode_steps"].append(record)
            state["grasped_so_far"] = state["grasped_so_far"] or bool(record["is_grasped"])
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def run_condition(
    policy: SafeDiffVLAPolicy, device: str, n_episodes: int, start_seed: int, mode: str, videos_dir: Path
) -> list[dict]:
    policy.config.execute_horizon = EXECUTE_HORIZON
    assert policy.config.action_horizon == EXECUTE_HORIZON
    assert policy.config.use_temporal_ensembling is False
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

    inst_state = install_instrumentation(mode)
    episode_records: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(inst_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps})
        logger.info("  [%s] episode %d len=%d n_steps=%d", mode, episode_ix, ep_len, len(steps))

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
        uninstall_instrumentation(inst_state)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]
    return episode_records


def summarize_episode(rec: dict, mode: str, videos_dir: Path) -> dict:
    steps = rec["steps"]
    close_step = first_close_step_compat(steps)
    grasp_success = any(s["is_grasped"] for s in steps)
    entry: dict[str, Any] = {
        "seed": rec["seed"],
        "mode": mode,
        "success": rec["success"],
        "grasp_success": grasp_success,
        "episode_len": rec["episode_len"],
        "close_step": close_step,
        "video": str(videos_dir / f"eval_episode_{rec['episode_ix']}.mp4"),
    }
    mstep = min_distance_step(steps)
    entry["episode_min_ee_to_card_dist"] = mstep["ee_to_card_dist"]
    entry["episode_min_dist_step_ix"] = mstep["step_ix"]
    entry["axis_error_at_min"] = mstep["ee_axis_error"]
    if close_step is not None:
        by_step = {s["step_ix"]: s for s in steps}
        s = by_step[close_step]
        entry["dist_at_close"] = s["ee_to_card_dist"]
        entry["axis_error_at_close"] = s["ee_axis_error"]
        entry["classification"] = classify_approach(steps, close_step, mstep)
        entry["reaches_near_contact_at_close"] = entry["classification"]["pattern"] == "reaches_near_contact_at_close"
    else:
        entry["dist_at_close"] = None
        entry["axis_error_at_close"] = None
        entry["classification"] = None
        entry["reaches_near_contact_at_close"] = None

    if mode == "oracle_xyz_correction":
        corrected_steps = [s for s in steps if s.get("oracle_corrected_this_step")]
        if corrected_steps:
            before = [s["predicted_target_axis_error_before_correction"] for s in corrected_steps]
            entry["mean_predicted_bias_before_correction"] = {
                axis: sum(b[axis] for b in before) / len(before) for axis in ("dx", "dy", "dz")
            }
            entry["n_steps_corrected"] = len(corrected_steps)
            entry["n_steps_correction_active_until_grasp"] = len(corrected_steps)
        else:
            entry["mean_predicted_bias_before_correction"] = None
            entry["n_steps_corrected"] = 0
    return entry


def first_close_step_compat(steps: list[dict]) -> int | None:
    """Same definition as `grasp_approach_precision_closedloop.first_close_step`, but read from
    the applied action's gripper channel (index 6, untouched by the oracle correction in either
    condition) instead of a separate binary list."""
    for s in steps[1:]:
        applied = s.get("applied_action_raw")
        if applied is not None and applied[6] <= ppf.GRIPPER_THRESHOLD:
            return s["step_ix"]
    return None


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_grasp_oracle_correction")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    logger.info("=== loading canonical checkpoint from %s (unmodified) ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_episode_records = {}
    all_summaries = []
    for mode in CONDITIONS:
        videos_dir = out_dir / f"videos_{mode}"
        logger.info("=== running condition: %s (seeds %d..%d) ===", mode, args.start_seed, args.start_seed + args.n_episodes - 1)
        records = run_condition(policy, args.device, args.n_episodes, args.start_seed, mode, videos_dir)
        all_episode_records[mode] = records
        for rec in records:
            all_summaries.append(summarize_episode(rec, mode, videos_dir))

    (out_dir / "episodes_raw.json").write_text(json.dumps(all_episode_records, indent=2))

    by_seed: dict[int, dict] = {}
    for entry in all_summaries:
        by_seed.setdefault(entry["seed"], {})[entry["mode"]] = entry

    paired = []
    for seed, conditions in sorted(by_seed.items()):
        b, o = conditions.get("baseline"), conditions.get("oracle_xyz_correction")
        paired.append(
            {
                "seed": seed,
                "baseline": b,
                "oracle_xyz_correction": o,
                "min_dist_improved": (o["episode_min_ee_to_card_dist"] < b["episode_min_ee_to_card_dist"]) if b and o else None,
                "min_dist_delta": (o["episode_min_ee_to_card_dist"] - b["episode_min_ee_to_card_dist"]) if b and o else None,
                "grasp_success_flipped_to_true": (o["grasp_success"] and not b["grasp_success"]) if b and o else None,
                "task_success_flipped_to_true": (o["success"] and not b["success"]) if b and o else None,
            }
        )

    n = len(paired)
    n_grasp_baseline = sum(1 for p in paired if p["baseline"]["grasp_success"])
    n_grasp_oracle = sum(1 for p in paired if p["oracle_xyz_correction"]["grasp_success"])
    n_success_baseline = sum(1 for p in paired if p["baseline"]["success"])
    n_success_oracle = sum(1 for p in paired if p["oracle_xyz_correction"]["success"])

    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "execute_horizon": EXECUTE_HORIZON,
        "action_horizon": EXECUTE_HORIZON,
        "seeds": [p["seed"] for p in paired],
        "diagnostic_caveat": (
            "oracle_xyz_correction is a PRIVILEGED, NON-DEPLOYABLE diagnostic that reads the live "
            "simulator target-card position and overwrites only the xyz component of the policy's "
            "own commanded action, every step, until the first live grasp-contact detection (then "
            "reverts to the policy's own xyz for the rest of the episode). Rotation and gripper "
            "channels are always the unmodified policy prediction in both conditions. Model "
            "weights/architecture/loss/checkpoint and execution.ActionExecutor queueing/replanning "
            "logic are all untouched."
        ),
        "paired_results": paired,
        "aggregate": {
            "n_episodes": n,
            "n_grasp_success_baseline": n_grasp_baseline,
            "n_grasp_success_oracle": n_grasp_oracle,
            "n_task_success_baseline": n_success_baseline,
            "n_task_success_oracle": n_success_oracle,
            "mean_min_dist_baseline": sum(p["baseline"]["episode_min_ee_to_card_dist"] for p in paired) / n if n else None,
            "mean_min_dist_oracle": sum(p["oracle_xyz_correction"]["episode_min_ee_to_card_dist"] for p in paired) / n if n else None,
            "n_reaches_near_contact_at_close_baseline": sum(1 for p in paired if p["baseline"]["reaches_near_contact_at_close"]),
            "n_reaches_near_contact_at_close_oracle": sum(1 for p in paired if p["oracle_xyz_correction"]["reaches_near_contact_at_close"]),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("=== wrote %s and %s ===", out_dir / "summary.json", out_dir / "episodes_raw.json")
    logger.info("aggregate: %s", json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
