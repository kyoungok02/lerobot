#!/usr/bin/env python
"""Measurement-only grasp-approach precision forensics for canonical execute_horizon=50
select_poker rollouts (checkpoint: sincos_20k_padfix, seeds 1000-1009), extending
`place_phase_forensics.py`'s finding that all 9/10 failures are `never_grasped` with an
axis-resolved, chunk-aware trace of *why* the end effector never gets close enough.

No model, loss, execution-policy, or gripper-logic code is touched anywhere in this script --
it only reads live physics/env state via the same non-invasive `VLABenchEnv.step`/`.reset`
monkeypatch technique as `place_phase_forensics.py` (imported and reused directly, not
reimplemented), and additionally records the raw *commanded* action each step already passed
into `env.step()` by the unmodified eval loop.

Key fact this script leans on (verified against `VLABenchEnv._build_ctrl_from_action` in
`lerobot/envs/vlabench.py`): action_mode="eef" (VLABenchEnv's default, used here) means
`action[:3]` is an **absolute end-effector target position in robot-base frame**, IK-solved
fresh every single env step -- NOT a per-step delta. `pos_world = action[:3] + robot_base_xyz`.
This lets us decompose the residual EE<->card distance into two independent, directly
measurable pieces at every timestep:
  - target localization bias: commanded_target_world <-> card_pos_world distance/axis-error --
    is the policy even *aiming* at the right place?
  - execution/tracking drift: achieved ee_pos_world <-> commanded_target_world distance -- does
    the IK-solved pose actually reach where it was told to go, or does contact/joint-limit/
    stale-state behavior leave it short?
Both are logged every step, so "never_grasped" failures can be attributed to one, the other, or
both, without guessing.

Outputs:
    outputs/eval/safediff_vla_grasp_approach_precision/episodes_raw.json   (per-episode, per-step trace)
    outputs/eval/safediff_vla_grasp_approach_precision/summary.json        (close-event alignment,
        axis decomposition, overshoot/undershoot/lateral classification, success-vs-failure diff)

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/grasp_approach_precision_closedloop.py
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import place_phase_forensics as ppf  # noqa: E402  (reuse extract_physics_record, not reimplement)

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = ppf.CHECKPOINT  # outputs/train/safediff_vla_temporal_decoder_sincos_20k_padfix/.../020000
TASK = ppf.TASK  # select_poker
RENAME_MAP = ppf.RENAME_MAP
GRIPPER_INDEX = ppf.GRIPPER_INDEX  # 6
GRIPPER_THRESHOLD = ppf.GRIPPER_THRESHOLD  # 0.5 -- >threshold == open
EXECUTE_HORIZON = 50
ALIGN_WINDOW = 30  # steps before first commanded close to align across episodes
CONTACT_NEAR_THRESHOLD = 0.05  # 5cm -- "close enough that contact is plausible" (matches the
# user's own observation that successes reach ~2-4cm and failures stall at 7-27cm)
RETREAT_MARGIN = 0.02  # 2cm -- "distance grew back by more than this after the approach minimum"
RETREAT_LEAD = 3  # steps -- minimum gap between the distance minimum and the close attempt for it
# to count as "approached then retreated" rather than "still approaching when it closed"
LATERAL_DOMINANCE_RATIO = 1.5


def install_instrumentation() -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": [], "step_ix": 0}

    def robot_base(self) -> np.ndarray:
        base = getattr(self, "_robot_base_xyz", None)
        return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])

    def annotate(record: dict, commanded_action: np.ndarray | None, base: np.ndarray, chunk_step: int | None) -> None:
        card_pos = np.asarray(record["card_pos"], dtype=float)
        ee_pos = np.asarray(record["ee_pos"], dtype=float)
        ee_err = ee_pos - card_pos
        record["ee_axis_error"] = {"dx": float(ee_err[0]), "dy": float(ee_err[1]), "dz": float(ee_err[2])}
        if commanded_action is None:
            record["commanded_action_raw"] = None
            record["commanded_gripper_value"] = None
            record["commanded_target_world"] = None
            record["commanded_target_to_card_dist"] = None
            record["commanded_target_axis_error"] = None
            record["ee_tracking_error_to_commanded_target"] = None
            record["chunk_id"] = None
            record["chunk_offset"] = None
            return
        commanded_target_world = commanded_action[:3] + base
        cmd_err = commanded_target_world - card_pos
        record["commanded_action_raw"] = commanded_action.tolist()
        record["commanded_gripper_value"] = float(commanded_action[GRIPPER_INDEX])
        record["commanded_target_world"] = commanded_target_world.tolist()
        record["commanded_target_to_card_dist"] = float(np.linalg.norm(cmd_err))
        record["commanded_target_axis_error"] = {"dx": float(cmd_err[0]), "dy": float(cmd_err[1]), "dz": float(cmd_err[2])}
        record["ee_tracking_error_to_commanded_target"] = float(np.linalg.norm(ee_pos - commanded_target_world))
        record["chunk_id"] = chunk_step // EXECUTE_HORIZON
        record["chunk_offset"] = chunk_step % EXECUTE_HORIZON

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            record = ppf.extract_physics_record(self, step_ix=0)
            annotate(record, commanded_action=None, base=robot_base(self), chunk_step=None)
            state["current_episode_steps"] = [record]
            state["step_ix"] = 0
        return result

    def patched_step(self, action):
        base = robot_base(self)
        commanded_action = np.asarray(action, dtype=float).copy()
        result = orig_step(self, action)  # unmodified call, same args
        terminated = result[2]
        chunk_step = state["step_ix"]  # 0-indexed position in the executed action sequence
        state["step_ix"] += 1
        if not terminated:
            # See place_phase_forensics.py's module docstring: on the terminal (success) step,
            # `orig_step` has already reset for the next episode by the time control returns.
            record = ppf.extract_physics_record(self, step_ix=state["step_ix"])
            annotate(record, commanded_action=commanded_action, base=base, chunk_step=chunk_step)
            state["current_episode_steps"].append(record)
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def first_close_step(steps: list[dict]) -> int | None:
    """First step index (>=1, matching `chunk_step`+1 / `state["step_ix"]` after increment)
    whose commanded gripper value was <=GRIPPER_THRESHOLD (closed). Mirrors
    `place_phase_forensics.py`'s `first_commanded_close_step` definition exactly, recomputed here
    from the raw per-step value instead of the binary list, for cross-check."""
    for s in steps[1:]:
        if s["commanded_gripper_value"] is not None and s["commanded_gripper_value"] <= GRIPPER_THRESHOLD:
            return s["step_ix"]
    return None


def min_distance_step(steps: list[dict]) -> dict:
    idx = min(range(len(steps)), key=lambda i: steps[i]["ee_to_card_dist"])
    return steps[idx]


def alignment_window(steps: list[dict], close_step: int) -> list[dict]:
    by_step = {s["step_ix"]: s for s in steps}
    window = []
    for r in range(-ALIGN_WINDOW, 1):
        idx = close_step + r
        s = by_step.get(idx)
        window.append(
            {
                "relative_step": r,
                "step_ix": idx,
                "ee_to_card_dist": s["ee_to_card_dist"] if s else None,
                "ee_axis_error": s["ee_axis_error"] if s else None,
                "commanded_target_to_card_dist": s["commanded_target_to_card_dist"] if s else None,
                "commanded_target_axis_error": s["commanded_target_axis_error"] if s else None,
                "ee_tracking_error_to_commanded_target": s["ee_tracking_error_to_commanded_target"] if s else None,
                "commanded_gripper_value": s["commanded_gripper_value"] if s else None,
                "chunk_offset": s["chunk_offset"] if s else None,
            }
        )
    return window


def classify_approach(steps: list[dict], close_step: int, min_step: dict) -> dict:
    dist_at_close = next(s["ee_to_card_dist"] for s in steps if s["step_ix"] == close_step)
    dist_at_min = min_step["ee_to_card_dist"]
    min_step_ix = min_step["step_ix"]

    retreated = (close_step - min_step_ix >= RETREAT_LEAD) and (dist_at_close > dist_at_min + RETREAT_MARGIN)
    stalled_short = (not retreated) and dist_at_min > CONTACT_NEAR_THRESHOLD

    axis = min_step["ee_axis_error"]
    lateral = float(np.hypot(axis["dx"], axis["dy"]))
    vertical = abs(axis["dz"])
    if vertical < 1e-9:
        axis_dominance = "lateral_dominant" if lateral > 1e-9 else "comparable"
    elif lateral > vertical * LATERAL_DOMINANCE_RATIO:
        axis_dominance = "lateral_dominant"
    elif vertical > lateral * LATERAL_DOMINANCE_RATIO:
        axis_dominance = "vertical_dominant"
    else:
        axis_dominance = "comparable"

    return {
        "dist_at_close": dist_at_close,
        "dist_at_min": dist_at_min,
        "min_step_ix": min_step_ix,
        "steps_min_precedes_close_by": close_step - min_step_ix,
        "pattern": "approach_then_retreat_before_close" if retreated else (
            "stalls_short_of_contact" if stalled_short else "reaches_near_contact_at_close"
        ),
        "residual_axis_dominance_at_min": axis_dominance,
        "lateral_residual_m_at_min": lateral,
        "vertical_residual_m_at_min": vertical,
    }


def run(policy: SafeDiffVLAPolicy, device: str, n_episodes: int, start_seed: int, out_dir: Path) -> None:
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

    inst_state = install_instrumentation()
    episode_records: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(inst_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps})
        logger.info("  episode %d len=%d n_steps_recorded=%d", episode_ix, ep_len, len(steps))

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
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))

    per_episode_summary = []
    for rec in episode_records:
        steps = rec["steps"]
        close_step = first_close_step(steps)
        entry: dict[str, Any] = {"seed": rec["seed"], "success": rec["success"], "episode_len": rec["episode_len"], "close_step": close_step}
        if close_step is None:
            entry["note"] = "no commanded gripper closure at all in this episode"
            per_episode_summary.append(entry)
            continue

        by_step = {s["step_ix"]: s for s in steps}
        for lag, name in ((0, "at_close"), (5, "close_minus_5"), (10, "close_minus_10"), (20, "close_minus_20")):
            s = by_step.get(close_step - lag)
            entry[f"dist_{name}"] = s["ee_to_card_dist"] if s else None
            entry[f"axis_error_{name}"] = s["ee_axis_error"] if s else None
            entry[f"commanded_target_dist_{name}"] = s["commanded_target_to_card_dist"] if s else None

        mstep = min_distance_step(steps)
        entry["episode_min_ee_to_card_dist"] = mstep["ee_to_card_dist"]
        entry["episode_min_dist_step_ix"] = mstep["step_ix"]
        entry["episode_min_dist_chunk_id"] = mstep["chunk_id"]
        entry["episode_min_dist_chunk_offset"] = mstep["chunk_offset"]
        entry["axis_error_at_min"] = mstep["ee_axis_error"]
        entry["commanded_target_axis_error_at_min"] = mstep["commanded_target_axis_error"]
        entry["commanded_target_dist_at_min"] = mstep["commanded_target_to_card_dist"]
        entry["ee_tracking_error_at_min"] = mstep["ee_tracking_error_to_commanded_target"]

        entry["classification"] = classify_approach(steps, close_step, mstep)
        entry["alignment_window_pm30_before_close"] = alignment_window(steps, close_step)
        per_episode_summary.append(entry)

    success_entries = [e for e in per_episode_summary if e["success"] and "axis_error_at_min" in e]
    failure_entries = [e for e in per_episode_summary if not e["success"] and "axis_error_at_min" in e]

    def axis_stats(entries: list[dict], field: str, axis: str) -> dict:
        vals = [abs(e[field][axis]) for e in entries if e.get(field) is not None]
        return {"mean_abs": statistics.mean(vals) if vals else None, "n": len(vals)}

    cross_group = {
        "n_success": len(success_entries),
        "n_failure": len(failure_entries),
        "axis_error_at_min_abs_mean": {
            axis: {"success": axis_stats(success_entries, "axis_error_at_min", axis), "failure": axis_stats(failure_entries, "axis_error_at_min", axis)}
            for axis in ("dx", "dy", "dz")
        },
        "commanded_target_axis_error_at_min_abs_mean": {
            axis: {"success": axis_stats(success_entries, "commanded_target_axis_error_at_min", axis), "failure": axis_stats(failure_entries, "commanded_target_axis_error_at_min", axis)}
            for axis in ("dx", "dy", "dz")
        },
        "episode_min_dist": {
            "success": [e["episode_min_ee_to_card_dist"] for e in success_entries],
            "failure": [e["episode_min_ee_to_card_dist"] for e in failure_entries],
        },
        "commanded_target_dist_at_min": {
            "success": [e["commanded_target_dist_at_min"] for e in success_entries],
            "failure": [e["commanded_target_dist_at_min"] for e in failure_entries],
        },
        "ee_tracking_error_at_min": {
            "success": [e["ee_tracking_error_at_min"] for e in success_entries],
            "failure": [e["ee_tracking_error_at_min"] for e in failure_entries],
        },
        "failure_pattern_counts": {
            pattern: sum(1 for e in failure_entries if e["classification"]["pattern"] == pattern)
            for pattern in ("approach_then_retreat_before_close", "stalls_short_of_contact", "reaches_near_contact_at_close")
        },
        "failure_residual_axis_dominance_counts": {
            dom: sum(1 for e in failure_entries if e["classification"]["residual_axis_dominance_at_min"] == dom)
            for dom in ("lateral_dominant", "vertical_dominant", "comparable")
        },
    }

    # Which axis differs most between the success episode and the failure group, at the
    # global-min-distance step (the closest each episode ever got) -- both for the *achieved* EE
    # position (bias + drift combined) and for the *commanded target* alone (bias only, execution
    # noise excluded).
    def biggest_axis_gap(field: str) -> dict | None:
        if not success_entries or not failure_entries:
            return None
        succ = success_entries[0][field]
        gaps = {}
        for axis in ("dx", "dy", "dz"):
            fail_mean = statistics.mean(abs(e[field][axis]) for e in failure_entries if e.get(field))
            gaps[axis] = {"success_abs": abs(succ[axis]), "failure_mean_abs": fail_mean, "gap": fail_mean - abs(succ[axis])}
        biggest = max(gaps, key=lambda a: gaps[a]["gap"])
        return {"per_axis": gaps, "largest_gap_axis": biggest}

    cross_group["biggest_axis_gap_ee_at_min"] = biggest_axis_gap("axis_error_at_min")
    cross_group["biggest_axis_gap_commanded_target_at_min"] = biggest_axis_gap("commanded_target_axis_error_at_min")

    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "execute_horizon": EXECUTE_HORIZON,
        "action_horizon": EXECUTE_HORIZON,
        "n_episodes": n_episodes,
        "start_seed": start_seed,
        "action_space_note": (
            "action_mode='eef': action[:3] is an ABSOLUTE end-effector target in robot-base frame, "
            "IK-solved fresh every env step (verified in VLABenchEnv._build_ctrl_from_action); "
            "commanded_target_to_card_dist / commanded_target_axis_error isolate target-localization "
            "bias, and ee_tracking_error_to_commanded_target isolates execution/IK-tracking drift."
        ),
        "per_episode": per_episode_summary,
        "cross_group": cross_group,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    logger.info("cross_group: %s", json.dumps(cross_group, indent=2)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_grasp_approach_precision")
    args = parser.parse_args()

    logger.info("=== loading canonical checkpoint from %s (unmodified) ===", args.checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"

    run(policy, args.device, args.n_episodes, args.start_seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
