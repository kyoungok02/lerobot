#!/usr/bin/env python
"""Closed-loop evaluation of the fresh 5k-step `temporal_decoder_grounded_grasp` checkpoint on
`select_poker` seeds 1000-1002, measurement-only, compared against the already-computed canonical
`temporal_decoder` (sincos_20k_padfix) baseline for the same seeds
(`outputs/eval/safediff_vla_grasp_approach_precision/summary.json` --
`grasp_approach_precision_closedloop.py`, not rerun here).

No model/loss/execution-policy code is touched by this script -- it only reads live physics state
via the same non-invasive `VLABenchEnv.step`/`.reset` monkeypatch as
`grasp_approach_precision_closedloop.py` (reused, not reimplemented), plus a thin *instance-level*
wrap of this one policy object's own `_grounded_grasp_reactive_close` method (to observe, not
change, its already-implemented proximity-trigger/one-shot-guard behavior and the decoder's raw
pre-override gripper channel).

Outputs:
    outputs/eval/safediff_vla_grounded_grasp_5k/episodes_raw.json
    outputs/eval/safediff_vla_grounded_grasp_5k/summary.json
    outputs/eval/safediff_vla_grounded_grasp_5k/videos/eval_episode_*.mp4

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_5k.py
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
from grasp_approach_precision_closedloop import EXECUTE_HORIZON  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_grounded_grasp_5k/checkpoints/005000/pretrained_model"
TASK = ppf.TASK
RENAME_MAP = ppf.RENAME_MAP
GRIPPER_THRESHOLD = ppf.GRIPPER_THRESHOLD
BASELINE_SUMMARY = Path("outputs/eval/safediff_vla_grasp_approach_precision/summary.json")


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def all_card_positions(env: VLABenchEnvImpl) -> dict[str, list[float]]:
    physics = env._env.physics
    task = env._env.task
    out = {}
    for name, ent in task.entities.items():
        if name == "table" or name.startswith("card_holder"):
            continue
        try:
            out[name] = np.asarray(ent.get_xpos(physics), dtype=float).tolist()
        except Exception:  # noqa: BLE001 - best-effort diagnostic only
            pass
    return out


def install_policy_instrumentation(policy: SafeDiffVLAPolicy) -> dict:
    """Instance-level wrap (not a class patch) of this one policy's own
    `_grounded_grasp_reactive_close` -- observes (doesn't change) its already-implemented
    behavior: the raw pre-override gripper value the decoder itself predicted, and whether the
    proximity trigger newly fired this call."""
    orig = SafeDiffVLAPolicy._grounded_grasp_reactive_close
    log: list[dict] = []

    def wrapped(self, action, current_state):
        original_gripper = float(action[0, GRIPPER_INDEX_RAW].item())
        was_triggered_before = self._grounded_grasp_close_triggered
        predicted_target_world = None
        if self._grounded_grasp_last_target_xyz is not None:
            target_pos_m = self._grounded_grasp_last_target_xyz * self.action_pos_std + self.action_pos_mean
            predicted_target_world = target_pos_m[0].detach().cpu().numpy()  # robot-base frame; +base -> world
        result = orig(self, action, current_state)
        log.append(
            {
                "original_gripper": original_gripper,
                "final_gripper": float(result[0, GRIPPER_INDEX_RAW].item()),
                "close_triggered_after": bool(self._grounded_grasp_close_triggered),
                "just_triggered_this_step": bool((not was_triggered_before) and self._grounded_grasp_close_triggered),
                "predicted_target_robot_frame": predicted_target_world.tolist() if predicted_target_world is not None else None,
            }
        )
        return result

    policy._grounded_grasp_reactive_close = wrapped.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log, "_original": orig}


def uninstall_policy_instrumentation(policy: SafeDiffVLAPolicy) -> None:
    if "_grounded_grasp_reactive_close" in policy.__dict__:
        del policy.__dict__["_grounded_grasp_reactive_close"]


def install_env_instrumentation(policy_log: list[dict]) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": [], "step_ix": 0}

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            record = ppf.extract_physics_record(self, step_ix=0)
            record["all_card_positions"] = all_card_positions(self)
            state["current_episode_steps"] = [record]
            state["step_ix"] = 0
        return result

    def patched_step(self, action):
        base = robot_base(self)
        result = orig_step(self, action)
        terminated = result[2]
        chunk_step = state["step_ix"]
        state["step_ix"] += 1
        if not terminated:
            record = ppf.extract_physics_record(self, step_ix=state["step_ix"])
            record["all_card_positions"] = all_card_positions(self)
            record["chunk_id"] = chunk_step // EXECUTE_HORIZON
            record["chunk_offset"] = chunk_step % EXECUTE_HORIZON

            # Merged in lockstep from the policy-side wrap: exactly one entry per env step,
            # appended by `select_action` immediately before this same step's `env.step()` call.
            if policy_log:
                p = policy_log[-1]
                record["original_decoder_gripper"] = p["original_gripper"]
                record["applied_gripper"] = p["final_gripper"]
                record["close_triggered_after_this_step"] = p["close_triggered_after"]
                record["proximity_trigger_fired_this_step"] = p["just_triggered_this_step"]
                if p["predicted_target_robot_frame"] is not None:
                    predicted_target_world = np.asarray(p["predicted_target_robot_frame"], dtype=float) + base
                    record["predicted_target_world"] = predicted_target_world.tolist()
                    card_pos = np.asarray(record["card_pos"], dtype=float)
                    ee_pos = np.asarray(record["ee_pos"], dtype=float)
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

            # wrong-object grasp signal: is any OTHER card entity currently grasped, per
            # `extract_physics_record`'s own contact query (independent of the target's own
            # `is_grasped`) -- and each entity's current height, for "max object/card z".
            record["wrong_object_grasped_this_step"] = any(
                info.get("is_grasped") for info in record.get("other_card_entities", {}).values()
            )
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
    heights_while_grasped = [s["card_height"] for s in steps if s["is_grasped"]]

    orig_close_step = next(
        (s["step_ix"] for s in steps[1:] if s.get("original_decoder_gripper", 1.0) <= GRIPPER_THRESHOLD), None
    )
    proximity_trigger_step = next((s["step_ix"] for s in steps if s.get("proximity_trigger_fired_this_step")), None)

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
            "nearest_card_to_predicted_target": s.get("nearest_card_to_predicted_target"),
            "target_is_nearest": s.get("target_is_nearest"),
            "predicted_target_dist_to_gt_card": s.get("predicted_target_dist_to_gt_card"),
            "ee_to_card_dist": s["ee_to_card_dist"],
        }

    target_is_nearest_values = [s["target_is_nearest"] for s in steps if s.get("target_is_nearest") is not None]

    return {
        "seed": seed,
        "success": success,
        "grasp_success": grasp_success,
        "lift_success": bool(heights_while_grasped) and max(heights_while_grasped) >= ppf.LIFT_THRESHOLD,
        "episode_len": episode_len,
        "original_decoder_close_step": orig_close_step,
        "proximity_trigger_step": proximity_trigger_step,
        "episode_min_ee_to_card_dist": min_step["ee_to_card_dist"],
        "episode_min_dist_step_ix": min_step["step_ix"],
        "at_first_chunk": ref(1),
        "at_proximity_trigger": ref(proximity_trigger_step),
        "at_min_distance": ref(min_step["step_ix"]),
        "fraction_steps_target_is_nearest": (
            sum(target_is_nearest_values) / len(target_is_nearest_values) if target_is_nearest_values else None
        ),
        "video": str(video),
    }


def run(policy: SafeDiffVLAPolicy, device: str, n_episodes: int, start_seed: int, out_dir: Path) -> None:
    policy.config.execute_horizon = EXECUTE_HORIZON
    assert policy.config.action_horizon == EXECUTE_HORIZON
    assert policy.config.use_temporal_ensembling is False
    assert policy.config.architecture == "temporal_decoder_grounded_grasp"
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
            n_episodes=n_episodes,
            max_episodes_rendered=n_episodes,
            videos_dir=videos_dir,
            start_seed=start_seed,
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

    baseline_by_seed = {}
    if BASELINE_SUMMARY.exists():
        baseline = json.loads(BASELINE_SUMMARY.read_text())
        baseline_by_seed = {e["seed"]: e for e in baseline["per_episode"]}

    comparison = []
    for e in per_episode_summary:
        b = baseline_by_seed.get(e["seed"])
        comparison.append(
            {
                "seed": e["seed"],
                "grounded_grasp_5k": {
                    "success": e["success"],
                    "grasp_success": e["grasp_success"],
                    "episode_min_ee_to_card_dist": e["episode_min_ee_to_card_dist"],
                    "fraction_steps_target_is_nearest": e["fraction_steps_target_is_nearest"],
                },
                "sincos_20k_padfix_baseline": (
                    {
                        "success": b["success"],
                        "pattern": b.get("classification", {}).get("pattern") if b.get("classification") else None,
                        "episode_min_ee_to_card_dist": b.get("episode_min_ee_to_card_dist"),
                    }
                    if b
                    else None
                ),
            }
        )

    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "execute_horizon": EXECUTE_HORIZON,
        "n_episodes": n_episodes,
        "start_seed": start_seed,
        "baseline_summary_source": str(BASELINE_SUMMARY) if baseline_by_seed else None,
        "per_episode": per_episode_summary,
        "vs_baseline": comparison,
        "aggregate": {
            "n_success": sum(1 for e in per_episode_summary if e["success"]),
            "n_grasp_success": sum(1 for e in per_episode_summary if e["grasp_success"]),
            "mean_min_ee_to_card_dist": sum(e["episode_min_ee_to_card_dist"] for e in per_episode_summary) / len(per_episode_summary),
            "n_approach_then_retreat_like": None,  # see per-episode "at_proximity_trigger" vs "at_min_distance" manually
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    logger.info("aggregate: %s", json.dumps(summary["aggregate"], indent=2))


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_5k")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint

    logger.info("=== loading temporal_decoder_grounded_grasp checkpoint from %s ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    run(policy, args.device, args.n_episodes, args.start_seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
