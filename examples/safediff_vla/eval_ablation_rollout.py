#!/usr/bin/env python
"""Closed-loop rollout evaluation for the target-supervision/conditioning ablation (A: canonical
`temporal_decoder`, B: auxiliary-target-only, C: target-conditioned), all evaluated under
IDENTICAL conditions -- crucially, reactive close is DISABLED for all three (a no-op passthrough
wrap for B/C, and simply never present for A), so this measures the pose-trajectory/decoder's own
behavior only, not the separate reactive-close mechanism evaluated previously.

No model/loss/execution-policy/threshold/lambda code is touched -- this only reads live physics
state via the same non-invasive `VLABenchEnv.step`/`.reset` monkeypatch used throughout this
directory, plus (B/C only) a thin *instance-level* wrap of this one policy object's own
`_grounded_grasp_reactive_close` that always returns the action UNCHANGED (diagnostic passthrough,
logging the raw decoder gripper and predicted target for reporting only).

Outputs:
    outputs/eval/safediff_vla_ablation_<A|B|C>/episodes_raw.json
    outputs/eval/safediff_vla_ablation_<A|B|C>/summary.json
    outputs/eval/safediff_vla_ablation_<A|B|C>/videos/eval_episode_*.mp4

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_ablation_rollout.py --condition A --checkpoint ...
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

TASK = ppf.TASK
RENAME_MAP = ppf.RENAME_MAP
GRIPPER_THRESHOLD = ppf.GRIPPER_THRESHOLD


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def install_policy_passthrough_instrumentation(policy: SafeDiffVLAPolicy) -> dict:
    """B/C only: instance-level wrap of `_grounded_grasp_reactive_close` that ALWAYS returns the
    action unchanged (no reactive close in this ablation) while logging the raw decoder gripper
    value and predicted target for reporting -- observes, never changes, model output."""
    log: list[dict] = []

    def passthrough(self, action, current_state):
        original_gripper = float(action[0, GRIPPER_INDEX_RAW].item())
        predicted_target_world = None
        if self._grounded_grasp_last_target_xyz is not None:
            target_pos_m = self._grounded_grasp_last_target_xyz * self.action_pos_std + self.action_pos_mean
            predicted_target_world = target_pos_m[0].detach().cpu().numpy()  # robot-base frame
        log.append(
            {
                "original_gripper": original_gripper,
                "predicted_target_robot_frame": predicted_target_world.tolist() if predicted_target_world is not None else None,
            }
        )
        return action  # UNCHANGED -- reactive close disabled for this ablation

    policy._grounded_grasp_reactive_close = passthrough.__get__(policy, SafeDiffVLAPolicy)
    return {"log": log}


def uninstall_policy_instrumentation(policy: SafeDiffVLAPolicy) -> None:
    if "_grounded_grasp_reactive_close" in policy.__dict__:
        del policy.__dict__["_grounded_grasp_reactive_close"]


def install_env_instrumentation(policy_log: list[dict] | None) -> dict:
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": [], "step_ix": 0}

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            record = ppf.extract_physics_record(self, step_ix=0)
            state["current_episode_steps"] = [record]
            state["step_ix"] = 0
        return result

    def patched_step(self, action):
        base = robot_base(self)
        commanded_action = np.asarray(action, dtype=float).copy()
        result = orig_step(self, action)  # unmodified call
        terminated = result[2]
        chunk_step = state["step_ix"]
        state["step_ix"] += 1
        if not terminated:
            record = ppf.extract_physics_record(self, step_ix=state["step_ix"])
            record["commanded_gripper_value"] = float(commanded_action[GRIPPER_INDEX_RAW])
            record["chunk_id"] = chunk_step // EXECUTE_HORIZON
            record["chunk_offset"] = chunk_step % EXECUTE_HORIZON
            record["wrong_object_grasped_this_step"] = any(
                info.get("is_grasped") for info in record.get("other_card_entities", {}).values()
            )
            if policy_log:
                p = policy_log[-1]
                record["original_decoder_gripper"] = p["original_gripper"]
                if p["predicted_target_robot_frame"] is not None:
                    predicted_target_world = np.asarray(p["predicted_target_robot_frame"], dtype=float) + base
                    card_pos = np.asarray(record["card_pos"], dtype=float)
                    ee_pos = np.asarray(record["ee_pos"], dtype=float)
                    record["predicted_target_world"] = predicted_target_world.tolist()
                    record["predicted_target_dist_to_gt_card"] = float(np.linalg.norm(predicted_target_world - card_pos))
                    record["ee_to_predicted_target_dist"] = float(np.linalg.norm(ee_pos - predicted_target_world))
                else:
                    record["predicted_target_world"] = None
                    record["predicted_target_dist_to_gt_card"] = None
                    record["ee_to_predicted_target_dist"] = None
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
    first_actual_grasp_step = next((s["step_ix"] for s in steps if s["is_grasped"]), None)
    wrong_object_grasp = any(s.get("wrong_object_grasped_this_step") for s in steps)
    max_target_card_height = max((s["card_height"] for s in steps), default=None)

    close_step = next(
        (s["step_ix"] for s in steps[1:] if s.get("commanded_gripper_value", 1.0) <= GRIPPER_THRESHOLD), None
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
            "ee_to_card_dist": s["ee_to_card_dist"],
            "predicted_target_world": s.get("predicted_target_world"),
            "predicted_target_dist_to_gt_card": s.get("predicted_target_dist_to_gt_card"),
            "ee_to_predicted_target_dist": s.get("ee_to_predicted_target_dist"),
        }

    return {
        "seed": seed,
        "success": success,
        "grasp_success": grasp_success,
        "episode_len": episode_len,
        "first_actual_grasp_step": first_actual_grasp_step,
        "wrong_object_grasp": wrong_object_grasp,
        "max_target_card_height": max_target_card_height,
        "close_step": close_step,
        "dist_at_close": ref(close_step)["ee_to_card_dist"] if close_step is not None else None,
        "episode_min_ee_to_card_dist": min_step["ee_to_card_dist"],
        "episode_min_dist_step_ix": min_step["step_ix"],
        "at_close": ref(close_step),
        "at_min_distance": ref(min_step["step_ix"]),
        "video": str(video),
    }


def run(policy: SafeDiffVLAPolicy, checkpoint: str, device: str, n_episodes: int, start_seed: int, out_dir: Path) -> None:
    policy.config.execute_horizon = EXECUTE_HORIZON
    assert policy.config.action_horizon == EXECUTE_HORIZON
    assert policy.config.use_temporal_ensembling is False
    is_grounded = policy.config.architecture == "temporal_decoder_grounded_grasp"
    set_seed(start_seed)

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

    policy_inst = install_policy_passthrough_instrumentation(policy) if is_grounded else None
    policy_log = policy_inst["log"] if policy_inst else None
    env_state = install_env_instrumentation(policy_log)
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(env_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps})
        if policy_log is not None:
            policy_log.clear()
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
        if is_grounded:
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
        "execute_horizon": EXECUTE_HORIZON,
        "architecture": policy.config.architecture,
        "grounded_grasp_condition_decoder_on_target": getattr(policy.config, "grounded_grasp_condition_decoder_on_target", None),
        "reactive_close": "disabled (passthrough) for this ablation" if is_grounded else "n/a (architecture has no reactive close)",
        "n_episodes": n_episodes,
        "start_seed": start_seed,
        "per_episode": per_episode_summary,
        "aggregate": {
            "n_success": sum(1 for e in per_episode_summary if e["success"]),
            "n_grasp_success": sum(1 for e in per_episode_summary if e["grasp_success"]),
            "n_wrong_object_grasp": sum(1 for e in per_episode_summary if e["wrong_object_grasp"]),
            "mean_min_ee_to_card_dist": sum(e["episode_min_ee_to_card_dist"] for e in per_episode_summary) / len(per_episode_summary),
            "mean_dist_at_close": (
                sum(e["dist_at_close"] for e in per_episode_summary if e["dist_at_close"] is not None)
                / max(1, sum(1 for e in per_episode_summary if e["dist_at_close"] is not None))
            ),
        },
    }
    if is_grounded:
        pred_dists = [
            e["at_min_distance"]["predicted_target_dist_to_gt_card"]
            for e in per_episode_summary
            if e["at_min_distance"] and e["at_min_distance"]["predicted_target_dist_to_gt_card"] is not None
        ]
        summary["aggregate"]["mean_predicted_target_dist_to_gt_card_at_min"] = (
            sum(pred_dists) / len(pred_dists) if pred_dists else None
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    logger.info("aggregate: %s", json.dumps(summary["aggregate"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=["A", "B", "C"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or f"outputs/eval/safediff_vla_ablation_{args.condition}"

    logger.info("=== loading condition %s checkpoint from %s ===", args.condition, args.checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()

    run(policy, args.checkpoint, args.device, args.n_episodes, args.start_seed, Path(output_dir))


if __name__ == "__main__":
    main()
