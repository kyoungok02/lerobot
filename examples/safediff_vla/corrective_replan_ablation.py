#!/usr/bin/env python
"""Pose-only corrective-replan ablation on the CANONICAL `temporal_decoder` (sincos+padfix)
checkpoint -- no retraining, no `TargetPointHead`/conditioning/reactive-close/phase-conditioning,
no changes to `execution.ActionExecutor` or any other production code.

Idea: the canonical policy commits a full 50-step open-loop chunk every `execute_horizon` steps.
`grasp_close_timing_oracle.py` showed the policy's *own* gripper-close timing, once xyz was
corrected, lagged far behind actual proximity (a privileged-oracle finding). This experiment asks
a *deployable* question instead: using only the model's OWN prediction (no simulator privilege),
can a single small corrective re-plan, triggered N steps before the CURRENTLY ACTIVE chunk's own
predicted first open->close step, recover some of that lost precision by replacing just the
xyz/rotation portion of the remaining trajectory with a fresh plan from the CURRENT image/state --
while leaving the gripper channel exactly as the original chunk already committed to (no
replan-induced flicker)?

Mechanism (this script only, not `execution.ActionExecutor`): `SafeDiffVLAPolicy.select_action`
is replaced at the INSTANCE level (not the class -- restored after the run) with a small custom
queue that:
  - on a fresh chunk (episode start or previous chunk exhausted): calls `plan_action_chunk` once,
    caches the 50-step [xyz, rotation, gripper] chunk, and finds `predicted_close_step` = the
    first position (in PHYSICAL units, via the same `postprocessor` the eval loop would apply
    anyway) where the gripper reads closed.
  - N steps before that position (once, per chunk): calls `plan_action_chunk` again with the
    CURRENT observation, and splices ONLY indices `[:6]` (xyz + raw rotation) of the remaining
    positions from the new plan into the cached chunk -- index `6` (gripper) is left completely
    untouched, still the ORIGINAL chunk's own value.
  - otherwise: dequeues the next cached position, exactly like the real executor would for
    `execute_horizon == action_horizon` (no ensembling, no gate).

Three conditions, same seeds: baseline (no corrective replan), N=10, N=5.

Outputs:
    outputs/eval/safediff_vla_corrective_replan_ablation/<condition>/episodes_raw.json
    outputs/eval/safediff_vla_corrective_replan_ablation/<condition>/summary.json
    outputs/eval/safediff_vla_corrective_replan_ablation/<condition>/videos/eval_episode_*.mp4

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/corrective_replan_ablation.py --condition baseline
    MUJOCO_GL=egl uv run python examples/safediff_vla/corrective_replan_ablation.py --condition N10
    MUJOCO_GL=egl uv run python examples/safediff_vla/corrective_replan_ablation.py --condition N5
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
from lerobot.scripts.lerobot_eval import eval_policy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = ppf.CHECKPOINT  # canonical sincos_20k_padfix, unchanged
TASK = ppf.TASK
RENAME_MAP = ppf.RENAME_MAP
GRIPPER_INDEX = ppf.GRIPPER_INDEX  # 6
GRIPPER_THRESHOLD = ppf.GRIPPER_THRESHOLD  # 0.5, physical-scale convention
CONDITION_N = {"baseline": None, "N10": 10, "N5": 5}


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def install_corrective_replan(policy: SafeDiffVLAPolicy, postprocessor, n_steps_before_close: int | None) -> dict:
    """Instance-level replacement of `select_action` (restored by `uninstall_corrective_replan`) --
    entirely bypasses `policy._executor`/`execution.ActionExecutor` for this experiment; no
    production execution code is read or modified. See module docstring for the exact mechanism."""
    orig_select_action = policy.select_action
    orig_reset = policy.reset
    state: dict[str, Any] = {}

    def reset_state() -> None:
        state["chunk"] = None  # [action_horizon, 7], NORMALIZED (pre-postprocessor) scale
        state["queue_idx"] = 0
        state["predicted_close_step"] = None
        state["corrective_done"] = False
        state["n_replans_this_episode"] = 0
        state["n_corrective_replans_this_episode"] = 0

    reset_state()

    def wrapped_reset() -> None:
        orig_reset()
        reset_state()

    def find_close_step(chunk: torch.Tensor) -> int | None:
        with torch.no_grad():
            physical = postprocessor(chunk.unsqueeze(0).cpu()).squeeze(0)
        is_open = physical[:, GRIPPER_INDEX] > GRIPPER_THRESHOLD
        closed_positions = (~is_open).nonzero(as_tuple=True)[0]
        return int(closed_positions[0].item()) if len(closed_positions) else None

    @torch.no_grad()
    def custom_select_action(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        policy.eval()
        chunk_exhausted = state["chunk"] is None or state["queue_idx"] >= state["chunk"].shape[0]
        if chunk_exhausted:
            actions, _ = policy.plan_action_chunk(batch)
            state["chunk"] = actions[0].clone()
            state["queue_idx"] = 0
            state["predicted_close_step"] = find_close_step(state["chunk"])
            state["corrective_done"] = False
            state["n_replans_this_episode"] += 1
        elif (
            n_steps_before_close is not None
            and not state["corrective_done"]
            and state["predicted_close_step"] is not None
            and state["queue_idx"] == state["predicted_close_step"] - n_steps_before_close
        ):
            new_actions, _ = policy.plan_action_chunk(batch)
            new_chunk = new_actions[0]
            remaining = state["chunk"].shape[0] - state["queue_idx"]
            k = min(remaining, new_chunk.shape[0])
            # Splice ONLY xyz + raw rotation [0:6]; index 6 (gripper) stays the ORIGINAL chunk's
            # own value -- the whole point is no gripper-channel disturbance from this replan.
            state["chunk"][state["queue_idx"] : state["queue_idx"] + k, :6] = new_chunk[:k, :6]
            state["corrective_done"] = True
            state["n_corrective_replans_this_episode"] += 1

        action = state["chunk"][state["queue_idx"]].unsqueeze(0).clone()
        state["queue_idx"] += 1
        return action

    policy.select_action = custom_select_action
    policy.reset = wrapped_reset
    return {"state": state, "_originals": (policy, orig_select_action, orig_reset)}


def uninstall_corrective_replan(handle: dict) -> None:
    policy, orig_select_action, orig_reset = handle["_originals"]
    policy.select_action = orig_select_action
    policy.reset = orig_reset


def install_env_instrumentation() -> dict:
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
        commanded_action = np.asarray(action, dtype=float).copy()
        result = orig_step(self, action)
        terminated = result[2]
        chunk_step = state["step_ix"]
        state["step_ix"] += 1
        if not terminated:
            record = ppf.extract_physics_record(self, step_ix=state["step_ix"])
            record["commanded_gripper_value"] = float(commanded_action[GRIPPER_INDEX])
            record["chunk_id"] = chunk_step // EXECUTE_HORIZON
            record["chunk_offset"] = chunk_step % EXECUTE_HORIZON
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


def count_gripper_transitions(steps: list[dict]) -> int:
    values = [s["commanded_gripper_value"] for s in steps[1:]]
    if len(values) < 2:
        return 0
    is_open = [v > GRIPPER_THRESHOLD for v in values]
    return sum(1 for a, b in zip(is_open[:-1], is_open[1:], strict=True) if a != b)


def summarize_episode(steps: list[dict], seed: int, success: bool, episode_len: int, video: Path, replan_counts: dict) -> dict:
    grasp_success = any(s["is_grasped"] for s in steps)
    wrong_object_grasp = any(s.get("wrong_object_grasped_this_step") for s in steps)
    close_step = next(
        (s["step_ix"] for s in steps[1:] if s.get("commanded_gripper_value", 1.0) <= GRIPPER_THRESHOLD), None
    )
    min_idx = min(range(len(steps)), key=lambda i: steps[i]["ee_to_card_dist"])
    dist_at_close = next((s["ee_to_card_dist"] for s in steps if s["step_ix"] == close_step), None) if close_step else None

    return {
        "seed": seed,
        "success": success,
        "grasp_success": grasp_success,
        "wrong_object_grasp": wrong_object_grasp,
        "episode_len": episode_len,
        "close_step": close_step,
        "dist_at_close": dist_at_close,
        "episode_min_ee_to_card_dist": steps[min_idx]["ee_to_card_dist"],
        "episode_min_dist_step_ix": steps[min_idx]["step_ix"],
        "gripper_transition_count": count_gripper_transitions(steps),
        "n_replans": replan_counts.get("n_replans_this_episode"),
        "n_corrective_replans": replan_counts.get("n_corrective_replans_this_episode"),
        "video": str(video),
    }


def run(policy: SafeDiffVLAPolicy, device: str, n_episodes: int, start_seed: int, condition: str, out_dir: Path) -> None:
    n_steps_before_close = CONDITION_N[condition]
    policy.config.execute_horizon = EXECUTE_HORIZON
    assert policy.config.action_horizon == EXECUTE_HORIZON
    assert policy.config.use_temporal_ensembling is False
    assert policy.config.architecture == "temporal_decoder"
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

    replan_handle = install_corrective_replan(policy, postprocessor, n_steps_before_close)
    env_state = install_env_instrumentation()
    episode_records: list[dict] = []
    videos_dir = out_dir / "videos"

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(env_state["current_episode_steps"])
        replan_counts = {
            "n_replans_this_episode": replan_handle["state"]["n_replans_this_episode"],
            "n_corrective_replans_this_episode": replan_handle["state"]["n_corrective_replans_this_episode"],
        }
        episode_records.append({"episode_ix": episode_ix, "episode_len": ep_len, "steps": steps, "replan_counts": replan_counts})
        logger.info(
            "  [%s] episode %d len=%d n_replans=%s n_corrective=%s",
            condition, episode_ix, ep_len, replan_counts.get("n_replans_this_episode"), replan_counts.get("n_corrective_replans_this_episode"),
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
            max_episodes_rendered=n_episodes,
            videos_dir=videos_dir,
            start_seed=start_seed,
            episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)
        uninstall_corrective_replan(replan_handle)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))

    per_episode_summary = [
        summarize_episode(
            rec["steps"], rec["seed"], rec["success"], rec["episode_len"],
            videos_dir / f"eval_episode_{rec['episode_ix']}.mp4", rec["replan_counts"],
        )
        for rec in episode_records
    ]

    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "condition": condition,
        "n_steps_before_close": n_steps_before_close,
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
            "mean_gripper_transition_count": sum(e["gripper_transition_count"] for e in per_episode_summary) / len(per_episode_summary),
            "total_corrective_replans": sum(e["n_corrective_replans"] or 0 for e in per_episode_summary),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s and %s", out_dir / "episodes_raw.json", out_dir / "summary.json")
    logger.info("aggregate [%s]: %s", condition, json.dumps(summary["aggregate"], indent=2))


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--condition", choices=list(CONDITION_N), required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint
    output_dir = args.output_dir or f"outputs/eval/safediff_vla_corrective_replan_ablation/{args.condition}"

    logger.info("=== loading canonical checkpoint from %s (unmodified) ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()

    run(policy, args.device, args.n_episodes, args.start_seed, args.condition, Path(output_dir))


if __name__ == "__main__":
    main()
