#!/usr/bin/env python
"""Instrumented closed-loop rollout for the canonical `sincos_20k_padfix` checkpoint that logs, per
episode, everything needed to directly measure two hypotheses about `execute_horizon`-driven
gripper dithering (see `execute_horizon_gripper_analysis.py`'s H1/H2):

  H1: at every chunk boundary (replan), does the *new* chunk's gripper intent disagree with what
      the *old* chunk had already committed to predicting for that same future timestep?
  H2 (canonical execute_horizon=50 only): do failures mostly happen *after* a successful grasp,
      during the place phase?

INSTRUMENTATION ONLY -- NO POLICY/EXECUTION CHANGE. The only hook is a monkeypatch of
`SafeDiffVLAPolicy.plan_action_chunk` that calls the original unmodified method, records a copy of
its exact return value, and returns that exact same (unmodified) value to the caller
(`execution.py::ActionExecutor`, itself also untouched). `plan_action_chunk` is called by
`ActionExecutor.select_action` if and only if its action queue is empty, i.e. precisely once per
replan (see `execution.py`, read but not edited) -- so every call this hook observes *is* a chunk
boundary, with no separate step-counting logic needed on this script's side. `policy.reset` is
also monkeypatched, purely to know when a new episode starts (to reset the per-episode chunk log),
again calling the original and changing nothing. Executed actions are read from `rollout_data`
returned by the *unmodified* `lerobot.scripts.lerobot_eval.eval_policy` / `rollout()` -- this
script adds no new execution-affecting code path of any kind. Validation (`--verify-against`)
cross-checks the newly recorded success/episode_len/transition timing against the earlier
`execute_horizon_ablation.json` run (same checkpoint, same seeds, no logging) to make that
"unchanged behavior" claim checkable rather than asserted.

Explicitly NOT implemented here (out of scope for this measurement pass): hysteresis, hold-after-
close, phase conditioning, event-triggered replanning, retraining, or any loss/architecture change.

Metrics computed (see module docstring sections below for exact definitions):
  - Replan consistency: `abs(new_chunk_gripper[0] - old_chunk_gripper[execute_horizon])`, binary
    open/close disagreement, disagreement -> actual-transition correspondence, boundary-to-
    transition distance.
  - Gripper-only smoothness: `mean_abs_delta_gripper` / `mean_abs_delta2_gripper` (exact, from the
    real executed per-step gripper channel -- no longer a transition-density proxy), transitions
    per episode / per replan / per 100 steps.
  - execute_horizon=50 failure-phase classification (never_grasped / grasped_then_dropped /
    grasp_held_but_place_failed / placed_but_task_not_registered-or-unknown), using the executed
    gripper trajectory plus the per-step reward trace already returned by `rollout_data["reward"]`
    -- `unknown` where the reward signal gives no distinguishing evidence, not a guess.

Outputs (one file per execute_horizon condition, to keep raw per-step logs manageable):
    outputs/eval/safediff_vla_replan_consistency/execute_horizon_{eh}/episodes_raw.json
    outputs/eval/safediff_vla_replan_consistency/execute_horizon_{eh}/summary.json

Usage:
    MUJOCO_GL=egl python examples/safediff_vla/replan_consistency_logging.py
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import torch
from torch import Tensor

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
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
REAL_HOLD_MIN_STEPS = 20  # same threshold as execute_horizon_gripper_analysis.py, for continuity
VERIFY_AGAINST = "outputs/eval/safediff_vla_execute_horizon_ablation/execute_horizon_ablation.json"


def install_instrumentation(policy: SafeDiffVLAPolicy) -> dict:
    """Monkeypatch `plan_action_chunk` (record-and-passthrough) and `reset` (episode-boundary
    marker only) on this policy *instance's class*. Returns a mutable `state` dict the caller
    reads `state["chunks"]` from inside its own `episode_callback` -- see module docstring for why
    this ordering is safe (the callback fires before the next episode's `reset`)."""
    cls = type(policy)
    orig_plan = cls.plan_action_chunk
    orig_reset = cls.reset
    state = {"chunks": [], "chunk_id": 0}

    def patched_reset(self):
        state["chunks"] = []
        state["chunk_id"] = 0
        return orig_reset(self)

    def patched_plan_action_chunk(self, batch):
        actions, metrics = orig_plan(self, batch)  # unmodified call
        state["chunks"].append(
            {
                "chunk_id": state["chunk_id"],
                "start_timestep": state["chunk_id"] * self.config.execute_horizon,
                "actions_normalized": actions.detach().cpu().clone(),
            }
        )
        state["chunk_id"] += 1
        return actions, metrics  # unmodified return value -- no transformation

    cls.reset = patched_reset
    cls.plan_action_chunk = patched_plan_action_chunk
    state["_originals"] = (cls, orig_plan, orig_reset)
    return state


def uninstall_instrumentation(state: dict) -> None:
    cls, orig_plan, orig_reset = state["_originals"]
    cls.plan_action_chunk = orig_plan
    cls.reset = orig_reset


def transitions_from_trajectory(gripper: Tensor) -> list[int]:
    side = (gripper > GRIPPER_THRESHOLD).to(torch.int8)
    changes = (side[1:] != side[:-1]).nonzero(as_tuple=True)[0]
    return [int(i) + 1 for i in changes.tolist()]


def gripper_smoothness(gripper: Tensor) -> dict:
    if gripper.shape[0] < 2:
        return {"mean_abs_delta_gripper": float("nan"), "mean_abs_delta2_gripper": float("nan")}
    velocity = gripper[1:] - gripper[:-1]
    mad = velocity.abs().mean().item()
    if gripper.shape[0] < 3:
        return {"mean_abs_delta_gripper": mad, "mean_abs_delta2_gripper": float("nan")}
    accel = velocity[1:] - velocity[:-1]
    mad2 = accel.abs().mean().item()
    return {"mean_abs_delta_gripper": mad, "mean_abs_delta2_gripper": mad2}


def failure_phase(gripper: Tensor, reward: Tensor, episode_len: int) -> dict:
    """4-way classification of a failed episode's gripper behavior, using only the executed
    gripper channel and the per-step reward trace `eval_policy`/`rollout()` already returns
    (`rollout_data["reward"]`) -- nothing guessed. `success_condition_mismatch` is only ever
    asserted as `unknown` here unless the reward trace itself shows a nonzero value that never
    triggered the binary success flag (this run's reward turned out to be purely sparse/binary in
    every episode inspected -- see summary.json's `reward_trace_is_purely_binary` field, computed
    per condition -- so this script never has positive evidence for that 4th category; it is
    included in the schema for a future run where a shaped reward is available)."""
    transitions = transitions_from_trajectory(gripper)
    if not transitions:
        return {"phase": "never_grasped", "evidence": "gripper never crossed the open/close threshold"}

    side = (gripper > GRIPPER_THRESHOLD).to(torch.int8)
    bounds = [0, *transitions, episode_len]
    segments = [
        {"state": "closed" if i % 2 == 1 else "open", "start": bounds[i], "end": bounds[i + 1], "duration": bounds[i + 1] - bounds[i]}
        for i in range(len(bounds) - 1)
    ]
    closed_durations = [s["duration"] for s in segments if s["state"] == "closed"]
    ends_closed = segments[-1]["state"] == "closed"
    real_hold = max(closed_durations, default=0) >= REAL_HOLD_MIN_STEPS

    reward_nonzero_ever = bool((reward != 0).any().item()) if reward.numel() else False

    if not real_hold:
        return {
            "phase": "never_grasped",
            "evidence": f"longest closed segment was {max(closed_durations, default=0)} steps (< {REAL_HOLD_MIN_STEPS})",
        }
    if not ends_closed:
        return {
            "phase": "grasped_then_dropped",
            "evidence": f"held closed for {max(closed_durations)} steps, then reopened at step {segments[-2]['end']} and never reclosed",
        }
    # Held a real grasp through to episode end but the episode still isn't `success`. Gripper
    # mechanics look fine; the only automatable evidence this script has for *why* the task-level
    # success check never fired is whether the reward trace ever showed any partial/shaped signal.
    if reward_nonzero_ever:
        return {
            "phase": "unknown",
            "evidence": "held grasp to episode end; reward trace had a nonzero value that never triggered success -- "
            "possible near-miss / success-condition mismatch, but this script cannot confirm without task-specific "
            "success-check introspection",
        }
    return {
        "phase": "grasp_held_but_place_failed",
        "evidence": f"held closed for the final {max(closed_durations)} steps to episode end; reward trace was all-zero "
        "throughout (no partial-credit signal observed), consistent with the object simply never reaching the "
        "task's target placement -- but this is inferred from absence of a shaped-reward signal, not confirmed "
        "against VLABench's own success-check logic",
    }


def run_condition(policy: SafeDiffVLAPolicy, eh: int, device: str, n_episodes: int, start_seed: int, out_dir: Path) -> dict:
    policy.config.execute_horizon = eh  # eval-time-only override, same mechanism as execute_horizon_ablation.py
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

    inst_state = install_instrumentation(policy)
    episode_records: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        executed_action = rollout_data[ACTION][env_idx, :ep_len].clone()  # raw/env-scale, exactly what was sent to env.step
        reward = rollout_data["reward"][env_idx, :ep_len].clone()
        gripper = executed_action[:, GRIPPER_INDEX]

        chunks_raw = []
        for c in inst_state["chunks"]:
            raw = postprocessor(c["actions_normalized"].clone()).squeeze(0)  # [50, 7], env-scale
            chunks_raw.append({"chunk_id": c["chunk_id"], "start_timestep": c["start_timestep"], "gripper_50step_raw": raw[:, GRIPPER_INDEX].tolist()})

        replan_consistency = []
        for i in range(1, len(chunks_raw)):
            new_c, old_c = chunks_raw[i], chunks_raw[i - 1]
            boundary_t = new_c["start_timestep"]
            offset = boundary_t - old_c["start_timestep"]  # == execute_horizon, by construction
            if offset >= len(old_c["gripper_50step_raw"]) or boundary_t >= ep_len:
                continue
            new_g0 = new_c["gripper_50step_raw"][0]
            old_g_at_boundary = old_c["gripper_50step_raw"][offset]
            executed_at_boundary = gripper[boundary_t].item() if boundary_t < ep_len else None
            new_binary = new_g0 > GRIPPER_THRESHOLD
            old_binary = old_g_at_boundary > GRIPPER_THRESHOLD
            transitions = transitions_from_trajectory(gripper)
            dist_to_transition = min((abs(boundary_t - t) for t in transitions), default=None)
            replan_consistency.append(
                {
                    "chunk_id": new_c["chunk_id"],
                    "boundary_timestep": boundary_t,
                    "new_plan_gripper_first": new_g0,
                    "old_plan_gripper_at_offset": old_g_at_boundary,
                    "offset_used": offset,
                    "abs_diff": abs(new_g0 - old_g_at_boundary),
                    "new_binary_open": new_binary,
                    "old_binary_open": old_binary,
                    "binary_disagree": new_binary != old_binary,
                    "executed_gripper_at_boundary": executed_at_boundary,
                    "matches_new_plan_first_step": (abs(executed_at_boundary - new_g0) < 1e-4) if executed_at_boundary is not None else None,
                    "actual_transition_at_boundary": boundary_t in transitions,
                    "distance_to_nearest_actual_transition": dist_to_transition,
                }
            )

        smooth_7d = {
            "mean_abs_delta_action": None,  # filled from eval_policy's own per-episode info below
        }
        smooth_gripper = gripper_smoothness(gripper)
        transitions_exec = transitions_from_trajectory(gripper)
        n_replans = max(len(chunks_raw) - 1, 0)

        record = {
            "episode_ix": episode_ix,
            "execute_horizon": eh,
            "episode_len": ep_len,
            "n_chunks": len(chunks_raw),
            "n_replans": n_replans,
            "executed_action_7d": executed_action.tolist(),
            "executed_gripper": gripper.tolist(),
            "reward_trace": reward.tolist(),
            "chunks": chunks_raw,
            "replan_consistency": replan_consistency,
            "gripper_transitions_exact": transitions_exec,
            "gripper_smoothness": smooth_gripper,
            "gripper_transitions_per_episode": len(transitions_exec),
            "gripper_transitions_per_replan": (len(transitions_exec) / n_replans) if n_replans else None,
            "gripper_transitions_per_100_steps": len(transitions_exec) / ep_len * 100,
        }
        episode_records.append(record)
        logger.info(
            "  episode %d len=%d n_chunks=%d gripper_trans=%d disagreements=%d/%d",
            episode_ix,
            ep_len,
            len(chunks_raw),
            len(transitions_exec),
            sum(1 for r in replan_consistency if r["binary_disagree"]),
            len(replan_consistency),
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
        uninstall_instrumentation(inst_state)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]
        rec["mean_abs_delta_action_7d"] = None
        actions = torch.tensor(rec["executed_action_7d"])
        if actions.shape[0] > 1:
            velocity = actions[1:] - actions[:-1]
            rec["mean_abs_delta_action_7d"] = velocity.abs().mean().item()
            if actions.shape[0] > 2:
                accel = velocity[1:] - velocity[:-1]
                rec["mean_abs_delta2_action_7d"] = accel.abs().mean().item()
            else:
                rec["mean_abs_delta2_action_7d"] = None
        if not rec["success"]:
            gripper_t = torch.tensor(rec["executed_gripper"])
            reward_t = torch.tensor(rec["reward_trace"])
            rec["failure_phase"] = failure_phase(gripper_t, reward_t, rec["episode_len"])
        else:
            rec["failure_phase"] = None

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes_raw.json").write_text(json.dumps(episode_records, indent=2))

    n_success = sum(1 for r in episode_records if r["success"])
    all_replans = [rc for rec in episode_records for rc in rec["replan_consistency"]]
    disagree = [rc for rc in all_replans if rc["binary_disagree"]]
    agree = [rc for rc in all_replans if not rc["binary_disagree"]]

    def transition_rate(subset: list[dict]) -> float | None:
        if not subset:
            return None
        return sum(1 for rc in subset if rc["actual_transition_at_boundary"]) / len(subset)

    summary = {
        "execute_horizon": eh,
        "n_episodes": n_episodes,
        "n_success": n_success,
        "reward_trace_is_purely_binary": all(set(r) <= {0.0, 1.0} for rec in episode_records for r in [set(rec["reward_trace"])]),
        "replan_consistency": {
            "n_replans_total": len(all_replans),
            "n_binary_disagreements": len(disagree),
            "disagreement_rate": (len(disagree) / len(all_replans)) if all_replans else None,
            "mean_abs_diff": statistics.mean(rc["abs_diff"] for rc in all_replans) if all_replans else None,
            "mean_abs_diff_when_disagree": statistics.mean(rc["abs_diff"] for rc in disagree) if disagree else None,
            "mean_abs_diff_when_agree": statistics.mean(rc["abs_diff"] for rc in agree) if agree else None,
            "p_actual_transition_at_boundary_given_disagree": transition_rate(disagree),
            "p_actual_transition_at_boundary_given_agree": transition_rate(agree),
            "mean_dist_to_transition_given_disagree": statistics.mean(rc["distance_to_nearest_actual_transition"] for rc in disagree if rc["distance_to_nearest_actual_transition"] is not None) if any(rc["distance_to_nearest_actual_transition"] is not None for rc in disagree) else None,
            "mean_dist_to_transition_given_agree": statistics.mean(rc["distance_to_nearest_actual_transition"] for rc in agree if rc["distance_to_nearest_actual_transition"] is not None) if any(rc["distance_to_nearest_actual_transition"] is not None for rc in agree) else None,
            "executed_matches_new_plan_first_step_rate": (sum(1 for rc in all_replans if rc["matches_new_plan_first_step"]) / len(all_replans)) if all_replans else None,
        },
        "gripper_smoothness": {
            "mean_abs_delta_gripper": statistics.mean(r["gripper_smoothness"]["mean_abs_delta_gripper"] for r in episode_records),
            "mean_abs_delta2_gripper": statistics.mean(r["gripper_smoothness"]["mean_abs_delta2_gripper"] for r in episode_records if r["gripper_smoothness"]["mean_abs_delta2_gripper"] == r["gripper_smoothness"]["mean_abs_delta2_gripper"]),
            "mean_abs_delta_action_7d": statistics.mean(r["mean_abs_delta_action_7d"] for r in episode_records if r["mean_abs_delta_action_7d"] is not None),
            "mean_abs_delta2_action_7d": statistics.mean(r["mean_abs_delta2_action_7d"] for r in episode_records if r.get("mean_abs_delta2_action_7d") is not None),
            "mean_transitions_per_episode": statistics.mean(r["gripper_transitions_per_episode"] for r in episode_records),
            "mean_transitions_per_replan": statistics.mean(r["gripper_transitions_per_replan"] for r in episode_records if r["gripper_transitions_per_replan"] is not None),
            "mean_transitions_per_100_steps": statistics.mean(r["gripper_transitions_per_100_steps"] for r in episode_records),
        },
        "failure_phase_counts": {
            phase: sum(1 for r in episode_records if r["failure_phase"] is not None and r["failure_phase"]["phase"] == phase)
            for phase in ("never_grasped", "grasped_then_dropped", "grasp_held_but_place_failed", "unknown")
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("overall eh=%d: success=%d/%d disagreement_rate=%s", eh, n_success, n_episodes, summary["replan_consistency"]["disagreement_rate"])
    return summary


def verify_unchanged(new_records: list[dict], eh: int) -> dict:
    """Cross-check against the earlier (non-instrumented) execute_horizon_ablation.json run for
    the same checkpoint/seeds -- if instrumentation truly added no transformation, success,
    episode_len, and transition timing must match exactly."""
    old_path = Path(VERIFY_AGAINST)
    if not old_path.exists():
        return {"checked": False, "reason": f"{old_path} not found"}
    old = json.loads(old_path.read_text())
    old_eps = {ep["seed"]: ep for ep in old["conditions"][str(eh)]["closed_loop_rollout"]["per_episode"]}
    mismatches = []
    for rec in new_records:
        old_ep = old_eps.get(rec["seed"])
        if old_ep is None:
            mismatches.append({"seed": rec["seed"], "issue": "no matching old episode"})
            continue
        if old_ep["success"] != rec["success"]:
            mismatches.append({"seed": rec["seed"], "issue": "success mismatch", "old": old_ep["success"], "new": rec["success"]})
        if old_ep["episode_len"] != rec["episode_len"]:
            mismatches.append({"seed": rec["seed"], "issue": "episode_len mismatch", "old": old_ep["episode_len"], "new": rec["episode_len"]})
        if old_ep["gripper_transition_steps"] != rec["gripper_transitions_exact"]:
            mismatches.append(
                {"seed": rec["seed"], "issue": "gripper_transition_steps mismatch", "old": old_ep["gripper_transition_steps"], "new": rec["gripper_transitions_exact"]}
            )
    return {"checked": True, "n_compared": len(new_records), "n_mismatches": len(mismatches), "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--execute-horizons", default="50,20,10")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_replan_consistency")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    logger.info("=== loading canonical checkpoint from %s (unmodified) ===", args.checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"
    assert policy.config.use_temporal_ensembling is False
    assert not hasattr(policy.config, "phase_conditioning")
    assert not hasattr(policy.config, "replan_on_gripper_close")

    all_verification = {}
    for eh in [int(x) for x in args.execute_horizons.split(",")]:
        logger.info("\n=== execute_horizon=%d ===", eh)
        out_dir = out_root / f"execute_horizon_{eh}"
        summary = run_condition(policy, eh, args.device, args.n_episodes, args.start_seed, out_dir)
        records = json.loads((out_dir / "episodes_raw.json").read_text())
        verification = verify_unchanged(records, eh)
        all_verification[str(eh)] = verification
        logger.info("verification vs prior non-instrumented run: %s", verification)

    (out_root / "verification_against_prior_run.json").write_text(json.dumps(all_verification, indent=2))
    logger.info("\n=== done. wrote results under %s ===", out_root)


if __name__ == "__main__":
    main()
