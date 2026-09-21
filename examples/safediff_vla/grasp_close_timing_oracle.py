#!/usr/bin/env python
"""Diagnostic-only close-TIMING oracle, following up on `grasp_approach_oracle_correction.py`'s
finding that xyz-only oracle correction fixes contact (3/3 grasp_success) but not task success
(1/3, same as baseline, with seed 1001 even flipping from success to failure). This experiment
asks: given contact is reached, is *when the gripper closes* the remaining bottleneck?

THIS IS A PRIVILEGED DIAGNOSTIC ORACLE, NOT A DEPLOYABLE METHOD -- same status as
`grasp_approach_oracle_correction.py`. No model weights, architecture, loss, checkpoint, or
`execution.ActionExecutor` queueing/replanning logic are touched. Only specific action-vector
components are overwritten at the `env.step()` boundary, using the simulator's own live state.

Three conditions, same seeds, same everything else:
  - baseline: policy's own `plan_action_chunk` output, executed unmodified.
  - xyz_oracle_only: identical to `grasp_approach_oracle_correction.py`'s `oracle_xyz_correction`
    condition -- action[:3] replaced by the live card position (robot-base frame) every step
    until first live contact, then reverts to the policy's own xyz. Gripper/rotation always the
    policy's own prediction.
  - xyz_oracle_plus_close_oracle: same xyz-oracle behavior as above, PLUS: starting the step
    *after* first live contact is detected (`Entity.is_grasped`, pure MuJoCo contact query -- the
    same signal `place_phase_forensics.py` uses as ground truth), the gripper channel
    (`action[6]`) is forced to 0.0 (fully closed, per `VLABenchEnv._build_ctrl_from_action`'s
    `np.clip(action[6], 0.0, 1.0)`, 0=closed/1=open) for the REST of the episode (a single
    trigger + indefinite hold, chosen to avoid introducing a second arbitrary "how long to hold"
    parameter). Rotation is always the policy's own prediction in every condition.

Non-anticipatory by construction: both interventions only ever use physics state already observed
as of the *previous* step (xyz correction gates off, and close-forcing gates on, one step after
the live `is_grasped` reading that triggers them) -- never a future value.

Interpretation (not automated -- read the printed/saved numbers):
  - If xyz_oracle_plus_close_oracle raises task success well above xyz_oracle_only, the remaining
    bottleneck is gripper-close timing / grasp<->lift phase transition.
  - If it still doesn't raise task success, look at orientation or post-grasp lift/transport
    behavior instead.

Outputs:
    outputs/eval/safediff_vla_grasp_close_timing_oracle/summary.json
    outputs/eval/safediff_vla_grasp_close_timing_oracle/episodes_raw.json
    outputs/eval/safediff_vla_grasp_close_timing_oracle/videos_<condition>/eval_episode_*.mp4

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/grasp_close_timing_oracle.py --start-seed 1000 --n-episodes 3
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

CHECKPOINT = ppf.CHECKPOINT
TASK = ppf.TASK
RENAME_MAP = ppf.RENAME_MAP
LIFT_THRESHOLD = ppf.LIFT_THRESHOLD  # 0.9m, VLABench's own SelectPokerTask condition_config
NEAR_CONTACT_TOLERANCE = 0.05  # 5cm -- same "plausible contact distance" constant used throughout
# this directory's analyses (grasp_approach_precision_closedloop.py's CONTACT_NEAR_THRESHOLD)
FORCED_CLOSED_GRIPPER_VALUE = 0.0  # per VLABenchEnv._build_ctrl_from_action's clip(action[6],0,1)
CONDITIONS = ("baseline", "xyz_oracle_only", "xyz_oracle_plus_close_oracle")
USES_XYZ_ORACLE = {"baseline": False, "xyz_oracle_only": True, "xyz_oracle_plus_close_oracle": True}
USES_CLOSE_ORACLE = {"baseline": False, "xyz_oracle_only": False, "xyz_oracle_plus_close_oracle": True}


def robot_base(env: VLABenchEnvImpl) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def install_instrumentation(mode: str) -> dict:
    assert mode in CONDITIONS
    use_xyz_oracle = USES_XYZ_ORACLE[mode]
    use_close_oracle = USES_CLOSE_ORACLE[mode]
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {
        "current_episode_steps": [],
        "step_ix": 0,
        "contact_ever_seen": False,  # gates xyz-oracle off / close-oracle on, from the NEXT step
        "close_forcing_active": False,
        "first_contact_step": None,
        "first_near_contact_step": None,
    }

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            record = ppf.extract_physics_record(self, step_ix=0)
            state["current_episode_steps"] = [record]
            state["step_ix"] = 0
            state["contact_ever_seen"] = bool(record["is_grasped"])
            state["close_forcing_active"] = False
            state["first_contact_step"] = 0 if record["is_grasped"] else None
            state["first_near_contact_step"] = 0 if record["ee_to_card_dist"] <= NEAR_CONTACT_TOLERANCE else None
        return result

    def patched_step(self, action):
        base = robot_base(self)
        original_action = np.asarray(action, dtype=float).copy()
        applied_action = original_action.copy()
        xyz_corrected_this_step = False
        close_forced_this_step = False

        if use_xyz_oracle and not state["contact_ever_seen"]:
            physics = self._env.physics
            task = self._env.task
            target = task.entities[task.target_entity]
            card_pos_world = np.asarray(target.get_xpos(physics), dtype=float)
            applied_action[:3] = card_pos_world - base
            xyz_corrected_this_step = True

        if use_close_oracle and state["close_forcing_active"]:
            applied_action[6] = FORCED_CLOSED_GRIPPER_VALUE
            close_forced_this_step = True

        result = orig_step(self, applied_action)
        terminated = result[2]
        chunk_step = state["step_ix"]
        state["step_ix"] += 1
        if not terminated:
            record = ppf.extract_physics_record(self, step_ix=state["step_ix"])
            card_pos = np.asarray(record["card_pos"], dtype=float)
            ee_pos = np.asarray(record["ee_pos"], dtype=float)
            ee_err = ee_pos - card_pos
            record["ee_axis_error"] = {"dx": float(ee_err[0]), "dy": float(ee_err[1]), "dz": float(ee_err[2])}
            record["original_predicted_action_raw"] = original_action.tolist()
            record["applied_action_raw"] = applied_action.tolist()
            record["xyz_corrected_this_step"] = xyz_corrected_this_step
            record["close_forced_this_step"] = close_forced_this_step
            record["chunk_id"] = chunk_step // EXECUTE_HORIZON
            record["chunk_offset"] = chunk_step % EXECUTE_HORIZON
            state["current_episode_steps"].append(record)

            if state["first_near_contact_step"] is None and record["ee_to_card_dist"] <= NEAR_CONTACT_TOLERANCE:
                state["first_near_contact_step"] = record["step_ix"]
            if record["is_grasped"]:
                if state["first_contact_step"] is None:
                    state["first_contact_step"] = record["step_ix"]
                if not state["contact_ever_seen"]:
                    state["contact_ever_seen"] = True  # disables xyz-oracle from the NEXT step
                if use_close_oracle and not state["close_forcing_active"]:
                    state["close_forcing_active"] = True  # forces close from the NEXT step
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state, state  # second ref kept for readability at call sites


def uninstall_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def first_policy_close_step(steps: list[dict]) -> int | None:
    """First step whose ORIGINAL (pre-override) policy gripper prediction was <=0.5 (closed) --
    what the unmodified policy itself would have done, regardless of any oracle intervention."""
    for s in steps[1:]:
        orig = s.get("original_predicted_action_raw")
        if orig is not None and orig[6] <= ppf.GRIPPER_THRESHOLD:
            return s["step_ix"]
    return None


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

    inst_state, _ = install_instrumentation(mode)
    episode_records: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        ep_len = done_index + 1
        steps = list(inst_state["current_episode_steps"])
        episode_records.append(
            {
                "episode_ix": episode_ix,
                "episode_len": ep_len,
                "steps": steps,
                "first_contact_step": inst_state["first_contact_step"],
                "first_near_contact_step": inst_state["first_near_contact_step"],
            }
        )
        logger.info("  [%s] episode %d len=%d n_steps=%d first_contact=%s", mode, episode_ix, ep_len, len(steps), inst_state["first_contact_step"])

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
    grasp_success = any(s["is_grasped"] for s in steps)
    heights_while_grasped = [s["card_height"] for s in steps if s["is_grasped"]]
    lift_success = bool(heights_while_grasped) and max(heights_while_grasped) >= LIFT_THRESHOLD

    orig_close_step = first_policy_close_step(steps)
    first_contact_step = rec["first_contact_step"]
    oracle_close_step = next((s["step_ix"] for s in steps if s.get("close_forced_this_step")), None)

    return {
        "seed": rec["seed"],
        "mode": mode,
        "success": rec["success"],
        "grasp_success": grasp_success,
        "lift_success": lift_success,
        "episode_len": rec["episode_len"],
        "first_near_contact_step": rec["first_near_contact_step"],
        "first_contact_step": first_contact_step,
        "original_policy_close_step": orig_close_step,
        "oracle_close_step": oracle_close_step,
        "policy_close_minus_contact_delay": (orig_close_step - first_contact_step) if (orig_close_step is not None and first_contact_step is not None) else None,
        "oracle_close_minus_contact_delay": (oracle_close_step - first_contact_step) if (oracle_close_step is not None and first_contact_step is not None) else None,
        "max_card_height": max((s["card_height"] for s in steps), default=None),
        "video": str(videos_dir / f"eval_episode_{rec['episode_ix']}.mp4"),
    }


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_grasp_close_timing_oracle")
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
        logger.info("=== condition: %s (seeds %d..%d) ===", mode, args.start_seed, args.start_seed + args.n_episodes - 1)
        records = run_condition(policy, args.device, args.n_episodes, args.start_seed, mode, videos_dir)
        all_episode_records[mode] = records
        for rec in records:
            all_summaries.append(summarize_episode(rec, mode, videos_dir))

    (out_dir / "episodes_raw.json").write_text(json.dumps(all_episode_records, indent=2))

    by_seed: dict[int, dict] = {}
    for entry in all_summaries:
        by_seed.setdefault(entry["seed"], {})[entry["mode"]] = entry

    paired = []
    for seed, conds in sorted(by_seed.items()):
        paired.append({"seed": seed, **{m: conds.get(m) for m in CONDITIONS}})

    aggregate = {}
    for mode in CONDITIONS:
        entries = [p[mode] for p in paired if p[mode] is not None]
        n = len(entries)
        aggregate[mode] = {
            "n_episodes": n,
            "n_grasp_success": sum(1 for e in entries if e["grasp_success"]),
            "n_lift_success": sum(1 for e in entries if e["lift_success"]),
            "n_task_success": sum(1 for e in entries if e["success"]),
        }

    summary = {
        "checkpoint": CHECKPOINT,
        "task": TASK,
        "execute_horizon": EXECUTE_HORIZON,
        "action_horizon": EXECUTE_HORIZON,
        "seeds": [p["seed"] for p in paired],
        "diagnostic_caveat": (
            "PRIVILEGED, NON-DEPLOYABLE diagnostic. xyz_oracle_only / xyz_oracle_plus_close_oracle "
            "read live simulator state (card position, gripper<->card contact) to override only "
            "action[:3] (xyz, until first contact) and/or action[6] (gripper, forced to 0.0/closed "
            "from the step after first contact to episode end) of the policy's own commanded "
            "action. Rotation is always the unmodified policy prediction in every condition. Both "
            "gates are non-anticipatory: they only ever act on physics state already observed as "
            "of the previous step. Model weights/architecture/loss/checkpoint and "
            "execution.ActionExecutor queueing/replanning logic are all untouched."
        ),
        "near_contact_tolerance_m": NEAR_CONTACT_TOLERANCE,
        "forced_closed_gripper_value": FORCED_CLOSED_GRIPPER_VALUE,
        "paired_results": paired,
        "aggregate": aggregate,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("=== wrote %s and %s ===", out_dir / "summary.json", out_dir / "episodes_raw.json")
    logger.info("aggregate: %s", json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
