#!/usr/bin/env python
"""Evaluation for the two no-`TargetPointHead` instruction-conditioning experiments:
`temporal_decoder_instruction` (`train_instruction_conditioned_decoder_5k.py` -- the instruction
added as global conditioning on every horizon query) and `temporal_decoder_text_crossattn`
(`train_text_crossattn_decoder_5k.py` -- an extra cross-attention over the raw, un-pooled text
token sequence, applied after the canonical decoder stack). Pass `--checkpoint` to select which.
Neither architecture has a `TargetPointHead`, so "where is it aiming" is read off the decoder's
own predicted action-chunk trajectory directly, not a separate head's output.

Part A -- same-image instruction-swap probe (same 5 scenes/seeds as the two prior
`TargetPointHead`-pooling experiments, for direct comparison): for each frozen scene, swaps ONLY
the instruction text across the real card identities present, and for each instruction runs
`plan_action_chunk` to get the full predicted action-chunk trajectory (physical units). The
gripper-close step's xyz (first step the postprocessed gripper channel crosses closed, falling
back to the chunk's last-step xyz if it never closes) is the "predicted target" proxy -- exactly
`instruction_counterfactual_audit.py`'s convention for a checkpoint with no explicit target head.
Reports, per pair of instructions in a scene:
  - full-chunk trajectory L2 (mean pointwise distance across all `action_horizon` steps) --
    "predicted action trajectory 변화".
  - whether the nearest real card to the predicted-close xyz is the SAME card despite the
    different instruction (1 - this = identity-flip rate).
  - whether each instruction's own nearest-card prediction matches the card it named
    ("correct-instructed-card rate").

Part B -- closed-loop rollout, seeds 1000-1009 (10 seeds), each with the environment's OWN
(correct) instruction -- NOT an instruction swap. Measures real task outcomes via a read-only
`VLABenchEnv.step`/`.reset` monkeypatch (same non-invasive pattern as prior rollout scripts):
  - success: `eval_policy`'s own per-episode success flag (the task's real success condition).
  - grasp/contact: whether the actual target card entity was ever grasped
    (`entity.is_grasped(physics, robot)`, ground-truth contact query, not inferred from actions).
  - wrong-object grasp rate: whether any OTHER (non-target) card entity was ever grasped instead.

Outputs:
    outputs/eval/safediff_vla_instruction_conditioned_decoder_5k/summary.json

Usage:
    # Part A only (no simulator needed for the probe itself, though the scene capture still
    # needs one):
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_instruction_conditioned_decoder_5k.py --skip-rollout
    # Both parts:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_instruction_conditioned_decoder_5k.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.scripts.lerobot_eval import eval_policy

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_OPEN_THRESHOLD = 0.5
CHECKPOINT = "outputs/train/safediff_vla_instruction_conditioned_decoder_5k/checkpoints/005000/pretrained_model"
TASK = "select_poker"
PROBE_SEEDS = [1000, 1001, 1002, 1003, 1004]  # same scenes as the prior two pooling experiments
ROLLOUT_SEEDS = list(range(1000, 1010))  # 1000-1009


# ---------------------------------------------------------------------------------------
# Part A -- same-image instruction-swap probe (no TargetPointHead: read the decoder's own
# predicted action-chunk trajectory instead)
# ---------------------------------------------------------------------------------------


def pretty_identity(identity: str) -> str:
    rank, _, suit = identity.partition("_of_")
    return f"{rank} of {suit}"


def get_scene_cards(env_impl: VLABenchEnvImpl) -> dict[str, np.ndarray]:
    physics = env_impl._env.physics
    task = env_impl._env.task
    base = env_impl._robot_base_xyz if env_impl._robot_base_xyz is not None else np.zeros(3, dtype=float)
    cards: dict[str, np.ndarray] = {}
    for name, ent in task.entities.items():
        if type(ent).__name__ != "Poker":
            continue
        pos_world = np.asarray(ent.get_xpos(physics), dtype=float)
        cards[name] = pos_world - base
    return cards


def capture_scene(seed: int) -> dict[str, Any]:
    """One `env.reset()`, zero `env.step()` calls. Returns the frozen observation (CPU tensors)
    plus ground-truth card identities/positions (robot frame) for that exact reset."""
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    env_impl: VLABenchEnvImpl = env.envs[0]
    try:
        observation, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
        target_name = env_impl._env.task.target_entity
        cards = get_scene_cards(env_impl)
        obs_t = preprocess_observation(observation)
        obs_t = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs_t.items()}
    finally:
        env.close()
    return {"seed": seed, "obs_t": obs_t, "cards": cards, "target_name": target_name}


def run_instruction(
    policy: SafeDiffVLAPolicy, env_preprocessor, preprocessor, postprocessor, obs_t_base: dict[str, Any], instruction: str
) -> dict[str, Any]:
    obs_t = dict(obs_t_base)
    obs_t["task"] = [instruction]
    obs_t = env_preprocessor(obs_t)
    obs_norm = preprocessor(obs_t)

    with torch.no_grad():
        actions_norm, _metrics = policy.plan_action_chunk(obs_norm)
    actions_physical = postprocessor(actions_norm[0].clone()).cpu().numpy()  # [H, 7]

    gripper = actions_physical[:, GRIPPER_INDEX_RAW]
    closed = gripper <= GRIPPER_OPEN_THRESHOLD
    close_idx = int(np.argmax(closed)) if closed.any() else None
    close_xyz = actions_physical[close_idx, :3] if close_idx is not None else actions_physical[-1, :3]

    return {
        "instruction": instruction,
        "full_trajectory_xyz": actions_physical[:, :3].tolist(),
        "predicted_close_step": close_idx,
        "predicted_close_xyz": close_xyz.tolist(),
        "used_last_step_fallback": close_idx is None,
    }


def nearest_card(point: np.ndarray, cards: dict[str, np.ndarray]) -> tuple[str, float]:
    dists = {name: float(np.linalg.norm(point - pos)) for name, pos in cards.items()}
    nearest_name = min(dists, key=dists.get)
    return nearest_name, dists[nearest_name]


def run_instruction_swap_probe(checkpoint: str, device: str, seeds: list[int]) -> dict:
    logger.info("=== Part A: same-image instruction-swap probe (%s) ===", checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(checkpoint).to(device)
    policy.eval()
    assert policy.config.architecture in ("temporal_decoder_instruction", "temporal_decoder_text_crossattn")
    assert not hasattr(policy, "target_point_head")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_cfg = VLABenchEnv(task=TASK)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    scene_results = []
    for seed in seeds:
        logger.info("=== scene seed=%d: single reset, zero steps ===", seed)
        scene = capture_scene(seed)
        cards = scene["cards"]
        if len(cards) < 2:
            logger.info("seed=%d: fewer than 2 cards in scene, skipping", seed)
            continue

        instructions = {name: f"primitive: Please pick the poker {pretty_identity(name)}" for name in cards}
        per_instruction = {}
        for name, instr in instructions.items():
            res = run_instruction(policy, env_preprocessor, preprocessor, postprocessor, scene["obs_t"], instr)
            point = np.array(res["predicted_close_xyz"])
            nearest_name, nearest_dist = nearest_card(point, cards)
            res["nearest_card_identity"] = nearest_name
            res["nearest_card_dist"] = nearest_dist
            res["is_correct_instructed_card"] = nearest_name == name
            per_instruction[name] = res
            logger.info(
                "  instruction card=%s -> predicted_close_xyz=%s nearest_card=%s(%.3fm) correct=%s",
                name, res["predicted_close_xyz"], nearest_name, nearest_dist, res["is_correct_instructed_card"],
            )

        pair_stats = []
        names = list(instructions.keys())
        for a, b in itertools.combinations(names, 2):
            ra, rb = per_instruction[a], per_instruction[b]
            traj_a, traj_b = np.array(ra["full_trajectory_xyz"]), np.array(rb["full_trajectory_xyz"])
            traj_l2_mean = float(np.linalg.norm(traj_a - traj_b, axis=1).mean())
            same_nearest_card = ra["nearest_card_identity"] == rb["nearest_card_identity"]
            pair_stats.append(
                {
                    "instruction_a_card": a,
                    "instruction_b_card": b,
                    "full_trajectory_mean_l2": traj_l2_mean,
                    "same_nearest_card_despite_different_instruction": same_nearest_card,
                }
            )

        scene_results.append(
            {
                "seed": seed,
                "target_entity_env_default": scene["target_name"],
                "cards_in_scene": {name: pos.tolist() for name, pos in cards.items()},
                "per_instruction": {
                    name: {k: v for k, v in res.items() if k != "full_trajectory_xyz"} for name, res in per_instruction.items()
                },
                "pairwise": pair_stats,
            }
        )

    all_pairs = [p for s in scene_results for p in s["pairwise"]]

    def agg(key: str) -> dict | None:
        if not all_pairs:
            return None
        vals = [p[key] for p in all_pairs]
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}

    frac_same = (
        float(np.mean([p["same_nearest_card_despite_different_instruction"] for p in all_pairs]))
        if all_pairs
        else None
    )
    frac_correct = (
        float(np.mean([r["is_correct_instructed_card"] for s in scene_results for r in s["per_instruction"].values()]))
        if scene_results
        else None
    )

    aggregate = {
        "n_scenes": len(scene_results),
        "n_instruction_pairs": len(all_pairs),
        "full_trajectory_mean_l2_across_instructions": agg("full_trajectory_mean_l2"),
        "frac_same_nearest_card_despite_different_instruction": frac_same,
        "frac_pairs_identity_flips_with_instruction": (1 - frac_same) if frac_same is not None else None,
        "frac_predictions_nearest_to_the_instructed_card": frac_correct,
    }
    del policy
    torch.cuda.empty_cache()
    return {"checkpoint": checkpoint, "scenes": scene_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------------------
# Part B -- closed-loop rollout, seeds 1000-1009, each with the env's own correct instruction
# ---------------------------------------------------------------------------------------


def install_env_instrumentation() -> dict:
    """Read-only `VLABenchEnv.step`/`.reset` monkeypatch (same non-invasive pattern as prior
    rollout scripts) recording, per step: whether the real target card entity is under contact
    grasp (`entity.is_grasped(physics, robot)`, ground truth, independent of any model output),
    and whether any OTHER card entity is grasped instead (wrong-object grasp)."""
    cls = VLABenchEnvImpl
    orig_step = cls.step
    orig_reset = cls.reset
    state: dict[str, Any] = {"current_episode_steps": []}

    def _card_grasp_status(env_impl: VLABenchEnvImpl) -> dict[str, Any]:
        physics = env_impl._env.physics
        task = env_impl._env.task
        robot = task.robot
        target_name = task.target_entity
        grasped = {}
        for name, ent in task.entities.items():
            if type(ent).__name__ != "Poker":
                continue
            try:
                grasped[name] = bool(ent.is_grasped(physics, robot))
            except Exception:  # noqa: BLE001 - best-effort diagnostic only
                grasped[name] = False
        return {
            "target_entity": target_name,
            "target_is_grasped": grasped.get(target_name, False),
            "wrong_object_grasped": any(g for name, g in grasped.items() if name != target_name),
        }

    def patched_reset(self, seed=None, **kwargs):
        result = orig_reset(self, seed=seed, **kwargs)
        if seed is not None:
            state["current_episode_steps"] = [_card_grasp_status(self)]
        return result

    def patched_step(self, action):
        result = orig_step(self, action)
        terminated = result[2]
        if not terminated:
            state["current_episode_steps"].append(_card_grasp_status(self))
        return result

    cls.step = patched_step
    cls.reset = patched_reset
    state["_originals"] = (cls, orig_step, orig_reset)
    return state


def uninstall_env_instrumentation(state: dict) -> None:
    cls, orig_step, orig_reset = state["_originals"]
    cls.step = orig_step
    cls.reset = orig_reset


def run_closed_loop_rollout(checkpoint: str, device: str, seeds: list[int], output_dir: Path) -> dict:
    logger.info("=== Part B: closed-loop rollout, seeds %d-%d (%s) ===", seeds[0], seeds[-1], checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(checkpoint).to(device)
    policy.eval()
    assert policy.config.architecture in ("temporal_decoder_instruction", "temporal_decoder_text_crossattn")

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

    env_state = install_env_instrumentation()
    episode_records: list[dict] = []

    def episode_callback(episode_ix: int, rollout_data: dict, env_idx: int, done_index: int) -> None:
        steps = list(env_state["current_episode_steps"])
        episode_records.append({"episode_ix": episode_ix, "episode_len": done_index + 1, "steps": steps})
        logger.info("  episode %d len=%d n_steps=%d", episode_ix, done_index + 1, len(steps))

    try:
        info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=len(seeds),
            max_episodes_rendered=0,
            videos_dir=output_dir / "videos",
            start_seed=seeds[0],
            episode_callback=episode_callback,
        )
    finally:
        env.close()
        uninstall_env_instrumentation(env_state)

    for ep, rec in zip(info["per_episode"], episode_records, strict=True):
        rec["seed"] = ep["seed"]
        rec["success"] = ep["success"]
        rec["grasp_contact_achieved"] = any(s["target_is_grasped"] for s in rec["steps"])
        rec["wrong_object_grasped"] = any(s["wrong_object_grasped"] for s in rec["steps"])
        logger.info(
            "  seed=%d success=%s grasp_contact=%s wrong_object_grasped=%s",
            rec["seed"], rec["success"], rec["grasp_contact_achieved"], rec["wrong_object_grasped"],
        )

    per_episode_summary = [
        {k: rec[k] for k in ("seed", "episode_len", "success", "grasp_contact_achieved", "wrong_object_grasped")}
        for rec in episode_records
    ]
    aggregate = {
        "n_episodes": len(per_episode_summary),
        "success_rate": float(np.mean([e["success"] for e in per_episode_summary])),
        "grasp_contact_rate": float(np.mean([e["grasp_contact_achieved"] for e in per_episode_summary])),
        "wrong_object_grasp_rate": float(np.mean([e["wrong_object_grasped"] for e in per_episode_summary])),
    }
    del policy
    torch.cuda.empty_cache()
    return {"checkpoint": checkpoint, "per_episode": per_episode_summary, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=PROBE_SEEDS)
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=ROLLOUT_SEEDS)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-instruction-swap-probe", action="store_true")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_instruction_conditioned_decoder_5k")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    part_a = None
    if not args.skip_instruction_swap_probe:
        part_a = run_instruction_swap_probe(args.checkpoint, args.device, args.probe_seeds)

    part_b = None
    if not args.skip_rollout:
        part_b = run_closed_loop_rollout(args.checkpoint, args.device, args.rollout_seeds, output_dir)

    summary = {"part_a_instruction_swap_probe": part_a, "part_b_closed_loop_rollout": part_b}
    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("=== wrote %s ===", out_path)


if __name__ == "__main__":
    main()
