#!/usr/bin/env python
"""Controlled counterfactual audit of instruction-conditioned grounding: fix ONE real
image+state observation (a single `env.reset()`, not a rollout), vary ONLY the language
instruction across several real card identities present in that exact scene, and compare the
model's internal representations and outputs. No training, no rollout, no threshold/replan/
reactive-close changes.

MAJOR CONTEXT (found while building this script, verified directly against a live env, reported
prominently in the output -- not fixed here, out of this audit's scope): `lerobot_eval.py` sets
`observation["task"] = list(env.call("task_description"))` every step, and
`VLABenchEnv.task_description` (see `lerobot/envs/vlabench.py`, checks for `task_obj.task_description`
then `task_obj.language_instruction`, else falls back to `self.task`) finds NEITHER attribute on
VLABench's own `SelectPokerTask` (verified: `hasattr(task_obj, "task_description")` and
`hasattr(task_obj, "language_instruction")` are both False; the real per-episode instruction lives
in `task_obj.instructions`, e.g. `["Please pick the poker 7 of spades"]`) -- so it silently falls
back to the generic task NAME, `"select_poker"`, for every single episode regardless of which card
is actually the target. Verified live: seeds 1000/1001/1002/1003 (target entities 7_of_spades /
2_of_diamonds / 5_of_diamonds / 3_of_hearts respectively) ALL report `task_description ==
"select_poker"`. Meanwhile `lerobot/vlabench_unified`'s own training task strings ARE fully
specific (e.g. `"primitive: Please pick the poker 2 of hearts"`). This means EVERY closed-loop
rollout this session (canonical baseline, oracle experiments, grounded-grasp variants, ablations)
ran with zero card-identity information in the language channel -- a highly plausible, and much
simpler, explanation for "always reaches toward roughly the same place" than any of the three
model-internal failure modes this script is designed to distinguish. This script sidesteps that
bug entirely by constructing its own batches with hand-specified, correctly-formatted instructions
(matching the training data's own phrasing), never going through `VLABenchEnv.task_description`.

For each of several scenes (env resets) and, per scene, several real card identities present in
that same scene (the true target + 2-3 others, all confirmed by reading live physics state), with
the image/state frozen and ONLY the instruction text changed, records:
  - pooled VLM multimodal latent (`SafeDiffVLAPolicy._pooled_latent`)
  - `TargetPointHead` predicted xyz (`_predict_target_xyz`, un-normalized to physical meters)
  - the decoder's first-10-step xyz trajectory (`plan_action_chunk`, physical meters)
  - the decoder's own predicted first-close xyz (physical meters, at the first step its own
    gripper channel reads closed)

Then reports, per scene: instruction-pair L2 distances for each of the above, which real card
entity each instruction's predicted target/first-close xyz is nearest to, the fraction of
instruction pairs landing on the SAME nearest card despite naming different ones, and pooled-latent
sensitivity to the instruction change (L2 and cosine distance).

Uses the `temporal_decoder_grounded_grasp` (fixed-supervision) checkpoint for TargetPointHead/
pooled-latent measurements, and separately the canonical `temporal_decoder` checkpoint for the
first-10-trajectory/first-close comparison (no TargetPointHead there, but the same instruction-
sensitivity question applies to its decoder).

No training, no threshold changes, no reactive close, no corrective replan added here.

Outputs:
    outputs/eval/safediff_vla_instruction_grounding_audit/summary.json

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/instruction_grounding_audit.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import place_phase_forensics as ppf  # noqa: E402

from lerobot.envs import make_env  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

TASK = "select_poker"
RENAME_MAP = ppf.RENAME_MAP
GROUNDED_CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_grounded_grasp_5k/checkpoints/005000/pretrained_model"
CANONICAL_CHECKPOINT = ppf.CHECKPOINT
GRIPPER_THRESHOLD = ppf.GRIPPER_THRESHOLD
SCENE_SEEDS = [1000, 1001, 1002]
N_CARDS_PER_SCENE = 4  # true target + up to 3 other real cards present in that same scene


def pretty_card_name(entity_name: str) -> str:
    """`"7_of_spades"` -> `"7 of spades"` (matches VLABench's own phrasing, e.g.
    `task_obj.instructions[0] == "Please pick the poker 7 of spades"`)."""
    return entity_name.replace("_", " ")


def robot_base(env) -> np.ndarray:
    base = getattr(env, "_robot_base_xyz", None)
    return np.asarray(base, dtype=float) if base is not None else np.array([0.0, -0.4, 0.78])


def capture_scene(seed: int) -> dict:
    """One `env.reset()` (NOT a rollout) -- real image+state observation, plus ground-truth card
    identities/positions read directly from live physics (independent of the broken
    `task_description` extraction)."""
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    set_seed(seed)
    observation, _ = env.reset(seed=[seed])
    sub_env = env.envs[0]
    physics = sub_env._env.physics
    task_obj = sub_env._env.task
    target_name = task_obj.target_entity
    base = robot_base(sub_env)

    card_positions = {}
    for name, ent in task_obj.entities.items():
        if name == "table" or name.startswith("card_holder"):
            continue
        with contextlib.suppress(Exception):  # best-effort
            card_positions[name] = np.asarray(ent.get_xpos(physics), dtype=float)

    result = {
        "seed": seed,
        "target_entity": target_name,
        "card_positions_world": {k: v.tolist() for k, v in card_positions.items()},
        "robot_base_world": base.tolist(),
        # Raw, still-batched ([1, ...] per key) env observation -- `preprocess_observation`
        # (the exact same function `lerobot_eval.py`'s own rollout loop calls) expects this shape.
        "raw_observation": observation,
    }
    env.close()
    return result


def build_batch(scene: dict, instruction: str) -> dict[str, torch.Tensor]:
    batch = preprocess_observation(scene["raw_observation"])
    batch["task"] = [instruction]
    return batch


def nearest_card(xyz_world: np.ndarray, card_positions_world: dict[str, list[float]]) -> tuple[str, float]:
    dists = {name: float(np.linalg.norm(xyz_world - np.asarray(pos))) for name, pos in card_positions_world.items()}
    nearest = min(dists, key=dists.get)
    return nearest, dists[nearest]


def run_grounded_grasp_probe(policy, preprocessor, postprocessor, scene: dict, instructions: list[str]) -> list[dict]:
    base = np.asarray(scene["robot_base_world"])
    records = []
    for instruction in instructions:
        raw_batch = build_batch(scene, instruction)
        batch = preprocessor(raw_batch)
        with torch.no_grad():
            latent_tokens, latent_pad_mask, latent_modality_ids = policy._encode_multimodal_latent(batch)
            pooled = policy._pooled_latent(latent_tokens, latent_pad_mask, latent_modality_ids)
            predicted_target_norm = policy._predict_target_xyz(latent_tokens, latent_pad_mask, latent_modality_ids)
            predicted_target_robot_frame = (predicted_target_norm * policy.action_pos_std + policy.action_pos_mean)[0].cpu().numpy()
            predicted_target_world = predicted_target_robot_frame + base
            actions, _ = policy.plan_action_chunk(batch)
        actions_phys = postprocessor(actions.cpu()).numpy()[0]  # [H, 7] physical scale, robot-base frame
        first10_xyz_world = actions_phys[:10, :3] + base
        is_open = actions_phys[:, GRIPPER_INDEX_RAW] > GRIPPER_THRESHOLD
        closed_idx = np.flatnonzero(~is_open)
        first_close_xyz_world = (actions_phys[closed_idx[0], :3] + base) if len(closed_idx) else None

        nearest_to_target, dist_to_nearest_target = nearest_card(predicted_target_world, scene["card_positions_world"])
        nearest_to_close = (
            nearest_card(first_close_xyz_world, scene["card_positions_world"]) if first_close_xyz_world is not None else (None, None)
        )
        records.append(
            {
                "instruction": instruction,
                "pooled_latent": pooled[0].cpu().numpy().tolist(),
                "predicted_target_world": predicted_target_world.tolist(),
                "nearest_card_to_predicted_target": nearest_to_target,
                "dist_to_nearest_card": dist_to_nearest_target,
                "first10_xyz_world": first10_xyz_world.tolist(),
                "first_close_xyz_world": first_close_xyz_world.tolist() if first_close_xyz_world is not None else None,
                "nearest_card_to_first_close": nearest_to_close[0],
            }
        )
    return records


def run_canonical_probe(policy, preprocessor, postprocessor, scene: dict, instructions: list[str]) -> list[dict]:
    base = np.asarray(scene["robot_base_world"])
    records = []
    for instruction in instructions:
        raw_batch = build_batch(scene, instruction)
        batch = preprocessor(raw_batch)
        with torch.no_grad():
            # `_pooled_latent` (`modeling_safediff_vla.py`) is now an instance method reading
            # `self.modality_pool_projection`, built only for `temporal_decoder_subgoal`/
            # `temporal_decoder_grounded_grasp` -- the canonical `temporal_decoder` checkpoint
            # probed here never has it (nothing about the canonical architecture itself changed;
            # this is only about whether the *pooled-latent diagnostic* below can be computed).
            latent_tokens, latent_pad_mask, latent_modality_ids = policy._encode_multimodal_latent(batch)
            pooled = (
                policy._pooled_latent(latent_tokens, latent_pad_mask, latent_modality_ids)
                if hasattr(policy, "modality_pool_projection")
                else None
            )
            actions, _ = policy.plan_action_chunk(batch)
        actions_phys = postprocessor(actions.cpu()).numpy()[0]
        first10_xyz_world = actions_phys[:10, :3] + base
        is_open = actions_phys[:, GRIPPER_INDEX_RAW] > GRIPPER_THRESHOLD
        closed_idx = np.flatnonzero(~is_open)
        first_close_xyz_world = (actions_phys[closed_idx[0], :3] + base) if len(closed_idx) else None
        nearest_to_close = nearest_card(first_close_xyz_world, scene["card_positions_world"]) if first_close_xyz_world is not None else (None, None)
        records.append(
            {
                "instruction": instruction,
                "pooled_latent": pooled[0].cpu().numpy().tolist() if pooled is not None else None,
                "first10_xyz_world": first10_xyz_world.tolist(),
                "first_close_xyz_world": first_close_xyz_world.tolist() if first_close_xyz_world is not None else None,
                "nearest_card_to_first_close": nearest_to_close[0],
            }
        )
    return records


def pairwise_stats(records: list[dict], key: str) -> dict:
    values = [np.asarray(r[key]) for r in records if r.get(key) is not None]
    if len(values) < 2:
        return {"n": len(values), "pairs": []}
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            vi, vj = records[i].get(key), records[j].get(key)
            if vi is None or vj is None:
                continue
            a, b = np.asarray(vi), np.asarray(vj)
            l2 = float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)))
            denom = np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1))
            cos = float(np.dot(a.reshape(-1), b.reshape(-1)) / denom) if denom > 0 else None
            pairs.append(
                {
                    "instruction_a": records[i]["instruction"],
                    "instruction_b": records[j]["instruction"],
                    "l2": l2,
                    "cosine_similarity": cos,
                }
            )
    return {"pairs": pairs, "mean_l2": sum(p["l2"] for p in pairs) / len(pairs)}


def build_scene_report(scene: dict, grounded_records: list[dict], canonical_records: list[dict]) -> dict:
    n = len(grounded_records)
    same_nearest_count = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if grounded_records[i]["nearest_card_to_predicted_target"] == grounded_records[j]["nearest_card_to_predicted_target"]
    )
    n_pairs = n * (n - 1) // 2
    return {
        "seed": scene["seed"],
        "true_target_entity": scene["target_entity"],
        "instructions_tested": [r["instruction"] for r in grounded_records],
        "grounded_grasp_checkpoint": GROUNDED_CHECKPOINT,
        "predicted_target_by_instruction": [
            {"instruction": r["instruction"], "predicted_target_world": r["predicted_target_world"], "nearest_card": r["nearest_card_to_predicted_target"], "dist_to_nearest": r["dist_to_nearest_card"]}
            for r in grounded_records
        ],
        "pooled_latent_pairwise": pairwise_stats(grounded_records, "pooled_latent"),
        "predicted_target_xyz_pairwise": pairwise_stats(grounded_records, "predicted_target_world"),
        "first10_trajectory_pairwise": pairwise_stats(grounded_records, "first10_xyz_world"),
        "fraction_instruction_pairs_same_nearest_card": (same_nearest_count / n_pairs) if n_pairs else None,
        "canonical_checkpoint": CANONICAL_CHECKPOINT,
        "canonical_pooled_latent_pairwise": pairwise_stats(canonical_records, "pooled_latent"),
        "canonical_first10_trajectory_pairwise": pairwise_stats(canonical_records, "first10_xyz_world"),
        "canonical_nearest_card_to_first_close_by_instruction": [
            {"instruction": r["instruction"], "nearest_card": r["nearest_card_to_first_close"]} for r in canonical_records
        ],
    }


def main() -> None:
    global GROUNDED_CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/safediff_vla_instruction_grounding_audit/summary.json")
    parser.add_argument(
        "--grounded-checkpoint",
        default=GROUNDED_CHECKPOINT,
        help="temporal_decoder_grounded_grasp checkpoint to probe (e.g. to compare the pre-fix "
        "all-token-mean pooling checkpoint against a modality-aware-pooling one) -- the canonical "
        "temporal_decoder checkpoint used for comparison is unaffected, always CANONICAL_CHECKPOINT.",
    )
    args = parser.parse_args()
    GROUNDED_CHECKPOINT = args.grounded_checkpoint

    scene_reports = []
    verified_bug = {}
    try:
        env_cfg = VLABenchEnv(task=TASK)
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        env = envs[TASK][0]
        for seed in SCENE_SEEDS:
            env.reset(seed=[seed])
            td = list(env.call("task_description"))[0]
            sub = env.envs[0]
            verified_bug[seed] = {"task_description_seen_by_model_during_rollout": td, "true_target_entity": sub._env.task.target_entity}
        env.close()
    except Exception as e:  # noqa: BLE001
        verified_bug["error"] = str(e)

    device = args.device
    grounded_policy = SafeDiffVLAPolicy.from_pretrained(GROUNDED_CHECKPOINT).to(device)
    grounded_policy.eval()
    grounded_pre, grounded_post = make_pre_post_processors(
        policy_cfg=grounded_policy.config,
        pretrained_path=GROUNDED_CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": device}, "rename_observations_processor": {"rename_map": RENAME_MAP}},
    )
    canonical_policy = SafeDiffVLAPolicy.from_pretrained(CANONICAL_CHECKPOINT).to(device)
    canonical_policy.eval()
    canonical_pre, canonical_post = make_pre_post_processors(
        policy_cfg=canonical_policy.config,
        pretrained_path=CANONICAL_CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": device}, "rename_observations_processor": {"rename_map": RENAME_MAP}},
    )

    for seed in SCENE_SEEDS:
        logger.info("=== scene seed=%d ===", seed)
        scene = capture_scene(seed)
        true_target = scene["target_entity"]
        other_cards = [c for c in scene["card_positions_world"] if c != true_target]
        chosen_others = other_cards[: N_CARDS_PER_SCENE - 1]
        instructions = [f"primitive: Please pick the poker {pretty_card_name(true_target)}"] + [
            f"primitive: Please pick the poker {pretty_card_name(c)}" for c in chosen_others
        ]
        logger.info("  true_target=%s, instructions=%s", true_target, instructions)

        grounded_records = run_grounded_grasp_probe(grounded_policy, grounded_pre, grounded_post, scene, instructions)
        canonical_records = run_canonical_probe(canonical_policy, canonical_pre, canonical_post, scene, instructions)
        scene_reports.append(build_scene_report(scene, grounded_records, canonical_records))
        logger.info("  fraction_same_nearest_card=%s", scene_reports[-1]["fraction_instruction_pairs_same_nearest_card"])

    summary = {
        "eval_harness_bug_verified": {
            "description": (
                "lerobot_eval.py sets observation['task'] = env.call('task_description') every "
                "step; VLABenchEnv.task_description falls back to the generic task NAME "
                "('select_poker') for SelectPokerTask because neither 'task_description' nor "
                "'language_instruction' exists on that VLABench task class (the real per-episode "
                "instruction is in task_obj.instructions). Every closed-loop rollout this session "
                "ran with this generic, card-identity-free instruction. NOT fixed in this audit "
                "(out of scope) -- reported for context."
            ),
            "per_seed_verification": verified_bug,
            "training_dataset_task_strings_are_specific": True,
            "training_dataset_example": "primitive: Please pick the poker 2 of hearts",
        },
        "scenes": scene_reports,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("=== wrote %s ===", out_path)


if __name__ == "__main__":
    main()
