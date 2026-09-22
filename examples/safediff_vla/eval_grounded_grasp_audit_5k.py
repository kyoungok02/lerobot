#!/usr/bin/env python
"""INDEPENDENT held-out + instruction-counterfactual evaluation of the fresh
`train_grounded_grasp_audit_5k.py` checkpoint (`temporal_decoder_grounded_grasp`,
`grounded_grasp_condition_decoder_on_target=False` i.e. target conditioning OFF, modality-aware
pooling, genuine `eval_split=0.1` held-out split). No rollout, no reactive close, no target
conditioning -- purely offline dataset/scene probes, self-contained (does not import or depend on
any file/checkpoint outside this repo's own `outputs/train/safediff_vla_grounded_grasp_audit_5k`).

Part A -- held-out `TargetPointHead` physical xyz L2:
  Reconstructs the EXACT same held-out split the training run used (`make_train_eval_datasets`
  with the identical `repo_id`+`eval_split` -- the split is a deterministic function of those two
  inputs only, per `lerobot.datasets.factory.make_train_eval_datasets`'s docstring). For every
  held-out frame with a valid grasp-transition label (`utils.find_grasp_target`), computes the
  model's predicted target xyz (physical meters) against GT, and against two baselines that need
  no model at all:
    - current-EE baseline: predict the observation's own current EE position.
    - mean-target baseline: predict the held-out split's own mean GT target position.
  Reports mean L2 and mean |dx|/|dy|/|dz| for the model and both baselines.

Part B -- same-image instruction-swap probe: for a handful of frozen real scenes (one
  `env.reset()` each, zero `env.step()` calls), varies ONLY the instruction text across the real
  card identities present in that scene, reads the model's OWN `TargetPointHead` output directly
  (via `plan_action_chunk`'s `predicted_target_xyz` metric -- no gripper-crossing heuristic needed,
  unlike `instruction_counterfactual_audit.py`'s canonical-checkpoint version, since this
  architecture has an explicit target head), and reports:
    - predicted-target movement (L2, meters) between every pair of card-instructions in a scene.
    - the fraction of instruction pairs whose nearest real card to the prediction is THE SAME card
      despite the different instruction (1 - this = the identity-flip rate the instruction is
      supposed to cause).

Usage:
    # Part A only (no simulator needed):
    uv run python examples/safediff_vla/eval_grounded_grasp_audit_5k.py --skip-instruction-swap-probe
    # Both parts (needs MuJoCo/EGL for the simulator):
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_grounded_grasp_audit_5k.py
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

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.policies.safediff_vla.utils import find_grasp_target, pad_or_crop_horizon, pad_or_crop_mask
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
EVAL_SPLIT = 0.1  # must match train_grounded_grasp_audit_5k.py's EVAL_SPLIT exactly
CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_audit_5k/checkpoints/005000/pretrained_model"
TASK = "select_poker"
SEEDS = [1000, 1001, 1002, 1003, 1004]


# ---------------------------------------------------------------------------------------
# Part A -- held-out TargetPointHead xyz L2 vs trivial baselines
# ---------------------------------------------------------------------------------------


def _held_out_dataloader(
    batch_size: int, num_workers: int, shuffle: bool
) -> tuple[torch.utils.data.DataLoader, int, list[str]]:
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", eval_split=EVAL_SPLIT),
        policy=SafeDiffVLAConfig(
            device="cpu", architecture="temporal_decoder_grounded_grasp", action_horizon=50, execute_horizon=50
        ),
        rename_map=RENAME_MAP,
    )
    _train_dataset, eval_dataset = make_train_eval_datasets(cfg)
    assert eval_dataset is not None, f"eval_split={EVAL_SPLIT} produced no held-out split"
    collate_fn = lerobot_collate_fn if eval_dataset.meta.has_language_columns else None
    generator = torch.Generator().manual_seed(1000) if shuffle else None
    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader, len(eval_dataset), eval_dataset.meta.camera_keys


def _physical_target_and_baselines(
    policy: SafeDiffVLAPolicy, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        _actions, plan_metrics = policy.plan_action_chunk(batch)
    predicted_norm = plan_metrics["predicted_target_xyz"]  # [B, 3], normalized action-position space
    predicted_m = predicted_norm * policy.action_pos_std + policy.action_pos_mean

    current_state_raw = policy._current_state(batch)
    clean_raw = pad_or_crop_horizon(batch[ACTION], policy.config.action_horizon)
    action_is_pad = batch.get("action_is_pad")
    action_is_pad_h = pad_or_crop_mask(action_is_pad, policy.config.action_horizon) if action_is_pad is not None else None
    state_for_grasp, action_for_grasp = policy._degrip_for_transition_detection(current_state_raw, clean_raw)
    gt_norm, valid_mask = find_grasp_target(
        state_for_grasp,
        action_for_grasp,
        gripper_index=GRIPPER_INDEX_RAW,
        gripper_open_threshold=policy.config.grounded_grasp_gripper_open_threshold,
        action_is_pad=action_is_pad_h,
    )
    gt_m = gt_norm * policy.action_pos_std + policy.action_pos_mean
    current_ee_m = current_state_raw[..., :3] * policy.state_pos_std + policy.state_pos_mean
    return predicted_m, gt_m, current_ee_m, valid_mask


def run_held_out_eval(checkpoint: str, device: str, batch_size: int, num_workers: int, max_samples: int) -> dict:
    logger.info("=== Part A: held-out TargetPointHead xyz L2 (%s) ===", checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(checkpoint).to(device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder_grounded_grasp"
    assert policy.config.grounded_grasp_condition_decoder_on_target is False, (
        "this audit round requires target conditioning OFF"
    )
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    loader, n_total, camera_keys = _held_out_dataloader(batch_size, num_workers, shuffle=max_samples > 0)
    logger.info("held-out split: %d frames", n_total)

    all_pred, all_gt, all_ee = [], [], []
    n_seen = 0
    for raw_batch in loader:
        batch = _preprocess_dataset_batch(raw_batch, camera_keys, RENAME_MAP, preprocessor)
        predicted_m, gt_m, current_ee_m, valid_mask = _physical_target_and_baselines(policy, batch)
        if valid_mask.any():
            all_pred.append(predicted_m[valid_mask].cpu())
            all_gt.append(gt_m[valid_mask].cpu())
            all_ee.append(current_ee_m[valid_mask].cpu())
        n_seen += valid_mask.numel()
        if max_samples > 0 and n_seen >= max_samples:
            break

    pred = torch.cat(all_pred)
    gt = torch.cat(all_gt)
    ee = torch.cat(all_ee)
    mean_target = gt.mean(dim=0, keepdim=True).expand_as(gt)
    n_valid = gt.shape[0]
    logger.info("  %d/%d frames had a valid grasp-transition label", n_valid, n_seen)

    def stats(pred_xyz: torch.Tensor, name: str) -> dict:
        err = pred_xyz - gt
        l2 = err.norm(dim=-1)
        return {
            "name": name,
            "n": n_valid,
            "mean_l2_m": l2.mean().item(),
            "median_l2_m": l2.median().item(),
            "mean_abs_dx_m": err[:, 0].abs().mean().item(),
            "mean_abs_dy_m": err[:, 1].abs().mean().item(),
            "mean_abs_dz_m": err[:, 2].abs().mean().item(),
        }

    result = {
        "checkpoint": checkpoint,
        "eval_split": EVAL_SPLIT,
        "n_frames_seen": n_seen,
        "n_valid_labels": n_valid,
        "model": stats(pred, "target_point_head"),
        "current_ee_baseline": stats(ee, "current_ee_baseline"),
        "mean_target_baseline": stats(mean_target, "mean_target_baseline"),
    }
    logger.info(
        "  model mean_l2_m=%.4f  current_ee mean_l2_m=%.4f  mean_target mean_l2_m=%.4f",
        result["model"]["mean_l2_m"], result["current_ee_baseline"]["mean_l2_m"], result["mean_target_baseline"]["mean_l2_m"],
    )
    del policy
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------------------
# Part B -- same-image instruction-swap probe + nearest-card identity flip rate
# ---------------------------------------------------------------------------------------


def pretty_identity(identity: str) -> str:
    rank, _, suit = identity.partition("_of_")
    return f"{rank} of {suit}"


def get_scene_cards(env_impl) -> dict[str, np.ndarray]:
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
    from lerobot.envs import make_env
    from lerobot.envs.configs import VLABenchEnv
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation

    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    env_impl = env.envs[0]
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
    policy: SafeDiffVLAPolicy, env_preprocessor, preprocessor, obs_t_base: dict[str, Any], instruction: str
) -> dict[str, Any]:
    obs_t = dict(obs_t_base)
    obs_t["task"] = [instruction]
    obs_t = env_preprocessor(obs_t)
    obs_norm = preprocessor(obs_t)

    with torch.no_grad():
        _actions, plan_metrics = policy.plan_action_chunk(obs_norm)
    predicted_norm = plan_metrics["predicted_target_xyz"]
    predicted_m = (predicted_norm * policy.action_pos_std + policy.action_pos_mean)[0].float().cpu().numpy()

    return {"instruction": instruction, "predicted_target_xyz": predicted_m.tolist()}


def nearest_card(point: np.ndarray, cards: dict[str, np.ndarray]) -> tuple[str, float]:
    dists = {name: float(np.linalg.norm(point - pos)) for name, pos in cards.items()}
    nearest_name = min(dists, key=dists.get)
    return nearest_name, dists[nearest_name]


def run_instruction_swap_probe(checkpoint: str, device: str, seeds: list[int]) -> dict:
    logger.info("=== Part B: same-image instruction-swap probe (%s) ===", checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(checkpoint).to(device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder_grounded_grasp"

    from lerobot.envs import make_env_pre_post_processors
    from lerobot.envs.configs import VLABenchEnv

    env_cfg = VLABenchEnv(task=TASK)
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
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
            res = run_instruction(policy, env_preprocessor, preprocessor, scene["obs_t"], instr)
            point = np.array(res["predicted_target_xyz"])
            nearest_name, nearest_dist = nearest_card(point, cards)
            res["nearest_card_identity"] = nearest_name
            res["nearest_card_dist"] = nearest_dist
            res["is_correct_instructed_card"] = nearest_name == name
            per_instruction[name] = res
            logger.info(
                "  instruction card=%s -> predicted_target=%s nearest_card=%s(%.3fm) correct=%s",
                name, res["predicted_target_xyz"], nearest_name, nearest_dist, res["is_correct_instructed_card"],
            )

        pair_stats = []
        names = list(instructions.keys())
        for a, b in itertools.combinations(names, 2):
            ra, rb = per_instruction[a], per_instruction[b]
            pa, pb = np.array(ra["predicted_target_xyz"]), np.array(rb["predicted_target_xyz"])
            target_l2 = float(np.linalg.norm(pa - pb))
            same_nearest_card = ra["nearest_card_identity"] == rb["nearest_card_identity"]
            pair_stats.append(
                {
                    "instruction_a_card": a,
                    "instruction_b_card": b,
                    "predicted_target_xyz_l2": target_l2,
                    "same_nearest_card_despite_different_instruction": same_nearest_card,
                }
            )

        scene_results.append(
            {
                "seed": seed,
                "target_entity_env_default": scene["target_name"],
                "cards_in_scene": {name: pos.tolist() for name, pos in cards.items()},
                "per_instruction": per_instruction,
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
        "predicted_target_xyz_l2_across_instructions": agg("predicted_target_xyz_l2"),
        "frac_same_nearest_card_despite_different_instruction": frac_same,
        "frac_pairs_identity_flips_with_instruction": (1 - frac_same) if frac_same is not None else None,
        "frac_predictions_nearest_to_the_instructed_card": frac_correct,
    }
    del policy
    torch.cuda.empty_cache()
    return {"checkpoint": checkpoint, "scenes": scene_results, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples", type=int, default=4000,
        help="0 = use the entire held-out split (very slow); default is a shuffled subset.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--skip-instruction-swap-probe", action="store_true")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_audit_5k")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    part_a = run_held_out_eval(args.checkpoint, args.device, args.batch_size, args.num_workers, args.max_samples)

    part_b = None
    if not args.skip_instruction_swap_probe:
        part_b = run_instruction_swap_probe(args.checkpoint, args.device, args.seeds)

    summary = {"part_a_held_out_target_point_head": part_a, "part_b_instruction_swap_probe": part_b}
    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("=== wrote %s ===", out_path)


if __name__ == "__main__":
    main()
