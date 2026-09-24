#!/usr/bin/env python
"""Held-out offline evaluation of `TargetPointHead` for the modality-aware `_pooled_latent` fix
(see `modeling_safediff_vla.py::SafeDiffVLAPolicy._pooled_latent` and
`utils.compute_prefix_modality_ids`) against the pre-fix all-token-mean-pooling checkpoint.

No rollout, no reactive close, no target conditioning -- purely offline dataset/scene probes,
matching this round's explicit scope (target conditioning OFF, reactive close OFF).

Part A -- held-out `TargetPointHead` physical xyz L2 (`--held-out-checkpoints`, default: the new
    modality-pool checkpoint only; pass the old checkpoint too for a side-by-side row):
  For every held-out-split frame with a valid grasp-transition label (`utils.find_grasp_target`,
  same held-out split as training via `eval_split=0.1` -- see
  `train_grounded_grasp_modality_pool_5k.py`), computes the model's predicted target xyz (physical
  meters) against GT, and against two baselines that need no model at all:
    - current-EE baseline: predict the observation's own current EE position (i.e. "target ==
      where the arm already is").
    - mean-target baseline: predict the held-out split's own mean GT target position (i.e. "target
      == the average across the whole split", ignoring the observation entirely).
  Reports mean L2 and mean |dx|/|dy|/|dz| for the model and both baselines.

Part B -- same-image instruction-swap probe (reuses `instruction_grounding_audit.py`'s scene
    capture/probe functions directly, not reimplemented): for a fixed real observation (one
    `env.reset()`, not a rollout) and several real card identities present in that same scene,
    varies ONLY the instruction text and measures predicted-target-xyz movement and how often the
    nearest real card to the prediction actually changes with it -- comparing the OLD (pre-fix)
    and NEW (modality-aware) checkpoints under identical scenes/instructions.

Outputs:
    outputs/eval/safediff_vla_pooling_fix/summary.json

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/eval_target_point_head_pooling_fix.py
    # Part A only (no simulator needed):
    uv run python examples/safediff_vla/eval_target_point_head_pooling_fix.py --skip-instruction-swap-probe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import instruction_grounding_audit as iga  # noqa: E402

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_train_eval_datasets  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.utils import find_grasp_target  # noqa: E402
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch  # noqa: E402
from lerobot.utils.collate import lerobot_collate_fn  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

RENAME_MAP = iga.RENAME_MAP
EVAL_SPLIT = 0.1  # must match train_grounded_grasp_modality_pool_5k.py's EVAL_SPLIT exactly
# `ablation_B_auxiliary_target_5k` (from `train_ablation_5k.py --condition B`), not the original
# `grounded_grasp_5k` checkpoint: both were trained with `grounded_grasp_condition_decoder_on_target
# =False` (this round's setting -- see `train_grounded_grasp_modality_pool_5k.py`'s docstring) and
# an otherwise IDENTICAL recipe/seed, so this isolates the pooling-code change as the only
# difference. `grounded_grasp_condition_decoder_on_target` doesn't change how `TargetPointHead`
# itself is trained (only whether the decoder consumes its prediction), so this is still a valid
# "old pooling" reference for TargetPointHead's own regression quality either way.
OLD_POOLING_CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_ablation_B_auxiliary_target_5k/checkpoints/005000/pretrained_model"
NEW_POOLING_CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_grounded_grasp_modality_pool_5k/checkpoints/005000/pretrained_model"


def _held_out_dataloader(
    batch_size: int, num_workers: int, shuffle: bool
) -> tuple[torch.utils.data.DataLoader, int, list[str]]:
    """The exact same held-out split `train_grounded_grasp_modality_pool_5k.py` trained against
    (`make_train_eval_datasets` splits deterministically off `eval_split`+`repo_id`, independent of
    which policy config is passed -- only `dataset`/`trainable_config.chunk_size` matter here).
    Also returns `dataset.meta.camera_keys` -- the RAW (pre-`rename_map`) camera key names
    `_preprocess_dataset_batch` needs, exactly as `lerobot_train.py` itself sources them.

    `shuffle`: the held-out split is ~348K FRAMES (not episodes) across 1254 episodes/295 tasks --
    consecutive frame indices are temporally adjacent within the same episode, so an unshuffled
    pass truncated by `--max-samples` would only ever see the first few episodes' own (nearly
    identical, since a grasp-target barely moves within one short window) targets, badly biasing
    both the L2 numbers and especially the `mean_target_baseline` (which would trivially match
    whatever few targets happened to be sampled). Shuffle (fixed seed, reproducible) whenever
    `--max-samples` truncates the pass; a `--max-samples 0` full pass doesn't need it."""
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
    """One batch -> `(predicted_xyz_m, gt_xyz_m, current_ee_xyz_m, valid_mask)`, all `[B, 3]`
    physical meters except `valid_mask` (`[B]` bool) -- exactly mirrors
    `SafeDiffVLAPolicy._forward_temporal_decoder`'s own GT-label extraction (same
    `_degrip_for_transition_detection` + `find_grasp_target` call), but keeps the raw xyz instead
    of only the scalar loss."""
    from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
    from lerobot.policies.safediff_vla.utils import pad_or_crop_horizon, pad_or_crop_mask

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
        "n_frames_seen": n_seen,
        "n_valid_labels": n_valid,
        "model": stats(pred, "target_point_head"),
        "current_ee_baseline": stats(ee, "current_ee_baseline"),
        "mean_target_baseline": stats(mean_target, "mean_target_baseline"),
    }
    logger.info("  model mean_l2_m=%.4f  current_ee mean_l2_m=%.4f  mean_target mean_l2_m=%.4f",
                result["model"]["mean_l2_m"], result["current_ee_baseline"]["mean_l2_m"], result["mean_target_baseline"]["mean_l2_m"])
    del policy
    torch.cuda.empty_cache()
    return result


def run_instruction_swap_probe(device: str, output_dir: Path) -> dict:
    """Part B: reruns `instruction_grounding_audit.py`'s own live-scene probe once per checkpoint
    (OLD pre-fix pooling, NEW modality-aware pooling), against the SAME scenes/instructions, and
    reports both plus a direct comparison."""
    logger.info("=== Part B: same-image instruction-swap probe (old vs new pooling) ===")
    reports = {}
    for label, checkpoint in (("old_all_token_mean_pooling", OLD_POOLING_CHECKPOINT), ("new_modality_aware_pooling", NEW_POOLING_CHECKPOINT)):
        out_path = output_dir / f"instruction_grounding_audit_{label}.json"
        logger.info("  running instruction_grounding_audit for %s (%s)", label, checkpoint)
        iga.GROUNDED_CHECKPOINT = checkpoint
        argv_backup = sys.argv
        try:
            sys.argv = ["instruction_grounding_audit.py", "--device", device, "--output", str(out_path), "--grounded-checkpoint", checkpoint]
            iga.main()
        finally:
            sys.argv = argv_backup
        reports[label] = json.loads(out_path.read_text())

    comparison = []
    for old_scene, new_scene in zip(reports["old_all_token_mean_pooling"]["scenes"], reports["new_modality_aware_pooling"]["scenes"], strict=True):
        assert old_scene["seed"] == new_scene["seed"]
        comparison.append(
            {
                "seed": old_scene["seed"],
                "true_target_entity": old_scene["true_target_entity"],
                "old_pooled_latent_mean_l2_across_instructions": old_scene["pooled_latent_pairwise"].get("mean_l2"),
                "new_pooled_latent_mean_l2_across_instructions": new_scene["pooled_latent_pairwise"].get("mean_l2"),
                "old_predicted_target_xyz_mean_l2_across_instructions": old_scene["predicted_target_xyz_pairwise"].get("mean_l2"),
                "new_predicted_target_xyz_mean_l2_across_instructions": new_scene["predicted_target_xyz_pairwise"].get("mean_l2"),
                "old_fraction_instruction_pairs_same_nearest_card": old_scene["fraction_instruction_pairs_same_nearest_card"],
                "new_fraction_instruction_pairs_same_nearest_card": new_scene["fraction_instruction_pairs_same_nearest_card"],
            }
        )
    return {"per_checkpoint_reports": {k: str(output_dir / f"instruction_grounding_audit_{k}.json") for k in reports}, "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4000,
        help="0 = use the entire held-out split (~348K frames -- very slow); default is a shuffled subset.",
    )
    parser.add_argument(
        "--held-out-checkpoints",
        nargs="+",
        default=[NEW_POOLING_CHECKPOINT, OLD_POOLING_CHECKPOINT],
        help="Part A runs held-out xyz-L2 for each of these checkpoints (default: new fix, then old baseline).",
    )
    parser.add_argument("--skip-instruction-swap-probe", action="store_true")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_pooling_fix")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    held_out_results = [
        run_held_out_eval(ckpt, args.device, args.batch_size, args.num_workers, args.max_samples)
        for ckpt in args.held_out_checkpoints
    ]

    # Part B needs the VLABench/dm_control simulator (a manual, non-pip install -- see
    # `pyproject.toml`'s vlabench note); not every environment running this script has it. Never
    # let that swallow Part A's (already computed, often much slower to redo) results -- write
    # what succeeded either way and report the failure in the summary instead of crashing.
    instruction_swap = None
    if not args.skip_instruction_swap_probe:
        try:
            instruction_swap = run_instruction_swap_probe(args.device, output_dir)
        except Exception as e:  # noqa: BLE001
            logger.exception("Part B (instruction-swap probe) failed; Part A results are unaffected")
            instruction_swap = {"error": f"{type(e).__name__}: {e}"}

    summary = {
        "part_a_held_out_target_point_head": held_out_results,
        "part_b_instruction_swap_probe": instruction_swap,
    }
    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("=== wrote %s ===", out_path)


if __name__ == "__main__":
    main()
