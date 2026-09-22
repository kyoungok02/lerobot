#!/usr/bin/env python
"""Fresh (no resume) 5k-step training run for `architecture="temporal_decoder_grounded_grasp"`
with `LanguageGroundedTargetPooling` (text-queried cross-attention over image tokens feeding
`TargetPointHead`, see `language_grounded_target_pooling.py`) -- replacing the modality-aware
masked-mean pooling `train_grounded_grasp_audit_5k.py` used for the same target head.

Identical condition to `train_grounded_grasp_audit_5k.py` in every other respect, so the two
checkpoints isolate exactly this one architectural change:
  - target conditioning OFF: `grounded_grasp_condition_decoder_on_target=False`.
  - reactive close OFF: not exercised (training never calls `select_action`).
  - no phase/replan: not applicable to this architecture (no subgoal/completion-gate logic).
  - sin/cos rotation encoding + padfix action-mask: unconditional, unchanged.
  - action_horizon=execute_horizon=50, backbone frozen, same optimizer/scheduler as
    `train_grounded_grasp_audit_5k.py` (lr=1e-4, wd=1e-6, warmup=1000, decay=30000), same
    seed=1000, batch_size=4, num_workers=8.
  - genuine 10% held-out split: `DatasetConfig(eval_split=0.1)` -- same deterministic split
    (repo_id + eval_split are the only inputs) as the modality-aware-pooling baseline, so both
    checkpoints are evaluated against the identical held-out episodes.

Usage:
    uv run python examples/safediff_vla/train_language_grounded_target_5k.py
"""

import argparse
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.scripts.lerobot_train import train

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}

EVAL_SPLIT = 0.1  # genuine held-out split; must match eval_grounded_grasp_audit_5k.py's EVAL_SPLIT


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_language_grounded_target_5k")
    parser.add_argument("--job-name", default="safediff_vla_language_grounded_target_5k")
    args = parser.parse_args()

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp",
        action_horizon=50,
        execute_horizon=50,
        use_temporal_ensembling=False,
        backbone_name="lerobot/smolvla_vlabench",
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        freeze_backbone=True,
        freeze_vision_encoder=True,
        use_lora=False,
        decoder_hidden_dim=512,
        decoder_num_layers=4,
        decoder_num_heads=8,
        decoder_ffn_dim=1024,
        decoder_dropout=0.1,
        use_backbone_domain_adapter=False,
        backbone_action_conversion_semantics="per_step",
        lambda_smooth=0.0,
        lambda_target=1.0,
        target_head_hidden_dim=256,
        grounded_grasp_condition_decoder_on_target=False,  # target conditioning OFF
        grounded_grasp_gripper_open_threshold=0.5,
        grounded_grasp_close_threshold_m=0.03,
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-6,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", eval_split=EVAL_SPLIT),
        policy=policy_cfg,
        rename_map=RENAME_MAP,
        output_dir=Path(args.output_dir),
        job_name=args.job_name,
        resume=False,
        seed=1000,
        num_workers=8,
        batch_size=4,
        steps=args.steps,
        log_freq=100,
        eval_steps=0,
        save_checkpoint=True,
        save_freq=args.save_freq,
        use_policy_training_preset=True,
    )
    train(cfg)


if __name__ == "__main__":
    main()
