#!/usr/bin/env python
"""INDEPENDENT fresh (no resume) 5k-step training run for
`architecture="temporal_decoder_grounded_grasp"`, run from this audit branch
(`b-grounded-grasp-audit`) to independently reproduce a comparable data point to whatever prior
run(s) exist under `experimental-grounded-grasp` -- this script does not read, import, or depend
on anything from that branch or from any pre-existing checkpoint/output directory.

Condition (as specified for this audit round):
  - modality-aware pooling: on unconditionally -- this is no longer a toggle, it's just how
    `SafeDiffVLAPolicy._pooled_latent` works after commit "Use modality-aware pooling for grounded
    grasp target prediction" (already present on this branch; verified via `git diff` against
    `origin/experimental-grounded-grasp`'s tip to be byte-identical except for unrelated read-only
    analysis scripts).
  - target conditioning OFF: `grounded_grasp_condition_decoder_on_target=False` -- `TargetPointHead`
    is still built and its auxiliary `lambda_target` regression loss still trains it, but the
    decoder never sees its prediction (see that field's docstring in
    `configuration_safediff_vla.py`).
  - reactive close OFF: nothing to set here -- `_grounded_grasp_reactive_close` only ever runs
    inside `select_action` (closed-loop env rollout). This script only calls `train()`, which never
    calls `select_action`, so reactive close is simply never exercised.
  - sin/cos rotation encoding + padfix action-mask: both unconditional in the current
    `temporal_decoder` path (no toggle left to set).
  - action_horizon=execute_horizon=50, backbone frozen, same optimizer/scheduler as
    `train_grounded_grasp_5k.py` (lr=1e-4, wd=1e-6, warmup=1000, decay=30000), same seed=1000,
    batch_size=4, num_workers=8.
  - genuine 10% held-out split: `DatasetConfig(eval_split=0.1)` -- `make_train_eval_datasets`
    deterministically holds out the last ceil(n_eps*0.1) episodes per task and never shows them to
    the training dataloader (see `lerobot.datasets.factory.make_train_eval_datasets`). Any held-out
    evaluation against this checkpoint must reconstruct the same split by calling
    `make_train_eval_datasets` with an identical `TrainPipelineConfig.dataset` (repo_id + eval_split
    are the only inputs that matter) -- see `eval_grounded_grasp_audit_5k.py`.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_audit_5k.py
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

EVAL_SPLIT = 0.1  # genuine held-out split; must match eval_grounded_grasp_audit_5k.py exactly


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_grounded_grasp_audit_5k")
    parser.add_argument("--job-name", default="safediff_vla_grounded_grasp_audit_5k")
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
