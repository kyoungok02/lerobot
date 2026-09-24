#!/usr/bin/env python
"""Fresh (no resume) 5k-step training for the target-supervision/conditioning ablation, isolating
two previously-coupled effects of `architecture="temporal_decoder_grounded_grasp"`: "does an
auxiliary grasp-target regression loss alone improve the pose trajectory" (condition B) vs "does
conditioning the decoder on that prediction help or hurt" (condition C) -- against the canonical
no-target-head baseline (condition A). See
`SafeDiffVLAConfig.grounded_grasp_condition_decoder_on_target`'s docstring for the mechanism.

  A. canonical: `architecture="temporal_decoder"` -- no TargetPointHead, no target conditioning,
     no reactive close. Identical recipe to `train_padfix_20k.py`'s own config, just 5k steps.
  B. auxiliary-target: `architecture="temporal_decoder_grounded_grasp"`,
     `grounded_grasp_condition_decoder_on_target=False` -- TargetPointHead + masked target loss
     trained, but the decoder never sees the predicted target; decoder's own gripper regression
     used as-is (no reactive close at eval time -- that's an eval-script choice, not a training one).
  C. target-conditioned: same as the already-completed "fixed-supervision" 5k run
     (`outputs/train/safediff_vla_temporal_decoder_grounded_grasp_5k/`,
     `grounded_grasp_condition_decoder_on_target=True`, the default) -- NOT retrained here
     (identical recipe/seed already exists; reused directly, see the ablation report script).

Same canonical hyperparameters as `train_padfix_20k.py`/`train_grounded_grasp_5k.py` throughout:
backbone frozen, action_horizon=execute_horizon=50, sin/cos + padfix (current canonical code,
nothing to toggle), use_temporal_ensembling=False, no subgoal, lambda_smooth=0.0, same
optimizer/scheduler. No threshold/lambda tuning, no reactive-close changes, no 20k run here.

Usage:
    uv run python examples/safediff_vla/train_ablation_5k.py --condition A
    uv run python examples/safediff_vla/train_ablation_5k.py --condition B
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

CONDITION_OVERRIDES = {
    "A": {"architecture": "temporal_decoder"},
    "B": {"architecture": "temporal_decoder_grounded_grasp", "grounded_grasp_condition_decoder_on_target": False},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=["A", "B"], required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--job-name", default=None)
    args = parser.parse_args()

    name = {
        "A": "safediff_vla_temporal_decoder_ablation_A_canonical_5k",
        "B": "safediff_vla_temporal_decoder_ablation_B_auxiliary_target_5k",
    }[args.condition]
    output_dir = args.output_dir or f"outputs/train/{name}"
    job_name = args.job_name or name

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
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
        grounded_grasp_gripper_open_threshold=0.5,
        grounded_grasp_close_threshold_m=0.03,
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-6,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
        **CONDITION_OVERRIDES[args.condition],
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", eval_split=0.0),
        policy=policy_cfg,
        rename_map=RENAME_MAP,
        output_dir=Path(output_dir),
        job_name=job_name,
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
