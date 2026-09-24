#!/usr/bin/env python
"""Fresh (no resume) 5k-step training for `architecture="temporal_decoder_grounded_grasp"` with the
modality-aware `_pooled_latent` fix (see `modeling_safediff_vla.py::SafeDiffVLAPolicy._pooled_latent`
and `utils.compute_prefix_modality_ids`): image tokens and text (instruction) tokens are now
masked-mean-pooled SEPARATELY (never diluting the instruction against the much larger image-token
count) and the single state token is used as-is, then concatenated + projected back down to
`_multimodal_latent_dim()` -- replacing the old uniform all-token mean.

Same "canonical" hyperparameters/recipe as `train_ablation_5k.py`'s condition B (auxiliary-target,
`grounded_grasp_condition_decoder_on_target=False`) -- backbone frozen, sin/cos + padfix, action_horizon=
execute_horizon=50, use_temporal_ensembling=False, no subgoal, lambda_smooth=0.0 -- per this round's
explicit scope: target conditioning OFF (decoder never sees the predicted target, isolating the
pooling fix's effect on TargetPointHead's own regression) and no reactive-close eval (this script
only trains; `eval_target_point_head_pooling_fix.py` does the held-out comparison).

ONLY new-vs-`train_ablation_5k.py --condition B` difference: `eval_split=0.1` (10% of episodes per
task held out, deterministically, by `lerobot.datasets.factory.make_train_eval_datasets` -- reused
verbatim by the eval script to recover the exact same held-out set) instead of `eval_split=0.0`, so
this run has genuine held-out data for the offline TargetPointHead xyz L2 evaluation the user asked
for. This is a quick 5k-step check, not the full run -- 20k is a deliberately separate, later decision.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_modality_pool_5k.py
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

EVAL_SPLIT = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument(
        "--output-dir", default="outputs/train/safediff_vla_temporal_decoder_grounded_grasp_modality_pool_5k"
    )
    parser.add_argument("--job-name", default="safediff_vla_temporal_decoder_grounded_grasp_modality_pool_5k")
    args = parser.parse_args()

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp",
        grounded_grasp_condition_decoder_on_target=False,
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
