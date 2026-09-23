#!/usr/bin/env python
"""Fresh (no resume) 20k-step training run for `architecture="temporal_decoder_grounded_grasp_v2"`
-- a training-length/checkpoint sweep following the 5k pilot (`train_grounded_grasp_v2_5k.py`),
which was the first design in this line of experiments to show closed-loop target grounding and
task success improving together (2/3 select_poker seeds succeeded with 100% per-step
target_is_nearest; the third failed with 0%). This run does NOT modify the architecture, loss, or
execution logic in any way -- identical `SafeDiffVLAConfig` to the 5k pilot, just more steps and
more checkpoints, to see whether the 5k grounding signal holds, improves, or collapses with longer
training (the sin/cos-rotation-encoding precedent on this codebase found a fix that helped at 5k
partially regress at 20k -- this sweep exists to check whether the same thing happens here, not to
assume 20k is automatically the best checkpoint).

Condition (byte-identical to `train_grounded_grasp_v2_5k.py`'s `SafeDiffVLAConfig`, only
`steps`/`save_freq`/`output_dir`/`job_name` differ):
  - frozen backbone (`freeze_backbone=True`, `freeze_vision_encoder=True`).
  - action_horizon=execute_horizon=50.
  - use_temporal_ensembling=False.
  - not `temporal_decoder_subgoal` -- no subgoal predictor built at all.
  - lambda_smooth=0.0.
  - lambda_target=1.0.
  - sin/cos rotation encoding + padfix action-mask: unconditional, unchanged.
  - reactive close / binary grasp-phase conditioning: unchanged (`_grounded_grasp_v2_reactive_close`,
    `TemporalActionDecoder`'s `use_grounded_grasp_v2`) -- no code touched.
  - same optimizer/scheduler (lr=1e-4, wd=1e-6, warmup=1000, decay=30000), same seed=1000,
    batch_size=4, num_workers=8, genuine 10% held-out split (`eval_split=0.1`, same repo_id -- same
    deterministic split as the 5k pilot).

Checkpoints saved at steps 5000, 10000, 15000, 20000 (`save_freq=5000`, `should_save_checkpoint`
always saves on an exact multiple of `save_freq` and on the final step). Written to
`outputs/train/safediff_vla_grounded_grasp_v2_20k/` -- a SEPARATE directory from the 5k pilot's
`outputs/train/safediff_vla_grounded_grasp_v2_5k/`, which this script never reads, writes, or
deletes.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_v2_20k.py
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

EVAL_SPLIT = 0.1  # genuine held-out split; same repo_id+eval_split as the 5k pilot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_grounded_grasp_v2_20k")
    parser.add_argument("--job-name", default="safediff_vla_grounded_grasp_v2_20k")
    args = parser.parse_args()
    smoke_5k_dir = Path("outputs/train/safediff_vla_grounded_grasp_v2_5k")
    if Path(args.output_dir).resolve() == smoke_5k_dir.resolve():
        raise ValueError(f"refusing to write into the 5k pilot's own directory ({smoke_5k_dir}) -- it must be preserved.")

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp_v2",
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
