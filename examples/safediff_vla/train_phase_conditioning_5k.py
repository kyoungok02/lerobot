#!/usr/bin/env python
"""Quick 5k-step training run for the minimal binary phase-conditioning experiment
(`SafeDiffVLAConfig.use_phase_conditioning`, see its own docstring in
`configuration_safediff_vla.py` and `TemporalActionDecoder`'s `phase_embedding`).

Same architecture, backbone, dataset, and hyperparameters as the existing
`safediff_vla_temporal_decoder_sincos_5k`/`_20k` runs (see their saved
`pretrained_model/train_config.json`) -- the *only* change is `use_phase_conditioning=True` plus
the new `phase_labels_path`, which attaches a per-frame binary phase label
(`examples/safediff_vla/compute_phase_labels.py`) to every training batch. sin/cos rotation
representation is untouched (`temporal_decoder` already sin/cos-encodes internally); no subgoal
predictor, no smoothness loss (`lambda_subgoal`/`lambda_smooth` stay at their `temporal_decoder`
defaults of unused/0).

Prerequisite: run `examples/safediff_vla/compute_phase_labels.py` first (writes
`outputs/data/vlabench_phase_labels/labels.parquet` by default).

Usage:
    uv run python examples/safediff_vla/train_phase_conditioning_5k.py
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--phase-labels-path", default="outputs/data/vlabench_phase_labels/labels.parquet"
    )
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_temporal_decoder_phase_5k")
    parser.add_argument("--job-name", default="safediff_vla_temporal_decoder_phase_5k")
    args = parser.parse_args()

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder",
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
        use_phase_conditioning=True,
        phase_labels_path=args.phase_labels_path,
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-6,
        scheduler_warmup_steps=1_000,
        scheduler_decay_steps=30_000,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", eval_split=0.0),
        policy=policy_cfg,
        rename_map=RENAME_MAP,
        output_dir=Path(args.output_dir),
        job_name=args.job_name,
        seed=1000,
        num_workers=8,
        batch_size=4,
        steps=args.steps,
        log_freq=100,
        eval_steps=0,
        save_checkpoint=True,
        save_freq=args.steps,  # quick run -- only the final checkpoint
        use_policy_training_preset=True,
    )
    train(cfg)


if __name__ == "__main__":
    main()
