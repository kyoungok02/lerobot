#!/usr/bin/env python
"""Fresh (no resume) 5k-step training run for `architecture="temporal_decoder_instruction"` --
the canonical `temporal_decoder` decoder with ONE change: the instruction itself (masked-mean
TEXT_MODALITY tokens, see `utils.masked_mean_by_modality`) is added as global conditioning to
every horizon query, exactly like `subgoal_state`/`target_xyz` already are for the other
experimental architectures (see `temporal_decoder.py`'s `use_instruction`).

Deliberately NOT the `temporal_decoder_grounded_grasp` path: no `TargetPointHead`, no target
conditioning, no auxiliary regression loss, no reactive close (that method is only ever invoked
for `architecture="temporal_decoder_grounded_grasp"`), no subgoal/phase logic. Two independent
`TargetPointHead`-pooling designs (masked-mean-query and per-token cross-attention) both improved
held-out xyz L2 while leaving instruction-conditioned card selection at or worse than baseline
(identity-flip rate 25% -> 0%, correct-instructed-card rate 33.3% -> 27.8% for both). This
experiment tests a structurally different hypothesis: put the instruction signal directly on the
action-generating query instead of funneling it through a separate auxiliary head.

Same condition as `train_grounded_grasp_audit_5k.py`/`train_language_grounded_target_5k.py`
otherwise: action_horizon=execute_horizon=50, backbone frozen, identical optimizer/scheduler
(lr=1e-4, wd=1e-6, warmup=1000, decay=30000), seed=1000, batch_size=4, num_workers=8, genuine
`eval_split=0.1` held-out split (same deterministic split as both prior runs, since repo_id+
eval_split are the only inputs).

Usage:
    uv run python examples/safediff_vla/train_instruction_conditioned_decoder_5k.py
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

EVAL_SPLIT = 0.1  # genuine held-out split; same repo_id+eval_split as the prior two runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_instruction_conditioned_decoder_5k")
    parser.add_argument("--job-name", default="safediff_vla_instruction_conditioned_decoder_5k")
    args = parser.parse_args()

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_instruction",
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
