#!/usr/bin/env python
"""Fresh (no resume) 5k-step training run for `architecture="temporal_decoder_text_crossattn"` --
the canonical `temporal_decoder` decoder plus one EXTRA cross-attention over the raw, un-pooled
per-token text sequence, applied after the canonical multimodal cross-attention/self-attention
stack (see `temporal_decoder.py`'s `use_text_crossattn`):

    queries -> canonical decoder cross-attn(full multimodal memory) [untouched]
            -> extra text-only cross-attn(text tokens, no mean/max/last-token pooling of any kind)
            -> action head

Every prior instruction-conditioning attempt pooled the instruction into a single vector before
it ever reached the query -- `temporal_decoder_instruction`'s additive masked-mean-pooled
embedding, and `temporal_decoder_grounded_grasp`'s two `TargetPointHead` pooling designs
(masked-mean-query and per-token-query cross-attention over IMAGE tokens). All three produced the
same collapse: identity-flip rate 0-25%, correct-instructed-card rate 27.8-33.3%. This is the
first design to keep the text sequence fully un-pooled all the way to the action head.

No `TargetPointHead`, no auxiliary target loss, no reactive close (only ever invoked for
`architecture="temporal_decoder_grounded_grasp"`), no contrastive loss, no subgoal/phase logic.
Same condition as the three prior experiments otherwise: action_horizon=execute_horizon=50,
backbone frozen, identical optimizer/scheduler (lr=1e-4, wd=1e-6, warmup=1000, decay=30000),
seed=1000, batch_size=4, num_workers=8, genuine `eval_split=0.1` held-out split (same
deterministic split as all three prior runs).

Usage:
    uv run python examples/safediff_vla/train_text_crossattn_decoder_5k.py
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

EVAL_SPLIT = 0.1  # genuine held-out split; same repo_id+eval_split as all three prior runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_text_crossattn_decoder_5k")
    parser.add_argument("--job-name", default="safediff_vla_text_crossattn_decoder_5k")
    args = parser.parse_args()

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_text_crossattn",
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
