#!/usr/bin/env python
"""5k-step LOCAL-REFINER-ONLY training run for `architecture="temporal_decoder_grounded_grasp_v2_5"`
(see `local_grasp_refiner.py` and `configuration_safediff_vla.py`'s architecture docstring): v2's
global grounding/approach pipeline (`TargetQueryExtractor`, `VisualGroundingCrossAttention`,
`TargetPointHead`, `TemporalActionDecoder`, and the frozen SmolVLA backbone) is INITIALIZED from an
existing v2 checkpoint (`--pretrained-path`, defaulting to the 15k checkpoint of the
`safediff_vla_grounded_grasp_v2_20k` run already on this branch) and kept FROZEN
(`local_grasp_freeze_global_modules=True`, the default) -- ONLY the new `LocalGraspRefiner` module
trains. This is the first experiment for this architecture: confirm the local closed-loop residual
controller can improve grasp precision WITHOUT touching v2's own already-checkpointed global
behavior at all. A fresh full-model v2.5 run (`local_grasp_freeze_global_modules=False`) is a
separate, later decision -- not run by this script.

Condition:
  - `pretrained_path` initializes backbone + all v2 modules from an existing v2 checkpoint; weight
    loading is non-strict (`PreTrainedPolicy.from_pretrained`'s own default) so the NEW
    `local_grasp_refiner.*` parameters (absent from that checkpoint) are reported as "missing" and
    simply keep their fresh random init -- this is expected, not an error.
  - `local_grasp_freeze_global_modules=True` (default): backbone, `TargetQueryExtractor`,
    `VisualGroundingCrossAttention`, `TargetPointHead`, `TemporalActionDecoder` all frozen --
    `get_optim_params()` (used automatically via `use_policy_training_preset=True`) then contains
    ONLY `LocalGraspRefiner`'s own parameters.
  - Every OTHER hyperparameter matches v2's own 5k/20k runs on this branch exactly (action_horizon=
    execute_horizon=50, use_temporal_ensembling=False, lambda_smooth=0.0, same optimizer/scheduler/
    seed/batch_size/num_workers, same genuine 10% held-out split) -- the global stage is meant to
    behave identically to v2, so nothing about its own training condition changes.
  - `local_grasp_*` fields (radius, max deltas, action horizon, execute horizon, window_k) and
    `lambda_local_grasp` are all left at their conservative config defaults -- no sweep this pilot.

5k steps only -- do not extend to 20k.

Usage:
    uv run python examples/safediff_vla/train_grounded_grasp_v2_5_5k.py
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

EVAL_SPLIT = 0.1  # genuine held-out split; same repo_id+eval_split as every prior run on this branch
DEFAULT_V2_CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/015000/pretrained_model"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--pretrained-path", default=DEFAULT_V2_CHECKPOINT)
    parser.add_argument("--output-dir", default="outputs/train/safediff_vla_grounded_grasp_v2_5_5k")
    parser.add_argument("--job-name", default="safediff_vla_grounded_grasp_v2_5_5k")
    args = parser.parse_args()
    if args.steps > 5000:
        raise ValueError("This pilot is 5k steps only -- do not extend to 20k without a separate decision.")

    policy_cfg = SafeDiffVLAConfig(
        device="cuda",
        push_to_hub=False,
        architecture="temporal_decoder_grounded_grasp_v2_5",
        pretrained_path=args.pretrained_path,
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
        local_grasp_radius_m=0.10,
        local_grasp_max_delta_xyz_m=0.03,
        local_grasp_max_delta_rot_rad=0.05,
        local_grasp_action_horizon=5,
        local_grasp_execute_horizon=1,
        local_grasp_window_k=10,
        lambda_local_grasp=1.0,
        local_grasp_hidden_dim=256,
        local_grasp_freeze_global_modules=True,
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
