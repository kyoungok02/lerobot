#!/usr/bin/env python
"""DIAGNOSTIC ONLY -- no training, no architecture/config/threshold changes.

Compares `temporal_decoder_grounded_grasp_v2_5`'s LOCAL_GRASP training-distribution metrics
(`loss_local_pos`/`loss_local_rot`/`loss_local_grip`, `local_eligible_frac`, and mean predicted
residual |delta_xyz| magnitude) across the 5k/10k/15k checkpoints of the SAME continued run
(`train_grounded_grasp_v2_5_5k.py` resumed via `lerobot-train --resume` to 15k). These metrics
were not logged as training-time scalars for this architecture when the 5k run completed, so this
script recomputes them post hoc, identically for all three checkpoints, over the SAME held-out
eval-split batches (same deterministic task-wise split `lerobot-train` itself uses, built via
`make_train_eval_datasets` -- never episodes any checkpoint was trained on), so the comparison is
apples-to-apples and not an artifact of different random batches.

Usage:
    uv run python examples/safediff_vla/compare_v2_5_checkpoint_training_metrics.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
RUN_DIR = "outputs/train/safediff_vla_grounded_grasp_v2_5_5k"
CHECKPOINTS = {
    "5k": f"{RUN_DIR}/checkpoints/005000/pretrained_model",
    "10k": f"{RUN_DIR}/checkpoints/010000/pretrained_model",
    "15k": f"{RUN_DIR}/checkpoints/015000/pretrained_model",
}


def build_eval_batches(n_batches: int, batch_size: int, seed: int) -> tuple[list[dict[str, torch.Tensor]], list[str]]:
    """Same deterministic task-wise eval split `lerobot-train` itself builds
    (`make_train_eval_datasets`) -- a placeholder policy config is only used to supply
    `action_horizon`/`chunk_size` to the dataset factory; no policy is constructed from it.
    Returns `(raw_batches, camera_keys)` -- `camera_keys` (pre-rename dataset key names) is needed
    by `_preprocess_dataset_batch` below, exactly like the real training loop uses it."""
    placeholder_policy_cfg = SafeDiffVLAConfig(
        architecture="temporal_decoder_grounded_grasp_v2_5",
        device="cpu",
        action_horizon=50,
        execute_horizon=50,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="lerobot/vlabench_unified", eval_split=0.1),
        policy=placeholder_policy_cfg,
        rename_map=RENAME_MAP,
        output_dir=Path("outputs/train/_scratch_compare_v2_5_checkpoints"),
        job_name="scratch_compare_v2_5_checkpoints",
        resume=False,
        seed=seed,
        steps=1,
        use_policy_training_preset=True,
    )
    _, eval_dataset = make_train_eval_datasets(cfg)
    assert eval_dataset is not None
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= n_batches:
            break
    return batches, eval_dataset.meta.camera_keys


def evaluate_checkpoint(checkpoint: str, batches: list[dict[str, torch.Tensor]], camera_keys: list[str], device: str) -> dict:
    policy = SafeDiffVLAPolicy.from_pretrained(checkpoint)
    policy = policy.to(device)
    policy.eval()
    # Same preprocessing (image dtype -> float, key rename, normalize, device move) the real
    # training loop applies to every batch before `policy(batch)` -- see
    # `lerobot_train.py`'s own `_preprocess_dataset_batch` / `make_pre_post_processors` usage.
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )

    orig_forward = policy.local_grasp_refiner.forward
    delta_mags: list[float] = []

    def spy_forward(*args, **kwargs):
        delta_xyz, delta_rot, gripper = orig_forward(*args, **kwargs)
        delta_mags.extend(delta_xyz[:, 0].norm(dim=-1).detach().cpu().tolist())  # first (executed) step only
        return delta_xyz, delta_rot, gripper

    policy.local_grasp_refiner.forward = spy_forward

    agg = {"loss_local_pos": [], "loss_local_rot": [], "loss_local_grip": [], "local_eligible_frac": []}
    with torch.no_grad():
        for raw_batch in batches:
            batch = _preprocess_dataset_batch(dict(raw_batch), camera_keys, RENAME_MAP, preprocessor)
            _, metrics = policy(batch)
            for key in agg:
                if key in metrics:
                    agg[key].append(metrics[key])

    del policy.local_grasp_refiner.forward  # restore bound method lookup to the class's own
    result = {key: (sum(vals) / len(vals) if vals else None) for key, vals in agg.items()}
    # NOTE: this magnitude is averaged over EVERY sample the refiner ran a forward pass on, not
    # just the (small) `local_eligible_frac` fraction the loss is actually masked to -- most
    # samples in a random batch are far from any grasp event, so this reflects the network's
    # general output scale, not specifically its behavior on in-window samples.
    result["mean_local_residual_xyz_magnitude_m_all_samples"] = (sum(delta_mags) / len(delta_mags)) if delta_mags else None
    result["n_batches"] = len(batches)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/safediff_vla_grounded_grasp_v2_5_checkpoint_training_metrics.json")
    args = parser.parse_args()

    logger.info("=== building %d held-out eval-split batches (batch_size=%d, seed=%d) ===", args.n_batches, args.batch_size, args.seed)
    batches, camera_keys = build_eval_batches(args.n_batches, args.batch_size, args.seed)
    logger.info("built %d batches", len(batches))

    results = {}
    for tag, checkpoint in CHECKPOINTS.items():
        if not Path(checkpoint).is_dir():
            logger.warning("checkpoint %s (%s) does not exist yet -- skipping", tag, checkpoint)
            continue
        logger.info("=== evaluating checkpoint %s: %s ===", tag, checkpoint)
        results[tag] = evaluate_checkpoint(checkpoint, batches, camera_keys, args.device)
        logger.info("%s: %s", tag, json.dumps(results[tag], indent=2))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
