#!/usr/bin/env python
"""Trains ONLY a `TargetPointHead` + modality-aware pooling projection (the simplest, most
stable pooling design -- `SafeDiffVLAPolicy._pooled_latent`'s masked-mean image/text/state
concat+project, NOT either of the two abandoned cross-attention designs) on the counterfactual
grounding dataset (`generate_counterfactual_grounding_dataset.py`): same image/state, only the
instruction varies, and the correct label varies with it -- a contrast `select_poker`'s own
demonstrations never provide.

Deliberately NOT trained through `SafeDiffVLAPolicy.forward()`/the action loss at all: the
canonical `TemporalActionDecoder` is never constructed or touched, there is no target
conditioning, no reactive close, no action loss mixed in. This is a fresh `TargetPointHead`
(random init) plus a fresh `modality_pool_projection` (random init), reading a REAL
`architecture="temporal_decoder_subgoal"` policy instance purely as a frozen-backbone multimodal
encoder (`_encode_multimodal_latent`) and its own `_pooled_latent` method (untouched,
unmodified) -- no changes to any file under `src/lerobot/policies/safediff_vla/`. Only
`modality_pool_projection` and this script's own `TargetPointHead` are optimized; the backbone
stays frozen throughout.

`TargetPointHead` output here is trained directly in PHYSICAL METERS (robot-frame), not the
MEAN_STD-normalized action-position space `_predict_target_xyz` uses inside the real policy --
this is an intentionally decoupled pilot (see module docstring in
`generate_counterfactual_grounding_dataset.py`); reconciling the two spaces is a later step, only
if this pilot's own success criterion is met.

Success criterion (per the pilot's own priority): correct-instructed-card rate clearly above the
existing four-architecture baseline (~28-33%), not xyz L2 improving on its own.

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/train_counterfactual_grounding_head.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from lerobot.envs import make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.target_point_head import TargetPointHead

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
# Any already-trained checkpoint on this dataset -- used ONLY to source a valid config schema
# (input/output feature shapes) and the preprocessor's dataset-derived normalization stats
# (image/state), never its model weights: the policy here is constructed fresh, and this
# script's own TargetPointHead is a brand-new module, not loaded from anywhere.
BASELINE_CHECKPOINT = "outputs/train/safediff_vla_grounded_grasp_audit_5k/checkpoints/005000/pretrained_model"
DATASET_DIR = Path("outputs/data/safediff_vla_counterfactual_grounding")


def load_dataset(dataset_dir: Path) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    scene_obs: dict[int, dict[str, torch.Tensor]] = {}
    for scene in manifest["scenes"]:
        scene_obs[scene["seed"]] = torch.load(dataset_dir / "scenes" / f"scene_{scene['seed']}.pt", weights_only=False)
    train_samples = [s for s in manifest["samples"] if s["split"] == "train"]
    held_out_samples = [s for s in manifest["samples"] if s["split"] == "held_out"]
    return scene_obs, train_samples, held_out_samples, manifest


def collate_batch(
    samples: list[dict[str, Any]], scene_obs: dict[int, dict[str, torch.Tensor]]
) -> tuple[dict[str, Any], torch.Tensor]:
    batch: dict[str, Any] = {}
    for key in scene_obs[samples[0]["seed"]]:
        batch[key] = torch.cat([scene_obs[s["seed"]][key] for s in samples], dim=0)
    batch["task"] = [s["instruction"] for s in samples]
    labels = torch.tensor([s["label_xyz"] for s in samples], dtype=torch.float32)
    return batch, labels


def encode_and_pool(
    policy: SafeDiffVLAPolicy, preprocessor, env_preprocessor, batch: dict[str, Any], device: str
) -> torch.Tensor:
    obs_t = env_preprocessor(batch)
    obs_norm = preprocessor(obs_t)
    latent_tokens, latent_pad_mask, latent_modality_ids = policy._encode_multimodal_latent(obs_norm)
    return policy._pooled_latent(latent_tokens, latent_pad_mask, latent_modality_ids)


def evaluate_held_out(
    policy: SafeDiffVLAPolicy,
    target_point_head: TargetPointHead,
    preprocessor,
    env_preprocessor,
    scene_obs: dict[int, dict[str, torch.Tensor]],
    held_out_samples: list[dict[str, Any]],
    manifest: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    policy.eval()
    target_point_head.eval()
    scenes_by_seed = {s["seed"]: s for s in manifest["scenes"] if s["split"] == "held_out"}

    preds_by_seed: dict[int, dict[str, np.ndarray]] = {}
    all_l2 = []
    with torch.no_grad():
        for sample in held_out_samples:
            batch, label = collate_batch([sample], scene_obs)
            pooled = encode_and_pool(policy, preprocessor, env_preprocessor, batch, device)
            pred = target_point_head(pooled)[0].cpu().numpy()
            preds_by_seed.setdefault(sample["seed"], {})[sample["card"]] = pred
            all_l2.append(float(np.linalg.norm(pred - np.array(sample["label_xyz"]))))

    correct = []
    pair_same_nearest = []
    pair_movement = []
    for seed, preds in preds_by_seed.items():
        cards = {name: np.array(pos) for name, pos in scenes_by_seed[seed]["cards"].items()}

        def nearest(point: np.ndarray) -> str:
            return min(cards, key=lambda name: float(np.linalg.norm(point - cards[name])))

        nearest_by_card = {instructed_card: nearest(pred) for instructed_card, pred in preds.items()}
        for instructed_card, nearest_card in nearest_by_card.items():
            correct.append(nearest_card == instructed_card)
        for a, b in itertools.combinations(preds.keys(), 2):
            pair_same_nearest.append(nearest_by_card[a] == nearest_by_card[b])
            pair_movement.append(float(np.linalg.norm(preds[a] - preds[b])))

    return {
        "n_held_out_samples": len(held_out_samples),
        "n_held_out_scenes": len(scenes_by_seed),
        "mean_xyz_l2_m": float(np.mean(all_l2)),
        "median_xyz_l2_m": float(np.median(all_l2)),
        "correct_instructed_card_rate": float(np.mean(correct)) if correct else None,
        "identity_flip_rate": float(1 - np.mean(pair_same_nearest)) if pair_same_nearest else None,
        "same_nearest_card_rate": float(np.mean(pair_same_nearest)) if pair_same_nearest else None,
        "instruction_swap_movement_mean_m": float(np.mean(pair_movement)) if pair_movement else None,
        "n_instruction_pairs": len(pair_same_nearest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_counterfactual_grounding_head")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== loading counterfactual grounding dataset from %s ===", args.dataset_dir)
    scene_obs, train_samples, held_out_samples, manifest = load_dataset(Path(args.dataset_dir))
    logger.info(
        "  %d train samples (%d scenes), %d held-out samples (%d scenes)",
        len(train_samples), manifest["n_train_scenes"], len(held_out_samples), manifest["n_held_out_scenes"],
    )

    logger.info("=== building frozen-backbone policy (architecture=temporal_decoder_subgoal, for _pooled_latent only) ===")
    config = SafeDiffVLAConfig.from_pretrained(BASELINE_CHECKPOINT)
    config.architecture = "temporal_decoder_subgoal"
    config.device = device
    config.push_to_hub = False
    policy = SafeDiffVLAPolicy(config)
    policy = policy.to(device)
    assert all(not p.requires_grad for p in policy.backbone.parameters()), "backbone must stay frozen"

    target_point_head = TargetPointHead(policy._multimodal_latent_dim(), hidden_dim=256).to(device)

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=BASELINE_CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_cfg = VLABenchEnv(task=TASK)
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=config)

    trainable_params = list(policy.modality_pool_projection.parameters()) + list(target_point_head.parameters())
    logger.info("  trainable params: %d", sum(p.numel() for p in trainable_params))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    logger.info("=== training grounding head only (%d epochs, batch_size=%d) ===", args.epochs, args.batch_size)
    for epoch in range(args.epochs):
        policy.train()
        target_point_head.train()
        # `policy.backbone` stays in whatever mode `.train()` puts it in above, but its own
        # params never receive gradients (frozen at construction) -- `.eval()` isn't needed for
        # correctness here since nothing in `_encode_multimodal_latent`/`_pooled_latent` uses
        # dropout/batchnorm-style train/eval-dependent behavior at the pooling-projection level.
        order = list(range(len(train_samples)))
        random.shuffle(order)
        epoch_losses = []
        for i in range(0, len(order), args.batch_size):
            batch_samples = [train_samples[j] for j in order[i : i + args.batch_size]]
            batch, labels = collate_batch(batch_samples, scene_obs)
            labels = labels.to(device)
            pooled = encode_and_pool(policy, preprocessor, env_preprocessor, batch, device)
            pred = target_point_head(pooled)
            loss = F.mse_loss(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            logger.info("  epoch %d/%d: train_mse=%.5f", epoch + 1, args.epochs, float(np.mean(epoch_losses)))

    logger.info("=== evaluating on held-out scenes ===")
    result = evaluate_held_out(
        policy, target_point_head, preprocessor, env_preprocessor, scene_obs, held_out_samples, manifest, device
    )
    result["baseline_for_comparison"] = {
        "correct_instructed_card_rate_range": [0.278, 0.333],
        "identity_flip_rate_range": [0.0, 0.25],
        "note": "four independent architecture-only interventions (mean-query, per-token, decoder-query, un-pooled text cross-attn) all converged inside these ranges without any counterfactual supervision",
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    logger.info("=== result ===")
    logger.info(json.dumps(result, indent=2))
    logger.info("wrote %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
