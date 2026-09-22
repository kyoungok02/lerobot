#!/usr/bin/env python
"""Offline diagnostic of `TargetPointHead` localization quality on the fresh 5k
`temporal_decoder_grounded_grasp` checkpoint -- NOT a new training/rollout run (except the
explicitly-scoped tiny-overfit capacity test in section 4, which trains only
`TargetPointHead`'s own parameters on a small fixed batch, everything else frozen).

Goal: separate "needs more of the same training (20k)" from "TargetPointHead's design/input
can't represent grasp xyz well enough", per the diagnostic plan:
  1. Held-out-ish target-point validation: physical-meter xyz error over many real, verified-valid
     (pre-grasp, real open->close transition in the future 50-step window) samples.
  2. Trivial baselines on the SAME samples: current-EE-xyz-as-target, and dataset-mean-grasp-xyz.
  3. Training-step learning curve: NOTED AS UNAVAILABLE -- `train_grounded_grasp_5k.py` used
     `save_freq=5000`, so no 1k/2k intermediate checkpoints exist. Reproducing them would require
     a new training run, which this diagnostic is explicitly scoped not to do.
  4. Tiny-set overfit capacity test: 64 fixed valid samples, `TargetPointHead` re-initialized from
     scratch, ONLY its own parameters trained (backbone/decoder frozen and not even invoked for
     the training loop -- the multimodal latent is computed once and cached, since it does not
     depend on `TargetPointHead` at all), for many steps on the same fixed batch.

Calibration notes (found while building this script, reported, NOT fixed here -- out of scope):

1. `SafeDiffVLAConfig.grounded_grasp_gripper_open_threshold` (default 0.5) is applied, during
   actual training (`_forward_temporal_decoder`), to the MEAN_STD-*normalized* gripper channel --
   not the raw physical one every other gripper-threshold convention in this codebase
   (`place_phase_forensics.py`'s `GRIPPER_THRESHOLD`, `compute_subgoal_labels.py`'s
   `GRIPPER_OPEN_THRESHOLD`, etc.) assumes. Minor, likely a few-frame shift near the transition.

2. MUCH more significant, verified directly against `lerobot/vlabench_unified`'s raw parquet
   columns across multiple episodes: `observation.state[6]` (gripper) uses the OPPOSITE sign
   convention from `action[6]` -- `action`: 1.0=open, 0.0=closed; `observation.state`: 0.0=open,
   1.0=closed (state lags action's own transition by exactly one frame, i.e. it reads back the
   *previous* commanded action's mechanical result). `utils.find_grasp_target`'s "pre-grasp" gate
   (`current_state[..., gripper_index] > threshold`) silently assumes the SAME convention as
   `action`, so for a genuinely pre-grasp (open) sample, `current_state`'s gripper is ~0.0, which
   reads as `False` ("not pre-grasp") under that gate -- meaning nearly all real training samples
   were being masked OUT as invalid, and the "valid" samples that got through were an
   uncontrolled, inverted-convention artifact, not necessarily bona fide pre-grasp windows. This
   directly corrupts `loss_target`'s effective training signal during the actual 5k run -- a much
   more likely explanation for poor localization than "needs more of the same training".
   NOT fixed here (would change training/label-extraction code, out of this diagnostic's scope).
   To get a *correct* ground truth for this diagnostic despite that bug, this script never calls
   `find_grasp_target` on `observation.state` -- it reuses the already-correct, action-only
   transition detection from `grasp_approach_precision_openloop.py`'s `find_grasp_frames` (which
   only ever compares `action[t-1]` vs `action[t]`, never touching `observation.state`) and reads
   the exact GT xyz directly out of the same batch's `action` window at that known offset.

No decoder-conditioning, reactive-close-threshold, or lambda_target changes. No 20k run.

Outputs:
    outputs/eval/safediff_vla_target_point_head_diagnosis/summary.json

Usage:
    uv run python examples/safediff_vla/diagnose_target_point_head.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from grasp_approach_precision_openloop import find_grasp_frames, select_candidate_episodes  # noqa: E402

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.target_point_head import TargetPointHead  # noqa: E402
from lerobot.processor.rename_processor import rename_batch_keys  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_grounded_grasp_5k/checkpoints/005000/pretrained_model"
DATASET_REPO_ID = "lerobot/vlabench_unified"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
MAX_OFFSET = 45  # steps before the grasp frame -- <50 guarantees the transition falls inside the
# future action_horizon=50 window from every such frame, so every row here is valid by construction.


def build_dense_pregrasp_rows(episodes: list[int], grasp_info: dict[int, dict], max_offset: int) -> pd.DataFrame:
    """Every frame from `max(0, grasp_frame - max_offset)` up to (excluding) `grasp_frame`, for
    every episode with a demonstrated grasp -- each one guaranteed to have the transition inside
    its own future `max_offset < 50`-step window. `k_to_transition` (1..max_offset) is the
    window-relative index (`action_delta_indices` offset from this row's own frame) of the action
    at the true grasp frame, i.e. `raw_batch[ACTION][row_i, k_to_transition, :3]` is the exact GT
    grasp xyz -- computed directly from the already-correct, action-only `grasp_frame` (see module
    docstring's calibration note 2), never from `observation.state`'s inverted gripper channel."""
    rows = []
    cursor = 0
    for ep in episodes:
        info = grasp_info[ep]
        length, grasp_frame = info["length"], info["grasp_frame"]
        if grasp_frame is not None:
            lo = max(0, grasp_frame - max_offset)
            for frame in range(lo, grasp_frame):
                rows.append(
                    {
                        "episode_index": ep,
                        "frame_index": frame,
                        "row": cursor + frame,
                        "k_to_transition": grasp_frame - frame,
                    }
                )
        cursor += length
    return pd.DataFrame(rows)


def gt_target_from_action_window(raw_batch: dict[str, torch.Tensor], k_to_transition: list[int]) -> torch.Tensor:
    """`raw_batch[ACTION][b, k_to_transition[b], :3]` for each row `b` -- the exact commanded xyz
    at the true (action-only-verified) grasp frame, gathered per-row since `k_to_transition`
    differs row to row."""
    k = torch.as_tensor(k_to_transition, dtype=torch.long)
    return raw_batch[ACTION][torch.arange(len(k)), k, :3]


def fetch_raw_batch(policy: SafeDiffVLAPolicy, episodes: list[int], rows: list[int]) -> dict[str, torch.Tensor]:
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET_REPO_ID, episodes=episodes),
        policy=policy.config,
        rename_map=RENAME_MAP,
        batch_size=len(rows),
    )
    dataset = make_dataset(cfg)
    subset = Subset(dataset, rows)
    loader = DataLoader(subset, batch_size=len(rows), shuffle=False)
    raw_batch = next(iter(loader))
    for cam_key in dataset.meta.camera_keys:
        if cam_key in raw_batch and raw_batch[cam_key].dtype == torch.uint8:
            raw_batch[cam_key] = raw_batch[cam_key].to(dtype=torch.float32) / 255.0
    return rename_batch_keys(raw_batch, RENAME_MAP)


def current_state(raw_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Same squeeze `SafeDiffVLAPolicy._current_state` applies: the dataset's own `observation.state`
    carries a length-1 leading time dimension (`observation_delta_indices=[0]`)."""
    state = raw_batch[OBS_STATE]
    return state[:, 0] if state.ndim > 2 else state


def error_stats(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    diff = pred - gt  # [N, 3], physical meters
    l2 = diff.norm(dim=-1)
    return {
        "n": int(pred.shape[0]),
        "mean_l2_m": l2.mean().item(),
        "median_l2_m": l2.median().item(),
        "p90_l2_m": l2.quantile(0.9).item(),
        "max_l2_m": l2.max().item(),
        "mean_abs_dx": diff[:, 0].abs().mean().item(),
        "mean_abs_dy": diff[:, 1].abs().mean().item(),
        "mean_abs_dz": diff[:, 2].abs().mean().item(),
        "mean_signed_dx": diff[:, 0].mean().item(),
        "mean_signed_dy": diff[:, 1].mean().item(),
        "mean_signed_dz": diff[:, 2].mean().item(),
    }


def run_validation(policy: SafeDiffVLAPolicy, device: str, n_episodes: int, chunk_size: int) -> dict:
    episodes = select_candidate_episodes(n_episodes)
    grasp_info = find_grasp_frames(episodes)
    episodes = [e for e in episodes if grasp_info[e]["grasp_frame"] is not None]
    row_df = build_dense_pregrasp_rows(episodes, grasp_info, MAX_OFFSET)
    logger.info("=== %d valid pre-grasp samples across %d episodes ===", len(row_df), len(episodes))

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    all_pred_phys, all_gt_phys, all_ee_phys = [], [], []
    rows = row_df["row"].tolist()
    k_all = row_df["k_to_transition"].tolist()
    for start in range(0, len(rows), chunk_size):
        chunk_rows = rows[start : start + chunk_size]
        chunk_k = k_all[start : start + chunk_size]
        raw_batch = fetch_raw_batch(policy, episodes, chunk_rows)
        state = current_state(raw_batch)
        gt_target_raw = gt_target_from_action_window(raw_batch, chunk_k)
        ee_xyz_raw = state[:, :3]

        batch = preprocessor(raw_batch)
        with torch.no_grad():
            _, metrics = policy.plan_action_chunk(batch)
        pred_norm = metrics["predicted_target_xyz"].detach().cpu()
        pred_phys = pred_norm * policy.action_pos_std.cpu() + policy.action_pos_mean.cpu()

        all_pred_phys.append(pred_phys)
        all_gt_phys.append(gt_target_raw)
        all_ee_phys.append(ee_xyz_raw)
        logger.info("  processed %d/%d", min(start + chunk_size, len(rows)), len(rows))

    pred_phys = torch.cat(all_pred_phys)
    gt_phys = torch.cat(all_gt_phys)
    ee_phys = torch.cat(all_ee_phys)
    mean_grasp_xyz = gt_phys.mean(dim=0, keepdim=True)

    return {
        "n_samples": int(pred_phys.shape[0]),
        "target_point_head": error_stats(pred_phys, gt_phys),
        "trivial_baseline_current_ee_xyz": error_stats(ee_phys, gt_phys),
        "trivial_baseline_dataset_mean_grasp_xyz": error_stats(mean_grasp_xyz.expand_as(gt_phys), gt_phys),
        "dataset_mean_grasp_xyz_value": mean_grasp_xyz.squeeze(0).tolist(),
    }


def run_tiny_overfit_test(policy_source: str, device: str, n_samples: int, n_steps: int, lr: float) -> dict:
    policy = SafeDiffVLAPolicy.from_pretrained(policy_source).to(device)
    policy.eval()  # backbone frozen + eval (no dropout noise) for a deterministic cached latent
    fresh_head = TargetPointHead(policy._multimodal_latent_dim(), policy.config.target_head_hidden_dim).to(device)
    policy.target_point_head = fresh_head  # re-init from scratch: tests representational capacity, not 5k's own partial fit

    episodes = select_candidate_episodes(10)
    grasp_info = find_grasp_frames(episodes)
    episodes = [e for e in episodes if grasp_info[e]["grasp_frame"] is not None]
    row_df = build_dense_pregrasp_rows(episodes, grasp_info, MAX_OFFSET).head(n_samples)
    assert len(row_df) == n_samples, f"only found {len(row_df)} valid rows, wanted {n_samples}"

    raw_batch = fetch_raw_batch(policy, episodes, row_df["row"].tolist())
    gt_target_raw = gt_target_from_action_window(raw_batch, row_df["k_to_transition"].tolist())

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_source,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    batch = preprocessor(raw_batch)
    gt_target_raw = gt_target_raw.to(device)
    gt_target_norm = (gt_target_raw - policy.action_pos_mean) / policy.action_pos_std

    with torch.no_grad():
        latent_tokens, latent_pad_mask = policy._encode_multimodal_latent(batch)
        pooled_latent = policy._pooled_latent(latent_tokens, latent_pad_mask).detach()

    optimizer = torch.optim.Adam(policy.target_point_head.parameters(), lr=lr)
    history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        pred_norm = policy.target_point_head(pooled_latent)
        loss = torch.nn.functional.mse_loss(pred_norm, gt_target_norm)
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == n_steps - 1:
            with torch.no_grad():
                pred_phys = pred_norm.detach() * policy.action_pos_std + policy.action_pos_mean
                err = (pred_phys - gt_target_raw).norm(dim=-1)
                history.append(
                    {
                        "step": step,
                        "loss_target_normalized_mse": loss.item(),
                        "mean_l2_error_m": err.mean().item(),
                        "median_l2_error_m": err.median().item(),
                        "max_l2_error_m": err.max().item(),
                    }
                )
            logger.info("  overfit step %d: loss=%.5f mean_l2_m=%.4f max_l2_m=%.4f", step, loss.item(), history[-1]["mean_l2_error_m"], history[-1]["max_l2_error_m"])

    return {"n_samples": n_samples, "n_steps": n_steps, "lr": lr, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-episodes", type=int, default=60, help="candidate select_poker episodes for validation")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overfit-n-samples", type=int, default=64)
    parser.add_argument("--overfit-n-steps", type=int, default=500)
    parser.add_argument("--overfit-lr", type=float, default=1e-3)
    parser.add_argument("--output", default="outputs/eval/safediff_vla_target_point_head_diagnosis/summary.json")
    args = parser.parse_args()

    logger.info("=== 1+2. held-out-ish target-point validation + trivial baselines ===")
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint).to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder_grounded_grasp"
    validation = run_validation(policy, args.device, args.n_episodes, args.chunk_size)
    logger.info("target_point_head: %s", json.dumps(validation["target_point_head"], indent=2))
    logger.info("baseline (current EE xyz): %s", json.dumps(validation["trivial_baseline_current_ee_xyz"], indent=2))
    logger.info("baseline (dataset-mean grasp xyz): %s", json.dumps(validation["trivial_baseline_dataset_mean_grasp_xyz"], indent=2))

    logger.info("=== 3. training-step learning curve: unavailable (save_freq=5000, no 1k/2k checkpoints saved) ===")

    logger.info("=== 4. tiny-set overfit capacity test (n=%d, steps=%d) ===", args.overfit_n_samples, args.overfit_n_steps)
    overfit = run_tiny_overfit_test(args.checkpoint, args.device, args.overfit_n_samples, args.overfit_n_steps, args.overfit_lr)

    summary = {
        "checkpoint": args.checkpoint,
        "task": TASK,
        "calibration_notes": [
            "minor: grounded_grasp_gripper_open_threshold=0.5 is applied during actual training to "
            "the MEAN_STD-normalized gripper channel, not raw physical (raw gripper mean=0.4574, "
            "std=0.4982 per this checkpoint's saved stats -- normalized 0.5 ~= raw 0.706). "
            "Plausible few-frame label-timing noise, unlikely to explain multi-cm errors alone.",
            "SIGNIFICANT: observation.state[6] (gripper) uses the OPPOSITE sign convention from "
            "action[6] (verified directly against the raw dataset, multiple episodes: action "
            "1.0=open/0.0=closed, state 0.0=open/1.0=closed, state lagging action by one frame). "
            "utils.find_grasp_target's pre-grasp gate assumes the same convention for both, so "
            "during actual 5k training it likely masked out most genuine pre-grasp samples and let "
            "through an uncontrolled, convention-inverted subset instead -- corrupting loss_target's "
            "effective supervision independent of TargetPointHead's own capacity or training "
            "duration. This diagnostic bypasses that bug entirely (GT extracted from the "
            "already-correct action-only transition, never from observation.state) so the numbers "
            "below are against the true intended label. NOT fixed here (out of scope).",
        ],
        "held_out_validation": validation,
        "training_step_learning_curve": "unavailable: only a single (005000) checkpoint was saved (save_freq=5000); would require a new training run, out of scope for this diagnostic",
        "tiny_overfit_capacity_test": overfit,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("=== wrote %s ===", out_path)


if __name__ == "__main__":
    main()
