#!/usr/bin/env python
"""Measurement-only open-loop (teacher-forced, in-distribution) grasp-approach precision check,
complementing `grasp_approach_precision_closedloop.py`'s live-rollout forensics.

Purpose: separate "target localization bias" from "execution drift" / "regression averaging"
by asking a question the closed-loop rollout *cannot* answer on its own -- given the exact real
image/state the demonstrator saw at a frame near a real recorded grasp event, how far off is the
policy's single-shot predicted action chunk from the demonstrator's *actual* recorded action
chunk, at that same frame? This has no simulator, no IK controller, no execution/queueing/replan
logic in the loop at all -- `policy.plan_action_chunk` is called once per sampled frame on a
real (`lerobot/vlabench_unified`) `select_poker` frame, teacher-forced (real image + real current
state), and compared to the dataset's own recorded future action window. A large, systematic
error here -- especially one that's already large well before the close event and doesn't shrink
as the close event approaches -- points at target localization bias in the model itself, not at
simulator/execution dynamics (which never enter this script at all).

Action-space note (see `grasp_approach_precision_closedloop.py`'s docstring / `VLABenchEnv.
_build_ctrl_from_action`): `action[:3]` is an ABSOLUTE end-effector target position in robot-base
frame -- both the model's predicted action and the dataset's ground-truth action are already in
this same frame/units, so a plain per-step L2 on `[..., :3]` is directly a target-position error,
no unit conversion needed.

No model, loss, execution-policy, or gripper-logic code is touched -- this only calls
`policy.plan_action_chunk` (already the pure "given this observation, output a plan" API used by
`eval_padfix_reproducibility.py`'s own open-loop check) and reads dataset frames.

Outputs:
    outputs/eval/safediff_vla_grasp_approach_precision/openloop_report.json

Usage:
    uv run python examples/safediff_vla/grasp_approach_precision_openloop.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.processor.rename_processor import rename_batch_keys
from lerobot.utils.constants import ACTION

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_sincos_20k_padfix/checkpoints/020000/pretrained_model"
DATASET_REPO_ID = "lerobot/vlabench_unified"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_CHANNEL = 6
GRIPPER_OPEN_THRESHOLD = 0.5
OFFSETS = list(range(-30, 11, 2))  # relative to the demonstrated grasp (open->close) frame
FORWARD_BATCH = 32


def select_candidate_episodes(n_episodes: int) -> list[int]:
    meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
    eps_df = meta.episodes.to_pandas()
    eps_df["task0"] = eps_df["tasks"].apply(lambda t: t[0])
    poker_eps = sorted(
        eps_df.loc[eps_df["task0"].str.contains("poker", case=False, na=False), "episode_index"].tolist()
    )
    return sorted(poker_eps[:: max(1, len(poker_eps) // n_episodes)][:n_episodes])


def find_grasp_frames(episodes: list[int]) -> dict[int, dict]:
    """For each episode, the first frame index where the recorded action's gripper channel
    crosses open(>threshold) -> closed(<=threshold) -- the demonstrated grasp event -- plus the
    episode's length, from a bare (no delta-window) LeRobotDataset load."""
    bare = LeRobotDataset(DATASET_REPO_ID, episodes=episodes)
    raw = bare.select_columns(["action", "episode_index", "frame_index"]).to_pandas()
    out: dict[int, dict] = {}
    for ep, group in raw.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        actions = np.stack(group["action"].to_numpy())
        gripper = actions[:, GRIPPER_CHANNEL]
        is_open = gripper > GRIPPER_OPEN_THRESHOLD
        transitions = np.flatnonzero(is_open[:-1] & ~is_open[1:]) + 1  # first open->close frame
        length = len(group)
        out[int(ep)] = {
            "length": length,
            "grasp_frame": int(transitions[0]) if len(transitions) else None,
        }
    return out


def build_target_rows(episodes: list[int], grasp_info: dict[int, dict]) -> pd.DataFrame:
    """Row offsets (0-based, into the episode-filtered dataset built with `episodes=episodes`,
    ascending episode order -- verified empirically against `LeRobotDataset`'s own filtering) for
    every (episode, relative_offset) target frame, clipped to the episode's own bounds."""
    rows = []
    cursor = 0
    for ep in episodes:  # `episodes` must already be sorted ascending
        info = grasp_info[ep]
        length = info["length"]
        grasp_frame = info["grasp_frame"]
        if grasp_frame is not None:
            seen_frames = set()
            for off in OFFSETS:
                frame = int(np.clip(grasp_frame + off, 0, length - 1))
                if frame in seen_frames:
                    continue
                seen_frames.add(frame)
                rows.append(
                    {
                        "episode_index": ep,
                        "frame_index": frame,
                        "relative_offset": frame - grasp_frame,  # post-clip, may differ from `off`
                        "requested_offset": off,
                        "row": cursor + frame,
                    }
                )
        cursor += length
    return pd.DataFrame(rows)


def run(policy: SafeDiffVLAPolicy, checkpoint: str, device: str, episodes: list[int], row_df: pd.DataFrame) -> dict:
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET_REPO_ID, episodes=episodes),
        policy=policy.config,
        rename_map=RENAME_MAP,
        batch_size=FORWARD_BATCH,
    )
    dataset = make_dataset(cfg)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    subset = Subset(dataset, row_df["row"].tolist())
    loader = DataLoader(subset, batch_size=FORWARD_BATCH, shuffle=False)

    per_row_records = []
    row_cursor = 0
    for raw_batch in loader:
        bsz = raw_batch[ACTION].shape[0]
        for cam_key in dataset.meta.camera_keys:
            if cam_key in raw_batch and raw_batch[cam_key].dtype == torch.uint8:
                raw_batch[cam_key] = raw_batch[cam_key].to(dtype=torch.float32) / 255.0
        raw_batch = rename_batch_keys(raw_batch, RENAME_MAP)
        gt_actions_raw = raw_batch[ACTION].clone()  # [b, 50, 7], dataset-native scale
        action_is_pad = raw_batch.get("action_is_pad")  # [b, 50] True=padded/repeated, exclude from means

        batch = preprocessor(raw_batch)
        with torch.no_grad():
            pred_actions_norm, _ = policy.plan_action_chunk(batch)
        pred_actions_raw = postprocessor(pred_actions_norm.cpu()).to(gt_actions_raw.dtype)

        pred_pos = pred_actions_raw[..., :3]
        gt_pos = gt_actions_raw[..., :3]
        position_error_per_bh = (pred_pos - gt_pos).norm(dim=-1)  # [b, 50]
        gripper_abs_error_per_bh = (pred_actions_raw[..., 6] - gt_actions_raw[..., 6]).abs()

        valid = ~action_is_pad if action_is_pad is not None else torch.ones_like(position_error_per_bh, dtype=torch.bool)

        def masked_mean(t: torch.Tensor, n: int, row: int, row_valid: torch.Tensor) -> float | None:
            sel = t[row, :n][row_valid[:n]]
            return sel.mean().item() if sel.numel() else None

        for i in range(bsz):
            meta_row = row_df.iloc[row_cursor + i]
            v = valid[i]

            per_row_records.append(
                {
                    "episode_index": int(meta_row["episode_index"]),
                    "frame_index": int(meta_row["frame_index"]),
                    "relative_offset": int(meta_row["relative_offset"]),
                    "position_error_first5": masked_mean(position_error_per_bh, 5, i, v),
                    "position_error_first10": masked_mean(position_error_per_bh, 10, i, v),
                    "gripper_abs_error_first5": masked_mean(gripper_abs_error_per_bh, 5, i, v),
                    "predicted_first_target_xyz": pred_pos[i, 0].tolist(),
                    "gt_first_target_xyz": gt_pos[i, 0].tolist(),
                    "n_valid_horizon_steps": int(v.sum().item()),
                }
            )
        row_cursor += bsz

    # Aggregate by relative_offset (aligned window, matching the closed-loop script's
    # pm-steps-around-close alignment convention).
    df = pd.DataFrame(per_row_records)
    by_offset = []
    for off, group in df.groupby("relative_offset", sort=True):
        by_offset.append(
            {
                "relative_offset": int(off),
                "n_frames": len(group),
                "mean_position_error_first5": group["position_error_first5"].mean(),
                "mean_position_error_first10": group["position_error_first10"].mean(),
                "mean_gripper_abs_error_first5": group["gripper_abs_error_first5"].mean(),
            }
        )
    by_offset.sort(key=lambda r: r["relative_offset"])

    near_close = df[df["relative_offset"].between(-3, 3)]
    far_before = df[df["relative_offset"].between(-30, -20)]

    return {
        "checkpoint": checkpoint,
        "dataset": DATASET_REPO_ID,
        "task": TASK,
        "n_episodes": len(episodes),
        "episodes": episodes,
        "offsets_requested": OFFSETS,
        "per_frame": per_row_records,
        "aligned_by_relative_offset": by_offset,
        "near_close_window_pm3": {
            "n_frames": len(near_close),
            "mean_position_error_first5": near_close["position_error_first5"].mean() if len(near_close) else None,
            "mean_position_error_first10": near_close["position_error_first10"].mean() if len(near_close) else None,
        },
        "far_before_close_window_20_to_30": {
            "n_frames": len(far_before),
            "mean_position_error_first5": far_before["position_error_first5"].mean() if len(far_before) else None,
            "mean_position_error_first10": far_before["position_error_first10"].mean() if len(far_before) else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/safediff_vla_grasp_approach_precision/openloop_report.json")
    args = parser.parse_args()
    checkpoint = args.checkpoint

    logger.info("=== loading canonical checkpoint from %s (unmodified) ===", checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(checkpoint)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"
    assert policy.config.action_horizon == 50

    episodes = select_candidate_episodes(args.n_episodes)
    logger.info("=== candidate select_poker episodes (%d): %s ===", len(episodes), episodes)

    grasp_info = find_grasp_frames(episodes)
    n_with_grasp = sum(1 for v in grasp_info.values() if v["grasp_frame"] is not None)
    logger.info("=== found a demonstrated grasp frame in %d/%d episodes ===", n_with_grasp, len(episodes))
    episodes = [e for e in episodes if grasp_info[e]["grasp_frame"] is not None]

    row_df = build_target_rows(episodes, grasp_info)
    logger.info("=== %d target frames across %d episodes, %d unique relative offsets ===", len(row_df), len(episodes), row_df["relative_offset"].nunique())

    report = run(policy, checkpoint, args.device, episodes, row_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float))
    logger.info("=== wrote %s ===", out_path)
    for r in report["aligned_by_relative_offset"]:
        logger.info("  offset=%+3d n=%3d pos_err_first5=%.4f pos_err_first10=%.4f", r["relative_offset"], r["n_frames"], r["mean_position_error_first5"], r["mean_position_error_first10"])


if __name__ == "__main__":
    main()
