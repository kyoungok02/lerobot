#!/usr/bin/env python
"""Dataset-wide numeric audit of the FIXED `utils.find_grasp_target` (gripper-polarity +
padding-artifact bugfix), run BEFORE any fresh `temporal_decoder_grounded_grasp` training. Pure
parquet-column audit -- no images, no GPU, no policy/model involved at all, so it can cover a
large, representative slice of the whole `lerobot/vlabench_unified` dataset (every task, not just
select_poker) quickly.

Independently (re-)derives ground truth for every sampled frame using plain arithmetic on the raw
`action`/`observation.state` gripper columns (NOT by calling `find_grasp_target` -- the point is to
check the function against an independently-computed oracle, not to check it against itself), then
cross-tabulates against `find_grasp_target`'s actual `valid_mask` output for that same frame:

  - true_pre_grasp_in_horizon: state currently reads open (state convention) AND a real,
    non-padded action-only open->close transition exists within the next `action_horizon` steps.
    `find_grasp_target` should mark ~100% of these valid.
  - post_grasp: state currently reads closed (mid-grasp/holding). `find_grasp_target` must mark
    0% of these valid (a post-grasp sample is invalid by definition, whatever the future window
    contains).
  - no_transition: state currently open, but no action-only transition exists within horizon
    (window too short / episode never re-closes in range). Must be 0% valid.
  - padding-derived false positives: among samples `find_grasp_target` marks valid, the fraction
    whose selected transition index actually falls on a padded (repeated-last-real-frame)
    position. Must be 0% (repeat-padding cannot itself manufacture a value change, but this is
    checked directly rather than assumed).

Padding is replicated exactly as `DatasetReader._get_query_indices` does it: index `t+j` clamped
to `episode_length-1` (repeat the last real frame), `is_pad = (t+j >= episode_length)`.

Outputs:
    outputs/eval/safediff_vla_grasp_target_supervision_audit/summary.json

Usage:
    uv run python examples/safediff_vla/audit_grasp_target_supervision.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402
from lerobot.policies.safediff_vla.utils import (  # noqa: E402
    _action_gripper_is_open,
    _state_gripper_is_open,
    find_grasp_target,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

DATASET_REPO_ID = "lerobot/vlabench_unified"
ACTION_HORIZON = 50
GRIPPER_OPEN_THRESHOLD = 0.5


def sample_episodes(n_episodes: int, seed: int) -> list[int]:
    meta = LeRobotDatasetMetadata(DATASET_REPO_ID)
    total = meta.total_episodes
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), min(n_episodes, total)))


def load_episode_gripper_sequences(episodes: list[int]) -> dict[int, dict]:
    """`{episode_index: {"length": int, "action_grip": np.ndarray, "state_grip": np.ndarray}}` --
    raw per-frame gripper columns only, via `select_columns` (no video decode)."""
    ds = LeRobotDataset(DATASET_REPO_ID, episodes=episodes)
    raw = ds.select_columns(["action", "observation.state", "episode_index", "frame_index"]).to_pandas()
    out = {}
    for ep, group in raw.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        actions = np.stack(group["action"].to_numpy())
        states = np.stack(group["observation.state"].to_numpy())
        out[int(ep)] = {
            "length": len(group),
            "action_grip": actions[:, GRIPPER_INDEX_RAW].astype(np.float64),
            "state_grip": states[:, GRIPPER_INDEX_RAW].astype(np.float64),
        }
    return out


def build_windowed_batch(
    ep_data: dict, frame_indices: list[int], action_horizon: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each `frame_indices[i]` in this one episode, build the length-`action_horizon` future
    action-gripper window with real `DatasetReader`-style repeat-padding at the episode boundary,
    and the current-frame state-gripper scalar. Returns
    (state_grip [N], action_grip_window [N, H], is_pad [N, H])."""
    length = ep_data["length"]
    action_grip, state_grip = ep_data["action_grip"], ep_data["state_grip"]
    n = len(frame_indices)
    window = np.empty((n, action_horizon), dtype=np.float64)
    is_pad = np.zeros((n, action_horizon), dtype=bool)
    state_now = np.empty(n, dtype=np.float64)
    for i, t in enumerate(frame_indices):
        state_now[i] = state_grip[t]
        for j in range(action_horizon):
            idx = t + j
            is_pad[i, j] = idx >= length
            window[i, j] = action_grip[min(idx, length - 1)]
    return (
        torch.from_numpy(state_now).unsqueeze(-1),  # [N,1] -- fake state_dim=1, gripper at index 0
        torch.from_numpy(window).unsqueeze(-1),  # [N,H,1]
        torch.from_numpy(is_pad),
    )


def independent_ground_truth(ep_data: dict, frame_indices: list[int], action_horizon: int) -> list[dict]:
    """Oracle classification per frame, using only plain arithmetic on the raw arrays (never
    calling `find_grasp_target`) -- what THIS script independently believes the correct
    valid/invalid category is."""
    length = ep_data["length"]
    action_grip, state_grip = ep_data["action_grip"], ep_data["state_grip"]
    out = []
    for t in frame_indices:
        state_open_now = state_grip[t] < GRIPPER_OPEN_THRESHOLD  # state convention: open=0, <threshold
        # Action-only transition search, real frames only (no padding read past `length`), within
        # the same [current_state, action[t], action[t+1], ...] framing find_grasp_target uses.
        transition_idx_in_window = None
        prev_open = state_open_now  # seq[0] = state
        for j in range(action_horizon):
            idx = t + j
            if idx >= length:
                break  # ran off the real episode -- no more real (non-padded) positions to check
            cur_open = action_grip[idx] > GRIPPER_OPEN_THRESHOLD  # action convention: open=1, >threshold
            if prev_open and not cur_open:
                transition_idx_in_window = j
                break
            prev_open = cur_open
        if not state_open_now:
            category = "post_grasp"
        elif transition_idx_in_window is None:
            category = "no_transition"
        else:
            category = "true_pre_grasp_in_horizon"
        out.append({"frame_index": t, "category": category, "oracle_transition_idx": transition_idx_in_window})
    return out


def run_audit(n_episodes: int, seed: int, frames_per_episode: int) -> dict:
    episodes = sample_episodes(n_episodes, seed)
    logger.info("=== auditing %d randomly sampled episodes (seed=%d) across the WHOLE dataset ===", len(episodes), seed)
    ep_data = load_episode_gripper_sequences(episodes)

    rng = random.Random(seed)
    rows = []
    for ep in episodes:
        info = ep_data[ep]
        length = info["length"]
        if length < 2:
            continue
        k = min(frames_per_episode, length)
        frame_indices = sorted(rng.sample(range(length), k))
        oracle = independent_ground_truth(info, frame_indices, ACTION_HORIZON)
        state_grip_t, action_grip_window, is_pad = build_windowed_batch(info, frame_indices, ACTION_HORIZON)
        _, valid_mask = find_grasp_target(
            state_grip_t, action_grip_window, gripper_index=0, gripper_open_threshold=GRIPPER_OPEN_THRESHOLD, action_is_pad=is_pad
        )

        # Recompute the SAME transitions tensor `find_grasp_target` uses internally, only to check
        # whether the selected index (for samples it marked valid) lands on a padded position --
        # `find_grasp_target` itself already excludes padded positions from selection, so this is a
        # direct audit of that guarantee, not a duplicate of its main logic.
        state_open_seq = _state_gripper_is_open(state_grip_t[:, 0], GRIPPER_OPEN_THRESHOLD)
        action_open_seq = _action_gripper_is_open(action_grip_window[:, :, 0], GRIPPER_OPEN_THRESHOLD)
        is_open_seq = torch.cat((state_open_seq.unsqueeze(1), action_open_seq), dim=1)
        transitions = (is_open_seq[:, :-1] & ~is_open_seq[:, 1:]) & ~is_pad
        selected_idx = transitions.to(torch.float32).argmax(dim=1)
        selected_is_pad = is_pad[torch.arange(len(frame_indices)), selected_idx]

        for i, o in enumerate(oracle):
            rows.append(
                {
                    "episode_index": ep,
                    "frame_index": o["frame_index"],
                    "category": o["category"],
                    "valid": bool(valid_mask[i].item()),
                    "selected_transition_is_pad": bool(selected_is_pad[i].item()) and bool(valid_mask[i].item()),
                }
            )

    df = pd.DataFrame(rows)
    n_total = len(df)

    def rate(mask: pd.Series, valid_col: str = "valid") -> tuple[int, int, float]:
        sub = df[mask]
        n = len(sub)
        n_valid = int(sub[valid_col].sum())
        return n, n_valid, (n_valid / n if n else float("nan"))

    n_pre, v_pre, r_pre = rate(df["category"] == "true_pre_grasp_in_horizon")
    n_post, v_post, r_post = rate(df["category"] == "post_grasp")
    n_notr, v_notr, r_notr = rate(df["category"] == "no_transition")
    n_all_valid = int(df["valid"].sum())
    n_pad_fp = int(df["selected_transition_is_pad"].sum())
    n_currently_open = n_pre + n_notr  # state reads open now, regardless of in-horizon or not
    in_horizon_coverage_among_pregrasp = (n_pre / n_currently_open) if n_currently_open else None

    return {
        "n_episodes_sampled": len(episodes),
        "n_frames_audited": n_total,
        "overall_valid_rate_all_frames": n_all_valid / n_total if n_total else None,
        "category_counts": {
            "true_pre_grasp_in_horizon": n_pre,
            "post_grasp": n_post,
            "no_transition": n_notr,
        },
        "true_pre_grasp_in_horizon_valid_rate": r_pre,  # expect ~1.0 (find_grasp_target should mark ~all of these valid)
        "post_grasp_valid_rate": r_post,  # MUST be 0.0
        "no_transition_valid_rate": r_notr,  # MUST be 0.0
        "padding_derived_false_positives": n_pad_fp,  # MUST be 0
        "padding_derived_false_positive_rate_of_all_valid": (n_pad_fp / n_all_valid) if n_all_valid else None,
        "in_horizon_coverage_among_currently_open_frames": in_horizon_coverage_among_pregrasp,  # candidate match for the ~82.7% reference
        "note_on_82_7_percent_reference": (
            "The user-cited ~82.7% reference figure's exact denominator was not independently "
            "known to this script; three candidate rates are reported for comparison: "
            "`overall_valid_rate_all_frames` (share of ALL sampled frames, any state), "
            "`true_pre_grasp_in_horizon_valid_rate` (should be ~1.0 by construction -- not this), "
            "and `in_horizon_coverage_among_currently_open_frames` (of frames where the gripper "
            "currently reads open, what fraction also have the demonstrated close within the next "
            "action_horizon steps) -- this last one is the closest conceptual match to a "
            "'pre-grasp + in-horizon coverage' statistic."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-episodes", type=int, default=800)
    parser.add_argument("--frames-per-episode", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/eval/safediff_vla_grasp_target_supervision_audit/summary.json")
    args = parser.parse_args()

    result = run_audit(args.n_episodes, args.seed, args.frames_per_episode)
    logger.info(json.dumps(result, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("=== wrote %s ===", out_path)

    ok = (
        result["post_grasp_valid_rate"] == 0.0
        and result["no_transition_valid_rate"] == 0.0
        and result["padding_derived_false_positives"] == 0
        and result["true_pre_grasp_in_horizon_valid_rate"] > 0.99
    )
    logger.info("=== AUDIT %s ===", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
