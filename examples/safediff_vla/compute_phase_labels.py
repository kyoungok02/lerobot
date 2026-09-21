#!/usr/bin/env python
"""Derive a per-frame binary "task phase" label for `lerobot/vlabench_unified`, for
`TemporalActionDecoder`'s minimal `use_phase_conditioning` (see `configuration_safediff_vla.py`).

phase = 0 for every frame strictly *before* an episode's first demonstrated grasp (gripper
open->close transition), 1 for every frame at or after it (an episode that never grasps is all
0). Computed causally, from a strict prefix of the episode's own actions (`frame_index < t`) --
never from the future action chunk that frame `t` is itself a training target for, so training
never leaks the label through its own target.

Same zero-cost source as `compute_subgoal_labels.py`: every episode's own `action` gripper
channel already tells us exactly when the demonstrated grasp happens; no simulator access, no
seeds, no new data collection.

Usage:
    uv run python examples/safediff_vla/compute_phase_labels.py
    uv run python examples/safediff_vla/compute_phase_labels.py --out outputs/data/vlabench_phase_labels/labels.parquet

Output columns: `index` (global row index, matches the dataset's own `index` feature),
`episode_index`, `frame_index`, `phase` (int, 0 or 1), `grasp_frame_index` (int or -1,
diagnostic only -- the episode's first open->close frame, or -1 if it never grasps).
"""

import argparse
import os

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "lerobot/vlabench_unified"
GRIPPER_CHANNEL = 6
# Same convention/threshold as `compute_subgoal_labels.py`: demonstrated (near-binary) actions
# don't need the hysteresis band `execution.ActionExecutor` uses for live (noisier) predictions.
GRIPPER_OPEN_THRESHOLD = 0.5


def find_phase_labels(gripper: np.ndarray) -> tuple[np.ndarray, int]:
    """For an episode's `[T]` gripper-channel action stream, return (`phase` `[T]` int array,
    `grasp_frame_index` or -1). `phase[t] = 1` iff some strictly earlier frame (`< t`) was the
    first frame where the gripper reads closed right after reading open."""
    is_open = gripper > GRIPPER_OPEN_THRESHOLD
    n = len(gripper)
    # `closing[j]` True means action j-1 was open and action j is closed -- frame j is the first
    # closed frame of that transition.
    closing = np.flatnonzero(is_open[:-1] & ~is_open[1:]) + 1
    if len(closing) == 0:
        return np.zeros(n, dtype=np.int64), -1
    grasp_frame_index = int(closing[0])
    phase = (np.arange(n) > grasp_frame_index).astype(np.int64)
    return phase, grasp_frame_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=str, default="outputs/data/vlabench_phase_labels/labels.parquet")
    args = parser.parse_args()

    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    data_files = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    print(f"Found {len(data_files)} data files: {data_files}")

    frames = []
    for f in data_files:
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f)
        frames.append(pd.read_parquet(path, columns=["index", "episode_index", "frame_index", "action"]))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} frames across {df['episode_index'].nunique()} episodes")

    out_rows = []
    n_never_grasps = 0
    for episode_index, group in df.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        actions = np.stack(group["action"].to_numpy())
        gripper = actions[:, GRIPPER_CHANNEL]
        phase, grasp_frame_index = find_phase_labels(gripper)
        if grasp_frame_index < 0:
            n_never_grasps += 1
        frame_indices = group["frame_index"].to_numpy()
        global_indices = group["index"].to_numpy()
        for row_pos in range(len(group)):
            out_rows.append(
                {
                    "index": int(global_indices[row_pos]),
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_indices[row_pos]),
                    "phase": int(phase[row_pos]),
                    "grasp_frame_index": grasp_frame_index,
                }
            )

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(f"{n_never_grasps} / {df['episode_index'].nunique()} episodes never grasp (all-phase-0)")
    print(out_df["phase"].value_counts())


if __name__ == "__main__":
    main()
