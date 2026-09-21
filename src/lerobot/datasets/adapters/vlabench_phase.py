"""Wraps a `LeRobotDataset`(-like) instance with a per-frame binary `observation.phase` label,
precomputed offline by `examples/safediff_vla/compute_phase_labels.py`.

Why this exists: `TemporalActionDecoder`'s minimal phase conditioning (see
`configuration_safediff_vla.py`'s `use_phase_conditioning`) needs a real, causal training label --
0 (pre-grasp) for every frame *before* an episode's first demonstrated grasp (gripper open->close),
1 (post-grasp/transport) for every frame at or after it. Computed strictly from each frame's own
past (frames with a smaller `frame_index` in the same episode), never from the future action chunk
that frame is a training target for -- the model must not see its own label leak in through the
target it's learning to predict.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch


class PhaseLabelDataset:
    """Attaches a precomputed `observation.phase` feature (scalar `long`, 0 or 1) to every
    returned frame.

    Composes with (rather than subclasses) the base dataset so it works for both `LeRobotDataset`
    and `StreamingLeRobotDataset` alike -- same design as `vlabench_subgoal.py`'s
    `SubgoalLabelDataset`, including why labels are kept as plain `numpy` arrays rather than a
    dict of individual tensors (see that module's docstring: avoids exhausting `DataLoader`
    worker shared-memory file descriptors).
    """

    def __init__(self, base_dataset: Any, labels_path: str) -> None:
        self.base_dataset = base_dataset
        labels = pd.read_parquet(labels_path, columns=["index", "phase"]).sort_values("index")
        self._label_indices = labels["index"].to_numpy()
        self._label_values = labels["phase"].to_numpy().astype(np.int64)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _attach(self, frame: dict[str, Any]) -> dict[str, Any]:
        global_index = int(frame["index"])
        row = np.searchsorted(self._label_indices, global_index)
        if row >= len(self._label_indices) or self._label_indices[row] != global_index:
            raise KeyError(
                f"No precomputed phase label for dataset index {global_index}. Re-run "
                "examples/safediff_vla/compute_phase_labels.py so it covers this episode range."
            )
        frame["observation.phase"] = torch.tensor(int(self._label_values[row]), dtype=torch.long)
        return frame

    def __getitem__(self, idx: int | slice):
        item = self.base_dataset[idx]
        if isinstance(item, list):
            return [self._attach(frame) for frame in item]
        return self._attach(item)

    def __getattr__(self, name: str) -> Any:
        # See `vlabench_subgoal.py`'s `SubgoalLabelDataset.__getattr__` for why this guard is
        # needed (DataLoader worker pickling probes dunders before `__init__` has run).
        if name == "base_dataset" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.base_dataset, name)
