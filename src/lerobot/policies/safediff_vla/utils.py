import torch
from torch import Tensor


def pad_or_crop_horizon(actions: Tensor, horizon: int) -> Tensor:
    if actions.shape[1] >= horizon:
        return actions[:, :horizon]
    padding = actions[:, -1:].expand(-1, horizon - actions.shape[1], -1)
    return torch.cat((actions, padding), dim=1)


def pad_or_crop_mask(mask: Tensor, horizon: int) -> Tensor:
    """Crop/pad a `[B, T]` bool mask to `horizon` steps, mirroring `pad_or_crop_horizon`'s handling
    of the action tensor it accompanies. Any step added by padding (T < horizon) is marked `True`
    (excluded) rather than copying the last real step's mask value, since that last step's own
    validity says nothing about a position that doesn't exist in the original chunk.
    """
    if mask.shape[1] >= horizon:
        return mask[:, :horizon]
    extra = torch.ones(mask.shape[0], horizon - mask.shape[1], dtype=torch.bool, device=mask.device)
    return torch.cat((mask, extra), dim=1)


def _action_gripper_is_open(values: Tensor, threshold: float) -> Tensor:
    """`action`'s gripper channel convention: 1.0=open, 0.0=closed -- open means strictly ABOVE
    `threshold` (matches every other `action`-gripper convention in this codebase, e.g.
    `examples/safediff_vla/place_phase_forensics.py`'s `GRIPPER_THRESHOLD`)."""
    return values > threshold


def _state_gripper_is_open(values: Tensor, threshold: float) -> Tensor:
    """`observation.state`'s gripper channel convention: 0.0=open, 1.0=closed -- the OPPOSITE of
    `action`'s own convention (`_action_gripper_is_open`) -- open means strictly BELOW `threshold`.

    Verified directly against `lerobot/vlabench_unified`'s raw dataset across multiple episodes:
    `observation.state[..., gripper_index]` transitions open(0.0)->closed(1.0) exactly one frame
    AFTER `action[..., gripper_index]` transitions open(1.0)->closed(0.0) -- state reports the
    mechanical result of the *previous* commanded action, with the inverted sign. Never assume
    `observation.state` and `action` share a polarity for this channel.
    """
    return values < threshold


def find_grasp_target(
    current_state: Tensor,
    action: Tensor,
    gripper_index: int = 6,
    gripper_open_threshold: float = 0.5,
    action_is_pad: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """For `architecture="temporal_decoder_grounded_grasp"`'s `TargetPointHead` supervision: find
    each sample's first open->close gripper transition within its future `action` window, and
    return the commanded xyz at that transition step as the grasp-target regression label.

    A transition is checked across the FULL sequence `[current_state's own gripper value, then
    action[:, 0, gripper], action[:, 1, gripper], ...]` -- so a transition landing exactly at
    `action[:, 0]` (current state open, first predicted step already closed) is found correctly,
    not missed by only ever comparing adjacent *future* steps. `current_state` and `action` are
    read with their OWN, DIFFERENT gripper polarities (`_state_gripper_is_open` /
    `_action_gripper_is_open` respectively) -- see those functions' docstrings; this is not a
    single shared convention.

    A sample is only valid (see `valid_mask`) when ALL of:
      - it is itself "pre-grasp" (`current_state`'s own gripper reads open, `_state_gripper_is_open`
        convention) -- a window that starts already closed (mid-grasp/holding, i.e. post-grasp) has
        no fresh grasp event to ground, by definition, and is therefore always invalid;
      - an open->close transition actually exists somewhere in the `action` window (`_action_
        gripper_is_open` convention) -- a window that stays open throughout (nothing to reach for
        yet) or is too short to contain the real transition has no valid target either, with no
        fallback to e.g. the window's last frame; and
      - if `action_is_pad` is given, that transition does not land on a padded (repeated-last-real-
        frame) position -- a padding artifact is never a real demonstrated grasp event.

    Args:
        current_state: `[B, state_dim]`, `state_dim >= gripper_index + 1`. Must already be in RAW
            physical units for `gripper_index` -- MEAN_STD-normalized values silently miscalibrate
            `gripper_open_threshold` (see `SafeDiffVLAPolicy._degrip_for_transition_detection`,
            which un-normalizes exactly this channel at the policy's own boundary before calling
            this function; xyz/rotation may stay normalized, only the gripper index matters here
            for the open/closed decision, though `action[..., :3]` is returned as-is/whatever
            scale it was passed in).
        action: `[B, H, action_dim]`, `action_dim >= gripper_index + 1` (raw/un-encoded 7-D
            layout). Same RAW-physical-gripper requirement as `current_state`.
        gripper_index: gripper's index in the last dim (6 for the LIBERO-safety 7-D layout: xyz,
            rx, ry, rz, gripper -- see `SafeDiffVLAConfig.validate_features`).
        gripper_open_threshold: the raw-physical-scale open/closed decision boundary (matches the
            convention used throughout this codebase, e.g.
            `examples/safediff_vla/place_phase_forensics.py`'s `GRIPPER_THRESHOLD`).
        action_is_pad: optional `[B, H]` bool, `True` where that position in the `action` window is
            a padded repeat of the last real frame (see `DatasetReader._get_query_indices`), not a
            real future action -- e.g. `utils.pad_or_crop_mask(batch["action_is_pad"], action_horizon)`.
            Positions marked `True` are never selected as "the" transition.

    Returns:
        target_xyz: `[B, 3]`, the commanded xyz (`action[..., :3]`) at each sample's first
            open->close transition step (meaningless where `valid_mask` is `False`).
        valid_mask: `[B]` bool, `True` where `target_xyz` is a real, well-defined label.
    """
    state_open_now = _state_gripper_is_open(current_state[..., gripper_index], gripper_open_threshold)  # [B]
    action_open = _action_gripper_is_open(action[..., gripper_index], gripper_open_threshold)  # [B, H]
    is_open_seq = torch.cat((state_open_now.unsqueeze(1), action_open), dim=1)  # [B, H+1]

    # `transitions[:, i]` True means seq position i was open and i+1 is closed -- i.e. `action[:, i]`
    # (index i in the H-wide action window) is the first closed step.
    transitions = is_open_seq[:, :-1] & ~is_open_seq[:, 1:]  # [B, H]
    if action_is_pad is not None:
        transitions = transitions & ~action_is_pad
    has_transition = transitions.any(dim=1)  # [B]

    # `argmax` on a bool/float tensor returns the index of the FIRST maximal (True/1) entry when
    # there are ties -- exactly "first transition" here. Rows with no transition get index 0
    # (argmax's fallback on an all-False row), which is fine: `valid_mask` excludes them below, so
    # the corresponding `target_xyz` value is never actually used.
    first_transition_idx = transitions.to(torch.float32).argmax(dim=1)  # [B]

    valid_mask = has_transition & state_open_now
    batch_idx = torch.arange(action.shape[0], device=action.device)
    target_xyz = action[batch_idx, first_transition_idx, :3]
    return target_xyz, valid_mask


def masked_mse(pred: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Mean squared error over only the timesteps where `valid_mask` (`[B, T]`, True = include)
    holds, broadcast across `pred`/`target`'s trailing dim `[B, T, D]`. Exactly reproduces
    `F.mse_loss(pred, target)` (both default to a mean over every element) when `valid_mask` is all
    `True`. The denominator is clamped to at least 1 so an all-`False` mask returns a safe `0`
    instead of `0/0` -- the numerator is already exactly `0` in that case since every term is
    masked out, so the clamp only avoids the division-by-zero, it doesn't change the result.
    """
    sq_err = (pred - target).square()
    mask = valid_mask.unsqueeze(-1).to(sq_err.dtype)
    numerator = (sq_err * mask).sum()
    denominator = (mask.sum() * sq_err.shape[-1]).clamp_min(1.0)
    return numerator / denominator
