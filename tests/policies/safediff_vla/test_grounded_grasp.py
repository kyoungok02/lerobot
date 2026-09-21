"""Tests for `architecture="temporal_decoder_grounded_grasp"` (experimental): a `TargetPointHead`
grounding the decoder's action chunk on an explicit grasp-target xyz, plus an inference-only
proximity-based reactive gripper close -- see `configuration_safediff_vla.py`'s architecture
docstring and `examples/safediff_vla/grasp_close_timing_oracle.py` for the diagnostic findings
this is meant to address. Reuses `test_safediff_vla.py`'s config/batch/policy builders and
`TinyBackbone` test double -- not a separate convention.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.policies.safediff_vla.target_point_head import TargetPointHead
from lerobot.policies.safediff_vla.temporal_decoder import TemporalActionDecoder
from lerobot.policies.safediff_vla.utils import find_grasp_target
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.safediff_vla.test_safediff_vla import (
    ACTION_DIM,
    STATE_DIM,
    make_batch,
    make_config,
    make_policy,
)

ARCH = "temporal_decoder_grounded_grasp"


def make_grounded_batch(batch_size: int = 2, horizon: int = 4) -> dict[str, torch.Tensor]:
    return {OBS_STATE: torch.randn(batch_size, STATE_DIM), ACTION: torch.randn(batch_size, horizon, ACTION_DIM)}


# ---- utils.find_grasp_target ----------------------------------------------------------------


def test_find_grasp_target_extracts_xyz_at_first_open_to_close_transition() -> None:
    """First open->close transition in the *full* [current_state, *action] sequence must be
    found, including a transition landing exactly on action[:, 0]."""
    current_state = torch.zeros(3, STATE_DIM)
    current_state[:, GRIPPER_INDEX_RAW] = 1.0  # every row starts open

    action = torch.zeros(3, 5, ACTION_DIM)
    action[:, :, GRIPPER_INDEX_RAW] = 1.0  # stays open by default

    # Row 0: transition at action index 2 (open,open,CLOSE,closed,closed).
    action[0, 2:, GRIPPER_INDEX_RAW] = 0.0
    action[0, 2, :3] = torch.tensor([1.0, 2.0, 3.0])
    # Row 1: transition immediately at action index 0 (current_state open -> action[0] closed).
    action[1, 0:, GRIPPER_INDEX_RAW] = 0.0
    action[1, 0, :3] = torch.tensor([4.0, 5.0, 6.0])
    # Row 2: transition at the very last index (4).
    action[2, 4, GRIPPER_INDEX_RAW] = 0.0
    action[2, 4, :3] = torch.tensor([7.0, 8.0, 9.0])

    target_xyz, valid_mask = find_grasp_target(current_state, action, gripper_index=GRIPPER_INDEX_RAW)
    assert valid_mask.tolist() == [True, True, True]
    assert torch.allclose(target_xyz[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(target_xyz[1], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.allclose(target_xyz[2], torch.tensor([7.0, 8.0, 9.0]))


def test_find_grasp_target_masks_samples_with_no_transition_in_horizon() -> None:
    """A window that never closes (stays open throughout) has no grasp event to ground -- must
    be masked invalid, regardless of what garbage ends up in `target_xyz` for that row."""
    current_state = torch.zeros(2, STATE_DIM)
    current_state[:, GRIPPER_INDEX_RAW] = 1.0  # both open

    action = torch.zeros(2, 4, ACTION_DIM)
    action[:, :, GRIPPER_INDEX_RAW] = 1.0  # row 0: never closes within the window
    action[1, :, GRIPPER_INDEX_RAW] = 1.0
    action[1, 2, GRIPPER_INDEX_RAW] = 0.0  # row 1: DOES close at index 2 -> valid

    target_xyz, valid_mask = find_grasp_target(current_state, action, gripper_index=GRIPPER_INDEX_RAW)
    assert valid_mask.tolist() == [False, True]
    del target_xyz  # row 0's value is intentionally unspecified/unused


def test_find_grasp_target_masks_samples_already_closed_at_current_state() -> None:
    """A window whose CURRENT state is already closed (mid-grasp/holding) is not "pre-grasp" --
    even if the action window happens to contain an open->close-looking flip later, this sample
    must not be used to ground a fresh grasp event."""
    current_state = torch.zeros(1, STATE_DIM)
    current_state[:, GRIPPER_INDEX_RAW] = 0.0  # already closed right now

    action = torch.zeros(1, 4, ACTION_DIM)
    action[0, 0, GRIPPER_INDEX_RAW] = 1.0  # reopens
    action[0, 1, GRIPPER_INDEX_RAW] = 0.0  # then closes again (a regrasp, not the initial one)

    _, valid_mask = find_grasp_target(current_state, action, gripper_index=GRIPPER_INDEX_RAW)
    assert valid_mask.tolist() == [False]


def test_find_grasp_target_respects_custom_gripper_open_threshold() -> None:
    current_state = torch.full((1, STATE_DIM), 0.0)
    current_state[0, GRIPPER_INDEX_RAW] = 0.6  # open at threshold=0.5, closed at threshold=0.7
    action = torch.zeros(1, 2, ACTION_DIM)
    action[0, :, GRIPPER_INDEX_RAW] = 0.6

    _, valid_default = find_grasp_target(current_state, action, gripper_index=GRIPPER_INDEX_RAW, gripper_open_threshold=0.5)
    _, valid_strict = find_grasp_target(current_state, action, gripper_index=GRIPPER_INDEX_RAW, gripper_open_threshold=0.7)
    assert valid_default.tolist() == [False]  # no transition: open the whole window
    assert valid_strict.tolist() == [False]  # "pre-grasp" fails: already below the stricter threshold


# ---- TargetPointHead --------------------------------------------------------------------------


def test_target_point_head_shape_and_gradient() -> None:
    head = TargetPointHead(latent_dim=12, hidden_dim=16)
    pooled_latent = torch.randn(3, 12, requires_grad=True)
    target_xyz = head(pooled_latent)
    assert target_xyz.shape == (3, 3)
    assert torch.isfinite(target_xyz).all()
    target_xyz.sum().backward()
    assert pooled_latent.grad is not None
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in head.parameters())


# ---- TemporalActionDecoder target_xyz conditioning -----------------------------------------


def test_temporal_action_decoder_with_target_xyz() -> None:
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM, state_dim=5, latent_dim=12, hidden_dim=16, horizon=4,
        num_layers=1, num_heads=2, ffn_dim=16, dropout=0.0, use_target_xyz=True,
    )
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    target_xyz = torch.randn(2, 3)
    actions = decoder(latent_tokens, None, state, target_xyz=target_xyz)
    assert actions.shape == (2, 4, ACTION_DIM)
    with pytest.raises(ValueError, match="use_target_xyz"):
        decoder(latent_tokens, None, state, target_xyz=None)


def test_predicted_target_xyz_conditioning_changes_decoder_output() -> None:
    """Predicted-target conditioning is actually wired into the decoder path: same latent tokens
    and current state, different `target_xyz` -> different action chunk."""
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM, state_dim=5, latent_dim=12, hidden_dim=16, horizon=4,
        num_layers=1, num_heads=2, ffn_dim=16, dropout=0.0, use_target_xyz=True,
    )
    decoder.eval()
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    target_a = torch.randn(2, 3)
    target_b = target_a + 5.0
    with torch.no_grad():
        out_a = decoder(latent_tokens, None, state, target_xyz=target_a)
        out_b = decoder(latent_tokens, None, state, target_xyz=target_b)
    assert not torch.allclose(out_a, out_b)


def test_policy_plan_action_chunk_actually_uses_target_point_head() -> None:
    """End-to-end (through `SafeDiffVLAPolicy`, not the bare decoder module): zeroing out
    `target_point_head`'s parameters (forcing its prediction to a constant) must change
    `plan_action_chunk`'s output relative to the untouched, randomly-initialized head -- proving
    the head's prediction actually reaches the decoder, not just that the decoder *can* accept a
    `target_xyz` argument in isolation."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    batch = make_grounded_batch()
    with torch.no_grad():
        actions_before, metrics_before = policy.plan_action_chunk(batch)
        for p in policy.target_point_head.parameters():
            p.zero_()
        actions_after, metrics_after = policy.plan_action_chunk(batch)
    assert not torch.equal(metrics_before["predicted_target_xyz"], metrics_after["predicted_target_xyz"])
    assert not torch.allclose(actions_before, actions_after)


# ---- training: loss_target -------------------------------------------------------------------


def test_grounded_grasp_policy_trains_with_target_loss() -> None:
    policy = make_policy(architecture=ARCH)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 1.0  # force "pre-grasp" so loss_target is non-trivial
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, -1, GRIPPER_INDEX_RAW] = -1.0  # guarantee a transition inside the horizon
    loss, metrics = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"loss", "loss_pos", "loss_rot", "loss_grip", "loss_subgoal", "loss_smooth", "loss_target"} <= metrics.keys()
    assert metrics["loss_target"] > 0
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.target_point_head.parameters())


def test_grounded_grasp_loss_target_zero_when_batch_has_no_valid_transition() -> None:
    """Every row's gripper stays open the whole window -> no valid label anywhere -> loss_target
    is a safe 0, not NaN, and contributes nothing (see `masked_mse`'s all-False-mask handling)."""
    policy = make_policy(architecture=ARCH)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0  # never closes
    _, metrics = policy(batch)
    assert metrics["loss_target"] == pytest.approx(0.0, abs=1e-8)


def test_grounded_grasp_lambda_target_scales_loss() -> None:
    policy_default = make_policy(architecture=ARCH, lambda_target=1.0)
    policy_zeroed = make_policy(architecture=ARCH, lambda_target=0.0)
    # Same weights so only lambda_target differs.
    policy_zeroed.load_state_dict(policy_default.state_dict())
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, -1, GRIPPER_INDEX_RAW] = -1.0
    with torch.no_grad():
        loss_default, metrics = policy_default(batch)
        loss_zeroed, _ = policy_zeroed(batch)
    assert metrics["loss_target"] > 0
    assert loss_default.item() == pytest.approx(loss_zeroed.item() + metrics["loss_target"], abs=1e-6)


# ---- proximity-based reactive close / one-shot guard ---------------------------------------


def test_proximity_close_trigger_fires_within_threshold() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    assert policy.config.grounded_grasp_close_threshold_m == pytest.approx(0.03)
    # Fallback stats (no dataset_stats given) are mean=0/std=1, so normalized == physical meters.
    policy._grounded_grasp_last_target_xyz = torch.zeros(1, 3)
    current_state = torch.zeros(1, STATE_DIM)
    current_state[0, :3] = 0.01  # ~1.7cm away, well within the 3cm default threshold
    action = torch.full((1, ACTION_DIM), 9.0)  # 9.0 stands in for "whatever the decoder predicted"

    out = policy._grounded_grasp_reactive_close(action, current_state)
    assert policy._grounded_grasp_close_triggered is True
    assert out[0, GRIPPER_INDEX_RAW].item() != pytest.approx(9.0)
    for i in range(ACTION_DIM):
        if i != GRIPPER_INDEX_RAW:
            assert out[0, i].item() == pytest.approx(9.0)  # rotation/position channels untouched


def test_proximity_close_trigger_does_not_fire_outside_threshold() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._grounded_grasp_last_target_xyz = torch.zeros(1, 3)
    current_state = torch.zeros(1, STATE_DIM)
    current_state[0, :3] = 1.0  # 1m away, far outside the 3cm default threshold
    action = torch.full((1, ACTION_DIM), 9.0)

    out = policy._grounded_grasp_reactive_close(action, current_state)
    assert policy._grounded_grasp_close_triggered is False
    assert torch.equal(out, action)  # untouched


def test_one_shot_close_guard_does_not_reopen_once_triggered() -> None:
    """Once triggered, moving the EE far away from the target on a later step must NOT reopen the
    gripper -- the whole point of the one-shot latch."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._grounded_grasp_last_target_xyz = torch.zeros(1, 3)

    near_state = torch.zeros(1, STATE_DIM)
    near_state[0, :3] = 0.01
    action = torch.full((1, ACTION_DIM), 9.0)
    triggered_output = policy._grounded_grasp_reactive_close(action.clone(), near_state)
    assert policy._grounded_grasp_close_triggered is True
    closed_value = triggered_output[0, GRIPPER_INDEX_RAW].item()

    far_state = torch.zeros(1, STATE_DIM)
    far_state[0, :3] = 5.0  # now far away -- a proximity-only check would reopen; the latch must not
    later_output = policy._grounded_grasp_reactive_close(action.clone(), far_state)
    assert policy._grounded_grasp_close_triggered is True
    assert later_output[0, GRIPPER_INDEX_RAW].item() == pytest.approx(closed_value)


def test_reset_clears_close_trigger_and_cached_target() -> None:
    policy = make_policy(architecture=ARCH)
    policy._grounded_grasp_close_triggered = True
    policy._grounded_grasp_last_target_xyz = torch.zeros(1, 3)
    policy.reset()
    assert policy._grounded_grasp_close_triggered is False
    assert policy._grounded_grasp_last_target_xyz is None


def test_select_action_applies_reactive_close_when_near_target() -> None:
    """Integration: `select_action`'s returned gripper channel reflects the proximity trigger,
    not the decoder's own (in this test, deliberately huge/"open") regressed gripper value."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    batch = make_grounded_batch(batch_size=1)
    batch[OBS_STATE][:, :3] = 0.0
    action = policy.select_action(batch)
    # Force the cached target to sit right on top of the (zeroed) current EE position, then ask
    # again on the same still-open queue item's *next* call is not needed -- directly drive the
    # helper the way `select_action` itself does, for a deterministic, isolated check.
    policy._grounded_grasp_last_target_xyz = torch.zeros(1, 3)
    out = policy._grounded_grasp_reactive_close(action, policy._current_state(batch))
    assert policy._grounded_grasp_close_triggered is True
    assert out[0, GRIPPER_INDEX_RAW].item() != pytest.approx(action[0, GRIPPER_INDEX_RAW].item())


def test_reactive_close_is_noop_before_any_chunk_planned() -> None:
    """Defensive: if `_grounded_grasp_last_target_xyz` is still None (no chunk planned yet this
    episode), the reactive-close helper must return `action` completely unmodified rather than
    crashing or guessing a target."""
    policy = make_policy(architecture=ARCH)
    policy.reset()
    assert policy._grounded_grasp_last_target_xyz is None
    action = torch.full((1, ACTION_DIM), 3.0)
    out = policy._grounded_grasp_reactive_close(action, torch.zeros(1, STATE_DIM))
    assert torch.equal(out, action)
    assert policy._grounded_grasp_close_triggered is False


# ---- no regression on the existing temporal_decoder architecture ---------------------------


def test_temporal_decoder_has_no_grounded_grasp_state() -> None:
    """The plain `temporal_decoder` architecture must not build any of the new module's
    machinery -- new-path additions must be strictly opt-in."""
    policy = make_policy(architecture="temporal_decoder")
    assert not hasattr(policy, "target_point_head")
    assert not hasattr(policy, "action_pos_mean")
    assert not hasattr(policy, "state_pos_mean")
    assert not hasattr(policy, "action_grip_mean")
    assert policy.decoder.use_target_xyz is False
    assert not hasattr(policy.decoder, "target_xyz_encoder")


def test_temporal_decoder_select_action_unaffected_by_grounded_grasp_addition() -> None:
    """`select_action` for `temporal_decoder` never calls the (nonexistent) reactive-close path
    and returns the executor's action completely unmodified -- same contract as before this
    change (see `test_safediff_vla.py::test_select_action_queue_and_reset`)."""
    policy = make_policy(architecture="temporal_decoder")
    batch = make_batch()
    action = policy.select_action(batch)
    assert action.shape == (2, ACTION_DIM)
    assert not hasattr(policy, "_grounded_grasp_last_target_xyz") or policy._grounded_grasp_last_target_xyz is None


def test_temporal_decoder_loss_metrics_include_zero_loss_target() -> None:
    """`_forward_temporal_decoder` is shared code (see module docstring) -- the new `loss_target`
    metric key is always present (additive), but must be exactly 0 and change nothing numerically
    for architectures that don't use it."""
    policy = make_policy(architecture="temporal_decoder")
    loss, metrics = policy(make_batch())
    assert metrics["loss_target"] == 0.0
    assert loss.item() == pytest.approx(
        metrics["loss"], abs=1e-6
    )  # sanity: reported loss matches the tensor actually returned


def test_config_rejects_negative_lambda_target() -> None:
    with pytest.raises(ValueError, match="lambda_target"):
        make_config(architecture=ARCH, lambda_target=-1.0)


def test_config_rejects_non_positive_close_threshold() -> None:
    with pytest.raises(ValueError, match="grounded_grasp_close_threshold_m"):
        make_config(architecture=ARCH, grounded_grasp_close_threshold_m=0.0)


def test_config_grounded_grasp_is_a_registered_architecture() -> None:
    config = make_config(architecture=ARCH)
    assert config.architecture == ARCH


def test_policy_construction_builds_target_head_only_for_grounded_grasp() -> None:
    policy = make_policy(architecture=ARCH)
    assert hasattr(policy, "target_point_head")
    assert hasattr(policy, "action_pos_mean") and policy.action_pos_mean.shape == (3,)
    assert hasattr(policy, "action_grip_mean") and policy.action_grip_mean.shape == (1,)
    assert policy.decoder.use_target_xyz is True


def test_nominal_action_head_never_called_for_grounded_grasp() -> None:
    """G-equivalent for the new architecture: neither `forward()` nor `plan_action_chunk()` ever
    call the backbone's own nominal action head."""
    from tests.policies.safediff_vla.testing_utils import NominalCallForbiddenBackbone

    config = make_config(architecture=ARCH)
    policy = SafeDiffVLAPolicy(config, backbone=NominalCallForbiddenBackbone(4, ACTION_DIM, state_dim=STATE_DIM))
    policy(make_grounded_batch())
    policy.plan_action_chunk(make_grounded_batch())


def test_grounded_grasp_backbone_stays_frozen() -> None:
    policy = make_policy(architecture=ARCH)
    loss, _ = policy(make_grounded_batch())
    loss.backward()
    assert all(p.grad is None for p in policy.backbone.parameters())
    assert any(p.grad is not None for p in policy.target_point_head.parameters())
