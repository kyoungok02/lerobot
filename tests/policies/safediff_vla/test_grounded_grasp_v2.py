"""Tests for `architecture="temporal_decoder_grounded_grasp_v2"` (experimental) -- see
`configuration_safediff_vla.py`'s architecture docstring for the forward-path-tracing findings
motivating this design (the canonical model is NOT instruction-blind, but attention on the
tokens that actually name the target's identity is far lower than on boilerplate instruction
tokens). Reuses `test_safediff_vla.py`'s config/batch/policy builders and `test_grounded_grasp.py`'s
`make_grounded_batch`/`_build_modality_batch` helpers -- not a separate convention.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.safediff_vla.grounded_grasp_v2 import TargetQueryExtractor, VisualGroundingCrossAttention
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.policies.safediff_vla.temporal_decoder import TemporalActionDecoder
from lerobot.policies.safediff_vla.utils import _state_gripper_is_open
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.safediff_vla.test_grounded_grasp import _build_modality_batch, make_grounded_batch
from tests.policies.safediff_vla.test_safediff_vla import ACTION_DIM, STATE_DIM, make_policy

ARCH = "temporal_decoder_grounded_grasp_v2"


# ---- TargetQueryExtractor -----------------------------------------------------------------


def test_target_query_extractor_shape_and_gradient() -> None:
    d = 12
    extractor = TargetQueryExtractor(d)
    image_tokens = torch.randn(3, 20, d, requires_grad=True)
    text_tokens = torch.randn(3, 4, d, requires_grad=True)
    state_token = torch.randn(3, 1, d, requires_grad=True)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)

    query = extractor(tokens, pad_mask, modality_ids)
    assert query.shape == (3, d)
    assert torch.isfinite(query).all()

    query.sum().backward()
    assert text_tokens.grad is not None and torch.any(text_tokens.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in extractor.parameters())


def test_target_query_extractor_reacts_to_instruction_change() -> None:
    """Same image/state, different text tokens (standing in for a different instruction) -> a
    different target-language query -- proves the query is instruction-sensitive."""
    d = 12
    extractor = TargetQueryExtractor(d)
    image_tokens = torch.randn(1, 20, d)
    state_token = torch.randn(1, 1, d)
    text_a = torch.randn(1, 4, d)
    text_b = torch.randn(1, 4, d)
    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_a, state_token)
    tokens_b, _, _ = _build_modality_batch(image_tokens, text_b, state_token)
    with torch.no_grad():
        query_a = extractor(tokens_a, pad_mask, modality_ids)
        query_b = extractor(tokens_b, pad_mask, modality_ids)
    assert not torch.allclose(query_a, query_b)


def test_target_query_extractor_finite_when_every_text_token_padded() -> None:
    d = 12
    extractor = TargetQueryExtractor(d)
    image_tokens = torch.randn(2, 10, d)
    text_tokens = torch.randn(2, 4, d)
    state_token = torch.randn(2, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)
    pad_mask = pad_mask.clone()
    pad_mask[:, 10:14] = False  # mask out every text token
    query = extractor(tokens, pad_mask, modality_ids)
    assert torch.isfinite(query).all()


# ---- VisualGroundingCrossAttention --------------------------------------------------------


def test_visual_grounding_cross_attention_shape_and_gradient() -> None:
    d = 12
    cross_attn = VisualGroundingCrossAttention(d)
    target_query = torch.randn(3, d, requires_grad=True)
    image_tokens = torch.randn(3, 20, d, requires_grad=True)
    text_tokens = torch.randn(3, 4, d)
    state_token = torch.randn(3, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)

    grounded = cross_attn(target_query, tokens, pad_mask, modality_ids)
    assert grounded.shape == (3, d)
    assert torch.isfinite(grounded).all()

    grounded.sum().backward()
    assert target_query.grad is not None and torch.any(target_query.grad != 0)
    assert image_tokens.grad is not None and torch.any(image_tokens.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in cross_attn.parameters())


def test_visual_grounding_cross_attention_reacts_to_visual_token_change() -> None:
    """Same target query/text/state, different image tokens -> a different grounded feature."""
    d = 12
    cross_attn = VisualGroundingCrossAttention(d)
    target_query = torch.randn(1, d)
    text_tokens = torch.randn(1, 4, d)
    state_token = torch.randn(1, 1, d)
    image_a = torch.randn(1, 20, d)
    image_b = torch.randn(1, 20, d)
    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_a, text_tokens, state_token)
    tokens_b, _, _ = _build_modality_batch(image_b, text_tokens, state_token)
    with torch.no_grad():
        grounded_a = cross_attn(target_query, tokens_a, pad_mask, modality_ids)
        grounded_b = cross_attn(target_query, tokens_b, pad_mask, modality_ids)
    assert not torch.allclose(grounded_a, grounded_b)


def test_visual_grounding_cross_attention_finite_when_every_image_token_padded() -> None:
    d = 12
    cross_attn = VisualGroundingCrossAttention(d)
    target_query = torch.randn(2, d)
    image_tokens = torch.randn(2, 10, d)
    text_tokens = torch.randn(2, 4, d)
    state_token = torch.randn(2, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)
    pad_mask = pad_mask.clone()
    pad_mask[:, :10] = False  # mask out every image token
    grounded = cross_attn(target_query, tokens, pad_mask, modality_ids)
    assert torch.isfinite(grounded).all()


# ---- TemporalActionDecoder: use_grounded_grasp_v2 conditioning ----------------------------


def test_temporal_action_decoder_with_grounded_grasp_v2() -> None:
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM, state_dim=5, latent_dim=12, hidden_dim=16, horizon=4,
        num_layers=1, num_heads=2, ffn_dim=16, dropout=0.0,
        use_target_xyz=True, use_grounded_grasp_v2=True,
    )
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    target_xyz = torch.randn(2, 3)
    grounded_feature = torch.randn(2, 12)
    phase = torch.zeros(2, dtype=torch.long)
    actions = decoder(
        latent_tokens, None, state, target_xyz=target_xyz,
        grounded_target_feature=grounded_feature, grasp_phase=phase,
    )
    assert actions.shape == (2, 4, ACTION_DIM)
    assert torch.isfinite(actions).all()
    with pytest.raises(ValueError, match="use_grounded_grasp_v2"):
        decoder(latent_tokens, None, state, target_xyz=target_xyz, grounded_target_feature=None, grasp_phase=phase)
    with pytest.raises(ValueError, match="use_grounded_grasp_v2"):
        decoder(
            latent_tokens, None, state, target_xyz=target_xyz,
            grounded_target_feature=grounded_feature, grasp_phase=None,
        )


def test_grounded_target_feature_conditioning_changes_decoder_output() -> None:
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM, state_dim=5, latent_dim=12, hidden_dim=16, horizon=4,
        num_layers=1, num_heads=2, ffn_dim=16, dropout=0.0,
        use_target_xyz=True, use_grounded_grasp_v2=True,
    )
    decoder.eval()
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    target_xyz = torch.randn(2, 3)
    phase = torch.zeros(2, dtype=torch.long)
    feature_a = torch.randn(2, 12)
    feature_b = feature_a + 5.0
    with torch.no_grad():
        out_a = decoder(latent_tokens, None, state, target_xyz=target_xyz, grounded_target_feature=feature_a, grasp_phase=phase)
        out_b = decoder(latent_tokens, None, state, target_xyz=target_xyz, grounded_target_feature=feature_b, grasp_phase=phase)
    assert not torch.allclose(out_a, out_b)


def test_grasp_phase_conditioning_changes_decoder_output() -> None:
    """PRE_GRASP (0) vs POST_GRASP (1), everything else fixed -> different action chunk."""
    decoder = TemporalActionDecoder(
        action_dim=ACTION_DIM, state_dim=5, latent_dim=12, hidden_dim=16, horizon=4,
        num_layers=1, num_heads=2, ffn_dim=16, dropout=0.0,
        use_target_xyz=True, use_grounded_grasp_v2=True,
    )
    decoder.eval()
    latent_tokens = torch.randn(2, 6, 12)
    state = torch.randn(2, 5)
    target_xyz = torch.randn(2, 3)
    feature = torch.randn(2, 12)
    with torch.no_grad():
        out_pre = decoder(
            latent_tokens, None, state, target_xyz=target_xyz,
            grounded_target_feature=feature, grasp_phase=torch.zeros(2, dtype=torch.long),
        )
        out_post = decoder(
            latent_tokens, None, state, target_xyz=target_xyz,
            grounded_target_feature=feature, grasp_phase=torch.ones(2, dtype=torch.long),
        )
    assert not torch.allclose(out_pre, out_post)


# ---- grasp-phase label formula (training-time: read off observation.state's own gripper) -----


def test_grasp_phase_label_formula_pre_vs_post() -> None:
    """The exact formula `_forward_temporal_decoder`/`_plan_temporal_decoder` use to derive the
    training-time phase label from `observation.state`'s own (raw-physical) gripper channel:
    open (state convention: below threshold) -> PRE_GRASP (0), closed -> POST_GRASP (1)."""
    state_gripper = torch.tensor([0.0, 1.0])  # open, closed (state convention: 0=open, 1=closed)
    phase = (~_state_gripper_is_open(state_gripper, 0.5)).long()
    assert phase.tolist() == [0, 1]


# ---- policy-level: end-to-end target query/visual grounding wiring ------------------------


def test_policy_predicted_target_reacts_to_text_and_image_changes() -> None:
    """End-to-end through the policy's own modules (not the bare submodules in isolation): same
    image/state, different instruction -> different predicted target xyz, AND same instruction,
    different image -> different predicted target xyz."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    d = policy._multimodal_latent_dim()

    def predict(image: torch.Tensor, text: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tokens, pad_mask, modality_ids = _build_modality_batch(image, text, state)
        with torch.no_grad():
            query = policy.target_query_extractor(tokens, pad_mask, modality_ids)
            grounded = policy.visual_grounding_cross_attn(query, tokens, pad_mask, modality_ids)
            return policy.target_point_head(grounded)

    image = torch.randn(1, 20, d)
    state = torch.randn(1, 1, d)
    text_a = torch.randn(1, 4, d)
    text_b = torch.randn(1, 4, d)
    assert not torch.allclose(predict(image, text_a, state), predict(image, text_b, state))

    image_b = torch.randn(1, 20, d)
    assert not torch.allclose(predict(image, text_a, state), predict(image_b, text_a, state))


def test_policy_plan_action_chunk_actually_uses_grounded_grasp_v2_target_head() -> None:
    """Zeroing `target_point_head`'s parameters must change `plan_action_chunk`'s output -- proves
    the head's prediction (via `grounded_target_feature` AND `target_xyz`, both derived from it)
    actually reaches the decoder, not just that the decoder can accept these arguments."""
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


# ---- training: loss_target valid masking ---------------------------------------------------


def test_grounded_grasp_v2_policy_trains_with_target_loss() -> None:
    policy = make_policy(architecture=ARCH)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0  # pre-grasp (state convention: 0=open)
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, -1, GRIPPER_INDEX_RAW] = -1.0  # guarantee a transition inside the horizon
    loss, metrics = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {"loss", "loss_pos", "loss_rot", "loss_grip", "loss_subgoal", "loss_smooth", "loss_target"} <= metrics.keys()
    assert metrics["loss_target"] > 0
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.target_point_head.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.target_query_extractor.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.visual_grounding_cross_attn.parameters())


def test_grounded_grasp_v2_loss_target_zero_when_no_valid_transition_in_horizon() -> None:
    """Every row's gripper stays open the whole window -> no valid label anywhere -> loss_target
    is a safe 0, not NaN."""
    policy = make_policy(architecture=ARCH)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0  # pre-grasp (state convention: 0=open)
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0  # never closes
    _, metrics = policy(batch)
    assert metrics["loss_target"] == pytest.approx(0.0, abs=1e-8)


def test_grounded_grasp_v2_lambda_target_scales_loss() -> None:
    policy_default = make_policy(architecture=ARCH, lambda_target=1.0)
    policy_zeroed = make_policy(architecture=ARCH, lambda_target=0.0)
    policy_zeroed.load_state_dict(policy_default.state_dict())
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, -1, GRIPPER_INDEX_RAW] = -1.0
    with torch.no_grad():
        loss_default, metrics = policy_default(batch)
        loss_zeroed, _ = policy_zeroed(batch)
    assert metrics["loss_target"] > 0
    assert loss_default.item() == pytest.approx(loss_zeroed.item() + metrics["loss_target"], abs=1e-6)


# ---- proximity-based reactive close / phase transition / queue clear / one-shot guard -----


def test_v2_proximity_close_trigger_fires_and_flips_phase() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    assert policy._v2_grasp_phase == 0
    # Fallback stats (no dataset_stats given) are mean=0/std=1, so normalized == physical meters.
    policy._v2_last_target_xyz = torch.zeros(1, 3)
    current_state = torch.zeros(1, STATE_DIM)
    current_state[0, :3] = 0.01  # ~1.7cm away, well within the 3cm default threshold
    action = torch.full((1, ACTION_DIM), 9.0)

    out = policy._grounded_grasp_v2_reactive_close(action, current_state)
    assert policy._v2_close_triggered is True
    assert policy._v2_grasp_phase == 1
    assert out[0, GRIPPER_INDEX_RAW].item() != pytest.approx(9.0)
    for i in range(ACTION_DIM):
        if i != GRIPPER_INDEX_RAW:
            assert out[0, i].item() == pytest.approx(9.0)  # rotation/position channels untouched


def test_v2_proximity_close_trigger_does_not_fire_outside_threshold() -> None:
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._v2_last_target_xyz = torch.zeros(1, 3)
    current_state = torch.zeros(1, STATE_DIM)
    current_state[0, :3] = 1.0  # 1m away, far outside the 3cm default threshold
    action = torch.full((1, ACTION_DIM), 9.0)

    out = policy._grounded_grasp_v2_reactive_close(action, current_state)
    assert policy._v2_close_triggered is False
    assert policy._v2_grasp_phase == 0
    assert torch.equal(out, action)


def test_v2_proximity_trigger_clears_action_queue() -> None:
    policy = make_policy(architecture=ARCH, action_horizon=5, execute_horizon=5)
    policy.eval()
    policy.reset()
    policy._v2_last_target_xyz = torch.zeros(1, 3)
    policy._executor._action_queue.extend([torch.zeros(1, ACTION_DIM) for _ in range(5)])
    assert len(policy._executor._action_queue) == 5

    current_state = torch.zeros(1, STATE_DIM)
    current_state[0, :3] = 0.01
    action = torch.full((1, ACTION_DIM), 9.0)
    policy._grounded_grasp_v2_reactive_close(action, current_state)
    assert len(policy._executor._action_queue) == 0


def test_v2_queue_clear_forces_full_replan_on_next_select_action_call() -> None:
    """After the proximity trigger clears the queue, the very next `select_action` call must
    invoke a fresh `plan_action_chunk` (the queue refills to `execute_horizon - 1` after popping
    one) instead of silently popping from an already-empty queue."""
    policy = make_policy(architecture=ARCH, execute_horizon=4, action_horizon=4)
    policy.eval()
    policy.reset()
    batch = make_grounded_batch(batch_size=1)
    # Deliberately far from wherever the randomly-initialized target_point_head happens to
    # predict, so this first call's own proximity check can never accidentally trigger (RNG-order
    # dependent otherwise -- a real, if unlikely, flake, not a hypothetical one).
    batch[OBS_STATE][:, :3] = 1000.0

    policy.select_action(batch)  # commits a fresh chunk; queue now has execute_horizon-1 left
    assert len(policy._executor._action_queue) == policy.config.execute_horizon - 1
    assert policy._v2_close_triggered is False

    batch[OBS_STATE][:, :3] = 0.0
    policy._v2_last_target_xyz = torch.zeros(1, 3)  # force the trigger on the very next call
    policy.select_action(batch)
    # The trigger clears the queue INSIDE this call, after the executor already popped one action
    # from the (still populated) pre-trigger queue -- so it must now be fully empty, not left with
    # leftover actions planned under stale PRE_GRASP conditioning.
    assert len(policy._executor._action_queue) == 0

    # Next call: queue is empty, so `ActionExecutor.select_action`'s own `if not self._action_queue`
    # branch fires -> `plan_fn()` runs again -> a fresh full chunk is committed (queue refills).
    policy.select_action(batch)
    assert len(policy._executor._action_queue) == policy.config.execute_horizon - 1


def test_v2_one_shot_close_guard_does_not_reopen_or_flip_phase_back() -> None:
    """Once triggered, moving the EE far away from the target on a later step must NOT reopen the
    gripper or flip the phase back -- the whole point of the one-shot latch."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    policy._v2_last_target_xyz = torch.zeros(1, 3)

    near_state = torch.zeros(1, STATE_DIM)
    near_state[0, :3] = 0.01
    action = torch.full((1, ACTION_DIM), 9.0)
    triggered_output = policy._grounded_grasp_v2_reactive_close(action.clone(), near_state)
    assert policy._v2_close_triggered is True
    assert policy._v2_grasp_phase == 1
    closed_value = triggered_output[0, GRIPPER_INDEX_RAW].item()

    far_state = torch.zeros(1, STATE_DIM)
    far_state[0, :3] = 5.0  # now far away -- a proximity-only check would reopen; the latch must not
    later_output = policy._grounded_grasp_v2_reactive_close(action.clone(), far_state)
    assert policy._v2_close_triggered is True
    assert policy._v2_grasp_phase == 1  # does not flip back to PRE_GRASP
    assert later_output[0, GRIPPER_INDEX_RAW].item() == pytest.approx(closed_value)


def test_v2_reset_clears_phase_trigger_and_cached_target() -> None:
    policy = make_policy(architecture=ARCH)
    policy._v2_grasp_phase = 1
    policy._v2_close_triggered = True
    policy._v2_last_target_xyz = torch.zeros(1, 3)
    policy.reset()
    assert policy._v2_grasp_phase == 0
    assert policy._v2_close_triggered is False
    assert policy._v2_last_target_xyz is None


def test_v2_select_action_applies_reactive_close_when_near_target() -> None:
    """Integration: `select_action`'s returned gripper channel reflects the proximity trigger,
    not the decoder's own (in this test, deliberately huge/"open") regressed gripper value."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    policy.reset()
    batch = make_grounded_batch(batch_size=1)
    batch[OBS_STATE][:, :3] = 0.0
    action = policy.select_action(batch)
    action = action.clone()
    action[0, GRIPPER_INDEX_RAW] = 5.0
    policy._v2_last_target_xyz = torch.zeros(1, 3)
    out = policy._grounded_grasp_v2_reactive_close(action, policy._current_state(batch))
    assert policy._v2_close_triggered is True
    assert out[0, GRIPPER_INDEX_RAW].item() != pytest.approx(action[0, GRIPPER_INDEX_RAW].item())


def test_v2_reactive_close_is_noop_before_any_chunk_planned() -> None:
    """Defensive: if `_v2_last_target_xyz` is still None (no chunk planned yet this episode), the
    reactive-close helper must return `action` completely unmodified rather than crashing."""
    policy = make_policy(architecture=ARCH)
    policy.reset()
    assert policy._v2_last_target_xyz is None
    action = torch.full((1, ACTION_DIM), 3.0)
    out = policy._grounded_grasp_v2_reactive_close(action, torch.zeros(1, STATE_DIM))
    assert torch.equal(out, action)
    assert policy._v2_close_triggered is False
    assert policy._v2_grasp_phase == 0


# ---- no regression on existing architectures ------------------------------------------------


def test_other_architectures_never_build_grounded_grasp_v2_modules() -> None:
    for arch in (
        "temporal_decoder",
        "temporal_decoder_subgoal",
        "temporal_decoder_grounded_grasp",
        "temporal_decoder_instruction",
        "temporal_decoder_text_crossattn",
    ):
        policy = make_policy(architecture=arch)
        assert not hasattr(policy, "target_query_extractor")
        assert not hasattr(policy, "visual_grounding_cross_attn")
        assert policy.decoder.use_grounded_grasp_v2 is False
        assert not hasattr(policy.decoder, "grounded_feature_encoder")
        assert not hasattr(policy.decoder, "phase_encoder")


def test_grounded_grasp_v2_builds_expected_modules_only() -> None:
    policy = make_policy(architecture=ARCH)
    assert hasattr(policy, "target_query_extractor")
    assert hasattr(policy, "visual_grounding_cross_attn")
    assert hasattr(policy, "target_point_head")
    assert not hasattr(policy, "language_grounded_target_pooling")  # v1's module, not v2's
    assert not hasattr(policy, "modality_pool_projection")  # subgoal-only
    assert policy.decoder.use_grounded_grasp_v2 is True
    assert policy.decoder.use_target_xyz is True  # reused for the predicted_target_xyz embedding
    assert policy.decoder.use_subgoal is False
    assert policy.decoder.use_instruction is False
    assert policy.decoder.use_text_crossattn is False
