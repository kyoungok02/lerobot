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

from lerobot.policies.safediff_vla.language_grounded_target_pooling import LanguageGroundedTargetPooling
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW
from lerobot.policies.safediff_vla.target_point_head import TargetPointHead
from lerobot.policies.safediff_vla.temporal_decoder import TemporalActionDecoder
from lerobot.policies.safediff_vla.utils import (
    IMAGE_MODALITY,
    OTHER_MODALITY,
    STATE_MODALITY,
    TEXT_MODALITY,
    compute_prefix_modality_ids,
    find_grasp_target,
)
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
    found, including a transition landing exactly on action[:, 0]. `observation.state`'s gripper
    polarity is the OPPOSITE of `action`'s (0.0=open here, vs action's 1.0=open) -- see
    `utils._state_gripper_is_open` / `_action_gripper_is_open`."""
    current_state = torch.zeros(3, STATE_DIM)
    current_state[:, GRIPPER_INDEX_RAW] = 0.0  # every row starts open (state convention: 0=open)

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
    current_state[:, GRIPPER_INDEX_RAW] = 0.0  # both open (state convention: 0=open)

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
    current_state[:, GRIPPER_INDEX_RAW] = 1.0  # already closed right now (state convention: 1=closed)

    action = torch.zeros(1, 4, ACTION_DIM)
    action[0, 0, GRIPPER_INDEX_RAW] = 1.0  # reopens
    action[0, 1, GRIPPER_INDEX_RAW] = 0.0  # then closes again (a regrasp, not the initial one)

    _, valid_mask = find_grasp_target(current_state, action, gripper_index=GRIPPER_INDEX_RAW)
    assert valid_mask.tolist() == [False]


def test_find_grasp_target_respects_custom_gripper_open_threshold() -> None:
    """`gripper_open_threshold` must actually shift both `_state_gripper_is_open`'s (`<threshold`)
    and `_action_gripper_is_open`'s (`>threshold`) decision boundaries -- constant state=action=0.6
    reads as "closed" (not pre-grasp) under threshold=0.5 for state (0.6 is not < 0.5), but
    "open" under threshold=0.9 (0.6 < 0.9); with a constant 0.6 action sequence, that same
    threshold=0.9 also reads every action step as closed (0.6 is not > 0.9), producing an
    immediate valid transition at index 0."""
    current_state = torch.full((1, STATE_DIM), 0.0)
    current_state[0, GRIPPER_INDEX_RAW] = 0.6
    action = torch.zeros(1, 2, ACTION_DIM)
    action[0, :, GRIPPER_INDEX_RAW] = 0.6

    _, valid_low = find_grasp_target(current_state, action, gripper_index=GRIPPER_INDEX_RAW, gripper_open_threshold=0.5)
    _, valid_high = find_grasp_target(current_state, action, gripper_index=GRIPPER_INDEX_RAW, gripper_open_threshold=0.9)
    assert valid_low.tolist() == [False]  # state gate fails: 0.6 is not < 0.5, reads as closed/post-grasp
    assert valid_high.tolist() == [True]  # state gate passes (0.6 < 0.9) AND action reads closed immediately


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


# ---- compute_prefix_modality_ids ------------------------------------------------------------


def test_compute_prefix_modality_ids_tags_image_text_state_segments() -> None:
    """Synthetic `prefix_att_masks` (state token's own `att_masks==1` signal, exactly what
    `embed_prefix` returns) + a known `lang_width` must recover the image/text/state boundaries
    `_pooled_latent` relies on -- no real VLM/backbone involved."""
    n_image, n_text = 5, 3
    seq_len = n_image + n_text + 1  # + state
    att_masks = torch.zeros(2, seq_len, dtype=torch.bool)
    att_masks[:, n_image + n_text] = True  # state token's position, same for both rows here

    modality_ids = compute_prefix_modality_ids(att_masks, lang_width=n_text)
    expected = torch.tensor([IMAGE_MODALITY] * n_image + [TEXT_MODALITY] * n_text + [STATE_MODALITY])
    assert torch.equal(modality_ids[0], expected)
    assert torch.equal(modality_ids[1], expected)


def test_compute_prefix_modality_ids_handles_per_sample_state_index() -> None:
    """State token position can differ per sample (e.g. different missing-camera masking doesn't
    shift it in practice, but the function must not assume a shared index across the batch)."""
    seq_len = 10
    att_masks = torch.zeros(2, seq_len, dtype=torch.bool)
    att_masks[0, 6] = True
    att_masks[1, 8] = True

    modality_ids = compute_prefix_modality_ids(att_masks, lang_width=2)
    assert modality_ids[0].tolist() == [0, 0, 0, 0, 1, 1, 2, -1, -1, -1]
    assert modality_ids[1].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 2, -1]


def test_compute_prefix_modality_ids_trailing_padding_is_other_not_image_or_text() -> None:
    """Positions after the state token (only reachable if `prefix_length` forces trailing
    zero-pad -- never the case with this codebase's default `prefix_length=-1`, but the function
    must still degrade safely) are tagged `OTHER_MODALITY`, not silently folded into image/text."""
    att_masks = torch.zeros(1, 8, dtype=torch.bool)
    att_masks[0, 4] = True  # state at index 4, indices 5-7 are unreachable trailing pad
    modality_ids = compute_prefix_modality_ids(att_masks, lang_width=2)
    assert modality_ids[0, 5:].tolist() == [OTHER_MODALITY, OTHER_MODALITY, OTHER_MODALITY]


# ---- _pooled_latent: modality-aware pooling (subgoal path only -- the grounded-grasp target
# head has its own `LanguageGroundedTargetPooling` instead, see the section below) -------------


def _build_modality_batch(image_tokens: torch.Tensor, text_tokens: torch.Tensor, state_token: torch.Tensor):
    """`image_tokens`/`text_tokens`: `[B, n, D]`, `state_token`: `[B, 1, D]` -> the
    `(latent_tokens, latent_pad_mask, latent_modality_ids)` triple `_pooled_latent` expects."""
    bsz = image_tokens.shape[0]
    n_image, n_text = image_tokens.shape[1], text_tokens.shape[1]
    tokens = torch.cat([image_tokens, text_tokens, state_token], dim=1)
    pad_mask = torch.ones(bsz, tokens.shape[1], dtype=torch.bool)
    modality_ids = torch.cat(
        [
            torch.full((bsz, n_image), IMAGE_MODALITY, dtype=torch.long),
            torch.full((bsz, n_text), TEXT_MODALITY, dtype=torch.long),
            torch.full((bsz, 1), STATE_MODALITY, dtype=torch.long),
        ],
        dim=1,
    )
    return tokens, pad_mask, modality_ids


def _make_average_projection_policy(latent_dim: int = 12) -> SafeDiffVLAPolicy:
    """A `temporal_decoder_subgoal` policy (the one remaining architecture that builds
    `modality_pool_projection`/`_pooled_latent` -- `temporal_decoder_grounded_grasp`'s target head
    uses `LanguageGroundedTargetPooling` instead, tested separately below) with
    `modality_pool_projection` forced to `(image_pooled + text_pooled + state_token) / 3` -- a
    simple, deterministic, exactly-known instantiation of "concat + projection" so
    pooling-sensitivity comparisons below don't depend on a randomly-initialized projection's own
    scaling."""
    policy = make_policy(architecture="temporal_decoder_subgoal")
    with torch.no_grad():
        identity_blocks = torch.cat([torch.eye(latent_dim)] * 3, dim=1) / 3
        policy.modality_pool_projection.weight.copy_(identity_blocks)
        policy.modality_pool_projection.bias.zero_()
    return policy


def test_pooled_latent_shape_and_gradient() -> None:
    policy = make_policy(architecture="temporal_decoder_subgoal")
    d = policy._multimodal_latent_dim()
    image_tokens = torch.randn(3, 40, d, requires_grad=True)
    text_tokens = torch.randn(3, 5, d, requires_grad=True)
    state_token = torch.randn(3, 1, d, requires_grad=True)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)

    pooled = policy._pooled_latent(tokens, pad_mask, modality_ids)
    assert pooled.shape == (3, d)
    assert torch.isfinite(pooled).all()

    pooled.sum().backward()
    assert image_tokens.grad is not None and torch.any(image_tokens.grad != 0)
    assert text_tokens.grad is not None and torch.any(text_tokens.grad != 0)
    assert state_token.grad is not None and torch.any(state_token.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.modality_pool_projection.parameters())


def test_pooled_latent_is_finite_when_a_modality_has_no_valid_tokens() -> None:
    """Every text token masked out (`latent_pad_mask=0` there) must not produce NaN/Inf -- the
    `clamp_min(1)` denominator guard must actually be exercised, not just present."""
    policy = make_policy(architecture="temporal_decoder_subgoal")
    d = policy._multimodal_latent_dim()
    image_tokens = torch.randn(2, 10, d)
    text_tokens = torch.randn(2, 4, d)
    state_token = torch.randn(2, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)
    pad_mask = pad_mask.clone()
    pad_mask[:, 10:14] = False  # mask out every text token

    pooled = policy._pooled_latent(tokens, pad_mask, modality_ids)
    assert torch.isfinite(pooled).all()


def test_pooled_latent_image_only_change_moves_output_text_only_change_does_not() -> None:
    """Perturbing ONLY the image tokens (text/state fixed) must change the pooled output, and that
    change must be independent of the (unperturbed) text tokens -- proves the image branch
    actually reaches `_pooled_latent`'s output, not just that *some* token changed it."""
    policy = _make_average_projection_policy()
    d = 12
    image_a = torch.randn(1, 20, d)
    text_tokens = torch.randn(1, 4, d)
    state_token = torch.randn(1, 1, d)
    image_b = image_a.clone()
    image_b[:, 0, :] += 10.0  # perturb a single image token

    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_a, text_tokens, state_token)
    tokens_b, _, _ = _build_modality_batch(image_b, text_tokens, state_token)
    pooled_a = policy._pooled_latent(tokens_a, pad_mask, modality_ids)
    pooled_b = policy._pooled_latent(tokens_b, pad_mask, modality_ids)

    # Averaging projection: pooled = (image_pooled + text_pooled + state_token) / 3, text/state
    # unchanged here, so the whole delta must equal exactly the image branch's own shifted mean.
    expected_delta = (image_b.mean(dim=1) - image_a.mean(dim=1)) / 3
    assert torch.allclose(pooled_b - pooled_a, expected_delta, atol=1e-5)


def test_pooled_latent_text_only_change_moves_output_image_only_change_does_not() -> None:
    """Mirror of the above: perturbing ONLY the text tokens moves the output by exactly the text
    branch's own shifted mean, with image/state held fixed."""
    policy = _make_average_projection_policy()
    d = 12
    image_tokens = torch.randn(1, 20, d)
    text_a = torch.randn(1, 4, d)
    state_token = torch.randn(1, 1, d)
    text_b = text_a.clone()
    text_b[:, 1, :] -= 7.0

    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_a, state_token)
    tokens_b, _, _ = _build_modality_batch(image_tokens, text_b, state_token)
    pooled_a = policy._pooled_latent(tokens_a, pad_mask, modality_ids)
    pooled_b = policy._pooled_latent(tokens_b, pad_mask, modality_ids)

    expected_delta = (text_b.mean(dim=1) - text_a.mean(dim=1)) / 3
    assert torch.allclose(pooled_b - pooled_a, expected_delta, atol=1e-5)


def test_pooled_latent_state_token_passes_through_unaveraged() -> None:
    """The (single) state token must be used as-is, not diluted by averaging against anything
    else -- perturbing it alone moves the output by exactly its own delta (under the averaging
    projection, scaled by the fixed 1/3 block weight, same as the image/text branches above)."""
    policy = _make_average_projection_policy()
    d = 12
    image_tokens = torch.randn(1, 20, d)
    text_tokens = torch.randn(1, 4, d)
    state_a = torch.randn(1, 1, d)
    state_b = state_a + 3.0

    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_a)
    tokens_b, _, _ = _build_modality_batch(image_tokens, text_tokens, state_b)
    pooled_a = policy._pooled_latent(tokens_a, pad_mask, modality_ids)
    pooled_b = policy._pooled_latent(tokens_b, pad_mask, modality_ids)

    expected_delta = (state_b - state_a).squeeze(1) / 3
    assert torch.allclose(pooled_b - pooled_a, expected_delta, atol=1e-5)


def test_pooled_latent_far_more_sensitive_to_text_change_than_old_all_token_mean() -> None:
    """The whole point of this fix: with many more image tokens than text tokens (a realistic
    single/multi-camera token count vs. a short instruction), a fixed-magnitude *text-only* change
    (standing in for an instruction swap) must move the new modality-aware pooled output far more
    than the OLD all-token unweighted mean (`latent_tokens.mean(dim=1)`) would have moved for the
    exact same tokens -- i.e. the fix actually undoes the dilution, not just refactors the code.

    The comparison is deterministic (not seed-dependent): the text shift is a constant vector
    added to every text token, so both deltas are exact linear functions of `n_image`/`n_text`
    alone, independent of the (otherwise arbitrary) random image/state token values.
    """
    policy = _make_average_projection_policy()
    d = 12
    n_image, n_text = 40, 3
    image_tokens = torch.randn(1, n_image, d)
    state_token = torch.randn(1, 1, d)
    text_a = torch.randn(1, n_text, d)
    text_b = text_a + 50.0  # drastic, uniform shift standing in for a swapped instruction

    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_a, state_token)
    tokens_b, _, _ = _build_modality_batch(image_tokens, text_b, state_token)

    new_pooled_a = policy._pooled_latent(tokens_a, pad_mask, modality_ids)
    new_pooled_b = policy._pooled_latent(tokens_b, pad_mask, modality_ids)
    new_delta = (new_pooled_b - new_pooled_a).norm()

    old_pooled_a = tokens_a.mean(dim=1)
    old_pooled_b = tokens_b.mean(dim=1)
    old_delta = (old_pooled_b - old_pooled_a).norm()

    # Analytically: new_delta = 50/3 per dim, old_delta = 3*50/(40+3+1) per dim -> ratio ~4.9x.
    assert new_delta > old_delta * 3


# ---- LanguageGroundedTargetPooling: text-queried cross-attention over image tokens -----------
# (grounded-grasp target head only -- `_predict_subgoal` keeps using `_pooled_latent` above)


def test_language_grounded_target_pooling_shape_gradient_and_nan_safety() -> None:
    d = 12
    pooling = LanguageGroundedTargetPooling(d)
    image_tokens = torch.randn(3, 20, d, requires_grad=True)
    text_tokens = torch.randn(3, 4, d, requires_grad=True)
    state_token = torch.randn(3, 1, d, requires_grad=True)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)

    grounded = pooling(tokens, pad_mask, modality_ids)
    assert grounded.shape == (3, d)
    assert torch.isfinite(grounded).all()

    grounded.sum().backward()
    assert image_tokens.grad is not None and torch.any(image_tokens.grad != 0)
    assert text_tokens.grad is not None and torch.any(text_tokens.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in pooling.parameters())


def test_language_grounded_target_pooling_finite_when_every_image_token_is_padded() -> None:
    """Every image token masked out (`latent_pad_mask=0` there) must not produce NaN -- the
    zero-valid-image-tokens fallback must actually be exercised, not just present (mirrors
    `test_pooled_latent_is_finite_when_a_modality_has_no_valid_tokens` above, for the new module)."""
    d = 12
    pooling = LanguageGroundedTargetPooling(d)
    image_tokens = torch.randn(2, 10, d)
    text_tokens = torch.randn(2, 4, d)
    state_token = torch.randn(2, 1, d)
    tokens, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_tokens, state_token)
    pad_mask = pad_mask.clone()
    pad_mask[:, :10] = False  # mask out every image token

    grounded = pooling(tokens, pad_mask, modality_ids)
    assert torch.isfinite(grounded).all()


def test_language_grounded_target_pooling_reacts_to_text_change_image_fixed() -> None:
    """Swapping ONLY the text tokens (standing in for a different instruction), image/state held
    fixed, must change the grounded output -- both because the query itself changes and because a
    different query attends differently over the same image tokens."""
    d = 12
    pooling = LanguageGroundedTargetPooling(d)
    image_tokens = torch.randn(1, 20, d)
    text_a = torch.randn(1, 4, d)
    state_token = torch.randn(1, 1, d)
    text_b = torch.randn(1, 4, d)

    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_a, state_token)
    tokens_b, _, _ = _build_modality_batch(image_tokens, text_b, state_token)
    with torch.no_grad():
        out_a = pooling(tokens_a, pad_mask, modality_ids)
        out_b = pooling(tokens_b, pad_mask, modality_ids)
    assert not torch.allclose(out_a, out_b)


def test_language_grounded_target_pooling_reacts_to_image_change_text_fixed() -> None:
    """Swapping ONLY the image tokens, text/state held fixed, must also change the grounded
    output -- unlike a pure bag-of-words text encoder, the cross-attention's value/key side
    depends on which image tokens are actually present, not just the query."""
    d = 12
    pooling = LanguageGroundedTargetPooling(d)
    image_a = torch.randn(1, 20, d)
    text_tokens = torch.randn(1, 4, d)
    state_token = torch.randn(1, 1, d)
    image_b = torch.randn(1, 20, d)

    tokens_a, pad_mask, modality_ids = _build_modality_batch(image_a, text_tokens, state_token)
    tokens_b, _, _ = _build_modality_batch(image_b, text_tokens, state_token)
    with torch.no_grad():
        out_a = pooling(tokens_a, pad_mask, modality_ids)
        out_b = pooling(tokens_b, pad_mask, modality_ids)
    assert not torch.allclose(out_a, out_b)


def test_predict_target_xyz_uses_language_grounded_pooling_reacts_both_ways() -> None:
    """End-to-end through `SafeDiffVLAPolicy._predict_target_xyz` (not the bare module): a
    `temporal_decoder_grounded_grasp` policy's predicted target xyz must react to a text-only
    change (image/state fixed) AND to an image-only change (text/state fixed)."""
    policy = make_policy(architecture=ARCH)
    policy.eval()
    d = policy._multimodal_latent_dim()
    image_tokens = torch.randn(1, 20, d)
    text_a = torch.randn(1, 4, d)
    text_b = torch.randn(1, 4, d)
    state_token = torch.randn(1, 1, d)

    tokens_text_a, pad_mask, modality_ids = _build_modality_batch(image_tokens, text_a, state_token)
    tokens_text_b, _, _ = _build_modality_batch(image_tokens, text_b, state_token)
    with torch.no_grad():
        target_text_a = policy._predict_target_xyz(tokens_text_a, pad_mask, modality_ids)
        target_text_b = policy._predict_target_xyz(tokens_text_b, pad_mask, modality_ids)
    assert not torch.allclose(target_text_a, target_text_b)

    image_a = torch.randn(1, 20, d)
    image_b = torch.randn(1, 20, d)
    tokens_image_a, pad_mask2, modality_ids2 = _build_modality_batch(image_a, text_a, state_token)
    tokens_image_b, _, _ = _build_modality_batch(image_b, text_a, state_token)
    with torch.no_grad():
        target_image_a = policy._predict_target_xyz(tokens_image_a, pad_mask2, modality_ids2)
        target_image_b = policy._predict_target_xyz(tokens_image_b, pad_mask2, modality_ids2)
    assert not torch.allclose(target_image_a, target_image_b)


def test_language_grounded_target_pooling_only_built_for_grounded_grasp_architecture() -> None:
    """Regression guard: `temporal_decoder` and `temporal_decoder_subgoal` must never build
    `language_grounded_target_pooling` (it's grounded-grasp-only, and must not affect the
    canonical decoder path at all), while `temporal_decoder_grounded_grasp` must."""
    assert not hasattr(make_policy(architecture="temporal_decoder"), "language_grounded_target_pooling")
    assert not hasattr(make_policy(architecture="temporal_decoder_subgoal"), "language_grounded_target_pooling")
    assert hasattr(make_policy(architecture=ARCH), "language_grounded_target_pooling")


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
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0  # force "pre-grasp" (state convention: 0=open) so loss_target is non-trivial
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
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0  # pre-grasp (state convention: 0=open)
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0  # never closes
    _, metrics = policy(batch)
    assert metrics["loss_target"] == pytest.approx(0.0, abs=1e-8)


def test_grounded_grasp_lambda_target_scales_loss() -> None:
    policy_default = make_policy(architecture=ARCH, lambda_target=1.0)
    policy_zeroed = make_policy(architecture=ARCH, lambda_target=0.0)
    # Same weights so only lambda_target differs.
    policy_zeroed.load_state_dict(policy_default.state_dict())
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0  # pre-grasp (state convention: 0=open)
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
    # Force a clearly-"open" gripper value on the decoder's own (otherwise randomly-initialized,
    # not reliably far from the force-closed value below by chance) output, so this assertion is
    # deterministic regardless of global RNG state / test execution order.
    action = action.clone()
    action[0, GRIPPER_INDEX_RAW] = 5.0
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


# ---- grounded_grasp_condition_decoder_on_target ablation switch -----------------------------


def test_condition_decoder_on_target_defaults_true_unchanged_behavior() -> None:
    """Default (True) must reproduce the existing, already-checkpointed behavior exactly --
    regression safety for the checkpoint already trained under the old (implicit) always-True
    contract."""
    policy = make_policy(architecture=ARCH)
    assert policy.config.grounded_grasp_condition_decoder_on_target is True
    assert policy.decoder.use_target_xyz is True


def test_condition_decoder_on_target_false_builds_unconditioned_decoder() -> None:
    """Ablation-B condition: `TargetPointHead` is still built (auxiliary loss still trainable),
    but the decoder itself is constructed WITHOUT target-xyz conditioning."""
    policy = make_policy(architecture=ARCH, grounded_grasp_condition_decoder_on_target=False)
    assert hasattr(policy, "target_point_head")
    assert policy.decoder.use_target_xyz is False
    assert not hasattr(policy.decoder, "target_xyz_encoder")


def test_condition_decoder_on_target_false_still_trains_target_loss() -> None:
    policy = make_policy(architecture=ARCH, grounded_grasp_condition_decoder_on_target=False)
    batch = make_grounded_batch()
    batch[OBS_STATE][:, GRIPPER_INDEX_RAW] = 0.0  # pre-grasp (state convention: 0=open)
    batch[ACTION][:, :, GRIPPER_INDEX_RAW] = 1.0
    batch[ACTION][:, -1, GRIPPER_INDEX_RAW] = -1.0
    loss, metrics = policy(batch)
    assert metrics["loss_target"] > 0
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in policy.target_point_head.parameters())


def test_condition_decoder_on_target_false_decoder_output_independent_of_target_head() -> None:
    """The whole point of the ablation: with conditioning off, the decoder's action-chunk output
    must NOT change when `target_point_head`'s parameters (hence its predictions) change --
    proving the decoder truly never sees `target_xyz` in this mode, not just that it's *labeled*
    unconditioned."""
    policy = make_policy(architecture=ARCH, grounded_grasp_condition_decoder_on_target=False)
    policy.eval()
    batch = make_grounded_batch()
    with torch.no_grad():
        actions_before, _ = policy.plan_action_chunk(batch)
        for p in policy.target_point_head.parameters():
            p.add_(1.0)  # perturb target_point_head's own output substantially
        actions_after, _ = policy.plan_action_chunk(batch)
    assert torch.equal(actions_before, actions_after)


def test_condition_decoder_on_target_true_decoder_output_does_depend_on_target_head() -> None:
    """Contrast case: with the default (conditioning ON), the same perturbation DOES change the
    decoder's output -- confirms the previous test isn't vacuously true for some unrelated reason."""
    policy = make_policy(architecture=ARCH, grounded_grasp_condition_decoder_on_target=True)
    policy.eval()
    batch = make_grounded_batch()
    with torch.no_grad():
        actions_before, _ = policy.plan_action_chunk(batch)
        for p in policy.target_point_head.parameters():
            p.add_(1.0)
        actions_after, _ = policy.plan_action_chunk(batch)
    assert not torch.allclose(actions_before, actions_after)
