"""Deterministic Temporal Action Decoder -- the main SafeDiff-VLA architecture.

Replaces "diffuse a residual on top of SmolVLA's own nominal action chunk" (the old
`legacy_diffusion` architecture, see `legacy/diffusion_planner.py`): every configuration tried
there (critic-free, state-conditioned, subgoal-conditioned, temporal-conv-mixed) matched or
underperformed simply executing SmolVLA's own nominal chunk unmodified, once the
`action_horizon`/`execute_horizon` mismatch against SmolVLA's native `chunk_size` was fixed --
i.e. refining a nominal action chunk post hoc never once helped. This module instead treats
SmolVLA purely as a multimodal *encoder* (see `SafeDiffVLAPolicy._encode_multimodal_latent`) and
generates the action chunk directly:

    VLM latent tokens + current state (+ optional predicted subgoal state)
        -> Temporal Action Decoder (cross-attention + self-attention across the horizon axis)
        -> action trajectory [B, H, A]

`nominal` (SmolVLA's own action-head output) is not an input here at all -- it's kept around
elsewhere only as an ablation baseline (`architecture: smolvla_nominal`) and an optional
auxiliary-loss target, never as something this decoder edits.

The horizon-axis temporal mixing here is `nn.TransformerDecoder`'s self-attention among the
`horizon` action-query positions -- there is no additional Conv1D mixing layer in this decoder
(the Conv1D `temporal_kernel_size` / `num_temporal_layers` mixer lives only in the *legacy*
`ConditionalDiffusionPlanner`, see `legacy/diffusion_planner.py`).
"""

import math

import torch
from torch import Tensor, nn

from .utils import TEXT_MODALITY


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> Tensor:
    """Fixed sin/cos positional embedding for the `length` horizon positions, shape [length, dim]."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=device) / max(half - 1, 1))
    positions = torch.arange(length, device=device).float()[:, None] * frequencies[None]
    embedding = torch.cat((positions.sin(), positions.cos()), dim=-1)
    return torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))


class StateEncoder(nn.Module):
    """Small MLP embedding a `[B, state_dim]` (or `[B, H, state_dim]`) state into `hidden_dim`."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class TemporalActionDecoder(nn.Module):
    """Learnable action-query tokens, cross-attending to the VLM's multimodal latent tokens and
    conditioned on robot state, self-attending across the horizon axis (a plain Transformer
    decoder stack) so every position's prediction can depend on its neighbors -- unlike a
    per-position-independent MLP, which by construction cannot make the chunk cohere into a
    single smooth trajectory (see `diffusion_planner.py`'s docstring for the failure mode this
    caused in `legacy_diffusion`).
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int,
        horizon: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_subgoal: bool = False,
        use_target_xyz: bool = False,
        use_instruction: bool = False,
        use_text_crossattn: bool = False,
        use_grounded_grasp_v2: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.use_subgoal = use_subgoal
        self.use_target_xyz = use_target_xyz
        self.use_instruction = use_instruction
        self.use_text_crossattn = use_text_crossattn
        self.use_grounded_grasp_v2 = use_grounded_grasp_v2
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.action_queries = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        # Fixed (non-learnable) sin/cos position embedding, one per horizon slot. `action_queries`
        # is itself already per-position (so position is not strictly *required* here, the way it
        # would be for a shared/repeated query), but keeping this explicit rather than folding
        # position entirely into the learned query makes the horizon-axis structure legible and
        # is cheap.
        self.register_buffer(
            "horizon_position_embedding", sinusoidal_positions(horizon, hidden_dim, torch.device("cpu"))
        )
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        if use_subgoal:
            self.subgoal_encoder = StateEncoder(state_dim, hidden_dim)
        if use_target_xyz:
            # Same small-MLP shape as `state_encoder`/`subgoal_encoder` (reused via `StateEncoder`,
            # just with an input width of 3 instead of `state_dim`) -- added as one more additive
            # global-conditioning term, exactly mirroring how `subgoal_state` already works below.
            # `nn.TransformerDecoder`/`TransformerDecoderLayer` themselves are completely untouched.
            self.target_xyz_encoder = StateEncoder(3, hidden_dim)
        if use_instruction:
            # Same additive-global-conditioning pattern as `subgoal_state`/`target_xyz` above, just
            # with an input width of `latent_dim` (the raw, un-projected masked-mean text-token
            # pooling -- see `utils.masked_mean_by_modality` -- not `self.latent_projection`'s
            # `hidden_dim`-wide output). `nn.TransformerDecoder`/`TransformerDecoderLayer`
            # themselves are completely untouched; this only changes what gets added to `queries`.
            self.instruction_encoder = StateEncoder(latent_dim, hidden_dim)
        if use_grounded_grasp_v2:
            # Two MORE additive global-conditioning terms, same pattern as above, on top of
            # `target_xyz_encoder` (built by `use_target_xyz` -- callers set both flags together
            # for this architecture, see `modeling_safediff_vla.py`): `grounded_target_feature`
            # (from `grounded_grasp_v2.py`'s `VisualGroundingCrossAttention`, `latent_dim`-wide,
            # same as `instruction_encoder`'s input) and a binary grasp-phase embedding
            # (0=PRE_GRASP, 1=POST_GRASP). `nn.TransformerDecoder`/`TransformerDecoderLayer`
            # themselves are, again, completely untouched.
            self.grounded_feature_encoder = StateEncoder(latent_dim, hidden_dim)
            self.phase_encoder = nn.Embedding(2, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        if use_text_crossattn:
            # Unlike `instruction_encoder` above (which pools the text tokens into one vector
            # BEFORE the decoder ever sees them), this attends over the raw, un-pooled per-token
            # text sequence directly -- no mean/max/last-token pooling, no aggregation of any kind.
            # Applied AFTER the canonical multimodal cross-attention/self-attention stack
            # (`self.transformer`, completely untouched), as one extra residual cross-attention:
            # query = the transformer's own per-horizon-step output, key/value = the projected
            # multimodal memory restricted to TEXT_MODALITY positions only (via
            # `key_padding_mask` -- same "compute over the whole sequence, mask down to the
            # positions that matter" approach as `language_grounded_target_pooling.py`, so no
            # separate ragged text-only tensor/projection is needed).
            self.text_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        latent_tokens: Tensor,
        latent_pad_mask: Tensor | None,
        current_state: Tensor,
        subgoal_state: Tensor | None = None,
        target_xyz: Tensor | None = None,
        instruction_embedding: Tensor | None = None,
        latent_modality_ids: Tensor | None = None,
        grounded_target_feature: Tensor | None = None,
        grasp_phase: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            latent_tokens: [B, N_tokens, latent_dim] VLM multimodal latent (see
                `SafeDiffVLAPolicy._encode_multimodal_latent`).
            latent_pad_mask: [B, N_tokens] bool, True where `latent_tokens` is real (not padding),
                or None if every token is valid.
            current_state: [B, state_dim].
            subgoal_state: [B, state_dim] predicted subgoal state -- a single target, not a
                per-timestep trajectory (see `state_predictor.py`'s `SubgoalStatePredictor`) --
                added as global conditioning to every query position. Required iff
                `use_subgoal=True`.
            target_xyz: [B, 3] predicted grasp-target xyz (see `target_point_head.py`'s
                `TargetPointHead`) -- a single point, added as global conditioning to every query
                position exactly like `subgoal_state`. Required iff `use_target_xyz=True`.
            instruction_embedding: [B, latent_dim] raw (un-projected) masked-mean text-token
                pooling (see `utils.masked_mean_by_modality`) -- added as global conditioning to
                every query position exactly like `subgoal_state`/`target_xyz`. Required iff
                `use_instruction=True`.
            latent_modality_ids: [B, N_tokens] int modality tag per `latent_tokens` position (see
                `utils.compute_prefix_modality_ids`) -- used only to restrict the extra text-only
                cross-attention to TEXT_MODALITY positions. Required iff `use_text_crossattn=True`.
            grounded_target_feature: [B, latent_dim] (see `grounded_grasp_v2.py`'s
                `VisualGroundingCrossAttention`) -- added as global conditioning to every query
                position exactly like `subgoal_state`/`target_xyz`/`instruction_embedding`.
                Required iff `use_grounded_grasp_v2=True`.
            grasp_phase: [B] long, 0=PRE_GRASP / 1=POST_GRASP -- embedded and added as global
                conditioning exactly like the other terms above. Required iff
                `use_grounded_grasp_v2=True`.

        Returns: action trajectory [B, H, action_dim].
        """
        if self.use_subgoal and subgoal_state is None:
            raise ValueError("This decoder was built with use_subgoal=True but got none.")
        if self.use_target_xyz and target_xyz is None:
            raise ValueError("This decoder was built with use_target_xyz=True but got none.")
        if self.use_instruction and instruction_embedding is None:
            raise ValueError("This decoder was built with use_instruction=True but got none.")
        if self.use_text_crossattn and latent_modality_ids is None:
            raise ValueError("This decoder was built with use_text_crossattn=True but got no latent_modality_ids.")
        if self.use_grounded_grasp_v2 and (grounded_target_feature is None or grasp_phase is None):
            raise ValueError(
                "This decoder was built with use_grounded_grasp_v2=True but got no "
                "grounded_target_feature and/or grasp_phase."
            )
        batch_size = latent_tokens.shape[0]
        memory = self.latent_projection(latent_tokens)
        memory_key_padding_mask = None if latent_pad_mask is None else ~latent_pad_mask

        queries = (self.action_queries + self.horizon_position_embedding)[None].expand(batch_size, -1, -1)
        queries = queries + self.state_encoder(current_state)[:, None, :]
        if self.use_subgoal:
            queries = queries + self.subgoal_encoder(subgoal_state)[:, None, :]
        if self.use_target_xyz:
            queries = queries + self.target_xyz_encoder(target_xyz)[:, None, :]
        if self.use_instruction:
            queries = queries + self.instruction_encoder(instruction_embedding)[:, None, :]
        if self.use_grounded_grasp_v2:
            queries = queries + self.grounded_feature_encoder(grounded_target_feature)[:, None, :]
            queries = queries + self.phase_encoder(grasp_phase)[:, None, :]

        decoded = self.transformer(
            tgt=queries, memory=memory, memory_key_padding_mask=memory_key_padding_mask
        )
        if self.use_text_crossattn:
            valid = latent_tokens.new_ones(latent_tokens.shape[:2], dtype=torch.bool) if latent_pad_mask is None else latent_pad_mask.to(torch.bool)
            text_valid = (latent_modality_ids == TEXT_MODALITY) & valid  # [B, N_tokens], no pooling
            # Zero-valid-text-token row would mask out every key for that row's softmax (all
            # `-inf` -> NaN) -- never the case with a real instruction, but fall back to attending
            # over every valid token for that row alone rather than propagate NaN.
            no_text_tokens = text_valid.sum(dim=1, keepdim=True) == 0
            text_valid = text_valid | (no_text_tokens & valid)
            text_grounded, _ = self.text_cross_attn(
                query=decoded, key=memory, value=memory, key_padding_mask=~text_valid, need_weights=False
            )
            decoded = decoded + text_grounded
        return self.action_head(decoded)
