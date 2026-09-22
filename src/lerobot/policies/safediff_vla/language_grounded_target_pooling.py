import torch
from torch import Tensor, nn

from .utils import IMAGE_MODALITY, STATE_MODALITY, TEXT_MODALITY


class LanguageGroundedTargetPooling(nn.Module):
    """Text-queried cross-attention over image tokens -- `TargetPointHead`'s input feature only.
    Replaces `SafeDiffVLAPolicy._pooled_latent`'s language-agnostic masked-mean pooling (still
    used unchanged by `_predict_subgoal`) with a feature the instruction actually shapes: which
    image tokens matter is picked by the text itself, instead of averaging every image patch in
    regardless of what was asked for. Does not touch `TemporalActionDecoder`'s own full-token
    cross-attention (that reads `latent_tokens` directly, never this module's output).

    Query: masked-mean TEXT_MODALITY tokens (one vector per sample -- the same text pooling
    `_pooled_latent` already computes).
    Key/Value: IMAGE_MODALITY tokens only (`key_padding_mask` excludes text/state tokens and any
    invalid/padded position, so the query can only pull from the scene).
    Output: `[grounded_visual_feature ; text_pooled ; state_token]` projected back down to
    `latent_dim` -- same shape contract as `_pooled_latent`'s own output, so `TargetPointHead`
    needs no change.
    """

    def __init__(self, latent_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            num_heads = 1
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.output_projection = nn.Linear(3 * latent_dim, latent_dim)

    def forward(self, latent_tokens: Tensor, latent_pad_mask: Tensor | None, latent_modality_ids: Tensor) -> Tensor:
        """`latent_tokens`: [B, N, D]. `latent_pad_mask`: [B, N] bool or None (all-valid).
        `latent_modality_ids`: [B, N] int (see `utils.compute_prefix_modality_ids`). Returns
        [B, D]."""
        if latent_pad_mask is None:
            latent_pad_mask = latent_tokens.new_ones(latent_tokens.shape[:2], dtype=torch.bool)
        valid = latent_pad_mask.to(torch.bool)

        def masked_mean(modality: int) -> Tensor:
            mask = ((latent_modality_ids == modality) & valid).unsqueeze(-1).to(latent_tokens.dtype)
            return (latent_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        text_pooled = masked_mean(TEXT_MODALITY)
        state_mask = (latent_modality_ids == STATE_MODALITY).unsqueeze(-1).to(latent_tokens.dtype)
        state_token = (latent_tokens * state_mask).sum(dim=1) / state_mask.sum(dim=1).clamp_min(1)

        image_valid = (latent_modality_ids == IMAGE_MODALITY) & valid  # [B, N]
        # A row with zero valid image tokens would mask out every key for that row's attention
        # softmax (all `-inf` -> NaN) -- never the case with the real backbone (camera tokens are
        # always present), but fall back to attending over every valid token for that row alone
        # rather than propagate NaN.
        no_image_tokens = image_valid.sum(dim=1, keepdim=True) == 0
        image_valid = image_valid | (no_image_tokens & valid)
        query = text_pooled.unsqueeze(1)  # [B, 1, D]
        grounded, _ = self.cross_attn(
            query=query, key=latent_tokens, value=latent_tokens, key_padding_mask=~image_valid, need_weights=False
        )
        grounded = grounded.squeeze(1)  # [B, D]

        return self.output_projection(torch.cat([grounded, text_pooled, state_token], dim=-1))
