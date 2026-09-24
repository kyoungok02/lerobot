import torch
from torch import Tensor, nn

from .utils import IMAGE_MODALITY, STATE_MODALITY, TEXT_MODALITY


class LanguageGroundedTargetPooling(nn.Module):
    """Per-token text-queried cross-attention over image tokens -- `TargetPointHead`'s input
    feature only. Replaces `SafeDiffVLAPolicy._pooled_latent`'s language-agnostic masked-mean
    pooling (still used unchanged by `_predict_subgoal`) with a feature the instruction actually
    shapes. Does not touch `TemporalActionDecoder`'s own full-token cross-attention (that reads
    `latent_tokens` directly, never this module's output).

    A first version queried with a single masked-mean-pooled text vector (one query per sample).
    That collapsed the instruction into one summary BEFORE it ever saw the image, so a short
    shared prefix ("primitive: Please pick the poker ...") dominated the query and the few tokens
    that actually name the target were diluted away -- confirmed by a fresh 5k checkpoint: held-out
    xyz L2 improved (0.0635m -> 0.0413m) but instruction-conditioned card selection got WORSE, not
    better (nearest-card identity-flip rate across instructions 25% -> 0%, correct-instructed-card
    rate 33.3% -> 27.8%). That design is deliberately not reused.

    This version queries with EVERY text token independently: each instruction token attends over
    the image tokens on its own (no pre-pooling before the query), so a token specific to "7 of
    spades" can pull different image evidence than the shared "pick the poker" prefix tokens do.
    Query: all TEXT_MODALITY tokens, one query row per text token position (padded/invalid text
    positions are computed too -- attention is per-query-row independent, so this is exactly
    equivalent to querying only the valid text tokens -- and then masked out below).
    Key/Value: IMAGE_MODALITY tokens only (`key_padding_mask` excludes text/state tokens and any
    invalid/padded position).
    Aggregation: the per-text-token grounded outputs are masked-mean-pooled over valid text
    positions into a single `[B, D]` vector.
    Output: `[grounded_text_pooled ; text_pooled (the original, un-grounded masked-mean text
    summary) ; state_token]` projected back down to `latent_dim` -- same shape contract as
    `_pooled_latent`'s own output, so `TargetPointHead` needs no change.
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

        text_pooled = masked_mean(TEXT_MODALITY)  # original, un-grounded text summary
        state_mask = (latent_modality_ids == STATE_MODALITY).unsqueeze(-1).to(latent_tokens.dtype)
        state_token = (latent_tokens * state_mask).sum(dim=1) / state_mask.sum(dim=1).clamp_min(1)

        image_valid = (latent_modality_ids == IMAGE_MODALITY) & valid  # [B, N]
        # A row with zero valid image tokens would mask out every key for that row's attention
        # softmax (all `-inf` -> NaN) -- never the case with the real backbone (camera tokens are
        # always present), but fall back to attending over every valid token for that row alone
        # rather than propagate NaN.
        no_image_tokens = image_valid.sum(dim=1, keepdim=True) == 0
        image_valid = image_valid | (no_image_tokens & valid)

        # Query with every position (image/state rows included) -- attention is computed
        # independently per query row, so this is exactly equivalent to querying only the text
        # rows, without needing a ragged per-sample slice. Only the TEXT_MODALITY rows' outputs are
        # kept below; everything else is discarded.
        grounded_tokens, _ = self.cross_attn(
            query=latent_tokens, key=latent_tokens, value=latent_tokens, key_padding_mask=~image_valid, need_weights=False
        )  # [B, N, D]

        text_valid_f = ((latent_modality_ids == TEXT_MODALITY) & valid).unsqueeze(-1).to(latent_tokens.dtype)
        grounded_text_pooled = (grounded_tokens * text_valid_f).sum(dim=1) / text_valid_f.sum(dim=1).clamp_min(1)

        return self.output_projection(torch.cat([grounded_text_pooled, text_pooled, state_token], dim=-1))
