"""`architecture="temporal_decoder_grounded_grasp_v2"` only (see `configuration_safediff_vla.py`'s
architecture docstring for the forward-path-tracing findings motivating this): explicit
target-language-query -> visual-grounding-cross-attention pipeline, replacing both
`_pooled_latent`'s language-agnostic masked-mean pooling and `LanguageGroundedTargetPooling`'s
mean/per-token-query designs (both already tried and both left instruction-conditioned card
selection at or below the un-supervised baseline -- see `language_grounded_target_pooling.py`'s
own docstring). Neither prior design ever gave the model a QUERY that could learn, end-to-end
against the target-xyz supervision, which instruction tokens actually identify the target;
`TargetQueryExtractor` does exactly that instead of any hard-coded rank/suit parsing or averaging.
"""

import torch
from torch import Tensor, nn

from .utils import IMAGE_MODALITY, TEXT_MODALITY


class TargetQueryExtractor(nn.Module):
    """Learned-query attention pooling over TEXT_MODALITY tokens only -- deliberately NOT mean
    pooling and NOT hard-coded token-position parsing (no assumption about where a rank/suit word
    sits in the tokenized instruction). A single learned query vector (shared across the whole
    batch, `nn.Parameter`) attends over an instruction's own tokens; because it is trained
    end-to-end against the auxiliary grasp-target xyz loss (see
    `SafeDiffVLAPolicy._forward_temporal_decoder`'s `temporal_decoder_grounded_grasp_v2` branch),
    it can learn to upweight whichever tokens actually carry the target's identity over generic
    task-boilerplate tokens ("please", "pick", "the", "poker"), without anyone telling it where
    those tokens are.
    """

    def __init__(self, latent_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            num_heads = 1
        self.query = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)

    def forward(self, latent_tokens: Tensor, latent_pad_mask: Tensor | None, latent_modality_ids: Tensor) -> Tensor:
        """`latent_tokens`: [B, N, D]. `latent_pad_mask`: [B, N] bool or None (all-valid).
        `latent_modality_ids`: [B, N] int (see `utils.compute_prefix_modality_ids`). Returns the
        target-language query [B, D]."""
        batch_size = latent_tokens.shape[0]
        if latent_pad_mask is None:
            latent_pad_mask = latent_tokens.new_ones(latent_tokens.shape[:2], dtype=torch.bool)
        valid = latent_pad_mask.to(torch.bool)

        text_valid = (latent_modality_ids == TEXT_MODALITY) & valid
        # Zero-valid-text-token row would mask out every key for that row's softmax (all `-inf`
        # -> NaN) -- never the case with a real instruction, but fall back to attending over
        # every valid token for that row alone rather than propagate NaN.
        no_text_tokens = text_valid.sum(dim=1, keepdim=True) == 0
        text_valid = text_valid | (no_text_tokens & valid)

        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, D]
        pooled, _ = self.attn(
            query=query, key=latent_tokens, value=latent_tokens, key_padding_mask=~text_valid, need_weights=False
        )
        return pooled.squeeze(1)


class VisualGroundingCrossAttention(nn.Module):
    """Cross-attention: query = `TargetQueryExtractor`'s target-language query, key/value =
    IMAGE_MODALITY tokens only. Output (`grounded_target_feature`) is a visual feature explicitly
    bound to that query -- fed into `target_point_head.py`'s `TargetPointHead` (the "TargetXYZHead")
    for the xyz regression, and ALSO added as its own additive conditioning term to
    `TemporalActionDecoder`'s horizon queries (`temporal_decoder.py`'s `use_grounded_grasp_v2`),
    alongside the predicted xyz itself and the binary grasp phase.
    """

    def __init__(self, latent_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            num_heads = 1
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)

    def forward(
        self, target_query: Tensor, latent_tokens: Tensor, latent_pad_mask: Tensor | None, latent_modality_ids: Tensor
    ) -> Tensor:
        """`target_query`: [B, D] (from `TargetQueryExtractor`). `latent_tokens`/`latent_pad_mask`/
        `latent_modality_ids`: same as `TargetQueryExtractor.forward`. Returns
        `grounded_target_feature` [B, D]."""
        if latent_pad_mask is None:
            latent_pad_mask = latent_tokens.new_ones(latent_tokens.shape[:2], dtype=torch.bool)
        valid = latent_pad_mask.to(torch.bool)

        image_valid = (latent_modality_ids == IMAGE_MODALITY) & valid
        no_image_tokens = image_valid.sum(dim=1, keepdim=True) == 0
        image_valid = image_valid | (no_image_tokens & valid)

        query = target_query.unsqueeze(1)  # [B, 1, D]
        grounded, _ = self.attn(
            query=query, key=latent_tokens, value=latent_tokens, key_padding_mask=~image_valid, need_weights=False
        )
        return grounded.squeeze(1)
