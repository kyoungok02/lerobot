from torch import Tensor, nn


class TargetPointHead(nn.Module):
    """Predicts a single grasp-target xyz `[B, 3]` from the pooled VLM multimodal latent alone
    (no current-state input -- unlike `state_predictor.py`'s `SubgoalStatePredictor`, which also
    conditions on the current state). Used by `architecture="temporal_decoder_grounded_grasp"` to
    ground the decoder's action chunk on an explicit "where is the thing I'm reaching for"
    estimate, separating target-identity/localization from the pose trajectory itself.

    Output is in the same MEAN_STD-normalized space as `action[..., :3]` (the regression target
    is the un-encoded xyz slice of the dataset's own future action -- see
    `utils.find_grasp_target`), not raw physical units.
    """

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, pooled_latent: Tensor) -> Tensor:
        """`pooled_latent`: [B, latent_dim]. Returns [B, 3]."""
        return self.net(pooled_latent)
