"""`architecture="temporal_decoder_grounded_grasp_v2_5"` only. v2's `TemporalActionDecoder`
regresses a single 50-step chunk end to end, including the last few centimeters of the approach
where sub-cm precision actually matters for a successful grasp -- diagnostic oracle experiments on
this branch found close-*timing* (not just target-xyz accuracy) was a real bottleneck there, and
v2's own held-out eval never moved task success even after target-xyz localization improved. This
architecture does NOT change v2's grounding/target-localization/global-decoder pipeline AT ALL
(`TargetQueryExtractor`, `VisualGroundingCrossAttention`, `TargetPointHead`,
`TemporalActionDecoder` -- see `grounded_grasp_v2.py` and `temporal_decoder.py` -- every one of
these stays byte-for-byte the same computation v2 already does, in its own separate branch of
`modeling_safediff_vla.py` so nothing here can ever perturb v2's own code path). Instead it adds
ONE new module, `LocalGraspRefiner`, and a 3-stage runtime controller
(`SafeDiffVLAPolicy.select_action`'s v2_5 branch):

    PRE_GRASP (v2's own 50-step global approach, unchanged)
        -- until EE-to-predicted-target distance <= `local_grasp_radius_m` (one-shot, see below) --
    LOCAL_GRASP (`LocalGraspRefiner`, genuine closed-loop: re-observe and replan EVERY step)
        -- until the refiner's own predicted gripper first reads "closed" (one-shot latch) --
    POST_GRASP (v2's own global decoder again, now conditioned on phase=POST_GRASP)

`LocalGraspRefiner` never re-regresses an absolute target pose -- it predicts a BOUNDED residual
`(delta_xyz, delta_rotation, gripper)` relative to the CURRENT observed EE pose, explicitly
conditioned on `relative_xyz = predicted_target_xyz - current_ee_xyz` (v2's own predicted target,
just used as one more input here instead of directly driving 50 more steps of open-loop chunk
execution). It predicts a short (`local_grasp_action_horizon`, default 5) residual chunk purely as
a training-time receding-horizon target -- at inference only its first step is ever executed
(`local_grasp_execute_horizon=1`), and the whole chunk is thrown away and recomputed from a fresh
observation on the very next step. This is the one place in the whole architecture where control
is genuinely closed-loop at every single step; the 50-step global stage (both PRE_GRASP and
POST_GRASP) stays exactly the open-loop chunk-then-queue behavior it always was.
"""

import torch
from torch import Tensor, nn

# `SafeDiffVLAPolicy._v25_stage`'s three values (see `modeling_safediff_vla.py`'s
# `temporal_decoder_grounded_grasp_v2_5` branch). PRE_GRASP/POST_GRASP are the only two ever fed to
# `TemporalActionDecoder`'s existing binary `grasp_phase` embedding (0/1) -- LOCAL_GRASP never
# calls the decoder at all, so the decoder itself needs no third phase value and stays untouched.
PRE_GRASP = 0
LOCAL_GRASP = 1
POST_GRASP = 2


class LocalGraspRefiner(nn.Module):
    """Short-horizon closed-loop residual controller for the last few centimeters of a grasp
    approach. Inputs (all `[B, *]`, batch-first): `grounded_target_feature` (v2's own
    `VisualGroundingCrossAttention` output), `image_feature` (fresh masked-mean IMAGE_MODALITY
    pooling -- see `utils.masked_mean_by_modality` -- giving this module some broader visual
    context beyond the single target-grounded feature, computed fresh here rather than adding a
    new output to any v2 module), `current_state` (encoded 10-D, sin/cos rotation -- same
    convention as `TemporalActionDecoder`'s own input), `predicted_target_xyz` (v2's
    `TargetPointHead` output, normalized action-position space), `relative_xyz`
    (`predicted_target_xyz - current_ee_xyz`, in PHYSICAL METERS -- the explicit "how far and
    which direction from here" signal this controller is built around).

    Output: `(delta_xyz, delta_rotation, gripper)`, each `[B, horizon, *]`. `delta_xyz` (meters,
    robot-base frame) and `delta_rotation` (radians) are `tanh`-bounded local corrections relative
    to the CURRENT observed EE pose -- never an absolute pose regression -- so this module can only
    ever nudge the arm, never replace v2's own global localization. `gripper` is a direct
    (unbounded, linear) regressed value in the same normalized convention
    `TemporalActionDecoder.action_head`'s own gripper channel already uses -- not a delta, since
    "should the gripper be open or closed right now" is not naturally a residual quantity.
    """

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        hidden_dim: int,
        horizon: int,
        max_delta_xyz: float,
        max_delta_rot: float,
    ) -> None:
        super().__init__()
        if max_delta_xyz <= 0:
            raise ValueError("max_delta_xyz must be positive")
        if max_delta_rot <= 0:
            raise ValueError("max_delta_rot must be positive")
        self.horizon = horizon
        self.max_delta_xyz = max_delta_xyz
        self.max_delta_rot = max_delta_rot
        input_dim = 2 * latent_dim + state_dim + 3 + 3  # grounded + image feature, state, target, relative
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * 7),
        )

    def forward(
        self,
        grounded_target_feature: Tensor,
        image_feature: Tensor,
        current_state: Tensor,
        predicted_target_xyz: Tensor,
        relative_xyz: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = grounded_target_feature.shape[0]
        x = torch.cat(
            (grounded_target_feature, image_feature, current_state, predicted_target_xyz, relative_xyz), dim=-1
        )
        raw = self.net(x).view(batch_size, self.horizon, 7)
        delta_xyz = self.max_delta_xyz * torch.tanh(raw[..., :3])
        delta_rot = self.max_delta_rot * torch.tanh(raw[..., 3:6])
        gripper = raw[..., 6:7]
        return delta_xyz, delta_rot, gripper
