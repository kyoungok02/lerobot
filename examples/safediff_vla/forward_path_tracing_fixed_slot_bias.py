#!/usr/bin/env python
"""Forward-path tracing on the corrected-instruction CANONICAL checkpoint (`temporal_decoder`,
sincos+padfix, 20k) to locate WHERE the "fixed-slot" bias (closed-loop behavior that barely/never
changes with the instruction, found in `BASELINE_SUMMARY.md`) originates.

Read-only: forward hooks + direct calls into already-trained submodules for logging only. No
training, no new modules, no architecture change, existing checkpoint used as-is.

For each of 5 scenes (real `env.reset()`, not a rollout) and, per scene, 3-4 real card
instructions (true target + others actually present, reusing `instruction_grounding_audit.py`'s
scene/instruction construction), with image+state FROZEN and only the instruction text varied,
traces:

  1. Visual representation: image-token region of `_encode_multimodal_latent`'s output, alongside
     each present card's simulator GT world xyz. No detector is trained; "does the representation
     distinguish card positions" is checked via pairwise token diversity (spread among image
     tokens), not per-card localization (would need camera-projection geometry, out of scope
     here). Whether the image region itself shifts with the instruction (possible in principle --
     the VLM's own self-attention could let image tokens attend to text) is measured directly in
     stage 3, not assumed.
  2. Text representation: tokenizer ids + the text-token region of the same latent, pairwise
     cosine/L2 across instructions, and specifically for the token(s) naming the card identity
     (rank/suit words -- the only words that actually differ between instructions).
  3. Multimodal decoder memory: `TemporalActionDecoder.latent_projection(latent_tokens)` -- the
     literal `memory` tensor `nn.TransformerDecoder` cross-attends to -- full/image-region/
     text-region/state-token deltas across instructions.
  4. Decoder query: `action_queries + horizon_position_embedding + state_encoder(current_state)`
     (canonical architecture: no subgoal/target conditioning) -- the exact tensor entering
     cross-attention as `tgt`, computed via the real trained submodules.
  5. Decoder cross-attention: a forward-pre-hook on each `TransformerDecoderLayer.multihead_attn`
     re-invokes that SAME module with `need_weights=True` (a parallel, read-only call -- the real
     forward pass's own `need_weights=False` fast path is untouched) to recover per-layer/head
     attention mass on image/text/state token regions, and specifically on the card-identity
     token(s).
  6. Action trajectory: `plan_action_chunk`'s physical 50-step xyz trajectory, pairwise L2 (first-10
     and full-horizon), predicted gripper-close point, nearest real card to it.
  7. Per-scene sensitivity summary + a single A-E fixed-slot-bias classification.

Outputs:
    outputs/eval/safediff_vla_forward_path_tracing/summary.json

Usage:
    MUJOCO_GL=egl <vlabench-venv>/bin/python examples/safediff_vla/forward_path_tracing_fixed_slot_bias.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from instruction_grounding_audit import (  # noqa: E402
    build_batch,
    capture_scene,
    nearest_card,
    pretty_card_name,
)

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.rotation_encoding import GRIPPER_INDEX_RAW  # noqa: E402
from lerobot.policies.safediff_vla.utils import IMAGE_MODALITY, STATE_MODALITY, TEXT_MODALITY  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
GRIPPER_THRESHOLD = 0.5
CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_sincos_20k_padfix/checkpoints/020000/pretrained_model"
SCENE_SEEDS = [1000, 1001, 1002, 1003, 1004]
N_CARDS_PER_SCENE = 4

# Words that can appear as the card-identity slot in "Please pick the poker {X} of {Y}" -- the
# only tokens that actually differ between this task's instructions.
RANK_WORDS = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"}
SUIT_WORDS = {"spades", "clubs", "hearts", "diamonds"}
CARD_WORDS = RANK_WORDS | SUIT_WORDS


def l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    denom = a.norm() * b.norm()
    return float((a @ b / denom).item()) if denom > 0 else None


def pairwise(records: list[dict], key_fn) -> list[dict]:
    out = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = key_fn(records[i]), key_fn(records[j])
            if a is None or b is None:
                continue
            out.append(
                {
                    "instruction_a": records[i]["instruction"],
                    "instruction_b": records[j]["instruction"],
                    "l2": l2(a, b),
                    "cosine": cosine(a, b),
                }
            )
    return out


class CrossAttentionCapture:
    """Forward-pre-hook on each `TransformerDecoderLayer.multihead_attn`: re-invokes that same
    (already-trained) module a second time with `need_weights=True, average_attn_weights=False`
    to recover per-head attention weights, purely for logging -- the real forward call proceeds
    through its own default (`need_weights=False`) fast path completely unaffected, since the hook
    never touches `args`/`kwargs`."""

    def __init__(self, decoder) -> None:
        self.per_layer_weights: list[torch.Tensor] = []  # each [B, num_heads, horizon, N_tokens]
        self._handles = []
        self._in_hook = False  # reentrancy guard -- see _hook's docstring note below
        for layer in decoder.transformer.layers:
            handle = layer.multihead_attn.register_forward_pre_hook(self._hook, with_kwargs=True)
            self._handles.append(handle)

    def _hook(self, module, args, kwargs):
        # The extra `module(...)` call below is itself a full `__call__`, which re-triggers this
        # SAME forward_pre_hook -- without this guard that recurses infinitely. The guard makes
        # the inner call a no-op passthrough (falls straight through to the module's real
        # forward()), so only the OUTER (real) call's pre-hook invocation does any work.
        if self._in_hook:
            return
        self._in_hook = True
        try:
            query, key, value = args[0], args[1], args[2]
            with torch.no_grad():
                _out, weights = module(
                    query,
                    key,
                    value,
                    key_padding_mask=kwargs.get("key_padding_mask"),
                    attn_mask=kwargs.get("attn_mask"),
                    need_weights=True,
                    average_attn_weights=False,
                )
            self.per_layer_weights.append(weights.detach().cpu())
        finally:
            self._in_hook = False

    def reset(self) -> None:
        self.per_layer_weights = []

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def modality_mass(weights: torch.Tensor, modality_ids: torch.Tensor, modality: int) -> float:
    """`weights`: [B, num_heads, horizon, N_tokens] (one layer). Mean fraction of attention mass
    (averaged over batch/heads/horizon positions) landing on token positions tagged `modality`."""
    mask = (modality_ids[0] == modality).to(weights.dtype)  # [N_tokens]
    mass = (weights * mask).sum(dim=-1)  # [B, num_heads, horizon]
    return float(mass.mean().item())


def token_mass(weights: torch.Tensor, token_indices: list[int]) -> float:
    if not token_indices:
        return 0.0
    idx = torch.tensor(token_indices)
    mass = weights[..., idx].sum(dim=-1)  # [B, num_heads, horizon]
    return float(mass.mean().item())


def find_card_identity_token_positions(token_ids: torch.Tensor, tokenizer) -> list[int]:
    """Token positions (within the language block) whose decoded piece is a card rank/suit word --
    the only words that differ between this task's instructions."""
    positions = []
    for i, tid in enumerate(token_ids.tolist()):
        piece = tokenizer.convert_ids_to_tokens([tid])[0]
        cleaned = piece.replace("▁", "").replace("Ġ", "").strip(".,\n").lower()
        if cleaned in CARD_WORDS:
            positions.append(i)
    return positions


def trace_scene(policy, preprocessor, postprocessor, tokenizer, decoder, scene: dict, instructions: list[str]) -> dict:
    """NOTE on variable text length: `pad_language_to="longest"` (this checkpoint's config) pads
    to the longest sequence IN THE BATCH -- at batch_size=1 (used throughout this script) that
    means NO padding at all, so different instructions tokenize to DIFFERENT lengths (e.g. "7" vs
    "10" is 1 vs 2 sub-word pieces). Image/state regions are always fixed-length (unaffected by
    instruction text) so are kept as raw per-token stacks; the text/full regions are stored
    MEAN-POOLED so they remain fixed-size, comparable vectors across instructions regardless of
    this length difference -- same masked-mean-pooling convention `_pooled_latent` already uses
    per modality."""
    base = np.asarray(scene["robot_base_world"])
    records = []
    attn_capture = CrossAttentionCapture(decoder)
    try:
        for instruction in instructions:
            raw_batch = build_batch(scene, instruction)
            batch = preprocessor(raw_batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS][0].cpu()

            attn_capture.reset()
            with torch.no_grad():
                latent_tokens, latent_pad_mask, latent_modality_ids = policy._encode_multimodal_latent(batch)
                current_state = policy._encode_state(policy._current_state(batch))
                memory = decoder.latent_projection(latent_tokens)
                queries = (decoder.action_queries + decoder.horizon_position_embedding)[None].expand(1, -1, -1)
                queries = queries + decoder.state_encoder(current_state)[:, None, :]
                actions, _ = policy.plan_action_chunk(batch)
            actions_phys = postprocessor(actions.cpu()).numpy()[0]

            image_mask = latent_modality_ids[0] == IMAGE_MODALITY
            text_mask = latent_modality_ids[0] == TEXT_MODALITY
            state_mask = latent_modality_ids[0] == STATE_MODALITY
            text_start = int(text_mask.nonzero()[0, 0].item()) if text_mask.any() else 0
            text_n = int(text_mask.sum().item())

            card_local_positions = find_card_identity_token_positions(lang_tokens[:text_n], tokenizer)
            card_token_positions_global = [text_start + p for p in card_local_positions]

            is_open = actions_phys[:, GRIPPER_INDEX_RAW] > GRIPPER_THRESHOLD
            closed_idx = np.flatnonzero(~is_open)
            first_close_xyz_world = (actions_phys[closed_idx[0], :3] + base) if len(closed_idx) else None
            nearest_to_close = (
                nearest_card(first_close_xyz_world, scene["card_positions_world"]) if first_close_xyz_world is not None else (None, None)
            )

            latent_image = latent_tokens[0][image_mask].detach().cpu()
            latent_text = latent_tokens[0][text_mask].detach().cpu()
            latent_state = latent_tokens[0][state_mask].detach().cpu()
            memory_image = memory[0][image_mask].detach().cpu()
            memory_text = memory[0][text_mask].detach().cpu()
            memory_state = memory[0][state_mask].detach().cpu()

            # image-token diversity (stage 1 sanity check): mean pairwise cosine similarity among
            # image tokens -- low (<<1) means the representation is spatially diverse, not collapsed.
            with torch.no_grad():
                normed = torch.nn.functional.normalize(latent_image.float(), dim=-1)
                image_token_pairwise_cosine_mean = float((normed @ normed.T).fill_diagonal_(0).mean().item())

            records.append(
                {
                    "instruction": instruction,
                    "lang_token_ids": lang_tokens[:text_n].tolist(),
                    "card_identity_token_positions_in_text_block": card_local_positions,
                    # Fixed-length, raw per-token (image/state never vary in length with instruction):
                    "latent_image": latent_image,
                    "latent_state": latent_state,
                    "memory_image": memory_image,
                    "memory_state": memory_state,
                    # Variable-length text region -> mean-pooled to a fixed-size vector:
                    "latent_text_pooled": latent_text.mean(dim=0) if text_n else None,
                    "memory_text_pooled": memory_text.mean(dim=0) if text_n else None,
                    "latent_full_pooled": latent_tokens[0].detach().cpu().mean(dim=0),
                    "memory_full_pooled": memory[0].detach().cpu().mean(dim=0),
                    # The single card-identity token (first one, if multi-piece) -- fixed-size,
                    # directly comparable across instructions regardless of overall text length.
                    "card_identity_vector_latent": latent_text[card_local_positions[0]] if card_local_positions else None,
                    "card_identity_vector_memory": memory_text[card_local_positions[0]] if card_local_positions else None,
                    "query": queries[0].detach().cpu(),
                    "image_token_pairwise_cosine_mean": image_token_pairwise_cosine_mean,
                    "attn_weights_per_layer": list(attn_capture.per_layer_weights),  # [num_layers][1,heads,horizon,N]
                    "card_token_positions_global": card_token_positions_global,
                    "image_n": latent_image.shape[0],
                    "text_n": text_n,
                    "state_n": latent_state.shape[0],
                    "actions_phys": actions_phys,
                    "first_close_xyz_world": first_close_xyz_world,
                    "nearest_card_to_close": nearest_to_close[0],
                }
            )
    finally:
        attn_capture.remove()
    return records


def build_scene_report(scene: dict, records: list[dict]) -> dict:
    # Per-instruction attention-mass summary (averaged across layers/heads/horizon).
    for r in records:
        # Modality-id row matching attn weights' last dim ordering (image, text, state -- the same
        # concatenation order `compute_prefix_modality_ids` guarantees), rebuilt per-instruction
        # since text width varies (see `trace_scene`'s docstring).
        mids = torch.cat(
            [
                torch.full((r["image_n"],), IMAGE_MODALITY),
                torch.full((r["text_n"],), TEXT_MODALITY),
                torch.full((r["state_n"],), STATE_MODALITY),
            ]
        )[None]
        image_frac = float(np.mean([modality_mass(w, mids, IMAGE_MODALITY) for w in r["attn_weights_per_layer"]]))
        text_frac = float(np.mean([modality_mass(w, mids, TEXT_MODALITY) for w in r["attn_weights_per_layer"]]))
        state_frac = float(np.mean([modality_mass(w, mids, STATE_MODALITY) for w in r["attn_weights_per_layer"]]))
        card_frac = (
            float(np.mean([token_mass(w, r["card_token_positions_global"]) for w in r["attn_weights_per_layer"]]))
            if r["card_token_positions_global"]
            else 0.0
        )
        r["attention_mass"] = {"image": image_frac, "text": text_frac, "state": state_frac, "card_identity_tokens": card_frac}

    text_pairs = pairwise(records, lambda r: r["latent_text_pooled"])
    card_token_pairs = pairwise(records, lambda r: r["card_identity_vector_latent"])
    memory_full_pairs = pairwise(records, lambda r: r["memory_full_pooled"])
    memory_image_pairs = pairwise(records, lambda r: r["memory_image"])
    memory_text_pairs = pairwise(records, lambda r: r["memory_text_pooled"])
    memory_state_pairs = pairwise(records, lambda r: r["memory_state"])
    query_pairs = pairwise(records, lambda r: r["query"])
    traj_full_pairs = pairwise(records, lambda r: torch.from_numpy(r["actions_phys"][:, :3]))
    traj_first10_pairs = pairwise(records, lambda r: torch.from_numpy(r["actions_phys"][:10, :3]))

    nearest_cards = [r["nearest_card_to_close"] for r in records]
    # Only pairs where BOTH instructions actually produced a predicted gripper-close point are
    # "comparable" -- a `None`-vs-`None` pair (no close event in either instruction's first
    # open-loop chunk) is neither "same" nor "different", it's simply undefined, and must NOT be
    # counted as "changed" (an earlier version of this script did exactly that: `a == b` is True
    # for `None == None`, but the old code's `and nearest_cards[i] is not None` guard excluded
    # None-None pairs from the "same" count without excluding them from the denominator, making
    # every all-None scene register as spuriously "changed").
    comparable_pairs = [
        (nearest_cards[i], nearest_cards[j])
        for i in range(len(records))
        for j in range(i + 1, len(records))
        if nearest_cards[i] is not None and nearest_cards[j] is not None
    ]
    n_comparable = len(comparable_pairs)
    same_nearest = sum(1 for a, b in comparable_pairs if a == b)

    def mean_l2(pairs):
        return float(np.mean([p["l2"] for p in pairs])) if pairs else None

    text_changed = mean_l2(text_pairs)
    memory_changed = mean_l2(memory_full_pairs)
    mean_text_attn_mass = float(np.mean([r["attention_mass"]["text"] for r in records]))
    traj_changed = mean_l2(traj_full_pairs)
    # None = undefined (no instruction in this scene produced a predicted gripper-close point --
    # see note above), not "unchanged".
    nearest_card_changed = (same_nearest < n_comparable) if n_comparable > 0 else None

    return {
        "seed": scene["seed"],
        "true_target_entity": scene["target_entity"],
        "instructions": [r["instruction"] for r in records],
        "per_instruction": [
            {
                "instruction": r["instruction"],
                "card_identity_token_positions_global": r["card_token_positions_global"],
                "attention_mass": r["attention_mass"],
                "nearest_card_to_predicted_close": r["nearest_card_to_close"],
                "image_token_pairwise_cosine_mean": r["image_token_pairwise_cosine_mean"],
            }
            for r in records
        ],
        "stage1_visual": {
            "card_positions_world": scene["card_positions_world"],
            "mean_image_token_pairwise_cosine": float(np.mean([r["image_token_pairwise_cosine_mean"] for r in records])),
            "image_region_changed_across_instructions_l2": mean_l2(memory_image_pairs),
            "note": "mean_image_token_pairwise_cosine: low (<<1) = image tokens are spatially diverse (not collapsed to one repeated vector), a necessary condition for the representation to carry per-card position info at all -- NOT per-card localization (would need camera-projection geometry, out of scope here; card GT xyz is included above for external reference only). Whether the image REGION itself actually shifts with the instruction (it could, in principle, via the VLM's own bidirectional self-attention letting image tokens attend to text) is measured directly in stage3_memory.image_region_pairwise, not assumed here.",
        },
        "stage2_text": {"pairwise_l2_cosine": text_pairs, "card_identity_token_pairwise": card_token_pairs},
        "stage3_memory": {
            "full_pairwise": memory_full_pairs,
            "image_region_pairwise": memory_image_pairs,
            "text_region_pairwise": memory_text_pairs,
            "state_pairwise": memory_state_pairs,
        },
        "stage4_query": {"pairwise": query_pairs, "note": "canonical architecture (no subgoal/target conditioning): query = action_queries + horizon_position_embedding + state_encoder(current_state) -- structurally instruction-independent by construction (current_state carries no language)."},
        "stage5_attention": {
            "mean_text_attention_mass_across_instructions": mean_text_attn_mass,
            "mean_card_identity_token_attention_mass": float(np.mean([r["attention_mass"]["card_identity_tokens"] for r in records])),
            "per_instruction": [{"instruction": r["instruction"], **r["attention_mass"]} for r in records],
        },
        "stage6_trajectory": {
            "full_horizon_xyz_pairwise": traj_full_pairs,
            "first10_xyz_pairwise": traj_first10_pairs,
            "nearest_card_to_close_by_instruction": [{"instruction": r["instruction"], "nearest_card": r["nearest_card_to_close"]} for r in records],
            "nearest_card_changed_across_instructions": nearest_card_changed,
        },
        "stage7_sensitivity": {
            "text_representation_changed": {"yes": text_changed is not None and text_changed > 1e-4, "magnitude_l2": text_changed},
            "multimodal_memory_changed": {"yes": memory_changed is not None and memory_changed > 1e-4, "magnitude_l2": memory_changed},
            "decoder_attends_to_text": {"yes": mean_text_attn_mass > 0.01, "attention_mass": mean_text_attn_mass},
            "action_trajectory_changed": {"yes": traj_changed is not None and traj_changed > 1e-3, "magnitude_l2_m": traj_changed},
            "selected_nearest_card_changed": nearest_card_changed,
        },
    }


def classify(scene_report: dict) -> str:
    s = scene_report["stage7_sensitivity"]
    if not s["text_representation_changed"]["yes"]:
        return "B. text representation failure -- text tokens themselves barely differ across instructions"
    if not s["multimodal_memory_changed"]["yes"]:
        return "C. multimodal fusion failure -- text differs but the fused memory (post latent_projection) barely does"
    if not s["decoder_attends_to_text"]["yes"]:
        return "D. decoder ignores text despite text being present in memory (near-zero attention mass on text tokens)"
    if not s["action_trajectory_changed"]["yes"]:
        return "E. decoder attends to text but action head/trajectory output collapses regardless"
    card_changed = s["selected_nearest_card_changed"]
    if card_changed is None:
        return "not determinable -- no instruction in this scene produced a predicted gripper-close point within its first open-loop chunk (single cold-reset frame, not a rollout), so 'which card was selected' has no value to compare; stages 1-6 all show the trajectory itself responding to the instruction"
    if card_changed is False:
        return "E. decoder reacts (trajectory changes) but not enough to change which card is actually selected"
    return "no fixed-slot bias detected for this scene -- trajectory and card selection both respond to instruction"


def strip_tensors_for_json(obj):
    """Drop the large raw tensors (latent_tokens_full/image/text/state, memory_*, query,
    attn_weights_per_layer, actions_phys) before serializing -- only derived pairwise/summary
    stats go into the JSON result file, per the 'one short result file' scope."""
    if isinstance(obj, dict):
        return {k: strip_tensors_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_tensors_for_json(v) for v in obj]
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/eval/safediff_vla_forward_path_tracing/summary.json")
    args = parser.parse_args()

    logger.info("=== loading canonical checkpoint from %s (unmodified) ===", args.checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint).to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    decoder = policy.decoder

    scene_reports = []
    for seed in SCENE_SEEDS:
        logger.info("=== scene seed=%d ===", seed)
        scene = capture_scene(seed)
        true_target = scene["target_entity"]
        other_cards = [c for c in scene["card_positions_world"] if c != true_target]
        chosen_others = other_cards[: N_CARDS_PER_SCENE - 1]
        instructions = [f"primitive: Please pick the poker {pretty_card_name(true_target)}"] + [
            f"primitive: Please pick the poker {pretty_card_name(c)}" for c in chosen_others
        ]
        logger.info("  true_target=%s, instructions=%s", true_target, instructions)

        records = trace_scene(policy, preprocessor, postprocessor, tokenizer, decoder, scene, instructions)
        report = build_scene_report(scene, records)
        report["fixed_slot_bias_classification"] = classify(report)
        scene_reports.append(report)
        logger.info("  classification: %s", report["fixed_slot_bias_classification"])
        logger.info("  sensitivity: %s", json.dumps(report["stage7_sensitivity"], indent=2, default=str))

    summary = {
        "checkpoint": args.checkpoint,
        "task": TASK,
        "scenes": strip_tensors_for_json(scene_reports),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("=== wrote %s ===", out_path)


if __name__ == "__main__":
    main()
