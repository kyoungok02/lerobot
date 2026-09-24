# SafeDiff-VLA grounded-grasp experiment: frozen baseline summary

**Status: frozen.** This is the official baseline as of commit `1f2bfd427f9af4be3253d54b430dff13de96dca5`
on `experimental-grounded-grasp`. No further training, no further expansion of the
grounded-grasp / pooling / instruction-conditioning line of experiments happens under this
summary — it documents what's already been run and concluded. See "Status and next steps" at the
bottom for what happens next.

## Official baseline

**Canonical `temporal_decoder` (sincos+padfix, 20k steps), evaluated with the corrected
instruction, is the official baseline going forward.**

- Checkpoint: `outputs/train/safediff_vla_temporal_decoder_sincos_20k_padfix/checkpoints/020000/pretrained_model`
- Eval: `outputs/eval/sincos_20k_padfix_instruction_fixed/` (`summary.json` + `episodes_raw.json`,
  select_poker seeds 1000-1009, exact per-episode instruction string included)
- Instruction fix: commit `1f2bfd427` (`src/lerobot/envs/vlabench.py::VLABenchEnv._resolve_task_description`,
  `src/lerobot/scripts/lerobot_eval.py` provenance logging, `tests/envs/test_vlabench.py`)

**`temporal_decoder_grounded_grasp` (modality-aware `TargetPointHead` pooling, 20k steps) is
preserved as an ablation / negative closed-loop result** — see "Key finding" below. Its checkpoint
and both eval traces (buggy- and fixed-instruction) are kept for exactly this reason.

- Checkpoint: `outputs/train/safediff_vla_temporal_decoder_grounded_grasp_modality_pool_20k/checkpoints/020000/pretrained_model`
- Eval (corrected instruction): `outputs/eval/modality_pool_20k_instruction_fixed/`

## Reproduction

```bash
# Instruction-fix regression tests (needs a VLABench-capable venv, e.g. a separate Python 3.12
# venv per docs/source/vlabench.mdx -- the base .venv is Python 3.13, incompatible with
# dm_control's labmaze dependency, which has no cp313 wheel).
MUJOCO_GL=egl <vlabench-venv>/bin/python -m pytest tests/envs/test_vlabench.py -v

# Canonical baseline, corrected instruction, seeds 1000-1009:
MUJOCO_GL=egl <vlabench-venv>/bin/python examples/safediff_vla/eval_canonical_corrected_instruction.py \
  --checkpoint outputs/train/safediff_vla_temporal_decoder_sincos_20k_padfix/checkpoints/020000/pretrained_model \
  --n-episodes 10 --start-seed 1000 \
  --output-dir outputs/eval/sincos_20k_padfix_instruction_fixed

# Modality-pool 20k, corrected instruction, seeds 1000-1009:
MUJOCO_GL=egl <vlabench-venv>/bin/python examples/safediff_vla/eval_canonical_corrected_instruction.py \
  --checkpoint outputs/train/safediff_vla_temporal_decoder_grounded_grasp_modality_pool_20k/checkpoints/020000/pretrained_model \
  --n-episodes 10 --start-seed 1000 \
  --output-dir outputs/eval/modality_pool_20k_instruction_fixed

# Held-out TargetPointHead xyz L2 (5k and 20k modality-pool checkpoints), no simulator needed:
uv run python examples/safediff_vla/eval_target_point_head_pooling_fix.py --skip-instruction-swap-probe \
  --held-out-checkpoints outputs/train/safediff_vla_temporal_decoder_grounded_grasp_modality_pool_20k/checkpoints/020000/pretrained_model \
  --max-samples 4000 --output-dir outputs/eval/safediff_vla_pooling_fix_20k
```

Dataset: `lerobot/vlabench_unified`, `eval_split=0.1` (deterministic, same held-out 1254 episodes /
348476 frames both runs — see `lerobot.datasets.factory.make_train_eval_datasets`). Task: `select_poker`.
Seeds 1000-1009 for every closed-loop rollout in this document.

---

## 1. Canonical: buggy-instruction vs fixed-instruction

Same checkpoint (`sincos_20k_padfix`), same seeds. "Buggy" = preserved pre-fix rollout traces
(`outputs/eval/safediff_vla_place_forensics/episodes_raw.json` + `outputs/eval/sincos_20k_padfix/report.json`,
generic `"select_poker"` instruction on every step). "Fixed" = fresh rollout this session with the
real per-episode instruction (e.g. `"primitive: Please pick the poker 7 of spades"`).

| metric | buggy instruction | fixed instruction |
|---|---|---|
| success | 1/10 | 1/10 |
| correct-target grasp | 1/10 | 2/10 |
| wrong-object grasp | 1/10 | 0/10 |
| mean min EE↔correct-card dist | 0.1435 m | 0.1389 m |

**Caveat:** the "buggy" side is an *older preserved* run (different session), not a fresh rollout
under identical code/environment apart from the instruction — so this comparison can't rule out
unrelated environment drift as a partial cause of the difference. Treat the direction (small,
positive) as suggestive, not a tightly controlled ablation.

## 2. Modality-pool: 5k vs 20k training steps

Held-out `TargetPointHead` xyz L2 (1500 valid-transition frames, `eval_split=0.1`, offline,
no simulator):

| | 5k | 20k |
|---|---|---|
| mean L2 | 0.0635 m | 0.0471 m |
| median L2 | 0.0494 m | 0.0347 m |
| mean \|dx\| / \|dy\| / \|dz\| | 0.044 / 0.033 / 0.012 m | 0.031 / 0.026 / 0.008 m |

Closed-loop (buggy-instruction rollouts — both 5k and 20k were only ever eval'd closed-loop
*before* the instruction fix; not rerun with the fixed instruction, per "no further expansion" —
20k's fixed-instruction closed-loop numbers are in table 3 below):

| | 5k | 20k |
|---|---|---|
| success | 1/10 | 1/10 |
| correct-target grasp | 1/10 | 1/10 |
| wrong-object grasp | 1/10 | 3/10 |
| mean min EE↔correct-card dist | 0.1322 m | 0.1184 m |

Instruction-swap probe (3 live scenes, same image, only instruction text varies — see
`outputs/eval/safediff_vla_instruction_grounding_audit_{5k,20k}/summary.json`):

| | 5k | 20k |
|---|---|---|
| nearest-card identity change rate across instructions | 17% (0/50/0%) | 0% |
| correct-instructed-card rate | 40% | 30% |

**Key finding (negative result):** 20k steps improved the offline `TargetPointHead` regression
substantially (−26% mean L2) but did **not** improve instruction-grounded closed-loop behavior —
wrong-object grasp roughly tripled, correct-instructed-card rate went *down*, and nearest-card
identity became **completely instruction-invariant** at 20k (0% change rate across all 3 probed
scenes, down from an already-weak 17% at 5k). More training made the position regression fit
better without making the model actually read the instruction to disambiguate between present
objects.

**Additional, stronger version of the same finding, found while assembling this summary:**
comparing the modality-pool 20k checkpoint's buggy-instruction vs fixed-instruction closed-loop
rollouts (both run this session, same code, same seeds, same checkpoint — a genuinely controlled
comparison unlike table 1's canonical comparison) — **the per-episode EE↔card distance trajectory
is bit-for-bit identical across all 10 episodes.** Changing the instruction text from the generic
task name to the real target-card instruction produced *zero* measurable change in this
checkpoint's physical behavior. This is direct, controlled evidence that language conditioning
is not reaching this checkpoint's pose-trajectory output at all, not just weakly.

## 3. Corrected-instruction canonical vs modality-pool 20k

Same seeds (1000-1009), same fixed-instruction code path, both checkpoints frozen backbone,
`action_horizon=execute_horizon=50`, no temporal ensembling, no reactive close, no target
conditioning (`grounded_grasp_condition_decoder_on_target=False` for modality-pool).

| metric | canonical (sincos+padfix 20k) | modality-pool 20k |
|---|---|---|
| success | 1/10 | 1/10 |
| correct-target approach | 4/10 (40%) | 4/10 (40%) |
| correct-target grasp | 2/10 | 1/10 |
| wrong-object grasp | 0/10 | 3/10 |
| mean min EE↔correct-card dist | 0.1389 m | 0.1184 m |

**Conclusion: no closed-loop advantage for `TargetPointHead`/modality-aware pooling over the plain
canonical decoder, once the instruction bug is fixed.** Canonical wins on grasp correctness and
wrong-object avoidance; modality-pool only wins on mean approach distance (and per finding above,
that number doesn't even depend on the instruction for this checkpoint). The pooling fix and 20k
scaling improved a proxy metric (offline xyz regression) without improving the behavior that proxy
was meant to stand in for.

---

## Status and next steps

- **A (this line of work): frozen.** No new training, no new grounded-grasp/pooling/
  instruction-conditioning experiments starting from here.
- **B (counterfactual-grounding experiment): separate, in progress elsewhere.** If it succeeds,
  only that commit gets pulled into this branch for canonical-integration / closed-loop
  validation prep — not a continuation of the pooling/TargetPointHead line above.
- Preserved, do-not-delete: final checkpoints (`sincos_20k_padfix`, `grounded_grasp_modality_pool_5k`,
  `grounded_grasp_modality_pool_20k`, `frac30_150k`), the corrected-instruction eval traces
  (`sincos_20k_padfix_instruction_fixed`, `modality_pool_20k_instruction_fixed`), the preserved
  pre-fix baseline traces (`safediff_vla_place_forensics`, `sincos_20k_padfix`) referenced by
  `eval_canonical_corrected_instruction.py`'s buggy-vs-fixed comparison, and the held-out/
  instruction-swap sources cited in this document (`safediff_vla_pooling_fix{,_20k}`,
  `safediff_vla_instruction_grounding_audit_{5k,20k}`, `safediff_vla_grounded_grasp_modality_pool_{5k_closedloop,20k}`).
