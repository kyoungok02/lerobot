#!/usr/bin/env python
"""Generates a small COUNTERFACTUAL grounding dataset for the language-grounding pilot: for each
of N scenes (one `VLABenchEnv.reset()` each, zero `.step()` calls -- image/state frozen), records
one sample per real poker card present in that scene:

    same image, same robot state, card-specific instruction -> that card's simulator
    ground-truth xyz (robot-frame meters)

e.g. for one scene with 3 cards:
    image_001 + "primitive: Please pick the poker 7 of spades"  -> xyz(7_of_spades)
    image_001 + "primitive: Please pick the poker 10 of clubs"  -> xyz(10_of_clubs)
    image_001 + "primitive: Please pick the poker 5 of hearts"  -> xyz(5_of_hearts)

The point is the CONTRAST: same image/state, only the instruction changes, and the correct label
changes with it -- a training signal `select_poker`'s own demonstrations never provide (there,
instruction always matches the one target the scene was built around; no other card's xyz is ever
a valid label for the same image). This dataset does not touch, modify, or delete the real
`lerobot/vlabench_unified` demonstrations at all -- it's written to its own directory.

Train/held-out split is by SCENE (not by sample), so no scene's cards leak across the split.

Outputs:
    outputs/data/safediff_vla_counterfactual_grounding/scenes/scene_<seed>.pt
        -- one file per scene: the raw (unnormalized) observation dict, CPU tensors.
    outputs/data/safediff_vla_counterfactual_grounding/manifest.json
        -- {"scenes": [{"seed", "split", "n_cards", "cards": {name: xyz}}, ...],
            "samples": [{"seed", "instruction", "card", "label_xyz", "split"}, ...]}

Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/generate_counterfactual_grounding_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.envs import make_env
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

TASK = "select_poker"
FIRST_SEED = 2000  # fresh range, distinct from every prior probe/rollout script's 1000-1009/1000-1004
MAX_CARDS_PER_SCENE = 4
MIN_CARDS_PER_SCENE = 2
HELD_OUT_FRAC = 0.2
SPLIT_SEED = 1000


def pretty_identity(identity: str) -> str:
    rank, _, suit = identity.partition("_of_")
    return f"{rank} of {suit}"


def get_scene_cards(env_impl: VLABenchEnvImpl) -> dict[str, np.ndarray]:
    physics = env_impl._env.physics
    task = env_impl._env.task
    base = env_impl._robot_base_xyz if env_impl._robot_base_xyz is not None else np.zeros(3, dtype=float)
    cards: dict[str, np.ndarray] = {}
    for name, ent in task.entities.items():
        if type(ent).__name__ != "Poker":
            continue
        pos_world = np.asarray(ent.get_xpos(physics), dtype=float)
        cards[name] = pos_world - base
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-scenes", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--output-dir", default="outputs/data/safediff_vla_counterfactual_grounding")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    scenes_dir = output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    env_impl: VLABenchEnvImpl = env.envs[0]

    scenes: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    n_written = 0
    started = time.time()
    try:
        seed = args.first_seed
        while n_written < args.n_scenes:
            t0 = time.time()
            observation, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
            cards = get_scene_cards(env_impl)
            seed += 1
            if len(cards) < MIN_CARDS_PER_SCENE:
                continue

            names = sorted(cards.keys())[:MAX_CARDS_PER_SCENE]
            obs_t = preprocess_observation(observation)
            obs_t = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs_t.items()}
            torch.save(obs_t, scenes_dir / f"scene_{seed - 1}.pt")

            scenes.append(
                {
                    "seed": seed - 1,
                    "n_cards": len(names),
                    "cards": {name: cards[name].tolist() for name in names},
                }
            )
            for name in names:
                samples.append(
                    {
                        "seed": seed - 1,
                        "card": name,
                        "instruction": f"primitive: Please pick the poker {pretty_identity(name)}",
                        "label_xyz": cards[name].tolist(),
                    }
                )
            n_written += 1
            if n_written % 20 == 0 or n_written == 1:
                elapsed = time.time() - started
                logger.info(
                    "  scene %d/%d (seed=%d, %d cards, %.2fs/scene, %.1fs elapsed)",
                    n_written, args.n_scenes, seed - 1, len(names), time.time() - t0, elapsed,
                )
    finally:
        env.close()

    # Scene-level train/held-out split -- every sample from a held-out scene stays held-out, so no
    # card from a scene the training loop has seen leaks into evaluation.
    rng = random.Random(SPLIT_SEED)
    scene_seeds = [s["seed"] for s in scenes]
    rng.shuffle(scene_seeds)
    n_held_out = max(1, int(round(len(scene_seeds) * HELD_OUT_FRAC)))
    held_out_seeds = set(scene_seeds[:n_held_out])
    for s in scenes:
        s["split"] = "held_out" if s["seed"] in held_out_seeds else "train"
    for sample in samples:
        sample["split"] = "held_out" if sample["seed"] in held_out_seeds else "train"

    manifest = {
        "task": TASK,
        "n_scenes": len(scenes),
        "n_samples": len(samples),
        "n_train_scenes": len(scenes) - n_held_out,
        "n_held_out_scenes": n_held_out,
        "scenes": scenes,
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "wrote %d scenes / %d samples (%d train scenes, %d held-out scenes) to %s",
        len(scenes), len(samples), len(scenes) - n_held_out, n_held_out, output_dir,
    )


if __name__ == "__main__":
    main()
