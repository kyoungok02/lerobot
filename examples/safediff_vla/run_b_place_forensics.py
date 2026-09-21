#!/usr/bin/env python
"""Runs A's `place_phase_forensics.py` (unmodified) against B's own fresh sin/cos+padfix 20k
checkpoint.

Does NOT edit `place_phase_forensics.py`. Working around one bug found in it: `main()` loads the
policy from `args.checkpoint` correctly, but `run()` builds the pre/post-processors from the
*module-level* `CHECKPOINT` constant (A's own checkpoint path,
`outputs/train/safediff_vla_temporal_decoder_sincos_20k_padfix/...`) instead of the path actually
passed on the CLI -- so `--checkpoint <other path>` silently loads the right policy weights but the
wrong (or, here, nonexistent-on-this-machine) normalization stats. Same fix pattern already used
elsewhere in this directory for cross-checkpoint reuse (e.g. `compare_euler_branch_checkpoints.py`
setting `eba.CHECKPOINT`): import the module and set its `CHECKPOINT` attribute before calling
`run()`, rather than editing the file.

Usage:
    uv run python examples/safediff_vla/run_b_place_forensics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import place_phase_forensics as ppf  # noqa: E402

from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

B_CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_padfix_20k/checkpoints/020000/pretrained_model"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=B_CHECKPOINT)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_b_padfix20k_place_forensics")
    args = parser.parse_args()

    ppf.CHECKPOINT = args.checkpoint  # the actual fix -- see module docstring

    ppf.logger.info("=== loading B checkpoint from %s (unmodified) ===", args.checkpoint)
    policy = SafeDiffVLAPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"

    ppf.run(policy, args.device, args.n_episodes, args.start_seed, Path(args.output_dir))


if __name__ == "__main__":
    main()
