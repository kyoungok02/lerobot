"""Regression tests for `VLABenchEnv`'s seed plumbing (`reset(seed=...)`) and per-episode
instruction extraction (`task_description`).

The seed-reproducibility tests build and reset the actual VLABench/MuJoCo simulator (no mocking)
because the bug being guarded against -- `env.reset(seed=X)` silently not affecting the sampled
scene at all -- can only be caught by observing genuine simulator state, not a stand-in. Skipped
when VLABench isn't installed (it's a heavy optional dependency, not part of the base test extras).

The `_resolve_task_description` unit tests run without VLABench installed: that method takes a
plain `task_obj` argument and never touches the simulator, so its suite-prefix / fallback logic can
be verified with lightweight fakes -- keeping instruction-contract coverage in the base test suite
rather than only in the (usually skipped) simulator-backed tests below.
"""

import numpy as np
import pytest

from lerobot.envs import make_env
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from tests.utils import skip_if_package_missing

TASK = "select_poker"


def _build_env():
    envs = make_env(VLABenchEnv(task=TASK), n_envs=1, use_async_envs=False)
    return envs[TASK][0]


class _FakeTaskObj:
    """Stand-in for a VLABench `LM4ManipBaseTask` instance -- only exposes what
    `_resolve_task_description` reads, mirroring real VLABench task classes where neither
    `task_description` nor `language_instruction` is ever defined (verified against the installed
    VLABench package) and `get_instruction()` is the real per-episode instruction accessor."""

    def __init__(self, instruction: str | None = "Please pick the poker 7 of spades"):
        self._instruction = instruction

    def get_instruction(self):
        return self._instruction


class TestResolveTaskDescriptionUnit:
    """No simulator required: `_resolve_task_description(task_obj)` is pure logic over whatever
    `task_obj` exposes."""

    def test_primitive_suite_task_gets_primitive_prefix(self):
        env = VLABenchEnvImpl(task="select_poker")
        result = env._resolve_task_description(_FakeTaskObj("Please pick the poker 7 of spades"))
        assert result == "primitive: Please pick the poker 7 of spades"

    def test_composite_suite_task_gets_composite_prefix(self):
        env = VLABenchEnvImpl(task="cluster_book")
        result = env._resolve_task_description(_FakeTaskObj("Cluster the objects into two classes."))
        assert result == "composite: Cluster the objects into two classes."

    def test_never_falls_back_to_generic_task_name_when_instruction_available(self):
        env = VLABenchEnvImpl(task="select_poker")
        result = env._resolve_task_description(_FakeTaskObj("Please pick the poker 2 of hearts"))
        assert result != "select_poker"
        assert result != env.task

    def test_different_instructions_produce_different_task_descriptions(self):
        env = VLABenchEnvImpl(task="select_poker")
        a = env._resolve_task_description(_FakeTaskObj("Please pick the poker 7 of spades"))
        b = env._resolve_task_description(_FakeTaskObj("Please pick the poker 2 of hearts"))
        assert a != b

    def test_falls_back_to_task_name_when_no_instruction_available(self):
        """A task_obj exposing no `get_instruction` (and no `task_description`/`language_instruction`)
        falls back to the generic task name -- the only case where that fallback is correct."""
        env = VLABenchEnvImpl(task="select_poker")
        result = env._resolve_task_description(object())
        assert result == "select_poker"

    def test_falls_back_to_task_name_when_get_instruction_returns_none(self):
        env = VLABenchEnvImpl(task="select_poker")
        result = env._resolve_task_description(_FakeTaskObj(None))
        assert result == "select_poker"

    def test_task_description_attr_takes_priority_over_get_instruction(self):
        """Forward-compatibility path: if a future/custom task class *does* define
        `task_description`, it should win over `get_instruction()` -- not silently ignored."""

        class _WithTaskDescription(_FakeTaskObj):
            task_description = "Please pick the poker ace of clubs"

        env = VLABenchEnvImpl(task="select_poker")
        result = env._resolve_task_description(_WithTaskDescription("Please pick the poker 7 of spades"))
        assert result == "primitive: Please pick the poker ace of clubs"

    def test_unknown_task_name_gets_no_suite_prefix(self):
        """A task not present in either SUITE_TASKS list (e.g. a custom/unregistered task) should
        pass the raw instruction through unprefixed rather than guessing a suite."""
        env = VLABenchEnvImpl(task="some_custom_unlisted_task")
        result = env._resolve_task_description(_FakeTaskObj("Do the custom thing."))
        assert result == "Do the custom thing."


@skip_if_package_missing("VLABench")
def test_same_seed_reset_reproduces_initial_scene():
    """Resetting the same live env twice with the same seed must reproduce the same initial
    object layout/target and the same rendered frame -- the ability to replay a specific eval
    episode depends on this."""
    env = _build_env()
    raw_env = env.envs[0]
    try:
        obs_a, _ = env.reset(seed=[1000])
        frame_a = raw_env.render().copy()
        target_a = raw_env._env.task.target_entity

        obs_b, _ = env.reset(seed=[1000])
        frame_b = raw_env.render().copy()
        target_b = raw_env._env.task.target_entity

        np.testing.assert_array_equal(obs_a["agent_pos"], obs_b["agent_pos"])
        np.testing.assert_array_equal(frame_a, frame_b)
        assert target_a == target_b
    finally:
        env.close()


@skip_if_package_missing("VLABench")
def test_different_seed_reset_changes_scene():
    """Different seeds must produce a genuinely different sampled scene -- at least the
    rendered frame and the sampled target entity, both of which the previous (no-op)
    `_seed_inner_env` implementation left completely unaffected by `seed`."""
    env = _build_env()
    raw_env = env.envs[0]
    try:
        env.reset(seed=[1000])
        frame_1000 = raw_env.render().copy()
        target_1000 = raw_env._env.task.target_entity

        env.reset(seed=[1001])
        frame_1001 = raw_env.render().copy()
        target_1001 = raw_env._env.task.target_entity

        assert not np.array_equal(frame_1000, frame_1001), (
            "rendered frame is identical across different seeds -- scene randomization is not "
            "actually seed-controlled"
        )
        assert target_1000 != target_1001, (
            "sampled target entity is identical across different seeds -- scene randomization is "
            "not actually seed-controlled"
        )
    finally:
        env.close()


@skip_if_package_missing("VLABench")
def test_select_poker_task_description_contains_target_card_identity():
    """The eval-harness instruction fix's core requirement: `task_description` must name the real
    target card (e.g. "7 of spades"), not just the generic task name."""
    env = _build_env()
    raw_env = env.envs[0]
    try:
        env.reset(seed=[1000])
        target_entity = raw_env._env.task.target_entity  # e.g. "7_of_spades"
        card_phrase = target_entity.replace("_", " ")  # "7 of spades"

        task_description = list(env.call("task_description"))[0]

        assert card_phrase in task_description, (
            f"task_description {task_description!r} does not mention the real target card "
            f"{card_phrase!r}"
        )
    finally:
        env.close()


@skip_if_package_missing("VLABench")
def test_select_poker_task_description_does_not_fall_back_to_generic_task_name():
    """Regression guard for the exact bug found by the instruction-grounding audit: every episode
    silently reporting the bare task name `"select_poker"` instead of a real per-episode
    instruction."""
    env = _build_env()
    try:
        env.reset(seed=[1000])
        task_description = list(env.call("task_description"))[0]
        assert task_description != "select_poker"
    finally:
        env.close()


@skip_if_package_missing("VLABench")
def test_select_poker_task_description_differs_across_target_episodes():
    """Different episodes with different sampled target cards must produce different instruction
    strings -- otherwise the language channel still carries no card-identity information, even if
    it's no longer the literal generic task name."""
    env = _build_env()
    raw_env = env.envs[0]
    try:
        env.reset(seed=[1000])
        target_1000 = raw_env._env.task.target_entity
        task_description_1000 = list(env.call("task_description"))[0]

        env.reset(seed=[1001])
        target_1001 = raw_env._env.task.target_entity
        task_description_1001 = list(env.call("task_description"))[0]

        assert target_1000 != target_1001, "test fixture assumption broken: seeds sampled the same target"
        assert task_description_1000 != task_description_1001
    finally:
        env.close()


@skip_if_package_missing("VLABench")
@pytest.mark.parametrize("task", ["select_fruit", "cluster_book"])
def test_other_vlabench_tasks_still_produce_a_real_instruction(task):
    """No-regression check: the instruction fix generalizes from `select_poker` to VLABench tasks
    with a different `get_instruction()` phrasing/suite (primitive vs. composite), rather than
    special-casing poker cards."""
    envs = make_env(VLABenchEnv(task=task), n_envs=1, use_async_envs=False)
    env = envs[task][0]
    try:
        env.reset(seed=[1000])
        task_description = list(env.call("task_description"))[0]
        assert task_description != task
        assert task_description.startswith(("primitive: ", "composite: "))
    finally:
        env.close()
