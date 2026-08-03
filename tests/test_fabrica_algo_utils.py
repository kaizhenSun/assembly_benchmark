# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ALGO_UTILS_PATH = (
    REPO_ROOT
    / "source"
    / "assembly_benchmark"
    / "assembly_benchmark"
    / "tasks"
    / "direct"
    / "fabrica"
    / "fabrica_algo_utils.py"
)
SPEC = importlib.util.spec_from_file_location("fabrica_algo_utils", ALGO_UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
ALGO_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ALGO_UTILS)

build_asymmetric_observations = ALGO_UTILS.build_asymmetric_observations
dense_insertion_reward = ALGO_UTILS.dense_insertion_reward
do_deltapos_path_transform = ALGO_UTILS.do_deltapos_path_transform
episode_timeout_mask = ALGO_UTILS.episode_timeout_mask
preprocess_fabrica_actions = ALGO_UTILS.preprocess_fabrica_actions
project_points_to_paths = ALGO_UTILS.project_points_to_paths
undo_deltapos_path_transform = ALGO_UTILS.undo_deltapos_path_transform


def test_path_transform_round_trip_and_original_action_semantics() -> None:
    goal = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    preinsert = torch.tensor([[0.0, 0.0, 0.0202], [1.0202, 2.0, 3.0]])
    delta = torch.tensor([[0.002, -0.003, 0.004], [-0.001, 0.005, 0.002]])
    transformed = do_deltapos_path_transform(delta, goal, preinsert)
    assert undo_deltapos_path_transform(transformed, goal, preinsert) == pytest.approx(delta)

    plug = preinsert.clone()
    path_scale = torch.tensor([1.01, 1.01])
    step = preprocess_fabrica_actions(
        torch.zeros_like(plug), plug, goal, goal, preinsert, path_scale, position_action_scale=0.005
    )
    expected = (goal - plug) / torch.linalg.vector_norm(goal - plug, dim=-1, keepdim=True) * 0.005
    assert step == pytest.approx(expected)

    at_goal = preprocess_fabrica_actions(
        torch.zeros_like(goal), goal, goal, goal, preinsert, path_scale, position_action_scale=0.005
    )
    assert at_goal == pytest.approx(torch.zeros_like(goal))


def test_actor_critic_reward_timeout_and_polyline_contracts() -> None:
    nominal = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = nominal + 0.25
    actor, critic = build_asymmetric_observations(nominal, actual)
    assert actor.shape == (2, 3)
    assert critic.shape == (2, 6)
    assert critic == pytest.approx(torch.cat((nominal, actual), dim=-1))

    reward, distance = dense_insertion_reward(torch.tensor([[0.003, 0.004, 0.0], [0.1, 0.0, 0.0]]), 1000.0, 0.03)
    assert distance == pytest.approx(torch.tensor([0.005, 0.1]))
    assert reward == pytest.approx(torch.tensor([-5.0, -30.0]))
    assert episode_timeout_mask(torch.tensor([126, 127, 128, 129]), 128).tolist() == [False, False, True, True]

    paths = torch.tensor(
        [
            [[0.0, 0.0, 0.02], [0.0, 0.0, 0.01], [0.0, 0.0, 0.0]],
            [[0.02, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    closest, progress = project_points_to_paths(torch.tensor([[0.003, 0.0, 0.015], [0.015, 0.004, 0.0]]), paths)
    assert closest == pytest.approx(torch.tensor([[0.0, 0.0, 0.015], [0.015, 0.0, 0.0]]))
    assert progress == pytest.approx(torch.tensor([0.25, 0.25]))
