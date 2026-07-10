# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch
from assembly_benchmark.sensors.vt_refine_tactile import (
    DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_DAMPING,
    DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_STIFFNESS,
    DEFAULT_TABLE_TACTILE_NORMAL_AXIS,
    DEFAULT_TABLE_TACTILE_PAD_SIZE,
    DEFAULT_TABLE_TACTILE_POINT_DISTANCE,
    generate_table_tactile_points_local,
    normalize_tactile_normal_force,
    resolve_table_tactile_contact_part_names,
    tactile_force_grid,
    tactile_observation_size,
    tactile_point_count,
)

ISAAC_SIM_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("carb") is not None and importlib.util.find_spec("pxr") is not None
)


def test_table_tactile_shape_helpers() -> None:
    assert tactile_point_count(DEFAULT_TABLE_TACTILE_ARRAY_SIZE) == 384
    assert tactile_observation_size(DEFAULT_TABLE_TACTILE_ARRAY_SIZE) == 1536

    with pytest.raises(ValueError, match="two entries"):
        tactile_point_count((12,))
    with pytest.raises(ValueError, match="positive"):
        tactile_point_count((0, 32))


def test_resolve_table_tactile_contact_part_names() -> None:
    all_parts = ("base_tag", "square_table_top", "square_table_leg4")
    default_parts = ("square_table_top", "square_table_leg4")

    assert resolve_table_tactile_contact_part_names((), default_parts, all_parts) == default_parts
    assert resolve_table_tactile_contact_part_names(("square_table_leg4",), default_parts, all_parts) == (
        "square_table_leg4",
    )
    with pytest.raises(ValueError, match="Unknown table tactile contact part"):
        resolve_table_tactile_contact_part_names(("missing",), default_parts, all_parts)


def test_normalize_tactile_normal_force() -> None:
    force = torch.tensor([[0.0, -1.0, 0.0004], [0.0, 0.0001, 0.0002]])
    normalized = normalize_tactile_normal_force(force)

    assert torch.all(normalized >= 0.0)
    assert torch.isclose(normalized[0, 2], torch.tensor(1.0))
    assert torch.isclose(normalized[1, 1], torch.tensor(0.5))
    assert torch.isclose(normalized[1, 2], torch.tensor(1.0))


def test_normalize_tactile_normal_force_sanitizes_non_finite_values() -> None:
    force = torch.tensor([[float("nan"), float("inf"), float("-inf"), 0.0004]])

    normalized = normalize_tactile_normal_force(force)

    assert torch.equal(normalized, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_tactile_force_grid_reshapes_and_sanitizes_force_values() -> None:
    tactile_points = torch.zeros((384, 4))
    tactile_points[:, 3] = torch.arange(384, dtype=torch.float32)
    tactile_points[0, 3] = float("nan")
    tactile_points[1, 3] = float("inf")
    tactile_points[2, 3] = -1.0

    grid = tactile_force_grid(tactile_points)

    assert grid.shape == (12, 32)
    assert grid[0, 0] == 0.0
    assert grid[0, 1] == 0.0
    assert grid[0, 2] == 0.0
    assert grid[-1, -1] == 383.0


def test_tactile_force_grid_supports_batch_and_normalized_clamp() -> None:
    tactile_points = torch.zeros((2, 384, 4))
    tactile_points[0, :, 3] = 0.5
    tactile_points[1, :, 3] = 1.5
    tactile_points[1, 0, 3] = float("inf")

    grid = tactile_force_grid(tactile_points, clamp_max=1.0)

    assert grid.shape == (2, 12, 32)
    assert torch.allclose(grid[0], torch.full((12, 32), 0.5))
    assert torch.allclose(grid[1], torch.ones((12, 32)))


def test_tactile_force_grid_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="Expected tactile points shape"):
        tactile_force_grid(torch.zeros((383, 4)))


def test_table_tactile_points_cover_configured_pad_surface() -> None:
    points = generate_table_tactile_points_local()

    assert points.shape == (384, 3)
    assert torch.isclose(points[:, 0].min(), torch.tensor(-DEFAULT_TABLE_TACTILE_POINT_DISTANCE * 5.5))
    assert torch.isclose(points[:, 0].max(), torch.tensor(DEFAULT_TABLE_TACTILE_POINT_DISTANCE * 5.5))
    assert torch.isclose(points[:, 1].min(), torch.tensor(-DEFAULT_TABLE_TACTILE_POINT_DISTANCE * 15.5))
    assert torch.isclose(points[:, 1].max(), torch.tensor(DEFAULT_TABLE_TACTILE_POINT_DISTANCE * 15.5))
    assert torch.allclose(points[:, 2], torch.full((384,), DEFAULT_TABLE_TACTILE_PAD_SIZE[2] / 2.0))


def test_table_tactile_downward_compression_axis_is_positive() -> None:
    compression_force = torch.tensor([0.0, 0.0, -0.002])

    assert torch.dot(compression_force, torch.tensor(DEFAULT_TABLE_TACTILE_NORMAL_AXIS)) > 0.0


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_table_tactile_env_cfg_injection() -> None:
    env_cfg_module = importlib.import_module(
        "assembly_benchmark.tasks.direct.assembly_benchmark.assembly_benchmark_env_cfg"
    )
    sensor_module = importlib.import_module("assembly_benchmark.sensors.vt_refine_tactile")
    cfg_cls = getattr(env_cfg_module, env_cfg_module.assembly_env_cfg_class_name("one_leg"))
    cfg = cfg_cls()
    observation_space = cfg.observation_space

    cfg.enable_table_tactile = True
    cfg.table_tactile_contact_part_names = ("square_table_leg4",)

    contact_part_names = sensor_module.configure_table_tactile_scene_cfg(cfg)

    assert contact_part_names == ("square_table_leg4",)
    assert "table_tactile_pad" in cfg.scene.__dict__
    assert "table_tactile_sensor" in cfg.scene.__dict__
    assert cfg.observation_space == observation_space
    physics_material = cfg.scene.table_tactile_pad.spawn.physics_material
    assert physics_material.compliant_contact_stiffness == DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_STIFFNESS
    assert physics_material.compliant_contact_damping == DEFAULT_TABLE_TACTILE_COMPLIANT_CONTACT_DAMPING
