# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import assembly_benchmark.sensors as sensors_package
import assembly_benchmark.sensors.vt_refine_tactile as tactile_module
import pytest
import torch
from assembly_benchmark.sensors.vt_refine_tactile import (
    DEFAULT_TABLE_TACTILE_ARRAY_SIZE,
    DEFAULT_TABLE_TACTILE_POINT_DISTANCE,
    R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE,
    R1_PRO_GRIPPER_TACTILE_PAD_SPECS,
    R1ProGripperTactilePadSpec,
    generate_tactile_points_on_rectangular_pad_from_bounds,
    get_r1_pro_gripper_tactile_points,
    normalize_tactile_normal_force,
    tactile_observation_size,
    tactile_point_count,
)

EXPECTED_PAD_IDENTITIES = (
    (
        "right_pad1",
        "right_gripper_tactile_sensor1",
        "right_gripper_tactile_pad1",
        "right_gripper_finger_link1",
        "r1_pro_gripper_finger_link1_flat_pad.STL",
        -1.0,
        (0.0, 1.0, 0.0),
        (0, 0),
    ),
    (
        "right_pad2",
        "right_gripper_tactile_sensor2",
        "right_gripper_tactile_pad2",
        "right_gripper_finger_link2",
        "r1_pro_gripper_finger_link2_flat_pad.STL",
        1.0,
        (0.0, -1.0, 0.0),
        (1, 0),
    ),
    (
        "left_pad1",
        "left_gripper_tactile_sensor1",
        "left_gripper_tactile_pad1",
        "left_gripper_finger_link1",
        "r1_pro_gripper_finger_link1_flat_pad.STL",
        -1.0,
        (0.0, 1.0, 0.0),
        (0, 1),
    ),
    (
        "left_pad2",
        "left_gripper_tactile_sensor2",
        "left_gripper_tactile_pad2",
        "left_gripper_finger_link2",
        "r1_pro_gripper_finger_link2_flat_pad.STL",
        1.0,
        (0.0, -1.0, 0.0),
        (1, 1),
    ),
)


def test_sensor_package_exports_only_curated_runtime_interfaces() -> None:
    assert set(sensors_package.__all__) == {
        "DEFAULT_TABLE_TACTILE_ARRAY_SIZE",
        "DEFAULT_TABLE_TACTILE_POINT_DISTANCE",
        "R1_PRO_HEAD_CAMERA_SPEC",
        "R1_PRO_GRIPPER_TACTILE_MATERIAL",
        "R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE",
        "R1_PRO_GRIPPER_TACTILE_PAD_SPECS",
        "R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA",
        "R1ProHeadCameraSpec",
        "R1ProGripperTactilePadSpec",
        "TactileContactMaterialSpec",
        "TactilePadBounds",
        "VtRefineTactileSensor",
        "VtRefineTactileSensorCfg",
        "VtRefineTactileSensorData",
        "apply_tactile_compliant_material",
        "configure_r1_pro_gripper_tactile_scene_cfg",
        "configure_table_tactile_scene_cfg",
        "get_r1_pro_gripper_tactile_points",
        "make_r1_pro_head_camera_cfg",
        "make_r1_pro_gripper_tactile_sensor_cfg",
        "make_table_tactile_pad_cfg",
        "make_table_tactile_sensor_cfg",
        "tactile_force_grid",
    }


def test_contact_target_runtime_groups_the_five_contact_resources() -> None:
    assert tuple(tactile_module._ContactTargetRuntime.__dataclass_fields__) == (
        "sdf_view",
        "body_view",
        "com_b",
        "mesh_pos_local",
        "mesh_quat_local",
    )
    assert tactile_module._ContactTargetRuntime.__slots__ == (
        "sdf_view",
        "body_view",
        "com_b",
        "mesh_pos_local",
        "mesh_quat_local",
    )


def test_r1_pro_gripper_tactile_registry_has_stable_right_then_left_order() -> None:
    assert isinstance(R1_PRO_GRIPPER_TACTILE_PAD_SPECS, tuple)
    assert all(isinstance(spec, R1ProGripperTactilePadSpec) for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS)
    assert (
        tuple(
            (
                spec.label,
                spec.sensor_name,
                spec.pad_link_name,
                spec.parent_link_name,
                spec.mesh_path.name,
                spec.surface_sign,
                spec.normal_axis,
                spec.panel_grid_coordinate,
            )
            for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS
        )
        == EXPECTED_PAD_IDENTITIES
    )


def test_r1_pro_gripper_tactile_registry_paths_are_derived_and_unique() -> None:
    unique_attributes = (
        "label",
        "sensor_name",
        "pad_link_name",
        "prim_path_expr",
        "panel_grid_coordinate",
        "usd_visual_root_path",
        "usd_collider_root_path",
        "usd_material_path",
    )
    for attribute in unique_attributes:
        values = tuple(getattr(spec, attribute) for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS)
        assert len(values) == len(set(values)), f"Duplicate {attribute}: {values}"

    for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        assert spec.prim_path_expr == f"{{ENV_REGEX_NS}}/Robot/.*{spec.pad_link_name}"
        assert spec.usd_visual_root_path == f"/visuals/{spec.pad_link_name}"
        assert spec.usd_collider_root_path == f"/colliders/{spec.pad_link_name}"
        assert spec.usd_material_path == (
            f"{spec.usd_collider_root_path}/materials/{spec.pad_link_name}_compliant_material"
        )


def test_r1_pro_gripper_tactile_pad_specs_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        R1_PRO_GRIPPER_TACTILE_PAD_SPECS[0].surface_sign = 1.0  # type: ignore[misc]


def test_r1_pro_gripper_tactile_sensor_factory_accepts_one_pad_spec() -> None:
    parameters = inspect.signature(tactile_module.make_r1_pro_gripper_tactile_sensor_cfg).parameters

    assert tuple(parameters)[:2] == ("contact_prim_paths_expr", "pad_spec")
    assert "elastomer_mesh_path" not in parameters
    assert "prim_path" not in parameters
    assert "tactile_surface_sign" not in parameters
    assert "tactile_normal_axis" not in parameters


def test_r1_pro_gripper_tactile_points_use_mesh_inner_surfaces() -> None:
    for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        points = generate_tactile_points_on_rectangular_pad_from_bounds(
            pad_bounds=spec.bounds,
            surface_axis=1,
            surface_sign=spec.surface_sign,
            grid_axes=(0, 2),
        )
        mins, maxs = spec.bounds
        center_x = (mins[0] + maxs[0]) / 2.0
        center_z = (mins[2] + maxs[2]) / 2.0
        surface_y = mins[1] if spec.surface_sign < 0.0 else maxs[1]

        assert points.shape == (384, 3)
        assert torch.allclose(points[:, 1], torch.full((384,), surface_y))
        assert torch.isclose(points[:, 0].min(), torch.tensor(center_x - DEFAULT_TABLE_TACTILE_POINT_DISTANCE * 5.5))
        assert torch.isclose(points[:, 0].max(), torch.tensor(center_x + DEFAULT_TABLE_TACTILE_POINT_DISTANCE * 5.5))
        assert torch.isclose(points[:, 2].min(), torch.tensor(center_z - DEFAULT_TABLE_TACTILE_POINT_DISTANCE * 15.5))
        assert torch.isclose(points[:, 2].max(), torch.tensor(center_z + DEFAULT_TABLE_TACTILE_POINT_DISTANCE * 15.5))
        assert points[:, 0].min() > mins[0]
        assert points[:, 0].max() < maxs[0]
        assert points[:, 2].min() > mins[2]
        assert points[:, 2].max() < maxs[2]


def test_r1_pro_gripper_tactile_normal_axes_follow_compression_direction() -> None:
    for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        compression_force = torch.tensor([0.0, -spec.surface_sign * 0.002, 0.0])

        assert torch.dot(compression_force, torch.tensor(spec.normal_axis)) > 0.0


def test_r1_pro_gripper_tactile_observation_size_is_four_12_by_32_pads() -> None:
    points_per_pad = tactile_point_count(DEFAULT_TABLE_TACTILE_ARRAY_SIZE)
    values_per_pad = tactile_observation_size(DEFAULT_TABLE_TACTILE_ARRAY_SIZE)

    assert points_per_pad == 384
    assert points_per_pad * len(R1_PRO_GRIPPER_TACTILE_PAD_SPECS) == 1536
    assert values_per_pad * len(R1_PRO_GRIPPER_TACTILE_PAD_SPECS) == R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE
    assert R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE == 6144


class _FakeTactileSensor:
    def __init__(self, pad_index: int):
        self.pad_index = pad_index
        self.calls: list[tuple[bool, torch.Tensor | None]] = []
        point_count = tactile_point_count(DEFAULT_TABLE_TACTILE_ARRAY_SIZE)
        self.positions = torch.full((2, point_count, 3), float(pad_index))
        base_force = torch.linspace(0.0, float(pad_index + 1), point_count)
        self.normal_force = torch.stack((base_force, base_force * 2.0), dim=0)

    def get_tactile_points(
        self,
        *,
        normalize: bool = False,
        env_origins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append((normalize, env_origins))
        positions = self.positions
        if env_origins is not None:
            positions = positions - env_origins.unsqueeze(1)
        force = self.normal_force
        if normalize:
            force = normalize_tactile_normal_force(force, low_force_threshold=0.0)
        return torch.cat((positions, force.unsqueeze(-1)), dim=-1)


def test_r1_pro_gripper_tactile_observation_normalizes_each_pad_then_concatenates_registry_order() -> None:
    sensors = {
        spec.sensor_name: _FakeTactileSensor(pad_index)
        for pad_index, spec in reversed(tuple(enumerate(R1_PRO_GRIPPER_TACTILE_PAD_SPECS)))
    }
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]])

    observation = get_r1_pro_gripper_tactile_points(sensors, normalize=True, env_origins=env_origins)

    points_per_pad = tactile_point_count(DEFAULT_TABLE_TACTILE_ARRAY_SIZE)
    assert observation.shape == (2, points_per_pad * 4, 4)
    assert observation[0].numel() == R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE
    expected_normalized_force = torch.linspace(0.0, 1.0, points_per_pad)
    for pad_index, spec in enumerate(R1_PRO_GRIPPER_TACTILE_PAD_SPECS):
        pad_observation = observation[:, pad_index * points_per_pad : (pad_index + 1) * points_per_pad]
        expected_positions = sensors[spec.sensor_name].positions - env_origins.unsqueeze(1)
        assert torch.equal(pad_observation[..., :3], expected_positions)
        assert torch.allclose(pad_observation[0, :, 3], expected_normalized_force)
        assert torch.allclose(pad_observation[1, :, 3], expected_normalized_force)
        assert sensors[spec.sensor_name].calls == [(True, env_origins)]


def _fake_env_cfg(*, append_to_policy: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        enable_r1_pro_gripper_tactile=not append_to_policy,
        append_r1_pro_gripper_tactile_to_policy=append_to_policy,
        r1_pro_gripper_tactile_contact_part_names=(),
        assembly_reset_part_names=("resettable_part_a", "resettable_part_b"),
        assembly_part_names=("fixed_part", "resettable_part_a", "resettable_part_b"),
        observation_space=100,
        scene=SimpleNamespace(
            fixed_part=SimpleNamespace(prim_path="{ENV_REGEX_NS}/FixedPart"),
            resettable_part_a=SimpleNamespace(prim_path="{ENV_REGEX_NS}/ResettablePartA"),
            resettable_part_b=SimpleNamespace(prim_path="{ENV_REGEX_NS}/ResettablePartB"),
        ),
    )


def test_r1_pro_gripper_tactile_scene_injection_uses_registry_and_resettable_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[tuple[str, ...], R1ProGripperTactilePadSpec]] = []

    def fake_sensor_cfg(
        contact_prim_paths_expr: tuple[str, ...],
        pad_spec: R1ProGripperTactilePadSpec,
        **_: Any,
    ) -> SimpleNamespace:
        created.append((tuple(contact_prim_paths_expr), pad_spec))
        return SimpleNamespace(contact_prim_paths_expr=tuple(contact_prim_paths_expr), pad_spec=pad_spec)

    monkeypatch.setattr(tactile_module, "make_r1_pro_gripper_tactile_sensor_cfg", fake_sensor_cfg)
    cfg = _fake_env_cfg()

    contact_part_names = tactile_module.configure_r1_pro_gripper_tactile_scene_cfg(cfg)

    expected_contact_paths = ("{ENV_REGEX_NS}/ResettablePartA", "{ENV_REGEX_NS}/ResettablePartB")
    assert contact_part_names == cfg.assembly_reset_part_names
    assert created == [(expected_contact_paths, pad_spec) for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS]
    for pad_spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        sensor_cfg = getattr(cfg.scene, pad_spec.sensor_name)
        assert sensor_cfg.pad_spec is pad_spec
        assert sensor_cfg.contact_prim_paths_expr == expected_contact_paths
    assert cfg.observation_space == 100


def test_r1_pro_gripper_tactile_scene_injection_adds_policy_space_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tactile_module,
        "make_r1_pro_gripper_tactile_sensor_cfg",
        lambda contact_prim_paths_expr, pad_spec, **_: SimpleNamespace(
            contact_prim_paths_expr=tuple(contact_prim_paths_expr), pad_spec=pad_spec
        ),
    )
    cfg = _fake_env_cfg(append_to_policy=True)

    tactile_module.configure_r1_pro_gripper_tactile_scene_cfg(cfg)
    tactile_module.configure_r1_pro_gripper_tactile_scene_cfg(cfg)

    assert cfg.observation_space == 100 + R1_PRO_GRIPPER_TACTILE_OBSERVATION_SIZE
