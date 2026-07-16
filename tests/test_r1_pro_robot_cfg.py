# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path
from xml.etree import ElementTree

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ASSET_ROOT = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "assets"
R1_PRO_ASSET_DIR = PACKAGE_ASSET_ROOT / "robots" / "r1_pro"
REMOVED_PAD_SENSOR_MARKERS = ("tac" + "tile", "vt_" + "refine", "flat_" + "pad")
ISAAC_SIM_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("carb") is not None and importlib.util.find_spec("pxr") is not None
)


def _assert_removed_pad_sensor_markers_absent(value: str) -> None:
    normalized_value = value.casefold()
    assert all(marker not in normalized_value for marker in REMOVED_PAD_SENSOR_MARKERS)


def test_r1_pro_urdf_and_asset_paths_exclude_removed_pad_sensor_assets() -> None:
    urdf_root = ElementTree.parse(R1_PRO_ASSET_DIR / "robot.urdf").getroot()
    _assert_removed_pad_sensor_markers_absent(ElementTree.tostring(urdf_root, encoding="unicode"))

    for asset_path in (path for path in PACKAGE_ASSET_ROOT.rglob("*") if path.is_file()):
        _assert_removed_pad_sensor_markers_absent(asset_path.relative_to(PACKAGE_ASSET_ROOT).as_posix())


@pytest.mark.skipif(importlib.util.find_spec("pxr") is None, reason="USD runtime is not available")
def test_r1_pro_generated_usd_layers_exclude_removed_pad_sensor_assets() -> None:
    from pxr import Usd

    for usd_name in ("r1_pro_fixed_base.usd", "r1_pro_fixed_physics.usd", "r1_pro_fixed_robot.usd"):
        stage = Usd.Stage.Open(str(R1_PRO_ASSET_DIR / "configuration" / usd_name))

        assert stage is not None
        for layer in stage.GetUsedLayers():
            _assert_removed_pad_sensor_markers_absent(layer.ExportToString())


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_r1_pro_gripper_home_position_is_fully_open() -> None:
    from assembly_benchmark.robots.r1_pro import (
        R1_PRO_CFG,
        R1_PRO_GRIPPER_HOME_POS,
        R1_PRO_GRIPPER_JOINT_NAMES,
    )

    assert R1_PRO_GRIPPER_HOME_POS == 0.05
    assert R1_PRO_CFG.init_state.joint_pos is not None
    assert {joint_name: R1_PRO_CFG.init_state.joint_pos[joint_name] for joint_name in R1_PRO_GRIPPER_JOINT_NAMES} == {
        joint_name: R1_PRO_GRIPPER_HOME_POS for joint_name in R1_PRO_GRIPPER_JOINT_NAMES
    }


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim runtime is not available")
def test_contact_assets_use_conservative_collision_offsets() -> None:
    from assembly_benchmark.assembly import available_assemblies, make_assembly
    from assembly_benchmark.assembly.isaac import make_assembly_part_spawn_cfg
    from assembly_benchmark.assets.furniture.lab_table import make_lab_table_cfg
    from assembly_benchmark.robots.r1_pro import R1_PRO_CFG

    def assert_collision_offsets(spawn) -> None:
        assert spawn is not None
        assert spawn.collision_props is not None
        assert spawn.collision_props.contact_offset == pytest.approx(0.001)
        assert spawn.collision_props.rest_offset == pytest.approx(0.0)

    assert_collision_offsets(R1_PRO_CFG.spawn)
    assert_collision_offsets(make_lab_table_cfg().spawn)

    for assembly_name in available_assemblies():
        assembly = make_assembly(assembly_name)
        for part in assembly.parts:
            spawn = make_assembly_part_spawn_cfg(assembly, part)
            if part.body_type == "visual":
                assert spawn.collision_props is None
            else:
                assert_collision_offsets(spawn)
