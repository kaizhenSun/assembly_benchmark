# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from assembly_benchmark.sensors.vt_refine_tactile import (
    R1_PRO_GRIPPER_TACTILE_MATERIAL,
    R1_PRO_GRIPPER_TACTILE_PAD_SPECS,
    R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA,
    TactileContactMaterialSpec,
    read_mesh_vertex_bounds,
    read_stl_vertex_bounds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ASSET_ROOT = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "assets"
TACTILE_MESH_DIR = PACKAGE_ASSET_ROOT / "sensors" / "vt_refine_tactile"
R1_PRO_ASSET_DIR = PACKAGE_ASSET_ROOT / "robots" / "r1_pro"


def test_r1_pro_gripper_tactile_material_contract_is_centralized_and_frozen() -> None:
    assert isinstance(R1_PRO_GRIPPER_TACTILE_MATERIAL, TactileContactMaterialSpec)
    assert R1_PRO_GRIPPER_TACTILE_MATERIAL.stiffness == 100.0
    assert R1_PRO_GRIPPER_TACTILE_MATERIAL.damping == 0.01
    assert R1_PRO_GRIPPER_TACTILE_MATERIAL.acceleration_spring is False
    assert R1_PRO_GRIPPER_TACTILE_MATERIAL.static_friction == 1.0
    assert R1_PRO_GRIPPER_TACTILE_MATERIAL.dynamic_friction == 1.0
    assert R1_PRO_GRIPPER_TACTILE_MATERIAL.restitution == 0.0
    assert R1_PRO_GRIPPER_TACTILE_MATERIAL.contact_offset == 0.005
    assert R1_PRO_GRIPPER_TACTILE_MATERIAL.rest_offset == 0.0
    assert R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA == (0.42, 0.18, 0.07, 1.0)
    with pytest.raises(FrozenInstanceError):
        R1_PRO_GRIPPER_TACTILE_MATERIAL.stiffness = 1.0  # type: ignore[misc]


def test_r1_pro_gripper_tactile_registry_uses_two_canonical_stl_assets() -> None:
    expected_mesh_names = {
        "r1_pro_gripper_finger_link1_flat_pad.STL",
        "r1_pro_gripper_finger_link2_flat_pad.STL",
    }

    assert {spec.mesh_path.name for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS} == expected_mesh_names
    for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        assert spec.mesh_path == TACTILE_MESH_DIR / spec.mesh_path.name
        assert spec.mesh_path.is_file()
        bounds = read_stl_vertex_bounds(spec.mesh_path)
        size = tuple(max_value - min_value for min_value, max_value in zip(bounds[0], bounds[1], strict=True))
        assert read_mesh_vertex_bounds(spec.mesh_path) == bounds
        assert spec.bounds == bounds
        assert spec.size == pytest.approx(size, abs=5.0e-6)

    obsolete_mesh_names = {
        f"{hand}_gripper_finger_link{link_index}_flat_pad.STL" for hand in ("right", "left") for link_index in (1, 2)
    }
    assert all(not (TACTILE_MESH_DIR / mesh_name).exists() for mesh_name in obsolete_mesh_names)


def test_r1_pro_finger_collision_meshes_meet_tactile_pad_backs() -> None:
    robot_mesh_dir = R1_PRO_ASSET_DIR / "meshes"

    for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        finger_bounds = read_stl_vertex_bounds(robot_mesh_dir / f"{spec.parent_link_name}_collision.STL")
        pad_mins, pad_maxs = spec.bounds
        if spec.parent_link_name.endswith("link1"):
            assert finger_bounds[0][1] == pytest.approx(pad_maxs[1])
        else:
            assert finger_bounds[1][1] == pytest.approx(pad_mins[1])
        assert finger_bounds[0][2] == pytest.approx(-0.0415, abs=5.0e-6)


def test_r1_pro_urdf_has_shared_visual_material_and_registry_pad_contracts() -> None:
    root = ET.parse(R1_PRO_ASSET_DIR / "robot.urdf").getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
    tactile_materials = [
        material for material in root.findall("material") if material.attrib.get("name") == "r1_pro_tactile_pad_visual"
    ]

    assert len(tactile_materials) == 1
    assert tactile_materials[0].find("color").attrib["rgba"] == " ".join(
        str(value) for value in R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA
    )
    for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        pad_link = links[spec.pad_link_name]
        joint = joints[spec.pad_link_name]
        expected_mesh_path = f"../../sensors/vt_refine_tactile/{spec.mesh_path.name}"

        assert joint.attrib["type"] == "fixed"
        assert joint.find("parent").attrib["link"] == spec.parent_link_name
        assert joint.find("origin").attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}
        assert {mesh.attrib["filename"] for mesh in pad_link.findall(".//mesh")} == {expected_mesh_path}
        visual_material = pad_link.find("./visual/material")
        assert visual_material is not None
        assert visual_material.attrib == {"name": "r1_pro_tactile_pad_visual"}
        assert visual_material.find("color") is None
        assert pad_link.find("./collision/material") is None


@pytest.mark.skipif(importlib.util.find_spec("pxr") is None, reason="USD Python runtime is not available")
def test_r1_pro_generated_usd_colors_only_tactile_pad_visuals_brown() -> None:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(R1_PRO_ASSET_DIR / "configuration" / "r1_pro_fixed_base.usd"))
    assert stage is not None

    expected_color = R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA[:3]
    expected_tactile_mesh_paths = []
    for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        tactile_visual_root = stage.GetPrimAtPath(spec.usd_visual_root_path)
        tactile_mesh_prims = [prim for prim in Usd.PrimRange(tactile_visual_root) if prim.GetTypeName() == "Mesh"]
        assert len(tactile_mesh_prims) == 1
        material_targets = UsdShade.MaterialBindingAPI(tactile_mesh_prims[0]).GetDirectBindingRel().GetTargets()
        assert len(material_targets) == 1
        assert str(material_targets[0]).startswith(f"{spec.usd_visual_root_path}/")

        material_prim = stage.GetPrimAtPath(material_targets[0])
        shader_prims = [prim for prim in Usd.PrimRange(material_prim) if prim.GetTypeName() == "Shader"]
        assert len(shader_prims) == 1
        shader = UsdShade.Shader(shader_prims[0])
        assert tuple(shader.GetInput("diffuse_color_constant").Get()) == pytest.approx(expected_color)
        assert shader.GetInput("opacity_constant").Get() == pytest.approx(R1_PRO_GRIPPER_TACTILE_VISUAL_RGBA[3])
        expected_tactile_mesh_paths.append(str(tactile_mesh_prims[0].GetPath()))

    brown_bound_mesh_paths = []
    for prim in stage.TraverseAll():
        if prim.GetTypeName() != "Mesh":
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        shaders = [child for child in Usd.PrimRange(material.GetPrim()) if child.GetTypeName() == "Shader"]
        if len(shaders) != 1:
            continue
        diffuse_color = UsdShade.Shader(shaders[0]).GetInput("diffuse_color_constant")
        if diffuse_color and tuple(diffuse_color.Get()) == pytest.approx(expected_color):
            brown_bound_mesh_paths.append(str(prim.GetPath()))

    assert set(brown_bound_mesh_paths) == set(expected_tactile_mesh_paths)


@pytest.mark.skipif(importlib.util.find_spec("pxr") is None, reason="USD Python runtime is not available")
def test_r1_pro_generated_usd_binds_independent_compliant_materials_only_to_tactile_pads() -> None:
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(R1_PRO_ASSET_DIR / "configuration" / "r1_pro_fixed_physics.usd"))
    assert stage is not None

    material_paths = tuple(spec.usd_material_path for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS)
    assert len(material_paths) == len(set(material_paths)) == 4
    expected_bound_mesh_paths = []
    for spec in R1_PRO_GRIPPER_TACTILE_PAD_SPECS:
        material_prim = stage.GetPrimAtPath(spec.usd_material_path)
        assert material_prim.IsValid()
        assert not stage.GetPrimAtPath(f"/materials/{spec.pad_link_name}_compliant_material").IsValid()
        assert str(material_prim.GetPath()).startswith(f"{spec.usd_collider_root_path}/")
        assert set(material_prim.GetMetadata("apiSchemas").explicitItems) == {
            "PhysicsMaterialAPI",
            "PhysxMaterialAPI",
        }
        assert material_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(
            R1_PRO_GRIPPER_TACTILE_MATERIAL.static_friction
        )
        assert material_prim.GetAttribute("physics:dynamicFriction").Get() == pytest.approx(
            R1_PRO_GRIPPER_TACTILE_MATERIAL.dynamic_friction
        )
        assert material_prim.GetAttribute("physics:restitution").Get() == pytest.approx(
            R1_PRO_GRIPPER_TACTILE_MATERIAL.restitution
        )
        assert material_prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == pytest.approx(
            R1_PRO_GRIPPER_TACTILE_MATERIAL.stiffness
        )
        assert material_prim.GetAttribute("physxMaterial:compliantContactDamping").Get() == pytest.approx(
            R1_PRO_GRIPPER_TACTILE_MATERIAL.damping
        )
        assert (
            material_prim.GetAttribute("physxMaterial:compliantContactAccelerationSpring").Get()
            is R1_PRO_GRIPPER_TACTILE_MATERIAL.acceleration_spring
        )

        tactile_collision_root = stage.GetPrimAtPath(spec.usd_collider_root_path)
        tactile_mesh_prims = [prim for prim in Usd.PrimRange(tactile_collision_root) if prim.GetTypeName() == "Mesh"]
        assert len(tactile_mesh_prims) == 1
        tactile_mesh_prim = tactile_mesh_prims[0]
        assert tactile_mesh_prim.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(
            R1_PRO_GRIPPER_TACTILE_MATERIAL.contact_offset
        )
        assert tactile_mesh_prim.GetAttribute("physxCollision:restOffset").Get() == pytest.approx(
            R1_PRO_GRIPPER_TACTILE_MATERIAL.rest_offset
        )
        tactile_targets = UsdShade.MaterialBindingAPI(tactile_mesh_prim).GetDirectBindingRel("physics").GetTargets()
        assert [str(target) for target in tactile_targets] == [spec.usd_material_path]
        expected_bound_mesh_paths.append(str(tactile_mesh_prim.GetPath()))

    bound_mesh_paths: dict[str, str] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/colliders")):
        if prim.GetTypeName() != "Mesh":
            continue
        for target in UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel("physics").GetTargets():
            target_path = str(target)
            if target_path in material_paths:
                bound_mesh_paths[target_path] = str(prim.GetPath())

    assert bound_mesh_paths == dict(zip(material_paths, expected_bound_mesh_paths, strict=True))
