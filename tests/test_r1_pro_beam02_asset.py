# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark" / "assets" / "robots" / "r1_pro_beam02"
ASSET_PATH = ASSET_DIR / "r1_pro_beam02_fixed.usd"
GENERATOR_PATH = REPO_ROOT / "scripts" / "tools" / "generate_r1_pro_beam02_asset.py"
ISAAC_SIM_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("carb") is not None and importlib.util.find_spec("pxr") is not None
)


def test_r1_pro_beam02_asset_is_packaged_with_reproducible_generator() -> None:
    assert ASSET_PATH.is_file()
    assert (ASSET_DIR / "configuration" / "r1_pro_beam02_fixed_base.usd").is_file()
    assert (ASSET_DIR / "configuration" / "r1_pro_beam02_fixed_physics.usd").is_file()
    assert GENERATOR_PATH.is_file()

    generator_source = GENERATOR_PATH.read_text(encoding="utf-8")
    assert "merge_fixed_joints=False" in generator_source
    assert "SDF_RESOLUTION = 512" in generator_source
    assert 'PLUG_LINK_NAME = "beam_plug_0"' in generator_source
    assert "PLUG_JOINT_POS = BEAM02_GRIPPER_TO_PLUG_POS" in generator_source
    assert "PLUG_JOINT_RPY = _quaternion_to_rpy(BEAM02_GRIPPER_TO_PLUG_QUAT)" in generator_source
    assert '"left_realsense_link"' in generator_source


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim USD runtime is not available")
def test_r1_pro_beam02_usd_physics_contract() -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(ASSET_PATH))
    assert stage is not None

    root_path = "/r1_pro_with_gripper"
    plug = stage.GetPrimAtPath(f"{root_path}/beam_plug_0")
    assert plug.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(plug).GetMassAttr().Get() == pytest.approx(0.0128723126)

    joint_prim = stage.GetPrimAtPath(f"{root_path}/joints/beam_plug_0_joint")
    joint = UsdPhysics.FixedJoint(joint_prim)
    assert joint
    assert [target.name for target in joint.GetBody0Rel().GetTargets()] == ["left_gripper_link"]
    assert [target.name for target in joint.GetBody1Rel().GetTargets()] == ["beam_plug_0"]

    collision_root = stage.GetPrimAtPath(f"{root_path}/beam_plug_0/collisions")
    assert collision_root.IsValid()
    assert not collision_root.IsInstanceable()
    colliders = [
        prim
        for prim in stage.TraverseAll()
        if "beam_plug_0/collisions" in str(prim.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.MeshCollisionAPI(prim)
    ]
    assert len(colliders) == 1
    collider = colliders[0]
    assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == PhysxSchema.Tokens.sdf
    assert PhysxSchema.PhysxSDFMeshCollisionAPI(collider).GetSdfResolutionAttr().Get() == 512

    material = UsdShade.MaterialBindingAPI(collider).GetDirectBinding("physics").GetMaterial()
    material_api = UsdPhysics.MaterialAPI(material.GetPrim())
    assert material_api.GetStaticFrictionAttr().Get() == pytest.approx(1.0)
    assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(1.0)

    filtered_targets = UsdPhysics.FilteredPairsAPI(plug).GetFilteredPairsRel().GetTargets()
    assert {target.name for target in filtered_targets} == {
        "left_arm_link7",
        "left_gripper_link",
        "left_gripper_finger_link1",
        "left_gripper_finger_link2",
        "left_realsense_link",
    }
