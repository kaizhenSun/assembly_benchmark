# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
from assembly_benchmark.assembly import (
    BEAM_FABRICA_PLAN,
    PIPER_FABRICA_GRIPPER_BASE_POS,
    PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "assembly_benchmark" / "assembly_benchmark"
ROBOT_ASSET_ROOT = PACKAGE_ROOT / "assets" / "robots"
PIPER_DIR = ROBOT_ASSET_ROOT / "piper"
FIXED_PLUG_ASSET_DIR = PIPER_DIR / "fixed_plug" / "beam"
SOCKET_ASSET_DIR = PACKAGE_ROOT / "assets" / "fabrica" / "beam" / "usd" / "fixed_plug_socket"
GENERATOR_PATH = REPO_ROOT / "scripts" / "tools" / "generate_fabrica_fixedplug_assets.py"
ISAAC_SIM_RUNTIME_AVAILABLE = (
    importlib.util.find_spec("carb") is not None and importlib.util.find_spec("pxr") is not None
)


def test_piper_sources_and_all_fixedplug_assets_are_packaged() -> None:
    assert (PIPER_DIR / "piper_description.urdf").is_file()
    assert (PIPER_DIR / "pika2_gripper.urdf").is_file()
    assert not (PIPER_DIR / "piper_with_gripper_description.xacro").exists()
    assert (PIPER_DIR / "piper_fixed.usd").is_file()
    assert "MIT License" in (PIPER_DIR / "LICENSE.agx-arm-urdf-MIT.txt").read_text(encoding="utf-8")
    expected_meshes = {
        "base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
    }
    assert {path.stem for path in (PIPER_DIR / "meshes").glob("*.stl")} == expected_meshes
    assert {path.stem for path in (PIPER_DIR / "meshes" / "dae").glob("*.dae")} == expected_meshes | {
        "pika_gripper_base",
        "gripper_left_link",
        "gripper_right_link",
    }

    for relation_key in BEAM_FABRICA_PLAN.relation_keys:
        relation_dir = FIXED_PLUG_ASSET_DIR / relation_key
        assert (relation_dir / "piper_fixed_plug.usd").is_file()
        for layer in ("base", "physics", "robot", "sensor"):
            assert (relation_dir / "configuration" / f"piper_fixed_plug_{layer}.usd").is_file()

    expected_socket_part_ids = {relation.socket_part_id for relation in BEAM_FABRICA_PLAN.relations}
    assert [(relation.key, relation.socket_part_id) for relation in BEAM_FABRICA_PLAN.relations] == [
        ("0_2", "2"),
        ("1_3", "3"),
        ("2_6", "6"),
        ("3_6", "6"),
    ]
    assert {path.name for path in SOCKET_ASSET_DIR.iterdir() if path.is_dir()} == expected_socket_part_ids
    for part_id in expected_socket_part_ids:
        assert (SOCKET_ASSET_DIR / part_id / "socket.usd").is_file()

    generator = GENERATOR_PATH.read_text(encoding="utf-8")
    assert 'PLUG_LINK_NAME = "plug"' in generator
    assert "PLUG_DENSITY = 1250.0" in generator
    assert "PLUG_FRICTION = 1.0" in generator
    assert "SDF_RESOLUTION = 512" in generator
    assert "merge_fixed_joints=False" in generator
    assert "convert_mimic_joints_to_normal_joints=False" in generator
    assert "write_piper_pika2_urdf" in generator
    assert "dict.fromkeys(relation.socket_part_id for relation in plan.relations)" in generator
    assert "topology-compatible fixed-plug robots" in generator
    attributes = set((REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines())
    assert "*.stl filter=lfs diff=lfs merge=lfs -text" in attributes
    assert "*.dae filter=lfs diff=lfs merge=lfs -text" in attributes
    assert "*.usd filter=lfs diff=lfs merge=lfs -text" in attributes


@pytest.mark.skipif(not ISAAC_SIM_RUNTIME_AVAILABLE, reason="Isaac Sim USD runtime is not available")
def test_all_fixedplug_usds_have_common_topology_and_relation_transforms() -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    assert pytest.approx((0.0, 0.0, -0.0255)) == PIPER_FABRICA_GRIPPER_BASE_POS
    assert pytest.approx((0.0, 0.0, 0.5 * math.pi)) == PIPER_FABRICA_GRIPPER_BASE_ROTATION_RPY

    base_stage = Usd.Stage.Open(str(PIPER_DIR / "piper_fixed.usd"))
    assert base_stage is not None
    base_root_path = base_stage.GetDefaultPrim().GetPath()
    base_mount = UsdPhysics.FixedJoint(base_stage.GetPrimAtPath(base_root_path.AppendPath("joints/gripper_base_joint")))
    assert tuple(base_mount.GetLocalPos0Attr().Get()) == pytest.approx((0.0, 0.0, 0.0))
    base_mount_rot = base_mount.GetLocalRot0Attr().Get()
    assert abs(float(base_mount_rot.GetReal())) == pytest.approx(1.0)
    for joint_index in range(1, 7):
        joint = base_stage.GetPrimAtPath(base_root_path.AppendPath(f"joints/joint{joint_index}"))
        max_velocity_deg_s = joint.GetAttribute("physxJoint:maxJointVelocity").Get()
        assert math.radians(max_velocity_deg_s) == pytest.approx(3.0)

    topologies = []
    for relation in BEAM_FABRICA_PLAN.relations:
        stage = Usd.Stage.Open(str(FIXED_PLUG_ASSET_DIR / relation.key / "piper_fixed_plug.usd"))
        assert stage is not None
        root = stage.GetDefaultPrim()
        root_path = root.GetPath()
        gripper_mount = UsdPhysics.FixedJoint(stage.GetPrimAtPath(root_path.AppendPath("joints/gripper_base_joint")))
        assert [target.name for target in gripper_mount.GetBody0Rel().GetTargets()] == ["link6"]
        assert [target.name for target in gripper_mount.GetBody1Rel().GetTargets()] == ["gripper_base_link"]
        assert tuple(gripper_mount.GetLocalPos0Attr().Get()) == pytest.approx(PIPER_FABRICA_GRIPPER_BASE_POS)
        mount_rot = gripper_mount.GetLocalRot0Attr().Get()
        mount_quat = (mount_rot.GetReal(), *mount_rot.GetImaginary())
        expected_mount_quat = (2.0**-0.5, 0.0, 0.0, 2.0**-0.5)
        mount_dot = abs(sum(left * right for left, right in zip(mount_quat, expected_mount_quat, strict=True)))
        assert mount_dot == pytest.approx(1.0, abs=5.0e-7)

        plug = stage.GetPrimAtPath(root_path.AppendChild("plug"))
        assert plug.HasAPI(UsdPhysics.RigidBodyAPI)
        fixed_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(root_path.AppendPath("joints/plug_joint")))
        assert [target.name for target in fixed_joint.GetBody0Rel().GetTargets()] == ["gripper_base_link"]
        assert [target.name for target in fixed_joint.GetBody1Rel().GetTargets()] == ["plug"]
        assert tuple(fixed_joint.GetLocalPos0Attr().Get()) == pytest.approx(relation.piper_gripper_to_plug_pos)
        local_rot = fixed_joint.GetLocalRot0Attr().Get()
        local_quat = (local_rot.GetReal(), *local_rot.GetImaginary())
        dot = abs(
            sum(left * right for left, right in zip(local_quat, relation.piper_gripper_to_plug_quat, strict=True))
        )
        assert dot == pytest.approx(1.0, abs=5.0e-7)

        colliders = [
            prim
            for prim in stage.TraverseAll()
            if "/plug/collisions" in str(prim.GetPath())
            and prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.MeshCollisionAPI(prim)
        ]
        assert len(colliders) == 1
        collider = colliders[0]
        assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == PhysxSchema.Tokens.sdf
        assert PhysxSchema.PhysxSDFMeshCollisionAPI(collider).GetSdfResolutionAttr().Get() == 512
        material = UsdShade.MaterialBindingAPI(collider).GetDirectBinding("physics").GetMaterial()
        assert UsdPhysics.MaterialAPI(material.GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(1.0)
        filtered_names = {
            target.name for target in UsdPhysics.FilteredPairsAPI(plug).GetFilteredPairsRel().GetTargets()
        }
        assert filtered_names == {
            "link6",
            "gripper_base_link",
            "gripper_left_link",
            "gripper_right_link",
        }
        joint_root = root_path.AppendChild("joints")
        for joint_index in range(1, 7):
            joint = stage.GetPrimAtPath(joint_root.AppendChild(f"joint{joint_index}"))
            max_velocity_deg_s = joint.GetAttribute("physxJoint:maxJointVelocity").Get()
            assert math.radians(max_velocity_deg_s) == pytest.approx(3.0)
        left_joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_root.AppendChild("left_joint")))
        right_joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_root.AppendChild("right_joint")))
        assert left_joint and right_joint
        assert left_joint.GetLowerLimitAttr().Get() >= 0.0
        assert left_joint.GetUpperLimitAttr().Get() > 0.0
        assert right_joint.GetLowerLimitAttr().Get() < 0.0
        assert right_joint.GetUpperLimitAttr().Get() <= 0.0
        bodies = tuple(sorted(prim.GetName() for prim in stage.TraverseAll() if prim.HasAPI(UsdPhysics.RigidBodyAPI)))
        joints = tuple(sorted(prim.GetName() for prim in stage.TraverseAll() if prim.IsA(UsdPhysics.Joint)))
        topologies.append((bodies, joints))

        socket_stage = Usd.Stage.Open(str(SOCKET_ASSET_DIR / relation.socket_part_id / "socket.usd"))
        assert socket_stage is not None
        socket_bodies = [prim for prim in socket_stage.TraverseAll() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert [prim.GetName() for prim in socket_bodies] == ["socket"]
        socket_colliders = [
            prim
            for prim in socket_stage.TraverseAll()
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.MeshCollisionAPI(prim)
        ]
        assert len(socket_colliders) == 1
        socket_collider = socket_colliders[0]
        assert UsdPhysics.MeshCollisionAPI(socket_collider).GetApproximationAttr().Get() == PhysxSchema.Tokens.sdf
        socket_material = UsdShade.MaterialBindingAPI(socket_collider).GetDirectBinding("physics").GetMaterial()
        assert UsdPhysics.MaterialAPI(socket_material.GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(0.5)
    assert len(set(topologies)) == 1
