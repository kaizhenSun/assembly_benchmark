#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the Galaxea R1 Pro URDF asset to a fixed-base USD asset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "assembly_benchmark"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

parser = argparse.ArgumentParser(description="Convert R1 Pro URDF to USD.")
parser.add_argument("--force", action="store_true", help="Force USD regeneration even if the output exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

from assembly_benchmark.robots.r1_pro import R1_PRO_ASSET_DIR, R1_PRO_URDF_PATH, R1_PRO_USD_PATH  # noqa: E402

GRIPPER_COLLISION_PATHS = (
    "/colliders/left_gripper_link/left_gripper_link_collision",
    "/colliders/left_gripper_finger_link1/left_gripper_finger_link1_collision",
    "/colliders/left_gripper_finger_link2/left_gripper_finger_link2_collision",
    "/colliders/right_gripper_link/right_gripper_link_collision",
    "/colliders/right_gripper_finger_link1/right_gripper_finger_link1_collision",
    "/colliders/right_gripper_finger_link2/right_gripper_finger_link2_collision",
)


def _fix_nested_collision_mesh_apis(usd_path: Path) -> None:
    """Apply collision APIs to the imported STL mesh prim when the importer nests it under Xforms."""
    physics_usd_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
    stage_path = physics_usd_path if physics_usd_path.exists() else usd_path

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {stage_path}")

    fixed_mesh_paths: list[str] = []
    for collision_path in GRIPPER_COLLISION_PATHS:
        collision_root = stage.GetPrimAtPath(collision_path)
        if not collision_root.IsValid():
            raise RuntimeError(f"Could not find gripper collision root in generated USD: {collision_path}")

        mesh_prims = [prim for prim in Usd.PrimRange(collision_root) if prim.GetTypeName() == "Mesh"]
        if len(mesh_prims) != 1:
            raise RuntimeError(f"Expected exactly one mesh under {collision_path}, found {len(mesh_prims)}")

        mesh_prim = mesh_prims[0]
        collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        collision_api.CreateCollisionEnabledAttr(True)
        mesh_collision_api.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        applied_schemas = set(mesh_prim.GetAppliedSchemas())
        if not {"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"}.issubset(applied_schemas):
            raise RuntimeError(f"Failed to apply collision APIs to mesh prim: {mesh_prim.GetPath()}")
        fixed_mesh_paths.append(str(mesh_prim.GetPath()))

    stage.GetRootLayer().Save()
    print(f"[INFO] Fixed gripper collision APIs on {len(fixed_mesh_paths)} mesh prims in {stage_path}")


def main() -> None:
    """Convert the R1 Pro URDF to the runtime USD file."""
    R1_PRO_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=str(R1_PRO_URDF_PATH),
        usd_dir=str(R1_PRO_ASSET_DIR),
        usd_file_name=R1_PRO_USD_PATH.name,
        force_usd_conversion=args_cli.force,
        make_instanceable=True,
        fix_base=True,
        # Keep fixed frames such as left_gripper_link, right_gripper_link, and camera mounts available for IK/sensors.
        merge_fixed_joints=False,
        self_collision=False,
        collision_from_visuals=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1000.0, damping=200.0),
        ),
    )
    converter = UrdfConverter(cfg)
    _fix_nested_collision_mesh_apis(Path(converter.usd_path))
    print(f"[INFO] R1 Pro USD written to: {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
