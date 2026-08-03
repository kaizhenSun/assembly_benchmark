# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose the packaged Piper and Pika2 URDFs and convert them to a fixed-base USD asset."""

from __future__ import annotations

import argparse
import atexit
import sys
import tempfile
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "assembly_benchmark"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

parser = argparse.ArgumentParser(description="Convert the AgileX Piper + Pika2 URDFs to a fixed-base USD asset.")
parser.add_argument("--force", action="store_true", help="Force USD regeneration when output already exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from assembly_benchmark.robots.piper import (  # noqa: E402
    PIPER_ASSET_DIR,
    PIPER_BASE_URDF_PATH,
    PIPER_GRIPPER_URDF_PATH,
    PIPER_USD_PATH,
)
from assembly_benchmark.utils.piper_urdf import write_piper_pika2_urdf  # noqa: E402

from pxr import Usd, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def runtime_physics_stage_path(usd_path: Path) -> Path:
    """Return the sublayer that owns imported collision geometry."""
    physics_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
    return physics_path if physics_path.exists() else usd_path


def fix_nested_collision_meshes(usd_path: Path) -> None:
    """Ensure every imported mesh collider has enabled convex-hull collision APIs."""
    stage_path = runtime_physics_stage_path(usd_path)
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open generated Piper physics stage: {stage_path}")
    collision_meshes = [
        prim for prim in stage.TraverseAll() if prim.GetTypeName() == "Mesh" and "/colliders/" in str(prim.GetPath())
    ]
    if not collision_meshes:
        raise RuntimeError(f"No Piper collision meshes found in generated USD: {stage_path}")
    for prim in stage.TraverseAll():
        if prim.GetTypeName() == "Mesh" or "/colliders/" not in str(prim.GetPath()):
            continue
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    for mesh in collision_meshes:
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    stage.GetRootLayer().Save()


def validate_pika2_joints(usd_path: Path) -> None:
    """Validate the two independently actuated Pika2 prismatic finger joints."""
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to validate generated Piper stage: {usd_path}")
    root = stage.GetDefaultPrim()
    joint_root = root.GetPath().AppendChild("joints")
    limits = {}
    for joint_name in ("left_joint", "right_joint"):
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_root.AppendChild(joint_name)))
        if not joint:
            raise RuntimeError(f"Generated Piper + Pika2 USD is missing prismatic joint '{joint_name}'.")
        limits[joint_name] = (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
    left_lower, left_upper = limits["left_joint"]
    right_lower, right_upper = limits["right_joint"]
    if left_lower < -1.0e-6 or left_upper <= 0.0 or right_lower >= 0.0 or right_upper > 1.0e-6:
        raise RuntimeError(f"Generated Pika2 joint limits have unexpected signs: {limits}.")


def strip_converter_metadata(asset_dir: Path) -> None:
    """Remove cache metadata not needed by checked-in USD assets."""
    for metadata_file in (asset_dir / ".asset_hash", asset_dir / "config.yaml"):
        metadata_file.unlink(missing_ok=True)


atexit.register(strip_converter_metadata, PIPER_ASSET_DIR)


def main() -> None:
    PIPER_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="piper_pika2_urdf_") as temp_dir:
        composed_urdf = Path(temp_dir) / "piper_with_pika2.urdf"
        write_piper_pika2_urdf(
            composed_urdf,
            PIPER_BASE_URDF_PATH,
            PIPER_GRIPPER_URDF_PATH,
            PIPER_ASSET_DIR,
        )
        cfg = UrdfConverterCfg(
            asset_path=str(composed_urdf),
            usd_dir=str(PIPER_ASSET_DIR),
            usd_file_name=PIPER_USD_PATH.name,
            force_usd_conversion=args_cli.force,
            make_instanceable=True,
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            self_collision=False,
            collision_from_visuals=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=40.0),
            ),
        )
        generated_path = Path(UrdfConverter(cfg).usd_path)

    if generated_path.resolve() != PIPER_USD_PATH.resolve():
        raise RuntimeError(f"Unexpected generated Piper USD path: {generated_path}")
    fix_nested_collision_meshes(generated_path)
    validate_pika2_joints(generated_path)
    strip_converter_metadata(PIPER_ASSET_DIR)
    print(f"[INFO] Piper + Pika2 USD written to: {generated_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
